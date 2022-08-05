# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:49


import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from autoencoders import reg_VAE
from utilities import train, predict
from utilities import weights_init
from utilities import split_train_test
from plotlib import truncate_colormap, xfrm_labels


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bsize = 32
save = True
n_epoch = 20
n_real = 100
repeat = 5
latent_dims = range(1, 9)


pt_dir = './pt/compression'
fig_dir = './latex/figures'

norms = ['raw', 'log', 'norm', 'pv']

df_results = pd.DataFrame()
for norm in norms:
    print(norm)

    data_fpath = f'./data/dataset-{norm}.pt'
    data = torch.load(data_fpath)

    param_fpath = './data/parameters.pt'
    param = torch.load(param_fpath)

    scaler = {}
    scaler['param_min'] = param.min(0, keepdim=True)[0]
    scaler['param_max'] = param.max(0, keepdim=True)[0]

    # Apply transformations
    param -= scaler['param_min']
    param /= (scaler['param_max'] - scaler['param_min'])

    condition = torch.empty(len(data), 0)  # no conditioning
    label = torch.empty(len(data), 0)  # no MLP training

    dataset = TensorDataset(data.flatten(1),
                            condition,
                            label,
                            )  # Args: X, c, y (2D tensors)

    # TRAINING HERE
    net_param = {'input_dim': dataset[:][0].shape[-1],
                 'cond_dim': dataset[:][1].shape[-1],
                 'label_dim': dataset[:][2].shape[-1],
                 'num_hidden': 3,
                 'hidden_dim': 32,
                 'latent_dim': None,
                 'activation': nn.SiLU(),
                 }

    for r in range(repeat):
        print(f'Repeat {r + 1}/{repeat}')

        train_sampler, test_sampler = split_train_test(dataset, test_split=0.2)

        dataloader = {}
        dataloader['train'] = DataLoader(dataset, batch_size=bsize,
                                         sampler=train_sampler)
        dataloader['test'] = DataLoader(dataset, batch_size=bsize,
                                        sampler=test_sampler)

        ntrain = len(train_sampler.indices)
        ntest = len(test_sampler.indices)
        print('ntrain:', ntrain, 'ntest:', ntest)

        mse_list = []
        for latent_dim in latent_dims:
            print('zdim:', latent_dim)
            net_param.update(latent_dim=latent_dim)
            model = reg_VAE(**net_param)
            model.apply(weights_init)

            history = train(model,
                            train_loader=dataloader['train'],
                            device=device,
                            verbose=n_epoch,
                            lr=1e-3,
                            n_epoch=n_epoch,
                            )

            infer = predict(model, dataloader['test'], n_real=n_real,
                            disable_prog_bar=True)

            infer['outputs'] = infer['outputs'].reshape(-1, n_real, 32, 2)
            infer['inputs'] = infer['inputs'].reshape(-1, 32, 2)

            mse = F.mse_loss(infer['inputs'],
                             infer['outputs'].mean(1),
                             reduction='mean')
            mse_list.append(mse.item())

            out_fn = ("{!s}={!r}".format(k, v) for (k, v) in net_param.items())
            out_fn = ','.join(out_fn)

            weights_fpath = os.path.join(pt_dir, f'{out_fn}-weights.pt')
            losses_fpath = os.path.join(pt_dir, f'{out_fn}-losses.pt')

            if save:
                torch.save(model.state_dict(), weights_fpath)
                torch.save(history, losses_fpath)

            scores = pd.Series(mse_list, name=norm)

        df_results = df_results.append(scores)

        print()


# Remove outliers
# abs_z_scores = np.abs(stats.zscore(df_results))
# df_results = df_results[(abs_z_scores < 3).all(axis=1)]


df_avg = df_results.groupby(df_results.index, sort=False).mean()
df_std = df_results.groupby(df_results.index, sort=False).std()
df_std = df_std.fillna(0)

df = (df_avg.T).join(df_std.T, lsuffix='_avg', rsuffix='_std')
df.index = latent_dims


csv_fpath = "./results/compression/latent-dim-MSE.csv"
if save:
    df.to_csv(csv_fpath)


df = pd.read_csv(csv_fpath, index_col=0)

fig, ax = plt.subplots()
cmap = truncate_colormap('bone', minval=0.2, maxval=0.8, n=255)
kwargs = dict(color=cmap(0.0))
ls = [':', '-', '--', '-.']
avg_cols = df.columns[df.columns.str.contains('_avg')]
std_cols = df.columns[df.columns.str.contains('_std')]
for i, (avg_col, std_col) in enumerate(zip(avg_cols, std_cols)):
    ax.plot(df.index, df[avg_col]/df[avg_col].max(), ls=ls[i],
            label=xfrm_labels[i], **kwargs)
    ax.fill_between(df.index,
                    (df[avg_col] - df[std_col]/np.sqrt(repeat))/df[avg_col].max(),
                    (df[avg_col] + df[std_col]/np.sqrt(repeat))/df[avg_col].max(),
                    color=cmap(1.0), alpha=0.25)

ax.set_ylabel('Relative validation loss')
ax.set_xlabel('Latent dimensions')
ax.set_xticks(df.index)
ax.set_yscale('log')
ax.legend(handlelength=1.2)

if save:
    plt.savefig(f'{fig_dir}/latent-dim-MSE.pdf')
plt.show()
