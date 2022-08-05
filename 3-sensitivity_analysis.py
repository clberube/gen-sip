# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:85


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from autoencoders import reg_VAE
from utilities import train
from utilities import str_with_err
from utilities import weights_init
from utilities import split_train_test
from plotlib import truncate_colormap, ppip_labels, xfrm_labels


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bsize = 32
save = True
n_epoch = 1
repeat = 5


fig_dir = './latex/figures'
csv_dir = './results/sensitivity/'

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

    param -= scaler['param_min']
    param /= (scaler['param_max'] - scaler['param_min'])

    condition = param[:, :]  # all parameters for sensitivity analysis
    label = torch.empty(len(data), 0)  # no label training

    dataset = TensorDataset(data.flatten(1),
                            condition,
                            label,
                            )  # Args: X, c, y (2D tensors)

    container = []
    for r in range(repeat):
        print(f'Repeat {r + 1}/{repeat}')

        train_sampler, test_sampler = split_train_test(dataset, test_split=0.2)

        dataloader = {}
        dataloader['train'] = DataLoader(dataset, batch_size=bsize,
                                         sampler=train_sampler)
        dataloader['test'] = DataLoader(dataset, batch_size=bsize,
                                        sampler=test_sampler)

        # TRAINING HERE
        net_param = {'input_dim': dataset[:][0].shape[-1],
                     'cond_dim': dataset[:][1].shape[-1],
                     'label_dim': dataset[:][2].shape[-1],
                     'num_hidden': 3,
                     'hidden_dim': 32,
                     'latent_dim': 8,
                     'activation': nn.SiLU(),
                     'learn_sigma': False,
                     }

        model = reg_VAE(**net_param)
        model.to(device)

        model.apply(weights_init)
        ntrain = len(train_sampler.indices)
        ntest = len(test_sampler.indices)
        if r == 0:
            print('ntrain:', ntrain, 'ntest:', ntest)
        losses = train(model,
                       dataloader['train'],
                       device=device,
                       verbose=1,
                       lr=1e-3,
                       n_epoch=n_epoch,
                       valid_loader=None,
                       )

        # sens = torch.stack(losses['input_grad_total']).sum(0)
        container.append(losses['input_grad_total'])

    sens = torch.stack(container).numpy()
    sens /= sens.sum(-1, keepdims=True)
    sens *= 100

    df_results = df_results.append(pd.Series(sens.mean(0), name=norm+'_avg'))
    df_results = df_results.append(pd.Series(sens.std(0), name=norm+'_std'))
    print()


df_results = df_results.T
df_results.index = ppip_labels

csv_path = f'{csv_dir}/sensitivity-indices.csv'

if save:
    df_results.to_csv(csv_path)

df = pd.read_csv(csv_path, index_col=0)

if not save:
    df = df_results.copy()

df_annot = pd.DataFrame(columns=norms, index=ppip_labels)
for j in df_annot.columns:
    for k in df_annot.index:
        df_annot.loc[k, j] = str_with_err(df.loc[k, j+'_avg'],
                                          df.loc[k, j+'_std']/np.sqrt(repeat))


fig, ax = plt.subplots()
cbar_kws = dict(use_gridspec=False, location="top",
                label=r"Relative sensitivity (\%)")
cmap = truncate_colormap('bone_r', minval=0.05, maxval=1, n=255)
cmap = truncate_colormap('RdBu', minval=0.5, maxval=1, n=255)

sns.heatmap(df.loc[:, df.columns.str.contains('_avg')],
            ax=ax,
            cmap=cmap,
            annot=df_annot,
            fmt='',
            vmin=0,
            vmax=df.max().max(),
            cbar=True,
            cbar_kws=cbar_kws,
            linewidths=1,
            annot_kws=dict(fontsize='small', color='k'),
            square=False,
            alpha=0.5)

ax.set_yticklabels(ppip_labels, minor=False, rotation=0)
ax.set_xticklabels(xfrm_labels, minor=False, rotation=0, ha='center')
# ax.set_xlabel("Data transformation")

if save:
    plt.savefig(f'{fig_dir}/sensitivity-indices.pdf')
plt.show()
