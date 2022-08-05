# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:85


import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from autoencoders import reg_VAE
from ppip_model import mixture
from utilities import predict
from utilities import str_with_err
from utilities import split_train_test
from plotlib import truncate_colormap, ppip_labels, xfrm_labels


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

bsize = 32
save = True
n_real = 100
repeat = 5

norms = ['raw', 'log', 'norm', 'pv']

fig_dir = './latex/figures'
csv_dir = './results/estimation/'
pt_dir = './pt/baseline/'


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
    label = param[:, :]  # all parameters for estimation

    for r in range(repeat):
        print(f'Repeat {r + 1}/{repeat}')

        dataset = TensorDataset(data.flatten(1),
                                condition,
                                label,
                                )  # Args: X, c, y (2D tensors)

        train_sampler, test_sampler = split_train_test(dataset, test_split=0.2)

        dataloader = {}
        dataloader['train'] = DataLoader(
            dataset, batch_size=bsize, sampler=train_sampler)
        dataloader['test'] = DataLoader(
            dataset, batch_size=bsize, sampler=test_sampler)

        # TRAINING HERE
        net_param = {'input_dim': dataset[:][0].shape[-1],
                     'cond_dim': dataset[:][1].shape[-1],
                     'label_dim': dataset[:][2].shape[-1],
                     'num_hidden': 3,
                     'hidden_dim': 32,
                     'latent_dim': 8,
                     'activation': nn.SiLU(),
                     }

        model = reg_VAE(**net_param)
        weights_fpath = os.path.join(pt_dir, f'baseline-{norm}-weights.pt')
        model.load_state_dict(torch.load(weights_fpath))

        infer = predict(model, dataloader['test'], n_real=n_real,
                        disable_prog_bar=True)

        infer['outputs'] = infer['outputs'].reshape(-1, n_real, 32, 2)
        infer['inputs'] = infer['inputs'].reshape(-1, 32, 2)

        df_true = pd.DataFrame(infer['labels'].numpy())
        df_pred = pd.DataFrame(infer['preds'].numpy().mean(1))

        MAE = metrics.mean_absolute_error(df_true, df_pred,
                                          multioutput='raw_values')
        # constant is baseline predictor MAE,
        # see https://doi.org/10.1016/j.infsof.2016.01.003
        SA = 100*(1 - MAE/0.33396718164169986)

        scores = pd.Series(SA, name=norm)
        df_results = df_results.append(scores)

    print()


csv_fpath = f'{csv_dir}/parameter-estimation.csv'

if save:
    df_results.to_csv(csv_fpath)

df = pd.read_csv(csv_fpath, index_col=0)

if not save:
    df = df_results.copy()

df_avg = df.groupby(df.index, sort=False).mean()
df_std = df.groupby(df.index, sort=False).std()
df_std = df_std.fillna(0)
df_std[df_std < 0.1] = 0.1

df_annot = df_avg.copy()
for j in df_annot.columns:
    for i in df_annot.index:
        df_annot.loc[i, j] = str_with_err(df_avg.loc[i, j], df_std.loc[i, j])


fig, ax = plt.subplots()
cbar_kws = dict(use_gridspec=False, location="top",
                label=r"Standardized accuracy (\%)")
cmap = truncate_colormap('RdBu', minval=0, maxval=1, n=255)

sns.heatmap(df_avg.T,
            ax=ax,
            cmap=cmap,
            annot=df_annot.T,
            fmt='',
            vmin=0,
            vmax=100,
            cbar=True,
            cbar_kws=cbar_kws,
            alpha=0.5,
            linewidths=1,
            annot_kws=dict(fontsize='small', color='k'),
            square=False)

ax.set_yticklabels(ppip_labels, minor=False, rotation=0)
ax.set_xticklabels(xfrm_labels, minor=False, rotation=0, ha='center')

if save:
    for ext in ['png', 'pdf']:
        plt.savefig(f'{fig_dir}/parameter-estimation.{ext}')
plt.show()


norm = 'log'
model = reg_VAE(**net_param)
weights_fpath = os.path.join(pt_dir, f'baseline-{norm}-weights.pt')
model.load_state_dict(torch.load(weights_fpath))

Zn_0 = torch.load(f'./data/mixture-{norm}.pt')
p0 = torch.tensor(list(mixture.values())).unsqueeze(0)

dataset = TensorDataset(Zn_0.flatten(1),
                        torch.empty(len(Zn_0), 0),
                        p0)
dataloader = DataLoader(dataset)


infer = predict(model, dataloader, n_real=n_real, disable_prog_bar=True)

infer['preds'] *= (scaler['param_max'] - scaler['param_min'])
infer['preds'] += scaler['param_min']

infer['labels'] = 10**infer['labels']
infer['preds'] = 10**infer['preds']

tru = infer['labels'].squeeze().numpy()
nom = infer['preds'].mean(1).squeeze().numpy()
std = infer['preds'].std(1).squeeze().numpy()
l95 = infer['preds'].quantile(0.025, dim=1).squeeze().numpy()
u95 = infer['preds'].quantile(0.975, dim=1).squeeze().numpy()

fig, ax = plt.subplots()
idx = np.arange(len(tru))
ax.errorbar(idx, tru, color='C3', marker='x', ms=5, ls='')
ax.errorbar(idx, nom, yerr=std, color='k',
            marker='.', ls='', ms=3)
ax.set_ylabel(r'$\log_{10}{\eta}$')
ax.set_xticks(idx)
ax.set_xticklabels(ppip_labels)
ax.set_yscale('log')
plt.show()


plt.scatter(df_true.loc[:, 6], df_pred.loc[:, 6])
