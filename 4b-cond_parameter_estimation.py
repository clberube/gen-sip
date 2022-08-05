# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 14:09:99


import string

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from autoencoders import reg_VAE
from utilities import train, predict
from utilities import weights_init
from utilities import split_train_test
from utilities import str_with_err
from plotlib import truncate_colormap, ppip_labels


bsize = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save = True
n_epoch = 20
n_real = 20
repeat = 5


norm = 'log'

fig_dir = './latex/figures'
pt_dir = './pt/estimation/'
csv_dir = './results/estimation/'


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


cond_mask = torch.zeros((7, param.shape[-1]))
cond_mask[1, -3:] = 1
cond_mask[2, 2:5] = 1
cond_mask[3, 2:] = 1
cond_mask[4, :2] = 1
cond_mask[5, :5] = 1
cond_mask[6, :2] = 1
cond_mask[6, -3:] = 1
# cond_mask[7, :] = 1


label = param[:, :]
condition_list = [torch.empty(len(data), 0),  # empty conditions
                  param[:, -3:],  # only host conditions
                  param[:, 2:5],  # only inclusions conditions
                  param[:, 2:],  # everything except a and phi
                  param[:, :2],  # only a and phi
                  param[:, :5],  # a and phi + inclusions
                  torch.cat((param[:, :2], param[:, -3:]), 1),  # a phi + host
                  # param[:, :],  # all parameters for sensitivity analysis
                  ]


scenarios = list(string.ascii_uppercase[:len(condition_list)])
df_results = pd.DataFrame()
for r in range(repeat):
    print(f'\nRepeat {r + 1}/{repeat}')
    for i, condition in enumerate(condition_list):
        print(f'Condition {i + 1}/{len(condition_list)}')

        dataset = TensorDataset(data.flatten(1),
                                condition,
                                label,
                                )  # Args: X, c, y (2D tensors)

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
                     }

        model = reg_VAE(**net_param)
        model.to(device)

        model.apply(weights_init)
        ntrain = len(train_sampler.indices)
        ntest = len(test_sampler.indices)
        print('ntrain:', ntrain, 'ntest:', ntest)
        losses = train(model,
                       dataloader['train'],
                       device=device,
                       verbose=(n_epoch//5),
                       lr=1e-3,
                       n_epoch=n_epoch,
                       valid_loader=dataloader['test'])

        infer = predict(model, dataloader['test'], n_real=n_real)

        infer['outputs'] = infer['outputs'].reshape(-1, n_real, 32, 2)
        infer['inputs'] = infer['inputs'].reshape(-1, 32, 2)

        df_true = pd.DataFrame(infer['labels'].numpy())
        df_pred = pd.DataFrame(infer['preds'].numpy().mean(1))

        MAE = metrics.mean_absolute_error(df_true, df_pred,
                                          multioutput='raw_values')
        # constant is baseline predictor MAE,
        # see https://doi.org/10.1016/j.infsof.2016.01.003
        SA = 100*(1 - MAE/0.33396718164169986)

        scores = pd.Series(SA, name=scenarios[i])
        df_results = df_results.append(scores)

csv_fpath = f'{csv_dir}/cond-parameter-estimation-{norm}.csv'
if save:
    df_results.to_csv(csv_fpath)

df = pd.read_csv(csv_fpath, index_col=0)

if not save:
    df = df_results.copy()

df_avg = df.groupby(df.index).mean()
df_std = df.groupby(df.index).std()
df_std = df_std.fillna(0)
df_std[df_std < 0.1] = 0.1


df_cond = pd.DataFrame(cond_mask.numpy())
df_cond[df_cond == 1] = '*'
df_cond[df_cond == 0] = ''


df_annot = df_avg.copy()
for j in df_annot.columns:
    for i in df_annot.index:
        df_annot.loc[i, j] = str_with_err(df_avg.loc[i, j], df_std.loc[i, j])


fig, ax = plt.subplots()
cbar_kws = dict(use_gridspec=False, location="top",
                label=r"Standardized accuracy (\%)", drawedges=False)
# cmap = truncate_colormap('bone_r', minval=0.05, maxval=0.85, n=255)
# cmap = truncate_colormap('bone', minval=0.05, maxval=0.95, n=255)
cmap = truncate_colormap('RdBu', minval=0, maxval=1, n=255)

hax = sns.heatmap(
    df_avg.T,
    ax=ax,
    annot=df_cond.T,
    cmap=cmap,
    alpha=0.0,
    cbar=False,
    linewidths=1,
    square=False,
    fmt='',
    annot_kws=dict(fontsize='small', ha='center', va='center', color='k'),
)

hax = sns.heatmap(
    df_avg.T,
    ax=ax,
    cmap=cmap,
    annot=df_annot.T,
    fmt='',
    # vmin=df_avg.min().min(),
    # vmax=df_avg.max().max(),
    vmin=0,
    vmax=100,
    cbar=True,
    cbar_kws=cbar_kws,
    linewidths=1,
    annot_kws=dict(fontsize='x-small', color='k'),
    square=False,
    alpha=0.5,
    mask=cond_mask.numpy().T,
)


# hax.collections[-1].colorbar.solids.set_rasterized(True)

ax.set_xticklabels(scenarios, minor=False)
ax.set_yticklabels(ppip_labels, minor=False, rotation=0)

if save:
    plt.savefig(f'{fig_dir}/cond-parameter-estimation-{norm}.pdf')
plt.show()
