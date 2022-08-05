# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 14:09:99


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

from autoencoders import reg_VAE
from utilities import train
from utilities import weights_init
from utilities import split_train_test
from plotlib import truncate_colormap, corrfunc, ppip_labels


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


bsize = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save = True
n_epoch = 20
n_gen = int(1e6)

norm = 'log'

fig_dir = './latex/figures'

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

condition_list = [
                  # param[:, None, 0],  # only a_i
                  # param[:, None, 1],  # only phi_i
                  param[:, :2],  # all geometrical
                  ]

label = param[:, :]  # all parameters


container = []

for i, condition in enumerate(condition_list):

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

    pt_dir = './pt/generative'

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
                   verbose=(n_epoch//10),
                   lr=1e-3,
                   n_epoch=n_epoch)

    # Generative modelling
    z_samples = torch.randn(n_gen, net_param['latent_dim'])  # N(0, 1)
    c_samples = torch.rand(n_gen, net_param['cond_dim'])  # U[0, 1]
    gen_params = model.aux(torch.cat((z_samples, c_samples), 1))

    gen_df = pd.DataFrame(gen_params.detach().numpy())

    gen_df.columns = ppip_labels

    g = sns.PairGrid(gen_df, vars=ppip_labels, corner=False,
                     despine=False)
    g.fig.set_size_inches(4.54, 4.54)
    cmap = truncate_colormap('bone_r', minval=0, maxval=1, n=255)
    g.map_lower(sns.histplot, cmap=cmap, rasterized=True)
    g.map_diag(sns.histplot, color=cmap(0.75), fill=False, linewidth=0.5,
               element='step')
    g.map_upper(corrfunc)
    g.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0])
    for a in g.figure.axes:
        a.xaxis.label.set_size('medium')
        a.yaxis.label.set_size('medium')
        a.set_xticks((0, 1))
        a.set_yticks((0, 1))
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(left=False, bottom=False)
        for spine in ['top', 'bottom', 'left', 'right']:
            a.spines[spine].set_linewidth(0.5)
    plt.tight_layout(h_pad=0, w_pad=0)
    if save:
        plt.savefig(f'{fig_dir}/parameter-posterior-cond{i}.png')
    plt.show()
