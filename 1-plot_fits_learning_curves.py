# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 14:09:99


import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

from autoencoders import reg_VAE
from utilities import train, predict
from utilities import weights_init
from utilities import split_train_test
from plotlib import plot_learning_curves, plot_fit_example, truncate_colormap


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


bsize = 32
save = True
n_epoch = 100
n_real = 100

pt_dir = './pt/baseline'
fig_dir = './latex/figures'

norms = ['raw', 'log', 'norm', 'pv']

losses_dict = {}
for norm in norms:
    print(norm)

    data_fpath = f'./data/dataset-{norm}.pt'
    data = torch.load(data_fpath)

    data += 0.01*data*torch.randn_like(data)

    param_fpath = './data/parameters.pt'
    param = torch.load(param_fpath)

    scaler = {}
    scaler['param_min'] = param.min(0, keepdim=True)[0]
    scaler['param_max'] = param.max(0, keepdim=True)[0]

    # Apply transformations
    param -= scaler['param_min']
    param /= (scaler['param_max'] - scaler['param_min'])

    dataset = TensorDataset(data.flatten(1),
                            torch.empty(len(data), 0),
                            param,
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
                   n_epoch=n_epoch,
                   valid_loader=dataloader['test'])

    losses_dict[norm] = losses

    weights_fpath = os.path.join(pt_dir, f'baseline-{norm}-weights.pt')
    losses_fpath = os.path.join(pt_dir, f'baseline-{norm}-losses.pt')

    if save:
        torch.save(model.state_dict(), weights_fpath)
        torch.save(losses, losses_fpath)

    print()

dst_path = None
if save:
    dst_path = f'{fig_dir}/default-mixture-fit-example'


cmap = truncate_colormap('bone', minval=0.2, maxval=0.8, n=255)
labels = [r'$T_\mathrm{raw}$',
          r'$T_\mathrm{log}$',
          r'$T_\mathrm{norm}$',
          r'$T_\mathrm{pv}$',
          ]

fig, ax = plt.subplots()
styles = [':', '-', '--', '-.']
styles_train = 4*['-']
styles_valid = 4*[':']
colors = ['C0', 'C3', 'C1', 'C2']
for i in range(len(norms)):
    losses_fpath = os.path.join(pt_dir, f'baseline-{norms[i]}-losses.pt')
    train_losses = torch.load(losses_fpath)['train']
    nf = np.min(np.abs(train_losses))
    valid_losses = torch.load(losses_fpath)['valid']
    train_losses /= nf
    valid_losses /= nf
    plot_learning_curves(valid_losses,
                         ax=ax, color=cmap(0.5), lw=0.5,
                         ls=styles[i])
    plot_learning_curves(train_losses,
                         label=labels[i], lw=1,
                         ax=ax, color=cmap(0.0),
                         ls=styles[i])
ax.set_xlabel('Epoch')
ax.set_ylabel('Relative loss')
ax.legend(ncol=2, handlelength=1.2)
if save:
    plt.savefig(f'{fig_dir}/learning-curves.pdf')


f = np.logspace(2, 6, 32)
fig, axs = plt.subplots(4, 1, figsize=(3.54, 1.5*3.54), sharex=True)
for i in range(len(norms)):
    ax = axs.flat[i]
    norm = norms[i]

    Zn_0 = torch.load(f'./data/mixture-{norm}.pt')

    Zn_0 += 0.01*Zn_0*torch.randn_like(Zn_0)

    dataset = TensorDataset(Zn_0.flatten(1),
                            torch.empty(len(Zn_0), 0),
                            torch.zeros(len(Zn_0), net_param['label_dim']))
    dataloader = DataLoader(dataset)

    model = reg_VAE(**net_param)
    weights_fpath = os.path.join(pt_dir, f'baseline-{norm}-weights.pt')
    model.load_state_dict(torch.load(weights_fpath))

    infer = predict(model, dataloader, n_real=n_real)

    infer['outputs'] = infer['outputs'].reshape(-1, n_real, 32, 2)
    infer['inputs'] = infer['inputs'].reshape(-1, 32, 2)

    # preds = infer['outputs']
    preds = None

    if preds is None:
        legends = [None, None, r"$x'$", r"$x''$"]
        tag = 'data'
    else:
        legends = ['CVAE', r"95 \% CI", r"$x'$", r"$x''$"]
        tag = 'fits'

    plot_fit_example(1, infer['inputs'], preds=preds,
                     dst_path=dst_path,
                     freq_range=[np.log10(f.min()), np.log10(f.max())],
                     ylog=False,
                     ax=ax, labels=legends,
                     color=cmap(0))

    ax.set_xlabel(None)
    ax.set_ylabel(labels[i])
    # ax.set_title(norms[i])
    # ax.set_xlim([f.min(), f.max()])

axs[0].legend(fontsize='small', loc='center right', ncol=2)
axs[-1].set_xlabel('Frequency (Hz)')

plt.tight_layout()

if save:
    plt.savefig(f'{fig_dir}/default-mixture-{tag}.pdf')
