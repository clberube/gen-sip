#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author: charles
# @Date:   2021-01-24 20:01:11
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:83


import torch
import numpy as np
from tqdm import tqdm
from SALib.sample import latin
import matplotlib.pyplot as plt

from ppip_model import forward_spherical, mixture
from utilities import normalize, polarize, log_complex
import plotlib


norms = ['raw', 'log', 'norm', 'pv']

f = np.logspace(2, 6, 32)
w = 2*np.pi*f


problem = {
    'num_vars': 8,
    'names': list(mixture.keys()),
    'bounds': [[np.log10(1e-5), np.log10(1e-3)],
               [np.log10(1e-3), np.log10(2e-1)],
               [-7, -5],
               [0, 5],
               # [5, 25],
               [-11, -9],
               [-10, -8],
               [-3, 0],
               # [70, 90],
               [-11, -9],
               ],
}

n_ex = 100000
param_values = latin.sample(problem, n_ex)

Z = np.empty((param_values.shape[0], w.shape[0], 2))
for i, X in enumerate(tqdm(param_values)):
    f = forward_spherical(w, *X)
    Z[i, :, 0] = f.real
    Z[i, :, 1] = f.imag

Z_0 = forward_spherical(w, **mixture)
Z_0 = torch.view_as_real(torch.from_numpy(Z_0))
Z_0 = torch.unsqueeze(Z_0, 0).numpy()

Z = np.append(Z, Z_0, axis=0)

# Z += Z*1e-2*np.random.randn(*Z.shape)

Zn = {}
Zn_0 = {}
for norm in norms:
    if norm == 'raw':
        Zn[norm] = Z
    elif norm == 'log':
        Zn[norm] = np.log10(Z)
    elif norm == 'sqrt':
        Zn[norm] = np.sqrt(Z)
    elif norm == 'cbrt':
        Zn[norm] = np.cbrt(Z)
    elif norm == 'pv':
        Zn[norm] = log_complex(Z)
    elif norm == 'rec':
        Zn[norm] = 1/Z
    elif norm == 'polar':
        Zn[norm] = polarize(Z)
    elif norm == 'norm':
        Zn[norm] = normalize(Z, axis=1)
    # Zn[norm] += 0.001*Zn[norm]*np.random.randn(*Zn[norm].shape)

    Zn_0[norm] = np.expand_dims(Zn[norm][-1], axis=0).copy()

    data = torch.tensor(Zn_0[norm]).float()
    data_fpath = f'./data/mixture-{norm}.pt'
    torch.save(data, data_fpath)

    Zn[norm] = np.delete(Zn[norm], -1, axis=0)
    data = torch.tensor(Zn[norm]).float()
    data_fpath = f'./data/dataset-{norm}.pt'
    torch.save(data, data_fpath)

param_values = torch.tensor(param_values).float()
param_fpath = './data/parameters.pt'
torch.save(param_values, param_fpath)


f = np.logspace(2, 6, 32)
legends = [
    [None, None, r"$\sigma'$ (S/m)", r"$\sigma''$ (S/m)"],
    # [None for _ in range(4)],
    # [None for _ in range(4)],
    # [None for _ in range(4)],
    [None, None, r"$\log_{10}\sigma'$", r"$\log_{10}\sigma''$"],
    # [None, None, r"$\sqrt{\sigma'}$", r"$\sqrt{\sigma''}$"],
    [None, None, r"$\sigma'_\mathrm{norm}$", r"$\sigma''_\mathrm{norm}$"],
    [None, None, r"$\ln\vert\sigma_\mathrm{eff}\vert$", r"$\varphi$ (rad)"]
]


colors = ['C0', 'C3', 'C1', 'C2']
fig, axs = plt.subplots(2, 2, figsize=(1.3*3.5433, 3.5433), sharex=True)
for i, norm in enumerate(norms):
    kwargs = dict(s=5, color=colors[i])
    ax = axs.flat[i]
    ax.scatter(f, Zn_0[norm][0][:, 0], marker='^', label=legends[i][2],
               **kwargs)
    ax.scatter(f, Zn_0[norm][0][:, 1], marker='v', label=legends[i][3],
               **kwargs)
    ax.set_title(norm)
    ax.legend(fontsize='small')
    ax.set_xscale('log')
    ax.set_xlim([f.min(), f.max()])

# Set common labels
fig.text(0.5, 0.0, 'Frequency (Hz)', ha='center', va='center')
fig.text(0.0, 0.5, 'Effective complex conductivity', ha='center', va='center',
         rotation='vertical')
plt.tight_layout()

for ext in ['.png', '.pdf']:
    plt.savefig(f'./figures/default-mixture-transforms{ext}')
