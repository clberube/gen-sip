# @Author: charles
# @Date:   2021-09-08 14:09:74
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 14:09:99


import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import seaborn as sns
import matplotlib.pyplot as plt

from autoencoders import reg_VAE
from plotlib import truncate_colormap, corrfunc, ppip_labels


RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_gen = int(1e6)
save = True
pt_dir = './pt/baseline/'
fig_dir = './latex/figures'

norms = ['raw', 'log']

net_param = {'input_dim': 64,
             'cond_dim': 0,
             'label_dim': 8,
             'num_hidden': 3,
             'hidden_dim': 32,
             'latent_dim': 8,
             'activation': nn.SiLU(),
             }


for norm in norms:
    print(norm)

    model = reg_VAE(**net_param)
    weights_fpath = os.path.join(pt_dir, f'baseline-{norm}-weights.pt')
    model.load_state_dict(torch.load(weights_fpath))
    model.eval()

    # Generative modelling
    z_samples = torch.randn(n_gen, net_param['latent_dim'])
    with torch.no_grad():
        gen_params = model.aux(z_samples)

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
        a.set_xticklabels([], fontsize='medium')
        a.set_yticklabels([], fontsize='medium')
        a.tick_params(left=False, bottom=False)
        for spine in ['top', 'bottom', 'left', 'right']:
            a.spines[spine].set_linewidth(0.5)
    plt.tight_layout(h_pad=0, w_pad=0)
    if save:
        plt.savefig(f'{fig_dir}/parameter-posterior-{norm}.png')
    # plt.close('all')
