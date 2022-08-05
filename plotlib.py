# @Author: charles
# @Date:   2021-09-24 09:09:18
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2022-08-05 15:08:87


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

from utilities import min_max_scale


try:
    plt.style.use('seg')
except IOError:
    pass


ppip_labels = [r'$a_\mathrm{i}$', r'$\phi_\mathrm{i}$', r'$D_\mathrm{i}$',
               r'$\sigma_\mathrm{i}$', r'$\epsilon_\mathrm{i}$',
               r'$D_\mathrm{h}$', r'$\sigma_\mathrm{h}$',
               r'$\epsilon_\mathrm{h}$']

xfrm_labels = [r'$T_\mathrm{raw}$',
               r'$T_\mathrm{log}$',
               r'$T_\mathrm{norm}$',
               r'$T_\mathrm{pv}$',
               ]


def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("${:.2f}$".format(r),
                va='center', ha='center',
                xy=(.5, .5), xycoords=ax.transAxes,
                fontsize='small')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''
    https://stackoverflow.com/a/18926541
    '''
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


cmap = truncate_colormap('bone', minval=0.2, maxval=0.8, n=255)


def plot_learning_curves(losses, dst_path=None, ax=None, norm=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    nsteps = len(losses)

    if norm == 'unit':
        losses = min_max_scale(losses)
    elif norm == 'mean':
        losses = losses/np.mean(np.abs(losses))
    elif norm == 'maxabs':
        losses = losses/np.max(np.abs(losses))
    elif norm == 'minabs':
        losses = losses/np.min(np.abs(losses))
    elif norm == 'max':
        losses = losses/np.max(losses)
    elif norm == 'min':
        losses = losses/np.min(losses)
    elif norm == 'first':
        losses = losses/losses[0]
    elif norm == 'max':
        losses = losses/np.max(losses)

    ax.plot(range(1, nsteps+1), losses, **kwargs)
    if dst_path and ax is None:
        for ext in ['.png', '.pdf']:
            plt.savefig(dst_path + ext)


def plot_fit_example(ntest, inputs, preds=None, dst_path=None,
                     freq_range=[2, 6], ylog=True, ax=None, labels=None,
                     color=None, **kwargs):

    if labels is None:
        labels = ["Mean fit", "95% CI", r"$\sigma'$", r"$\sigma''$"]

    if ax is None:
        fig, ax = plt.subplots()

    w = np.random.choice(ntest)
    f = np.logspace(freq_range[0], freq_range[1], 32)
    cmap = truncate_colormap('bone', minval=0.2, maxval=0.8, n=255)

    ax.scatter(f, inputs[w][:, 0], s=5, marker='^', color=color,
               label=labels[2], **kwargs)
    ax.scatter(f, inputs[w][:, 1], s=5, marker='v', color=color,
               label=labels[3], **kwargs)

    if preds is not None:
        ax.plot(f, preds.mean(1)[w][:, 0], '-', color=cmap(0.0),
                label=labels[0], **kwargs)
        ax.fill_between(f,
                        preds.quantile(0.025, 1)[w][:, 0],
                        preds.quantile(0.975, 1)[w][:, 0],
                        color=cmap(1.0), alpha=1.0, zorder=0,
                        label=labels[1],
                        **kwargs)

        ax.plot(f, preds.mean(1)[w][:, 1], '-', color=cmap(0.0), **kwargs)
        ax.fill_between(f,
                        preds.quantile(0.025, 1)[w][:, 1],
                        preds.quantile(0.975, 1)[w][:, 1],
                        color=cmap(1.0), alpha=0.5, zorder=0,
                        **kwargs)

    ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized conductivity')
    if dst_path and ax is None:
        for ext in ['.png', '.pdf']:
            plt.savefig(dst_path + ext)
