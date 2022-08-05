# @Author: charles
# @Date:   2021-09-24 09:09:18
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-24 09:09:02


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from sklearn import metrics

from utilities import min_max_scale


plt.style.use('seg')


ppip_labels = [r'$a_\mathrm{i}$', r'$\phi_\mathrm{i}$', r'$D_\mathrm{i}$',
               r'$\sigma_\mathrm{i}$', r'$\epsilon_\mathrm{i}$',
               r'$D_\mathrm{h}$', r'$\sigma_\mathrm{h}$',
               r'$\epsilon_\mathrm{h}$']

xfrm_labels = [r'$T_\mathrm{raw}$',
               r'$T  _\mathrm{log}$',
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


def plot_roccurve(y_values, y_preds_proba, ax=None, model_name='User',
                  explain=False, **kwargs):
    """Plots the binary classification ROC curve
    and computes the AUC

    Args:
        y_values: 1D array of true y values
        y_preds_proba: 1D array of positive class prediction probability
        ax (optional): A predefined matplotlib axes object.
        model_name (str): The name of the model for plot legend
            (default: 'User').

    @author: charles 25/05/2019
    """

    if ax is None:
        fig, ax = plt.subplots()

    fpr, tpr, _ = metrics.roc_curve(y_values, y_preds_proba)
    xx = np.arange(101) / float(100)
    aur = metrics.auc(fpr, tpr)
    if explain:
        ax.plot([0.0, 0.0], [0.0, 1.0], 'k', ls=(0, (1, 1)), lw=0.5)
        ax.plot([0.0, 1.0], [1.0, 1.0], 'k', ls=(0, (1, 1)), lw=0.5,
                label='Perfect')
        ax.plot(xx, xx, 'k', ls=(0, (1, 10)), lw=0.5, label='Random')
    ax.plot(fpr, tpr, label=f'{model_name}', **kwargs)

    # ax.set_title('ROC-AUC: {:.3f}'.format(aur))
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_aspect('equal')
    # ax.legend(handlelength=2)
    return aur


def plot_prcurve(y_values, y_preds_proba, ax=None, model_name='User',
                 explain=False, **kwargs):
    """Plots the binary classification ROC curve
    and computes the AUC.

    Args:
        y_values: 1D array of true y values
        y_preds_proba: 1D array of positive class prediction probability
        ax (optional): A predefined matplotlib axes object

    @author: charles 25/05/2019
    """

    if ax is None:
        fig, ax = plt.subplots()

    pre, rec, _ = metrics.precision_recall_curve(y_values, y_preds_proba)
    avg_pre = metrics.average_precision_score(y_values, y_preds_proba)

    ax.plot(rec[:-1], pre[:-1], label=f'{model_name}', **kwargs)

    # ax.step(rec, pre, label=f'{model_name}', where='pre', **kwargs)
    # ax.fill_between(rec, pre, alpha=0.2, color='b', step='post')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    # ax.set_aspect('equal')
    return avg_pre


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


def dense_lines(ys, x=None, ax=None, ny=100, y_lim=None, y_pad=0.01,
                normalize=True, logscale=False, **kwargs):
    """Returns a Density Line Chart.

    Args:
        ys (:obj:`list` of :obj:`1darray`): The lines to plot. Can also be
            passed as a `2darray`.
        x (:obj:`1darray`, optional): The x values corresponding to
            the data passed with `ys`. If not provided, range(0, len(ys))
            is used.
        ax (:obj:`matplotlib axes`, optional): The axes to plot on. If not
            provided a new figure will be created.
        ny (:obj:`int`, optional): The vertical grid size. Higher values
            yield a smoother density estimation. Lower values may yield a
            pixelated result. Default: 100.
        y_pad (:obj:`float`, optional): The padding fraction to establish the
            grid limits past the data values. Must be greater than 0.
            Default: 0.01 (1%).
        normalize (:obj:`bool`, optional): Normalize the plot so the density
            is between 0 and 1. Default: True.
        **kwargs: Arbitrary keyword arguments to pass to plt.imshow().

    Returns:
        A plt.imshow() object.

    """
    if ax is None:
        ax = plt.gca()

    if isinstance(ys, list):
        ys = np.array(ys)

    assert isinstance(ys, np.ndarray), (
           "`ys` must be a list of 1D arrays or a 2D array")

    assert y_pad > 0, (
           "`y_pad` must be greater than 0")

    if x is None:
        x = np.arange(ys.shape[1])

    # kwargs.setdefault('aspect', 'auto')
    # kwargs.setdefault('origin', 'lower')

    nx = x.shape[0]
    x_range = np.arange(nx)

    if y_lim is None:
        y_pad *= (ys.max() - ys.min())
        y_grid = np.linspace(ys.min()-y_pad, ys.max()+y_pad, ny)
    else:
        if logscale:
            y_grid = np.logspace(np.log10(y_lim[0]), np.log10(y_lim[1]), ny)
        else:
            y_grid = np.linspace(y_lim[0], y_lim[1], ny)

    x_grid = x
    # x_grid = np.linspace(x.min(), x.max(), nx)

    grid = np.zeros((ny, nx))
    indices = np.searchsorted(y_grid, ys) - 1

    for idx in indices:
        grid[idx, x_range] += 1

    if normalize:
        grid /= grid.max()

    extent = (x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max())
    print(extent)
    # img = ax.imshow(grid, extent=extent, **kwargs)
    img = ax.pcolormesh(x_grid, y_grid, grid, **kwargs)
    return img


"""
Plots the component-wise sensitivities

csv_files = [f'{csv_dir}/sensitivity-{n}.csv' for n in norms]


fig, axs = plt.subplots(2, 2, figsize=(5, 4))
for i in range(4):
    ax = axs.flat[i]
    df_sens = pd.read_csv(csv_files[i], index_col=0)
    df_avg = df_sens[[t+'_avg' for t in titles]]
    df_std = df_sens[[t+'_std' for t in titles]]
    df_avg.columns = titles
    df_std.columns = titles

    df_annot = df_avg.copy()
    for j in df_annot.columns:
        for k in df_annot.index:
            df_annot.loc[k, j] = str_with_err(df_avg.loc[k, j],
                                              df_std.loc[k, j])

    cbar_kws = dict(use_gridspec=False, location="right",
                    label='Relative sensitivity')
    cmap = truncate_colormap('bone_r', minval=0.05, maxval=0.85, n=255)

    sns.heatmap(df_avg, ax=ax, cmap=cmap, annot=df_annot, fmt="",
                vmin=0,
                vmax=100,
                annot_kws=dict(fontsize='small'),
                cbar_kws=cbar_kws,
                linewidths=0.8,
                cbar=False,
                square=False)

    ax.set_title(f'{labels[i]}', fontsize='medium')
    ax.set_yticklabels(ylabels, minor=False, fontsize='medium', rotation=0)
    ax.set_xticklabels(titles)

for i in range(2):
    axs[0, i].set_xticks([])
    axs[i, 1].set_yticks([])

plt.tight_layout()
if save:
    for ext in ['png', 'pdf']:
        plt.savefig(f'{fig_dir}/sensitivity-indices.{ext}')
plt.show()
"""
