# @Author: charles
# @Date:   2021-09-08 14:09:79
# @Email:  charles.berube@polymtl.ca
# @Last modified by:   charles
# @Last modified time: 2021-09-08 14:09:28


import os
import math
import warnings
from statistics import mean
from timeit import default_timer as timer

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
from scipy import convolve
from tqdm import tqdm

from metrics import get_rmse


warnings.filterwarnings(
    "ignore", message="Initializing zero-element tensors is a no-op")


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway.
    Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)
    return result_tensor


def min_max_scale(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


def str_with_err(value, error):
    if error > 0:
        digits = -int(math.floor(math.log10(error)))
    else:
        digits = 0
    if digits < 0:
        digits = 0
    err10digits = math.floor(error*10**digits)
    return "${0:.{2}f}({1:.0f})$".format(value, err10digits, digits)


def append_suffix(fpath, suffix):
    name, ext = os.path.splitext(fpath)
    return f"{name}-{suffix}{ext}"


def compute_MARP0(X):
    MARP0 = 0
    for i in range(len(X)):
        for j in range(i):
            MARP0 += np.abs(X[i] - X[j])
    MARP0 *= 2/(len(X)**2)
    return MARP0


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)


def normalize(X, axis=1):
    X_max = X.max(axis=axis, keepdims=True)
    X_min = X.min(axis=axis, keepdims=True)
    return (X - X_min) / (X_max - X_min)


def standardize(X, axis=1):
    X_avg = X.mean(axis=axis, keepdims=True)
    X_std = X.std(axis=axis, keepdims=True)
    return (X - X_avg) / X_std


def max_abs_scale(X, axis=1):
    max_abs = np.abs(X).max(axis=axis, keepdims=True)
    return X / max_abs


def normalize_complex(X, axis=None):
    norm = np.linalg.norm(X, axis=-1, keepdims=True)
    if axis is not None:
        norm = norm.max(axis=axis)
    return X / norm


def log_complex(X):
    complex_X = X[:, :, 0] + 1j*X[:, :, 1]
    LC = np.empty(X.shape)
    LC[:, :, 0] = np.log(np.abs(complex_X))
    LC[:, :, 1] = np.angle(complex_X)
    return LC


def polarize(X):
    complex_X = X[:, :, 0] + 1j*X[:, :, 1]
    AP = np.empty(X.shape)
    AP[:, :, 0] = np.abs(complex_X)
    AP[:, :, 1] = np.angle(complex_X)
    return AP


def wide_normalize(X, X_min=None, X_max=None):
    if X_max is None:
        X_max = X.max(0, keepdims=True).max(1, keepdims=True)
        print("X_max:", X_max)
    if (X_min is None):
        X_min = X.min(0, keepdims=True).min(1, keepdims=True)
        print("X_min:", X_min)
    return (X - X_min) / (X_max - X_min)


def split_train_test(dataset, test_split=0.2, random_seed=None):
    shuffle_dataset = True
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    return train_sampler, test_sampler


def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # step is in [0,1]
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0 + np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L


def frange_cycle_cosine(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio)  # step is in [0,1]
    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop:
            L[int(i+c*period)] = 0.5-.5*math.cos(v*math.pi)
            v += step
            i += 1
    return L


def log_normal_pdf(sample, mu, logvar, raxis=1):
    log2pi = np.log(2. * np.pi)
    return torch.sum(
        -.5 * ((sample - mu) ** 2. * torch.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def train(model, train_loader, verbose, lr, n_epoch, device=None, beta=None,
          valid_loader=None):

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # if beta is None:
    #     beta = torch.ones(n_epoch)

    train_losses = ['log_sigma', 'NLL', 'KLD', 'AUX', 'train']
    valid_losses = ['valid']
    grads = ['input_grad_total']

    history = {k: np.zeros(n_epoch) for k in train_losses}
    history.update({k: torch.zeros(model.cond_dim) for k in grads})
    history.update({k: np.zeros(n_epoch) for k in valid_losses})

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # reg_loss = torch.nn.MSELoss(reduction='sum')
    # reg_loss = torch.nn.L1Loss(reduction='mean')
    # mlp_loss = torch.nn.BCEWithLogitsLoss()

    start_time = timer()
    model.to(device)
    for e in range(n_epoch):
        running_loss = {k: 0 for k in train_losses}  # reset running losses
        model.train()
        for X, c, y in train_loader:
            X = X.to(device)
            c = c.to(device)
            y = y.to(device)

            if c.shape[-1] > 0:
                c.requires_grad = True

            optimizer.zero_grad()

            # Forward pass
            Xp, mu, logvar, p = model(X, c)
            if y.shape[-1] == 0:
                AUX = torch.tensor(0)
            # elif y.shape[-1] == 1:
            #     AUX = mlp_loss(p, y)
            elif y.shape[-1] > 1:
                # AUX = reg_loss(p, y)
                AUX = model.reconstruction_loss(p, y)

            NLL, KLD = model.vae_loss(Xp, X, mu, logvar)

            total_loss = NLL + KLD + AUX

            running_loss['log_sigma'] += model.log_sigma.item()*X.size(0)
            running_loss['NLL'] += NLL.item()
            running_loss['KLD'] += KLD.item()
            running_loss['AUX'] += AUX.item()*X.size(0)
            running_loss['train'] += total_loss.item()

            # if (e + 1 == n_epoch) and (c.shape[-1] > 0):
            # if (e == 0) and (c.shape[-1] > 0):
            if c.shape[-1] > 0:
                y_grad_total = torch.autograd.grad(
                    NLL, c, retain_graph=True)[0]
                history['input_grad_total'] += y_grad_total.square().sum(0)

            # Backward pass
            total_loss.backward()
            optimizer.step()

        for k in train_losses:
            history[k][e] = running_loss[k]/len(train_loader.sampler)

        verbose_str = (f"Epoch: {(e+1):.0f}, "
                       f"log sigma: {history['log_sigma'][e]:.2f}, "
                       f"NLL: {history['NLL'][e]:.0f}, "
                       f"KLD: {history['KLD'][e]:.0f}, "
                       f"AUX: {history['AUX'][e]:.3f}, "
                       f"Train: {history['train'][e]:.0f}"
                       )

        if valid_loader:
            model.eval()
            running_loss = {k: 0 for k in valid_losses}  # reset running losses

            for X, c, y in valid_loader:
                X = X.to(device)
                c = c.to(device)
                y = y.to(device)

                # Forward pass
                Xp, mu, logvar, p = model(X, c)
                if y.shape[-1] == 0:
                    AUX = torch.tensor(0)
                # elif y.shape[-1] == 1:
                #     AUX = mlp_loss(p, y)
                elif y.shape[-1] > 1:
                    # AUX = reg_loss(p, y)
                    AUX = model.reconstruction_loss(p, y)

                NLL, KLD = model.vae_loss(Xp, X, mu, logvar)
                # total_loss = NLL + beta[e]*KLD + AUX
                # NLL_scaled = X.shape[-1]*NLL/(2*logvar.exp())
                # total_loss = X.shape[-1]*sigma.log() + NLL_scaled + KLD + AUX
                total_loss = NLL + KLD + AUX
                running_loss['valid'] += total_loss.item()

            for k in valid_losses:
                history[k][e] = running_loss[k]/len(valid_loader.sampler)

            verbose_str += f", Valid: {history['valid'][e]:.0f}"

        if verbose:
            if (e + 1) % verbose == 0:
                print(verbose_str)

    end_time = timer()
    print(f'Training time: {(end_time - start_time)/60:.2f} m')
    return history


def predict(model, dataloader, n_real=20, device=None, disable_prog_bar=False):
    """Infers reconstructions and labels from a test DataLoader.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = len(dataloader.sampler)

    inputs = torch.empty(N, model.input_dim)
    outputs = torch.empty(N, n_real, model.input_dim)
    conds = torch.empty(N, model.cond_dim)
    labels = torch.empty(N, model.label_dim)
    mus = torch.empty(N, model.latent_dim)
    logvars = torch.empty(N, model.latent_dim)
    preds = torch.empty(N, n_real, model.label_dim)

    i = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (X, c, y) in tqdm(dataloader, disable=disable_prog_bar):
            X = X.to(device)
            c = c.to(device)
            y = y.to(device)

            B = X.shape[0]

            inputs[i:i+B] = X
            conds[i:i+B] = c
            labels[i:i+B] = y

            X = X.unsqueeze(1).expand(-1, n_real,
                                      - 1).reshape(B*n_real, model.input_dim)
            c = c.unsqueeze(1).expand(-1, n_real,
                                      - 1).reshape(B*n_real, model.cond_dim)
            Xp, mu, logvar, p = model(X, c)

            outputs[i:i+B] = Xp.reshape(B, n_real, model.input_dim)
            mus[i:i+B] = mu.reshape(B, n_real, model.latent_dim)[:, 0, :]
            logvars[i:i+B] = logvar.reshape(B, n_real, model.latent_dim)[:, 0, :]
            preds[i:i+B] = p.reshape(B, n_real, model.label_dim)

            i += B

    keys = ['inputs', 'labels', 'outputs', 'mus', 'logvars', 'preds']
    scope = locals()
    return {k: scope[k] for k in keys}
