# @Author: charles
# @Date:   2021-05-15 15:05:15
# @Last modified by:   charles
# @Last modified time: 2021-05-15 15:05:38


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from utilities import softclip


log_two_pi = torch.tensor(2 * math.pi).log()


class reg_VAE(nn.Module):
    def __init__(self, input_dim, num_hidden=3, hidden_dim=128, latent_dim=2,
                 cond_dim=8, label_dim=2, activation=nn.Tanh(),
                 learn_sigma=False):

        super(reg_VAE, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.label_dim = label_dim
        self.activation = activation
        self.num_hidden = num_hidden
        self.hidden_dim = hidden_dim
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.mu_var_layers = nn.ModuleList([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, input_dim)])
        self.aux_layers = nn.ModuleList([nn.Linear(latent_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, label_dim)])
        self.learn_sigma = learn_sigma
        self.log_sigma = torch.zeros([])

    def reconstruction_loss(self, x_hat, x):
        """ Computes the NLL of the data given the latent variable.
        """
        self.log_sigma = ((x - x_hat) ** 2).mean([0, 1], keepdim=True).sqrt().log()
        # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
        # ensures stable training.
        log_sigma = softclip(self.log_sigma, -6)
        rec = self.gaussian_nll(x_hat, log_sigma, x).sum()
        return rec

    def vae_loss(self, recon_x, x, mu, logvar):
        rec = self.reconstruction_loss(recon_x, x)
        # see Appendix B of :
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return rec, kld

    def gaussian_nll(self, mu, log_sigma, x):
        return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * log_two_pi

    def encode(self, *args):
        x = torch.cat(args, 1)
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mu, logvar = (p(x) for p in self.mu_var_layers)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, *args):
        z = torch.cat(args, 1)
        for i, layer in enumerate(self.decoder_layers):
            if i <= self.num_hidden:
                z = self.activation(layer(z))
            else:
                z = layer(z)
        return z

    def aux(self, *args):
        z = torch.cat(args, 1)
        for i, layer in enumerate(self.aux_layers):
            if i <= self.num_hidden:
                z = self.activation(layer(z))
            else:
                z = layer(z)
        return z

    def forward(self, *args):
        mu, logvar = self.encode(*args)
        z = self.reparameterize(mu, logvar)
        p = self.aux(z, args[1])
        xp = self.decode(z, args[1])
        return xp, mu, logvar, p
