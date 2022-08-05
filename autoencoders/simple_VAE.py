# @Author: charles
# @Date:   2021-05-15 15:05:15
# @Last modified by:   charles
# @Last modified time: 2021-05-15 15:05:38


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class simple_VAE(nn.Module):
    def __init__(self, input_dim, num_hidden=3, hidden_dim=128, latent_dim=2,
                 param_dim=2, activation=nn.Tanh()):
        super(simple_VAE, self).__init__()

        self.activation = activation
        self.num_hidden = num_hidden
        self.param_dim = param_dim
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.mu_var_layers = nn.ModuleList([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, input_dim)])

    def encode(self, x):
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mu, logvar = (p(x) for p in self.mu_var_layers)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        for i, layer in enumerate(self.decoder_layers):
            if i < self.num_hidden:
                z = self.activation(layer(z))
            else:
                z = layer(z)
        return z

    def forward(self, x, y):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xp = self.decode(z)
        return xp, mu, logvar
