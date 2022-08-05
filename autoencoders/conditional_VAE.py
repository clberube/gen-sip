# @Author: charles
# @Date:   2021-05-15 15:05:15
# @Last modified by:   charles
# @Last modified time: 2021-05-15 15:05:38


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


class conditional_VAE(nn.Module):
    def __init__(self, input_dim, num_hidden=3, hidden_dim=128, latent_dim=2,
                 cond_dim=2, activation=nn.Tanh()):
        super(conditional_VAE, self).__init__()

        self.activation = activation
        self.num_hidden = num_hidden
        self.cond_dim = cond_dim
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.mu_var_layers = nn.ModuleList([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim + cond_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, input_dim)])

    def encode(self, x, c):
        x = torch.cat([x, c], 1)
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mu, logvar = (p(x) for p in self.mu_var_layers)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        z = torch.cat([z, c], 1)
        for i, layer in enumerate(self.decoder_layers):
            if i < self.num_hidden:
                z = self.activation(layer(z))
            else:
                z = layer(z)
        return z

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        xp = self.decode(z, c)
        return xp, mu, logvar
