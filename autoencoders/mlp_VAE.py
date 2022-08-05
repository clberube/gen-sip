# @Author: charles
# @Date:   2021-05-15 15:05:15
# @Last modified by:   charles
# @Last modified time: 2021-05-15 15:05:38


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


def VAE_loss(recon_x, x, mu, logvar, beta=1., ndata=1.0):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
    # MSE /= (x.size(0)*x.size(1))
    MSE /= x.size(0)
    KLD /= (mu.size(0)*mu.size(1))
    # KLD /= mu.size(0)
    # KLD *= ndata
    return MSE, beta*KLD


class mlp_VAE(nn.Module):
    def __init__(self, input_dim, num_hidden=3, hidden_dim=128, latent_dim=2,
                 cond_dim=8, label_dim=2, activation=nn.Tanh()):
        super(mlp_VAE, self).__init__()

        self.activation = activation
        self.num_hidden = num_hidden
        self.label_dim = label_dim
        self.encoder_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])
        self.mu_var_layers = nn.ModuleList([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
        self.decoder_layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, input_dim)])
        self.classifier_layers = nn.ModuleList([nn.Linear(latent_dim, hidden_dim)] + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)] + [nn.Linear(hidden_dim, label_dim)])

    def encode(self, x):
        # h = torch.cat([x, c], 1)
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

    def mlp(self, z):
        for i, layer in enumerate(self.classifier_layers):
            if i < self.num_hidden:
                z = self.activation(layer(z))
            else:
                z = layer(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        p = self.mlp(z)
        xp = self.decode(z)
        return xp, mu, logvar, p
