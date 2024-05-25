import torch.nn as nn
import torch


class VAEBottleneck(nn.Module):
    def __init__(self, external_dim: int, latent_dim: int):
        super().__init__()

        self.mean = nn.Linear(external_dim, latent_dim)
        self.logvar = nn.Linear(external_dim, latent_dim)

        if latent_dim == external_dim:
            self.output = nn.Identity()
        else:
            self.output = nn.Linear(latent_dim, external_dim)

    def forward(self, x):
        return self.mean(x)

    def train_sample(self, x):
        mean = self.mean(x)
        logvar = self.logvar(x)

        sigma = torch.exp(logvar * 0.5)
        standard_normal = torch.randn_like(logvar)

        sampled = mean + sigma * standard_normal

        kl_divergence = 0.5 * torch.mean(mean ** 2 + torch.exp(logvar) - logvar - 1)

        return self.output(sampled), kl_divergence

    def inference_sample(self, x):
        return self.output(self.mean(x))
