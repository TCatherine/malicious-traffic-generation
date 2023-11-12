import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, feature_dim, lr, betas):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=betas)

    def forward(self, x):
        x = self.model(x)
        return x
