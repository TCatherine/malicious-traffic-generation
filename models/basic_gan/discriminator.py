import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, feature_dims, lr, betas):
        super(Discriminator, self).__init__()
        feature_dim = feature_dims[0] * feature_dims[1]

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
        data = x.reshape(x.shape[0], -1)
        res = self.model(data)
        return res
