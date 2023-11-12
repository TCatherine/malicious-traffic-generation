import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, feature_dim, lr, betas):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(feature_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, feature_dim),
            nn.Tanh()
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=betas)

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        x = self.model(x)
        return x
