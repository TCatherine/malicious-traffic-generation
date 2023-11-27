import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, feature_dims, lr, betas):
        super(Generator, self).__init__()
        self.l_shape = feature_dims[0]
        self.dict_shape = feature_dims[1]
        feature_dim = feature_dims[0] * feature_dims[1]

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
            nn.Sigmoid()
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            betas=betas)

    def forward(self, x):
        data = x.reshape(x.shape[0], -1)
        res = self.model(data)
        return res.reshape(-1, self.l_shape, self.dict_shape)
