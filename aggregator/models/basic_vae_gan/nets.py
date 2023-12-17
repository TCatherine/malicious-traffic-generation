import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from plotly import express as px
import os


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x[:, None, :, :]

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))

        return x


class BottleneckVariational(torch.nn.Module):
    def __init__(self, hidden_sz):
        super().__init__()

        self.lin_mu = torch.nn.Linear(512, hidden_sz)
        self.lin_log_var = torch.nn.Linear(512, hidden_sz)

    def forward(self, x, use_noise: int):
        x = x.reshape(-1, 512)

        mu = self.lin_mu(x)  # bs x latent_size
        log_var = self.lin_log_var(x)  # bs x latent_size
        std = torch.exp(0.5 * log_var)

        noise = torch.randn(std.shape).to(x.device)
        noise *= use_noise

        z = noise * std + mu
        return z, mu, log_var


class Decoder(torch.nn.Module):
    def __init__(self, hidden_sz):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.fc1 = nn.Linear(hidden_sz, 512)

        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))

        x = x.reshape(-1, 128, 2, 2)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        x = x.squeeze()
        return x


class VariationalAutoEncoder(torch.nn.Module):

    def __init__(self, hidden_sz):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = BottleneckVariational(hidden_sz)
        self.decoder = Decoder(hidden_sz)

    def forward(self, x, use_noise):
        x = self.encoder(x)
        z, mu, log_var = self.bottleneck(x, use_noise)
        recon = self.decoder(z)
        return recon, mu, log_var


class LossVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_mse = nn.MSELoss()
        self.kl_coef = 0.0001

    def forward(self, track, track_rec, track_mu, log_var):
        dec_mse = self.dec_mse(track, track_rec)
        kl_loss = self.kl_divergence(track_mu, log_var) * self.kl_coef

        total = dec_mse + kl_loss

        detailed = [float(i) for i in [dec_mse, kl_loss]]
        return total, detailed

    def kl_divergence(self, mu, log_var):
        kl = 1 + log_var - mu.pow(2) - log_var.exp()
        kl = -0.5 * kl.sum(-1)
        kl = kl.mean()
        return kl



