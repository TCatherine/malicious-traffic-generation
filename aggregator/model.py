import torch
import torch.nn as nn

lr = 0.5
# class Discriminator(nn.Module):
# 	def __init__(self, feature_dim, lr, betas):
# 		super(Discriminator, self).__init__()
#
# 		self.model = nn.Sequential(
# 			nn.Linear(feature_dim, 512),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(512, 256),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Linear(256, 1),
# 			nn.Sigmoid()
# 		)
#
# 		self.optimizer = torch.optim.Adam(
# 			self.parameters(),
# 			lr=lr,
# 			betas=betas)
#
# 	def forward(self, x):
# 		x = self.model(x)
# 		return x
#
#
# class Generator(nn.Module):
# 	def __init__(self, feature_dim, lr, betas):
# 		super(Generator, self).__init__()
#
# 		def block(in_feat, out_feat, normalize=True):
# 			layers = [nn.Linear(in_feat, out_feat)]
# 			if normalize:
# 				layers.append(nn.BatchNorm1d(out_feat, 0.8))
# 			layers.append(nn.LeakyReLU(0.2, inplace=True))
# 			return layers
#
# 		self.model = nn.Sequential(
# 			*block(feature_dim, 128, normalize=False),
# 			*block(128, 256),
# 			*block(256, 512),
# 			*block(512, 1024),
# 			nn.Linear(1024, feature_dim),
# 			nn.Tanh()
# 		)
#
# 		self.optimizer = torch.optim.Adam(
# 			self.parameters(),
# 			lr=lr,
# 			betas=betas)
#
# 	def forward(self, x):
# 		x = x.reshape([x.shape[0], -1])
# 		x = self.model(x)
# 		return x

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


class ConvVariationalAutoEncoder(torch.nn.Module):

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


class VariationalAutoEncoderModel:
    device = torch.device('cuda')
    file_path = os.path.dirname(__file__)

    def __init__(self, hidden_sz):
        self.net = ConvVariationalAutoEncoder(hidden_sz)
        self.net = self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss = LossVAE()

        self.loss_track = []

        self.weights_path = f'{self.file_path}/weights/{self.net.__class__.__name__}_{hidden_sz}'

    def fit(self, x_train, x_test):
        x_train = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)

        for epoch in tqdm(range(1500)):
            self.optimizer.zero_grad()
            rec, mu, log_var = self.net(x_train, use_noise=True)
            train_l, detailed = self.loss(x_train, rec, mu, log_var)
            train_l.backward()
            self.optimizer.step()

            # test
            with torch.no_grad():
                rec, mu, log_var = self.net(x_test, use_noise=False)
                test_l, detailed = self.loss(x_test, rec, mu, log_var)

            # save losses
            self.loss_track.append({'train': float(train_l), 'test': float(test_l)})

        self.l = pd.DataFrame(self.loss_track)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred, _, _ = self.net(x, use_noise=False)
        pred = pred.cpu().numpy()
        return pred

    def encode(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, hidden, _ = self.net(x, use_noise=False)
        hidden = hidden.cpu().numpy()
        return hidden

    def decode(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            rec = self.net.decoder(x)
        rec = rec.cpu().numpy()
        return rec

    def plot_loss(self):
        l = self.loss_track[10:]
        px.line(l).show()

    def save_weights(self):
        torch.save(self.net.state_dict(), self.weights_path)
        return self.weights_path

    def load_weights(self):
        self.net.load_state_dict(torch.load(self.weights_path,
                                            map_location=self.device))
