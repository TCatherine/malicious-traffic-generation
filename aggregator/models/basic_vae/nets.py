import pandas as pd
import torch
import random
from torch import nn
from tqdm import tqdm
from plotly import express as px
import os


class Encoder(torch.nn.Module):
    hidden_sz = 64
    layers_num = 5

    def __init__(self, dict_size):
        super().__init__()

        self.rnn = nn.LSTM(dict_size, 64, 5)

        self.act = nn.LeakyReLU()
        self.dict_size = dict_size

    def forward(self, x):
        rnn_out, (h_n, c_n) = self.rnn(x)

        h_n = torch.transpose(h_n, 0, 1)
        h_n = h_n.reshape(-1, self.hidden_sz * self.layers_num)
        return h_n


class BottleneckVariational(torch.nn.Module):
    def __init__(self, inp_sz, hidden_sz):
        super().__init__()

        self.lin_mu = torch.nn.Linear(inp_sz, hidden_sz)
        self.lin_log_var = torch.nn.Linear(inp_sz, hidden_sz)

    def forward(self, x, use_noise: int):
        # x = x.reshape(-1, 512)

        mu = self.lin_mu(x)  # bs x latent_size
        log_var = self.lin_log_var(x)  # bs x latent_size
        std = torch.exp(0.5 * log_var)

        noise = torch.randn(std.shape).to(x.device)
        noise *= use_noise

        z = noise * std + mu
        return z, mu, log_var


class Decoder(torch.nn.Module):

    def __init__(self, hidden_sz, dict_size):
        super().__init__()

        self.fc1 = nn.Linear(64, 320)
        self.rnn = nn.LSTM(dict_size, 64, 5)

        self.relu = nn.LeakyReLU()

    def forward(self, x, h_n, c_n):
        output, (h_n, c_n) = self.rnn(x, (h_n, c_n))
        return x, h_n, c_n


class VariationalAutoEncoder(torch.nn.Module):

    def __init__(self, hidden_sz, dict_size, device):
        super().__init__()
        self.device = device
        self.dict_size = dict_size

        self.encoder = Encoder(dict_size)
        self.bottleneck = BottleneckVariational(self.encoder.hidden_sz * self.encoder.layers_num,
                                                hidden_sz)

        self.bottleneck2decoder = nn.Linear(64, 320)
        self.decoder = Decoder(hidden_sz, dict_size)

    def forward(self, x: torch.Tensor, use_noise):
        x = torch.nn.functional.one_hot(x, num_classes=self.dict_size)
        x = torch.tensor(x, dtype=torch.float32)
        x = torch.transpose(x, 0, 1)

        trg_len, batch_size, trg_vocab_size = x.shape

        encoded_x = self.encoder(x)
        z, mu, log_var = self.bottleneck(encoded_x, use_noise)

        bottleneck_input = self.bottleneck2decoder(z)
        bottleneck_input = bottleneck_input.reshape((-1, 5, 64))
        bottleneck_input = torch.transpose(bottleneck_input, 0, 1)
        input = x[0:1, :]
        recon = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        h_n = bottleneck_input.contiguous()
        c_n = torch.zeros(5, 48, 64).to(self.device)

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, h_n, c_n = self.decoder(input, h_n, c_n)

            # place predictions in a tensor holding predictions for each token
            recon[t] = output

        recon = torch.transpose(recon, 0, 1)
        return recon, mu, log_var


class LossVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_cel = nn.CrossEntropyLoss()
        self.kl_coef = 0.0001

    def forward(self, line, line_rec, line_mu, log_var):
        dec_loss = self.dec_cel(line_rec, line)
        kl_loss = self.kl_divergence(line_mu, log_var) * self.kl_coef

        return dec_loss, kl_loss

    def kl_divergence(self, mu, log_var):
        kl = 1 + log_var - mu.pow(2) - log_var.exp()
        kl = -0.5 * kl.sum(-1)
        kl = kl.mean()
        return kl
