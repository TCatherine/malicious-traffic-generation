import os

import pandas as pd
import torch
from plotly import express as px
from tqdm import tqdm

from .nets import VariationalAutoEncoder, LossVAE


class VAE_Model:
    # device = torch.device('cpu')
    device = torch.device('cuda')

    file_path = os.path.dirname(__file__)

    def __init__(self, hidden_sz):
        self.net = VariationalAutoEncoder(hidden_sz)
        self.net = self.net.to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.loss = LossVAE()

        self.loss_track = []

        self.weights_path = f'{self.file_path}/weights/{self.net.__class__.__name__}_{hidden_sz}'

    def fit(self, train_dataloader):
        # x_train = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        # x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)

        # batch_size = 48
        for epoch in tqdm(range(1500)):

            # train
            for train_batch, labels in train_dataloader:
                # train_batch = x_train[i:i + batch_size]
                train_batch = train_batch.to(self.device)

                self.optimizer.zero_grad()
                rec, mu, log_var = self.net(train_batch, use_noise=True)
                train_l, detailed = self.loss(train_batch, rec, mu, log_var)
                train_l.backward()
                self.optimizer.step()

            # # test
            # with torch.no_grad():
            #     rec, mu, log_var = self.net(x_test, use_noise=False)
            #     test_l, detailed = self.loss(x_test, rec, mu, log_var)

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
