import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from plotly import express as px
import os
from .discriminator import Discriminator
from .generator import Generator


class CGAN_Model:
    device = torch.device('cuda')
    file_path = os.path.dirname(__file__)
    batch_sz = 23

    def __init__(self, hidden_sz):
        self.hidden_sz = hidden_sz

        self.generator = Generator(hidden_sz)
        self.discriminator = Discriminator()

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        self.dis_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.001)

        self.criterion = nn.BCELoss()

        self.loss_track = []

        self.gen_weights_path = f'{self.file_path}/weights/cgan_{self.generator.__class__.__name__}_{hidden_sz}'
        self.disc_weights_path = f'{self.file_path}/weights/cgan_{self.discriminator.__class__.__name__}_{hidden_sz}'

    def train_generator_step(self):

        z = torch.randn(self.batch_sz, self.hidden_sz).to(self.device)
        y = torch.ones(self.batch_sz, 1).to(self.device)

        G_output = self.generator(z)
        D_output = self.discriminator(G_output)
        G_loss = self.criterion(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.gen_optimizer.step()

        return G_loss.data.item()

    def train_discrim_step(self):

        # train discriminator on real
        x_real, y_real = x.view(-1, mnist_dim), torch.ones(bs, 1)
        x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

        D_output = D(x_real)
        D_real_loss = criterion(D_output, y_real)
        D_real_score = D_output

        # train discriminator on facke
        z = Variable(torch.randn(bs, z_dim).to(device))
        x_fake, y_fake = G(z), Variable(torch.zeros(bs, 1).to(device))

        D_output = D(x_fake)
        D_fake_loss = criterion(D_output, y_fake)
        D_fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

        return D_loss.data.item()

    def generate_random_data_input(self, n_shape, l_shape):
        noise = torch.randn(n_shape, dtype=torch.float32, device=self.device)

        noise_labels = torch.randn(l_shape, dtype=torch.float32, device=self.device)
        noise_labels = noise_labels.argmax(dim=1)
        noise_labels = torch.nn.functional.one_hot(noise_labels) * 1.0

        return noise, noise_labels

    def fit(self, x_train, y_train, x_test, y_test):
        # https://github.com/rishabhd786/VAE-GAN-PYTORCH/blob/master/main.py
        x_train = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)

        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        for epoch in tqdm(range(1500)):
            z, mu, log_var = self.enc(x_train, y_train, use_noise=True)
            x_pred = self.dec(z, y_train)

            noise, noise_labels = self.generate_random_data_input(mu.shape, y_train.shape)
            x_from_noise = self.dec(noise, noise_labels)

            # train discrim
            class_gen_rec, vector_rec = self.disc(x_pred, y_train)
            class_gen_noise, vector_fake = self.disc(x_from_noise, noise_labels)
            class_pred_real, vector_real = self.disc(x_train, y_train)

            dl1 = self.disc_loss(class_gen_rec, torch.zeros_like(class_gen_rec))
            dl2 = self.disc_loss(class_gen_noise, torch.zeros_like(class_gen_noise))
            dl3 = self.disc_loss(class_pred_real, torch.zeros_like(class_pred_real))

            disc_loss = dl1 + dl2 + dl3

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            # train generator
            class_gen_rec, vector_rec = self.disc(x_pred, y_train)
            class_gen_noise, vector_fake = self.disc(x_from_noise, noise_labels)
            class_pred_real, vector_real = self.disc(x_train, y_train)

            dl1 = self.disc_loss(class_gen_rec, torch.zeros_like(class_gen_rec))
            dl2 = self.disc_loss(class_gen_noise, torch.zeros_like(class_gen_noise))
            dl3 = self.disc_loss(class_pred_real, torch.zeros_like(class_pred_real))

            disc_loss = dl1 + dl2 + dl3

            l_prior = KL_loss()
            l_disc_like = class_pred_fake
            self.train_discrim_step()

            self.optimizer.zero_grad()
            rec, mu, log_var = self.net(x_train, y_train, use_noise=True)
            train_l, detailed = self.loss(x_train, rec, mu, log_var)
            train_l.backward()
            self.optimizer.step()

            # test
            with torch.no_grad():
                rec, mu, log_var = self.net(x_test, y_test, use_noise=False)
                test_l, detailed = self.loss(x_test, rec, mu, log_var)

            # save losses
            self.loss_track.append({'train': float(train_l), 'test': float(test_l)})

        self.l = pd.DataFrame(self.loss_track)

    def predict(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            pred, _, _ = self.net(x, y, use_noise=False)
        pred = pred.cpu().numpy()
        return pred

    def encode(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, hidden, _ = self.net(x, y, use_noise=False)
        hidden = hidden.cpu().numpy()
        return hidden

    def decode(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            rec = self.net.decoder(x, y)
        rec = rec.cpu().numpy()
        return rec

    def transfer(self, x, y, y_out):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        y_out = torch.tensor(y_out, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, mu, _ = self.net(x, y, use_noise=False)
            rec = self.net.decoder(mu, y_out)

        rec = rec.cpu().numpy()
        return rec

        pass

    def plot_loss(self):
        l = self.loss_track[10:]
        px.line(l).show()

    def save_weights(self):
        torch.save(self.net.state_dict(), self.weights_path)
        return self.weights_path

    def load_weights(self):
        self.net.load_state_dict(torch.load(self.weights_path,
                                            map_location=self.device))
