import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from plotly import express as px
import os
from .discriminator import Discriminator
from .generator import Generator
from .locals import *
from torch.autograd import Variable


class CGAN_Model:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    file_path = os.path.dirname(__file__)

    def __init__(self, hidden_sz):
        self.lambda_gp = LAMBDA_GRADIENT_PENALTY
        self.generator = Generator(hidden_sz, lr=LEARNING_RATE, betas=BETAS)
        self.discriminator = Discriminator(hidden_sz, lr=LEARNING_RATE, betas=BETAS)

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        self.criterion = nn.BCELoss()

        self.loss_track = []

        self.gen_weights_path = f'{self.file_path}/weights/cgan_{self.generator.__class__.__name__}_{hidden_sz}'
        self.disc_weights_path = f'{self.file_path}/weights/cgan_{self.discriminator.__class__.__name__}_{hidden_sz}'

    def generate_random_data_input(self, n_shape, l_shape, dict):
        noise = torch.randn((n_shape, l_shape, dict), dtype=torch.float32, device=self.device)
        noise = noise.argmax(dim=2)
        noise_labels = torch.nn.functional.one_hot(noise, num_classes=dict) * 1.0

        return noise_labels

    def generate(self, data_shape):
        x = self.generate_random_data_input(data_shape[0], data_shape[1], data_shape[2])
        return self.generator(x)

    def train_generator_step(self, data_shape):
        self.generator.optimizer.zero_grad()

        g_output = self.generate(data_shape)
        y = Variable(torch.zeros(data_shape[0], 1).to(self.device))
        d_output = self.discriminator(g_output)
        g_loss = 1 - self.criterion(d_output, y)

        g_loss.backward()
        self.generator.optimizer.step()

        return g_loss.data.item()

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = self.generate_random_data_input(*real_samples.shape)

        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_discrim_step(self, x_real, y_real):
        self.discriminator.optimizer.zero_grad()

        d_real_score = self.discriminator(x_real)
        d_real_loss = self.criterion(d_real_score, y_real)

        # train discriminator on fake
        x_fake = self.generate(x_real.shape)
        y_fake = Variable(torch.zeros(x_real.shape[0], 1).to(self.device))

        d_fake_score = self.discriminator(x_fake)
        d_fake_loss = self.criterion(d_fake_score, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        gp = self.compute_gradient_penalty(x_real, x_fake)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.discriminator.optimizer.step()

        return d_loss.data.item()

    def fit(self, x_train, y_train):
        x_train = torch.tensor(x_train, dtype=torch.float32, device=self.device)
        # x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)

        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        # y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        for epoch in tqdm(range(EPOCHS)):
            d_loss = self.train_discrim_step(x_train, y_train)
            g_loss = self.train_generator_step(data_shape=x_train.shape)

            #     # test
            #     with torch.no_grad():
            #         rec, mu, log_var = self.net(x_test, y_test, use_noise=False)
            #         test_l, detailed = self.loss(x_test, rec, mu, log_var)
            #
            # save losses
            self.loss_track.append({'d': float(d_loss), 'g': float(g_loss)})
        #
        self.l = pd.DataFrame(self.loss_track)

    # def predict(self, x, y):
    #     x = torch.tensor(x, dtype=torch.float32, device=self.device)
    #     y = torch.tensor(y, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         pred, _, _ = self.net(x, y, use_noise=False)
    #     pred = pred.cpu().numpy()
    #     return pred

    # def encode(self, x, y):
    #     x = torch.tensor(x, dtype=torch.float32, device=self.device)
    #     y = torch.tensor(y, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         _, hidden, _ = self.encode(x, y, use_noise=False)
    #     hidden = hidden.cpu().numpy()
    #     return hidden
    #
    # def decode(self, x, y):
    #     x = torch.tensor(x, dtype=torch.float32, device=self.device)
    #     y = torch.tensor(y, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         rec = self.net.decoder(x, y)
    #     rec = rec.cpu().numpy()
    #     return rec

    # def transfer(self, x, y, y_out):
    #     x = torch.tensor(x, dtype=torch.float32, device=self.device)
    #     y = torch.tensor(y, dtype=torch.float32, device=self.device)
    #     y_out = torch.tensor(y_out, dtype=torch.float32, device=self.device)
    #
    #     with torch.no_grad():
    #         _, mu, _ = self.net(x, y, use_noise=False)
    #         rec = self.net.decoder(mu, y_out)
    #
    #     rec = rec.cpu().numpy()
    #     return rec
    #
    #     pass
    #
    def plot_loss(self):
        l = self.loss_track[10:]
        px.line(l).show()

    def save_weights(self):
        torch.save(self.generator.state_dict(), self.gen_weights_path)
        torch.save(self.discriminator.state_dict(), self.disc_weights_path)

    def load_weights(self):
        self.generator.load_state_dict(torch.load(self.gen_weights_path,
                                                  map_location=self.device))
        self.discriminator.load_state_dict(torch.load(self.disc_weights_path,
                                                      map_location=self.device))
