from parser import parse
from train import train
from model import Discriminator, Generator
from parameters import *
from parser.models.transformer_gan.cgan_model import CGAN_Model

import torch


def is_cuda() -> bool:
    return torch.cuda.is_available()


def train_cgan():
    model = CGAN_Model(hidden_sz=1000)


def main():
    cuda = is_cuda()
    tokenizer, dataset = parse(
        # batch_size=BATCH_SIZE,
        # is_cuda=cuda
        )

    feature_dim = int(list(dataset.sampler.data_source[0][0].shape)[0])
    det_step = DETERMINATOR_STEP

    D = Discriminator(
        feature_dim=feature_dim,
        lr=LEARNING_RATE,
        betas=BETAS)
    G = Generator(
        feature_dim=feature_dim,
        lr=LEARNING_RATE,
        betas=BETAS)

    if cuda:
        G, D = G.cuda(), D.cuda()

    train(
        data=dataset,
        model=(G, D),
        epochs=EPOCHS,
        det_step=det_step,
        is_cuda=cuda)


if __name__ == "__main__":
    main()
