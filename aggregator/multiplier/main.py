from parser import parse, get_strings
from train import train
from parameters import *
from aggregator.multiplier.models.transformer_gan.cgan_model import CGAN_Model

import torch


def is_cuda() -> bool:
    return torch.cuda.is_available()


def train_cgan(dataset):
    x, y = dataset
    model = CGAN_Model(hidden_sz=x.shape[1:])
    # model.fit(x, y)
    # model.plot_loss()
    return model

def main():
    cuda = is_cuda()
    tokenizer, dataset = parse(
        # batch_size=BATCH_SIZE,
        # is_cuda=cuda
        )
    model = train_cgan(dataset)
    data_shape = dataset[0].shape
    data_shape = (2, data_shape[1], data_shape[2])
    res = model.generate(data_shape).tolist()
    url = get_strings(res, tokenizer)
    print(url)

    # feature_dim = int(list(dataset.sampler.data_source[0][0].shape)[0])
    # det_step = DETERMINATOR_STEP
    #
    # D = Discriminator(
    #     feature_dim=feature_dim,
    #     lr=LEARNING_RATE,
    #     betas=BETAS)
    # G = Generator(
    #     feature_dim=feature_dim,
    #     lr=LEARNING_RATE,
    #     betas=BETAS)
    #
    # if cuda:
    #     G, D = G.cuda(), D.cuda()
    #
    # train(
    #     data=dataset,
    #     model=(G, D),
    #     epochs=EPOCHS,
    #     det_step=det_step,
    #     is_cuda=cuda)


if __name__ == "__main__":
    main()
