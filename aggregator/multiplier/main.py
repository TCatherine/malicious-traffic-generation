from parser import parse, get_strings
from train import train
from parameters import *
from aggregator.multiplier.models.basic_gan.cgan_model import CGAN_Model

import torch


def is_cuda() -> bool:
    return torch.cuda.is_available()


def train_cgan(dataset):
    x, y = dataset
    model = CGAN_Model(hidden_sz=x.shape[1:])
    model.fit(x, y)
    model.plot_loss()
    return model

def main():
    tokenizer, dataset = parse(bpe_params = BPE)
    model = train_cgan(dataset)
    data_shape = dataset[0].shape
    data_shape = (1, data_shape[1], data_shape[2])
    res = model.generate(data_shape).tolist()
    url = get_strings(res, tokenizer)
    a = 1

if __name__ == "__main__":
    main()
