from data import run as ref_run

from data import parse, DatasetURI, Tokenizer
from parameters import *
from models.basic_gan.cgan_model import CGAN_Model
from models.basic_vae.vae import VAE_Model

import torch


def is_cuda() -> bool:
    return torch.cuda.is_available()


def run(
        types: list[str],
        number: int,
        use_miltiplier: bool
) -> list:
    reference_samples = ref_run(types, number)
    if use_miltiplier:
        # TODO: implement miltiplier module
        bpe_dict = load_dict(bpe_params=BPE)
    else:
        samples = ref_run(types, number)

    samples = reference_samples
    return samples


def train_cgan(dataset):
    x, y = dataset
    model = CGAN_Model(hidden_sz=x.shape[1:])
    model.fit(x, y)
    model.plot_loss()
    model.save_weights()
    return model


def train_vae(data, tokenizer):
    # train_data, test_data = split(data)
    # train, test = train_test_split(data)

    dataset = DatasetURI(data, tokenizer)

    hidden_size = 64
    model = VAE_Model(hidden_sz=hidden_size, dict_size=tokenizer.dict_size)
    model.fit(dataset)
    model.plot_loss()
    model.save_weights()
    return model


def main():
    data = parse()
    tokenizer = Tokenizer(BPE, 'data/dataset/xss_url')
    tokenizer.metrics(data)

    model = train_vae(data, tokenizer)

    data_shape = dataset[0].shape
    data_shape = (1, data_shape[1], data_shape[2])
    res = model.generate(data_shape).tolist()
    url = get_strings(res, tokenizer)
    print(run(['xss'], 10, False, False))


if __name__ == "__main__":
    main()
