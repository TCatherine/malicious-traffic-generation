from typing import Any
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from .locals import *
from .tokenize import Tokenizer
from tqdm import tqdm
from .dataset import DatasetURI

import numpy as np
import re
import random
import torch


class Parser:
    def __init__(self,
                 dictionary):
        self.__re = re.compile(dictionary['re'])
        self.__file = dictionary['path']
        self.__groups = self.__fill_data()

    def read_file(self) -> str:
        d = self.__file.open().read()
        return d

    def __fill_data(self):
        d = self.read_file()
        return self.__re.findall(d)

    def data(self) -> list[Any]:
        return self.__groups


def get_strings(dataset: list[Any], tokenizer: Tokenizer) -> list:
    dictarr = np.asarray([tokenizer.vocab_stoi[d] for d in tokenizer.vocab_stoi]).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(dictarr)
    strings = []
    for line in dataset:
        res = enc.inverse_transform(line)
        enc_data = [d[0] for d in res]
        data = tokenizer.decode(enc_data)
        strings.append(data)
    return strings


def load_bpe(bpe_params: dict, data: list = None) -> Tokenizer:
    if (not bpe_params['dict_path'].is_file() or bpe_params['force_update']) and list is not None:
        data = ' '.join(data)
        Tokenizer.from_corpus(data, learn_bpe_args=bpe_params.copy())

    bpet = Tokenizer.load(bpe_params['dict_path'], fixed_length=bpe_params['fixed_length'])
    return bpet


def run(
        types: list[str],
        number: int,
):
    data_types = {
        'xss': xss_url_suricata,
        'benign': benign
    }
    type_number = number // len(types)

    samples = []
    for type in types:
        parser = Parser(data_types[type])
        data = parser.data()
        samples.extend(random.choices(data, k=type_number))
    return samples


def load_dict(bpe_params: dict) -> Tokenizer:
    bpet = Tokenizer.load(bpe_params['dict_path'], fixed_length=bpe_params['fixed_length'])
    return bpet


def parse() -> (Tokenizer, DataLoader):
    xss_parser = Parser(xss_url)
    # bpet = load_bpe(bpe_params)
    return xss_parser.data()

    # ds = DatasetURI(xss_parser.data(), bpet)
    #
    # tokens_dl = DataLoader(
    #     dataset=TensorDataset(torch.as_tensor(tokens)),
    #     shuffle=False,
    #     batch_size=2*batch_size,
    #     drop_last=False)
    #
    # ohe = torch.Tensor()
    # for t in tqdm(tokens_dl, desc="One hot encoding"):
    #     res = torch.nn.functional.one_hot(t[0], num_classes=bpet.dict_size)
    #     ohe = torch.cat((ohe, res))
    #     # ohe.extend(one_hot_encoding(t, bpet.vocab_stoi))
    #
    # data_target = torch.ones(len(tokens), 1)
    # dataset = DataLoader(
    #     dataset=TensorDataset(torch.LongTensor(tokens), data_target),
    #     shuffle=True,
    #     batch_size=batch_size,
    #     drop_last=True)
    # return bpet, dataset
    pass
