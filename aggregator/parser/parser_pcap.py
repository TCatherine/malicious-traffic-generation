from typing import Any
from sklearn.preprocessing import OneHotEncoder
from .locals import *
from .tokenize import Tokenizer

import numpy as np
import re
import random


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


def one_hot_encoding(dataset: list[Any], tokens_dict: dict) -> list:
    dictarr = np.asarray([tokens_dict[d] for d in tokens_dict]).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(dictarr)
    ohe_data = []
    for line in dataset:
        data = np.reshape(line, (-1, 1))
        ohe_data.append(enc.transform(data).toarray().tolist())
    return ohe_data


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


def parse(bpe_params: dict) -> (Tokenizer, list):
    xss_parser = Parser(xss_url)

    if not bpe_params['dict_path'].is_file():
        data = ' '.join(xss_parser.data())
        Tokenizer.from_corpus(data, learn_bpe_args=bpe_params.copy())

    bpet = Tokenizer.load(bpe_params['dict_path'], fixed_length=bpe_params['fixed_length'])
    tokens = [bpet.encode(data) for data in xss_parser.data()]

    dataset = one_hot_encoding(tokens, bpet.vocab_stoi)
    dataset = (np.asarray(dataset, dtype=np.int16), np.ones((len(dataset), 1), dtype=np.int16))
    return bpet, dataset
