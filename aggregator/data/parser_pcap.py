from typing import Any
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from .locals import *
from .tokenizer import Tokenizer
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
        d = self.__file.open(encoding="utf8").read()
        return d

    def __fill_data(self):
        d = self.read_file()
        return self.__re.findall(d)

    def data(self) -> list[Any]:
        return self.__groups


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


def parse() -> (Tokenizer, DataLoader):
    xss_parser = Parser(xss_url)
    # bpe = Tokenizer(bpe_params)
    # print(bpe.tokenize(xss_parser.data()[0]))
    # res = bpe.encode(xss_parser.data()[0])
    # print(res)
    # print(bpe.decode(res))

    return xss_parser.data()
