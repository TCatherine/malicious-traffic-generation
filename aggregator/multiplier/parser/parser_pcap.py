from typing import List, Any

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from .locals import xss_url
from .tokenize import Tokenizer
import numpy as np

import re

STRING_SIZE = 150


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
    dictarr = np.asarray(list(tokens_dict.keys())).reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(dictarr)
    ohe_data = []
    for line in dataset:
        data = np.reshape(line, (-1, 1))
        ohe_data.append(enc.transform(data).toarray().tolist())
    return ohe_data


def parse(
        # batch_size,
        # is_cuda=False
) -> (Tokenizer, list):
    xss_parser = Parser(xss_url)
    xss_tokenizer = Tokenizer(xss_parser.data())
    xss_tokenizer.fit()

    data = xss_tokenizer.transform(xss_parser.data(), 200)
    dataset = one_hot_encoding(data, xss_tokenizer.inverse_tokens_dict)

    return xss_tokenizer, dataset
