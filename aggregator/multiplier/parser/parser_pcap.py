from typing import List, Any

from torch.utils.data import DataLoader, TensorDataset
from .locals import xss, benign
import re
from .bpe.bpe import Encoder


class Parser:
    def __init__(self,
                 dictionary):
        self.__re = re.compile(dictionary['re'])
        self.__file = dictionary['path']
        self.__url_encoder = Encoder(200, pct_bpe=0.88)
        self.__groups = self.data()

    def read_file(self) -> str:
        d = self.__file.open().read()
        return d

    def __fill_data(self):
        d = self.read_file()
        self.groups = self.__re.findall(d)

    def data(self) -> list[Any]:
        return self.groups

    def enable_bpe(self):
        endpoints = [req[0] for req in self.groups]
        self.__url_encoder.fit(endpoints)


def parse(
        batch_size,
        is_cuda=False
) -> dict:
    xss_parser = Parser(xss)
    xss_data = xss_parser.data()
    tokenizer(xss_data)

    # benign_parser = Parser(benign)
    # benign_data = benign_parser.data()

    a = 1
