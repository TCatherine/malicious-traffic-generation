from typing import List, Any

from torch.utils.data import DataLoader, TensorDataset
from .locals import xss, benign
import re
from .bpe.bpe import Encoder


class Parser:
    def __init__(self,
                 dictionary):
        self.re = re.compile(dictionary['re'])
        self.file = dictionary['path']

    def read_file(self) -> str:
        d = self.file.open().read()
        return d

    def data(self) -> list[Any]:
        d = self.read_file()
        groups = self.re.findall(d)
        return groups


def tokenizer(data):
    encoder = Encoder(200, pct_bpe=0.88)
    endpoints = [req[0] for req in data]
    encoder.fit(endpoints)


def parse(
        batch_size,
        is_cuda=False
) -> dict:
    xss_parser = Parser(xss)
    xss_data = xss_parser.data()
    tokenizer(xss_data)

    benign_parser = Parser(benign)
    benign_data = benign_parser.data()

    a = 1
