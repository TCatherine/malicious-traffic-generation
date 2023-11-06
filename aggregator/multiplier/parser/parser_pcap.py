from typing import List, Any

from torch.utils.data import DataLoader, TensorDataset
from .locals import xss_url
from .tokenize import Tokenizer
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


def parse(
        # batch_size,
        # is_cuda=False
) -> (Tokenizer, list):
    xss_parser = Parser(xss_url)
    xss_tokenizer = Tokenizer(xss_parser.data())
    xss_tokenizer.fit()

    dataset = [xss_tokenizer.transform(sample) for sample in xss_parser.data()]
    aligned_dataset = dataset.copy()
    [data.extend([0 for _ in range(STRING_SIZE - len(data))]) for data in aligned_dataset]
    return xss_tokenizer, aligned_dataset
