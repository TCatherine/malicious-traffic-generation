from torch.utils.data import DataLoader, TensorDataset
from .locals import xss, benign
import re

class Parser:
    def __init__(self,
                 dictionary):
        self.re = re.compile(dictionary['re'])
        self.file = dictionary['path']

    def read_file(self) -> str:
        d = self.file.open().read()
        return d

    def data(self) -> dict:
        d = self.read_file()
        groups = self.re.findall(d)
        return groups

def parse(
        batch_size,
        is_cuda=False
) -> dict:
    xss_parser = Parser(xss)
    xss_data = xss_parser.data()

    benign_parser = Parser(benign)
    benign_data = benign_parser.data()

    a = 1
