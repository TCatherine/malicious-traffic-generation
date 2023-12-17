import torch
import re
from typing import List

from .locals import *
from .tokenize import Tokenizer


class DatasetURI(torch.utils.data.Dataset):
    def __init__(self,
                 data: list,
                 tokenize: Tokenizer
                 ):
        self.tokenize = tokenize
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> List[int]:
        return self.tokenize.encode(self.data[idx])

def collate_fn(data: List[List[int]], num_classes: int) -> torch.Torch:
    res = torch.as_tensor(data)
    res = torch.nn.functional.one_hot(res, num_classes=num_classes)
    return res

