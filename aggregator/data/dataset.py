import torch

from torch.nn.utils.rnn import pad_sequence
from typing import List
from .tokenizer import Tokenizer


class DatasetURI(torch.utils.data.Dataset):
    def __init__(self,
                 data: list,
                 tokenizer: Tokenizer
                 ):
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> List[int]:
        return self.tokenizer.encode(self.data[idx])


def collate_fn(data: List[List[int]]) -> torch.Tensor:
    dataset = [torch.as_tensor(d) for d in data]
    res = pad_sequence(dataset, batch_first=True)

    # res = torch.nn.functional.one_hot(res, num_classes=num_classes)
    return res
