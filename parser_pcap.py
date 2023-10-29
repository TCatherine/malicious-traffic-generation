from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import re


data = Path("data/xss")

def parse(
        batch_size,
        file=data,
        is_cuda=False
) -> DataLoader:
    with file.open() as f:
        data = f.read()

    http_header_re = re.compile(
        '\n'
        'GET (.*)\n'
        '\n'
        'HTTP\/(.*)\n'
        'Host: (.*)\n'
        'User-Agent: (.*)\n'
        'Accept-Encoding: (.*)\n'
        'Accept: (.*)\n'
        'Connection: (.*)\n'
    )

    groups = http_header_re.findall(data)




