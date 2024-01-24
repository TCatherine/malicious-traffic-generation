import re
import random

from typing import Any, List
from .locals import *


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


def parse() -> List[str]:
    xss_parser = Parser(xss_url_suricata)

    return xss_parser.data()
