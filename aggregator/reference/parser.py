import re
import random

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
        type,
        number
) -> list:
    parser = Parser(type)
    data = parser.data()
    return random.choices(data, k=number)
