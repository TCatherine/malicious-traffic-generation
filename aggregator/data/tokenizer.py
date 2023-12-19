from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_tokenizer, sentencepiece_numericalizer
from pathlib import Path
from typing import List


class Tokenizer:
    def __init__(self, bpe_params: dict, data: Path = None):
        self.vocab_size = bpe_params['vocab_size']
        self.model_path = bpe_params['model_path']
        self.fixed_length = bpe_params['fixed_length']
        self.dict_path = str(self.model_path).split('.model')[0] + '.vocab'
        self.vocab = self.__load_dict()
        self.model = self.__load_model(data)
        self.tokens_generator = sentencepiece_tokenizer(self.model)
        self.id_generator = sentencepiece_numericalizer(self.model)


    def __load_dict(self) -> dict:
        vocab = {}
        with open(self.dict_path, 'r', encoding="utf8") as f:
            data = f.readlines()
            for i, line in enumerate(data):
                vocab[i] = line.split()[0]
        return vocab


    def __load_model(self, data: Path):
        if not self.model_path.is_file():
            model_prefix = str(self.model_path).split('.model')[0]
            generate_sp_model(data, vocab_size=self.vocab_size, model_type='bpe', model_prefix=model_prefix)

        return load_sp_model(str(self.model_path))

    @property
    def tokens_dict(self) -> dict:
        return self.vocab

    @property
    def dict_size(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> List[str]:
        return next(self.tokens_generator([text]))

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens)

    def encode(self, text: str) -> List[int]:
        return next(self.id_generator([text]))

    def decode(self, encodings: List[int]) -> str:
        data = [self.vocab[i] for i in encodings]
        return data
