from .bpe.bpe import Encoder


class Tokenizer:
    def __init__(self, data: list):
        self.__url_encoder = Encoder(200, pct_bpe=0.45)
        self.data = data

    def fit(self):
        self.__url_encoder.fit(self.data)

    def transform(self, sample: str) -> list[int]:
        res = self.__url_encoder.transform([sample])
        return next(res)

    def get_tokens(self, sample: str) -> list[str]:
        res = self.__url_encoder.tokenize(sample)
        return res

    def inverse(self, token: iter(list[int])):
        res = self.__url_encoder.inverse_transform([token])
        return next(res)
