from .bpe.bpe import Encoder


class Tokenizer:
    def __init__(self, data: list):
        self.EOS = '__eos'  # end of string
        self.SOS = '__sos'  # start of string
        self.PAD = '__pad'
        self.data = data

        self.__url_encoder = Encoder(
            200,
            pct_bpe=0.65,
            ngram_max=10,
            required_tokens=[self.EOS, self.SOS],
            PAD=self.PAD
        )

    def fit(self):
        self.__url_encoder.fit(self.data)

    def __aligne(self, data: list[int], fixed_length: int = None) -> list[int]:
        data.insert(0, self.tokens_dict[self.SOS])
        data.append(self.tokens_dict[self.EOS])

        if fixed_length is not None:
            data = data[:fixed_length]
            while len(data) < fixed_length:
                data.append(self.tokens_dict[self.PAD])

        return data

    def transform(self, sample: list[str], fixed_length: int = None) -> list[list[int]]:
        data = list(self.__url_encoder.transform(sample))
        aligned_data = [self.__aligne(d, fixed_length) for d in data]
        return aligned_data

    def get_tokens(self, sample: str) -> list[str]:
        res = self.__url_encoder.tokenize(sample)
        return res

    def inverse(self, token: list[int]):
        res = self.__url_encoder.inverse_transform([token])
        return next(res)

    @property
    def tokens_dict(self) -> dict:
        vocab_dict = self.__url_encoder.vocabs_to_dict()
        byte_pairs = vocab_dict['byte_pairs']
        words = vocab_dict['words']

        tokens_dict = words.copy()
        tokens_dict.update(byte_pairs)
        return tokens_dict

    @property
    def inverse_tokens_dict(self) -> dict:
        tokens_dict = self.__url_encoder.inverse_bpe_vocab.copy()
        tokens_dict.update(self.__url_encoder.inverse_word_vocab)
        return tokens_dict

    def add_token(self, token: str):
        if token not in self.__url_encoder.required_tokens:
            self.__url_encoder.required_tokens.append(token)
            self.fit()
