import json 
from typing import Iterable, Iterator


class Tokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, encoding="utf-8") as f: 
            raw_vocab = json.load(f)

        vocab = {int(k) : v.encode("latin-1") for k, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = json.loads(line)

                merges.append((a.encode("latin1"), b.encode("latin1")))


        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        pass 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass 

    def decode(self, ids: list[int]) -> str:
        pass 