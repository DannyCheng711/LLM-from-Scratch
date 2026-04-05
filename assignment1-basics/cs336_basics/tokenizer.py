import json 
from typing import Iterable, Iterator
import regex as re 

# uv run pytest tests/test_tokenizer.py  

"""
→ split by special tokens
→ for each part:
    if special token:
        map to id
    else:
        PAT pretokenize
        → bytes
        → BPE merge by rank
        → ids
""" 
class Tokenizer:

    # BPE merges token pairs based on their frequency in the training data, 
    # and uses this ranking to determine the priority of merges during encoding.
    
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab # id -> bytes
        self.token_to_id = {v : k for k, v in vocab.items()} # bytes -> id
        self.merges = merges
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.rx = re.compile(PAT)

    # overloading + factory
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # read UTF-8 file  → latin-1 str ->  encode ("latin1") → bytes
        with open(vocab_filepath, encoding="utf-8") as f: 
            raw_vocab = json.load(f)

        vocab = {int(k) : v.encode("latin-1") for k, v in raw_vocab.items()} # id -> bytes

        merges = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                a, b = json.loads(line)

                merges.append((a.encode("latin1"), b.encode("latin1"))) # bytes 


        return cls(vocab, merges, special_tokens)
    
    def _split_by_special_tokens(self, text: str) -> list[int]:
        if not self.special_tokens:
            return [text]

        # escape regex metacharacters in tokens and build an | pattern
        pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        parts = re.split(pattern, text) # keep the tokens in (pattern) after split 
        return [part for part in parts if part != ""]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        
        encoded_results = []
        
        # ["hello", "<pad>", "world", "<bos>", "test"]
        # Step 1: split with special tokens and keep them 
        parts = self._split_by_special_tokens(text) 

        for part in parts:
            # 1. special tokens or part is a token in vocab 
            if part in self.special_tokens:
                part_bytes = part.encode("utf-8")
                encoded_results.append(self.token_to_id[part_bytes])
                continue

            # 2. normal tokens
            for m in self.rx.finditer(part):
                pre_tokenized_part = m.group(0)
                pre_tokenized_part_bytes = pre_tokenized_part.encode("utf-8")
                # text.encode("utf-8") is one bytes object
                # iterating over it gives int, so convert each int byte into bytes([b])
                tokens = [bytes([b]) for b in pre_tokenized_part_bytes]

                # bytes(5) -> length 5 with 0 bytes 
                # bytes([5]) -> bytes with content

                # repeatedly merge until no mergeable pair remains
                while True:
                    best_rank = None 
                    best_pair = None 

                    for i in range(len(tokens) - 1):
                        a, b = tokens[i], tokens[i + 1]
                        rank = self.merges_rank.get((a, b))

                        if rank is not None and (best_rank is None or rank < best_rank):
                            best_rank = rank 
                            best_pair = (a, b)

                    # no mergeable pairs
                    if best_pair is None:
                        break 
                    
                    # merge all occurences of best pair 
                    merged_tokens = []
                    i = 0 
                    while i < len(tokens):
                        if (i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]):
                            merged_tokens.append(tokens[i] + tokens[i + 1]) # concat bytes
                            i += 2
                        else:
                            merged_tokens.append(tokens[i])
                            i += 1

                    tokens = merged_tokens # update tokens
                
                # tokens (byte) -> token index 
                for tok in tokens:
                    token_idx = self.token_to_id.get(tok)
                    if token_idx is None:
                        raise ValueError(f"Token {tok!r} not found in vocab")
                    encoded_results.append(token_idx)
                
        return encoded_results


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for iter_str in iterable:
            token_ids = self.encode(iter_str)
            for token_id in token_ids:
                yield token_id



    def decode(self, ids: list[int]) -> str:
        
        byte_sequence = b"".join(self.vocab[id] for id in ids)
        
        return byte_sequence.decode("utf-8", errors="replace") # bytes -> readable str