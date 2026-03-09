import regex as re 
from collections import defaultdict

def train_bpe(input_path, vocab_size, special_tokens):
    # contraction suffix ('s), seq of letter, seq of number, seq of symbol, trailing whitespace, general whitespace
    # leading whitespace is more usual in a word 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Pre-tokenize
    chunks = re.findall(PAT, text)
    print(chunks[:20])

    chunks = [chunk.encode("utf-8") for chunk in chunks] # string to bytes 

    # Pairs of byte tokens inside each chunk
    # b is int, bytes expect an iterable of integers
    counts = defaultdict(int)
    for chunk in chunks:
        tokens = [bytes([b]) for b in chunk]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            counts[pair] += 1

    
    # merge 
    num_merge = vocab_size - 256 - len(special_tokens)

    # for i in range(num_merge):


        
    
        






    vocab = None 
    merges = None 
    return vocab, merges