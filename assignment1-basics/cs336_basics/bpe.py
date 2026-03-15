import regex as re 
from collections import defaultdict, Counter

# BPE training = rewrite unique tokenized chunks + maintain frequencies.
# 1 count pairs in unique words (weighted by frequency)
# 2 choose most frequent pair
# 3 merge pair in affected words
# 4 update pair frequencies incrementally

def count_pairs(word):
    counts = defaultdict(int)
    for p in zip(word, word[1:]):
        counts[p] += 1
    return counts 

def merge_pair_into_newword(word, pair, new_token):
    a, b = pair
    merged = []
    i = 0

    while i < len(word):
        # pair found
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)


def train_bpe(input_path, vocab_size, special_tokens):
    """
    vocab: dict[int, bytes] # token_id -> token_bytes
    merges: list[tuple[bytes, bytes]] # merge record
    """
    # contraction suffix ('s), seq of letter, seq of number, seq of symbol, trailing whitespace, general whitespace
    # leading whitespace is more usual in a word 
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # initialized
    merges = []
    n_merges = vocab_size - 256 - len(special_tokens)
    vocab = {i: bytes([i]) for i in range(256)} # token_id -> token_bytes
    next_idx = 256
    # add special tokens to vocab
    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1

    # 0. Remove special tokens from training text
    spans = [text]
    # update spans with len(special_tokens) times
    for tok in special_tokens:
        new_spans = []
        for sp in spans:
            new_spans.extend(sp.split(tok))
        spans = new_spans

    # split with special tokens and PAT
    chunks = []
    for sp in spans:
        chunks.extend(re.findall(PAT, sp))
    

    # 1. Pre-tokenize 
    # Build word_freq
    chunk_counter = Counter(chunks)
    # chunk str-> bytes -> tuple of int
    word_freq = {}
    for chunk, freq in chunk_counter.items():
        word = tuple(chunk.encode("utf-8"))
        word_freq[word] = freq

    # 2. build pair statistics
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set) # ('t', 'h') -> (the, they)

    for word, freq in word_freq.items():
        local_pair_count = count_pairs(word)
        for pair, occ in local_pair_count.items():
            pair_counts[pair] += occ * freq # pair occ * word freq = ttl pair counts
            pair_to_words[pair].add(word)

    # 3. iterative BPE
    for _ in range(n_merges):

        # filter valid pairs
        valid_pairs = [(p, c) for p, c in pair_counts.items() if c > 0]
        if not valid_pairs:
            break
    
        # find the max pair to merge
        # break ties by choosing the lexicographically greater pair
        (a, b), best_count = max(
            valid_pairs,
            key=lambda pair_count: (pair_count[1], (vocab[pair_count[0][0]], vocab[pair_count[0][1]])) # count, lexicographic
        )

        merges.append((vocab[a], vocab[b]))
        vocab[next_idx] = vocab[a] + vocab[b]

        # update affected words
        affected_words = list(pair_to_words.get((a, b), set()))
        add_back = defaultdict(int)

        for word in affected_words:
            freq = word_freq.get(word)

            if freq is None: 
                continue 

            # remove old word contribution (easier)
            local_pair_count = count_pairs(word)

            for pair, occ in local_pair_count.items():
                # remove this word's contribution from the global pair counts
                pair_counts[pair] -= occ * freq
                # remove the pair entirely if no occurrences remain in the corpus
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                # remove this word from the words containing the pair
                if pair in pair_to_words:
                    pair_to_words[pair].discard(word)
                    # remove the pair entry if no words contain this pair anymore
                    if not pair_to_words[pair]:
                        pair_to_words.pop(pair)
            
            del word_freq[word]
            new_word = merge_pair_into_newword(word, (a, b), next_idx)
            add_back[new_word] += freq
        
        for new_word, freq in add_back.items():
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            local_pair_count = count_pairs(new_word)
            for pair, occ in local_pair_count.items():
                pair_counts[pair] += occ * freq
                pair_to_words[pair].add(new_word)

        next_idx += 1

    return vocab, merges
