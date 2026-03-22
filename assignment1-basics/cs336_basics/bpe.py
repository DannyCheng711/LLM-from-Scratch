import regex as re 
from collections import defaultdict, Counter
from multiprocessing import Pool, cpu_count

# BPE training = rewrite unique tokenized chunks + maintain frequencies.
# 1 count pairs in unique words (weighted by frequency)
# 2 choose most frequent pair
# 3 merge pair in affected words
# 4 update pair frequencies incrementally


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _process_chunk(sp):
    """
    Build local word frequencies for one text span.
    """
    rx = re.compile(PAT)
    local_word_freq = Counter()
    for m in rx.finditer(sp):
        piece = m.group(0) # each token piece matched (split) by the regex PAT as input for byte-level BPE
        if piece: 
            local_word_freq[tuple(piece.encode("utf-8"))] += 1 # str → bytes → tuple
    return local_word_freq

def _count_pairs(word):
    """
    Count pair freq in a word 
    """
    counts = {}
    if len(word) < 2:
        return counts

    prev = word[0]
    for cur in word[1:]:
        pair = (prev, cur) # pair for adjacent words
        counts[pair] = counts.get(pair, 0) + 1
        prev = cur
    return counts

def _merge_pair_into_newword(word, pair, new_token):
    a, b = pair
    merged = []
    i = 0
    L = len(word)

    while i < L:
        # pair found
        if i < L - 1 and word[i] == a and word[i + 1] == b:
            merged.append(new_token)
            i += 2
        else:
            merged.append(word[i])
            i += 1

    return tuple(merged)

def _initialize_bpe_training(input_path, special_tokens):
    
    # contraction suffix ('s), seq of letter, seq of number, seq of symbol, trailing whitespace, general whitespace
    # leading whitespace is more usual in a word 
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # initialized
    vocab = {i: bytes([i]) for i in range(256)} # token_id -> token_bytes
    next_idx = 256
    # add special tokens to vocab
    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1

    # 0. Remove special tokens from training text
    # prepare spans
    spans = [text]
    # update spans with len(special_tokens) times
    for tok in special_tokens:
        new_spans = []
        for sp in spans:
            new_spans.extend(sp.split(tok))
        spans = new_spans

    return vocab, next_idx, spans

def _build_word_freq(spans):
    """
    Serial (non-parallel) fallback for building global word frequencies.
    """
    word_freq = Counter()
    for sp in spans:
        if not sp:
            continue
        word_freq.update(_process_chunk(sp))
    return word_freq

def _train_bpe_from_word_freq(vocab, next_idx, vocab_size, special_tokens, word_freq=None):
    """
    vocab: dict[int, bytes] # token_id -> token_bytes
    merges: list[tuple[bytes, bytes]] # merge record
    """

    merges = []
    n_merges = vocab_size - 256 - len(special_tokens)
    
    # 2. build pair statistics
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set) # ('t', 'h') -> (the, they)

    for word, freq in word_freq.items():
        local_pair_count = _count_pairs(word)
        for pair, occ in local_pair_count.items():
            pair_counts[pair] += occ * freq # pair occ * word freq = ttl pair counts
            pair_to_words[pair].add(word)

    # 3. iterative BPE
    for _ in range(n_merges):

        # find the max pair to merge
        # break ties by choosing the lexicographically greater pair
        (a, b), best_count = max(
            ((p, c) for p, c in pair_counts.items() if c > 0),
            key=lambda pair_count: (
                pair_count[1], (vocab[pair_count[0][0]], vocab[pair_count[0][1]])) # count, lexicographic
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
            local_pair_count = _count_pairs(word)

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
            new_word = _merge_pair_into_newword(word, (a, b), next_idx)
            add_back[new_word] += freq
        
        for new_word, freq in add_back.items():
            word_freq[new_word] = word_freq.get(new_word, 0) + freq

            local_pair_count = _count_pairs(new_word)
            for pair, occ in local_pair_count.items():
                pair_counts[pair] += occ * freq
                pair_to_words[pair].add(new_word)

        next_idx += 1

    return vocab, merges

def train_bpe(input_path, vocab_size, special_tokens):
    """
    Public API expected by the assignment tests.

    Returns:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    
    vocab, next_idx, spans = _initialize_bpe_training(input_path=input_path, special_tokens=special_tokens)
    word_freq = _build_word_freq(spans)
    vocab, merges = _train_bpe_from_word_freq(
        vocab=vocab, next_idx=next_idx, vocab_size=vocab_size, special_tokens=special_tokens, word_freq=word_freq)
    
    return vocab, merges