# BPE Training Experiments

## TinyStories

### Setup

- Vocabulary size: 10,000
- Special tokens: `<|endoftext|>`
- Implementation: byte-level BPE with incremental updates (`word_freq` + `pair_to_words`)
- Hardware: M2 MacBook Pro

---

### Results

- Training time: **~0.04 minutes (~2.4 seconds)**
- Memory usage: **~0.09 GB (~92 MB)**
- Longest token: `b"<|endoftext|>"` (length = 13)

The longest token corresponds to the special token `<|endoftext|>`, which is expected because special tokens are added directly to the vocabulary and are not affected by BPE merges.


---

### Profiling


Profiling shows that most of the training time is spent in : 

- regex-based pre-tokenization and 
- pair statistics operations. 

In particular, regex.findall dominates preprocessing, while repeated max selection over pair counts and pair counting contribute significantly during iterative merges.

```
Ordered by: cumulative time

ncalls  tottime  percall  cumtime  percall filename:lineno(function)
64/1    0.000    0.000    2.149    2.149 {built-in method builtins.exec}
1    0.103    0.103    2.149    2.149 train_bpe.py:1(<module>)
1    0.367    0.367    2.006    2.006 bpe.py:33(train_bpe)
6458    0.013    0.000    0.511    0.000 regex.py:331(findall)
6458    0.410    0.000    0.410    0.000 {method 'findall' of '_regex.Pattern' objects}
5715/5714    0.174    0.000    0.346    0.000 {built-in method builtins.max}
243    0.247    0.001    0.247    0.001 bpe.py:93(<listcomp>)
1    0.000    0.000    0.220    0.220 __init__.py:587(__init__)
1    0.000    0.000    0.220    0.220 __init__.py:660(update)
1    0.220    0.220    0.220    0.220 {built-in method _collections._count_elements}
737228    0.172    0.000    0.172    0.000 bpe.py:101(<lambda>)
51640    0.164    0.000    0.164    0.000 bpe.py:10(count_pairs)
```

---

### Notes

- Special tokens are removed before BPE training and added back to the vocabulary.
- Using `word_freq` significantly reduces redundant computation compared to naive corpus-level updates.
- Learned tokens tend to reflect common natural language patterns in stories.

---

## OpenWebText (OWT)

### Setup

- Vocabulary size: 32,000
- Special tokens: `<|endoftext|>`
- Implementation: same as TinyStories

---

### Results

- Longest token: `b"..."`

The longest token often corresponds to web-specific patterns such as URLs, HTML fragments, or code-like text, which reflects the noisy and diverse nature of web data.

---

### Comparison

Compared to TinyStories:

- TinyStories produces cleaner, word-like tokens (e.g., common phrases in narratives)
- OpenWebText produces more diverse tokens including punctuation, URLs, and markup-like patterns

This demonstrates that the learned tokenizer is highly dependent on the underlying data distribution.