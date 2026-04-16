# BPE Training Experiments

## TinyStories

### Setup

- Vocabulary size: 10,000
- Special tokens: `<|endoftext|>`
- Implementation: byte-level BPE with incremental updates (`word_freq` + `pair_to_words`)
- Hardware: M2 MacBook Pro

---

### Output

```
[TS] Training time: 7.71 minutes
[TS] Peak memory: 15594.28 MB
[TS] Longest token: b' accomplishment' length: 15
```

---

### Results

- Training time: **~7.71 minutes** (optimized with multiprocessing)
- Memory usage: **~15.6 GB**
- Longest token: `b' administration'`, length: 15

The longest token corresponds to a frequent substring in the dataset, which is expected since BPE merges commonly co-occurring byte sequences.

---

### Profiling


Profiling shows that most of the training time is spent in : 

- regex-based pre-tokenization and 
- pair statistics operations. 

This indicates that pretokenization is the primary CPU bottleneck in small-to-medium datasets, making it highly suitable for multiprocessing.

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
- Implementation: streaming-based preprocessing to avoid loading the entire corpus into memory

---

### Output

```
[OWT] Stage 1: initialize vocab
[OWT] Stage 1 done
[OWT] Stage 2: streaming word_freq
[OWT] Stage 2 done: 6601892 unique words
[OWT] Stage 3: BPE merge loop
[OWT] Stage 3 done
[OWT] Training time: 491.70 minutes
[OWT] Peak memory: 6135.18 MB
[OWT] Longest token: b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' length: 64
```

---

### Results

- Training time: **~491.7 minutes**
- Memory usage: **~6.1 GB**
- Longest token: `b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82' `, length: 64

The longest token appears to be a mojibake-like byte sequence, which is common in raw, unprocessed web-scale text such as OpenWebText. This reflects that the BPE tokenization process is driven purely by frequency patterns in byte sequences, rather than semantic meaning.

---

### Comparison

Compared to TinyStories:

- Unlike TinyStories, the bottleneck shifts from pretokenization to:
    
    1. Large-scale word_freq construction
	1. Expensive BPE merge loop over a much larger vocabulary space
	1. Loading the entire dataset at once is infeasible due to memory constraints.

- Therefore, streaming preprocessing is necessary to incrementally build word_freq without exceeding RAM limits.

--- 

### Observation

- Multiprocessing significantly accelerates pretokenization in TinyStories (~50 min → ~7 min).
- Streaming is essential for OWT to prevent memory overflow during preprocessing.
- For smaller datasets, CPU-bound preprocessing dominates and benefits from multiprocessing.
- For large datasets, memory and algorithmic complexity become the primary bottlenecks.

# Tokenizer Experiments

### Sampling 10 documents 

- TinyStories compression ratio: 3.8182 bytes/token
- OpenWebText compression ratio: 4.3372 bytes/token

###  Using TinyStories Tokenizer on OpenWebText
- Using the TinyStories tokenizer on OpenWebText yields a compression ratio of 3.3047 bytes per token, which is lower compared to using the OpenWebText tokenizer. The TinyStories tokenizer is trained on simpler data and cannot effectively capture the more diverse patterns in OpenWebText.


### Estimate the throughput 
-  The TinyStories tokenizer achieves a throughput of approximately 1.21 MB/s, while the OpenWebText tokenizer achieves around 1.02 MB/s. At this rate, tokenizing the 825GB Pile dataset would take approximately 8.5–10 days.

### Serializing the token IDs
- uint16 is appropriate because the vocabulary size is well below 65,536, allowing each token ID to be represented within 16 bits.