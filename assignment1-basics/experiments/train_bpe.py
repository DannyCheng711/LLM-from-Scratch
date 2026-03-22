from cs336_basics.bpe import train_bpe, _train_bpe_from_word_freq, _initialize_bpe_training, _process_chunk
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import json 
import time
import tracemalloc

# raw bytes (\xff)
# readable text (b"of")

# multiprocessing pattern:
# imap_unordered(func, iterable, chunksize)
# func: the worker function (what to run on each task)
# iterable: a collection of inputs; each element will be passed to func as func(x)
# chunksize: how many inputs are batched together and sent to each worker at once

def _build_word_freq_parallel(spans, num_workers=5):
    """
    Building global word frequencies.
    """
    word_freq = Counter()
    with Pool(num_workers) as pool:
        for local_word_freq in pool.imap_unordered(_process_chunk, spans, chunksize=100):
            word_freq.update(local_word_freq)
 
    return word_freq

def _build_word_freq_streaming_parallel(
    input_path, special_token, num_workers=6, batch_size=1000,chunksize=100):

    word_freq = Counter()
    buffer = ""
    doc_batch = []

    with Pool(num_workers) as pool:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                buffer += line
                parts = buffer.split(special_token)

                complete_docs = parts[:-1]
                buffer = parts[-1]

                for doc in complete_docs:
                    if doc:
                        doc_batch.append(doc)

                # once enough docs are accumulated, process them in parallel
                if len(doc_batch) >= batch_size:
                    for local_word_freq in pool.imap_unordered(_process_chunk, doc_batch, chunksize=chunksize):
                        word_freq.update(local_word_freq)
                    doc_batch = []

        # flush remaining buffer as one last document
        if buffer:
            doc_batch.append(buffer)

        # flush the last partial batch
        if doc_batch:
            for local_word_freq in pool.imap_unordered(_process_chunk, doc_batch, chunksize=chunksize):
                word_freq.update(local_word_freq)

    return word_freq

if __name__ == "__main__":
    # --- TinyStories ---
    # input_path = "data/TinyStoriesV2-GPT4-train.txt"
    # vocab_size = 10000
    # special_tokens = ["<|endoftext|>"]

    # tracemalloc.start()
    # start = time.time()

    # vocab, next_idx, spans = _initialize_bpe_training(input_path, special_tokens)

    # num_workers = min(5, cpu_count())
    # word_freq = _build_word_freq_parallel(spans, num_workers=num_workers)

    # vocab, merges = _train_bpe_from_word_freq(
    #     vocab, next_idx, vocab_size, special_tokens, word_freq
    # )

    # end = time.time()
    # current, peak = tracemalloc.get_traced_memory()

    # print(f"[TS] Training time: {(end - start)/60:.2f} minutes")
    # print(f"[TS] Peak memory: {peak / 1e6:.2f} MB")

    # # longest token
    # longest = max(vocab.values(), key=len)
    # print("[TS] Longest token:", longest, "length:", len(longest))

    # # save vocab and merges 
    # with open("./experiments/ts/vocab.json", "w", encoding="utf-8") as f:
    #     json.dump(
    #         {str(k): v.decode("latin1") for k, v in vocab.items()}, 
    #         f, ensure_ascii=False, indent=2
    #     )

    # with open("./experiments/ts/merges.txt", "w", encoding="utf-8") as f:
    #     for a, b in merges:
    #         f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")


    # --- OpenWebText ---

    input_path = "data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    tracemalloc.start()
    start = time.time()

    print("[OWT] Stage 1: initialize vocab", flush=True)
    vocab = {i: bytes([i]) for i in range(256)}
    next_idx = 256
    for tok in special_tokens:
        vocab[next_idx] = tok.encode("utf-8")
        next_idx += 1
    print("[OWT] Stage 1 done", flush=True)

    print("[OWT] Stage 2: streaming word_freq", flush=True)
    word_freq = _build_word_freq_streaming_parallel(input_path, special_tokens[0])
    print(f"[OWT] Stage 2 done: {len(word_freq)} unique words", flush=True)

    print("[OWT] Stage 3: BPE merge loop", flush=True)
    vocab, merges = _train_bpe_from_word_freq(
        vocab, next_idx, vocab_size, special_tokens, word_freq
    )

    print("[OWT] Stage 3 done", flush=True)

    end = time.time()
    current, peak = tracemalloc.get_traced_memory()

    print(f"[OWT] Training time: {(end - start)/60:.2f} minutes")
    print(f"[OWT] Peak memory: {peak / 1e6:.2f} MB")

    # longest token
    longest = max(vocab.values(), key=len)
    print("[OWT] Longest token:", longest, "length:", len(longest))

    # save vocab and merges 
    with open("./experiments/owt/vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v.decode("latin1") for k, v in vocab.items()}, 
            f, ensure_ascii=False, indent=2
        )

    with open("./experiments/owt/merges.txt", "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")
