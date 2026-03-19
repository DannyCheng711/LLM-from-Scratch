from cs336_basics.bpe import train_bpe
import json 
import time
import tracemalloc

# raw bytes (\xff)
# readable text (b"of")

# --- TinyStories ---

INPUT_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000

tracemalloc.start()
start = time.time()

vocab, merges = train_bpe(
    INPUT_PATH,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

end = time.time()
current, peak = tracemalloc.get_traced_memory()

print(f"[TS] Training time: {(end - start)/60:.2f} minutes")
print(f"[TS] Peak memory: {peak / 1e6:.2f} MB")

# longest token
longest = max(vocab.values(), key=len)
print("[TS] Longest token:", longest, "length:", len(longest))


# save vocab and merges 
with open("./experiments/ts/vocab.json", "w", encoding="utf-8") as f:
    json.dump(
        {str(k): v.decode("latin1") for k, v in vocab.items()}, 
        f, ensure_ascii=False, indent=2
    )

with open("./experiments/ts/merges.txt", "w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")


# --- OpenWebText ---

INPUT_PATH = "tests/fixtures/OpenWebText.txt"
VOCAB_SIZE = 32000

tracemalloc.start()
start = time.time()

vocab, merges = train_bpe(
    INPUT_PATH,
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)

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

