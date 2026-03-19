from cs336_basics.bpe import train_bpe

train_bpe(
    "tests/fixtures/corpus.en",
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
)