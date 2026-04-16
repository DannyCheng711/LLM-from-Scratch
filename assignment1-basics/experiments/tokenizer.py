from cs336_basics.tokenizer import Tokenizer
import regex as re 
import random
import time 
import numpy as np 

def split_documents(text, separator):
    pattern = re.escape(separator) 
    docs = [doc for doc in re.split(pattern, text) if doc.strip()] 
    return docs

# chunk_size = 1 MB
def split_documents_stream(input_path, separator, chunk_size=1024 * 1024): 
    pattern = re.escape(separator)
    buffer = ""

    with open(input_path, "r", encoding="utf-8") as f:
        chunk = f.read(chunk_size)
        while chunk:
            buffer += chunk
            parts = re.split(pattern, buffer)
            # the last part might not be a doc
            for doc in parts[:-1]:
                if doc.strip():
                    yield doc

            buffer = parts[-1]
            chunk = f.read(chunk_size) # move to the next chunk 
        
    if buffer.strip():
        yield buffer


# P(A_k is selected) = 1 / k  
# P(A_k is not replaced afterward) = k / N
def reservoir_sample(iterator, k):
    samples = []
    
    for i, item in enumerate(iterator):
        # create sampling set
        if i < k:
            samples.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                samples[j] = item
                
    return samples

# compression ratio (bytes/token) = bytes in original text / token num after tokenized
def compute_compression_ratio(tokenizer, docs):
    total_bytes = 0
    total_tokens = 0

    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    return total_bytes / total_tokens if total_tokens > 0 else 0.0

# throughput (bytes/s) = bytes in original text / time for encoding (s) 
def compute_throughput(tokenizer, docs):
    total_bytes = 0
    total_time = 0

    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        st = time.time()
        tokenizer.encode(doc)
        comp_time = time.time() - st 
        
        total_time += comp_time

    return total_bytes / total_time if total_time > 0 else 0.0


def encode_to_numpy(tokenizer, docs, output_path):
    all_ids = []

    for doc in docs:
        all_ids.extend(tokenizer.encode(doc))

    arr = np.array(all_ids, dtype=np.uint16)
    np.save(output_path, arr)

if __name__ == "__main__":
    random.seed(42)
    special_tokens = ["<|endoftext|>"]
    ts_input_path = "data/TinyStoriesV2-GPT4-train.txt"
    ts_valid_path = "data/TinyStoriesV2-GPT4-valid.txt"
    owt_input_path = "data/owt_train.txt"
    owt_valid_path = "data/owt_valid.txt"

    # --- TinyStories ---
    print(f"[TS] Reading vocab and merge of TinyStories ... ")
    ts_tokenizer = Tokenizer.from_files(
        vocab_filepath="experiments/ts/vocab.json", merges_filepath="experiments/ts/merges.txt", special_tokens=special_tokens)
    
    # iterator 
    ts_docs = split_documents_stream(ts_input_path, separator=special_tokens)

    # --- OpenWebText ---
    print(f"[OWT] Reading vocab and merge of OpenWebText ... ")
    owt_tokenizer = Tokenizer.from_files(
        vocab_filepath="experiments/owt/vocab.json", merges_filepath="experiments/owt/merges.txt", special_tokens=special_tokens)

    owt_docs = split_documents_stream(owt_input_path, separator=special_tokens)

    # --- Sample 10 documents ---
    print(f"[TS] Sampling doc in TinyStories ... ")
    print(f"[OWT] Sampling doc in OpenWebText ... ")
    ts_sample = reservoir_sample(ts_docs, 10)
    owt_sample = reservoir_sample(owt_docs, 10)


    # --- Compute Compression Ratio --- 
    print(f"[TS] Compute compression ratio in TinyStories ... ")
    print(f"[OWT] Compute compression ratio in OpenWebText ... ")
    ts_ratio = compute_compression_ratio(ts_tokenizer, ts_sample)
    owt_ratio = compute_compression_ratio(owt_tokenizer, owt_sample)

    print(f"TinyStories compression ratio: {ts_ratio:.4f} bytes/token")
    print(f"OpenWebText compression ratio: {owt_ratio:.4f} bytes/token")


    # --- Use TS Tokenizer on OWT Data ---
    # --- Compute Compression Ratio --- 
    print(f"[TS] Compute compression ratio in OpenWebText ... ")
    ts_owt_ratio = compute_compression_ratio(ts_tokenizer, owt_sample)

    print(f"TinyStories compression ratio on OWT: {ts_owt_ratio:.4f} bytes/token")


    # --- Compute Throughput --- 
    print(f"[TS] Compute throughput in TinyStories ... ")
    print(f"[OWT] Compute throughput in OpenWebText ... ")
    ts_throughput= compute_throughput(ts_tokenizer, ts_sample)
    owt_throughput = compute_throughput(owt_tokenizer, owt_sample)

    print(f"TinyStories throughput: {ts_throughput:.4f} bytes/s")
    print(f"OpenWebText throughput: {owt_throughput:.4f} bytes/s")

    # --- Serializing the token IDs ---
    ts_docs = split_documents_stream(ts_input_path, separator=special_tokens) # build again, as the iterator is comsumed before
    owt_docs = split_documents_stream(owt_input_path, separator=special_tokens)
    encode_to_numpy(ts_tokenizer, ts_docs, "experiments/ts/ts_train_ids.npy")
    encode_to_numpy(owt_tokenizer, owt_docs, "experiments/owt/owt_train_ids.npy")

    ts_valid_docs = split_documents_stream(ts_valid_path, separator=special_tokens) # build again, as the iterator is comsumed before
    owt_valid_docs = split_documents_stream(owt_valid_path, separator=special_tokens)
    encode_to_numpy(ts_tokenizer, ts_valid_docs, "experiments/ts/ts_valid_ids.npy")
    encode_to_numpy(owt_tokenizer, owt_valid_docs, "experiments/owt/owt_valid_ids.npy")

 
    

