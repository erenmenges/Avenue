import time
import config
from tokenizers import Tokenizer
from pathlib import Path
from datasets import load_dataset
import numpy as np
import json
from dataset import get_dataset

ds = get_dataset()

tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))

def pack(start_index: int, output_path: Path, token_budget: int):
    """
    Start from an index, and encode & pack number of token_budget items.
    """
    start_time = time.perf_counter()
    print(f"IN PROGRESS: Packing {str(output_path)}")

    assert tokenizer.token_to_id(config.EOS_TOKEN) == config.EOS_ID, "endoftext token is not 0"

    batch_size = 1000

    buffer_size = 10_000_000
    buffer = []

    tokens_written = 0
    last_index = None

    with open(output_path, "wb") as f:
        for i in range(start_index, len(ds), batch_size):
            texts = ds[i : i + batch_size]["text"]

            encoded_batch = tokenizer.encode_batch(texts)

            # put encodings to buffer
            for encoded_doc in encoded_batch:
                buffer.extend(encoded_doc.ids)
                buffer.append(config.EOS_ID)
                tokens_written += (len(encoded_doc.ids) + 1)
                
            # flush buffer
            if (len(buffer) >= buffer_size):
                buffer = np.array(buffer, dtype=config.TOKEN_DTYPE)
                buffer.tofile(f)
                buffer = []

            if tokens_written >= token_budget:
                break

        # final flush
        buffer = np.array(buffer, dtype=config.TOKEN_DTYPE)
        buffer.tofile(f)
        buffer = []
        last_index = len(texts) + i  - 1 ### add len texts since i only records starts of each batch
    
    time_taken = time.perf_counter() - start_time
    print(f"DONE: Packing {str(output_path)}, Time elapsed packing: {time_taken}s")
    return (tokens_written, last_index)

def prepare(val_token_budget: int, train_token_budget: int):
    val_n_tokens_written, val_last_index = pack(0, config.VAL_BIN, val_token_budget)
    train_n_tokens_written, train_last_index = pack(val_last_index + 1, config.TRAIN_BIN, train_token_budget)

    manifest = {"val_n_tokens_written": val_n_tokens_written,
                "val_last_index": val_last_index,
                "train_n_tokens_written": train_n_tokens_written,
                "train_last_index": train_last_index}
    
    # verify bin correctness
    verify(config.VAL_BIN, val_n_tokens_written)
    verify(config.TRAIN_BIN, train_n_tokens_written)

    with open(config.SPLIT_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    print(manifest)

def verify(bin_path: Path, reported_tokens: int):
    """
    Verify packing correctness. Output needs human review as well.
    """
    print(f"IN PROGRESS: Verify {str(bin_path)}")

    # does token number match or not
    n_tokens = bin_path.stat().st_size // 2
    print(f"Actual number of tokens: {n_tokens}, reported tokens: {reported_tokens}")

    bin_arr = np.memmap(bin_path, dtype=config.TOKEN_DTYPE, mode="r")

    # correctness etc
    print(f"tokens: {n_tokens}")
    print(f"min id: {bin_arr.min()}, max id: {bin_arr.max()}")
    assert bin_arr.max() < config.VOCAB_SIZE, f"token {bin_arr.max()} more than vocab size"

    # sample some text and check it manually
    sample = bin_arr[n_tokens//2 : n_tokens//2 + 100_000].tolist()
    print(f"EOS density: {len([token for token in sample if token == config.EOS_ID])} per 100,000 tokens")

    sample = bin_arr[n_tokens//2 : n_tokens//2 + 1000].tolist()
    print(f"Sample: {tokenizer.decode(sample, skip_special_tokens=False)}")
    print(f"DONE: Evaluated {str(bin_path)}. Double check the results.")

if __name__ == "__main__":
    if not config.SPLIT_MANIFEST_PATH.exists():
        prepare(val_token_budget=10_000_000, train_token_budget=600_000_000)
