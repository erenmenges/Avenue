from datasets import load_dataset
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from dataset import get_dataset
import config

def sample_for_tokenizer():
    """
    Creates tokenizer corpus for the next step: tokenizer training.
    """
    ds = get_dataset()
    bytes_written = 0
    total_docs_written = 0

    next_milestone = 3e9

    with open(config.TOKENIZER_CORPUS_PATH, "w", encoding="utf-8") as f:
        for i in range(0, len(ds)):
            example = ds[i]
            text = example["text"].replace("\n", " ")
            line = text + "\n"
            f.write(line)

            total_docs_written += 1
            bytes_written += len(line.encode("utf-8"))
            
            if bytes_written > next_milestone:
                print(f"GB: {bytes_written / 1e9}, Docs written: {total_docs_written}")
                next_milestone += 3e9
        
    print("DONE: Sampling for tokenizer training")

def train_tokenizer(vocab_size: int = 16_384, special_tokens: list[str] | None = None):
    """
    Trains tokenizer with BPE.
    """

    special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens

    tokenizer = Tokenizer(models.BPE(byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=True), 
                                                       pre_tokenizers.ByteLevel(add_prefix_space=False)])  ### 
    tokenizer.decoder = decoders.ByteLevel()  ### decode back to real text from internal mappings (e.g. space to weird G)

    trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                    special_tokens=special_tokens,
                                    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                    show_progress=True)   ### set up the trainer
    
    tokenizer.train([str(config.TOKENIZER_CORPUS_PATH)], trainer)
    tokenizer.save(str(config.TOKENIZER_PATH))

    print("DONE: Training tokenizer.")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    print(tokenizer.encode("Hello New York from 2026!").tokens)

def measure_compression() -> float:
    """
    Measures the average byte per token from the training data.
    """
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    ds = get_dataset()

    total_bytes = 0
    total_tokens = 0
    batch = []
    batch_size = 1000
    num_docs_to_sample = 30000
    docs_processed = 0

    for i in range(0, len(ds), 3):  ### lets use a batch approach
        batch.append(ds[i]["text"])
        docs_processed += 1

        if len(batch) == batch_size:
            encodings = tokenizer.encode_batch(batch)
            total_tokens += sum(len(enc.ids) for enc in encodings)
            total_bytes += sum(len(text.encode("utf-8")) for text in batch)
            batch = []
            
            if docs_processed > num_docs_to_sample:
                break
    
    average_bytes_per_token = float(total_bytes / total_tokens)
    print(f"DONE: Measuring avg bytes per token. Avg bytes per token: {average_bytes_per_token}, documents sampled: {docs_processed}")
    return average_bytes_per_token




if __name__ == "__main__":
    if not config.TOKENIZER_CORPUS_PATH.exists():
        sample_for_tokenizer()
    if not config.TOKENIZER_PATH.exists():
        train_tokenizer()
    measure_compression()