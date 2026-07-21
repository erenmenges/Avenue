from datasets import load_dataset
import config

def get_dataset():
    ds = load_dataset(config.HF_DATASET, name= config.HF_CONFIG, split="train", cache_dir=str(config.CACHE_DIR),)
    return ds
