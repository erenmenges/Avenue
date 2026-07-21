from pathlib import Path
import numpy as np

HF_DATASET = "HuggingFaceFW/fineweb-edu"
HF_CONFIG  = "sample-10BT"

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"

TOKENIZER_CORPUS_PATH = DATA_DIR / "tokenizer_corpus.txt"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"

TRAIN_BIN = DATA_DIR / "train.bin"
VAL_BIN = DATA_DIR / "val.bin"

SPLIT_MANIFEST_PATH = DATA_DIR / "split_manifest.json"

SEED = 42

EOS_TOKEN = "<|endoftext|>"
EOS_ID = 0
VOCAB_SIZE = 16384
TOKEN_DTYPE = np.uint16

SEQ_LEN = 1024
BATCH_SIZE = 64

K = 6
D = 256
H = 8
TRAINING_TOKEN_BUDGET = 160_000_000
MAX_STEPS = TRAINING_TOKEN_BUDGET // (SEQ_LEN * BATCH_SIZE) 