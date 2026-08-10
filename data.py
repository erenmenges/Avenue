import numpy as np
import config
import torch

val_bin_arr = np.memmap(config.VAL_BIN, dtype=config.TOKEN_DTYPE, mode="r")
train_bin_arr = np.memmap(config.TRAIN_BIN, dtype=config.TOKEN_DTYPE, mode="r")

def reset_rngs(split: str = "both"):
    global train_rng, val_rng
    if split == "train":
        train_rng = np.random.default_rng(seed=config.SEED)
    elif split == "both":
        train_rng = np.random.default_rng(seed=config.SEED)
        val_rng = np.random.default_rng(seed=config.SEED + 100)
    else:
        val_rng = np.random.default_rng(seed=config.SEED + 100)

reset_rngs("both")

def get_rng_states():
    return (train_rng.bit_generator.state, val_rng.bit_generator.state)

def set_rng_states(states: tuple):
    train_rng.bit_generator.state = states[0]
    val_rng.bit_generator.state = states[1]


def get_batch(split: str):
    """
    Creates a random batch of SEQ_LEN + 1.
    """
    rng = train_rng if split == "train" else val_rng
    bin_arr = val_bin_arr if split == "val" else train_bin_arr

    starts = rng.integers(low=0, high=len(bin_arr) - config.SEQ_LEN, size=config.BATCH_SIZE)

    batch = [bin_arr[start: start + config.SEQ_LEN + 1] for start in starts]
    batch = np.stack(batch).astype(np.int64)

    x_b = torch.from_numpy(batch[:, :-1])
    y_b = torch.from_numpy(batch[:, 1:])

    return (x_b, y_b)
