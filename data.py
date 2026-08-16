import json

import numpy as np
import torch

import config


class AvenueData:
    def __init__(self, seed: int):
        self.seed = seed
        self.val_bin_arr = np.memmap(config.VAL_BIN, dtype=config.TOKEN_DTYPE, mode="r")
        self.train_bin_arr = np.memmap(config.TRAIN_BIN, dtype=config.TOKEN_DTYPE, mode="r")
        self.reset_rngs(split="both")

        with open(config.SPLIT_MANIFEST_PATH) as f:
            self.val_bpt = json.load(f)["val_bpt"]

    def reset_rngs(self, split: str = "both"):
        if split == "train":
            self.train_rng = np.random.default_rng(seed=self.seed)
        elif split == "both":
            self.train_rng = np.random.default_rng(seed=self.seed)
            self.val_rng = np.random.default_rng(seed=self.seed + 100)
        else:
            self.val_rng = np.random.default_rng(seed=self.seed + 100)

    def get_rng_states(self):
        return (self.train_rng.bit_generator.state, self.val_rng.bit_generator.state)

    def set_rng_states(self, states: tuple):
        self.train_rng.bit_generator.state = states[0]
        self.val_rng.bit_generator.state = states[1]

    def get_batch(self, split: str):
        """
        Creates a random batch of SEQ_LEN + 1.
        """
        rng = self.train_rng if split == "train" else self.val_rng
        bin_arr = self.val_bin_arr if split == "val" else self.train_bin_arr

        starts = rng.integers(low=0, high=len(bin_arr) - config.SEQ_LEN, size=config.BATCH_SIZE)

        batch = [bin_arr[start : start + config.SEQ_LEN + 1] for start in starts]
        batch = np.stack(batch).astype(np.int64)

        x_b = torch.from_numpy(batch[:, :-1])
        y_b = torch.from_numpy(batch[:, 1:])

        return (x_b, y_b)
