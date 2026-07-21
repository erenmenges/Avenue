import torch
import time
from pathlib import Path
from torch import nn
import model
import config
import data

device = "mps "if torch.backends.mps.is_available() else "cpu"

lm = model.Transformer(K=config.K, D=config.D, H=config.H, K=config.VOCAB_SIZE)

optimizer = torch.optim.AdamW(params=lm.parameters(), betas=(0.9, 0.95), weight_decay=0.1, fused=True)
loss_fn = nn.CrossEntropyLoss()

tokens_per_step = config.SEQ_LEN * config.BATCH_SIZE
max_steps = config.TRAINING_TOKEN_BUDGET / tokens_per_step 

for step in range(max_steps):
    x_b, y_b = get