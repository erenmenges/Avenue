import torch
import time
from pathlib import Path
from torch import nn
import model
import config
from data import get_batch

torch.manual_seed(config.SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

lm = model.Transformer(K=config.K, D=config.D, H=config.H, V=config.VOCAB_SIZE)
lm.to(device)
lm = torch.compile(lm)

optimizer = torch.optim.AdamW(params=lm.parameters(), betas=(0.9, 0.95), weight_decay=0.1)
loss_fn = nn.CrossEntropyLoss()

print("IN PROGRESS: Starting model training loop")
print(f"TRAIN.PY: Will train for {config.MAX_STEPS} steps.")

start = time.perf_counter()
tokens_trained = 0

for step in range(config.MAX_STEPS):
    x_b, y_b = get_batch("train")
    x_b, y_b = x_b.to(device), y_b.to(device)

    # bf16 for faster training
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        y_hat = lm(x_b)  ### (B, N, V)

        model_logits = y_hat.reshape(-1 , y_hat.shape[-1])  ### (B, N, V) --> (B * N, V)
        y_b = y_b.flatten()  ### (B, N) --> (B*N,)

        loss = loss_fn(model_logits, y_b)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tokens_trained += y_b.shape[0]

    if step % 100 == 0:
        with torch.no_grad():
            lm.eval()
            x_val_b, y_val_b = get_batch("val")
            x_val_b, y_val_b = x_val_b.to(device), y_val_b.to(device)

            y_val_hat = lm(x_val_b)  ### (B, N, V)

            val_model_logits = y_val_hat.reshape(-1 , y_val_hat.shape[-1])  ### (B, N, V) --> (B * N, V)
            y_val_b = y_val_b.flatten()

            val_loss = loss_fn(val_model_logits, y_val_b)

            torch.mps.synchronize()
            print(f"Step: {step}, train loss: {loss.item():.3f}, val_loss: {val_loss.item():.3f}, time elapsed: {time.perf_counter() - start:.2f}s, tokens_trained: {tokens_trained}, tok/s:{tokens_trained/(time.perf_counter() - start):.2f}")
            lm.train()