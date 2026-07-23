import torch
import time
from torch import nn
import model
import config
import math
import wandb
from data import get_batch, reset_rngs

device = "mps" if torch.backends.mps.is_available() else "cpu"
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# helper function for LR schedule
def get_lr(step: int, peak_lr: float = config.PEAK_LR) -> float:
    # warmup
    min_lr = peak_lr / 10
    if step < config.WARMUP_STEPS:
        return peak_lr * ((step + 1) / config.WARMUP_STEPS)

    # cosine decay
    p = (step - config.WARMUP_STEPS) / (config.MAX_STEPS - config.WARMUP_STEPS)
    decay_factor = (math.cos(math.pi * p) + 1) / 2
    return min_lr + ((peak_lr - min_lr) * decay_factor)


def save_checkpoint(model: model.Transformer, optimizer: torch.optim.AdamW, step: int, peak_lr: float):
    savedict = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": {"K":config.K ,"D":config.D, "H":config.H, "V": config.VOCAB_SIZE}}
    timestamp = time.strftime("%m%d-%H%M")
    total_n_of_params = sum(param.numel() for param in model.parameters()) / 1e6
    save_path = config.CHECKPOINT_DIR / f"ckpt_{total_n_of_params:.0f}M_{timestamp}_lr{peak_lr:.5f}_step{step:06d}.pt"
    tmp_path = save_path.with_suffix(".tmp")
    torch.save(savedict, tmp_path)
    tmp_path.rename(save_path)
    print(f"CHECKPOINT: saved at step {step} to {save_path}")




def train(peak_lr: float = config.PEAK_LR, seed: int = config.SEED):
    torch.manual_seed(seed)

    # separate compile() optimized model from raw model to handle saving properly
    raw_lm = model.Transformer(K=config.K, D=config.D, H=config.H, V=config.VOCAB_SIZE)
    raw_lm.to(device)
    lm = torch.compile(raw_lm)
    reset_rngs()

    # initialize wandb
    n_params_in_millions = sum(p.numel() for p in raw_lm.parameters()) / 1e6
    run = wandb.init(project="Avenue",
                        name=f"{n_params_in_millions:.0f}M_lr{peak_lr:.1e}_seed{seed}",
                        config={
                            "peak_lr": peak_lr,
                            "min_lr": peak_lr / 10,
                            "warmup_steps": config.WARMUP_STEPS,
                            "max_steps": config.MAX_STEPS,
                            "batch_size": config.BATCH_SIZE,
                            "seq_len": config.SEQ_LEN,
                            "K": config.K, "D": config.D, "H": config.H,
                            "vocab_size": config.VOCAB_SIZE,
                            "seed": seed,
                            "weight_decay": 0.1,
                            "grad_clip": 1.0,
                            "n_params": n_params_in_millions
                        },)

    # selective weight decay
    params_to_decay = [param for param in lm.parameters() if param.dim() >= 2]
    params_to_not_decay = [param for param in lm.parameters() if param.dim() < 2]

    optimizer = torch.optim.AdamW(params=[{"params": params_to_decay, "weight_decay": 0.1},
                                        {"params": params_to_not_decay, "weight_decay": 0.0}],
                                    lr=peak_lr ,betas=(0.9, 0.95))  ### selective weight decay on only the matrices. no weight decay on layernorm.
    loss_fn = nn.CrossEntropyLoss()

    print("IN PROGRESS: Starting model training loop")
    print(f"TRAIN.PY: Will train for {config.MAX_STEPS} steps.")

    start = time.perf_counter()
    tokens_trained = 0

    for step in range(config.MAX_STEPS):
        lr = get_lr(step, peak_lr=peak_lr)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr

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
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=lm.parameters(), max_norm=1.0)   ### grad clipping to avoid huge weird gradients making us take a large step
        optimizer.step()
        tokens_trained += y_b.shape[0]

        # log train loss every 20 step for more precise metrics
        if step % 20 == 0:
            wandb.log({"train_loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "lr": lr},
                        step=step)

        if step % 100 == 0:
            with torch.no_grad():
                lm.eval()
                val_losses = torch.zeros(15)
                for i in range(15):
                    x_val_b, y_val_b = get_batch("val")
                    x_val_b, y_val_b = x_val_b.to(device), y_val_b.to(device)

                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        y_val_hat = lm(x_val_b)  ### (B, N, V)
                        val_model_logits = y_val_hat.reshape(-1 , y_val_hat.shape[-1])  ### (B, N, V) --> (B * N, V)
                        y_val_b = y_val_b.flatten()
                        val_loss = loss_fn(val_model_logits, y_val_b)

                    val_losses[i] = val_loss.item()

                torch.mps.synchronize()
                # log to wandb
                wandb.log({"train_loss": loss.item(),
                            "val_loss": val_losses.mean().item(),
                            "grad_norm": grad_norm.item(),
                            "lr": lr,
                            "tokens_trained": tokens_trained,
                            "tok_per_sec": tokens_trained / (time.perf_counter() - start)},
                         step=step)
                print(f"Step: {step}, train loss: {loss.item():.3f}, val_loss: {val_losses.mean():.3f}, time elapsed: {time.perf_counter() - start:.2f}s, tokens_trained: {tokens_trained}, tok/s:{tokens_trained/(time.perf_counter() - start):.2f}, grad_norm: {grad_norm.item():.2f}, LR: {lr:.6f}")
                lm.train()
    save_checkpoint(raw_lm, optimizer, step, peak_lr)

    # finish wandb logging
    run.finish()
    
    # clear memory
    del lm, raw_lm, optimizer
    torch.mps.empty_cache()

if __name__ == "__main__":
    train()