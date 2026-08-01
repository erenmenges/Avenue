import torch
import time
from torch import nn
from torch.nn import LinearCrossEntropyOptions
import torch.nn.functional as F
import model
import config
import math
import wandb
import argparse
from pathlib import Path
from data import get_batch, reset_rngs, get_rng_states, set_rng_states
from eval_inference import evaluate

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



def save_checkpoint(model: model.Transformer, model_config: dict, optimizer: torch.optim.AdamW, step: int, peak_lr: float, wandb_run_id: str, run_dir: Path, final: bool = False):
    savedict = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": model_config,
                "wandb_run_id": wandb_run_id,
                "rng_states": get_rng_states()}
    timestamp = time.strftime("%m%d-%H%M")
    total_n_of_params = sum(param.numel() for param in model.parameters()) / 1e6
    if not final:
        save_path = run_dir / f"ckpt_{total_n_of_params:.0f}M_{timestamp}_lr{peak_lr:.5f}_step{step:06d}.pt"
    else:
        save_path = run_dir / "final.pt"
    tmp_path = save_path.with_suffix(".tmp")
    torch.save(savedict, tmp_path)
    tmp_path.rename(save_path)
    print(f"CHECKPOINT: saved at step {step} to {save_path}")

def train(peak_lr: float = config.PEAK_LR, seed: int = config.SEED, resume_path: str | None = None, is_ternary: bool = False):
    torch.manual_seed(seed)

    if not resume_path:
        raw_lm = model.Transformer(K=config.K, D=config.D, H=config.H, V=config.VOCAB_SIZE, ternary=is_ternary)
        raw_lm.to(device)
        model_config = {"K":config.K ,"D":config.D, "H":config.H, "V": config.VOCAB_SIZE, "PEAK_LR": peak_lr, "IS_TERNARY": is_ternary}
        reset_rngs("both")

        # selective weight decay
        params_to_decay = [param for param in raw_lm.parameters() if param.dim() >= 2]
        params_to_not_decay = [param for param in raw_lm.parameters() if param.dim() < 2]
        optimizer = torch.optim.AdamW(params=[{"params": params_to_decay, "weight_decay": 0.1},
                                            {"params": params_to_not_decay, "weight_decay": 0.0}],
                                        lr=peak_lr ,betas=(0.90, 0.95))  ### selective weight decay on only the matrices. no weight decay on layernorm.
        lce_options = LinearCrossEntropyOptions(batch_chunk_size=16384, chunking_method=None, acc_policy="accurate")
        eval_loss_fn = nn.CrossEntropyLoss()
        start_step = 0
        tokens_trained = 0
        session_start_tokens = 0
    else:
        checkpoint = torch.load(resume_path, map_location=device)
        checkpoint_config = checkpoint["config"]
        model_config = {"K":checkpoint_config["K"] ,"D":checkpoint_config["D"], "H":checkpoint_config["H"], "V": checkpoint_config["V"], "PEAK_LR": checkpoint_config["PEAK_LR"], "IS_TERNARY": checkpoint_config["IS_TERNARY"]}
        raw_lm = model.Transformer(K=model_config["K"], D=model_config["D"], H=model_config["H"], V=model_config["V"], ternary=model_config["IS_TERNARY"])
        raw_lm.to(device)
        raw_lm.load_state_dict(checkpoint["model"])
        params_to_decay = [param for param in raw_lm.parameters() if param.dim() >= 2]
        params_to_not_decay = [param for param in raw_lm.parameters() if param.dim() < 2]
        peak_lr = checkpoint_config["PEAK_LR"]
        optimizer = torch.optim.AdamW(params=[{"params": params_to_decay, "weight_decay": 0.1},
                                                    {"params": params_to_not_decay, "weight_decay": 0.0}],
                                                lr=peak_lr ,betas=(0.9, 0.95))
        optimizer.load_state_dict(checkpoint["optimizer"])
        lce_options = LinearCrossEntropyOptions(batch_chunk_size=16384, chunking_method=None, acc_policy="accurate")
        eval_loss_fn = nn.CrossEntropyLoss()
        start_step = checkpoint["step"] + 1
        tokens_trained = start_step * config.BATCH_SIZE * config.SEQ_LEN
        session_start_tokens = tokens_trained
        set_rng_states(checkpoint["rng_states"])


    lm = torch.compile(raw_lm)  ### separate compile() optimized model from raw model to handle saving properly

    # initialize wandb
    if not resume_path:
        n_params_in_millions = sum(p.numel() for p in raw_lm.parameters()) / 1e6
        architecture_type = "fp" if not is_ternary else "ternary"
        run = wandb.init(project="Avenue",
                            name=f"{architecture_type}_{n_params_in_millions:.0f}M_lr{peak_lr:.1e}_seed{seed}",
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
    else:
        run = wandb.init(project="Avenue", id=checkpoint["wandb_run_id"], resume="must")

    # initialize the dir where checkpoints will be saved
    RUN_DIR = config.CHECKPOINT_DIR / f"run_{run.id}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("TRAIN.PY - IN PROGRESS: Starting model training loop")
    print(f"TRAIN.PY: Will train for {config.MAX_STEPS} steps.")
    print(f"TRAIN.PY - Model config: {model_config}")

    # prepare for flip rate calculation
    if is_ternary:
        previous_weight = None
        current_weight = None


    start = time.perf_counter()
    start2 = time.perf_counter()
    compute_times = []

    for step in range(start_step, config.MAX_STEPS):
        # lr scheduling
        lr = get_lr(step, peak_lr=peak_lr)
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = lr


        # get the batch
        x_b, y_b = get_batch("train")
        x_b, y_b = x_b.to(device), y_b.to(device)

        torch.mps.synchronize()
        t0 = time.perf_counter()

        # forward pass, bf16 for faster training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            h = lm(x_b, return_hidden=True)  ### (B, N, V), fp32

            h = h.reshape(-1 , h.shape[-1]).to(dtype=torch.bfloat16)  ### (B, N, V) --> (B * N, V)
            W = raw_lm.embeddings.weight.to(dtype=torch.bfloat16)
            y_b = y_b.flatten()  ### (B, N) --> (B*N,)

            loss = F.linear_cross_entropy(h, W, y_b, reduction="mean", options=lce_options)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=lm.parameters(), max_norm=1.0)   ### grad clipping to avoid huge weird gradients making us take a large step
        optimizer.step()
        tokens_trained += y_b.shape[0]

        torch.mps.synchronize()
        compute_times.append(time.perf_counter() - t0)


        # log train loss every 40 step for more precise metrics
        if step % 10 == 0 and step != 0:
            metrics = {"train_loss": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "lr": lr,
                        "tokens_trained": tokens_trained,
                        "compute_tok_per_sec": (len(compute_times) * (x_b.shape[0] * x_b.shape[1])) / (sum(compute_times)),
                        "compute_frac": sum(compute_times) / (time.perf_counter() - start2),}
            print(f"Step: {step}, time elapsed: {time.perf_counter() - start:.2f}s, tokens_trained: {tokens_trained}, compute_tok_per_sec: {(len(compute_times) * (x_b.shape[0] * x_b.shape[1])) / (sum(compute_times))}tok/s, compute_frac: {sum(compute_times) / (time.perf_counter() - start2):2f}")
            compute_times = []
            start2 = time.perf_counter()
            
            ## measure flip rate if ternary
            if is_ternary:
                with torch.no_grad():
                    layer = raw_lm.main[0].Q_layer
                    current_weight, _ = model.quantize_weights(layer.weight)
                    if previous_weight is not None:
                        metrics["flip_rate_40"] = (current_weight != previous_weight).float().mean().item()
                        metrics["frac_zero"] = (current_weight == 0).float().mean().item()
                        metrics["latent_absmax"] = layer.weight.abs().max().item()
                        metrics["latent_absmean"] = layer.weight.abs().mean().item()
                    previous_weight = current_weight
            wandb.log(metrics, step=step)

        # eval
        if step % 100 == 0:
            val_loss, val_bpb = evaluate(lm, eval_loss_fn, 10)

            torch.mps.synchronize()
            # log to wandb
            wandb.log({
                        "val_loss": val_loss,
                        "val_bpb": val_bpb,
                        "tok_per_sec": (tokens_trained - session_start_tokens) / (time.perf_counter() - start)},
                        step=step)
            print(f"Step: {step}, train loss: {loss.item():.3f}, val_loss: {val_loss:.3f}, time elapsed: {time.perf_counter() - start:.2f}s, tokens_trained: {tokens_trained}, tok/s:{(tokens_trained - session_start_tokens)/(time.perf_counter() - start):.2f}, grad_norm: {grad_norm.item():.2f}, LR: {lr:.6f}")

        # save every 300 steps
        if step % 300 == 0 and step > start_step:
            save_checkpoint(raw_lm, model_config, optimizer, step, peak_lr, run.id, RUN_DIR)

    # final eval
    reset_rngs("val")
    final_val_loss, final_val_bpb = evaluate(lm, eval_loss_fn, 50)
    wandb.log({"final_val_loss": final_val_loss, "final_val_bpb": final_val_bpb}, step=step)

    save_checkpoint(raw_lm, model_config, optimizer, step, peak_lr, run.id, RUN_DIR, True)

    # finish wandb logging
    run.finish()
    
    # clear memory
    del lm, raw_lm, optimizer
    torch.mps.empty_cache()

if __name__ == "__main__":
    # resume training functionality
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, metavar="LEARNING_RATE", default=config.PEAK_LR, help="Set specific learning rate")
    parser.add_argument("--ternary", action="store_true", help="To train in ternary weights or not")
    parser.add_argument("--resume", type=str, metavar="CHECKPOINT_PATH", help="Path of checkpoint that is to be resumed")
    args = parser.parse_args()
    train(resume_path=args.resume, is_ternary=args.ternary, peak_lr=args.lr)