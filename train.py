import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.nn import LinearCrossEntropyOptions

import config
import model
from data import AvenueData
from eval_inference import evaluate

device = "mps" if torch.backends.mps.is_available() else "cpu"
config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# helper function for LR schedule. return fraction of peak lr instead of absolute lr
def get_lr_mult(step: int, warmup_steps: int, max_steps: int) -> float:
    # warmup
    if step < warmup_steps:
        return (step + 1) / warmup_steps

    # cosine decay
    p = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_factor = (math.cos(math.pi * p) + 1) / 2
    return 0.1 + (0.9 * decay_factor)


def save_checkpoint(
    lm: model.Transformer,
    model_config: dict,
    optimizers: tuple,
    step: int,
    tokens_trained: int,
    total_token_budget: int,
    muon_lr: float,
    adamw_lr: float,
    wandb_run_id: str,
    run_dir: Path,
    data: AvenueData,
    final: bool = False,
):
    savedict = {
        "model": lm.state_dict(),
        "optimizers": {
            "muon": optimizers[0].state_dict(),
            "adamw": optimizers[1].state_dict(),
        },
        "step": step,
        "config": model_config,
        "tokens_trained": tokens_trained,
        "total_token_budget": total_token_budget,
        "wandb_run_id": wandb_run_id,
        "rng_states": data.get_rng_states(),
    }
    total_n_of_params = sum(param.numel() for param in lm.parameters()) / 1e6
    if not final:
        save_path = run_dir / f"ckpt_{total_n_of_params:.0f}M_muon_m{muon_lr:.1e}_a{adamw_lr:.1e}_step{step:06d}.pt"
    else:
        save_path = run_dir / f"final_{total_n_of_params:.0f}M_{total_token_budget / 1e7:.0f}BT_muon_m{muon_lr:.1e}_a{adamw_lr:.1e}.pt"
    tmp_path = save_path.with_suffix(".tmp")
    torch.save(savedict, tmp_path)
    tmp_path.rename(save_path)
    print(f"CHECKPOINT: saved at step {step} to {save_path}")


def build_optimizers(raw_lm: model.Transformer, muon_lr: float, adamw_lr: float):
    muon_params, decay_params, nondecay_params = [], [], []

    for name, p in raw_lm.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = name.startswith(("embeddings.", "output_head."))
        if p.ndim >= 2 and not is_embed:
            muon_params.append(p)
        elif p.ndim >= 2:
            decay_params.append(p)
        else:
            nondecay_params.append(p)

    total = len(muon_params) + len(decay_params) + len(nondecay_params)
    assert total == len(list(raw_lm.parameters())), "not all parameters are in the lists"
    assert all(p.ndim == 2 for p in muon_params), "muon was fed something non 2d"

    muon_optimizer = torch.optim.Muon(
        params=[{"params": muon_params, "weight_decay": 0.1}],
        lr=muon_lr,
        momentum=0.95,
        nesterov=True,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw_optimizer = torch.optim.AdamW(
        params=[
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nondecay_params, "weight_decay": 0.0},
        ],
        lr=adamw_lr,
        betas=(0.90, 0.95),
    )

    for optimizer in (muon_optimizer, adamw_optimizer):
        for g in optimizer.param_groups:
            g["base_lr"] = g["lr"]

    print(f"muon: {len(muon_params)} tensors | adamw decay: {len(decay_params)} | adamw nodecay: {len(nondecay_params)}")
    print(f"muon lr={muon_lr:.2e}  adamw lr={adamw_lr:.2e}")

    return muon_optimizer, adamw_optimizer


def train(
    K: int,
    D: int,
    H: int,
    token_budget: int,
    muon_lr: float,
    adamw_lr: float,
    seed: int,
    is_ternary: bool,
    resume_path: str | None = None,
):

    if not resume_path:
        torch.manual_seed(seed)
        data = AvenueData(seed)
        raw_lm = model.Transformer(K=K, D=D, H=H, V=config.VOCAB_SIZE, ternary=is_ternary)
        raw_lm.to(device)
        model_config = {
            "K": K,
            "D": D,
            "H": H,
            "V": config.VOCAB_SIZE,
            "MUON_LR": muon_lr,
            "ADAM_LR": adamw_lr,
            "SEED": seed,
            "IS_TERNARY": is_ternary,
        }
        data.reset_rngs("both")

        muon_optimizer, adamw_optimizer = build_optimizers(raw_lm, model_config["MUON_LR"], model_config["ADAM_LR"])
        lce_options = LinearCrossEntropyOptions(batch_chunk_size=16384, chunking_method=None, acc_policy="accurate")
        start_step = 0
        tokens_trained = 0
        session_start_tokens = 0
    else:
        checkpoint = torch.load(resume_path, map_location=device)
        model_config = {
            "K": checkpoint["config"]["K"],
            "D": checkpoint["config"]["D"],
            "H": checkpoint["config"]["H"],
            "V": checkpoint["config"]["V"],
            "MUON_LR": checkpoint["config"]["MUON_LR"],
            "ADAM_LR": checkpoint["config"]["ADAM_LR"],
            "SEED": checkpoint["config"]["SEED"],
            "IS_TERNARY": checkpoint["config"]["IS_TERNARY"],
        }
        raw_lm = model.Transformer(
            K=model_config["K"],
            D=model_config["D"],
            H=model_config["H"],
            V=model_config["V"],
            ternary=model_config["IS_TERNARY"],
        )
        raw_lm.to(device)
        raw_lm.load_state_dict(checkpoint["model"])

        muon_lr = model_config["MUON_LR"]
        adamw_lr = model_config["ADAM_LR"]
        muon_optimizer, adamw_optimizer = build_optimizers(raw_lm, muon_lr, adamw_lr)
        muon_optimizer.load_state_dict(checkpoint["optimizers"]["muon"])
        adamw_optimizer.load_state_dict(checkpoint["optimizers"]["adamw"])
        lce_options = LinearCrossEntropyOptions(batch_chunk_size=16384, chunking_method=None, acc_policy="accurate")
        start_step = checkpoint["step"] + 1
        tokens_trained = checkpoint["tokens_trained"]
        token_budget = checkpoint["total_token_budget"]
        session_start_tokens = tokens_trained
        seed = model_config["SEED"]
        torch.manual_seed(seed)
        data = AvenueData(seed)
        is_ternary = checkpoint["config"]["IS_TERNARY"]
        data.set_rng_states(checkpoint["rng_states"])

    lm = torch.compile(raw_lm)  ### separate compile() optimized model from raw model to handle saving properly

    max_steps = token_budget // (config.BATCH_SIZE * config.SEQ_LEN)
    warmup_steps = max(1, int(0.01 * max_steps))
    n_params = sum(p.numel() for p in raw_lm.parameters())
    n_params_in_millions = n_params / 1e6
    architecture_type = "fp" if not is_ternary else "ternary"

    # initialize wandb
    if not resume_path:
        run = wandb.init(
            project="Avenue",
            name=(f"{architecture_type}_{n_params_in_millions:.0f}M_muon_m{muon_lr:.1e}_a{adamw_lr:.1e}__H{H}_s{seed}"),
            config={
                "muon_lr": muon_lr,
                "adamw_lr": adamw_lr,
                "min_lr_frac": 0.1,
                "optimizer": "muon+adamw",
                "warmup_steps": warmup_steps,
                "max_steps": max_steps,
                "batch_size": config.BATCH_SIZE,
                "seq_len": config.SEQ_LEN,
                "K": K,
                "D": D,
                "H": H,
                "vocab_size": config.VOCAB_SIZE,
                "seed": seed,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "n_params": n_params_in_millions,
            },
        )
    else:
        run = wandb.init(project="Avenue", id=checkpoint["wandb_run_id"], resume="must")

    # initialize the dir where checkpoints will be saved
    RUN_DIR = config.CHECKPOINT_DIR / f"run_{run.id}_{n_params_in_millions:.0f}M_{token_budget / 1e9:.0f}BT_muon_m{muon_lr:.1e}_a{adamw_lr:.1e}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"TRAIN.PY: Starting training. Model has {n_params:,} parameters and will train for {max_steps:,} steps. Model config: {model_config}")

    # prepare for flip rate calculation
    if is_ternary:
        previous_weight = None
        current_weight = None

    start = time.perf_counter()

    for step in range(start_step, max_steps):
        # lr scheduling
        lr_mult = get_lr_mult(step, warmup_steps, max_steps)
        for optimizer in (muon_optimizer, adamw_optimizer):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = parameter_group["base_lr"] * lr_mult

        # get the batch
        x_b, y_b = data.get_batch("train")
        x_b, y_b = x_b.to(device), y_b.to(device)

        # forward pass, bf16 for faster training
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            h = lm(x_b, return_hidden=True)  ### (B, N, V), fp32

            h = h.reshape(-1, h.shape[-1]).to(dtype=torch.bfloat16)  ### (B, N, V) --> (B * N, V)
            W = raw_lm.embeddings.weight.to(dtype=torch.bfloat16)
            y_b = y_b.flatten()  ### (B, N) --> (B*N,)

            loss = F.linear_cross_entropy(h, W, y_b, reduction="mean", options=lce_options)

        # backprop
        muon_optimizer.zero_grad(set_to_none=True)
        adamw_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=lm.parameters(), max_norm=1.0)  ### grad clipping to avoid huge weird gradients making us take a large step
        muon_optimizer.step()
        adamw_optimizer.step()
        tokens_trained += y_b.shape[0]

        # log train loss every 50 steps for more precise metrics
        if step % 50 == 0 and step != 0:
            metrics = {
                "train_loss": loss.item(),
                "grad_norm": grad_norm.item(),
                "adamw_lr": adamw_optimizer.param_groups[0]["lr"],
                "muon_lr": muon_optimizer.param_groups[0]["lr"],
                "tokens_trained": tokens_trained,
            }

            ## measure flip rate if ternary
            if is_ternary:
                with torch.no_grad():
                    layer = raw_lm.main[0].Q_layer
                    current_weight = model.quantize_weights(layer.weight).sign()
                    if previous_weight is not None:
                        metrics["flip_rate_50"] = (current_weight != previous_weight).float().mean().item()
                        metrics["frac_zero"] = (current_weight == 0).float().mean().item()
                        metrics["latent_absmax"] = layer.weight.abs().max().item()
                        metrics["latent_absmean"] = layer.weight.abs().mean().item()
                    previous_weight = current_weight
            wandb.log(metrics, step=step)

        # eval
        if step % 300 == 0:
            val_loss, val_bpb = evaluate(lm, 10, data)

            torch.mps.synchronize()
            # log to wandb
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_bpb": val_bpb,
                    "tok_per_sec": (tokens_trained - session_start_tokens) / (time.perf_counter() - start),
                },
                step=step,
            )
            print(
                f"Step: {step} | train loss: {loss.item():.3f}  | val_loss: {val_loss:.3f} |",
                f"time elapsed: {time.perf_counter() - start:,.0f}s | tokens_trained: {tokens_trained:,} tokens |",
                f"tok/s:{(tokens_trained - session_start_tokens) / (time.perf_counter() - start):,.0f}tok/s |",
                f"grad_norm: {grad_norm.item():.2f} | MUON LR: {muon_optimizer.param_groups[0]['lr']:.6f} |",
                f"ADAMW LR: {adamw_optimizer.param_groups[0]['lr']:.6f}",
            )

        # save every 800 steps
        if step % 5000 == 0 and step > start_step:
            save_checkpoint(
                raw_lm,
                model_config,
                (muon_optimizer, adamw_optimizer),
                step,
                tokens_trained,
                token_budget,
                muon_lr,
                adamw_lr,
                run.id,
                RUN_DIR,
                data,
            )

    # final eval
    data.reset_rngs("val")
    final_val_loss, final_val_bpb = evaluate(lm, 50, data)
    wandb.log({"final_val_loss": final_val_loss, "final_val_bpb": final_val_bpb}, step=step)

    save_checkpoint(
        raw_lm,
        model_config,
        (muon_optimizer, adamw_optimizer),
        step,
        tokens_trained,
        token_budget,
        muon_lr,
        adamw_lr,
        run.id,
        RUN_DIR,
        data,
        True,
    )

    # finish wandb logging
    run.finish()

    # clear memory
    del lm, raw_lm, muon_optimizer, adamw_optimizer
    torch.mps.empty_cache()


if __name__ == "__main__":
    # resume training functionality
    parser = argparse.ArgumentParser()
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--adamw-lr", type=float)
    parser.add_argument("--K", type=int)
    parser.add_argument("--D", type=int)
    parser.add_argument("--H", type=int)
    parser.add_argument("--token-budget", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ternary", action="store_true", help="To train in ternary weights or not")
    parser.add_argument(
        "--resume",
        type=str,
        metavar="CHECKPOINT_PATH",
        help="Path of checkpoint that is to be resumed",
    )
    args = parser.parse_args()
    train(
        K=args.K,
        D=args.D,
        H=args.H,
        token_budget=args.token_budget,
        resume_path=args.resume,
        is_ternary=args.ternary,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        seed=args.seed,
    )
