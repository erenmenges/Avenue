import argparse

from torch import nn

import config
from data import AvenueData
from eval_inference import evaluate, load_model


def calculate_size(K, D, V, ternary=False, ptq=False):
    ternary_or_fp = 4 * (D * D) + (4 * D * D) + (D * 4 * D)  ### linear/bitlinear weights
    norm_params_fp = 2 * D
    norm_params_ternary = 9 * D
    scales_ternary = 9 * D  ### 4D (QVKO) + 4D (MLP1) + D (MLP2)
    embeddings = V * D

    if ternary and not ptq:
        return K * ((ternary_or_fp // 4) + (2 * norm_params_ternary) + (2 * scales_ternary)) + (2 * (embeddings + D)), K * (ternary_or_fp + norm_params_ternary) + (embeddings + D)
    elif ptq:
        return K * ((ternary_or_fp // 4) + (2 * norm_params_fp) + (2 * scales_ternary)) + (2 * (embeddings + D)), K * (ternary_or_fp + norm_params_fp + scales_ternary) + (embeddings + D)
    else:
        return 2 * (K * (ternary_or_fp + norm_params_fp) + embeddings + D), (K * (ternary_or_fp + norm_params_fp) + embeddings + D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model to evaluate")
    parser.add_argument("model_path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ptq", action="store_true")
    args = parser.parse_args()
    model_to_eval, info = load_model(checkpoint_path=args.model_path, quantize_to_ternary=args.ptq, return_info=True)
    data = AvenueData(args.seed)
    is_ternary = args.ptq or info["IS_TERNARY"]
    loss_fn = nn.CrossEntropyLoss()
    val_loss, val_bpb = evaluate(lm=model_to_eval, eval_size=50, data=data)
    size, num_params = calculate_size(info["K"], info["D"], config.VOCAB_SIZE, is_ternary, args.ptq)
    print(f"Is ternary: {is_ternary}, K: {info['K']}, D: {info['D']}, H: {info['H']}")
    print(f"Val loss: {val_loss:3f}, Val bpb: {val_bpb:3f}")
    print(f"Number of parameters: {num_params:,}, Size: {size:,} bytes")
