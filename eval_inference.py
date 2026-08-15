import argparse
import math

import torch
from tokenizers import Tokenizer
from torch import nn

import config
import model
from data import AvenueData

device = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(
    checkpoint_path, quantize_to_ternary: bool = False, return_info: bool = False
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    lm = model.Transformer(
        K=checkpoint_config["K"],
        D=checkpoint_config["D"],
        H=checkpoint_config["H"],
        V=checkpoint_config["V"],
        ternary=checkpoint_config["IS_TERNARY"],
    )
    lm.to(device)
    lm.load_state_dict(state_dict=checkpoint["model"])

    if quantize_to_ternary:
        assert (
            checkpoint_config["IS_TERNARY"] == False
        )  ### do not try to PTQ an already quantization trained model
        with torch.no_grad():
            for block in lm.main:
                for layer in (
                    block.Q_layer,
                    block.K_layer,
                    block.V_layer,
                    block.O_layer,
                    block.MLP[0],
                    block.MLP[2],
                ):
                    w_q = model.quantize_weights(layer.weight)
                    layer.weight.copy_(w_q)
    if return_info:
        return (lm, checkpoint_config)
    else:
        return lm


def evaluate(lm: model.Transformer, eval_size: int, data: AvenueData):
    data.reset_rngs(split="val")
    loss_fn = nn.CrossEntropyLoss()

    was_training = lm.training

    with torch.no_grad():
        lm.eval()
        val_losses = torch.zeros(eval_size)
        for i in range(eval_size):
            x_val_b, y_val_b = data.get_batch("val")
            x_val_b, y_val_b = x_val_b.to(device), y_val_b.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                y_val_hat = lm(x_val_b)  ### (B, N, V)
                val_model_logits = y_val_hat.reshape(
                    -1, y_val_hat.shape[-1]
                )  ### (B, N, V) --> (B * N, V)
                y_val_b = y_val_b.flatten()
                val_loss = loss_fn(val_model_logits, y_val_b)
            val_losses[i] = val_loss.item()

        val_losses_mean = val_losses.mean().item()

        ## bpb calculation
        val_bpb = val_losses_mean / (math.log(2) * data.val_bpt)

        if was_training:
            lm.train()

        return (val_losses_mean, val_bpb)


def predict(
    lm: model.Transformer,
    temperature: float,
    repetition_penalty: float,
    top_k: int,
    top_p: float,
    min_p: float,
    seed: int,
    max_tokens: int,
):
    torch.manual_seed(seed)
    prompt = input("\033[31mEnter a prompt: \033[0m")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    prompt_tokenized = tokenizer.encode(prompt).ids
    context = torch.tensor(prompt_tokenized, dtype=torch.long, device=device)[
        None, :
    ]  ### (1, N)
    assert max_tokens <= config.SEQ_LEN

    print(prompt, end="")
    lm.eval()
    with torch.no_grad():
        for _ in range(max_tokens - context.shape[1]):
            seen_logits = torch.ones((1, config.VOCAB_SIZE), device=device)  ### (1, V)
            seen_logits[0, context[0][-64:]] = (
                repetition_penalty  ### only apply the penalty to last 64 tokens (only remember last 64 tokens)
            )

            logits = lm(context)[:, -1]
            logits = logits / temperature  ### apply temperature
            logits = torch.where(
                logits > 0, logits / seen_logits, logits * seen_logits
            )  ### repetition penalty

            probabilities = torch.softmax(
                logits, dim=-1
            )  ### softmax the logits to turn them into a probability distribution

            # min p sampling
            threshold = min_p * probabilities.max(dim=-1, keepdim=True).values
            probabilities = torch.where(
                probabilities >= threshold,
                probabilities,
                torch.zeros_like(probabilities),
            )  ### min p sampling
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            )  ### renormalize probabilities to account for min p

            # top k sampling
            kth_largest_probability = torch.topk(probabilities, k=top_k, dim=-1).values[
                ..., -1:
            ]
            probabilities = torch.where(
                probabilities >= kth_largest_probability,
                probabilities,
                torch.zeros_like(probabilities),
            )  ### top k sampling
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            )  ### renormalize probabilities to account for top k

            # top p sampling
            sorted_probabilities, sorted_indices = torch.sort(
                probabilities, descending=True, dim=-1
            )
            cumulative_sum = torch.cumsum(sorted_probabilities, dim=-1)
            keep = (
                (cumulative_sum - sorted_probabilities) < top_p
            )  ### we shift the cumsum to the right so the cutoff happens after adding the probability that makes it reach top_p
            keep = torch.zeros_like(keep).scatter(
                dim=-1, index=sorted_indices, src=keep
            )  ### turn the sorted keep tensor to a probabilities-like indexed one
            probabilities = torch.where(
                keep, probabilities, torch.zeros_like(probabilities)
            )  ### top p sampling
            probabilities = probabilities / probabilities.sum(
                dim=-1, keepdim=True
            )  ### renormalize probabilities to account for top p

            prediction = torch.multinomial(probabilities, num_samples=1)
            print(tokenizer.decode(prediction[0].tolist()), end="", flush=True)
            if prediction.item() == config.EOS_ID:
                break
            context = torch.cat((context, prediction), dim=1)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint and run it")
    parser.add_argument("checkpoint_path", type=str, help="Location of the model")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=16384)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.12)
    parser.add_argument("--repetition-penalty", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--ptq", action="store_true")
    args = parser.parse_args()
    lm = load_model(args.checkpoint_path, quantize_to_ternary=args.ptq)
    predict(
        lm=lm,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )
