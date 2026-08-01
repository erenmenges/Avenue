from tokenizers import Tokenizer
import model
import torch
import argparse
from pathlib import Path
import config
import torch
from torch import nn
from data import get_batch, reset_rngs
import math
import json


torch.manual_seed(config.SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

with open(config.SPLIT_MANIFEST_PATH) as f:

    val_bpt = json.load(f)["val_bpt"]

def load_model(checkpoint_path, quantize_to_ternary: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    lm = model.Transformer(K=checkpoint_config["K"], D=checkpoint_config["D"], H=checkpoint_config["H"], V=checkpoint_config["V"], ternary=checkpoint_config["IS_TERNARY"])
    lm.to(device)
    lm.load_state_dict(state_dict=checkpoint["model"])

    if quantize_to_ternary:
        assert checkpoint_config["IS_TERNARY"] == False  ### do not try to PTQ an already quantization trained model
        with torch.no_grad():
            for block in lm.main:
                for layer in (block.Q_layer, block.K_layer, block.V_layer, block.O_layer, block.MLP[0], block.MLP[2]):
                    w_q, mu = model.quantize_weights(layer.weight)
                    layer.weight.copy_(w_q * mu)
    return lm


def evaluate(lm: model.Transformer, loss_fn: nn.CrossEntropyLoss, eval_size: int):
    reset_rngs(split="val")

    with torch.no_grad():
        lm.eval()
        val_losses = torch.zeros(eval_size)
        for i in range(eval_size):
            x_val_b, y_val_b = get_batch("val")
            x_val_b, y_val_b = x_val_b.to(device), y_val_b.to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                y_val_hat = lm(x_val_b)  ### (B, N, V)
                val_model_logits = y_val_hat.reshape(-1 , y_val_hat.shape[-1])  ### (B, N, V) --> (B * N, V)
                y_val_b = y_val_b.flatten()
                val_loss = loss_fn(val_model_logits, y_val_b)
            val_losses[i] = val_loss.item()

        lm.train()

        val_losses_mean = val_losses.mean().item()

        ## bpb calculation
        val_bpb = val_losses_mean / (math.log(2) * val_bpt)

        return (val_losses_mean, val_bpb)


def predict(lm: model.Transformer, temperature: float = 1.0, max_tokens: int = 1024, repetition_penalty: int = 1.2):
    prompt = input("Enter a prompt: ")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    prompt_tokenized = tokenizer.encode(prompt).ids
    prompt_tokenized = torch.tensor(prompt_tokenized, dtype=torch.long, device=device)[None, :]  ### (1, N)
    context = prompt_tokenized
    assert max_tokens <= config.SEQ_LEN

    print(prompt, end="")
    for _ in range(max_tokens - prompt_tokenized.shape[1]):
        seen_logits = torch.ones((1, config.VOCAB_SIZE), device=device)  ### (1, V)
        seen_logits[0, context[0][-64:]] = repetition_penalty  ### only apply the penalty to last 64 tokens (only remember last 64 tokens)

        logits = lm(context)[:, -1]
        logits = logits / temperature  ### apply temperature
        logits = torch.where(logits > 0, logits / seen_logits, logits * seen_logits) ### repetition penalty

        probabilities = torch.softmax(logits, dim=-1)   ### softmax the logits to turn them into a probability distribution
        prediction = torch.multinomial(probabilities, num_samples=1)
        print(tokenizer.decode(prediction[0].tolist()), end="", flush=True)
        if prediction.item() == config.EOS_ID:
            break
        context = torch.cat((context, prediction), dim=1)

    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a model checkpoint and run it")
    parser.add_argument("checkpoint_path", type=str, help="Location of the model")
    args = parser.parse_args()
    lm = load_model(args.checkpoint_path)
    predict(lm)
