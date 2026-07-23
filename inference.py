from tokenizers import Tokenizer
import model
import torch
import argparse
from pathlib import Path
import config


torch.manual_seed(config.SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Load a model checkpoint and run it")
parser.add_argument("checkpoint_path", type=str, help="Location of the model")
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_path, map_location=device)
checkpoint_config = checkpoint["config"]
lm = model.Transformer(K=checkpoint_config["K"], D=checkpoint_config["D"], H=checkpoint_config["H"], V=checkpoint_config["V"])
lm.to(device)
lm.load_state_dict(state_dict=checkpoint["model"])

prompt = "Germany and France are countries, and London and Paris are "

def predict(model: model.Transformer, tokenizer: Path, prompt: str, temperature: float = 1.0, max_tokens: int = 1024):
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_PATH))
    prompt_tokenized = tokenizer.encode(prompt).ids
    prompt_tokenized = torch.tensor(prompt_tokenized, dtype=torch.long, device=device)[None, :]  ### (1, N)
    input = prompt_tokenized
    assert max_tokens <= config.SEQ_LEN

    print(prompt, end="")
    for _ in range(max_tokens - prompt_tokenized.shape[1]):
        logits = model(input)[:, -1]
        probabilities = torch.softmax(logits / temperature, dim=-1)   ### softmax the logits to turn them into a probability distribution
        prediction = torch.multinomial(probabilities, num_samples=1)
        print(tokenizer.decode(prediction[0].tolist()), end="", flush=True)
        if prediction.item() == config.EOS_ID:
            break
        input = torch.cat((input, prediction), dim=1)

    print()

print("Temperature: 1.3")
predict(lm, config.TOKENIZER_PATH, prompt, temperature=1.3)
print("Temperature: 1")
predict(lm, config.TOKENIZER_PATH, prompt, temperature=1)
print("Temperature: 0.7")
predict(lm, config.TOKENIZER_PATH, prompt, temperature=0.7)


