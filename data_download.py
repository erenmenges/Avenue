from pathlib import Path
from datasets import load_dataset

cache_dir = Path(__file__).parent / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu", name= "sample-10BT", split="train", cache_dir=str(cache_dir),
)

print(ds)
print(ds[0]["text"][:500])