"""
Create a 1 000-example train slice + 200-example dev slice from GSM8K,
normalise each entry, and save to data/gsm8k_train.jsonl + gsm8k_dev.jsonl.

After writing the files, print a token-count sanity check.
"""
from pathlib import Path
import json, numpy as np, datasets, tiktoken, warnings

ds = datasets.load_dataset("openai/gsm8k", "main")
train_full = list(ds["train"])            

train = train_full[:1_000]
dev   = train_full[1_000:1_200]

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

def dump(split, name):
    with open(out_dir / f"gsm8k_{name}.jsonl", "w") as f:
        for row in split:
            q = row["question"].strip()
            a = row["answer"].strip()
            try:
                cot, final = a.split("####")
            except ValueError:
                warnings.warn(f"Answer missing #### delimiter: {a[:60]}…")
                continue
            f.write(json.dumps({
                "question": q,
                "answer"  : final.strip(),
                "cot"     : cot.strip()
            }) + "\n")

dump(train, "train")
dump(dev,   "dev")

enc = tiktoken.get_encoding("cl100k_base")
cot_lens = [
    len(enc.encode(json.loads(line)["cot"]))
    for line in open(out_dir / "gsm8k_train.jsonl", encoding="utf-8")
]
print(f"{np.mean(cot_lens):.0f} avg CoT tokens  |  {sum(cot_lens):,} tokens total")
