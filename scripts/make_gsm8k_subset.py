from pathlib import Path
import json, numpy as np, datasets, tiktoken, warnings

ds = datasets.load_dataset("openai/gsm8k", "main")
train_full = list(ds["train"])

train = train_full[:1_000]
dev   = train_full[1_000:1_200]

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

def dump(split, name):
    with open(out_dir / f"gsm8k_{name}.jsonl", "w", encoding="utf‑8") as f:
        for row in split:
            q = row["question"].strip()
            a = row["answer"].strip()
            try:
                cot, final = a.split("####")
            except ValueError:
                warnings.warn(f"Answer missing #### delimiter: {a[:60]}…")
                continue
            cot = cot.strip().split("####")[0].rstrip()
            final = final.strip()
            cot_with_answer = f"{cot}\nThe final answer is: \\boxed{{{final}}}"
            f.write(
                json.dumps({
                    "question": q,
                    "answer": final,
                    "cot": cot_with_answer,
                })
                + "\n"
            )


dump(train, "train")
dump(dev, "dev")

# (unchanged) quick token‑count sanity check
enc = tiktoken.get_encoding("cl100k_base")
cot_lens = [
    len(enc.encode(json.loads(line)["cot"]))
    for line in open(out_dir / "gsm8k_train.jsonl", encoding="utf‑8")
]
print(
    f"{np.mean(cot_lens):.0f} avg CoT tokens  |  {sum(cot_lens):,} tokens total (train)"
)