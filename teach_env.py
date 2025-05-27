#!/usr/bin/env python3
"""
Tiny RL environment for Phase 3.
Implements:
  • 3.1 Action space      – (example_id, reveal_level)
  • 3.2 Reward            – (Δ dev-accuracy) – 5e-4 · tokens_shown
  • 3.3 Gym-style API     – reset() / step()
No dependency on the old student_trainer.py.
"""

# ---------- stdlib ----------
from dataclasses import dataclass
from pathlib import Path
import math, json, re, random, logging, itertools, collections
from typing import Dict, Any, List, Tuple

# ---------- third-party ----------
import torch, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import re

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3.1  ACTION SPACE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
REVEAL_QUESTION_ONLY = 0
REVEAL_HINT          = 1
REVEAL_FULL_COT      = 2

@dataclass(frozen=True, slots=True)
class Action:
    example_id: int
    reveal_level: int   # 0 / 1 / 2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Helper: build the *student* exactly as in train_finetune.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_student(device: str = "cuda") -> Tuple[Any, Any]:
    """
    Returns (model, tokenizer) initialised with 4-bit LoRA.
    Specs copied from train_finetune.py  :contentReference[oaicite:0]{index=0}
    """
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )                                       # :contentReference[oaicite:1]{index=1}

    base_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_id,
        device_map="auto" if device.startswith("cuda") else None,
        quantization_config=bnb_cfg,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"],
    )                                       # :contentReference[oaicite:2]{index=2}
    model = get_peft_model(model, lora_cfg)
    model.train()
    return model, tokenizer


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3.2  REWARD = Δacc – λ·tokens   (λ = 5e-4)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TOKEN_LAMBDA = 5e-4
def reward_delta(prev_acc: float, new_acc: float, tokens: int) -> float:
    return (new_acc - prev_acc) - TOKEN_LAMBDA * tokens

ANSWER_PATTERNS = [
    re.compile(r"####\s*(-?\d+)"),     # standard GSM8K
    re.compile(r"\\boxed\{(-?\d+)\}"), # LaTeX box
]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3.3  ENVIRONMENT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class TeachEnv:
    """
    *state*  = last k (loss, dev_acc, tokens) – a fixed-length float vector
    *action* = (example_id, reveal_level)
    *reward* = advantage (reward – moving_baseline)
    """
    def __init__(
        self,
        train_jsonl: str,
        dev_jsonl:   str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        hint_tokens: int = 40,
        eval_every:  int = 25,
        lr: float = 2e-4,
        history_len: int = 32,
        max_seq_len: int = 1024,
    ):
        self.device = device
        self.hint_tokens = hint_tokens
        self.eval_every = eval_every
        self.max_seq_len = max_seq_len

        # data
        self.train_ds = load_dataset("json", data_files=train_jsonl)["train"]
        self.dev_ds   = load_dataset("json", data_files=dev_jsonl)["train"]
        self.num_examples = len(self.train_ds)

        # student model
        self.model, self.tok = build_student(device)
        self.opt = torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad), lr=lr)

        # bookkeeping
        self.step_idx = 0
        self.history  = collections.deque(maxlen=history_len)
        self.baseline = 0.0
        self.dev_acc  = self._full_eval()      # initial accuracy

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    def extract_answer(text: str) -> str | None:
        for pat in ANSWER_PATTERNS:
            if m := pat.search(text):
                return m.group(1)
        return None

    def _full_eval(self, batch_size: int = 8) -> float:
        """Greedy dev-set accuracy, no CoT."""
        self.model.eval()
        correct = 0
        for start in range(0, len(self.dev_ds), batch_size):
            batch = self.dev_ds.select(range(start, min(start+batch_size, len(self.dev_ds))))
            prompts = [ex["question"] for ex in batch]
            with torch.inference_mode():
                inputs = self.tok(prompts, return_tensors="pt",
                                  padding=True, truncation=True,
                                  max_length=self.max_seq_len).to(self.device)
                outs = self.model.generate(**inputs, max_new_tokens=64, temperature=0.0)
                decoded = self.tok.batch_decode(outs, skip_special_tokens=True)
            for pred, ex in zip(decoded, batch):
                pred_ans = self.extract_answer(pred)
                if pred_ans is None:
                    continue
                gold = re.search(r"####\s*(-?\d+)", ex["answer"])
                if gold and pred_ans.strip() == gold.group(1).strip():
                    correct += 1
        self.model.train()
        return correct / len(self.dev_ds)

    def _build_text(self, ex: Dict[str, str], rv: int) -> Tuple[str, int]:
        q  = ex["question"].strip()
        cot = ex.get("cot", "").strip()
        if rv == REVEAL_QUESTION_ONLY:
            txt = q
        elif rv == REVEAL_HINT:
            hint = " ".join(cot.split()[: self.hint_tokens]) + " …"
            txt  = f"{q}\n\n### Hint\n{hint}"
        elif rv == REVEAL_FULL_COT:
            txt = f"{q}\n\n### Solution\n{cot}"
        else:
            raise ValueError("Bad reveal level")
        toks = len(self.tok.encode(txt))
        return txt, toks

    def _single_update(self, prompt: str) -> float:
        enc = self.tok(prompt, return_tensors="pt",
                       padding=False, truncation=True,
                       max_length=self.max_seq_len).to(self.device)
        labels = enc["input_ids"].clone()
        loss = self.model(**enc, labels=labels).loss
        loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(loss)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def reset(self):
        self.step_idx = 0
        self.history.clear()
        self.baseline = 0.0
        self.dev_acc  = self._full_eval()
        return self._state_vec()

    def _state_vec(self) -> np.ndarray:
        flat = list(itertools.chain.from_iterable(self.history))
        pad  = [0.0] * (self.history.maxlen*3 - len(flat))
        return np.asarray(flat + pad, dtype=np.float32)

    def step(self, action: Action):
        ex = self.train_ds[int(action.example_id)]
        txt, toks = self._build_text(ex, action.reveal_level)

        # a) optional train
        train_loss = math.nan
        if action.reveal_level != REVEAL_QUESTION_ONLY:
            train_loss = self._single_update(txt)

        # b) occasional dev eval
        new_acc = self.dev_acc
        recomputed = False
        if (self.step_idx + 1) % self.eval_every == 0:
            new_acc = self._full_eval()
            recomputed = True

        # c) reward → advantage
        raw_r = reward_delta(self.dev_acc, new_acc, toks)
        self.baseline = 0.9 * self.baseline + 0.1 * raw_r
        adv = raw_r - self.baseline

        # d) history & counters
        self.history.append((float(train_loss if not math.isnan(train_loss) else 0.0),
                             float(new_acc), float(toks)))
        self.dev_acc = new_acc
        self.step_idx += 1

        info = dict(train_loss=train_loss, dev_acc=new_acc,
                    tokens=toks, raw_reward=raw_r,
                    recomputed=recomputed)
        done = False     # infinite horizon
        return self._state_vec(), float(adv), done, info


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5  SMOKE TEST  (optional)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    import random, logging
    logging.basicConfig(level=logging.INFO)

    # 1. download / cache the dataset from Hugging Face
    from datasets import load_dataset

    TRAIN_SPLIT = "train"      # 7473 items
    TEST_SPLIT  = "test"       # 1319 items – we’ll subsample as “dev”
    DEV_N       = 200

    full_train = load_dataset("gsm8k", "main", split=TRAIN_SPLIT)
    full_test  = load_dataset("gsm8k", "main", split=TEST_SPLIT)

    # take a quick dev subset for speed
    dev_subset = full_test.shuffle(seed=42).select(range(DEV_N))

    # 2. write them to temporary JSONL files (TeachEnv still expects paths)
    import tempfile, json, os
    tmp_dir = tempfile.TemporaryDirectory()
    train_path = os.path.join(tmp_dir.name, "gsm8k_train.jsonl")
    dev_path   = os.path.join(tmp_dir.name, "gsm8k_dev.jsonl")

    with open(train_path, "w") as f:
        for ex in full_train:
            json.dump(ex, f); f.write("\n")
    with open(dev_path, "w") as f:
        for ex in dev_subset:
            json.dump(ex, f); f.write("\n")

    # 3. create environment
    env = TeachEnv(
        train_jsonl=train_path,
        dev_jsonl=dev_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_every=10,
    )

    # 4. random-policy smoke-test (30 steps)
    state = env.reset()
    for t in range(30):
        act = Action(
            example_id=random.randrange(env.num_examples),
            reveal_level=random.choice([REVEAL_QUESTION_ONLY,
                                        REVEAL_HINT,
                                        REVEAL_FULL_COT])
        )
        state, reward, done, info = env.step(act)
        print(f"{t:02d}  reward {reward:+.4f}   dev_acc {info['dev_acc']:.3%}")

    # cleanup temporary dir automatically when program exits