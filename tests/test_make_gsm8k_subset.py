import pytest  # noqa
from pathlib import Path
import scripts.make_gsm8k_subset as gsm8k
import json

def test_make_gsm8k_subset(tmp_path, monkeypatch):
    monkeypatch.setattr(gsm8k, "out_dir", tmp_path)
    
    gsm8k.dump(gsm8k.train, "train")
    gsm8k.dump(gsm8k.dev, "dev")
    
    train_file = tmp_path / "gsm8k_train.jsonl"
    dev_file = tmp_path / "gsm8k_dev.jsonl"
    
    assert train_file.exists(), "Train file was not created"
    assert dev_file.exists(), "Dev file was not created"
    
    train_count = sum(1 for _ in open(train_file))
    dev_count = sum(1 for _ in open(dev_file))
    
    assert train_count == 1000, f"Expected 1000 lines in train file, got {train_count}"
    assert dev_count == 200, f"Expected 200 lines in dev file, got {dev_count}"
    
    with open(train_file) as f:
        first_line = json.loads(next(f))
        assert all(k in first_line for k in ["question", "answer", "cot"]), \
            "Train file JSON missing required fields" 