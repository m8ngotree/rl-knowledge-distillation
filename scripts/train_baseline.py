#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from student_trainer import StudentTrainer

def get_device(device_arg: str) -> str:
    """Get the appropriate device based on availability and user preference."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg

def main():
    parser = argparse.ArgumentParser(description="Train a student model on GSM8K dataset")
    parser.add_argument("--train_file", required=True, help="Path to training JSONL file")
    parser.add_argument("--dev_file", required=True, help="Path to dev JSONL file")
    parser.add_argument("--out_dir", required=True, help="Output directory for metrics and model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to train on (auto selects best available)")
    
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device(args.device)
    trainer = StudentTrainer(
        lora_r=8,
        lora_alpha=32,
        learning_rate=2e-4,
        device=device,
        fp16=device == "cuda",
        output_dir=str(out_dir / "lora_output")
    )
    
    trainer.fit(
        train_jsonl=args.train_file,
        num_epochs=args.epochs,
    )
    
    accuracy = trainer.score(args.dev_file)
    
    with open(args.train_file) as f:
        train_tokens = sum(len((json.loads(line)["question"] + " " + json.loads(line)["cot"]).split()) for line in f)
    
    metrics = {
        "accuracy": accuracy,
        "train_tokens": train_tokens * args.epochs,
        "epochs": args.epochs
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    trainer.save(str(out_dir / "student_lora"))
    
    print(f"Training complete! Accuracy: {accuracy:.2%}")
    print(f"Metrics saved to {out_dir}/metrics.json")
    print(f"Model saved to {out_dir}/student_lora/")

if __name__ == "__main__":
    main() 