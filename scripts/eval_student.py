#!/usr/bin/env python3
import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate a trained student model on GSM8K dataset")
    parser.add_argument("--adapter_dir", required=True, help="Path to trained LoRA adapter directory")
    parser.add_argument("--dev_file", required=True, help="Path to dev JSONL file")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                       help="Device to evaluate on (auto selects best available)")
    parser.add_argument("--out_file", help="Optional path to save metrics JSON file")
    
    args = parser.parse_args()
    
    device = get_device(args.device)
    trainer = StudentTrainer.from_adapter(
        args.adapter_dir,
        device=device,
        fp16=device == "cuda"
    )
    
    accuracy = trainer.score(args.dev_file)
    
    metrics = {
        "accuracy": accuracy,
        "device": device
    }
    
    print(f"Evaluation complete! Accuracy: {accuracy:.2%}")
    
    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {out_path}")

if __name__ == "__main__":
    main() 