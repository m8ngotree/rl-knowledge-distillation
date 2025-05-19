import logging
import re
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StudentTrainer:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fp16: bool = True,
        output_dir: str = "lora_output",
        wandb_project: str = "rl-optimized-teaching",
        wandb_run_name: Optional[str] = None,
    ):
        self.device = device
        self.output_dir = output_dir
        
        # Initialize wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "base_model": base_model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "device": device,
                "fp16": fp16,
            }
        )
        
        # Force fp16=False if not CUDA
        if not (isinstance(device, str) and device.startswith("cuda")):
            fp16 = False
            if device == "cpu":
                print("Warning: Loading model on CPU. This may require significant RAM.")
                print("Consider using a quantized model or GPU if available.")
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use 8-bit quantization for CPU to reduce memory usage
        load_in_8bit = device == "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto" if device != "cpu" else None,
            load_in_8bit=load_in_8bit,
        )
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)
        
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=25,
            save_strategy="no",
            optim="adamw_torch",
        )

    def fit(
        self,
        train_jsonl: str,
        num_epochs: int = 1,
        max_seq_len: int = 768,
    ):
        dataset = load_dataset("json", data_files=train_jsonl)["train"]
        
        def format_prompt(example):
            return {
                "text": f"{example['question']}\n\n### Solution\n{example['cot']}"
            }
        dataset = dataset.map(format_prompt)
        
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=self.training_args.learning_rate,
            fp16=self.training_args.fp16,
            logging_steps=25,
            save_strategy="no",
            optim="adamw_torch",
            max_seq_length=max_seq_len,
            dataset_text_field="text",
            report_to="wandb",  # Enable wandb reporting
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        
    def score(self, dev_jsonl: str) -> float:
        dataset = load_dataset("json", data_files=dev_jsonl)["train"]
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Scoring"):
            prompt = f"{example['question']}\n\n### Solution\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0,
                do_sample=False,
            )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            match = re.search(r"####\s*(\d+)", generated)
            
            if match:
                pred_answer = int(match.group(1))
                true_answer = int(example["answer"])
                correct += pred_answer == true_answer
            
            total += 1
            
        accuracy = correct / total if total > 0 else 0.0
        # Log accuracy to wandb
        wandb.log({"dev_accuracy": accuracy})
        return accuracy
    
    def save(self, adapter_dir: str):
        # Note: This only saves the LoRA adapter weights, not the base model.
        # When loading with from_adapter(), the base model name must be specified separately.
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)
        # Log model artifacts to wandb
        wandb.save(adapter_dir)
    
    @classmethod
    def from_adapter(cls, adapter_dir: str, **kwargs):
        instance = cls(**kwargs)
        instance.model = PeftModel.from_pretrained(
            instance.model,
            adapter_dir,
            is_trainable=False,
        )
        return instance

if __name__ == "__main__":
    trainer = StudentTrainer(
        device="cpu",
        wandb_project="rl-optimized-teaching",
        wandb_run_name="test-run"
    )
    
    # Load and subset dataset
    dataset = load_dataset("json", data_files="data/gsm8k_train.jsonl")["train"]
    subset = dataset.select(range(2))
    subset.to_json("data/train_subset.jsonl")
    
    # Train on 2 examples
    trainer.fit("data/train_subset.jsonl", num_epochs=1)
    
    # Score on dev set
    accuracy = trainer.score("data/gsm8k_dev.jsonl")
    print(f"Dev accuracy: {accuracy:.2%}")
    
    # Close wandb run
    wandb.finish() 