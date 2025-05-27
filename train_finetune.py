#!/usr/bin/env python
"""
Fine-tune Meta-Llama-3.1-8B-Instruct on GSM8K with 4-bit LoRA.
Run:  python train_finetune.py --help  for CLI flags.
"""
import argparse, torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

SYSTEM_PROMPT = "You are a helpful math tutor."

def format_example(example, tokenizer):
    chat = [
        {"role": "system",     "content": SYSTEM_PROMPT},
        {"role": "user",       "content": example["question"]},
        {"role": "assistant",  "content": example["answer"]},
    ]
    return {
        "text":
            tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            ) + tokenizer.eos_token
    }

def main(cfg):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",   # ← correct HF repo name
        device_map="auto",                       # keeps layer off-loading
        quantization_config=bnb_cfg,             # ← new field
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = LoraConfig(
        r=16, 
        lora_alpha=16, 
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity-check

    # ----- 3. Prepare GSM8K -----
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"].map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=["question", "answer"],
    )
    val_ds = ds["test"].map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=["question", "answer"],
    )

    # ----- 4. Trainer -----
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=TrainingArguments(
            output_dir=cfg.out_dir,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_acc,
            num_train_epochs=cfg.epochs,
            learning_rate=2e-4,
            lr_scheduler_type="linear",
            warmup_ratio=0.03,           # 3 %
            logging_steps=10,
            save_strategy="steps",
            save_steps=250,
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            report_to="none",
        ),
    )
    trainer.train()

    # ----- 5. Save artifacts -----
    model.save_pretrained(f"{cfg.out_dir}/lora-adapter")
    if cfg.merge_fp16:
        merged = model.merge_and_unload()
        merged.save_pretrained(f"{cfg.out_dir}/merged")  # ≈ 16 GB

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="llama3-gsm8k")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--merge_fp16", action="store_true")
    main(p.parse_args())