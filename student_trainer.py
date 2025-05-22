import logging, re
from pathlib import Path
from typing import Optional

import torch, wandb
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StopAfterBoxed(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.box_pattern = re.compile(r"\\boxed\{[-+]?\d+\}")

    def __call__(self, input_ids, scores, **kwargs):
        # Only decode the last ~100 tokens for speed
        text = self.tokenizer.decode(input_ids[0, -100:], skip_special_tokens=True)
        # Check for complete boxed answer pattern
        return bool(self.box_pattern.search(text))


class StudentTrainer:
    def __init__(
        self,
        base_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        lora_r: int = 8,
        lora_alpha: int = 16,
        learning_rate: float = 1e-4,  # Lower learning rate for better stability
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fp16: bool = True,
        output_dir: str = "lora_output",
        wandb_project: str = "rl-optimized-teaching",
        wandb_run_name: Optional[str] = None,
        max_seq_len: int = 1024,
    ):
        self.device = device
        self.output_dir = output_dir
        self.max_seq_len = max_seq_len

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
                "max_seq_len": self.max_seq_len,
            },
        )

        if not torch.cuda.is_available():
            fp16 = False
            if device == "cpu":
                print("Warning: loading model on CPU; this may be slow.")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token == self.tokenizer.eos_token:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

        # Ensure the explicit end‑of‑answer token is in the vocab
        self.end_token = "<|answer_end|>"
        if self.end_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.end_token]})

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if fp16 else torch.float32,
            device_map="auto",
            load_in_8bit=False,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.end_token_id = self.tokenizer.convert_tokens_to_ids(self.end_token)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, lora_config)

        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        
        self.sft_learning_rate = learning_rate 
        self.sft_fp16 = fp16

    def fit(
        self,
        train_jsonl: str,
        num_epochs: int = 2,  # Increased to 2 epochs
    ):
        dataset = load_dataset("json", data_files=train_jsonl)["train"]
        
        def format_prompt(example):
            return {
                "text": (
                    f"### Question:\n{example['question']}\n\n"
                    f"### Solution (show your work and end with answer in \\boxed{{}}):\n"
                    f"{example['cot'].strip()}{self.tokenizer.eos_token}"
                )
            }
        dataset = dataset.map(format_prompt)
        
        per_device_batch_size = 4 if torch.cuda.is_available() else 4
        
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.sft_learning_rate,
            fp16=self.sft_fp16,
            logging_steps=25,
            save_strategy="no",
            optim="adamw_torch",
            max_seq_length=self.max_seq_len,
            dataset_text_field="text",
            report_to="wandb",
            gradient_checkpointing=True,
            warmup_ratio=0.1,  # Add warmup
            lr_scheduler_type="cosine",  # Use cosine learning rate schedule
        )

        self.model.train()
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=sft_config,
            processing_class=self.tokenizer, 
        )

        trainer.train()
        
    def score(
        self,
        dev_jsonl: str,
        batch_size: int = 8, 
        max_new_tokens: int = 512, 
    ) -> float:
        dataset = load_dataset("json", data_files=dev_jsonl)["train"]
        
        self.model.eval()
        self.model.gradient_checkpointing_disable() 
        self.model.config.use_cache = True 

        correct = 0
        total = len(dataset)

        scoring_batch_size = batch_size if torch.cuda.is_available() else 2 

        with torch.inference_mode():
            for start in tqdm(range(0, total, scoring_batch_size), desc="Scoring (debug)"):
                end = min(start + scoring_batch_size, total)
                batch = dataset.select(range(start, end))
                prompts = [
                    f"### Question:\n{ex['question']}\n\n"
                    f"### Solution (show your work and end with answer in \\boxed{{}}):\n"
                    for ex in batch
                ]
                
                inputs = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True, 
                    truncation=True,
                    max_length=self.max_seq_len - max_new_tokens, 
                ).to(self.device)
                
                stopping_criteria = StoppingCriteriaList([StopAfterBoxed(self.tokenizer)])
                
                outs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=None,
                    stopping_criteria=stopping_criteria,
                    temperature=0.1,  # More deterministic
                    num_beams=1,  # Greedy decoding
                )
                
                full_decoded = self.tokenizer.batch_decode(outs, skip_special_tokens=True)
                
                for i, text_output in enumerate(full_decoded):
                    current_example_data = batch[i] 
                    
                    extracted_ans_str = extract_boxed_answer(text_output)
                    
                    parsed_value = None
                    if extracted_ans_str:
                        try:
                            # Remove any commas from the answer string before converting to int
                            cleaned_ans_str = extracted_ans_str.replace(",", "")
                            parsed_value = int(cleaned_ans_str)
                        except ValueError:
                            print(f"  Warning: Could not convert '{extracted_ans_str}' to int for example {start + i}.")
                    
                    # Clean the correct answer string similarly
                    correct_answer_str = str(current_example_data["answer"]).replace(",", "")
                    correct_answer_val = int(correct_answer_str)

                    print(f"\nExample {start + i} Debug:")
                    print(f"  Prompt: {repr(prompts[i])}") 
                    print(f"  Model Output (full): {repr(text_output)}")
                    print(f"  Extracted Answer String: {extracted_ans_str}")
                    print(f"  Parsed Answer: {parsed_value}")
                    print(f"  Correct Answer: {correct_answer_val}")

                    if parsed_value is not None and parsed_value == correct_answer_val:
                        print("  -> CORRECT")
                        correct += 1
                    else:
                        print("  -> INCORRECT")
                        
        accuracy = correct / total if total > 0 else 0.0
        print(f"\nFinal Debug: {correct} / {total} correct. Accuracy: {accuracy:.2%}")
        wandb.log({"dev_accuracy": accuracy})
        return accuracy

    def save(self, adapter_dir: str):
        self.model.save_pretrained(adapter_dir)
        self.tokenizer.save_pretrained(adapter_dir)

    @classmethod
    def from_adapter(cls, adapter_dir: str, **kwargs):
        instance = cls(**kwargs) 
        instance.model = PeftModel.from_pretrained(
            instance.model.base_model.model, 
            adapter_dir,
            is_trainable=False, 
        )
        return instance

def extract_boxed_answer(text):
    matches = list(re.finditer(r"\\boxed\{(-?\d+)\}", text))
    if matches:
        return matches[-1].group(1)
    return None

if __name__ == "__main__":
    trainer = StudentTrainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        wandb_project="rl-optimized-teaching", 
        wandb_run_name="test-run-fixed-stopping",
    )
    
    try:
        full_train_dataset = load_dataset("json", data_files="data/gsm8k_train.jsonl")["train"]
        subset_size = 250
        if len(full_train_dataset) >= subset_size:
            subset = full_train_dataset.select(range(subset_size)) 
            subset.to_json("data/train_subset_small.jsonl")
            train_file_to_use = "data/train_subset_small.jsonl"
        else:
            print(f"Warning: Full training data has less than {subset_size} examples. Using all.")
            train_file_to_use = "data/gsm8k_train.jsonl"
    except Exception as e:
        print(f"Error creating subset, ensure 'data/gsm8k_train.jsonl' exists and is valid: {e}")
        print("Falling back to trying to use 'data/gsm8k_train.jsonl' directly if it exists.")
        train_file_to_use = "data/gsm8k_train.jsonl"

    print(f"Using training file: {train_file_to_use}")
    
    dev_file_path = Path("data/gsm8k_dev.jsonl")
    if not dev_file_path.exists():
        print(f"ERROR: Dev file {dev_file_path} not found. Please run make_gsm8k_subset.py.")
    else:
        try:
            trainer.fit(train_file_to_use, num_epochs=2) 
            accuracy = trainer.score("data/gsm8k_dev.jsonl")
            print(f"Dev accuracy: {accuracy:.2%}")
        except Exception as e:
            print(f"An error occurred during training or scoring: {e}")
            import traceback
            traceback.print_exc()

    wandb.finish()