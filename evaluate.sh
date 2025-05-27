#!/usr/bin/env bash
accelerate launch -m lm_eval \
  --model hf \
  --model_args "pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct,peft=llama3-gsm8k/lora-adapter" \
  --tasks gsm8k_cot_llama \
  --num_fewshot 8 \
  --batch_size 8 \
  --apply_chat_template \
  --fewshot_as_multiturn # <-- Add this flag
