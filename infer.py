import torch, json, transformers
from peft import PeftModel, PeftConfig

base = "meta-llama/Meta-Llama-3.1-8B-Instruct"
peft = "llama3-gsm8k/lora-adapter"          # or merged/ for FP16

# 1-line helper: returns merged fp16 weights on-the-fly
def load_peft(base_id, peft_dir):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, peft_dir)
    return model.merge_and_unload()         # keep memory modest

model = load_peft(base, peft)
tokenizer = transformers.AutoTokenizer.from_pretrained(base)
tokenizer.pad_token = tokenizer.eos_token

pipe = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    temperature=0.2,
    top_p=0.9,
    device_map="auto",
)

prompt = [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user",   "content": "If 5 pencils cost 30 cents, how much do 3 pencils cost?"}
]
print(pipe(prompt)[0]["generated_text"])
