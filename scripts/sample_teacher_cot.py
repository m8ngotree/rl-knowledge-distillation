#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT solutions using DeepSeek API")
    parser.add_argument("--in_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--out_file", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, default="deepseek-r1", help="Model to use")
    parser.add_argument("--api_key", type=str, required=True, help="DeepSeek API key")
    parser.add_argument("--max_tokens", type=int, default=768, help="Max tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--cache_file", type=str, default="cache/cot_cache.jsonl", help="Cache file path")
    return parser.parse_args()

def load_cache(cache_file: str) -> Dict[str, str]:
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            for line in f:
                entry = json.loads(line)
                cache[entry["qid"]] = entry["cot"]
    return cache

def save_to_cache(cache_file: str, qid: str, cot: str):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "a") as f:
        json.dump({"qid": qid, "cot": cot}, f)
        f.write("\n")

def generate_cot(question: str, api_key: str, max_tokens: int, temperature: float) -> Optional[str]:
    prompt = f"""You are a math tutor. Provide a concise step-by-step solution before '####'.
Question: {question}

Solution:"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-r1",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    max_retries = 3
    base_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate CoT after {max_retries} attempts: {e}")
                return None
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)

def main():
    args = parse_args()
    cache = load_cache(args.cache_file)
    
    total_calls = 0
    cached_count = 0
    
    with open(args.in_file, "r") as f_in, open(args.out_file, "w") as f_out:
        for line in tqdm(f_in, desc="Processing questions"):
            entry = json.loads(line)
            qid = entry.get("qid", str(hash(entry["question"])))
            
            if entry.get("cot") and entry["cot"].strip():
                f_out.write(json.dumps(entry) + "\n")
                continue
                
            if qid in cache:
                entry["cot"] = cache[qid]
                f_out.write(json.dumps(entry) + "\n")
                cached_count += 1
                continue
            
            cot = generate_cot(entry["question"], args.api_key, args.max_tokens, args.temperature)
            if cot:
                entry["cot"] = cot
                save_to_cache(args.cache_file, qid, cot)
                total_calls += 1
            f_out.write(json.dumps(entry) + "\n")
    
    print(f"\nSummary:")
    print(f"Total API calls: {total_calls}")
    print(f"Cached solutions used: {cached_count}")

if __name__ == "__main__":
    main() 