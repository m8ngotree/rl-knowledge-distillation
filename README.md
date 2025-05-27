# RL-Optimized Teaching

This project implements a reinforcement learning (RL) based approach to optimize teaching strategies using language models. It fine-tunes Meta-Llama-3.1-8B-Instruct on GSM8K dataset with 4-bit LoRA for efficient training.

## Project Structure

```
.
├── train_finetune.py     # Main script for fine-tuning Llama model
├── teach_env.py         # RL environment for teaching optimization
├── infer.py            # Script for model inference
├── evaluate.sh         # Evaluation script
└── requirements.txt    # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rl-optimized-teaching.git
cd rl-optimized-teaching
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Components

### 1. Model Fine-tuning (`train_finetune.py`)

This script fine-tunes Meta-Llama-3.1-8B-Instruct on the GSM8K dataset using 4-bit LoRA for efficient training.

Key features:
- Uses 4-bit quantization with BitsAndBytes
- Implements LoRA for parameter-efficient fine-tuning
- Supports gradient accumulation for larger effective batch sizes
- Configurable training parameters (batch size, epochs, etc.)

Usage:
```bash
python train_finetune.py \
    --out_dir llama3-gsm8k \
    --batch_size 2 \
    --grad_acc 16 \
    --epochs 3 \
    --merge_fp16
```

### 2. RL Environment (`teach_env.py`)

Implements a reinforcement learning environment for optimizing teaching strategies. The environment is designed to work with language models and teaching scenarios.

Key features:
- Custom RL environment for teaching optimization
- Integration with language models
- Reward function based on teaching effectiveness
- Support for various teaching scenarios

### 3. Inference (`infer.py`)

Script for running inference with the fine-tuned model.

### 4. Evaluation (`evaluate.sh`)

Shell script for evaluating model performance.

## Dependencies

Key dependencies include:
- `torch>=2.1.0`: PyTorch for deep learning
- `transformers==4.51.3`: Hugging Face Transformers
- `datasets>=2.18.0`: Dataset handling
- `accelerate>=0.26.1`: Training acceleration
- `bitsandbytes==0.42.0`: 4-bit quantization
- `peft>=0.15.0`: Parameter-efficient fine-tuning
- `trl==0.17.0`: Training with RL
- `sentencepiece`: Tokenization
- `tiktoken`: Token counting
- `lm-eval==0.4.0`: Model evaluation

See `requirements.txt` for complete list of dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add your license information here]

## Acknowledgments

- Meta AI for the Llama model
- Hugging Face for the Transformers library
- OpenAI for the GSM8K dataset