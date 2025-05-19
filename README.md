# RL-Optimized Teaching

This project implements a reinforcement learning (RL) based approach to optimize teaching strategies. It uses PPO (Proximal Policy Optimization) to learn optimal teaching policies and includes tools for generating Chain-of-Thought (CoT) solutions.

## Project Structure

```
.
├── scripts/
│   └── sample_teacher_cot.py    # Script for generating CoT solutions using DeepSeek API
├── tests/
│   ├── test_mock_rl_loop.py     # Unit tests for RL environment and PPO training
│   └── test_make_gsm8k_subset.py # Tests for GSM8K dataset processing
└── requirements.txt             # Project dependencies
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

### 1. Chain-of-Thought Generation (`scripts/sample_teacher_cot.py`)

This script generates step-by-step solutions for math problems using the DeepSeek API. It includes caching functionality to avoid redundant API calls.

Key features:
- Uses DeepSeek API for generating solutions
- Implements caching to save API calls
- Supports batch processing of questions
- Configurable parameters (temperature, max tokens, etc.)

Usage:
```bash
python scripts/sample_teacher_cot.py \
    --in_file input.jsonl \
    --out_file output.jsonl \
    --api_key YOUR_DEEPSEEK_API_KEY \
    --model deepseek-r1 \
    --max_tokens 768 \
    --temperature 0.2
```

### 2. RL Environment and Training (`tests/test_mock_rl_loop.py`)

Implements a simple RL environment for testing PPO training. The environment is designed to be learnable with a clear optimal policy.

Key features:
- `LearnableEnv`: A simple environment where the optimal action is 0.5
- Reward function: `1.0 - (action - 0.5)²`
- Uses stable-baselines3 PPO implementation
- Includes comprehensive unit tests

The environment is designed to be:
- Deterministic
- Easy to learn
- Fast to train
- Suitable for testing RL algorithms

Example usage in tests:
```python
env = DummyVecEnv([lambda: LearnableEnv()])
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=64,
    batch_size=64,
    policy_kwargs=dict(net_arch=[dict(pi=[32], vf=[32])])
)
model.learn(total_timesteps=5000)
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_mock_rl_loop.py -v
```

## Dependencies

Key dependencies include:
- `stable-baselines3`: For PPO implementation
- `gymnasium`: For RL environment
- `transformers`: For language model integration
- `trl`: For RL with language models
- `torch`: For deep learning
- `numpy`: For numerical computations

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

- DeepSeek for providing the API
- Stable-Baselines3 team for the RL implementation
- OpenAI Gymnasium for the environment interface