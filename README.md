# RL-Optimized Curriculum Learning for Knowledge Distillation

This project implements reinforcement learning-based curriculum optimization for knowledge distillation, extending SEC (Self-Evolving Curriculum) methods to teacher-student learning scenarios.

## 🏗️ Project Structure

```
rl_curriculum_distillation/
├── README.md
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   └── experiment_config.py     # Experiment configurations
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py           # MATH dataset handling
│   ├── models/
│   │   ├── __init__.py
│   │   ├── curriculum.py        # Curriculum strategies (SEC, fixed, random)
│   │   ├── reward_computation.py # Reward functions for curriculum
│   │   └── trainers.py          # Training classes for different methods
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py         # Model evaluation utilities
│   └── utils/
│       ├── __init__.py
│       ├── monitoring.py        # Training monitoring and logging
│       └── serialization.py     # Result saving utilities
├── scripts/
│   ├── run_single_experiment.py # Run individual experiments
│   ├── run_all_experiments.py   # Run complete comparison
│   └── quick_test.py            # Verify installation and functionality
└── experiments/                 # Generated experiment results
```

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd rl_curriculum_distillation
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Verify installation:**
```bash
python scripts/quick_test.py
```

### Running Experiments

**Single experiment:**
```bash
# RL curriculum method
python scripts/run_single_experiment.py --method rl --n_epochs 5

# Traditional distillation
python scripts/run_single_experiment.py --method traditional --n_epochs 5

# Fixed curricula
python scripts/run_single_experiment.py --method easy_to_hard --n_epochs 5
python scripts/run_single_experiment.py --method hard_to_easy --n_epochs 5
```

**Complete comparison:**
```bash
python scripts/run_all_experiments.py
```

**Available options:**
```bash
python scripts/run_single_experiment.py --help
```

## 📊 Methods Implemented

### 1. RL Curriculum (SEC-inspired)
- **Class**: `RLDistillationTrainer`
- **Description**: Uses reinforcement learning to adaptively select training examples
- **Algorithm**: TD(0) Q-learning with Boltzmann exploration
- **Reward**: Teacher-student confidence gaps and knowledge transfer signals

### 2. Traditional Distillation
- **Class**: `TraditionalDistillationTrainer` 
- **Description**: Uniform random sampling baseline
- **Algorithm**: Standard knowledge distillation with KL divergence

### 3. Fixed Curricula
- **Class**: `FixedCurriculumTrainer`
- **Easy-to-Hard**: Progressively increase difficulty
- **Hard-to-Easy**: Start with hardest problems first

## 🧠 Key Components

### Curriculum Strategies (`src/models/curriculum.py`)
- `SECDistillationCurriculum`: SEC-inspired adaptive curriculum
- `EasyToHardCurriculum`: Linear difficulty progression
- `HardToEasyCurriculum`: Reverse difficulty progression
- `RandomCurriculum`: Uniform sampling baseline

### Reward Computation (`src/models/reward_computation.py`)
- `FixedDistillationRewardComputer`: SEC-style rewards for distillation
- `MultiObjectiveRewardComputer`: Extended multi-objective rewards

### Evaluation (`src/evaluation/evaluator.py`)
- Comprehensive model evaluation
- Teacher-student agreement metrics
- Deterministic evaluation for consistency

## 📈 Results and Analysis

After running experiments, you'll find:

- **`results/`**: Complete experimental results
- **`all_results_enhanced.json`**: Numerical results
- **`comprehensive_comparison.png`**: Visualization
- **`experiment_report.md`**: Detailed analysis
- **Individual experiment directories**: Detailed logs and plots

### Key Metrics
- **Accuracy**: Overall and per-difficulty level
- **Sample Efficiency**: Training examples needed for convergence
- **Teacher-Student Agreement**: Consistency between models
- **Training Time**: Computational efficiency

## 🔧 Configuration

Modify `config/experiment_config.py` to customize:

```python
# Model settings
ModelConfig(
    teacher_model="Qwen/Qwen2-1.5B",
    student_model="Qwen/Qwen2-0.5B"
)

# Training settings
TrainingConfig(
    batch_size=8,
    learning_rate=1e-5,
    temperature=3.0
)

# Curriculum settings
CurriculumConfig(
    alpha=0.5,  # Q-learning rate
    tau=1.0     # Exploration temperature
)
```

## 🧪 Testing

**Quick verification:**
```bash
python scripts/quick_test.py
```

**Test specific components:**
```python
from src.data.dataset import MATHDataset
from src.models.trainers import RLDistillationTrainer

# Test dataset loading
dataset = MATHDataset(split='train')
print(f"Loaded {len(dataset)} examples")

# Test model initialization
trainer = RLDistillationTrainer(batch_size=4)
```

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.37+
- CUDA-capable GPU (8GB+ recommended)
- See `requirements.txt` for complete list

## 🐛 Troubleshooting

**Common issues:**

1. **CUDA out of memory**: Reduce `batch_size` in config
2. **Dataset loading fails**: Install `datasets` and `huggingface_hub`
3. **Model loading errors**: Check CUDA/PyTorch compatibility
4. **Import errors**: Run `pip install -e .` in project root

**Debug mode:**
```bash
# Run with minimal settings for debugging
python scripts/run_single_experiment.py --method rl --n_epochs 1 --steps_per_epoch 10 --batch_size 2
```

## 🤝 Contributing

The modular structure makes it easy to extend:

1. **Add new curriculum strategies**: Extend `src/models/curriculum.py`
2. **Implement new rewards**: Add to `src/models/reward_computation.py`
3. **Create new trainers**: Inherit from `BaseDistillationTrainer`
4. **Add evaluation metrics**: Extend `src/evaluation/evaluator.py`

## 📖 Research Background

This implementation is based on:
- **SEC (Self-Evolving Curriculum)**: TD(0) curriculum learning for RL
- **Knowledge Distillation**: Teacher-student model compression
- **MATH Dataset**: Mathematical reasoning evaluation

### Key Innovation
Extends SEC's single-model curriculum learning to knowledge distillation scenarios, addressing the unique challenges of teacher-student learning dynamics.

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@article{rl_curriculum_distillation_2024,
  title={RL-Optimized Curriculum Learning for Knowledge Distillation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```