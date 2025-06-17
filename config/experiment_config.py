"""
Unified experiment configuration settings
"""

from dataclasses import dataclass, field
from typing import Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration with validation"""
    teacher_model: str = "Qwen/Qwen2-Math-1.5B-Instruct"
    student_model: str = "Qwen/Qwen2-0.5B-Instruct"
    device: str = "cuda"
    torch_dtype: str = "float16"
    max_length: int = 512
    max_new_tokens: int = 512
    
    def __post_init__(self):
        """Validate model configuration"""
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

@dataclass
class TrainingConfig:
    """Training configuration with validation"""
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    temperature: float = 3.0
    alpha_distill: float = 0.7
    n_epochs: int = 5
    steps_per_epoch: int = 200
    
    def __post_init__(self):
        """Validate training configuration"""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.alpha_distill <= 1:
            raise ValueError("alpha_distill must be between 0 and 1")

@dataclass
class CurriculumConfig:
    """Curriculum learning configuration with validation"""
    alpha: float = 0.5  # Q-value learning rate
    tau: float = 1.0    # Temperature for Boltzmann sampling
    method: str = "rl"  # rl, traditional, easy_to_hard, hard_to_easy
    
    def __post_init__(self):
        """Validate curriculum configuration"""
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if self.tau <= 0:
            raise ValueError("tau must be positive")
        if self.method not in ["rl", "traditional", "easy_to_hard", "hard_to_easy"]:
            raise ValueError(f"Invalid method: {self.method}")
    
@dataclass
class EvaluationConfig:
    """Evaluation configuration with validation"""
    eval_samples: int = 100
    eval_interval: int = 1
    log_solutions: bool = True
    
    def __post_init__(self):
        """Validate evaluation configuration"""
        if self.eval_samples <= 0:
            raise ValueError("eval_samples must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

@dataclass
class ExperimentConfig:
    """Complete experiment configuration with validation and utilities"""
    name: str = "default_experiment"
    save_dir: str = "./experiments"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def __post_init__(self):
        """Validate and setup experiment configuration"""
        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Validate OpenAI API key if needed for evaluation
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        # Extract nested configs
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        curriculum_config = CurriculumConfig(**config_dict.get('curriculum', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        # Create main config
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'training', 'curriculum', 'evaluation']}
        
        return cls(
            model=model_config,
            training=training_config,
            curriculum=curriculum_config,
            evaluation=evaluation_config,
            **main_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'name': self.name,
            'save_dir': self.save_dir,
            'model': {
                'teacher_model': self.model.teacher_model,
                'student_model': self.model.student_model,
                'device': self.model.device,
                'torch_dtype': self.model.torch_dtype,
                'max_length': self.model.max_length,
                'max_new_tokens': self.model.max_new_tokens,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'gradient_accumulation_steps': self.training.gradient_accumulation_steps,
                'learning_rate': self.training.learning_rate,
                'temperature': self.training.temperature,
                'alpha_distill': self.training.alpha_distill,
                'n_epochs': self.training.n_epochs,
                'steps_per_epoch': self.training.steps_per_epoch,
            },
            'curriculum': {
                'alpha': self.curriculum.alpha,
                'tau': self.curriculum.tau,
                'method': self.curriculum.method,
            },
            'evaluation': {
                'eval_samples': self.evaluation.eval_samples,
                'eval_interval': self.evaluation.eval_interval,
                'log_solutions': self.evaluation.log_solutions,
            }
        }

# Predefined experiment configurations using the unified system
def get_experiment_config(experiment_type: str, **overrides) -> ExperimentConfig:
    """Get predefined experiment configuration with optional overrides"""
    
    base_configs = {
        'rl_curriculum': {
            'name': 'rl_curriculum',
            'curriculum': {
                'method': 'rl',
                'alpha': 0.5,
                'tau': 1.0,
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'n_epochs': 5,
                'steps_per_epoch': 200,
            },
            'evaluation': {
                'eval_samples': 100,
                'eval_interval': 1,
            }
        },
        'traditional': {
            'name': 'traditional',
            'curriculum': {
                'method': 'traditional',
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'n_epochs': 5,
                'steps_per_epoch': 200,
            },
            'evaluation': {
                'eval_samples': 100,
                'eval_interval': 1,
            }
        },
        'easy_to_hard': {
            'name': 'easy_to_hard',
            'curriculum': {
                'method': 'easy_to_hard',
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'n_epochs': 5,
                'steps_per_epoch': 200,
            },
            'evaluation': {
                'eval_samples': 100,
                'eval_interval': 1,
            }
        },
        'hard_to_easy': {
            'name': 'hard_to_easy',
            'curriculum': {
                'method': 'hard_to_easy',
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-5,
                'n_epochs': 5,
                'steps_per_epoch': 200,
            },
            'evaluation': {
                'eval_samples': 100,
                'eval_interval': 1,
            }
        }
    }
    
    if experiment_type not in base_configs:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Apply overrides to the base config
    config_dict = base_configs[experiment_type].copy()
    
    # Deep merge overrides
    for key, value in overrides.items():
        if key in config_dict and isinstance(config_dict[key], dict) and isinstance(value, dict):
            config_dict[key].update(value)
        else:
            config_dict[key] = value
    
    return ExperimentConfig.from_dict(config_dict)

# Backward compatibility
BASE_CONFIG = get_experiment_config('traditional').to_dict()
EXPERIMENT_CONFIGS = {
    'rl_curriculum': {
        'method': 'rl',
        'alpha': 0.5,
        'tau': 1.0,
        'description': 'RL-based curriculum (SEC-inspired)'
    },
    'traditional': {
        'method': 'traditional',
        'description': 'Traditional distillation (uniform sampling)'
    },
    'easy_to_hard': {
        'method': 'easy_to_hard',
        'description': 'Fixed curriculum: Easy to Hard'
    },
    'hard_to_easy': {
        'method': 'hard_to_easy',
        'description': 'Fixed curriculum: Hard to Easy'
    }
}