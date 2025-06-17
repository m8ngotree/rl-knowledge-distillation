"""
Test configuration system
"""

import pytest
from config.experiment_config import (
    ModelConfig, TrainingConfig, CurriculumConfig, 
    EvaluationConfig, ExperimentConfig, get_experiment_config
)


class TestConfigValidation:
    """Test configuration validation"""
    
    def test_model_config_validation(self):
        """Test model configuration validation"""
        # Valid config
        config = ModelConfig()
        assert config.max_length > 0
        assert config.max_new_tokens > 0
        
        # Invalid max_length
        with pytest.raises(ValueError, match="max_length must be positive"):
            ModelConfig(max_length=0)
            
        # Invalid max_new_tokens  
        with pytest.raises(ValueError, match="max_new_tokens must be positive"):
            ModelConfig(max_new_tokens=-1)
    
    def test_training_config_validation(self):
        """Test training configuration validation"""
        # Valid config
        config = TrainingConfig()
        assert config.batch_size > 0
        assert config.learning_rate > 0
        assert 0 <= config.alpha_distill <= 1
        
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainingConfig(batch_size=0)
            
        # Invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-1)
            
        # Invalid alpha_distill
        with pytest.raises(ValueError, match="alpha_distill must be between 0 and 1"):
            TrainingConfig(alpha_distill=1.5)
    
    def test_curriculum_config_validation(self):
        """Test curriculum configuration validation"""
        # Valid config
        config = CurriculumConfig()
        assert 0 <= config.alpha <= 1
        assert config.tau > 0
        assert config.method in ["rl", "traditional", "easy_to_hard", "hard_to_easy"]
        
        # Invalid alpha
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            CurriculumConfig(alpha=2.0)
            
        # Invalid tau
        with pytest.raises(ValueError, match="tau must be positive"):
            CurriculumConfig(tau=0)
            
        # Invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            CurriculumConfig(method="invalid_method")
    
    def test_evaluation_config_validation(self):
        """Test evaluation configuration validation"""
        # Valid config
        config = EvaluationConfig()
        assert config.eval_samples > 0
        assert config.eval_interval > 0
        
        # Invalid eval_samples
        with pytest.raises(ValueError, match="eval_samples must be positive"):
            EvaluationConfig(eval_samples=0)
            
        # Invalid eval_interval
        with pytest.raises(ValueError, match="eval_interval must be positive"):
            EvaluationConfig(eval_interval=-1)


class TestExperimentConfig:
    """Test experiment configuration creation"""
    
    def test_default_config_creation(self):
        """Test default configuration creation"""
        config = ExperimentConfig()
        assert config.name == "default_experiment"
        assert config.save_dir == "./experiments"
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.curriculum, CurriculumConfig)
        assert isinstance(config.evaluation, EvaluationConfig)
    
    def test_config_from_dict(self):
        """Test configuration creation from dictionary"""
        config_dict = {
            'name': 'test_experiment',
            'training': {
                'batch_size': 16,
                'learning_rate': 1e-4
            },
            'curriculum': {
                'method': 'traditional'
            }
        }
        
        config = ExperimentConfig.from_dict(config_dict)
        assert config.name == 'test_experiment'
        assert config.training.batch_size == 16
        assert config.training.learning_rate == 1e-4
        assert config.curriculum.method == 'traditional'
    
    def test_config_to_dict(self):
        """Test configuration conversion to dictionary"""
        config = ExperimentConfig(name='test')
        config_dict = config.to_dict()
        
        assert config_dict['name'] == 'test'
        assert 'model' in config_dict
        assert 'training' in config_dict
        assert 'curriculum' in config_dict
        assert 'evaluation' in config_dict
    
    def test_predefined_experiments(self):
        """Test predefined experiment configurations"""
        # Test RL curriculum
        rl_config = get_experiment_config('rl_curriculum')
        assert rl_config.name == 'rl_curriculum'
        assert rl_config.curriculum.method == 'rl'
        
        # Test traditional
        trad_config = get_experiment_config('traditional')
        assert trad_config.name == 'traditional'
        assert trad_config.curriculum.method == 'traditional'
        
        # Test easy to hard
        eth_config = get_experiment_config('easy_to_hard')
        assert eth_config.name == 'easy_to_hard'
        assert eth_config.curriculum.method == 'easy_to_hard'
        
        # Test hard to easy
        hte_config = get_experiment_config('hard_to_easy')
        assert hte_config.name == 'hard_to_easy'
        assert hte_config.curriculum.method == 'hard_to_easy'
        
        # Test invalid experiment type
        with pytest.raises(ValueError, match="Unknown experiment type"):
            get_experiment_config('invalid_type')
    
    def test_config_overrides(self):
        """Test configuration overrides"""
        overrides = {
            'training': {
                'batch_size': 32
            },
            'name': 'custom_rl_experiment'
        }
        
        config = get_experiment_config('rl_curriculum', **overrides)
        assert config.training.batch_size == 32
        assert config.name == 'custom_rl_experiment'