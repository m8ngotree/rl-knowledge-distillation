"""
Test reward computation
"""

import pytest
import torch
from unittest.mock import Mock
from src.models.reward_computation import FixedDistillationRewardComputer


class TestRewardComputation:
    """Test reward computation functionality"""
    
    def test_fixed_distillation_reward_computer(self):
        """Test fixed distillation reward computation"""
        # Create mock model outputs
        batch_size, seq_len, vocab_size = 2, 10, 100
        
        # Mock teacher outputs (confident and correct)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits[:, :, 0] = 10.0  # Make token 0 very likely
        teacher_outputs = Mock()
        teacher_outputs.logits = teacher_logits
        
        # Mock student outputs (less confident)
        student_logits = torch.randn(batch_size, seq_len, vocab_size) * 0.5
        student_outputs = Mock()
        student_outputs.logits = student_logits
        
        # Mock labels (target tokens are 0, some positions masked)
        labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
        labels[:, :3] = -100  # Mask first 3 positions (prompt tokens)
        
        # Mock problems
        problems = [
            {'level': 1, 'problem': 'test1'},
            {'level': 2, 'problem': 'test2'}
        ]
        
        # Compute rewards
        computer = FixedDistillationRewardComputer()
        rewards = computer.compute_batch_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Check that rewards are computed for each difficulty level
        assert isinstance(rewards, dict)
        assert 1 in rewards
        assert 2 in rewards
        
        # Check that rewards are reasonable values
        for level, reward in rewards.items():
            assert isinstance(reward, (int, float))
            assert reward >= 0.0  # Rewards should be non-negative
    
    def test_reward_computer_with_no_valid_tokens(self):
        """Test reward computation when all tokens are masked"""
        batch_size, seq_len, vocab_size = 1, 5, 10
        
        # Mock outputs
        teacher_outputs = Mock()
        teacher_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        student_outputs = Mock()
        student_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # All tokens masked
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        
        problems = [{'level': 1, 'problem': 'test'}]
        
        computer = FixedDistillationRewardComputer()
        rewards = computer.compute_batch_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Should handle masked tokens gracefully
        assert isinstance(rewards, dict)
        assert 1 in rewards
        assert rewards[1] == computer.epsilon  # Should be epsilon, not 0, to prevent Q-value freezing
    
    def test_reward_computer_division_by_zero_protection(self):
        """Test that division by zero is prevented"""
        batch_size, seq_len, vocab_size = 2, 5, 10
        
        # Mock outputs
        teacher_outputs = Mock()
        teacher_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        student_outputs = Mock()
        student_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Some valid tokens
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, 0] = -100  # Mask first token
        
        # Empty problems list (edge case)
        problems = []
        
        computer = FixedDistillationRewardComputer()
        rewards = computer.compute_batch_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Should return empty dict without crashing
        assert isinstance(rewards, dict)
        assert len(rewards) == 0
    
    def test_reward_aggregation_by_difficulty(self):
        """Test that rewards are properly aggregated by difficulty level"""
        batch_size, seq_len, vocab_size = 4, 5, 10
        
        # Mock outputs
        teacher_outputs = Mock()
        teacher_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        student_outputs = Mock()
        student_outputs.logits = torch.randn(batch_size, seq_len, vocab_size)
        
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Multiple problems with same difficulty levels
        problems = [
            {'level': 1, 'problem': 'test1'},
            {'level': 1, 'problem': 'test2'},  # Same level as first
            {'level': 2, 'problem': 'test3'},
            {'level': 2, 'problem': 'test4'}   # Same level as third
        ]
        
        computer = FixedDistillationRewardComputer()
        rewards = computer.compute_batch_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Should aggregate rewards for each difficulty level
        assert len(rewards) == 2  # Two unique difficulty levels
        assert 1 in rewards
        assert 2 in rewards
        
        # Each reward should be the average of the individual problem rewards
        for reward in rewards.values():
            assert isinstance(reward, (int, float))
            assert reward >= 0.0