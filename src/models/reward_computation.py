"""
Reward computation for curriculum learning in knowledge distillation
"""

import torch
import torch.nn.functional as F
from typing import Dict, List

class FixedDistillationRewardComputer:
    """FIXED: Improved reward computation with proper scaling and robustness"""
    
    def __init__(self, reward_scale: float = 10.0, epsilon: float = 1e-6):
        """
        Initialize reward computer with scaling and numerical stability
        
        Args:
            reward_scale: Scale factor to make rewards comparable to main loss magnitude
            epsilon: Small value to prevent division by zero and ensure non-zero updates
        """
        self.reward_scale = reward_scale
        self.epsilon = epsilon
    
    def compute_batch_rewards(self, teacher_outputs, student_outputs, labels, problems):
        """
        FIXED: Improved reward computation with proper scaling and robustness
        - Scale rewards to be meaningful compared to main loss (5-6 range)
        - Add epsilon to prevent Q-value freezing
        - More robust aggregation
        """
        with torch.no_grad():
            # Proper alignment for next-token prediction
            teacher_logits = teacher_outputs.logits[:, :-1]  # [B, L-1, V]
            student_logits = student_outputs.logits[:, :-1]  # [B, L-1, V]
            gold_tokens = labels[:, 1:]  # [B, L-1] - what we want to predict
            
            # Get probabilities and predictions
            teacher_probs = F.softmax(teacher_logits, dim=-1)  # [B, L-1, V]
            student_probs = F.softmax(student_logits, dim=-1)  # [B, L-1, V]
            
            teacher_conf, teacher_preds = teacher_probs.max(dim=-1)  # [B, L-1]
            student_conf, student_preds = student_probs.max(dim=-1)  # [B, L-1]
            
            # Correct mask alignment
            valid_mask = (gold_tokens != -100)  # [B, L-1]
            
            # Check correctness with proper alignment
            teacher_correct = (teacher_preds == gold_tokens).float() * valid_mask
            student_correct = (student_preds == gold_tokens).float() * valid_mask
            
            # IMPROVED: Multiple reward signals with better scaling
            # Signal 1: SEC-style (teacher confident + correct, student uncertain)
            sec_reward = teacher_correct * teacher_conf * (1 - student_conf)
            
            # Signal 2: Knowledge transfer (teacher right, student wrong/uncertain)  
            knowledge_transfer = teacher_correct * teacher_conf * (1 - student_correct * student_conf)
            
            # Signal 3: Difficulty-based bonus (reward harder problems more)
            difficulty_bonus = torch.zeros_like(sec_reward)
            for i, problem in enumerate(problems):
                level = problem['level']
                # Higher level = higher bonus (normalized by max level 5)
                bonus_multiplier = level / 5.0
                difficulty_bonus[i] = bonus_multiplier
            
            # Combine signals with scaling
            base_reward = 0.5 * sec_reward + 0.3 * knowledge_transfer + 0.2 * difficulty_bonus * sec_reward
            
            # FIXED: Scale rewards to be comparable to main loss magnitude
            scaled_reward = base_reward * self.reward_scale
            
            # Average per example (only over valid tokens)
            valid_token_counts = valid_mask.sum(dim=1)  # [B]
            example_rewards = torch.zeros(valid_token_counts.shape[0], device=valid_mask.device)
            
            # Only compute rewards for examples with valid tokens
            for i in range(len(example_rewards)):
                if valid_token_counts[i] > 0:
                    example_rewards[i] = (scaled_reward[i] * valid_mask[i]).sum() / valid_token_counts[i]
                else:
                    # FIXED: Add small epsilon instead of zero to prevent Q-value freezing
                    example_rewards[i] = self.epsilon
            
            # Group by difficulty level
            difficulty_rewards = {}
            for i, problem in enumerate(problems):
                level = problem['level']
                reward_value = example_rewards[i].item()
                
                if level not in difficulty_rewards:
                    difficulty_rewards[level] = []
                difficulty_rewards[level].append(reward_value)
            
            # FIXED: More robust aggregation with epsilon baseline
            avg_difficulty_rewards = {}
            for level, rewards in difficulty_rewards.items():
                if len(rewards) > 0:
                    # Average rewards but ensure minimum epsilon value
                    avg_reward = sum(rewards) / len(rewards)
                    avg_difficulty_rewards[level] = max(avg_reward, self.epsilon)
                else:
                    # FIXED: Use epsilon instead of 0 to prevent curriculum freezing
                    avg_difficulty_rewards[level] = self.epsilon
            
            return avg_difficulty_rewards

class MultiObjectiveRewardComputer:
    """Multi-objective reward computation extending SEC"""
    
    def __init__(self, 
                 sec_weight: float = 0.5,
                 efficiency_weight: float = 0.2,
                 retention_weight: float = 0.2,
                 transfer_weight: float = 0.1,
                 reward_scale: float = 10.0):
        self.sec_weight = sec_weight
        self.efficiency_weight = efficiency_weight
        self.retention_weight = retention_weight
        self.transfer_weight = transfer_weight
        
        self.base_computer = FixedDistillationRewardComputer(reward_scale=reward_scale)
        
    def compute_batch_rewards(self, teacher_outputs, student_outputs, labels, problems):
        """Compute multi-objective rewards"""
        # Get base SEC-style rewards
        sec_rewards = self.base_computer.compute_batch_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Compute additional reward components
        efficiency_rewards = self._compute_efficiency_rewards(
            teacher_outputs, student_outputs, labels, problems
        )
        
        # Combine with proper weighting
        combined_rewards = {}
        for level in sec_rewards:
            combined_rewards[level] = (
                self.sec_weight * sec_rewards[level] +
                self.efficiency_weight * efficiency_rewards.get(level, self.base_computer.epsilon)
            )
            
        return combined_rewards
    
    def _compute_efficiency_rewards(self, teacher_outputs, student_outputs, labels, problems):
        """Compute sample efficiency rewards"""
        # IMPROVED: More meaningful efficiency computation
        with torch.no_grad():
            teacher_logits = teacher_outputs.logits[:, :-1]
            student_logits = student_outputs.logits[:, :-1]
            
            # Compute KL divergence as efficiency metric (lower = more efficient)
            kl_div = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='none'
            ).sum(dim=-1)  # [B, L-1]
            
            # Convert to reward (inverse of KL, scaled)
            efficiency_reward = torch.exp(-kl_div * 0.1)  # Small scaling factor
            
            # Average per example
            valid_mask = (labels[:, 1:] != -100)
            example_efficiency = torch.zeros(len(problems), device=efficiency_reward.device)
            
            for i in range(len(problems)):
                if valid_mask[i].any():
                    example_efficiency[i] = (efficiency_reward[i] * valid_mask[i]).sum() / valid_mask[i].sum()
                else:
                    example_efficiency[i] = self.base_computer.epsilon
        
        # Group by difficulty
        difficulty_rewards = {}
        for i, problem in enumerate(problems):
            level = problem['level']
            if level not in difficulty_rewards:
                difficulty_rewards[level] = []
            difficulty_rewards[level].append(example_efficiency[i].item())
        
        # Average per difficulty
        avg_difficulty_rewards = {}
        for level, rewards in difficulty_rewards.items():
            if rewards:
                avg_difficulty_rewards[level] = sum(rewards) / len(rewards)
            else:
                avg_difficulty_rewards[level] = self.base_computer.epsilon
                
        return avg_difficulty_rewards