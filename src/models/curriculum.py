"""
Curriculum learning strategies for knowledge distillation
"""

import numpy as np
from collections import defaultdict
from typing import List, Dict

class SECDistillationCurriculum:
    """SEC-inspired curriculum for knowledge distillation"""
    
    def __init__(self, difficulties: List[int], alpha: float = 0.5, tau: float = 1.0):
        self.difficulties = difficulties
        self.Q_values = {d: 0.0 for d in difficulties}
        self.alpha = alpha  # Learning rate for Q-value updates
        self.tau = tau      # Temperature for Boltzmann sampling
        
        # Tracking for analysis
        self.q_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        self.selection_counts = defaultdict(int)
        
    def select_difficulties(self, batch_size: int) -> List[int]:
        """Select difficulties using Boltzmann distribution over Q-values"""
        # Compute probabilities
        exp_values = np.array([np.exp(self.Q_values[d] / self.tau) for d in self.difficulties])
        probs = exp_values / exp_values.sum()
        
        # Sample difficulties
        selected = np.random.choice(self.difficulties, size=batch_size, p=probs)
        
        # Track selections
        for d in selected:
            self.selection_counts[d] += 1
            
        return selected.tolist()
    
    def update_q_values(self, difficulty_rewards: Dict[int, float]):
        """Update Q-values using TD(0) as in SEC"""
        for difficulty, reward in difficulty_rewards.items():
            old_q = self.Q_values[difficulty]
            self.Q_values[difficulty] = self.alpha * reward + (1 - self.alpha) * old_q
            
            # Track history
            self.q_history[difficulty].append(self.Q_values[difficulty])
            self.reward_history[difficulty].append(reward)

class FixedCurriculum:
    """Base class for fixed curriculum strategies"""
    
    def __init__(self, difficulties: List[int], steps_per_difficulty: int = 50):
        self.difficulties = difficulties
        self.steps_per_difficulty = steps_per_difficulty
        self.current_step = 0
        
    def get_current_difficulty(self) -> int:
        """Get current difficulty based on step count"""
        raise NotImplementedError
        
    def select_difficulties(self, batch_size: int) -> List[int]:
        """Select current difficulty for all samples in batch"""
        current_difficulty = self.get_current_difficulty()
        return [current_difficulty] * batch_size
    
    def update_q_values(self, difficulty_rewards: Dict[int, float]):
        """No-op for fixed curricula"""
        pass
    
    def step(self):
        """Increment step counter"""
        self.current_step += 1

class EasyToHardCurriculum(FixedCurriculum):
    """Fixed curriculum: Easy to Hard"""
    
    def __init__(self, difficulties: List[int], steps_per_difficulty: int = 50):
        super().__init__(sorted(difficulties), steps_per_difficulty)
        
    def get_current_difficulty(self) -> int:
        """Get current difficulty (increasing)"""
        difficulty_idx = min(
            self.current_step // self.steps_per_difficulty,
            len(self.difficulties) - 1
        )
        return self.difficulties[difficulty_idx]

class HardToEasyCurriculum(FixedCurriculum):
    """Fixed curriculum: Hard to Easy"""
    
    def __init__(self, difficulties: List[int], steps_per_difficulty: int = 50):
        super().__init__(sorted(difficulties, reverse=True), steps_per_difficulty)
        
    def get_current_difficulty(self) -> int:
        """Get current difficulty (decreasing)"""
        difficulty_idx = min(
            self.current_step // self.steps_per_difficulty,
            len(self.difficulties) - 1
        )
        return self.difficulties[difficulty_idx]

class RandomCurriculum:
    """Random/uniform curriculum (baseline)"""
    
    def __init__(self, difficulties: List[int]):
        self.difficulties = difficulties
        
    def select_difficulties(self, batch_size: int) -> List[int]:
        """Randomly select difficulties"""
        return np.random.choice(self.difficulties, size=batch_size).tolist()
    
    def update_q_values(self, difficulty_rewards: Dict[int, float]):
        """No-op for random curriculum"""
        pass