#!/usr/bin/env python3
"""
ε-greedy table over TeachEnv actions.
Implements:
  • choose_action()  – ε-greedy
  • update()         – Q-table incremental update
  • diagnostics()    – running averages for plots
"""

from collections import defaultdict, deque
import random, math, numpy as np
from typing import Dict, Tuple

class BanditTeacher:
    def __init__(self, num_examples: int, epsilon: float = 0.1, gamma: float = 0.1):
        """
        num_examples  – size of action space = num_examples × 3 reveal_levels
        epsilon       – exploration rate
        gamma         – α in Q ← Q + α (r – Q)
        """
        self.num_examples = num_examples
        self.epsilon = epsilon
        self.gamma   = gamma
        self.Q : Dict[Tuple[int,int], float] = defaultdict(float)
        self.diag_reward = deque(maxlen=100)   # for live plotting

    # ---------------------------------------------------------------
    def choose_action(self) -> Tuple[int,int]:
        if random.random() < self.epsilon:
            # explore ✔
            return random.randrange(self.num_examples), random.randrange(3)
        # exploit: argmax over three reveal levels for *one* random example;
        # cheaper than global argmax but still works well in practice
        ex = random.randrange(self.num_examples)
        best_level = max(range(3), key=lambda r: self.Q[(ex, r)])
        return ex, best_level

    def update(self, action, reward: float):
        key = (action.example_id, action.reveal_level)
        self.Q[key] += self.gamma * (reward - self.Q[key])
        self.diag_reward.append(reward)

    # ---------------------------------------------------------------
    def diagnostics(self):
        if not self.diag_reward:
            return 0.0
        return float(np.mean(self.diag_reward))
