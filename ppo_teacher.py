#!/usr/bin/env python3
"""
Tiny PPO teacher using TRL’s PPOTrainer.
State  = TeachEnv._state_vec()  (length = 3 * history_len)
Action = (example_id, reveal_level)  encoded as single int:  id*3 + reveal
"""

import torch, torch.nn as nn
from trl import PPOTrainer, PPOConfig
from typing import Tuple
from teach_env import TeachEnv, Action, REVEAL_QUESTION_ONLY, REVEAL_HINT, REVEAL_FULL_COT

# ---------- policy network ----------
class Policy(nn.Module):
    def __init__(self, state_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# ---------- wrapper ----------
class PPOTeacher:
    def __init__(self, env: TeachEnv,
                 clip: float = 0.2,
                 kl_penalty: float = 0.01,
                 lr: float = 1e-4,
                 ent: float = 1e-3):

        self.env = env
        self.act_dim = env.num_examples * 3
        self.policy = Policy(state_dim=len(env._state_vec()), act_dim=self.act_dim)
        # TRL config
        cfg = PPOConfig(
            batch_size=64, mini_batch_size=32,
            learning_rate=lr, clip_range=clip, vf_coef=0.0,
            kl_penalty=kl_penalty, ent_coef=ent,
            seed=42,
        )
        self.trainer = PPOTrainer(cfg,
                                  actor_model=self.policy,
                                  critic_model=None, # use advantage as reward directly
                                  ref_model=None)

    # -----------------------------------------------------------
    def _int2action(self, idx: int) -> Action:
        ex_id, rv = divmod(idx, 3)
        return Action(example_id=ex_id, reveal_level=rv)

    def train(self, total_steps: int = 1_000):
        state = self.env.reset()
        for step in range(total_steps):
            logits = self.policy(torch.tensor(state))
            dist   = torch.distributions.Categorical(logits=logits)
            act_idx = dist.sample().item()
            action  = self._int2action(act_idx)

            new_state, reward, done, info = self.env.step(action)

            # TRL interface expects lists
            self.trainer.step([state], [act_idx], [reward], [new_state])
            state = new_state

            # optional live print
            if (step+1) % 100 == 0:
                print(f"[PPO] {step+1} / {total_steps}  mean_reward "
                      f"{self.trainer.reward_mean:.4f}   dev_acc {info['dev_acc']:.3%}")
