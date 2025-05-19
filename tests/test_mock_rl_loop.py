import numpy as np
import pytest
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.losses = []
        
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            self.losses.append(self.model.ep_info_buffer[-1]["r"])
        return True

class SimpleEnv(gym.Env):
    """A simple environment that returns random normal rewards."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.steps = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0.0]), {}
    
    def step(self, action):
        self.steps += 1
        reward = np.random.normal()
        done = self.steps >= 200
        return np.array([0.0]), reward, done, False, {}
    
class LearnableEnv(gym.Env):
    """
    Reward = 1 - (action - 0.5)^2  ∈ (-∞, 1].
    PPO can discover that the optimal action is ≈ 0.5.
    """
    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.action_space      = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.steps = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        a          = np.clip(action[0], -1, 1)
        reward     = 1.0 - (a - 0.5) ** 2               # peak = 1 at a = 0.5
        done       = self.steps >= 200
        return np.array([0.0], dtype=np.float32), reward, done, False, {}

@pytest.mark.fast
def test_ppo_training():
    env = DummyVecEnv([lambda: LearnableEnv()])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=64,
        batch_size=64,
        seed=42,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[dict(pi=[32], vf=[32])]),
    )

    rewards = []

    class RewardCallback(BaseCallback):
        def _on_step(self):
            rewards.append(float(self.locals["rewards"][0]))
            return True

    model.learn(total_timesteps=5_000, callback=RewardCallback())

    initial = np.mean(rewards[:500])
    final   = np.mean(rewards[-500:])
    assert final > initial + 0.1, f"reward did not improve: {initial:.3f} → {final:.3f}"