#!/usr/bin/env python3
"""
Thin wrapper: run PPO teacher for N episodes, checkpoint when dev_acc improves.
"""

import time, json
from teach_env import TeachEnv
from ppo_teacher import PPOTeacher

env = TeachEnv("gsm8k_train.jsonl", "gsm8k_dev.jsonl", device="cuda")
teacher = PPOTeacher(env)

best_acc = env.dev_acc
curricula = []

for step in range(2_000):
    teacher.train(total_steps=1)   # 1 PPO step = env.step + optimiser
    acc = env.dev_acc
    curricula.append(teacher.trainer.actor_actions[-1])

    if acc > best_acc:
        best_acc = acc
        ts = int(time.time())
        env.student.model.save_pretrained(f"ppo_student_best_{ts}")
        json.dump(curricula, open(f"ppo_curriculum_{ts}.json", "w"))
        print(f"💾 saved new best (+{(best_acc-env.baseline):.3%})")
