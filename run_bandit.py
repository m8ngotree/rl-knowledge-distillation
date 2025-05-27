#!/usr/bin/env python3
"""
Runs BanditTeacher until *student* beats plain SFT by +2 pts on dev.
Logs JSON of picked curricula when improvement is reached.
"""

import json, time, random, os, tempfile
from pathlib import Path

from teach_env import TeachEnv, Action
from bandit_teacher import BanditTeacher
from teach_env import REVEAL_QUESTION_ONLY, REVEAL_HINT, REVEAL_FULL_COT

# ---------- 1 env ----------
env = TeachEnv(
    train_jsonl="gsm8k_train.jsonl",   # <-- point to full train or subset
    dev_jsonl="gsm8k_dev.jsonl",
    device="cuda",
    eval_every=25,
)

# ---------- 2 teacher ----------
teacher = BanditTeacher(num_examples=env.num_examples,
                        epsilon=0.1, gamma=0.1)

best_acc   = env.dev_acc
curricula  = []          # list of (example_id, reveal_level)
target_gain = 0.02       # +2 pts

# ---------- 3 loop ----------
for step in range(2_000):        # ≈ 4 h wall-clock on 4090
    ex_id, rv = teacher.choose_action()
    state, r, _, info = env.step(Action(ex_id, rv))
    teacher.update(Action(ex_id, rv), r)
    curricula.append((ex_id, rv))

    if info["recomputed_dev"] and info["dev_acc"] > best_acc:
        best_acc = info["dev_acc"]

    if (step+1) % 100 == 0:
        print(f"{step+1:05d}  mean_reward {teacher.diagnostics():+.4f}  "
              f"dev_acc {info['dev_acc']:.3%}  best {best_acc:.3%}")

    # ---------- 4 exit & save ----------
    if best_acc - env.baseline >= target_gain:
        ts = int(time.time())
        json.dump(curricula,
                  open(f"bandit_curriculum_{ts}.json", "w"))
        env.student.model.save_pretrained(f"student_best_{ts}")
        print("🎉 Bandit reached +2 pts!  models & curriculum saved.")
        break
