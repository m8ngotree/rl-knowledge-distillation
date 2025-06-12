"""
RL-based Knowledge Distillation with SEC-inspired Curriculum Learning
Implements adaptive curriculum selection for distilling math reasoning capabilities
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, deque
import json
import wandb
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd
from tqdm import tqdm


class MATHDataset(Dataset):
    """MATH dataset with difficulty categorization"""
    
    def __init__(self, split='train', tokenizer=None, max_length=1024):
        self.dataset = load_dataset('hendrycks/math', split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Organize by difficulty level (1-5)
        self.difficulty_indices = defaultdict(list)
        for idx, item in enumerate(self.dataset):
            difficulty = item['level']  # e.g., "Level 1", "Level 2", etc.
            level = int(difficulty.split()[-1])
            self.difficulty_indices[level].append(idx)
        
        self.difficulties = list(self.difficulty_indices.keys())
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            'problem': item['problem'],
            'solution': item['solution'],
            'level': int(item['level'].split()[-1]),
            'type': item['type']
        }
    
    def get_by_difficulty(self, difficulty: int, n_samples: int):
        """Sample n problems from a specific difficulty level"""
        indices = np.random.choice(
            self.difficulty_indices[difficulty], 
            min(n_samples, len(self.difficulty_indices[difficulty])),
            replace=False
        )
        return [self[idx] for idx in indices]


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


class DistillationRewardComputer:
    """Compute rewards for curriculum learning in distillation context"""
    
    @staticmethod
    def compute_learning_potential(teacher_outputs, student_outputs, labels=None):
        """
        Compute learning potential as reward signal
        High reward when teacher is confident but student is not
        """
        rewards = []
        
        with torch.no_grad():
            # Get probabilities
            teacher_probs = F.softmax(teacher_outputs.logits, dim=-1)
            student_probs = F.softmax(student_outputs.logits, dim=-1)
            
            # Teacher and student confidence
            teacher_conf, teacher_pred = teacher_probs.max(dim=-1)
            student_conf, student_pred = student_probs.max(dim=-1)
            
            # Compute per-token rewards
            for i in range(len(teacher_conf)):
                # Base learning potential: teacher knows, student doesn't
                base_potential = teacher_conf[i] * (1 - student_conf[i])
                
                # Bonus for correct teacher predictions (if labels available)
                correctness_bonus = 1.0
                if labels is not None and i < len(labels):
                    if teacher_pred[i] == labels[i]:
                        correctness_bonus = 1.5
                
                # SEC insight: optimal learning at intermediate difficulty
                # Peak reward when student confidence around 0.5
                difficulty_factor = 4 * student_conf[i] * (1 - student_conf[i])
                
                reward = base_potential * correctness_bonus * (0.5 + 0.5 * difficulty_factor)
                rewards.append(reward)
        
        return torch.stack(rewards).mean().item()
    
    @staticmethod
    def compute_kl_divergence_reward(teacher_outputs, student_outputs):
        """KL divergence as learning signal"""
        with torch.no_grad():
            teacher_logprobs = F.log_softmax(teacher_outputs.logits, dim=-1)
            student_logprobs = F.log_softmax(student_outputs.logits, dim=-1)
            
            # KL(Teacher || Student)
            kl_div = F.kl_div(student_logprobs, teacher_logprobs.exp(), reduction='none').sum(dim=-1)
            
            # Higher KL = more to learn, but cap to avoid extreme values
            reward = torch.tanh(kl_div / 10.0).mean().item()
            
        return reward


class RLKnowledgeDistillation:
    """Main RL-based knowledge distillation trainer"""
    
    def __init__(
        self,
        teacher_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        student_name: str = "Qwen/Qwen2.5-1.5B",
        device: str = "cuda",
        alpha: float = 0.5,
        tau: float = 1.0,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-5,
        temperature: float = 3.0,
        alpha_distill: float = 0.7,
        save_dir: str = "./experiments"
    ):
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.alpha_distill = alpha_distill  # Weight for distillation vs. hard label loss
        
        # Create save directory
        self.save_dir = Path(save_dir) / f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models
        print(f"Loading teacher model: {teacher_name}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.teacher.eval()
        
        print(f"Loading student model: {student_name}")
        self.student = AutoModelForCausalLM.from_pretrained(
            student_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize curriculum
        self.curriculum = SECDistillationCurriculum(
            difficulties=[1, 2, 3, 4, 5],
            alpha=alpha,
            tau=tau
        )
        
        # Initialize dataset
        self.dataset = MATHDataset(split='train', tokenizer=self.tokenizer)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        
        # Tracking
        self.metrics = defaultdict(list)
        self.step = 0
        
    def prepare_batch(self, problems: List[Dict]) -> Dict:
        """Prepare a batch of problems for training"""
        # Format problems with instruction
        prompts = []
        solutions = []
        
        for p in problems:
            prompt = f"Problem: {p['problem']}\n\nSolution: "
            prompts.append(prompt)
            solutions.append(p['solution'])
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        targets = self.tokenizer(
            solutions,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': targets['input_ids'],
            'problems': problems
        }
    
    def distillation_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict[int, float]]:
        """Single distillation training step"""
        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
        
        # Get student outputs
        student_outputs = self.student(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Compute distillation loss
        loss_kd = F.kl_div(
            F.log_softmax(student_outputs.logits / self.temperature, dim=-1),
            F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Compute hard label loss
        loss_ce = F.cross_entropy(
            student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
            batch['labels'].view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Combined loss
        loss = self.alpha_distill * loss_kd + (1 - self.alpha_distill) * loss_ce
        
        # Compute rewards for each difficulty level in batch
        difficulty_rewards = defaultdict(list)
        reward_computer = DistillationRewardComputer()
        
        for i, problem in enumerate(batch['problems']):
            # Extract outputs for this example
            teacher_out_i = type('', (), {
                'logits': teacher_outputs.logits[i:i+1]
            })()
            student_out_i = type('', (), {
                'logits': student_outputs.logits[i:i+1]
            })()
            
            # Compute reward
            reward = reward_computer.compute_learning_potential(
                teacher_out_i, 
                student_out_i,
                batch['labels'][i:i+1]
            )
            
            difficulty_rewards[problem['level']].append(reward)
        
        # Average rewards by difficulty
        avg_rewards = {
            d: np.mean(rewards) for d, rewards in difficulty_rewards.items()
        }
        
        return loss, avg_rewards
    
    def train_epoch(self, n_steps: int = 100):
        """Train for one epoch with curriculum learning"""
        self.student.train()
        
        pbar = tqdm(range(n_steps), desc="Training")
        accumulated_loss = 0.0
        
        for step in pbar:
            # Select difficulties using curriculum
            difficulties = self.curriculum.select_difficulties(self.batch_size)
            
            # Sample problems from selected difficulties
            problems = []
            for d in difficulties:
                sampled = self.dataset.get_by_difficulty(d, 1)
                if sampled:
                    problems.extend(sampled)
            
            if not problems:
                continue
                
            # Prepare batch
            batch = self.prepare_batch(problems)
            
            # Forward pass
            loss, difficulty_rewards = self.distillation_step(batch)
            
            # Accumulate gradients
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update curriculum Q-values
                self.curriculum.update_q_values(difficulty_rewards)
                
                # Log metrics
                self.metrics['loss'].append(accumulated_loss)
                self.metrics['step'].append(self.step)
                
                for d, r in difficulty_rewards.items():
                    self.metrics[f'reward_level_{d}'].append(r)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{accumulated_loss:.4f}",
                    'q_values': {d: f"{q:.3f}" for d, q in self.curriculum.Q_values.items()}
                })
                
                accumulated_loss = 0.0
                self.step += 1
    
    def evaluate(self, n_samples: int = 100):
        """Evaluate student on test set"""
        self.student.eval()
        test_dataset = MATHDataset(split='test', tokenizer=self.tokenizer)
        
        results_by_difficulty = defaultdict(list)
        
        with torch.no_grad():
            for _ in range(n_samples):
                idx = np.random.randint(len(test_dataset))
                problem = test_dataset[idx]
                
                # Prepare input
                prompt = f"Problem: {problem['problem']}\n\nSolution: "
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).to(self.device)
                
                # Generate solution
                outputs = self.student.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True
                )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Simple correctness check (you'd want more sophisticated evaluation)
                # For now, just check if key parts of solution appear
                solution_text = problem['solution'].lower()
                generated_text = generated.lower()
                
                # Extract final answer if present
                correct = self._check_answer_match(solution_text, generated_text)
                
                results_by_difficulty[problem['level']].append(correct)
        
        # Compute accuracy by difficulty
        accuracy_by_level = {
            level: np.mean(results) if results else 0.0
            for level, results in results_by_difficulty.items()
        }
        
        return accuracy_by_level
    
    def _check_answer_match(self, solution: str, generated: str) -> bool:
        """Simple answer matching (would need more sophisticated evaluation)"""
        # Extract boxed answers
        import re
        
        def extract_boxed(text):
            pattern = r'\\boxed\{([^}]+)\}'
            matches = re.findall(pattern, text)
            return matches[-1] if matches else None
        
        sol_answer = extract_boxed(solution)
        gen_answer = extract_boxed(generated)
        
        if sol_answer and gen_answer:
            return sol_answer.strip() == gen_answer.strip()
        
        return False
    
    def save_results(self):
        """Save training results and plots"""
        # Save metrics
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save curriculum history
        curriculum_data = {
            'q_values': dict(self.curriculum.Q_values),
            'q_history': dict(self.curriculum.q_history),
            'reward_history': dict(self.curriculum.reward_history),
            'selection_counts': dict(self.curriculum.selection_counts)
        }
        with open(self.save_dir / 'curriculum_data.json', 'w') as f:
            json.dump(curriculum_data, f, indent=2)
        
        # Generate plots
        self._generate_plots()
    
    def _generate_plots(self):
        """Generate analysis plots"""
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['step'], self.metrics['loss'])
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(self.save_dir / 'loss_curve.png')
        plt.close()
        
        # Q-values evolution
        plt.figure(figsize=(12, 8))
        for difficulty in self.curriculum.difficulties:
            if difficulty in self.curriculum.q_history:
                plt.plot(self.curriculum.q_history[difficulty], label=f'Level {difficulty}')
        plt.xlabel('Updates')
        plt.ylabel('Q-value')
        plt.title('Q-values Evolution by Difficulty')
        plt.legend()
        plt.savefig(self.save_dir / 'q_values_evolution.png')
        plt.close()
        
        # Selection frequency
        plt.figure(figsize=(10, 6))
        difficulties = list(self.curriculum.selection_counts.keys())
        counts = [self.curriculum.selection_counts[d] for d in difficulties]
        plt.bar(difficulties, counts)
        plt.xlabel('Difficulty Level')
        plt.ylabel('Selection Count')
        plt.title('Curriculum Selection Frequency')
        plt.savefig(self.save_dir / 'selection_frequency.png')
        plt.close()


class TraditionalDistillation(RLKnowledgeDistillation):
    """Traditional knowledge distillation baseline"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override curriculum with uniform sampling
        self.use_curriculum = False
    
    def train_epoch(self, n_steps: int = 100):
        """Train with uniform sampling (no curriculum)"""
        self.student.train()
        
        pbar = tqdm(range(n_steps), desc="Training (Traditional)")
        accumulated_loss = 0.0
        
        for step in pbar:
            # Uniform random sampling from all difficulties
            problems = []
            for _ in range(self.batch_size):
                idx = np.random.randint(len(self.dataset))
                problems.append(self.dataset[idx])
            
            # Prepare batch
            batch = self.prepare_batch(problems)
            
            # Forward pass (same distillation loss)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
            
            student_outputs = self.student(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Compute losses
            loss_kd = F.kl_div(
                F.log_softmax(student_outputs.logits / self.temperature, dim=-1),
                F.softmax(teacher_outputs.logits / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            loss_ce = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                batch['labels'].view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
            
            loss = self.alpha_distill * loss_kd + (1 - self.alpha_distill) * loss_ce
            
            # Accumulate gradients
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Log metrics
                self.metrics['loss'].append(accumulated_loss)
                self.metrics['step'].append(self.step)
                
                pbar.set_postfix({'loss': f"{accumulated_loss:.4f}"})
                
                accumulated_loss = 0.0
                self.step += 1


def run_experiment(
    method: str = "rl",  # "rl" or "traditional"
    n_epochs: int = 3,
    steps_per_epoch: int = 100,
    eval_interval: int = 50
):
    """Run a complete experiment"""
    
    # Initialize trainer
    if method == "rl":
        trainer = RLKnowledgeDistillation(
            teacher_name="Qwen/Qwen2.5-Math-7B-Instruct",
            student_name="Qwen/Qwen2.5-1.5B",
            batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            alpha=0.5,  # SEC Q-value learning rate
            tau=1.0,    # SEC temperature
            save_dir=f"./experiments/{method}"
        )
    else:
        trainer = TraditionalDistillation(
            teacher_name="Qwen/Qwen2.5-Math-7B-Instruct",
            student_name="Qwen/Qwen2.5-1.5B",
            batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            save_dir=f"./experiments/{method}"
        )
    
    # Training loop
    eval_results = []
    
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")
        
        # Train
        trainer.train_epoch(steps_per_epoch)
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0:
            print("\nEvaluating...")
            accuracy_by_level = trainer.evaluate(n_samples=50)
            eval_results.append({
                'epoch': epoch + 1,
                'step': trainer.step,
                'accuracy_by_level': accuracy_by_level,
                'overall_accuracy': np.mean(list(accuracy_by_level.values()))
            })
            
            print(f"Accuracy by level: {accuracy_by_level}")
            print(f"Overall accuracy: {eval_results[-1]['overall_accuracy']:.3f}")
    
    # Save results
    trainer.save_results()
    
    # Save evaluation results
    with open(trainer.save_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    return trainer, eval_results


if __name__ == "__main__":
    # Example: Run both methods for comparison
    
    # Run RL-based distillation
    print("Running RL-based knowledge distillation...")
    rl_trainer, rl_results = run_experiment(method="rl", n_epochs=5, steps_per_epoch=100)
    
    # Run traditional distillation  
    print("\nRunning traditional knowledge distillation...")
    trad_trainer, trad_results = run_experiment(method="traditional", n_epochs=5, steps_per_epoch=100)
    
    # Compare results
    print("\n=== Final Comparison ===")
    print(f"RL-based final accuracy: {rl_results[-1]['overall_accuracy']:.3f}")
    print(f"Traditional final accuracy: {trad_results[-1]['overall_accuracy']:.3f}")