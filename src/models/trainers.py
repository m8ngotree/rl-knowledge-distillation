"""
Training classes for different distillation methods
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from ..data.dataset import MATHDataset
from ..models.curriculum import (
    SECDistillationCurriculum, 
    EasyToHardCurriculum, 
    HardToEasyCurriculum,
    RandomCurriculum
)
from ..models.reward_computation import FixedDistillationRewardComputer
from ..evaluation.evaluator import ModelEvaluator
from ..utils.monitoring import TrainingMonitor, MetricsTracker
from ..utils.serialization import ExperimentSaver
from ..utils.logging_config import setup_experiment_logging, log_gpu_memory

class BaseDistillationTrainer:
    """Base class for knowledge distillation training"""
    
    def __init__(
        self,
        teacher_model_name: str = "Qwen/Qwen2-Math-1.5B-Instruct",
        student_model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        device: str = "cuda",
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-5,
        temperature: float = 3.0,
        alpha_distill: float = 0.7,
        save_dir: str = "./experiments",
        max_length: int = 512,
        seed: int = 42
    ):
        # Basic attributes
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.alpha_distill = alpha_distill
        self.max_length = max_length

        # ------------------------------------------------------------------
        # Logger must be created BEFORE any method that logs is invoked
        # ------------------------------------------------------------------
        from ..utils.logging_config import get_logger
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        # ------------------------------------------------------------------
        # Reproducibility – set global seed
        # ------------------------------------------------------------------
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create save directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(save_dir) / f"exp_{timestamp}"
        self.saver = ExperimentSaver(self.save_dir)
        
        # Load models (now safe because logger exists)
        self._load_models(teacher_model_name, student_model_name)
        
        # Initialize dataset and logger
        self.logger.info("Loading MATH dataset...")
        self.dataset = MATHDataset(split='train', tokenizer=self.tokenizer)
        
        # Initialize evaluation with LLM-based answer comparison
        self.evaluator = ModelEvaluator(self.tokenizer, self.device)
        
        # Initialize tracking
        self.metrics_tracker = MetricsTracker()
        self.step = 0
        self.total_training_examples = 0
        
        # Optimizer & AMP scaler
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=learning_rate)
        self.use_amp = (self.device == "cuda" and torch.cuda.is_available())
        self.scaler: Optional[GradScaler] = GradScaler() if self.use_amp else None
        
        # GPU memory management
        self._setup_memory_management()
        
    def _load_models(self, teacher_model_name: str, student_model_name: str):
        """Load teacher and student models"""
        self.logger.info(f"Loading teacher model: {teacher_model_name}")
        self.teacher = AutoModelForCausalLM.from_pretrained(
            teacher_model_name, 
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        self.teacher.eval()
        
        self.logger.info(f"Loading student model: {student_model_name}")
        self.student = AutoModelForCausalLM.from_pretrained(
            student_model_name,
            torch_dtype=torch.float16,
            device_map=None
        ).to(self.device)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Verify model loading
        self._verify_model_loading()
    
    def _verify_model_loading(self):
        """Verify both models loaded correctly"""
        self.logger.info("="*60)
        self.logger.info("MODEL LOADING VERIFICATION")
        self.logger.info("="*60)
        
        test_input = "What is 2 + 2?"
        
        try:
            # Test tokenizer
            tokens = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            self.logger.info(f"✅ Tokenizer: {len(tokens['input_ids'][0])} tokens")
            
            # Test teacher model
            with torch.no_grad():
                teacher_out = self.teacher(**tokens)
                self.logger.info(f"✅ Teacher forward pass: {teacher_out.logits.shape}")
            
            # Test student model
            student_out = self.student(**tokens)
            self.logger.info(f"✅ Student forward pass: {student_out.logits.shape}")
            
            # Verify compatibility
            assert teacher_out.logits.shape[-1] == student_out.logits.shape[-1], \
                f"Vocab size mismatch: Teacher {teacher_out.logits.shape[-1]} vs Student {student_out.logits.shape[-1]}"
            
            self.logger.info(f"✅ All model verification tests passed!")
            self.logger.info(f"✅ Vocabulary size: {teacher_out.logits.shape[-1]}")
            self.logger.info("="*60)
            
        except Exception as e:
            self.logger.error(f"❌ Model verification failed: {e}")
            raise e
    
    def _setup_memory_management(self):
        """Setup GPU memory management and monitoring"""
        if self.device == "cuda" and torch.cuda.is_available():
            # Clear cache at start
            torch.cuda.empty_cache()
            
            # Log initial memory usage
            log_gpu_memory(self.logger, "Initial Setup")
            
            # Use explicit device index to avoid string device issues
            device_index = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
            self.logger.info(f"Total GPU Memory: {total_memory:.2f}GB")
    
    def _cleanup_memory(self):
        """Clean up GPU memory between steps"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _check_memory_usage(self):
        """Check and log current memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device_index) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
            return {'allocated': memory_allocated, 'reserved': memory_reserved}
        return {'allocated': 0, 'reserved': 0}
    
    def prepare_batch(self, problems: List[Dict], mode: str = "train") -> Dict:
        """
        FIXED: Prepare batch for proper knowledge distillation
        
        Args:
            problems: List of problem dictionaries
            mode: "train" or "eval" - in train mode, only use prompts (no ground truth)
        """
        if mode == "train":
            # FIXED: For training, only use prompts - no ground truth solutions
            # This forces the model to actually generate solutions, not just copy them
            texts = []
            for p in problems:
                prompt = f"Problem: {p['problem']}\n\nSolution: Please solve this step-by-step. Use LaTeX formatting for mathematical expressions (fractions, matrices, etc.). Provide your final answer in \\boxed{{}} format."
                texts.append(prompt)
            
            # Tokenize prompts only
            tokens = self.tokenizer(
                texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # For training, we don't have labels - model will generate from prompt
            return {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'problems': problems,
                'mode': mode
            }
        
        else:
            # FIXED: For evaluation, include ground truth for comparison
            full_texts = []
            prompts = []
            
            for p in problems:
                prompt = f"Problem: {p['problem']}\n\nSolution: Please solve this step-by-step. Use LaTeX formatting for mathematical expressions (fractions, matrices, etc.). Provide your final answer in \\boxed{{}} format."
                full_text = prompt + p['solution']
                full_texts.append(full_text)
                prompts.append(prompt)
            
            # Tokenize full sequences
            tokens = self.tokenizer(
                full_texts,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Create labels with prompt tokens masked
            labels = tokens['input_ids'].clone()
            for i, prompt in enumerate(prompts):
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
                prompt_len = len(prompt_tokens)
                labels[i, :prompt_len] = -100  # Ignore prompt tokens in loss
            
            return {
                'input_ids': tokens['input_ids'],
                'attention_mask': tokens['attention_mask'],
                'labels': labels,
                'problems': problems,
                'mode': mode
            }
    
    def distillation_step(self, batch: Dict) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        FIXED: Distillation step that returns teacher outputs to avoid double forward pass
        """
        if batch['mode'] == 'train':
            # FIXED: For training mode, use generation-based distillation
            return self._generation_based_distillation(batch)
        else:
            # For evaluation mode, use the original teacher-forcing approach
            return self._teacher_forcing_distillation(batch)
    
    def _generation_based_distillation(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        FIXED: Generation-based distillation for actual knowledge transfer
        """
        amp_enabled = self.use_amp
        with autocast(enabled=amp_enabled):
            # Generate teacher responses
            with torch.no_grad():
                teacher_outputs = self.teacher.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Get teacher logits for the generated sequence
                teacher_sequences = teacher_outputs.sequences
                teacher_logits_list = teacher_outputs.scores  # List of [B, V] tensors
                
                # Stack teacher logits
                if teacher_logits_list:
                    teacher_logits = torch.stack(teacher_logits_list, dim=1)  # [B, new_tokens, V]
                else:
                    # Fallback if no generation happened
                    teacher_forward = self.teacher(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    teacher_logits = teacher_forward.logits
                    teacher_sequences = batch['input_ids']
            
            # Generate student responses for the same inputs
            student_outputs = self.student.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            student_sequences = student_outputs.sequences
            student_logits_list = student_outputs.scores
            
            if student_logits_list:
                student_logits = torch.stack(student_logits_list, dim=1)  # [B, new_tokens, V]
            else:
                # Fallback
                student_forward = self.student(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                student_logits = student_forward.logits
            
            # Compute KL divergence loss between teacher and student generation distributions
            min_length = min(teacher_logits.size(1), student_logits.size(1))
            teacher_logits_trunc = teacher_logits[:, :min_length, :]
            student_logits_trunc = student_logits[:, :min_length, :]
            
            loss_kd = F.kl_div(
                F.log_softmax(student_logits_trunc / self.temperature, dim=-1),
                F.softmax(teacher_logits_trunc / self.temperature, dim=-1),
                reduction='batchmean'
            ) * (self.temperature ** 2)
            
            # Return teacher outputs for reward computation
            teacher_output_dict = type('', (), {
                'logits': teacher_logits,
                'sequences': teacher_sequences
            })()
            
            student_output_dict = type('', (), {
                'logits': student_logits,
                'sequences': student_sequences
            })()
            
            return loss_kd, {
                'teacher_outputs': teacher_output_dict,
                'student_outputs': student_output_dict
            }
    
    def _teacher_forcing_distillation(self, batch: Dict) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Original teacher-forcing distillation for evaluation"""
        amp_enabled = self.use_amp
        with autocast(enabled=amp_enabled):
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )

            # Student forward
            student_outputs = self.student(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Alignment for KD + CE
            teacher_logits = teacher_outputs.logits[:, :-1]  # [B, L-1, V]
            student_logits = student_outputs.logits[:, :-1]  # [B, L-1, V]
            target_tokens = batch['labels'][:, 1:]           # [B, L-1]

            valid_mask = (target_tokens != -100)

            # --- KL Divergence ---
            if valid_mask.any():
                loss_kd = F.kl_div(
                    F.log_softmax(student_logits[valid_mask] / self.temperature, dim=-1),
                    F.softmax(teacher_logits[valid_mask] / self.temperature, dim=-1),
                    reduction='batchmean'
                ) * (self.temperature ** 2)
            else:
                loss_kd = torch.tensor(0.0, device=self.device)

            # --- Cross-Entropy --- (use shifted logits & labels)
            if valid_mask.any():
                loss_ce = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)),
                    target_tokens.reshape(-1),
                    ignore_index=-100
                )
            else:
                loss_ce = torch.tensor(0.0, device=self.device)

            loss = self.alpha_distill * loss_kd + (1 - self.alpha_distill) * loss_ce
        
        return loss, {
            'teacher_outputs': teacher_outputs,
            'student_outputs': student_outputs
        }
    
    def train_epoch(self, n_steps: int = 100):
        """Train for one epoch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def evaluate(self, n_samples: int = 100, log_solutions: bool = True) -> Dict:
        """FIXED: Evaluate the student model using proper evaluation mode"""
        # For evaluation, we want to use the traditional approach with ground truth
        # so we can compute proper accuracy metrics
        results = self.evaluator.evaluate_model(
            self.student, 
            self.dataset, 
            n_samples=n_samples, 
            log_solutions=log_solutions,
            model_type='student',
            response_storage=self.saver.get_response_storage() if log_solutions else None
        )
        
        # Log to metrics tracker
        self.metrics_tracker.log_evaluation(
            step=self.step,
            accuracy=results['overall_accuracy'],
            teacher_student_agreement=results.get('teacher_student_agreement', 0.0),
            generated_solutions=results.get('generated_solutions', [])
        )
        
        return results
    
    def save_results(self):
        """Save all training results"""
        # Get training summary
        summary = self.metrics_tracker.get_summary()
        
        # Save all components
        self.saver.save_metrics(summary['metrics'])
        self.saver.save_training_history(summary['training_history'])
        self.saver.save_model_checkpoint(self.student, 'student_model')
        
        # Save organized responses
        self.saver.get_response_storage().save_responses()
        
        return self.saver.get_save_dir()

class RLDistillationTrainer(BaseDistillationTrainer):
    """RL-based curriculum distillation trainer"""
    
    def __init__(self, alpha: float = 0.5, tau: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize SEC curriculum
        available_difficulties = self.dataset.difficulties
        self.logger.info(f"Initializing RL curriculum with difficulties: {available_difficulties}")
        
        self.curriculum = SECDistillationCurriculum(
            difficulties=available_difficulties,
            alpha=alpha,
            tau=tau
        )
        
        # Initialize reward computer
        self.reward_computer = FixedDistillationRewardComputer()
    
    def train_epoch(self, n_steps: int = 100):
        """FIXED: Train for one epoch with RL curriculum - no double teacher forward pass"""
        self.student.train()
        
        monitor = TrainingMonitor(log_interval=20)
        pbar = tqdm(range(n_steps), desc="Training (RL Curriculum)")
        
        for step in pbar:
            # Select difficulties using curriculum
            difficulties = self.curriculum.select_difficulties(self.batch_size)
            
            # Efficiently sample problems using batch method
            problems = self.dataset.get_batch_by_difficulties(difficulties, n_per_difficulty=1)
            
            if not problems:
                continue
            
            # Track curriculum selections
            difficulty_counts = defaultdict(int)
            for p in problems:
                difficulty_counts[p['level']] += 1
            
            # FIXED: Prepare batch for training mode (prompts only)
            batch = self.prepare_batch(problems, mode="train")
            
            # FIXED: Get both loss and model outputs in one call
            loss, outputs = self.distillation_step(batch)
            
            # FIXED: Use actual batch size for gradient accumulation scaling
            actual_batch_size = len(problems)
            effective_accumulation_steps = max(1, self.gradient_accumulation_steps * actual_batch_size // self.batch_size)
            
            # FIXED: Extract teacher and student outputs (no double forward pass)
            if outputs and 'teacher_outputs' in outputs and 'student_outputs' in outputs:
                teacher_outputs = outputs['teacher_outputs']
                student_outputs = outputs['student_outputs']
                
                # Create mock labels for reward computation (since we're using generation)
                # Use the generated teacher sequences as "ground truth" for reward computation
                mock_labels = teacher_outputs.sequences.clone()
                # Mask the prompt part
                prompt_length = batch['input_ids'].size(1)
                mock_labels[:, :prompt_length] = -100
                
                # Compute curriculum rewards using the outputs we already have
                difficulty_rewards = self.reward_computer.compute_batch_rewards(
                    teacher_outputs, student_outputs, mock_labels, batch['problems']
                )
            else:
                # Fallback: empty rewards if outputs not available
                difficulty_rewards = {level: self.reward_computer.epsilon for level in difficulties}
            
            # FIXED: Scale loss by effective accumulation steps
            micro_loss = loss / effective_accumulation_steps
            if self.scaler:
                self.scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Update curriculum Q-values
                self.curriculum.update_q_values(difficulty_rewards)
                
                # FIXED: Log metrics with correct loss scaling
                full_batch_loss = micro_loss.item() * effective_accumulation_steps
                self.metrics_tracker.log_training_step(
                    step=self.step,
                    loss=full_batch_loss,
                    curriculum_selections=difficulty_counts,
                    q_values=self.curriculum.Q_values
                )
                
                # Monitor training
                monitor.log_training_step(
                    loss=full_batch_loss,
                    curriculum_selections=difficulty_counts,
                    q_values=self.curriculum.Q_values,
                    batch_info=f"actual_batch_size={actual_batch_size}, effective_accum={effective_accumulation_steps}"
                )
                
                pbar.set_postfix({'loss': f"{full_batch_loss:.4f}"})
                
                # Clean up memory periodically
                if self.step % 50 == 0:
                    self._cleanup_memory()
                
                self.step += 1
                self.total_training_examples += actual_batch_size
    
    def save_results(self):
        """Save RL training results including curriculum data"""
        # Save base results
        save_dir = super().save_results()
        
        # Save curriculum-specific data
        curriculum_data = {
            'q_values': dict(self.curriculum.Q_values),
            'q_history': dict(self.curriculum.q_history),
            'reward_history': dict(self.curriculum.reward_history),
            'selection_counts': dict(self.curriculum.selection_counts)
        }
        self.saver.save_curriculum_data(curriculum_data)
        
        return save_dir
    
    def evaluate_teacher_student_pair(self, n_samples: int = 100) -> Dict:
        """Evaluate teacher-student agreement with organized storage"""
        results = self.evaluator.evaluate_teacher_student_pair(
            self.teacher,
            self.student,
            self.dataset,
            n_samples=n_samples,
            response_storage=self.saver.get_response_storage()
        )
        
        # Log comparative results
        self.metrics_tracker.log_evaluation(
            step=self.step,
            accuracy=results['student_overall_accuracy'],
            teacher_student_agreement=results['overall_agreement']
        )
        
        return results
    
    def get_response_storage_stats(self) -> Dict:
        """Get statistics about stored responses"""
        return self.saver.get_response_storage().get_stats()

class TraditionalDistillationTrainer(BaseDistillationTrainer):
    """Traditional knowledge distillation baseline"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.curriculum = RandomCurriculum(self.dataset.difficulties)
    
    def train_epoch(self, n_steps: int = 100):
        """FIXED: Train with uniform sampling (no curriculum) - proper batch handling"""
        self.student.train()
        
        pbar = tqdm(range(n_steps), desc="Training (Traditional)")
        
        for step in pbar:
            # Uniform random sampling using optimized batch method
            problems = self.dataset.get_random_batch(self.batch_size)
            
            # FIXED: Use training mode for proper knowledge distillation
            batch = self.prepare_batch(problems, mode="train")
            loss, _ = self.distillation_step(batch)
            
            # FIXED: Use actual batch size for gradient accumulation scaling
            actual_batch_size = len(problems)
            effective_accumulation_steps = max(1, self.gradient_accumulation_steps * actual_batch_size // self.batch_size)
            
            # FIXED: Scale loss by effective accumulation steps
            micro_loss = loss / effective_accumulation_steps
            if self.scaler:
                self.scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # FIXED: Log metrics with correct loss scaling
                full_batch_loss = micro_loss.item() * effective_accumulation_steps
                self.metrics_tracker.log_training_step(
                    step=self.step,
                    loss=full_batch_loss
                )
                
                pbar.set_postfix({'loss': f"{full_batch_loss:.4f}"})
                
                self.step += 1
                self.total_training_examples += actual_batch_size

class FixedCurriculumTrainer(BaseDistillationTrainer):
    """Base class for fixed curriculum trainers"""
    
    def __init__(self, curriculum_type: str = "easy_to_hard", **kwargs):
        super().__init__(**kwargs)
        
        if curriculum_type == "easy_to_hard":
            self.curriculum = EasyToHardCurriculum(self.dataset.difficulties)
        elif curriculum_type == "hard_to_easy":
            self.curriculum = HardToEasyCurriculum(self.dataset.difficulties)
        else:
            raise ValueError(f"Unknown curriculum type: {curriculum_type}")
        
        self.curriculum_type = curriculum_type
    
    def train_epoch(self, n_steps: int = 100):
        """FIXED: Train with fixed curriculum - proper batch handling"""
        self.student.train()
        
        pbar = tqdm(range(n_steps), desc=f"Training ({self.curriculum_type})")
        
        for step in pbar:
            # Sample problems from current curriculum using batch method
            difficulties = self.curriculum.select_difficulties(self.batch_size)
            problems = self.dataset.get_batch_by_difficulties(difficulties, n_per_difficulty=1)
            
            if not problems:
                continue
            
            # FIXED: Use training mode for proper knowledge distillation
            batch = self.prepare_batch(problems, mode="train")
            loss, _ = self.distillation_step(batch)
            
            # FIXED: Use actual batch size for gradient accumulation scaling
            actual_batch_size = len(problems)
            effective_accumulation_steps = max(1, self.gradient_accumulation_steps * actual_batch_size // self.batch_size)
            
            # FIXED: Scale loss by effective accumulation steps
            micro_loss = loss / effective_accumulation_steps
            if self.scaler:
                self.scaler.scale(micro_loss).backward()
            else:
                micro_loss.backward()
            
            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Step curriculum
                if hasattr(self.curriculum, 'step'):
                    self.curriculum.step()
                
                # FIXED: Log metrics with correct loss scaling
                full_batch_loss = micro_loss.item() * effective_accumulation_steps
                self.metrics_tracker.log_training_step(
                    step=self.step,
                    loss=full_batch_loss
                )
                
                current_difficulty = getattr(self.curriculum, 'get_current_difficulty', lambda: 'N/A')()
                pbar.set_postfix({
                    'loss': f"{full_batch_loss:.4f}",
                    'difficulty': current_difficulty,
                    'actual_batch': actual_batch_size
                })
                
                self.step += 1
                self.total_training_examples += actual_batch_size