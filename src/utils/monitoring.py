"""
Training monitoring and logging utilities
"""

import time
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, Any

class TrainingMonitor:
    """Real-time training monitoring and logging"""
    
    def __init__(self, log_interval: int = 10, window_size: int = 50):
        self.log_interval = log_interval
        self.window_size = window_size
        self.step_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Moving averages
        self.loss_window = deque(maxlen=window_size)
        self.step_times = deque(maxlen=window_size)
        
    def log_training_step(self, 
                         loss: float, 
                         curriculum_selections: Optional[Dict] = None, 
                         q_values: Optional[Dict] = None, 
                         generated_examples: Optional[Any] = None, 
                         batch_info: Optional[str] = None):
        """Log training step with comprehensive monitoring"""
        current_time = time.time()
        step_time = current_time - self.last_log_time
        
        self.loss_window.append(loss)
        self.step_times.append(step_time)
        self.last_log_time = current_time
        
        if self.step_count % self.log_interval == 0:
            self._print_status(loss, curriculum_selections, q_values, 
                             generated_examples, batch_info, current_time)
        
        self.step_count += 1
    
    def _print_status(self, 
                     loss: float, 
                     curriculum_selections: Optional[Dict], 
                     q_values: Optional[Dict], 
                     generated_examples: Optional[Any], 
                     batch_info: Optional[str], 
                     current_time: float):
        """Print detailed training status"""
        elapsed = current_time - self.start_time
        avg_loss = np.mean(self.loss_window) if self.loss_window else loss
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        
        print(f"\n{'='*80}")
        print(f"TRAINING STATUS - Step {self.step_count}")
        print(f"{'='*80}")
        print(f"Time: {elapsed:.1f}s elapsed, {avg_step_time:.2f}s/step avg")
        print(f"Loss: current={loss:.4f}, avg={avg_loss:.4f}")
        
        if curriculum_selections:
            print(f"Curriculum selections: {dict(curriculum_selections)}")
        
        if q_values:
            print("Q-values:")
            for difficulty, q_val in sorted(q_values.items()):
                print(f"  Level {difficulty}: {q_val:+.4f}")
        
        if batch_info:
            print(f"Batch info: {batch_info}")
        
        print(f"{'='*80}")

class MetricsTracker:
    """Track and aggregate training metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.training_history = {
            'steps': [],
            'losses': [],
            'accuracies': [],
            'q_values_history': defaultdict(list),
            'curriculum_selections': [],
            'teacher_student_agreement': [],
            'generated_solutions': [],
            'training_examples_seen': []
        }
        
    def log_metric(self, key: str, value: Any, step: Optional[int] = None):
        """Log a metric"""
        self.metrics[key].append(value)
        if step is not None:
            self.metrics[f'{key}_step'].append(step)
    
    def log_training_step(self, 
                         step: int, 
                         loss: float, 
                         curriculum_selections: Optional[Dict] = None,
                         q_values: Optional[Dict] = None):
        """Log training step data"""
        self.training_history['steps'].append(step)
        self.training_history['losses'].append(loss)
        
        if curriculum_selections:
            self.training_history['curriculum_selections'].append(dict(curriculum_selections))
            
        if q_values:
            for difficulty, q_val in q_values.items():
                self.training_history['q_values_history'][difficulty].append(q_val)
    
    def log_evaluation(self, 
                      step: int, 
                      accuracy: float, 
                      teacher_student_agreement: float,
                      generated_solutions: Optional[list] = None):
        """Log evaluation results"""
        self.training_history['accuracies'].append(accuracy)
        self.training_history['teacher_student_agreement'].append(teacher_student_agreement)
        
        if generated_solutions:
            self.training_history['generated_solutions'].extend(generated_solutions)
    
    def get_summary(self) -> Dict:
        """Get training summary"""
        return {
            'total_steps': len(self.training_history['steps']),
            'final_accuracy': self.training_history['accuracies'][-1] if self.training_history['accuracies'] else 0,
            'final_loss': self.training_history['losses'][-1] if self.training_history['losses'] else 0,
            'metrics': dict(self.metrics),
            'training_history': dict(self.training_history)
        }