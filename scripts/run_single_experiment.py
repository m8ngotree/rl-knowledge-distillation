#!/usr/bin/env python3
"""
Run a single knowledge distillation experiment
"""

import sys
import argparse
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.trainers import (
    RLDistillationTrainer,
    TraditionalDistillationTrainer,
    FixedCurriculumTrainer
)
from config.experiment_config import BASE_CONFIG

def run_experiment(
    method: str = "rl",
    n_epochs: int = 3,
    steps_per_epoch: int = 100,
    eval_interval: int = 1,
    **kwargs
):
    """Run a complete experiment"""
    
    print(f"Running {method} experiment...")
    print(f"Configuration: {kwargs}")
    
    # Initialize trainer based on method
    if method == "rl":
        trainer = RLDistillationTrainer(
            batch_size=kwargs.get('batch_size', 4),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            alpha=kwargs.get('alpha', 0.5),
            tau=kwargs.get('tau', 1.0),
            save_dir=f"./experiments/{method}"
        )
    elif method == "traditional":
        trainer = TraditionalDistillationTrainer(
            batch_size=kwargs.get('batch_size', 4),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            save_dir=f"./experiments/{method}"
        )
    elif method in ["easy_to_hard", "hard_to_easy"]:
        trainer = FixedCurriculumTrainer(
            curriculum_type=method,
            batch_size=kwargs.get('batch_size', 4),
            gradient_accumulation_steps=kwargs.get('gradient_accumulation_steps', 4),
            learning_rate=kwargs.get('learning_rate', 5e-5),
            save_dir=f"./experiments/{method}"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Training loop
    eval_results = []
    
    for epoch in range(n_epochs):
        print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")
        
        # Train
        trainer.train_epoch(steps_per_epoch)
        
        # Evaluate periodically
        if (epoch + 1) % eval_interval == 0:
            print("\nEvaluating...")
            eval_results_dict = trainer.evaluate(n_samples=50, log_solutions=True)
            eval_results.append({
                'epoch': epoch + 1,
                'step': trainer.step,
                **eval_results_dict
            })
            
            print(f"Overall accuracy: {eval_results_dict['overall_accuracy']:.3f}")
            print(f"Teacher-Student agreement: {eval_results_dict.get('teacher_student_agreement', 0.0):.3f}")
            
            # Print accuracy by level
            level_accs = {k: v for k, v in eval_results_dict.items() if k.startswith('accuracy_level_')}
            if level_accs:
                print("Accuracy by level:")
                for level_key, acc in sorted(level_accs.items()):
                    level = level_key.split('_')[-1]
                    print(f"  Level {level}: {acc:.3f}")
    
    # Save results
    save_dir = trainer.save_results()
    
    # Save evaluation results
    trainer.saver.save_evaluation_results(eval_results)
    
    print(f"\nExperiment completed! Results saved to: {save_dir}")
    print(f"Final accuracy: {eval_results[-1]['overall_accuracy']:.3f}")
    
    return trainer, eval_results

def main():
    parser = argparse.ArgumentParser(description="Run a single distillation experiment")
    parser.add_argument("--method", type=str, default="rl", 
                       choices=["rl", "traditional", "easy_to_hard", "hard_to_easy"],
                       help="Training method")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=100, help="Steps per epoch")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Curriculum learning rate (RL only)")
    parser.add_argument("--tau", type=float, default=1.0, help="Curriculum temperature (RL only)")
    
    args = parser.parse_args()
    
    # Convert args to kwargs
    kwargs = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'alpha': args.alpha,
        'tau': args.tau,
        'gradient_accumulation_steps': 4,  # Fixed for now
    }
    
    try:
        trainer, results = run_experiment(
            method=args.method,
            n_epochs=args.n_epochs,
            steps_per_epoch=args.steps_per_epoch,
            eval_interval=args.eval_interval,
            **kwargs
        )
        
        print("\n✅ Experiment completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()