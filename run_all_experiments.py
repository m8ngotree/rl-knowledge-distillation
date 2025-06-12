"""
Complete experimental pipeline for RL knowledge distillation
"""
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
from rl_distillation import (
    RLKnowledgeDistillation, 
    TraditionalDistillation,
    run_experiment
)

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path(f'results/full_experiment_{timestamp}')
results_dir.mkdir(parents=True, exist_ok=True)

# Experimental configuration
BASE_CONFIG = {
    'teacher_name': 'Qwen/Qwen2.5-Math-7B-Instruct',
    'student_name': 'Qwen/Qwen2.5-1.5B',
    'n_epochs': 5,
    'steps_per_epoch': 200,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'eval_interval': 1
}

# Different experimental conditions
EXPERIMENTS = {
    'rl_curriculum': {
        'method': 'rl',
        'alpha': 0.5,
        'tau': 1.0,
        'description': 'RL-based curriculum (SEC-inspired)'
    },
    'traditional': {
        'method': 'traditional',
        'description': 'Traditional distillation (uniform sampling)'
    },
    'rl_high_explore': {
        'method': 'rl',
        'alpha': 0.5,
        'tau': 2.0,
        'description': 'RL curriculum with high exploration'
    },
    'rl_low_explore': {
        'method': 'rl',
        'alpha': 0.5,
        'tau': 0.5,
        'description': 'RL curriculum with low exploration'
    }
}

def run_all_experiments():
    """Run all experiments and save results"""
    all_results = {}
    
    for exp_name, exp_config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Running experiment: {exp_name}")
        print(f"Description: {exp_config['description']}")
        print(f"{'='*60}\n")
        
        # Merge configurations
        config = {**BASE_CONFIG, **exp_config}
        
        # Run experiment
        trainer, results = run_experiment(**config)
        
        # Save experiment-specific results
        exp_dir = results_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Copy all generated files
        for file in trainer.save_dir.glob('*'):
            file.rename(exp_dir / file.name)
        
        all_results[exp_name] = {
            'config': config,
            'results': results,
            'final_accuracy': results[-1]['overall_accuracy'],
            'accuracy_by_level': results[-1]['accuracy_by_level']
        }
        
        print(f"\nExperiment {exp_name} completed!")
        print(f"Final accuracy: {results[-1]['overall_accuracy']:.3f}")
    
    # Save combined results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def analyze_results(all_results):
    """Analyze and compare results"""
    analysis = {}
    
    # Compare RL vs Traditional
    rl_acc = all_results['rl_curriculum']['final_accuracy']
    trad_acc = all_results['traditional']['final_accuracy']
    
    improvement = (rl_acc - trad_acc) / trad_acc * 100
    
    analysis['accuracy_comparison'] = {
        'rl_curriculum': rl_acc,
        'traditional': trad_acc,
        'improvement_percent': improvement
    }
    
    # Statistical significance (would need multiple runs)
    analysis['statistical_test'] = {
        'note': 'Run multiple seeds for proper statistical testing'
    }
    
    # Save analysis
    with open(results_dir / 'analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"RL Curriculum Accuracy: {rl_acc:.3f}")
    print(f"Traditional Accuracy: {trad_acc:.3f}")
    print(f"Improvement: {improvement:.1f}%")
    
    return analysis

def create_comparison_plots(all_results):
    """Create publication-quality comparison plots"""
    plt.style.use('seaborn-v0_8-paper')
    
    # Plot 1: Final accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(all_results.keys())
    accuracies = [all_results[m]['final_accuracy'] for m in methods]
    
    bars = ax.bar(range(len(methods)), accuracies)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([all_results[m]['config']['description'] for m in methods], 
                       rotation=45, ha='right')
    ax.set_ylabel('Final Accuracy')
    ax.set_title('Knowledge Distillation Method Comparison')
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Accuracy by difficulty level
    fig, ax = plt.subplots(figsize=(12, 6))
    
    difficulties = ['1', '2', '3', '4', '5']
    x = np.arange(len(difficulties))
    width = 0.2
    
    for i, (method, data) in enumerate(all_results.items()):
        if 'accuracy_by_level' in data:
            acc_by_level = data['accuracy_by_level']
            accuracies = [acc_by_level.get(d, 0) for d in difficulties]
            ax.bar(x + i*width, accuracies, width, 
                   label=data['config']['description'])
    
    ax.set_xlabel('Difficulty Level')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Problem Difficulty')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(difficulties)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_by_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run all experiments
    all_results = run_all_experiments()
    
    # Analyze results
    analysis = analyze_results(all_results)
    
    # Create plots
    create_comparison_plots(all_results)
    
    print(f"\nAll results saved to: {results_dir}")