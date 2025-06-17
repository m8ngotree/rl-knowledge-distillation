#!/usr/bin/env python3
"""
Complete experimental pipeline for RL knowledge distillation
UPDATED: Uses new modular structure
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
from collections import defaultdict
import time

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.trainers import (
    RLDistillationTrainer,
    TraditionalDistillationTrainer,
    FixedCurriculumTrainer
)
from src.data.dataset import MATHDataset
from config.experiment_config import BASE_CONFIG, EXPERIMENT_CONFIGS

def test_setup():
    """Test that dataset loading works before running experiments"""
    print("Testing setup...")
    try:
        dataset = MATHDataset(split='train')
        print(f"✅ EleutherAI MATH dataset loaded: {len(dataset)} examples")
        print(f"✅ Available difficulties: {sorted(dataset.difficulties)}")
        return True
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        print("Please run: pip install -U datasets huggingface_hub fsspec")
        return False

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path(f'results/full_experiment_{timestamp}')
results_dir.mkdir(parents=True, exist_ok=True)

def run_single_experiment_enhanced(exp_name, exp_config, results_dir):
    """Run a single experiment with enhanced monitoring and comprehensive analysis"""
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Description: {exp_config['description']}")
    print(f"Config: {exp_config}")
    print(f"{'='*60}\n")
    
    # Merge configurations
    config = {**BASE_CONFIG, **exp_config}
    
    experiment_start_time = time.time()
    
    try:
        # Initialize trainer based on method
        method = config['method']
        trainer_kwargs = {
            'batch_size': config.get('batch_size', 4),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 4),
            'learning_rate': config.get('learning_rate', 5e-5),
            'save_dir': str(results_dir / exp_name)
        }
        
        if method == "rl":
            trainer = RLDistillationTrainer(
                alpha=config.get('alpha', 0.5),
                tau=config.get('tau', 1.0),
                **trainer_kwargs
            )
        elif method == "traditional":
            trainer = TraditionalDistillationTrainer(**trainer_kwargs)
        elif method in ["easy_to_hard", "hard_to_easy"]:
            trainer = FixedCurriculumTrainer(
                curriculum_type=method,
                **trainer_kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Training loop
        eval_results = []
        n_epochs = config.get('n_epochs', 5)
        steps_per_epoch = config.get('steps_per_epoch', 200)
        eval_interval = config.get('eval_interval', 1)
        
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
        
        # Save results
        save_dir = trainer.save_results()
        trainer.saver.save_evaluation_results(eval_results)
        trainer.saver.save_config(config)
        
        # Get comprehensive training summary
        training_summary = trainer.metrics_tracker.get_summary()
        
        experiment_results = {
            'config': config,
            'results': eval_results,
            'training_summary': training_summary,
            'final_accuracy': eval_results[-1]['overall_accuracy'],
            'teacher_student_agreement': eval_results[-1].get('teacher_student_agreement', 0),
            'experiment_duration': time.time() - experiment_start_time
        }
        
        # Add accuracy by level if available
        level_accuracies = {k: v for k, v in eval_results[-1].items() if k.startswith('accuracy_level_')}
        if level_accuracies:
            experiment_results['accuracy_by_level'] = level_accuracies
        
        # Add subject accuracies if available
        subject_accuracies = {k: v for k, v in eval_results[-1].items() if k.startswith('accuracy_') and not k.startswith('accuracy_level_')}
        if subject_accuracies:
            experiment_results['accuracy_by_subject'] = subject_accuracies
        
        print(f"\n✅ Experiment {exp_name} completed!")
        print(f"Duration: {experiment_results['experiment_duration']:.1f} seconds")
        print(f"Final accuracy: {eval_results[-1]['overall_accuracy']:.3f}")
        print(f"Teacher-student agreement: {experiment_results['teacher_student_agreement']:.3f}")
        
        return experiment_results
        
    except Exception as e:
        print(f"❌ Experiment {exp_name} failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_info = {
            'experiment': exp_name,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        with open(results_dir / f'{exp_name}_error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        return None

def run_all_experiments():
    """Run all experiments with enhanced analysis and monitoring"""
    # Test setup first
    if not test_setup():
        print("❌ Setup failed - cannot proceed with experiments")
        return None
    
    all_results = {}
    total_start_time = time.time()
    
    # Show experiment plan
    print("Available experiments:")
    for i, (exp_name, exp_config) in enumerate(EXPERIMENT_CONFIGS.items(), 1):
        print(f"{i}. {exp_name}: {exp_config['description']}")
    
    print(f"\nRunning {len(EXPERIMENT_CONFIGS)} experiments...")
    print(f"Expected total time: ~{len(EXPERIMENT_CONFIGS) * 20:.0f} minutes (rough estimate)")
    
    for i, (exp_name, exp_config) in enumerate(EXPERIMENT_CONFIGS.items(), 1):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i}/{len(EXPERIMENT_CONFIGS)}: {exp_name}")
        print(f"{'='*80}")
        
        exp_start_time = time.time()
        result = run_single_experiment_enhanced(exp_name, exp_config, results_dir)
        
        if result:
            all_results[exp_name] = result
            elapsed = time.time() - exp_start_time
            remaining_experiments = len(EXPERIMENT_CONFIGS) - i
            estimated_remaining = elapsed * remaining_experiments
            
            print(f"\n✅ Experiment {i}/{len(EXPERIMENT_CONFIGS)} completed in {elapsed:.1f} seconds")
            if remaining_experiments > 0:
                print(f"⏱️ Estimated time remaining: {estimated_remaining:.1f} seconds ({estimated_remaining/60:.1f} minutes)")
        else:
            print(f"\n❌ Experiment {i}/{len(EXPERIMENT_CONFIGS)} failed")
    
    total_duration = time.time() - total_start_time
    print(f"\nAll experiments completed in {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    # Save combined results with enhanced metrics
    with open(results_dir / 'all_results_enhanced.json', 'w') as f:
        # Convert numpy/torch types for JSON serialization
        from src.utils.serialization import convert_for_json
        json.dump(convert_for_json(all_results), f, indent=2)
    
    return all_results

def analyze_results(all_results):
    """Analyze and compare results with enhanced metrics"""
    if not all_results:
        print("No results to analyze")
        return None
        
    analysis = {}
    
    # Compare all methods
    method_accuracies = {
        name: data['final_accuracy'] 
        for name, data in all_results.items()
    }
    
    # Calculate improvements relative to traditional baseline
    if 'traditional' in all_results:
        traditional_acc = all_results['traditional']['final_accuracy']
        improvements = {}
        
        for method, acc in method_accuracies.items():
            if method != 'traditional':
                improvement = (acc - traditional_acc) / traditional_acc * 100
                improvements[method] = improvement
        
        analysis['improvements_vs_traditional'] = improvements
    
    analysis['accuracy_comparison'] = method_accuracies
    
    # Enhanced metrics comparison
    enhanced_metrics = ['teacher_student_agreement']
    for metric in enhanced_metrics:
        method_values = {
            name: data.get(metric, 0) 
            for name, data in all_results.items()
        }
        analysis[f'{metric}_comparison'] = method_values
    
    # Difficulty level analysis
    difficulty_analysis = {}
    all_levels = set()
    for data in all_results.values():
        if 'accuracy_by_level' in data:
            all_levels.update([int(k.split('_')[-1]) for k in data['accuracy_by_level'].keys()])
    
    for level in sorted(all_levels):
        level_accuracies = {}
        for method, data in all_results.items():
            if 'accuracy_by_level' in data:
                level_key = f'accuracy_level_{level}'
                if level_key in data['accuracy_by_level']:
                    level_accuracies[method] = data['accuracy_by_level'][level_key]
        if level_accuracies:
            difficulty_analysis[f'level_{level}'] = level_accuracies
    
    analysis['difficulty_level_analysis'] = difficulty_analysis
    
    # Save analysis
    with open(results_dir / 'analysis_enhanced.json', 'w') as f:
        from src.utils.serialization import convert_for_json
        json.dump(convert_for_json(analysis), f, indent=2)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print("\nFINAL ACCURACY COMPARISON:")
    for method, acc in sorted(method_accuracies.items(), key=lambda x: x[1], reverse=True):
        duration = all_results[method].get('experiment_duration', 0)
        print(f"{method:20}: {acc:.3f} (duration: {duration:.1f}s)")
    
    if 'improvements_vs_traditional' in analysis:
        print("\nIMPROVEMENTS vs TRADITIONAL:")
        for method, improvement in sorted(analysis['improvements_vs_traditional'].items(), key=lambda x: x[1], reverse=True):
            print(f"{method:20}: {improvement:+.1f}%")
    
    print("\nTEACHER-STUDENT AGREEMENT:")
    agreement_comparison = analysis.get('teacher_student_agreement_comparison', {})
    for method, agreement in sorted(agreement_comparison.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:20}: {agreement:.3f}")
    
    if 'difficulty_level_analysis' in analysis:
        print("\nACCURACY BY DIFFICULTY LEVEL:")
        for level, accuracies in sorted(analysis['difficulty_level_analysis'].items()):
            level_num = level.split('_')[1]
            print(f"  Level {level_num}:")
            for method, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
                print(f"    {method:18}: {acc:.3f}")
    
    return analysis

def create_comprehensive_comparison_plots(all_results):
    """Create comprehensive comparison plots with all enhanced metrics"""
    if not all_results:
        print("No results to plot")
        return
        
    plt.style.use('default')
    
    # Super comprehensive comparison figure
    fig = plt.figure(figsize=(20, 16))
    
    methods = list(all_results.keys())
    descriptions = [all_results[m]['config']['description'] for m in methods]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Final accuracy comparison
    plt.subplot(3, 3, 1)
    accuracies = [all_results[m]['final_accuracy'] for m in methods]
    bars = plt.bar(range(len(methods)), accuracies, color=colors[:len(methods)])
    plt.xticks(range(len(methods)), [d.split(':')[0] for d in descriptions], rotation=45, ha='right')
    plt.ylabel('Final Accuracy')
    plt.title('Final Accuracy Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Teacher-Student Agreement
    plt.subplot(3, 3, 2)
    agreements = [all_results[m]['teacher_student_agreement'] for m in methods]
    plt.bar(range(len(methods)), agreements, color=colors[:len(methods)])
    plt.xticks(range(len(methods)), [d.split(':')[0] for d in descriptions], rotation=45, ha='right')
    plt.ylabel('Teacher-Student Agreement')
    plt.title('Teacher-Student Agreement')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 3. Training Duration Comparison
    plt.subplot(3, 3, 3)
    durations = [all_results[m].get('experiment_duration', 0) / 60 for m in methods]  # Convert to minutes
    plt.bar(range(len(methods)), durations, color=colors[:len(methods)])
    plt.xticks(range(len(methods)), [d.split(':')[0] for d in descriptions], rotation=45, ha='right')
    plt.ylabel('Training Duration (minutes)')
    plt.title('Training Time Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy by Difficulty Level
    plt.subplot(3, 3, 4)
    level_data = defaultdict(list)
    level_methods = []
    
    for method in methods:
        if 'accuracy_by_level' in all_results[method]:
            level_methods.append(method)
            acc_by_level = all_results[method]['accuracy_by_level']
            for level_key, acc in acc_by_level.items():
                level = int(level_key.split('_')[-1])
                level_data[level].append(acc)
    
    if level_data:
        levels = sorted(level_data.keys())
        x = np.arange(len(levels))
        width = 0.8 / len(level_methods)
        
        for i, method in enumerate(level_methods):
            acc_by_level = all_results[method]['accuracy_by_level']
            accs = [acc_by_level.get(f'accuracy_level_{level}', 0) for level in levels]
            plt.bar(x + i*width, accs, width, label=descriptions[methods.index(method)].split(':')[0], 
                   color=colors[methods.index(method)])
        
        plt.xlabel('Difficulty Level')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Difficulty Level')
        plt.xticks(x + width * (len(level_methods)-1) / 2, [f'L{l}' for l in levels])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    # 5. Improvement over Traditional
    plt.subplot(3, 3, 5)
    if 'traditional' in all_results:
        traditional_acc = all_results['traditional']['final_accuracy']
        improvements = []
        method_labels = []
        
        for method in methods:
            if method != 'traditional':
                acc = all_results[method]['final_accuracy']
                improvement = (acc - traditional_acc) / traditional_acc * 100
                improvements.append(improvement)
                method_labels.append(descriptions[methods.index(method)].split(':')[0])
        
        colors_subset = [colors[methods.index(m)] for m in methods if m != 'traditional']
        bars = plt.bar(range(len(improvements)), improvements, 
                      color=colors_subset)
        
        plt.xticks(range(len(improvements)), method_labels, rotation=45, ha='right')
        plt.ylabel('Improvement (%)')
        plt.title('Improvement over Traditional')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{improvement:+.1f}%', ha='center', va=va, fontweight='bold')
    
    plt.suptitle('Comprehensive Knowledge Distillation Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive comparison plots generated!")

def generate_experiment_report(all_results, analysis):
    """Generate a comprehensive experiment report"""
    
    report_lines = []
    report_lines.append("# RL Knowledge Distillation Experiment Report")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    report_lines.append("## Executive Summary")
    if len(all_results) >= 2:
        best_method = max(all_results.keys(), key=lambda x: all_results[x]['final_accuracy'])
        best_acc = all_results[best_method]['final_accuracy']
        report_lines.append(f"- **Best performing method**: {best_method} ({best_acc:.3f} accuracy)")
        
        if 'traditional' in all_results:
            trad_acc = all_results['traditional']['final_accuracy']
            improvement = (best_acc - trad_acc) / trad_acc * 100
            report_lines.append(f"- **Best improvement over traditional**: {improvement:+.1f}%")
    
    total_experiments = len(all_results)
    total_duration = sum(result.get('experiment_duration', 0) for result in all_results.values())
    report_lines.append(f"- **Total experiments**: {total_experiments}")
    report_lines.append(f"- **Total experiment time**: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    report_lines.append("")
    
    report_lines.append("## Detailed Results")
    report_lines.append("")
    
    for method, result in all_results.items():
        report_lines.append(f"### {method}")
        report_lines.append(f"- **Description**: {result['config']['description']}")
        report_lines.append(f"- **Final Accuracy**: {result['final_accuracy']:.3f}")
        report_lines.append(f"- **Teacher-Student Agreement**: {result['teacher_student_agreement']:.3f}")
        report_lines.append(f"- **Training Duration**: {result.get('experiment_duration', 0):.1f} seconds")
        
        # Accuracy by level
        if 'accuracy_by_level' in result:
            report_lines.append("- **Accuracy by Level**:")
            for level_key, acc in sorted(result['accuracy_by_level'].items()):
                level = level_key.split('_')[-1]
                report_lines.append(f"  - Level {level}: {acc:.3f}")
        
        report_lines.append("")
    
    report_lines.append("## Statistical Analysis")
    if analysis and 'improvements_vs_traditional' in analysis:
        report_lines.append("### Improvements over Traditional Baseline")
        for method, improvement in sorted(analysis['improvements_vs_traditional'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- **{method}**: {improvement:+.1f}%")
    
    report_lines.append("")
    report_lines.append("## Generated Files")
    report_lines.append("- `all_results_enhanced.json`: Complete experimental results")
    report_lines.append("- `analysis_enhanced.json`: Statistical analysis and comparisons")
    report_lines.append("- `comprehensive_comparison.png`: Main comparison plot")
    report_lines.append("- Individual experiment directories with detailed plots and logs")
    
    # Save report
    with open(results_dir / 'experiment_report.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Experiment report generated: {results_dir / 'experiment_report.md'}")

if __name__ == "__main__":
    print("Starting comprehensive experimental comparison...")
    print("Dataset: EleutherAI/hendrycks_math")
    print("Using modular structure with new trainers")
    print()
    
    # Show which experiments will run
    print("Experiments to run:")
    for exp_name, exp_config in EXPERIMENT_CONFIGS.items():
        print(f"• {exp_name}: {exp_config['description']}")
    print()
    
    if len(EXPERIMENT_CONFIGS) == 1:
        print("Running single experiment (others commented out)")
    else:
        print(f"Running {len(EXPERIMENT_CONFIGS)} experiments in sequence")
    print()
    
    # Run all experiments
    all_results = run_all_experiments()
    
    if all_results:
        print(f"\n{'='*80}")
        print("ANALYSIS AND VISUALIZATION")
        print(f"{'='*80}")
        
        # Analyze results
        print("Performing comprehensive analysis...")
        analysis = analyze_results(all_results)
        
        # Create plots
        print("Creating comparison plots...")
        create_comprehensive_comparison_plots(all_results)
        
        # Generate report
        print("Generating experiment report...")
        generate_experiment_report(all_results, analysis)
        
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        print(f"\nAll results saved to: {results_dir}")
        print("\nGenerated files:")
        print("• all_results_enhanced.json: Complete experimental results")
        print("• analysis_enhanced.json: Statistical analysis and comparisons") 
        print("• experiment_report.md: Comprehensive markdown report")
        print("• comprehensive_comparison.png: Main comparison visualization")
        print("• Individual experiment directories with detailed plots and logs")
        
        # Final summary highlighting key findings
        print(f"\n{'='*60}")
        print("KEY FINDINGS")
        print(f"{'='*60}")
        
        if len(all_results) >= 2:
            # Best method
            best_method = max(all_results.keys(), key=lambda x: all_results[x]['final_accuracy'])
            best_acc = all_results[best_method]['final_accuracy']
            best_desc = all_results[best_method]['config']['description']
            
            print(f"\n🏆 Best performing method:")
            print(f"   {best_method}: {best_desc}")
            print(f"   Final accuracy: {best_acc:.3f}")
            
            # Performance ranking
            print(f"\n📊 Performance ranking:")
            sorted_methods = sorted(all_results.items(), key=lambda x: x[1]['final_accuracy'], reverse=True)
            for i, (method, result) in enumerate(sorted_methods, 1):
                duration = result.get('experiment_duration', 0)
                print(f"   {i}. {method}: {result['final_accuracy']:.3f} ({duration:.0f}s)")
            
            # Key improvements
            if 'traditional' in all_results and analysis and 'improvements_vs_traditional' in analysis:
                print(f"\n🚀 Improvements over traditional distillation:")
                improvements = analysis['improvements_vs_traditional']
                for method, improvement in sorted(improvements.items(), key=lambda x: x[1], reverse=True):
                    print(f"   {method}: {improvement:+.1f}%")
        
        print(f"\n{'='*60}")
        print("✅ All experiments completed successfully!")
        print("Open the generated plots and report for detailed analysis!")
        print(f"{'='*60}")
        
    else:
        print("❌ No experiments completed successfully")
        print("Check the error logs and ensure all dependencies are installed")
        print("Try running: pip install -U datasets huggingface_hub fsspec torch transformers")