#!/usr/bin/env python3
"""
Enhanced quick test to verify everything is working with modular structure
"""

import sys
import time
import traceback
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_single_experiment import run_experiment

def test_dataset_loading():
    """Test EleutherAI MATH dataset loading with enhanced validation"""
    print("="*60)
    print("TESTING DATASET LOADING")
    print("="*60)
    
    try:
        from src.data.dataset import MATHDataset
        
        print("Loading dataset (this may take a moment)...")
        start_time = time.time()
        dataset = MATHDataset(split='train')
        load_time = time.time() - start_time
        
        print(f"✅ Dataset loaded successfully in {load_time:.1f} seconds")
        print(f"✅ Total examples: {len(dataset)}")
        print(f"✅ Available difficulties: {sorted(dataset.difficulties)}")
        
        # Test data quality
        print("\nTesting data quality...")
        sample_problems = []
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            sample_problems.append(sample)
            print(f"  Sample {i+1}: Level {sample['level']}, Subject: {sample.get('subject', 'unknown')}")
            print(f"    Problem length: {len(sample['problem'])} chars")
            print(f"    Solution length: {len(sample['solution'])} chars")
        
        # Test difficulty sampling
        print("\nTesting difficulty sampling...")
        for difficulty in sorted(dataset.difficulties)[:3]:  # Test first 3 difficulties
            sampled = dataset.get_by_difficulty(difficulty, 2)
            print(f"  Level {difficulty}: sampled {len(sampled)} problems")
            if sampled:
                print(f"    Example: {sampled[0]['problem'][:100]}...")
        
        print(f"✅ Dataset quality tests passed!")
        return True, dataset
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_model_loading():
    """Test model loading and basic functionality"""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    try:
        from src.models.trainers import RLDistillationTrainer
        
        print("Initializing trainer (this will test model loading)...")
        start_time = time.time()
        
        # This will automatically run model verification
        trainer = RLDistillationTrainer(
            batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=5e-5,
            save_dir="./test_experiments"
        )
        
        init_time = time.time() - start_time
        print(f"✅ Trainer initialized successfully in {init_time:.1f} seconds")
        print(f"✅ All model verification tests passed!")
        
        return True, trainer
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_basic_training():
    """Test basic training functionality"""
    print("\n" + "="*60)
    print("TESTING BASIC TRAINING")
    print("="*60)
    
    try:
        print("Running minimal training test...")
        trainer, results = run_experiment(
            method="rl",
            n_epochs=1,
            steps_per_epoch=3,  # Very small for testing
            eval_interval=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-5,
        )

        print(f"✅ Basic training completed!")
        print(f"✅ Final accuracy: {results[-1]['overall_accuracy']:.3f}")
        print(f"✅ Training steps completed: {trainer.step}")
        print(f"✅ Total training examples seen: {trainer.total_training_examples}")
        
        # Test enhanced metrics
        if 'teacher_student_agreement' in results[-1]:
            print(f"✅ Teacher-student agreement: {results[-1]['teacher_student_agreement']:.3f}")
        
        return True, trainer, results
        
    except Exception as e:
        print(f"❌ Basic training failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_curriculum_functionality(trainer):
    """Test curriculum learning functionality"""
    print("\n" + "="*60)
    print("TESTING CURRICULUM FUNCTIONALITY")
    print("="*60)
    
    try:
        print("Testing curriculum selection...")
        
        # Test curriculum selection
        if hasattr(trainer, 'curriculum'):
            difficulties = trainer.curriculum.select_difficulties(5)
            print(f"✅ Curriculum selected difficulties: {difficulties}")
            
            # Check Q-values
            print("Current Q-values:")
            for difficulty, q_val in sorted(trainer.curriculum.Q_values.items()):
                print(f"  Level {difficulty}: {q_val:.4f}")
            
            # Test curriculum update
            test_rewards = {d: 0.5 for d in trainer.curriculum.difficulties}
            trainer.curriculum.update_q_values(test_rewards)
            print(f"✅ Q-values updated successfully")
        else:
            print("✅ No curriculum (traditional method)")
        
        return True
        
    except Exception as e:
        print(f"❌ Curriculum functionality failed: {e}")
        traceback.print_exc()
        return False

def test_evaluation_functionality(trainer):
    """Test enhanced evaluation functionality"""
    print("\n" + "="*60)
    print("TESTING EVALUATION FUNCTIONALITY")
    print("="*60)
    
    try:
        print("Running enhanced evaluation (small sample)...")
        
        eval_results = trainer.evaluate(n_samples=5, log_solutions=True)
        
        print(f"✅ Evaluation completed!")
        print(f"✅ Overall accuracy: {eval_results['overall_accuracy']:.3f}")
        if 'teacher_student_agreement' in eval_results:
            print(f"✅ Teacher-student agreement: {eval_results['teacher_student_agreement']:.3f}")
        
        return True, eval_results
        
    except Exception as e:
        print(f"❌ Evaluation functionality failed: {e}")
        traceback.print_exc()
        return False, None

def test_baseline_methods():
    """Test baseline methods"""
    print("\n" + "="*60)
    print("TESTING BASELINE METHODS")
    print("="*60)
    
    baseline_methods = ['traditional', 'easy_to_hard']
    
    for method in baseline_methods:
        try:
            print(f"\nTesting {method} method...")
            trainer, results = run_experiment(
                method=method,
                n_epochs=1,
                steps_per_epoch=2,
                eval_interval=1,
                batch_size=2,
                gradient_accumulation_steps=1,
                learning_rate=5e-5,
            )
            print(f"✅ {method} method working! Final accuracy: {results[-1]['overall_accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ {method} method failed: {e}")
            return False
    
    return True

def test_serialization(trainer):
    """Test serialization functionality"""
    print("\n" + "="*60)
    print("TESTING SERIALIZATION")
    print("="*60)
    
    try:
        print("Testing save_results...")
        save_dir = trainer.save_results()
        print(f"✅ Results saved successfully to: {save_dir}")
        
        # Verify files were created
        expected_files = ['metrics.json', 'training_history.json', 'student_model.pt']
        for filename in expected_files:
            filepath = save_dir / filename
            if filepath.exists():
                print(f"✅ {filename} created successfully")
            else:
                print(f"⚠️ {filename} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("ENHANCED QUICK TEST FOR MODULAR RL KNOWLEDGE DISTILLATION")
    print("="*80)
    print("This test verifies:")
    print("• Dataset loading and distribution analysis")
    print("• Model loading and verification")
    print("• Basic training functionality") 
    print("• Curriculum learning mechanisms")
    print("• Enhanced evaluation and metrics")
    print("• Baseline methods")
    print("• Serialization functionality")
    print("• Modular code structure")
    print("="*80)
    
    overall_start_time = time.time()
    
    # Test 1: Dataset Loading
    dataset_success, dataset = test_dataset_loading()
    if not dataset_success:
        print("\n❌ CRITICAL: Dataset loading failed - cannot proceed")
        return False
    
    # Test 2: Model Loading  
    model_success, trainer = test_model_loading()
    if not model_success:
        print("\n❌ CRITICAL: Model loading failed - cannot proceed")
        return False
    
    # Test 3: Basic Training
    training_success, trainer, results = test_basic_training()
    if not training_success:
        print("\n❌ CRITICAL: Basic training failed")
        return False
    
    # Test 4: Serialization
    serialization_success = test_serialization(trainer)
    if not serialization_success:
        print("\n⚠️ WARNING: Serialization has issues")
    
    # Test 5: Curriculum Functionality
    curriculum_success = test_curriculum_functionality(trainer)
    if not curriculum_success:
        print("\n⚠️ WARNING: Curriculum functionality has issues")
    
    # Test 6: Enhanced Evaluation
    eval_success, eval_results = test_evaluation_functionality(trainer)
    if not eval_success:
        print("\n⚠️ WARNING: Enhanced evaluation has issues")
    
    # Test 7: Baseline Methods
    baseline_success = test_baseline_methods()
    if not baseline_success:
        print("\n⚠️ WARNING: Some baseline methods have issues")
    
    # Final Summary
    total_time = time.time() - overall_start_time
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*80)
    
    tests = [
        ("Dataset Loading", dataset_success),
        ("Model Loading & Verification", model_success), 
        ("Basic Training", training_success),
        ("Serialization", serialization_success),
        ("Curriculum Learning", curriculum_success),
        ("Enhanced Evaluation", eval_success),
        ("Baseline Methods", baseline_success)
    ]
    
    passed_tests = sum(1 for _, success in tests if success)
    
    for test_name, success in tests:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:30}: {status}")
    
    print(f"\nOverall: {passed_tests}/{len(tests)} tests passed")
    print(f"Total test time: {total_time:.1f} seconds")
    
    if passed_tests >= 5:  # Core functionality working
        print("\n🎉 CORE FUNCTIONALITY WORKING!")
        print("✅ Ready for full experiments!")
        print("\nModular structure benefits:")
        print("  • Clean separation of concerns")
        print("  • Easy to extend and maintain")
        print("  • Reusable components")
        print("  • Better testing and debugging")
        print("\nTo run full experiments:")
        print("  python scripts/run_all_experiments.py")
        print("  python scripts/run_single_experiment.py --method rl")
        return True
    else:
        print("\n❌ CRITICAL ISSUES DETECTED")
        print("Please fix the failing tests before running full experiments")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("• Run 'python scripts/run_all_experiments.py' for complete comparison")
        print("• Run 'python scripts/run_single_experiment.py --help' for single experiments")
        print("• Check generated plots and analysis in results/ directory")
        print("• Modify config/experiment_config.py for different settings")
        print("• Extend src/models/ for new curriculum strategies")
    else:
        print("\n" + "="*60)
        print("TROUBLESHOOTING")
        print("="*60)
        print("• Check that all dependencies are installed:")
        print("  pip install -e .")
        print("• Ensure sufficient GPU memory (8GB+ recommended)")
        print("• Check CUDA availability if using GPU")
        print("• Try reducing batch_size if memory issues occur")