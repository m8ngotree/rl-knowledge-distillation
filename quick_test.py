"""
Quick test to verify everything is working
"""
from rl_distillation import run_experiment

# Quick test with minimal settings
print("Running quick test...")
trainer, results = run_experiment(
    method="rl",
    n_epochs=1,
    steps_per_epoch=10,
    eval_interval=1,
    batch_size=2
)

print(f"Test completed! Final accuracy: {results[-1]['overall_accuracy']:.3f}")
print("If you see this, the setup is working correctly!")