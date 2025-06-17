"""
Setup file for RL Curriculum Distillation package
"""

from setuptools import setup, find_packages

setup(
    name="rl_curriculum_distillation",
    version="0.1.0",
    description="RL-Optimized Curriculum Learning for Knowledge Distillation",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.37.0",
        "datasets>=2.12.0",
        "accelerate>=0.25.0",
        "tokenizers>=0.15.0",
        "huggingface_hub>=0.19.0",
        "fsspec>=2023.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "typing_extensions>=4.0.0",
        "psutil>=5.9.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "ipython>=8.12.0",
            "jupyter>=1.0.0",
            "wandb>=0.15.0",
            "tensorboard>=2.13.0"
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)