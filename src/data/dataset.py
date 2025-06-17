"""
Dataset classes for MATH dataset and curriculum learning
"""

import numpy as np
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Any
from ..utils.logging_config import get_logger

class MATHDataset(Dataset):
    """EleutherAI MATH dataset with difficulty categorization and enhanced distribution analysis"""
    
    def __init__(self, split: str = 'train', tokenizer: Optional[Any] = None, max_length: int = 1024) -> None:
        # Input validation
        if not isinstance(split, str):
            raise TypeError(f"split must be a string, got {type(split)}")
        if split not in ['train', 'test']:
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")
        if max_length <= 0:
            raise ValueError(f"max_length must be positive, got {max_length}")
        
        # Initialize logger
        self.logger = get_logger(__name__)
        self.logger.info(f"Loading EleutherAI MATH dataset ({split})...")
        
        # Load all subject configs from EleutherAI
        configs = ['algebra', 'counting_and_probability', 'geometry',
                  'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        
        self.data = []
        subject_data_counts = {}  # Track per-subject loading
        
        for config in configs:
            try:
                dataset = load_dataset('EleutherAI/hendrycks_math', config, split=split)
                config_data = list(dataset)
                
                # Add subject type to each item
                for item in config_data:
                    item['subject'] = config
                
                self.data.extend(config_data)
                subject_data_counts[config] = len(config_data)
                self.logger.info(f"  ✅ {config}: {len(config_data)} examples")
            except Exception as e:
                self.logger.warning(f"  ⚠️ Failed to load {config}: {e}")
                subject_data_counts[config] = 0
        
        self.logger.info(f"Total loaded: {len(self.data)} examples")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Organize by difficulty level (1-5)
        self.difficulty_indices = defaultdict(list)
        
        # Enhanced distribution analysis
        subject_counts = defaultdict(int)
        level_counts = defaultdict(int)
        subject_level_counts = defaultdict(lambda: defaultdict(int))
        malformed_levels = []
        
        for idx, item in enumerate(self.data):
            level_str = item.get('level', 'Level 1')
            level = self._parse_level(level_str)
            subject = item.get('subject', 'unknown')
            
            self.difficulty_indices[level].append(idx)
            
            # Track distributions
            subject_counts[subject] += 1
            level_counts[level] += 1
            subject_level_counts[subject][level] += 1
            
            # Track malformed levels for debugging
            if '?' in str(level_str):
                malformed_levels.append((idx, level_str, subject))
        
        self.difficulties = list(self.difficulty_indices.keys())
        
        # ENHANCED DISTRIBUTION ANALYSIS
        self.logger.info("="*60)
        self.logger.info("DATASET DISTRIBUTION ANALYSIS")
        self.logger.info("="*60)
        self.logger.info(f"Total examples: {len(self.data)}")
        self.logger.info(f"Available difficulties: {sorted(self.difficulties)}")
        
        self.logger.info("Distribution by subject:")
        for subject, count in sorted(subject_counts.items()):
            percentage = count/len(self.data)*100
            self.logger.info(f"  {subject.ljust(25)}: {count:>4} ({percentage:>5.1f}%)")
        
        self.logger.info("Distribution by difficulty level:")
        for level, count in sorted(level_counts.items()):
            percentage = count/len(self.data)*100
            self.logger.info(f"  Level {level}: {count:>4} ({percentage:>5.1f}%)")
        
        self.logger.info("="*60)
    
    def _parse_level(self, level_str: Union[str, None]) -> int:
        """Parse level string to integer, default malformed to level 3"""
        if not level_str or '?' in level_str:
            return 3  # Default for malformed values
        
        if 'Level' in level_str:
            try:
                return int(level_str.split()[-1])
            except (ValueError, IndexError):
                return 3
        
        try:
            return int(level_str)
        except (ValueError, TypeError):
            return 3
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Input validation - accept both int and numpy integer types
        import numbers
        if not isinstance(idx, numbers.Integral):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        idx = int(idx)  # Convert numpy int to Python int
        if not 0 <= idx < len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
            
        item = self.data[idx]
        level_str = item.get('level', 'Level 1')
        level = self._parse_level(level_str)
        
        return {
            'problem': item.get('problem', ''),
            'solution': item.get('solution', ''),
            'level': level,
            'type': item.get('type', 'Mathematics'),
            'subject': item.get('subject', 'unknown')
        }
    
    def get_by_difficulty(self, difficulty: int, n_samples: int) -> List[Dict[str, Any]]:
        """Sample n problems from a specific difficulty level"""
        # Input validation
        if not isinstance(difficulty, int):
            raise TypeError(f"difficulty must be an integer, got {type(difficulty)}")
        if not isinstance(n_samples, int):
            raise TypeError(f"n_samples must be an integer, got {type(n_samples)}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
            
        if difficulty not in self.difficulty_indices:
            return []
            
        indices = np.random.choice(
            self.difficulty_indices[difficulty], 
            min(n_samples, len(self.difficulty_indices[difficulty])),
            replace=False
        )
        return [self[idx] for idx in indices]
    
    def get_batch_by_difficulties(self, difficulties: List[int], n_per_difficulty: int = 1) -> List[Dict[str, Any]]:
        """Efficiently sample problems from multiple difficulty levels in batch"""
        # Input validation
        if not isinstance(difficulties, list):
            raise TypeError(f"difficulties must be a list, got {type(difficulties)}")
        if not all(isinstance(d, int) for d in difficulties):
            raise TypeError("All difficulties must be integers")
        if not isinstance(n_per_difficulty, int):
            raise TypeError(f"n_per_difficulty must be an integer, got {type(n_per_difficulty)}")
        if n_per_difficulty <= 0:
            raise ValueError(f"n_per_difficulty must be positive, got {n_per_difficulty}")
            
        problems = []
        
        # Collect all required indices at once
        all_indices = []
        for difficulty in difficulties:
            if difficulty in self.difficulty_indices:
                available_indices = self.difficulty_indices[difficulty]
                if available_indices:
                    sample_size = min(n_per_difficulty, len(available_indices))
                    sampled_indices = np.random.choice(
                        available_indices, 
                        size=sample_size, 
                        replace=False
                    )
                    all_indices.extend(sampled_indices)
        
        # Batch retrieve all problems
        problems = [self[idx] for idx in all_indices]
        return problems
    
    def get_balanced_batch(self, batch_size: int, difficulties: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Get a balanced batch across all or specified difficulties"""
        # Input validation
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if difficulties is not None:
            if not isinstance(difficulties, list):
                raise TypeError(f"difficulties must be a list or None, got {type(difficulties)}")
            if not all(isinstance(d, int) for d in difficulties):
                raise TypeError("All difficulties must be integers")
                
        if difficulties is None:
            difficulties = self.difficulties
        
        # Calculate samples per difficulty to reach target batch size
        n_difficulties = len(difficulties)
        if n_difficulties == 0:
            return []
        
        base_samples = batch_size // n_difficulties
        extra_samples = batch_size % n_difficulties
        
        problems = []
        for i, difficulty in enumerate(difficulties):
            n_samples = base_samples + (1 if i < extra_samples else 0)
            if n_samples > 0:
                difficulty_problems = self.get_by_difficulty(difficulty, n_samples)
                problems.extend(difficulty_problems)
        
        return problems
    
    def get_random_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a random batch of problems (for traditional distillation)"""
        # Input validation
        if not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size)}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
            
        if batch_size >= len(self.data):
            return [self[i] for i in range(len(self.data))]
        
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        return [self[idx] for idx in indices]