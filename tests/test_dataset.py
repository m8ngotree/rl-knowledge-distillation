"""
Test dataset functionality
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.data.dataset import MATHDataset


class TestMATHDataset:
    """Test MATH dataset functionality"""
    
    @patch('src.data.dataset.load_dataset')
    def test_dataset_initialization(self, mock_load_dataset):
        """Test dataset initialization with mocked data"""
        # Mock dataset data - return different data for each subject config
        def mock_load_dataset_side_effect(dataset_name, config, split):
            if config == 'algebra':
                return [{'problem': 'What is 2+2?', 'solution': '4', 'level': 'Level 1', 'type': 'Arithmetic'}]
            elif config == 'geometry':
                return [{'problem': 'What is 3+3?', 'solution': '6', 'level': 'Level 2', 'type': 'Arithmetic'}]
            else:
                return []  # Return empty for other configs
        
        mock_load_dataset.side_effect = mock_load_dataset_side_effect
        
        # Create dataset
        dataset = MATHDataset(split='train')
        
        assert len(dataset) == 2  # Only algebra and geometry have data
        assert 1 in dataset.difficulties
        assert 2 in dataset.difficulties
    
    def test_level_parsing(self):
        """Test level string parsing"""
        from src.data.dataset import MATHDataset
        
        dataset = MATHDataset.__new__(MATHDataset)  # Create without __init__
        
        # Test valid level strings
        assert dataset._parse_level('Level 1') == 1
        assert dataset._parse_level('Level 5') == 5
        assert dataset._parse_level('1') == 1
        assert dataset._parse_level('3') == 3
        
        # Test malformed/invalid levels
        assert dataset._parse_level('Level ?') == 3  # Default
        assert dataset._parse_level('Invalid') == 3  # Default
        assert dataset._parse_level('') == 3  # Default
        assert dataset._parse_level(None) == 3  # Default
    
    @patch('src.data.dataset.load_dataset')
    def test_get_by_difficulty(self, mock_load_dataset):
        """Test sampling by difficulty level"""
        # Mock dataset with multiple difficulty levels
        def mock_load_dataset_side_effect(dataset_name, config, split):
            if config == 'algebra':
                mock_data = []
                for i in range(10):  # 10 problems per level
                    for level in [1, 2, 3]:
                        mock_data.append({
                            'problem': f'Problem {i} level {level}',
                            'solution': f'Solution {i}',
                            'level': f'Level {level}',
                            'type': 'Test'
                        })
                return mock_data
            else:
                return []  # Return empty for other configs
        
        mock_load_dataset.side_effect = mock_load_dataset_side_effect
        dataset = MATHDataset(split='train')
        
        # Test sampling from specific difficulty
        level_1_problems = dataset.get_by_difficulty(1, 5)
        assert len(level_1_problems) == 5
        assert all(p['level'] == 1 for p in level_1_problems)
        
        # Test sampling more than available
        level_2_problems = dataset.get_by_difficulty(2, 20)
        assert len(level_2_problems) == 10  # Only 10 available
        
        # Test sampling from non-existent difficulty
        empty_problems = dataset.get_by_difficulty(99, 5)
        assert len(empty_problems) == 0
    
    @patch('src.data.dataset.load_dataset')
    def test_batch_sampling_methods(self, mock_load_dataset):
        """Test efficient batch sampling methods"""
        # Create mock data with known distribution
        def mock_load_dataset_side_effect(dataset_name, config, split):
            if config == 'algebra':
                mock_data = []
                for level in [1, 2, 3]:
                    for i in range(5):  # 5 problems per level
                        mock_data.append({
                            'problem': f'Problem {i} level {level}',
                            'solution': f'Solution {i}',
                            'level': f'Level {level}',
                            'type': 'Test'
                        })
                return mock_data
            else:
                return []  # Return empty for other configs
        
        mock_load_dataset.side_effect = mock_load_dataset_side_effect
        dataset = MATHDataset(split='train')
        
        # Test batch sampling by difficulties
        difficulties = [1, 2, 3]
        batch_problems = dataset.get_batch_by_difficulties(difficulties, n_per_difficulty=2)
        assert len(batch_problems) == 6  # 2 per difficulty * 3 difficulties
        
        # Count problems by level
        level_counts = {}
        for p in batch_problems:
            level = p['level']
            level_counts[level] = level_counts.get(level, 0) + 1
        
        assert level_counts[1] == 2
        assert level_counts[2] == 2
        assert level_counts[3] == 2
        
        # Test balanced batch
        balanced_batch = dataset.get_balanced_batch(9)  # Should get 3 from each level
        assert len(balanced_batch) == 9
        
        # Test random batch
        random_batch = dataset.get_random_batch(6)
        assert len(random_batch) == 6
        
        # Test random batch larger than dataset
        large_batch = dataset.get_random_batch(100)
        assert len(large_batch) == len(dataset)  # Should return full dataset
    
    @patch('src.data.dataset.load_dataset')
    def test_dataset_getitem(self, mock_load_dataset):
        """Test dataset item access"""
        def mock_load_dataset_side_effect(dataset_name, config, split):
            if config == 'algebra':
                return [{
                    'problem': 'Test problem',
                    'solution': 'Test solution', 
                    'level': 'Level 1',
                    'type': 'Arithmetic'
                }]
            else:
                return []  # Return empty for other configs
        
        mock_load_dataset.side_effect = mock_load_dataset_side_effect
        dataset = MATHDataset(split='train')
        
        item = dataset[0]
        assert item['problem'] == 'Test problem'
        assert item['solution'] == 'Test solution' 
        assert item['level'] == 1  # Parsed to int
        assert item['type'] == 'Arithmetic'
        assert item['subject'] == 'algebra'  # Subject added by dataset loader
    
    @patch('src.data.dataset.load_dataset')
    def test_empty_difficulties_handling(self, mock_load_dataset):
        """Test handling of empty difficulty lists"""
        def mock_load_dataset_side_effect(dataset_name, config, split):
            if config == 'algebra':
                return [{'problem': 'test', 'solution': 'test', 'level': 'Level 1', 'type': 'test'}]
            else:
                return []  # Return empty for other configs
        
        mock_load_dataset.side_effect = mock_load_dataset_side_effect
        dataset = MATHDataset(split='train')
        
        # Test empty difficulties list
        empty_batch = dataset.get_balanced_batch(5, difficulties=[])
        assert len(empty_batch) == 0
        
        # Test non-existent difficulties
        empty_batch2 = dataset.get_batch_by_difficulties([99, 100])
        assert len(empty_batch2) == 0