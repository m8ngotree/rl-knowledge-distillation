"""
Serialization utilities for saving experiment results
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

def convert_for_json(obj: Any) -> Any:
    """Convert numpy/torch types to native Python types for JSON serialization"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().item() if obj.numel() == 1 else obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, defaultdict):
        return dict(obj)
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj

def save_json(data: Dict, filepath: Path):
    """Save data to JSON file with proper type conversion"""
    clean_data = convert_for_json(data)
    with open(filepath, 'w') as f:
        json.dump(clean_data, f, indent=2)

def load_json(filepath: Path) -> Dict:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

class ResponseStorage:
    """Handles organized storage of teacher-student response pairs"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.responses_file = self.save_dir / 'response_pairs.json'
        self.responses = []
    
    def add_response_pair(self, 
                         problem: Dict,
                         teacher_response: str,
                         student_response: str,
                         teacher_extracted_answer: str,
                         student_extracted_answer: str,
                         answers_equivalent: bool,
                         llm_evaluation_details: Dict = None):
        """Add a teacher-student response pair to storage"""
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'problem': {
                'text': problem.get('problem', ''),
                'solution': problem.get('solution', ''),
                'level': problem.get('level', 'unknown'),
                'subject': problem.get('subject', 'unknown'),
                'problem_id': problem.get('id', problem.get('idx', len(self.responses)))
            },
            'teacher_response': {
                'full_response': teacher_response,
                'extracted_answer': teacher_extracted_answer,
                'response_length': len(teacher_response.split())
            },
            'student_response': {
                'full_response': student_response,
                'extracted_answer': student_extracted_answer,
                'response_length': len(student_response.split())
            },
            'evaluation': {
                'answers_equivalent': answers_equivalent,
                'teacher_correct': None,  # To be filled by individual evaluations
                'student_correct': None,  # To be filled by individual evaluations
                'llm_evaluation_details': llm_evaluation_details or {}
            },
            'metadata': {
                'entry_id': len(self.responses),
                'added_at': datetime.now().isoformat()
            }
        }
        
        self.responses.append(response_entry)
        return response_entry
    
    def add_single_response(self,
                          problem: Dict,
                          response: str,
                          model_type: str,  # 'teacher' or 'student'
                          extracted_answer: str,
                          is_correct: bool):
        """Add a single model response (for individual evaluations)"""
        response_entry = {
            'timestamp': datetime.now().isoformat(),
            'problem': {
                'text': problem.get('problem', ''),
                'solution': problem.get('solution', ''),
                'level': problem.get('level', 'unknown'),
                'subject': problem.get('subject', 'unknown'),
                'problem_id': problem.get('id', problem.get('idx', len(self.responses)))
            },
            'model_type': model_type,
            'response': {
                'full_response': response,
                'extracted_answer': extracted_answer,
                'response_length': len(response.split())
            },
            'evaluation': {
                'is_correct': is_correct,
                'evaluated_at': datetime.now().isoformat()
            },
            'metadata': {
                'entry_id': len(self.responses),
                'added_at': datetime.now().isoformat()
            }
        }
        
        self.responses.append(response_entry)
        return response_entry
    
    def save_responses(self):
        """Save all collected responses to JSON file"""
        save_json(self.responses, self.responses_file)
    
    def load_responses(self):
        """Load existing responses from JSON file"""
        if self.responses_file.exists():
            self.responses = load_json(self.responses_file)
    
    def get_response_pairs(self) -> List[Dict]:
        """Get all teacher-student response pairs"""
        return [r for r in self.responses if 'teacher_response' in r and 'student_response' in r]
    
    def get_single_responses(self, model_type: str = None) -> List[Dict]:
        """Get single model responses, optionally filtered by model type"""
        single_responses = [r for r in self.responses if 'model_type' in r]
        if model_type:
            return [r for r in single_responses if r['model_type'] == model_type]
        return single_responses
    
    def get_stats(self) -> Dict:
        """Get statistics about stored responses"""
        pairs = self.get_response_pairs()
        teacher_singles = self.get_single_responses('teacher')
        student_singles = self.get_single_responses('student')
        
        stats = {
            'total_entries': len(self.responses),
            'response_pairs': len(pairs),
            'teacher_only_responses': len(teacher_singles),
            'student_only_responses': len(student_singles),
            'problems_by_level': defaultdict(int),
            'problems_by_subject': defaultdict(int)
        }
        
        for response in self.responses:
            problem = response.get('problem', {})
            stats['problems_by_level'][problem.get('level', 'unknown')] += 1
            stats['problems_by_subject'][problem.get('subject', 'unknown')] += 1
        
        if pairs:
            equivalent_pairs = sum(1 for p in pairs if p['evaluation']['answers_equivalent'])
            stats['pair_equivalence_rate'] = equivalent_pairs / len(pairs)
        
        return stats

class ExperimentSaver:
    """Handle saving of experiment results and artifacts"""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize response storage
        self.response_storage = ResponseStorage(self.save_dir)
    
    def save_metrics(self, metrics: Dict):
        """Save training metrics"""
        save_json(metrics, self.save_dir / 'metrics.json')
    
    def save_training_history(self, history: Dict):
        """Save training history"""
        save_json(history, self.save_dir / 'training_history.json')
    
    def save_curriculum_data(self, curriculum_data: Dict):
        """Save curriculum learning data"""
        save_json(curriculum_data, self.save_dir / 'curriculum_data.json')
    
    def save_evaluation_results(self, eval_results: list):
        """Save evaluation results"""
        save_json(eval_results, self.save_dir / 'eval_results.json')
    
    def save_config(self, config: Dict):
        """Save experiment configuration"""
        save_json(config, self.save_dir / 'config.json')
    
    def save_model_checkpoint(self, model, checkpoint_name: str = 'student_model'):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / f'{checkpoint_name}.pt'
        torch.save(model.state_dict(), checkpoint_path)
    
    def get_response_storage(self) -> ResponseStorage:
        """Get the response storage instance"""
        return self.response_storage
    
    def get_save_dir(self) -> Path:
        """Get save directory path"""
        return self.save_dir