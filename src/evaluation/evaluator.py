"""
Evaluation utilities for knowledge distillation models using LLM-based evaluation
"""

import os
import re
import time
import torch
import numpy as np
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from openai import OpenAI, RateLimitError, APIError, APIConnectionError, APITimeoutError
from dotenv import load_dotenv

from ..data.dataset import MATHDataset

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """LLM-based model evaluation for mathematical reasoning"""
    
    def __init__(self, tokenizer, device: str = "cuda", model: str = "gpt-4o-mini", max_retries: int = 3):
        """
        Initialize evaluator with OpenAI LLM for answer comparison
        
        Args:
            tokenizer: Tokenizer for the models being evaluated
            device: Device for model inference
            model: OpenAI model to use for evaluation
            max_retries: Maximum number of retries for API calls
        """
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.max_retries = max_retries
        
        # Initialize caches for efficiency
        self._prompt_cache = {}  # Cache tokenized prompts
        self._api_cache = {}     # Cache API evaluation results
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize logger
        from ..utils.logging_config import get_logger
        self.logger = get_logger(__name__)
        self.logger.info("✅ LLM-based evaluation enabled with OpenAI API")
    
    def evaluate_model(self, 
                      model, 
                      dataset: MATHDataset,
                      n_samples: int = 100, 
                      log_solutions: bool = True,
                      model_type: str = 'student',
                      response_storage = None) -> Dict:
        """
        Evaluate model using LLM-based answer comparison
        """
        model.eval()
        
        results = {
            'accuracy_by_level': defaultdict(list),
            'accuracy_by_subject': defaultdict(list),
            'generated_solutions': [],
            'teacher_student_agreement': defaultdict(list),
            'response_lengths': defaultdict(list)
        }
        
        self.logger.info(f"Evaluating on {n_samples} samples using LLM evaluation...")
        eval_pbar = tqdm(range(n_samples), desc="LLM Evaluation")
        
        with torch.no_grad():
            for eval_step in eval_pbar:
                idx = np.random.randint(len(dataset))
                problem = dataset[idx]
                
                prompt = f"Problem: {problem['problem']}\n\nSolution: Please solve this step-by-step. Use LaTeX formatting for mathematical expressions (fractions, matrices, etc.). Provide your final answer in \\boxed{{}} format."
                
                solution, is_correct = self._generate_and_evaluate(
                    model, prompt, problem
                )
                
                # Calculate metrics
                level = problem['level']
                subject = problem.get('subject', 'unknown')
                
                results['accuracy_by_level'][level].append(is_correct)
                results['accuracy_by_subject'][subject].append(is_correct)
                
                solution_length = len(solution.split())
                results['response_lengths'][level].append(solution_length)
                
                if log_solutions:
                    solution_log = {
                        'eval_step': eval_step,
                        'problem': problem,
                        'generated_solution': solution,
                        'is_correct': is_correct,
                    }
                    results['generated_solutions'].append(solution_log)
                    
                    # Store in organized response storage if provided
                    if response_storage:
                        extracted_answer = self._extract_answer(solution)
                        response_storage.add_single_response(
                            problem=problem,
                            response=solution,
                            model_type=model_type,
                            extracted_answer=extracted_answer or 'No answer extracted',
                            is_correct=is_correct
                        )
                
                current_acc = np.mean([np.mean(accs) for accs in results['accuracy_by_level'].values()])
                eval_pbar.set_postfix({'accuracy': f"{current_acc:.3f}"})
        
        return self._compute_final_metrics(results)
    
    def evaluate_teacher_student_pair(self,
                                    teacher_model,
                                    student_model,
                                    dataset: MATHDataset,
                                    n_samples: int = 100,
                                    response_storage = None) -> Dict:
        """Evaluate teacher-student agreement using LLM evaluation"""
        teacher_model.eval()
        student_model.eval()
        
        results = {
            'teacher_accuracy': [],
            'student_accuracy': [],
            'agreement': [],
            'accuracy_by_level': defaultdict(lambda: {'teacher': [], 'student': [], 'agreement': []}),
        }
        
        with torch.no_grad():
            for eval_step in tqdm(range(n_samples), desc="Teacher-Student LLM Evaluation"):
                idx = np.random.randint(len(dataset))
                problem = dataset[idx]
                
                prompt = f"Problem: {problem['problem']}\n\nSolution: Please solve this step-by-step. Use LaTeX formatting for mathematical expressions (fractions, matrices, etc.). Provide your final answer in \\boxed{{}} format."
                
                # Evaluate teacher
                teacher_solution, teacher_correct = self._generate_and_evaluate(
                    teacher_model, prompt, problem
                )
                
                # Evaluate student
                student_solution, student_correct = self._generate_and_evaluate(
                    student_model, prompt, problem
                )
                
                # Calculate agreement
                agreement = 1.0 if teacher_correct == student_correct else 0.0
                level = problem['level']
                
                # Store results
                results['teacher_accuracy'].append(teacher_correct)
                results['student_accuracy'].append(student_correct)
                results['agreement'].append(agreement)
                
                results['accuracy_by_level'][level]['teacher'].append(teacher_correct)
                results['accuracy_by_level'][level]['student'].append(student_correct)
                results['accuracy_by_level'][level]['agreement'].append(agreement)
                
                # Store in organized response storage if provided
                if response_storage:
                    teacher_extracted = self._extract_answer(teacher_solution)
                    student_extracted = self._extract_answer(student_solution)
                    
                    # Check if extracted answers are equivalent
                    answers_equivalent = False
                    llm_details = {}
                    if teacher_extracted and student_extracted:
                        answers_equivalent = self._llm_compare_answers(
                            teacher_extracted, student_extracted
                        )
                        llm_details = {
                            'comparison_made': True,
                            'teacher_vs_ground_truth': teacher_correct,
                            'student_vs_ground_truth': student_correct,
                            'teacher_vs_student': answers_equivalent
                        }
                    
                    response_entry = response_storage.add_response_pair(
                        problem=problem,
                        teacher_response=teacher_solution,
                        student_response=student_solution,
                        teacher_extracted_answer=teacher_extracted or 'No answer extracted',
                        student_extracted_answer=student_extracted or 'No answer extracted',
                        answers_equivalent=answers_equivalent,
                        llm_evaluation_details=llm_details
                    )
                    
                    # Fill in individual correctness
                    response_entry['evaluation']['teacher_correct'] = teacher_correct
                    response_entry['evaluation']['student_correct'] = student_correct
        
        # Compute summary metrics
        final_results = {
            'teacher_overall_accuracy': np.mean(results['teacher_accuracy']),
            'student_overall_accuracy': np.mean(results['student_accuracy']),
            'overall_agreement': np.mean(results['agreement']),
        }
        
        # Add per-level metrics
        for level, level_results in results['accuracy_by_level'].items():
            final_results[f'teacher_accuracy_level_{level}'] = np.mean(level_results['teacher'])
            final_results[f'student_accuracy_level_{level}'] = np.mean(level_results['student'])
            final_results[f'agreement_level_{level}'] = np.mean(level_results['agreement'])
        
        return final_results
    
    def _generate_and_evaluate(self, model, prompt: str, problem: Dict) -> Tuple[str, bool]:
        """Generate response and evaluate using LLM with caching"""
        # Check cache for tokenized prompt
        cache_key = hash(prompt)
        if cache_key in self._prompt_cache:
            inputs = self._prompt_cache[cache_key]
        else:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            self._prompt_cache[cache_key] = inputs
        
        # Use deterministic generation
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # Greedy decoding for consistency
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Extract only the tokens generated *after* the prompt to avoid character count drift
        generated_ids = outputs[0][inputs['input_ids'].shape[-1]:]
        solution_part = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Use LLM to compare answers
        correct = self._llm_compare_answers(problem['solution'], solution_part)
        
        return solution_part, correct
    
    def _llm_compare_answers(self, reference_solution: str, generated_solution: str) -> bool:
        """
        Compare two mathematical answers using LLM evaluation
        
        Args:
            reference_solution: The correct answer from dataset
            generated_solution: The generated answer to evaluate
            
        Returns:
            Boolean indicating if answers are equivalent
        """
        # Extract answers from both texts
        ref_answer = self._extract_answer(reference_solution)
        gen_answer = self._extract_answer(generated_solution)
        
        if not ref_answer or not gen_answer:
            return False
        
        # Create evaluation prompt
        prompt = self._create_evaluation_prompt(ref_answer, gen_answer)
        
        # Check cache for API results
        api_cache_key = hash(f"{ref_answer}|||{gen_answer}")
        if api_cache_key in self._api_cache:
            return self._api_cache[api_cache_key]
        
        # Call OpenAI API with retries and specific error handling
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a mathematical expert evaluating answer equivalence. Be precise and consider mathematical equality, not just string similarity."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    timeout=30  # 30 second timeout
                )
                
                result = response.choices[0].message.content.strip()
                parsed_result = self._parse_evaluation_result(result)
                
                # Cache the result
                self._api_cache[api_cache_key] = parsed_result
                return parsed_result
                
            except RateLimitError as e:
                wait_time = min(60, (2 ** attempt) * 10)  # Cap at 60 seconds
                logger.warning(f"Rate limit hit on attempt {attempt + 1}. Waiting {wait_time}s...")
                if attempt == self.max_retries - 1:
                    logger.error(f"Rate limit exceeded after {self.max_retries} attempts")
                    return False
                time.sleep(wait_time)
                
            except APIConnectionError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                if attempt == self.max_retries - 1:
                    logger.error(f"Connection failed after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(wait_time)
                
            except APITimeoutError as e:
                wait_time = 2 ** attempt
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                if attempt == self.max_retries - 1:
                    logger.error(f"Timeout after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(wait_time)
                
            except APIError as e:
                # For other API errors, don't retry if it's a client error (4xx)
                if hasattr(e, 'status_code') and 400 <= e.status_code < 500:
                    logger.error(f"Client error (non-retryable): {e}")
                    return False
                
                wait_time = 2 ** attempt
                logger.warning(f"API error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                if attempt == self.max_retries - 1:
                    logger.error(f"API error after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(wait_time)
                
            except Exception as e:
                wait_time = 2 ** attempt
                logger.warning(f"Unexpected error on attempt {attempt + 1}: {e}. Retrying in {wait_time}s...")
                if attempt == self.max_retries - 1:
                    logger.error(f"Unexpected error after {self.max_retries} attempts: {e}")
                    return False
                time.sleep(wait_time)
        
        return False
    
    def _extract_answer(self, text: str) -> Optional[str]:
        """Extract answer from solution text"""
        # Try to extract from \boxed{} first
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, text)
        if matches:
            return matches[-1]
        
        # Fallback answer extraction using common patterns
        patterns = [
            r'(?:answer|result|solution)\s*(?:is|=)\s*([^.\n]+)',
            r'therefore[,\s]+([^.\n]+)',
            r'thus[,\s]+([^.\n]+)',
            r'=\s*([^.\n]+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            if matches:
                return matches[-1].strip()
        
        # If no pattern matches, try to get the last line that looks like an answer
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('Problem:') and not line.startswith('Solution:'):
                # Check if line contains mathematical content
                if any(char in line for char in '0123456789=+-*/^()[]{}'):
                    return line
        
        return None
    
    def _create_evaluation_prompt(self, reference: str, generated: str) -> str:
        """Create prompt for LLM evaluation"""
        return f"""Compare these two mathematical answers and determine if they are equivalent:

Reference Answer: {reference}
Generated Answer: {generated}

Consider the following:
1. Mathematical equivalence (e.g., 1/2 = 0.5 = 50%)
2. Different valid forms of the same answer (e.g., simplified vs. unsimplified fractions)
3. Ignore minor formatting differences or LaTeX syntax variations
4. Consider equivalent mathematical expressions (e.g., 2x + 3x = 5x)

Respond with exactly one of:
- "EQUIVALENT: [brief explanation]" if the answers are mathematically equivalent
- "NOT_EQUIVALENT: [brief explanation]" if the answers are different

Your response:"""
    
    def _parse_evaluation_result(self, result: str) -> bool:
        """Parse LLM evaluation result"""
        result = result.strip().upper()
        
        if result.startswith("EQUIVALENT:"):
            return True
        elif result.startswith("NOT_EQUIVALENT:"):
            return False
        else:
            # Try to infer from content
            if "equivalent" in result.lower() and "not" not in result.lower():
                return True
            else:
                return False
    
    def _compute_final_metrics(self, results: Dict) -> Dict:
        """Compute final evaluation metrics"""
        final_results = {}
        
        # Accuracy by level
        for level, accuracies in results['accuracy_by_level'].items():
            final_results[f'accuracy_level_{level}'] = np.mean(accuracies)
        
        # Accuracy by subject
        for subject, accuracies in results['accuracy_by_subject'].items():
            final_results[f'accuracy_{subject}'] = np.mean(accuracies)
        
        # Overall accuracy
        all_accuracies = [acc for accs in results['accuracy_by_level'].values() for acc in accs]
        final_results['overall_accuracy'] = np.mean(all_accuracies)
        
        # Teacher-student agreement (if available)
        if results['teacher_student_agreement']:
            all_agreements = [agr for agrs in results['teacher_student_agreement'].values() for agr in agrs]
            final_results['teacher_student_agreement'] = np.mean(all_agreements)
        
        # Response lengths
        all_lengths = [length for lengths in results['response_lengths'].values() for length in lengths]
        final_results['avg_response_length'] = np.mean(all_lengths)
        
        return final_results
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        self._prompt_cache.clear()
        self._api_cache.clear()
        self.logger.info("Cleared evaluation caches")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'prompt_cache_size': len(self._prompt_cache),
            'api_cache_size': len(self._api_cache)
        }