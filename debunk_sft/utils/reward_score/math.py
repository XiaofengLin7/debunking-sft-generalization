"""
Math reward function using MATH-Perturb's official answer_check function.

This module wraps the answer_check function from the MATH-Perturb repository
to provide reward scoring for math problems in the REIL framework.

Reference: https://github.com/Kaffaljidhmah2/MATH-Perturb
"""

import sys
import os

# Add MATH-Perturb to path
_MATH_PERTURB_PATH = os.path.join(os.path.dirname(__file__), '../../../thirdparty/MATH-Perturb')
if _MATH_PERTURB_PATH not in sys.path:
    sys.path.insert(0, _MATH_PERTURB_PATH)

from evaluate import answer_check


def compute_score_math(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """
    Compute reward score for math problems using MATH-Perturb's answer_check.
    
    Args:
        solution_str (str): The model's solution string containing the predicted answer.
        ground_truth (str): The ground truth answer string.
        data_source (str, optional): The data source identifier (e.g., 'math_perturb_simple').
        extra_info (dict, optional): Additional information including the problem statement.
        **kwargs: Additional keyword arguments (ignored).
    
    Returns:
        float: 1.0 if the answer is correct, 0.0 otherwise.
    """
    # Extract problem from extra_info if available
    problem = ''
    if extra_info is not None:
        if isinstance(extra_info, dict):
            problem = extra_info.get('question', '')
    
    # Determine dataset type based on data_source
    # Use 'perturb' for MATH-Perturb datasets, 'original' for standard MATH
    dataset_type = 'original'  # default
    if data_source is not None:
        data_source_lower = data_source.lower()
        if 'perturb' in data_source_lower:
            dataset_type = 'perturb'
    
    try:
        is_correct = answer_check(problem, solution_str, ground_truth, dataset_type)
        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"[compute_score_math] Error during answer_check: {e}")
        return 0.0


def compute_score_math_perturb(solution_str, ground_truth, data_source=None, extra_info=None, **kwargs):
    """
    Compute reward score specifically for MATH-Perturb evaluation.
    Always uses 'perturb' dataset type.
    
    Args:
        solution_str (str): The model's solution string containing the predicted answer.
        ground_truth (str): The ground truth answer string.
        data_source (str, optional): The data source identifier.
        extra_info (dict, optional): Additional information including the problem statement.
        **kwargs: Additional keyword arguments (ignored).
    
    Returns:
        float: 1.0 if the answer is correct, 0.0 otherwise.
    """
    # Extract problem from extra_info if available
    problem = ''
    if extra_info is not None:
        if isinstance(extra_info, dict):
            problem = extra_info.get('question', '')
    
    try:
        is_correct = answer_check(problem, solution_str, ground_truth, 'perturb')
        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"[compute_score_math_perturb] Error during answer_check: {e}")
        return 0.0


if __name__ == "__main__":
    # Test the reward function
    problem = "What is 2 + 2?"
    solution = "The answer is \\boxed{4}."
    ground_truth = "4"
    
    score = compute_score_math(solution, ground_truth, data_source='diverse_MATH', 
                               extra_info={'question': problem})
    print(f"Test 1 - Correct answer: score = {score}")
    assert score == 1.0, "Expected score 1.0 for correct answer"
    
    # Test with wrong answer
    wrong_solution = "The answer is \\boxed{5}."
    score = compute_score_math(wrong_solution, ground_truth, data_source='diverse_MATH',
                               extra_info={'question': problem})
    print(f"Test 2 - Wrong answer: score = {score}")
    assert score == 0.0, "Expected score 0.0 for wrong answer"
    
    # Test MATH-Perturb specific function
    score = compute_score_math_perturb(solution, ground_truth, 
                                        extra_info={'question': problem})
    print(f"Test 3 - MATH-Perturb correct: score = {score}")
    assert score == 1.0, "Expected score 1.0 for correct answer"
    
    print("All tests passed!")
