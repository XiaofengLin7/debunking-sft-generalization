import re
def extract_solution(solution_str):
    """Extract the action sequence from the solution string."""
    # processed_str = solution_str.split('\n')[-1]
    processed_str = solution_str
    action_pattern = r'<answer>(.*?)</answer>|<action>(.*?)</action>'
    match = re.finditer(action_pattern, processed_str)
    matches = list(match)
    if matches:
        # Get the last match and check both answer and action groups
        final_match = matches[-1]
        final_answer = final_match.group(1) or final_match.group(2)
        final_answer = final_answer.strip() if final_answer else None
    else:
        final_answer = None
    return final_answer, processed_str

def validate_answer_structure(processed_str: str) -> bool:
    """Adapted from https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py"""
    """Performs validation of answer structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True

    # Check required tags
    tags = {
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        
        if count != expected_count:
            validation_passed = False

    # Verify tag order
    if (positions['answer_start'] > positions['answer_end']):
        validation_passed = False

    return validation_passed


def compute_score(solution_str, ground_truth, format_score=0.1, score=1.0, *args, **kwargs):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return 0
    
    format_correct = validate_answer_structure(processed_str)
    if not format_correct:
        return 0
    else:
        final_answer, _ = extract_solution(processed_str)
        if final_answer is None:
            return 0
        else:
            if final_answer.strip() == ground_truth.strip():
                return score
            return format_score
                
    return 0

def main():
    solution_str = "Assistant: <answer>take apple 1 from countertop 1 </answer>"
    print(compute_score(solution_str, "take apple 1 from countertop 1"))

if __name__ == "__main__":
    main()