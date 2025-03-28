import re

def extract_solution(solution_str):
    """Extract the action sequence from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     processed_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     print("[Error] Failed to locate model response header")
    #     return None, solution_str
    # print(f"\n[Model Response]\n{solution_str}")
    processed_str = solution_str.split('\n')[-1]
    
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

def extract_action(text):
    """
    Extract action from text.
    - 0: Still (Invalid Action)
    - 1: Up
    - 2: Down
    - 3: Left
    - 4: Right
    """
        
    DIRECTION_MAP = {"Up": 1, "Down": 2, "Left": 3, "Right": 4}
    # TODO: originally, we parse either number (key of direction_map) or direction (value of direction_map).
    # here we remove numbers and preserve directions only, but regex has not been removed. please remove them later.
    pattern = r'^\s*(([1-4])\s*\((up|down|left|right)\)|(up|down|left|right)|([1-4]))\s*$'
    match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
    
    if not match:
        return 0
    
    if match.group(2):   
        return int(match.group(2))
    elif match.group(4): 
        return DIRECTION_MAP[match.group(4).capitalize()]
    elif match.group(5): 
        return int(match.group(5))
    
    return 0

def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0, *args, **kwargs):
    """The scoring function for Sokoban."""
    final_answer, processed_str = extract_solution(solution_str)

    if final_answer is None:
        return 0
    else:
        action = extract_action(final_answer)
        if action == ground_truth:
            return score
        return format_score

def main():
    # solution_str = "Assistant: <answer>up</answer> <answer>right</answer> <answer>down</answer> <answer>left</answer>"
    # print(extract_solution(solution_str))
    # solution_str = "<|im_start|>assistant\n\n<|im_end|>\n\n<answer>up</answer> <answer>right</answer> <answer>down</answer> <answer>left</answer>"
    # print(extract_solution(solution_str))
    # solution_str = "<|im_start|>assistant\n\n<|im_end|>\n\n<answer>up</answer>"
    # extracted_solution = extract_solution(solution_str)
    # print(extract_action(extracted_solution))
    # print(compute_score(solution_str, 1))
    # print(compute_score(solution_str, 2))
    # print(compute_score(solution_str, 3))
    # print(compute_score(solution_str, 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>Right</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>right</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>RIGHT</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>4</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>4(right)</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>3</answer>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<action>4(right)</action>", 4))
    print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<action>up</action> <answer>left</answer>", 4))
if __name__ == "__main__":
    main()