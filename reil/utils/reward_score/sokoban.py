import re

def extract_solution(solution_str):
    """Extract the action sequence from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    action_pattern = r'<action>(.*?)</action>'
    match = re.finditer(action_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

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
    solution = extract_solution(solution_str)
    if solution is None:
        return 0
    else:
        action = extract_action(solution)
        if action == ground_truth:
            return score
        return format_score

def main():
    solution_str = "Assistant: <action>up</action> <action>right</action> <action>down</action> <action>left</action>"
    print(extract_solution(solution_str))
    solution_str = "<|im_start|>assistant\n\n<|im_end|>\n\n<action>up</action> <action>right</action> <action>down</action> <action><action>left</action></action>"
    print(extract_solution(solution_str))
    solution_str = "<|im_start|>assistant\n\n<|im_end|>\n\n<action>up</action>"
    extracted_solution = extract_solution(solution_str)
    print(extract_action(extracted_solution))
    print(compute_score(solution_str, 1))
    print(compute_score(solution_str, 2))
    print(compute_score(solution_str, 3))
    print(compute_score(solution_str, 4))

if __name__ == "__main__":
    main()