import re
import numpy as np
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

def extract_action_base(text):
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

def extract_action_cardinal(text):
    """
    Extract action from text.
    - 0: Still (Invalid Action)
    - 1: North
    - 2: South
    - 3: West
    - 4: East
    """
    DIRECTION_MAP = {"North": 1, "South": 2, "West": 3, "East": 4}
    pattern = r'^\s*(([1-4])\s*\((north|south|west|east)\)|(north|south|west|east)|([1-4]))\s*$'
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

def extract_action_emoji(text):
    """
    Extract action from text.
    - 0: Still (Invalid Action)
    - 1: ⬆️
    - 2: ⬇️
    - 3: ⬅️
    - 4: ➡️
    """
    DIRECTION_MAP = {"⬆️": 1, "⬇️": 2, "⬅️": 3, "➡️": 4}
    pattern = r'^\s*(([1-4])\s*\((⬆️|⬇️|⬅️|➡️)\)|(⬆️|⬇️|⬅️|➡️)|([1-4]))\s*$'
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
    
def validate_response_structure(processed_str: str) -> bool:
    """Adapted from https://github.com/Unakar/Logic-RL/blob/main/verl/utils/reward_score/kk.py"""
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
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
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        validation_passed = False

    return validation_passed

def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0, *args, **kwargs):
    """The scoring function for Sokoban."""
    final_answer, _ = extract_solution(solution_str)

    if final_answer is None:
        return 0
    else:
        action = extract_action_base(final_answer)
        if action == ground_truth:
            return score
        return format_score
    
def compute_score_with_format(solution_str, ground_truth, format_score=0.1, score=1.0, *args, **kwargs):
    """The scoring function for Sokoban. with format score."""
    final_answer, _ = extract_solution(solution_str)

    if final_answer is None:
        return 0
    else:
        action = extract_action_base(final_answer)
        if action == ground_truth:
            return score
        return format_score

def compute_score_with_logic(solution_str, ground_truth, format_score=0.1, score=1.0, *args, **kwargs):
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return 0
    
    format_correct = validate_response_structure(processed_str)
    if not format_correct:
        return 0
    else:
        final_answer, _ = extract_solution(processed_str)
        if final_answer is None:
            return 0
        else:
            action = extract_action_base(final_answer)
            if action == ground_truth:
                return score
    return format_score

def convert_action_sequence(action_sequence, data_source="sokoban"):
    """
    Convert a sequence of actions to their numerical representations.
    
    Args:
        action_sequence: Can be either:
            - A string containing multiple actions (e.g., "up, down, left, right")
            - A list of action strings (e.g., ["up", "down", "left", "right"])
    
    Returns:
        A list of numerical action values:
        - 0: Still (Invalid Action)
        - 1: Up
        - 2: Down
        - 3: Left
        - 4: Right
    """
    # If input is a string, split it into individual actions
    if isinstance(action_sequence, str):
        # Split by common separators (comma, semicolon, space)
        actions = re.split(r'[,;\s]+', action_sequence.strip())
        actions = [a for a in actions if a]  # Remove empty strings
    else:
        # Assume it's already a list/iterable of actions
        actions = action_sequence
    
    # Convert each action using the existing extract_action function
    numerical_actions = []
    for action in actions:
        if "cardinal" in data_source:
            numerical_actions.append(extract_action_cardinal(action))
        elif "emoji" in data_source:
            numerical_actions.append(extract_action_emoji(action))
        else:
            numerical_actions.append(extract_action_base(action))
    
    return numerical_actions

def compute_score_with_action_sequence(solution_str, ground_truth, data_source, format_score=0.1, score=1.0, *args, **kwargs):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return 0
    
    format_correct = validate_response_structure(processed_str)
    if not format_correct:
        return 0
    else:
        final_answer, _ = extract_solution(processed_str)
        if final_answer is None:
            return 0
        else:
            action_sequence = convert_action_sequence(final_answer, data_source)
            len_horizon = len(ground_truth)
            if len(action_sequence) < len_horizon:
                return 0
            else:
                if np.array_equal(action_sequence[:len_horizon], ground_truth):
                    return score
                elif len(action_sequence) == len_horizon and not any(action == 0 for action in action_sequence):
                    return format_score
                
    return 0

def compute_score_with_action_sequence_zero_format_score(solution_str, ground_truth, data_source, score=1.0, *args, **kwargs):
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return 0
    
    format_correct = validate_response_structure(processed_str)
    if not format_correct:
        return 0
    else:
        final_answer, _ = extract_solution(processed_str)
        if final_answer is None:
            return 0
        else:
            action_sequence = convert_action_sequence(final_answer, data_source)
            len_horizon = len(ground_truth)
            if len(action_sequence) < len_horizon:
                return 0
            else:
                if np.array_equal(action_sequence[:len_horizon], ground_truth):
                    return score
                elif len(action_sequence) == len_horizon and not any(action == 0 for action in action_sequence):
                    return 0
                
    return 0

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
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>Right</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>right</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>RIGHT</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>4</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>4(right)</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<answer>3</answer>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<action>4(right)</action>", 4))
    # print(compute_score("<|im_start|>assistant\n\n<|im_end|>\n\n<action>up</action> <answer>left</answer>", 4))
    # print(compute_score_with_logic("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up</answer>", 1))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, down, right</answer>", [1, 3, 2, 4],'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, down, right, right</answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, </answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer></answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, </answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up  left down left</answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>north, west, south, east</answer>", [1, 3, 2, 4], 'sokoban_cardinal', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>⬆️, ⬅️, ⬇️, ➡️</answer>", [1, 3, 2, 4], 'sokoban_emoji', 0.1, 1.0))
    print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>⬆️</answer>", [1], 'sokoban_emoji', 0.1, 1.0))
if __name__ == "__main__":
    main()