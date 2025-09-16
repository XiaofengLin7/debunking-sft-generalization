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
    
def extract_action_numerical(text):
    """
    Extract action from text.
    - 0: Still (Invalid Action)
    - 1: 1
    - 2: 2
    - 3: 3
    - 4: 4
    """
    pattern = r'^\s*(([1-4])\s*\((1|2|3|4)\)|(1|2|3|4)|([1-4]))\s*$'
    match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)

    if not match:
        return 0
    
    if match.group(2):
        return int(match.group(2))
    elif match.group(4):
        return int(match.group(4))
    elif match.group(5):
        return int(match.group(5))
    
    return 0

def extract_action_alphabetical(text):
    """
    Extract action from text.
    - 0: Still (Invalid Action)
    - 1: A (Up)
    - 2: B (Down)
    - 3: C (Left)
    - 4: D (Right)
    """
    DIRECTION_MAP = {"A": 1, "B": 2, "C": 3, "D": 4}
    pattern = r'^\s*(([1-4])\s*\((A|B|C|D)\)|(A|B|C|D)|([1-4]))\s*$'
    match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)

    if not match:
        return 0
    
    if match.group(2):  # Number in parentheses like "1(A)"
        return int(match.group(2))
    elif match.group(4):  # Letter like A, B, C, D
        # Map letters to action numbers
        return DIRECTION_MAP[match.group(4).capitalize()]
    elif match.group(5):  # Number like 1, 2, 3, 4
        return int(match.group(5))
    
    return 0

def extract_action(action, action_mapping):
    """
    Extract action from action mapping. If the action is not in the action mapping, return 0.
    action_mapping is a dictionary with keys as strings and values as integers.
    """
    # Build a case-insensitive lookup for labels (assume mapping is valid)
    normalized_map = {}
    label_candidates = []
    for k, v in action_mapping.items():
        key_stripped_lower = str(k).strip().lower()
        normalized_map[key_stripped_lower] = v
        label_candidates.append(str(k))

    text = "" if action is None else str(action)

    # If we have labels, try to parse in the same grammar as specialized extractors:
    #   1) "n(label)" -> return n
    #   2) label-only -> return mapped value (case-insensitive)
    #   3) bare number 1-4 -> return number
    if label_candidates:
        label_alt = "(?:" + "|".join(re.escape(lbl) for lbl in label_candidates) + ")"
        pattern = rf'^\s*(([1-4])\s*\(({label_alt})\)|({label_alt})|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        if match:
            if match.group(2):
                return int(match.group(2))
            elif match.group(4):
                matched_label = match.group(4).strip().lower()
                return normalized_map.get(matched_label, 0)
            elif match.group(5):
                return int(match.group(5))

    # Fallback: direct key match (case-insensitive)
    value = normalized_map.get(text.strip().lower())
    if value is not None:
        return value


    return 0

def extract_action_random(text):
    """
    - 0: "None"
    - 1: "*"
    - 2: "&"
    - 3: "1"
    - 4: "M"
    """
    DIRECTION_MAP = {"*": 1, "&": 2, "1": 3, "M": 4}
    pattern = r'^\s*(([1-4])\s*\(([*&1M])\)|([*&1M])|([1-4]))\s*$'
    match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)

    if not match:
        return 0
    
    if match.group(2):
        return int(match.group(2))
    elif match.group(4):
        # print(match.group(4).capitalize())
        return DIRECTION_MAP[match.group(4).capitalize()]
    elif match.group(5):
        return int(match.group(5))
    
    return 0

def extract_action_mapping(prompt):
    """
    Extract action mapping from prompt. 
    The prompt is expected to be in the following format:
    ```
    You are a Sokoban solver.

    Sokoban Quick Guide
    Goal: Push all boxes (X) onto targets (O).

    Symbols:
    # Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

    Rules:
    1. Push boxes (can't pull).
    2. Avoid walls (#).

    Answers:
    <answer> {action_1} </answer> | <answer> {action_2} </answer> | <answer> {action_3} </answer> | <answer> {action_4} </answer>

    Rewards:
    Move: -0.1
    Box on target: +1.0
    All boxes placed: +10.0


    [Current Observation]:
    {observation}
    Decide the next action:\
    ```
    """
    # Extract the four actions from the Answers section of the prompt
    # Look for a line like: <answer> {action_1} </answer> | <answer> {action_2} </answer> | <answer> {action_3} </answer> | <answer> {action_4} </answer>
    answers_pattern = r'Answers:\s*(?:\n)?\s*<answer>\s*(.*?)\s*</answer>\s*\|\s*<answer>\s*(.*?)\s*</answer>\s*\|\s*<answer>\s*(.*?)\s*</answer>\s*\|\s*<answer>\s*(.*?)\s*</answer>'
    match = re.search(answers_pattern, prompt, re.IGNORECASE | re.DOTALL)
    if match:
        action_1, action_2, action_3, action_4 = [a.strip() for a in match.groups()]
        return {action_1: 1, action_2: 2, action_3: 3, action_4: 4}
    
    return None

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


def convert_action_sequence(action_sequence, data_source="sokoban", action_mapping=None):
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
    if action_mapping is not None:
        for action in actions:
            numerical_actions.append(extract_action(action, action_mapping))
        return numerical_actions
    else:
        for action in actions:
            if "cardinal" in data_source:
                numerical_actions.append(extract_action_cardinal(action))
            elif "emoji" in data_source:
                numerical_actions.append(extract_action_emoji(action))
            elif "numerical" in data_source:
                numerical_actions.append(extract_action_numerical(action))
            elif "alphabetical" in data_source:
                numerical_actions.append(extract_action_alphabetical(action))
            elif "random" in data_source:
                numerical_actions.append(extract_action_random(action))
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

def compute_score_with_action_sequence_and_diverse_prompt(solution_str, ground_truth, data_source, score=1.0, *args, **kwargs):
    
    if "Assistant:" in solution_str:
        prompt = solution_str.split("Assistant:", 1)[0]
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        prompt = solution_str.split("<|im_start|>assistant", 1)[0]
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
            action_mapping = extract_action_mapping(prompt)
            if action_mapping is None:
                return 0
            else:
                action_sequence = convert_action_sequence(final_answer, data_source, action_mapping)
                len_horizon = len(ground_truth)
                if len(action_sequence) < len_horizon:
                    return 0
                else:
                    if np.array_equal(action_sequence[:len_horizon], ground_truth):
                        return score
                    elif len(action_sequence) == len_horizon and not any(action == 0 for action in action_sequence):
                        return 0
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
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, down, right</answer>", [1, 3, 2, 4],'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, down, right, right</answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, left, </answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer></answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up, </answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>up  left down left</answer>", [1, 3, 2, 4], 'sokoban', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>north, west, south, east</answer>", [1, 3, 2, 4], 'sokoban_cardinal', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>⬆️, ⬅️, ⬇️, ➡️</answer>", [1, 3, 2, 4], 'sokoban_emoji', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>⬆️</answer>", [1], 'sokoban_emoji', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>1, 3, 2, 4</answer>", [1, 3, 2, 4], 'sokoban_numerical', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>1</answer>", [1], 'sokoban_numerical', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>A, C, B, D</answer>", [1, 3, 2, 4], 'sokoban_alphabetical', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>A, C</answer>", [1, 3], 'sokoban_alphabetical', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>*</answer>", [1], 'sokoban_random', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>1</answer>", [3], 'sokoban_random', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>M</answer>", [4], 'sokoban_random', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>&</answer>", [2], 'sokoban_random', 0.1, 1.0))
    # print(compute_score_with_action_sequence("<|im_start|>assistant\n\n<|im_end|>\n\n<think></think><answer>&, M</answer>", [2, 4], 'sokoban_random', 0.1, 1.0))
    
    prompt_0 = """
    
    <|im_start|>user\nYou are a Sokoban solver.\n\nSokoban Quick Guide\nGoal: Push all boxes (X) onto targets (O).\n\nSymbols:\n# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target\n\nRules:\n1. Push boxes (can't pull).\n2. Avoid walls (#).\n\nAnswers:\n<answer> A </answer> | <answer> B </answer> | <answer> C </answer> | <answer> D </answer>\nwhere A is Up, B is Down, C is Left, D is Right.\n\n\nRewards:\nMove: -0.1\nBox on target: +1.0\nAll boxes placed: +10.0\n\n\n[Current Observation]:\n # \t # \t # \t # \t # \t # \t\n # \t _ \t # \t # \t # \t # \t\n # \t _ \t _ \t # \t # \t # \t\n # \t _ \t _ \t # \t # \t # \t\n # \t _ \t P \t X \t O \t # \t\n # \t # \t # \t # \t # \t # \t\nDecide the next action:\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>
    """
    prompt_1 = """<|im_start|>user\nYou are a Sokoban solver.\n\nSokoban Quick Guide\nGoal: Push all boxes (X) onto targets (O).\n\nSymbols:\n# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target\n\nRules:\n1. Push boxes (can't pull).\n2. Avoid walls (#).\n\nAnswers:\n<answer> tree </answer> | <answer> fix </answer> | <answer> jade </answer> | <answer> freedom </answer>\nwhere tree is Up, fix is Down, jade is Left, freedom is Right.\n\nRewards:\nMove: -0.1\nBox on target: +1.0\nAll boxes placed: +10.0\n\n\n[Current Observation]:\n # \t # \t # \t # \t # \t # \t\n # \t P \t X \t _ \t O \t # \t\n # \t _ \t _ \t _ \t _ \t # \t\n # \t # \t # \t # \t # \t # \t\n # \t # \t # \t # \t # \t # \t\n # \t # \t # \t # \t # \t # \t\nDecide the next action:\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>"""
    print(action_mapping_0:=extract_action_mapping(prompt_0))
    print(action_mapping_1:=extract_action_mapping(prompt_1))
    print(extract_action("freedom", action_mapping_1))
    print(extract_action("a", action_mapping_0))
if __name__ == "__main__":
    main()