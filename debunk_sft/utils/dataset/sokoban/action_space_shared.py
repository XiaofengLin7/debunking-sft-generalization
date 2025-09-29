"""
Shared utilities for action space management in Sokoban dataset creation.
Common functions used by both create_mixed_dataset.py and create_ultra_mixed_dataset.py
"""

import re
from typing import Dict, List


def create_answers_block(action_mapping: Dict[int, str], add_explanation: bool = True) -> str:
    """
    Create answers block from action mapping.
    
    Args:
        action_mapping: Dictionary mapping action indices to words/symbols
        add_explanation: Whether to add explanation of mappings
        
    Returns:
        Formatted answers block string
    """
    # Get the 4 direction actions (indices 1-4)
    up_action = action_mapping[1]
    down_action = action_mapping[2] 
    left_action = action_mapping[3]
    right_action = action_mapping[4]
    
    # Create answer tags
    answer_tags = [
        f"<answer> {up_action} </answer>",
        f"<answer> {down_action} </answer>", 
        f"<answer> {left_action} </answer>",
        f"<answer> {right_action} </answer>"
    ]
    
    answers_line = "Answers:\n" + " | ".join(answer_tags) + "\n"
    
    # Add explanation if requested and not using standard directional words
    if add_explanation:
        standard_words = {"Up", "Down", "Left", "Right", "North", "South", "West", "East"}
        if not any(action in standard_words for action in [up_action, down_action, left_action, right_action]):
            explanation = f"where {up_action} is Up, {down_action} is Down, {left_action} is Left, {right_action} is Right.\n"
            answers_line += explanation
    
    return answers_line


def replace_answers_in_prompt(prompt_text: str, new_answers_block: str) -> str:
    """
    Replace the Answers section in the prompt with a new answers block.
    
    Args:
        prompt_text: Original prompt text
        new_answers_block: New answers block to insert
        
    Returns:
        Updated prompt text
    """
    # Pattern to capture from 'Answers:' up to the line before 'Rewards:'
    pattern = r"(Answers:\n[\s\S]*?)\n\s*Rewards:"
    
    def replacement(match):
        return f"{new_answers_block}\nRewards:"
    
    new_text, n = re.subn(pattern, replacement, prompt_text, flags=re.IGNORECASE)
    
    if n == 0:
        # If no Answers block found, try to insert before Rewards
        rewards_pattern = r"\n\s*Rewards:"
        new_text, n = re.subn(rewards_pattern, f"\n{new_answers_block}\nRewards:", prompt_text, flags=re.IGNORECASE)
        if n == 0:
            print("Warning: Could not replace answers block in prompt")
            return prompt_text
    
    return new_text


def convert_actions_to_vocabulary(action_sequence: List[int], action_mapping: Dict[int, str]) -> str:
    """
    Convert action sequence from indices to vocabulary string.
    
    Args:
        action_sequence: List of action indices
        action_mapping: Mapping from indices to action vocabulary
        
    Returns:
        Space-separated string of action words
    """
    return " ".join([action_mapping[action] for action in action_sequence])


def create_standard_action_mapping(action_words: List[str]) -> Dict[int, str]:
    """
    Create standard action mapping from 4 action words.
    
    Args:
        action_words: List of 4 words for [Up, Down, Left, Right]
        
    Returns:
        Dictionary mapping {0: "None", 1: Up, 2: Down, 3: Left, 4: Right}
    """
    if len(action_words) != 4:
        raise ValueError("Must provide exactly 4 action words for [Up, Down, Left, Right]")
    
    return {
        0: "None",
        1: action_words[0],  # Up
        2: action_words[1],  # Down
        3: action_words[2],  # Left
        4: action_words[3]   # Right
    }


def validate_action_mapping(action_mapping: Dict[int, str]) -> bool:
    """
    Validate that action mapping has required keys and structure.
    
    Args:
        action_mapping: Action mapping to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_keys = {0, 1, 2, 3, 4}
    if set(action_mapping.keys()) != required_keys:
        raise ValueError(f"Action mapping must have keys {required_keys}, got {set(action_mapping.keys())}")
    
    if action_mapping[0] != "None":
        raise ValueError("Action mapping[0] must be 'None'")
    
    # Check for unique non-None values
    action_values = [action_mapping[i] for i in range(1, 5)]
    if len(set(action_values)) != 4:
        raise ValueError("Action values for indices 1-4 must be unique")
    
    return True


if __name__ == "__main__":
    # Test the shared utilities
    print("Testing shared action space utilities:")
    
    # Test 1: Standard directional mapping
    print("\n1. Testing standard directional mapping:")
    standard_mapping = create_standard_action_mapping(["Up", "Down", "Left", "Right"])
    print(f"Mapping: {standard_mapping}")
    validate_action_mapping(standard_mapping)
    print("✓ Validation passed")
    
    answers_block = create_answers_block(standard_mapping, add_explanation=False)
    print(f"Answers block:\n{answers_block}")
    
    # Test 2: Custom word mapping
    print("\n2. Testing custom word mapping:")
    custom_mapping = create_standard_action_mapping(["cat", "dog", "bird", "fish"])
    print(f"Mapping: {custom_mapping}")
    validate_action_mapping(custom_mapping)
    print("✓ Validation passed")
    
    custom_answers_block = create_answers_block(custom_mapping, add_explanation=True)
    print(f"Answers block:\n{custom_answers_block}")
    
    # Test 3: Action conversion
    print("\n3. Testing action conversion:")
    test_actions = [1, 4, 2]  # Up, Right, Down
    standard_result = convert_actions_to_vocabulary(test_actions, standard_mapping)
    custom_result = convert_actions_to_vocabulary(test_actions, custom_mapping)
    print(f"Actions {test_actions} -> Standard: '{standard_result}'")
    print(f"Actions {test_actions} -> Custom: '{custom_result}'")
    
    # Test 4: Prompt replacement
    print("\n4. Testing prompt replacement:")
    sample_prompt = """You are a Sokoban solver.

Answers:
<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>

Rewards:
Move: -0.1"""
    
    updated_prompt = replace_answers_in_prompt(sample_prompt, custom_answers_block)
    print("Original answers: Up | Down | Left | Right")
    print(f"Updated answers: cat | dog | bird | fish")
    
    # Verify the replacement worked
    if "cat" in updated_prompt and "dog" in updated_prompt:
        print("✓ Prompt replacement successful")
    else:
        print("✗ Prompt replacement failed")
    
    print("\n✓ All shared utility tests passed!")
