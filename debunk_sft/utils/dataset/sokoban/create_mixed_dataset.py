"""
This script is used to create a mixed prompt dataset (varied action space).
Randomly converts half of the datapoints' prompts by replacing the old action
space with a new action space, and maps the responses accordingly.
"""

import pandas as pd
import copy
import re
from debunk_sft.env.sokoban.env import SokobanEnvReil
from datasets import Dataset
import numpy as np
qwen_response_template = """\
</think> <answer> {action} </answer>
"""


# Local lookup tables for mapping action ids to vocabulary for each space
BASE_ACTION_LOOKUP = {
    0: "None",
    1: "Up",
    2: "Down",
    3: "Left",
    4: "Right",
}
CARDINAL_ACTION_LOOKUP = {
    0: "None",
    1: "North",
    2: "South",
    3: "West",
    4: "East",
}
EMOJI_ACTION_LOOKUP = {
    0: "None",
    1: "⬆️",
    2: "⬇️",
    3: "⬅️",
    4: "➡️",
}
NUMERICAL_ACTION_LOOKUP = {
    0: "None",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
}
ALPHABETICAL_ACTION_LOOKUP = {
    0: "None",
    1: "A",
    2: "B",
    3: "C",
    4: "D",
}
RANDOM_ACTION_LOOKUP = {
    0: "None",
    1: "*",
    2: "&",
    3: "1",
    4: "M",
}


def _normalize_space_name(name: str | None) -> str:
    if not name:
        return "base"
    name_l = name.strip().lower()
    alias = {
        "base": "base",
        "udlr": "base",
        "original": "base",
        "cardinal": "cardinal",
        "news": "cardinal",
        "emoji": "emoji",
        "empty": "empty",
        "numerical": "numerical",
        "num": "numerical",
        "alphabetical": "alphabetical",
        "alpha": "alphabetical",
        "random": "random",
    }
    return alias.get(name_l, name_l)


def _get_action_lookup(space: str):
    space = _normalize_space_name(space)
    if space == "base":
        # Prefer env's lookup if present, else local default
        return getattr(SokobanEnvReil, "ACTION_LOOKUP", BASE_ACTION_LOOKUP)
    if space == "cardinal":
        return CARDINAL_ACTION_LOOKUP
    if space == "emoji":
        return EMOJI_ACTION_LOOKUP
    if space == "empty":
        # Fallback to base mapping for response vocabulary if needed
        return getattr(SokobanEnvReil, "ACTION_LOOKUP", BASE_ACTION_LOOKUP)
    if space == "numerical":
        return NUMERICAL_ACTION_LOOKUP
    if space == "alphabetical":
        return ALPHABETICAL_ACTION_LOOKUP
    if space == "random":
        return RANDOM_ACTION_LOOKUP
    raise ValueError(f"Unsupported action space: {space}")


def _answers_block(space: str) -> str:
    space = _normalize_space_name(space)
    if space == "base":
        return (
            "Answers:\n"
            "<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>\n"
        )
    if space == "cardinal":
        return (
            "Answers:\n"
            "<answer> North </answer> | <answer> South </answer> | <answer> West </answer> | <answer> East </answer>\n"
        )
    if space == "emoji":
        return (
            "Answers:\n"
            "<answer> ⬆️ </answer> | <answer> ⬇️ </answer> | <answer> ⬅️ </answer> | <answer> ➡️ </answer>\n"
        )
    if space == "empty":
        return ""
    if space == "numerical":
        return (
            "Answers:\n"
            "<answer> 1 </answer> | <answer> 2 </answer> | <answer> 3 </answer> | <answer> 4 </answer>\n"
            "where 1 is Up, 2 is Down, 3 is Left, 4 is Right.\n"
        )
    if space == "alphabetical":
        return (
            "Answers:\n"
            "<answer> A </answer> | <answer> B </answer> | <answer> C </answer> | <answer> D </answer>\n"
            "where A is Up, B is Down, C is Left, D is Right.\n"
        )
    if space == "random":
        return (
            "Answers:\n"
            "<answer> * </answer> | <answer> & </answer> | <answer> 1 </answer> | <answer> M </answer>\n"
            "where * is Up, & is Down, 1 is Left, M is Right.\n"
        )
    raise ValueError(f"Unsupported action space for answers block: {space}")


def _replace_action_space_in_prompt(prompt_text: str, target_space: str) -> str:
    """
    Replace the Answers section in the Sokoban instruction prompt with the
    target action space's answer options. If target is 'empty', remove the
    Answers section entirely.
    """
    target_space = _normalize_space_name(target_space)

    # Pattern to capture from 'Answers:' up to the line before 'Rewards:'
    pattern = r"(Answers:\n[\s\S]*?)\n\s*Rewards:"
    answers = _answers_block(target_space)

    if target_space == "empty":
        # Remove the Answers block if present
        def _rem_empty(m):
            return "Rewards:"

        new_text, n = re.subn(pattern, _rem_empty, prompt_text, flags=re.IGNORECASE)
        return new_text if n > 0 else prompt_text

    # Replace or insert Answers block before Rewards
    def _repl(m):
        return f"{answers}\n\nRewards:"

    new_text, n = re.subn(pattern, _repl, prompt_text, flags=re.IGNORECASE)
    if n == 0:
        # If original had no Answers block (e.g., 'empty'), try inserting before Rewards:
        rewards_pat = r"\n\s*Rewards:"
        new_text, n = re.subn(rewards_pat, f"\n{answers}\n\nRewards:", prompt_text, flags=re.IGNORECASE)
        return new_text if n > 0 else prompt_text
    return new_text

def convert_to_mixed_data(
    data_file: str,
    original_action_space: str | None = None,
    new_action_spaces: list[str] | None = None,
    seed: int = 42,
):
    # Set seed for reproducible results
    np.random.seed(seed)
    
    df = pd.read_parquet(data_file)
    assert 'data_source' in df.columns, "data_source is required"
    assert 'prompt' in df.columns, "prompt is required"
    assert 'reward_model' in df.columns, "reward_model is required"
    instance_template = {
            "data_source": None,
            "prompt": None,
            "response": None,
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": None}
        }
    sft_instances = []
    rl_instances = []
    data_length = len(df)
    
    # Normalize action space names and defaults
    orig_space = _normalize_space_name(original_action_space)
    
    # Handle new action spaces - if None, default to cardinal; if single string, convert to list
    if new_action_spaces is None:
        new_action_spaces = ["cardinal"]
    elif isinstance(new_action_spaces, str):
        new_action_spaces = [new_action_spaces]
    
    # Normalize all new action spaces
    new_action_spaces = [_normalize_space_name(space) for space in new_action_spaces]
    n = len(new_action_spaces)
    
    # Calculate distribution: n/(n+1) for new action spaces, 1/(n+1) for original
    total_new_samples = int(data_length * n / (n + 1))
    samples_per_new_space = int(data_length / (n + 1))
    original_samples = data_length - total_new_samples
    
    # Handle remainder from integer division - assign to original action space
    remainder = data_length - (original_samples + samples_per_new_space * n)
    original_samples += remainder  # Add remainder to original samples
    
    print(f"Dataset distribution:")
    print(f"  Total samples: {data_length}")
    print(f"  Original action space ({orig_space}): {original_samples} samples (includes {remainder} remainder)")
    print(f"  New action spaces: {total_new_samples} samples total")
    for i, space in enumerate(new_action_spaces):
        print(f"    {space}: {samples_per_new_space} samples")
    
    # Randomly shuffle indices for fair distribution
    all_indices = np.random.permutation(data_length)
    
    # Assign indices to each action space
    current_idx = 0
    
    # Original action space gets the first 1/(n+1) portion + remainder
    original_indices = set(all_indices[current_idx:current_idx + original_samples])
    current_idx += original_samples
    
    # Each new action space gets exactly 1/(n+1) portion
    action_space_assignments = {}
    for space in new_action_spaces:
        space_indices = set(all_indices[current_idx:current_idx + samples_per_new_space])
        for idx in space_indices:
            action_space_assignments[idx] = space
        current_idx += samples_per_new_space
    
    # Debug: Verify all samples are assigned
    total_assigned = len(original_indices) + len(action_space_assignments)
    print(f"  Verification: {total_assigned} samples assigned out of {data_length}")
    if total_assigned != data_length:
        print(f"  WARNING: Sample count mismatch! Expected {data_length}, got {total_assigned}")
    
    # Safety check: Ensure all positions are covered
    all_positions = set(range(data_length))
    covered_positions = original_indices.union(set(action_space_assignments.keys()))
    uncovered_positions = all_positions - covered_positions
    if uncovered_positions:
        print(f"  WARNING: Uncovered positions: {len(uncovered_positions)}")
        print(f"  First few uncovered: {list(uncovered_positions)[:10]}")

    for pos, (index, row) in enumerate(df.iterrows()):
        sft_instance = copy.deepcopy(instance_template)
        rl_instance = copy.deepcopy(instance_template)
        sft_instance['data_source'] = row['data_source']
        rl_instance['data_source'] = row['data_source']
        sft_instance['reward_model'] = row['reward_model'].copy()
        rl_instance['reward_model'] = row['reward_model'].copy()
        sft_instance['extra_info'] = row['extra_info'].copy()
        rl_instance['extra_info'] = row['extra_info'].copy()
        
        rl_instance['prompt'] = row['prompt'].copy()
        assert len(row['prompt']) == 1, "prompt should be a list with one element"
        assert 'content' in row['prompt'][0]
        prompt_text = row['prompt'][0]['content']
        
        action = row['reward_model']['ground_truth'][:]
        if "sokoban" in row['data_source']:
            # Determine which action space to use for this datapoint
            if pos in original_indices:
                target_space = orig_space
            elif pos in action_space_assignments:
                target_space = action_space_assignments[pos]
            else:
                # Safety fallback: assign to original space if somehow not assigned
                print(f"  WARNING: Position {pos} not assigned to any action space, using original")
                target_space = orig_space

            # Update prompt to target action space
            sft_instance['prompt'] = _replace_action_space_in_prompt(prompt_text, target_space)
            rl_instance['prompt'][0]['content'] = _replace_action_space_in_prompt(prompt_text, target_space)

            # Map response to correct action vocabulary
            lookup = _get_action_lookup(target_space)
            action_str = " ".join([lookup[a] for a in action])
            sft_instance['data_source'] = f"{sft_instance['data_source']}_{target_space}"
            rl_instance['data_source'] = f"{rl_instance['data_source']}_{target_space}"
        else:
            # Non-sokoban data left untouched
            sft_instance['prompt'] = prompt_text
            rl_instance['prompt'][0]['content'] = prompt_text
            if isinstance(action, list):
                action_str = " ".join([a for a in action])
            else:
                action_str = action
        sft_instance['response'] = qwen_response_template.format(action=action_str)
        rl_instance['response'] = qwen_response_template.format(action=action_str)
        sft_instances.append(sft_instance)
        rl_instances.append(rl_instance)

    return sft_instances, rl_instances


def main():
    # Train dataset: mix of 3 new action spaces (cardinal, emoji, numerical)
    # Distribution: 3/4 to new action spaces, 1/4 to original
    # Each new action space gets 1/4 of the dataset
    print("Creating train dataset with 3 new action spaces...")
    train_sft_instances, train_rl_instances = convert_to_mixed_data(
        "./data/sokoban_one_horizon_large_envs/train.parquet",
        original_action_space="base",
        new_action_spaces=["cardinal", "alphabetical", "numerical"],
        seed=42,
    )
    
    # Test dataset: mix of 2 new action spaces (alphabetical, random)
    # Distribution: 2/3 to new action spaces, 1/3 to original
    # Each new action space gets 1/3 of the dataset
    print("\nCreating test dataset with 2 new action spaces...")
    test_sft_instances, test_rl_instances = convert_to_mixed_data(
        "./data/sokoban_one_horizon_large_envs/test.parquet",
        original_action_space="base",
        new_action_spaces=["emoji", "random"],
        seed=42,
    )
    
    train_sft_dataset = Dataset.from_list(train_sft_instances)
    test_sft_dataset = Dataset.from_list(test_sft_instances)
    train_rl_dataset = Dataset.from_list(train_rl_instances)
    test_rl_dataset = Dataset.from_list(test_rl_instances)
    train_sft_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/ultradiverse/sft/train.parquet")
    test_sft_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/ultradiverse/sft/test.parquet")
    train_rl_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/ultradiverse/rl/train.parquet")
    test_rl_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/ultradiverse/rl/test.parquet")

    # Optional: push to hub with a different repo name if desired
    train_rl_dataset.push_to_hub("Xiaofeng77/sokoban_ultradiverse", split="train")
    test_rl_dataset.push_to_hub("Xiaofeng77/sokoban_ultradiverse", split="test")
    
if __name__ == "__main__":
    main()

    