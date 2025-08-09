"""
This script is used to create a mixed prompt dataset (varied action space).
Randomly converts half of the datapoints' prompts by replacing the old action
space with a new action space, and maps the responses accordingly.
"""

import pandas as pd
import copy
import re
from reil.env.sokoban.env import SokobanEnvReil
from datasets import Dataset
import numpy as np
qwen_response_template = """\
</think> <answer> {action} </answer> <|im_end|>
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
    new_action_space: str | None = None,
):
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
    # Randomly choose half indices to convert
    convert_idx = np.random.permutation(data_length)
    convert_set = set(convert_idx[: data_length // 2])

    # Normalize action space names and defaults
    orig_space = _normalize_space_name(original_action_space)
    new_space = _normalize_space_name(new_action_space or "cardinal")

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
            # Whether to convert this datapoint's prompt to new action space
            to_convert = pos in convert_set
            target_space = new_space if to_convert else orig_space

            # Update prompt to target action space
            sft_instance['prompt'] = _replace_action_space_in_prompt(prompt_text, target_space)
            rl_instance['prompt'][0]['content'] = _replace_action_space_in_prompt(prompt_text, target_space)
            # print(instance['prompt'] if to_convert else "")

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
    train_sft_instances, train_rl_instances = convert_to_mixed_data(
        "./data/sokoban_one_horizon_large_envs/train.parquet",
        original_action_space="base",
        new_action_space="cardinal",
    )
    test_sft_instances, test_rl_instances = convert_to_mixed_data(
        "./data/sokoban_one_horizon_large_envs/test.parquet",
        original_action_space="base",
        new_action_space="emoji",
    )
    train_sft_dataset = Dataset.from_list(train_sft_instances)
    test_sft_dataset = Dataset.from_list(test_sft_instances)
    train_rl_dataset = Dataset.from_list(train_rl_instances)
    test_rl_dataset = Dataset.from_list(test_rl_instances)
    train_sft_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/mixed/sft/train.parquet")
    test_sft_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/mixed/sft/test.parquet")
    train_rl_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/mixed/rl/train.parquet")
    test_rl_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/mixed/rl/test.parquet")

    # Optional: push to hub with a different repo name if desired
    # train_dataset.push_to_hub("Xiaofeng77/reil_small_sokoban_mixed", split="train")
    # test_dataset.push_to_hub("Xiaofeng77/reil_small_sokoban_mixed", split="test")
    
if __name__ == "__main__":
    main()

    