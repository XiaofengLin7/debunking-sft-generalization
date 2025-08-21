"""
Simple word sampling for creating random action spaces.
"""

import numpy as np
import pandas as pd
import copy
from typing import List, Dict
from datasets import Dataset
from .action_space_shared import (
    create_answers_block,
    replace_answers_in_prompt,
    convert_actions_to_vocabulary,
    create_standard_action_mapping,
    validate_action_mapping
)

qwen_response_template = """\
</think> <answer> {action} </answer>
"""


def get_word_pool() -> List[str]:
    """Get a pool of words for sampling."""
    return [
        # Animals
        "cat", "dog", "bird", "fish", "lion", "tiger", "bear", "wolf", "fox", "deer",
        "eagle", "hawk", "owl", "duck", "swan", "frog", "snake", "turtle", "rabbit", "mouse",
        "elephant", "giraffe", "zebra", "monkey", "panda", "koala", "dolphin", "whale", "shark", "butterfly",
        
        # Nature
        "tree", "flower", "leaf", "rock", "stone", "water", "fire", "wind", "earth", "sun",
        "moon", "star", "cloud", "rain", "snow", "mountain", "valley", "river", "ocean", "forest",
        "desert", "island", "beach", "cave", "volcano", "meadow", "field", "garden", "pond", "lake",
        
        # Colors
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white",
        "gray", "silver", "gold", "copper", "bronze", "maroon", "navy", "teal", "lime", "olive",
        "aqua", "crimson", "scarlet", "amber", "jade", "ruby", "sapphire", "emerald", "pearl", "coral",
        
        # Objects
        "book", "pen", "cup", "key", "phone", "chair", "table", "lamp", "door", "window",
        "box", "bag", "hat", "shoe", "clock", "mirror", "brush", "knife", "fork", "spoon",
        "plate", "bowl", "bottle", "glass", "camera", "computer", "keyboard", "mouse", "screen", "speaker",
        
        # Food
        "apple", "banana", "orange", "grape", "cherry", "strawberry", "peach", "pear", "lemon", "lime",
        "bread", "cheese", "milk", "egg", "meat", "fish", "rice", "pasta", "pizza", "burger",
        "salad", "soup", "cake", "cookie", "candy", "chocolate", "honey", "sugar", "salt", "pepper",
        
        # Actions
        "run", "jump", "fly", "swim", "walk", "dance", "sing", "laugh", "cry", "sleep",
        "wake", "eat", "drink", "play", "work", "study", "read", "write", "draw", "paint",
        "build", "create", "fix", "open", "close", "start", "stop", "move", "push", "pull",
        
        # Abstract
        "happy", "quick", "bright", "calm", "strong", "wise", "kind", "brave", "gentle", "fierce",
        "magic", "power", "energy", "force", "spirit", "dream", "hope", "love", "peace", "freedom",
        "truth", "beauty", "wonder", "mystery", "secret", "destiny", "luck", "chance", "victory", "glory"
    ]


def sample_random_words(n_words: int, seed: int = None) -> List[str]:
    """
    Sample random words from the word pool.
    
    Args:
        n_words: Number of words to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of randomly sampled words
    """
    if seed is not None:
        np.random.seed(seed)
    
    word_pool = get_word_pool()
    
    # Sample without replacement if possible
    if n_words <= len(word_pool):
        return list(np.random.choice(word_pool, size=n_words, replace=False))
    else:
        # Sample with replacement if we need more words than available
        return list(np.random.choice(word_pool, size=n_words, replace=True))





def process_dataset_sample(sample_data: dict, words: List[str]) -> dict:
    """
    Process a single dataset sample by replacing action template and converting actions.
    
    Args:
        sample_data: Original dataset sample
        words: 4 random words for action space
        
    Returns:
        Updated sample with new action template and converted actions
    """
    # Create action mapping using shared utility
    action_mapping = create_standard_action_mapping(words)
    validate_action_mapping(action_mapping)
    
    # Create answers block using shared utility
    answers_block = create_answers_block(action_mapping, add_explanation=True)
    
    # Get original prompt
    prompt_content = sample_data['prompt'][0]['content']
    
    # Replace answers block in prompt using shared utility
    new_prompt = replace_answers_in_prompt(prompt_content, answers_block)
    
    # Convert expert actions using shared utility
    original_actions = sample_data['reward_model']['ground_truth']
    new_actions = convert_actions_to_vocabulary(original_actions, action_mapping)
    
    # Create updated sample
    updated_sample = sample_data.copy()
    updated_sample['prompt'] = [{"role": "user", "content": new_prompt}]
    updated_sample['converted_actions'] = new_actions
    updated_sample['action_words'] = words
    updated_sample['action_mapping'] = action_mapping
    
    return updated_sample


def create_rl_sft_datasets(data_file: str, seed: int = 42):
    """
    Create both RL and SFT datasets with randomly generated action spaces.
    Each data point gets unique randomly sampled words for action space.
    
    Args:
        data_file: Path to input parquet file
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (sft_instances, rl_instances)
    """
    # Read the original dataset
    df = pd.read_parquet(data_file)
    print(f"Processing {len(df)} samples from {data_file}")
    
    # Instance template - exact format from create_mixed_dataset.py
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
    
    # Track action space diversity
    unique_action_spaces = set()
    
    for pos, (index, row) in enumerate(df.iterrows()):
        # Generate unique random words for this sample
        words = sample_random_words(4, seed=seed + pos)
        unique_action_spaces.add(tuple(words))
        
        # Create instances using exact format from create_mixed_dataset.py
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
            # Create action mapping and update prompt
            action_mapping = create_standard_action_mapping(words)
            answers_block = create_answers_block(action_mapping, add_explanation=True)
            updated_prompt = replace_answers_in_prompt(prompt_text, answers_block)
            
            # Update prompts
            sft_instance['prompt'] = updated_prompt
            rl_instance['prompt'][0]['content'] = updated_prompt
            
            # Map response to random words
            action_str = convert_actions_to_vocabulary(action, action_mapping)
            sft_instance['data_source'] = f"{sft_instance['data_source']}_random"
            rl_instance['data_source'] = f"{rl_instance['data_source']}_random"
        else:
            # Non-sokoban data left untouched
            sft_instance['prompt'] = prompt_text
            rl_instance['prompt'][0]['content'] = prompt_text
            if isinstance(action, list):
                action_str = " ".join([str(a) for a in action])
            else:
                action_str = str(action)
        
        sft_instance['response'] = qwen_response_template.format(action=action_str)
        rl_instance['response'] = qwen_response_template.format(action=action_str)
        sft_instances.append(sft_instance)
        rl_instances.append(rl_instance)
    
    # Report diversity
    print(f"Generated {len(unique_action_spaces)} unique action spaces out of {len(df)} samples")
    print(f"Uniqueness ratio: {len(unique_action_spaces)/len(df):.3f}")
    
    return sft_instances, rl_instances


def save_datasets(sft_instances: List[dict], rl_instances: List[dict], output_dir: str, split: str):
    """
    Save SFT and RL datasets to parquet files.
    
    Args:
        sft_instances: List of SFT instances
        rl_instances: List of RL instances
        output_dir: Output directory path
    """
    # Create datasets
    sft_dataset = Dataset.from_list(sft_instances)
    rl_dataset = Dataset.from_list(rl_instances)
    
    # Save to files
    sft_dataset.to_parquet(f"{output_dir}/sft/{split}.parquet")
    rl_dataset.to_parquet(f"{output_dir}/rl/{split}.parquet")
    
    print(f"Saved {len(sft_instances)} SFT instances to {output_dir}/sft/{split}.parquet")
    print(f"Saved {len(rl_instances)} RL instances to {output_dir}/rl/{split}.parquet")
    
    return sft_dataset, rl_dataset


def main():
    sft_train_instances, rl_train_instances = create_rl_sft_datasets("./data/sokoban_one_horizon_large_envs/train.parquet", seed=42)
    sft_test_instances, rl_test_instances = create_rl_sft_datasets("./data/sokoban_one_horizon_large_envs/test.parquet", seed=42)
    save_datasets(sft_train_instances, rl_train_instances, "./data/sokoban_one_horizon_large_envs/super_random", "train")
    save_datasets(sft_test_instances, rl_test_instances, "./data/sokoban_one_horizon_large_envs/super_random", "test")

    #push to hub
    rl_train_dataset = Dataset.from_list(rl_train_instances)
    rl_test_dataset = Dataset.from_list(rl_test_instances)
    rl_train_dataset.push_to_hub("Xiaofeng77/sokoban_super_random", split="train")
    rl_test_dataset.push_to_hub("Xiaofeng77/sokoban_super_random", split="test")

if __name__ == "__main__":
    main()