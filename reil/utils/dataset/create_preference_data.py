"""
Create preference dataset for DPO training.
"""

import pandas as pd
import copy
from reil.env.sokoban.env import SokobanEnvReil
from datasets import Dataset
import random

qwen_response_template = """\
</think> <answer> {action} </answer> <|im_end|>
"""

def create_preference_data(data_file: str):
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
    instances = []
    for index, row in df.iterrows():
        instance = copy.deepcopy(instance_template)
        instance['data_source'] = row['data_source']
        assert len(row['prompt']) == 1, "prompt should be a list with one element"
        assert 'content' in row['prompt'][0]
        instance['prompt'] = row['prompt'][0]['content']
        
        instance['extra_info']['index'] = index
        action = row['reward_model']['ground_truth'][:]
        assert "sokoban" in row['data_source'], "Only sokoban data is supported"
        assert len(action) == 1, "action should be a list with one element"
        action_str = SokobanEnvReil.ACTION_LOOKUP[action[0]]
        negative_action = sample_negative_action(action[0])

        negative_action_str = SokobanEnvReil.ACTION_LOOKUP[negative_action]  
        instance['rejected'] = qwen_response_template.format(action=negative_action_str)
        instance['chosen'] = qwen_response_template.format(action=action_str)
        instances.append(instance)

    return instances

def sample_negative_action(pos_action: int):
    "randomly sample a negative sample in [1,2,3,4] beside pos_action"
    assert pos_action in [1,2,3,4], "pos_action should be in [1,2,3,4]"
    return random.choice([a for a in [1,2,3,4] if a != pos_action])

def main():
    random.seed(42)
    train_data = create_preference_data("./data/sokoban_one_horizon_large_envs/train.parquet")
    train_dataset = Dataset.from_list(train_data)
    train_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/dpo/train.parquet")

    train_dataset.push_to_hub("Xiaofeng77/reil_sokoban_preference", split="train")



if __name__ == "__main__":
    main()