import pandas as pd
import copy
from reil.env.sokoban.env import SokobanEnvReil
from datasets import Dataset
qwen_response_template = """\
</think> <answer> {action} </answer> <|im_end|>
"""

def convert_to_sft_data(data_file: str):
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
        if "sokoban" in row['data_source']:
            action_str = " ".join([SokobanEnvReil.ACTION_LOOKUP[a] for a in action])
        else:
            if type(action) == list:
                action_str = " ".join([a for a in action])
            else:
                action_str = action
        instance['response'] = qwen_response_template.format(action=action_str)
        instances.append(instance)

    return instances


def main():
    train_instances = convert_to_sft_data("./data/small_sokoban/train.parquet")
    test_instances = convert_to_sft_data("./data/small_sokoban/test.parquet")
    train_dataset = Dataset.from_list(train_instances)
    test_dataset = Dataset.from_list(test_instances)
    train_dataset.to_parquet("./data/small_sokoban/sft/train.parquet")
    test_dataset.to_parquet("./data/small_sokoban/sft/test.parquet")

    train_dataset.push_to_hub("Xiaofeng77/reil_small_sokoban_sft", split="train")
    test_dataset.push_to_hub("Xiaofeng77/reil_small_sokoban_sft", split="test")
    
if __name__ == "__main__":
    main()

    