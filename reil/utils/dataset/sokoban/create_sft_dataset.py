import pandas as pd
import copy
from reil.env.sokoban.env import SokobanEnvReil
from datasets import Dataset
from pprint import pprint
qwen_response_template = """\
</think> <answer> {action} </answer>
"""

def convert_parquet_to_sft_data(data_file: str):
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

def convert_jsonl_to_sft_data(data_file: str):
    df = pd.read_json(data_file, lines=True)
    instance_template = {
            "data_source": None,
            "prompt": None,
            "response": None,
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": None}
        }
    instances = []
    for idx, row in df.iterrows():
        pprint(row)
        instance = copy.deepcopy(instance_template)
        instance['data_source'] = row['data_source'] if 'data_source' in row else 'sokoban'
        instance['prompt'] = row['prompt']
        instance['response'] = row['response']
        instance['extra_info']['index'] = row['question_id']
        instance["reward_model"] = row['answer']
        instances.append(instance)
    return instances

def main():
    train_instances = convert_jsonl_to_sft_data("./results/sokoban_one_horizon_large_envs-train-temp_1.0-top_p_1.0-top_k_-1-20250903_175938.jsonl")
    # test_instances = convert_parquet_to_sft_data("./data/sokoban_one_horizon_large_envs/test.parquet")
    train_dataset = Dataset.from_list(train_instances)
    # test_dataset = Dataset.from_list(test_instances)
    train_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/cot-sft/train.parquet")
    # test_dataset.to_parquet("./data/sokoban_one_horizon_large_envs/sft/test.parquet")
    train_dataset.push_to_hub("Xiaofeng77/Qwen3-8b-cot-sokoban", split="train")
    # train_dataset.push_to_hub("Xiaofeng77/reil_sokoban_one_horizon_large_envs_sft", split="train")
    # test_dataset.push_to_hub("Xiaofeng77/reil_sokoban_one_horizon_large_envs_sft", split="test")
    
if __name__ == "__main__":
    main()

    