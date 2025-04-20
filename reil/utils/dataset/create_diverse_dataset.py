# create behavior cloning dataset for sokoban agent
# adapted from ragen/sft/utils/generate_sft_verl_sokoban.py

from ragen.env.sokoban import SokobanEnv
from ragen.env.sokoban.room_utils import get_shortest_action_path
from datasets import Dataset
import os
import argparse

INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Answers:
<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Current Observation]:
{observation}
Decide the next {len_horizon} actions:\
"""

templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

def main():
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate behavior cloning dataset for Sokoban agent.")
    parser.add_argument("--num_boxes", type=int, default=1, help="Number of boxes (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum steps per episode (default: 10)")
    parser.add_argument("--search_depth", type=int, default=30, help="Maximum search depth for BFS (default: 30)")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'], 
                       help="Template prefix to use (default: qwen-instruct)")
    parser.add_argument("--train_size_each_instance", type=int, default=200, help="Number of train instances to generate (default: 200)")
    parser.add_argument("--test_size_each_instance", type=int, default=60, help="Number of test instances to generate (default: 60)")
    parser.add_argument("--num_test_envs", type=int, default=200, help="Number of test environments(default: 200)")
    parser.add_argument("--output", type=str, default='./data/sokoban_diverse', help="Output directory (default: ./data/sokoban_diverse)")
    # parser.add_argument("--is_push_to_hub", type=bool, default=True, help="Whether to push to hub (default: True)")
    parser.add_argument("--push_to_hub", action='store_true', help="Push to hub (default: False)")
    parser.add_argument("--hf_name", type=str, default='sokoban')

    args = parser.parse_args()

    # Extract arguments
    # train_dim_x = [6, 8, 10]
    # train_dim_y = [6, 8, 10]
    # test_dim_x = [7, 9]
    # test_dim_y = [7, 9]
    # horizons = [1, 2, 3, 4]
    train_dim_x = [6]
    train_dim_y = [6]
    test_dim_x = [5, 6, 7, 8]
    test_dim_y = [5, 6, 7, 8]
    horizons = [1]
    assert len(train_dim_x) == len(train_dim_y), "train_dim_x and train_dim_y must have the same length"
    assert len(test_dim_x) == len(test_dim_y), "test_dim_x and test_dim_y must have the same length"
    print(f"We will have {len(train_dim_x) * len(train_dim_y) * len(horizons) * args.train_size_each_instance} train instances in total.\n")
    print(f"We will have {len(test_dim_x) * len(test_dim_y) * len(horizons) * args.test_size_each_instance} test instances in total.\n")

    num_boxes = args.num_boxes
    seed = args.seed
    max_steps = args.max_steps
    search_depth = args.search_depth
    prefix = args.prefix
    train_envs = [SokobanEnv(dim_room=(dim_x, dim_y), num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth) for dim_x, dim_y in zip(train_dim_x, train_dim_y)]
    test_envs = [SokobanEnv(dim_room=(dim_x, dim_y), num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth) for dim_x, dim_y in zip(test_dim_x, test_dim_y)]
    # Create training instances
    train_instances = []
    for seed_train in range(seed, seed + args.train_size_each_instance):
        for env in train_envs:
            for len_horizon in horizons:
                obs = env.reset(seed=seed_train)
                gt_action_sequence = get_shortest_action_path(env.room_fixed, env.room_state, MAX_DEPTH=search_depth)
                if gt_action_sequence is None:
                    print(f"No action sequence found for seed {seed_train}")
                    continue
        
                if len_horizon > len(gt_action_sequence):
                    print(f"No enough actions for seed {seed_train}, skip")
                    continue
                
                for i in range(len(gt_action_sequence) - len_horizon + 1):
                    instruction = templates[prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs, len_horizon=len_horizon))
                    action_sequence = gt_action_sequence[i:i+len_horizon]
                    train_instances.append({
                        'instruction': instruction,
                        'gt_action': action_sequence,
                        'horizon': len_horizon,
                        'room_dim': (env.dim_room[0], env.dim_room[1])
                    })
            
                    obs, reward, done, info = env.step(action_sequence[0])

    # Create test instances 
    test_instances = []
    for seed_test in range(seed+args.train_size_each_instance, seed+args.train_size_each_instance+args.test_size_each_instance):
        for env in test_envs:
            for len_horizon in horizons:
                obs = env.reset(seed=seed_test)
                gt_action_sequence = get_shortest_action_path(env.room_fixed, env.room_state, MAX_DEPTH=search_depth)
                if gt_action_sequence is None:
                    print(f"No action sequence found for seed {seed_test}")
                    continue
        
                if len_horizon > len(gt_action_sequence):
                    print(f"No enough actions for seed {seed_test}, skip")
                    continue
                
                for i in range(len(gt_action_sequence) - len_horizon + 1):
                    instruction = templates[prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs, len_horizon=len_horizon))
                    action_sequence = gt_action_sequence[i:i+len_horizon]
                    test_instances.append({
                        'instruction': instruction,
                        'gt_action': action_sequence,
                        'horizon': len_horizon,
                        'room_dim': (env.dim_room[0], env.dim_room[1])
                    })
            
                    obs, reward, done, info = env.step(action_sequence[0])
    
    # Create test instances for each test environment
    test_env_instances = []
    for seed_test_env in range(seed+args.train_size_each_instance+args.test_size_each_instance, seed+args.train_size_each_instance+args.test_size_each_instance+args.num_test_envs):
        obs = test_envs[0].reset(seed=seed_test_env)
        instruction = templates[prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs, len_horizon=len_horizon))
        test_env_instances.append(
            {
                'instruction': instruction,
                'seed': seed_test_env
            }
        )

    def _create_instance(idx, instance):
        prompt_formatted = instance['instruction']
        gt_action = instance['gt_action']

        return {
            "data_source": f"sokoban_{instance['room_dim'][0]}x{instance['room_dim'][1]}_{instance['horizon']}horizon",
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": gt_action},
            "extra_info": {"split": "train", "index": idx}
        }
    
    def _create_test_env_instance(idx, instance):
        prompt_formatted = instance['instruction']

        return {
            "data_source": "sokoban",
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": [-1]},
            "extra_info": {"split": "test", "index": idx}
        }

    train_dataset = Dataset.from_list([_create_instance(i, train_instances[i]) for i in range(len(train_instances))])
    test_dataset = Dataset.from_list([_create_instance(i, test_instances[i]) for i in range(len(test_instances))])
    # test_env_dataset = Dataset.from_list([_create_test_env_instance(args.seed + i, test_env_instances[i-args.train_size_each_instance-args.test_size_each_instance]) for i in range(args.train_size_each_instance+args.test_size_each_instance, args.train_size_each_instance+args.test_size_each_instance+args.num_test_envs)])
    test_env_dataset = Dataset.from_list([_create_test_env_instance(instance['seed'], instance) for instance in test_env_instances])
    def make_map_fn(split):
        def process_fn(example, idx):
            # Add split information to each example
            example['extra_info']['split'] = split  # Currently in your code
            if split == 'train':
                # Apply training-specific transformations
                example['id'] = f"train_{idx}"
            elif split == 'test':
                # Apply test-specific transformations
                example['id'] = f"test_{idx}"
            elif split == 'test_env':
                # Apply test-specific transformations
                example['id'] = f"test_env_{idx}"
                
            return example
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))
    test_env_dataset = test_env_dataset.map(function=make_map_fn('test_env'), with_indices=True)
    test_env_dataset.to_parquet(os.path.join(args.output, 'test_env.parquet'))
    # push to hub
    if args.push_to_hub:
        train_dataset.push_to_hub("Xiaofeng77/"+args.hf_name, split="train")
        test_dataset.push_to_hub("Xiaofeng77/"+args.hf_name, split="test")
        test_env_dataset.push_to_hub("Xiaofeng77/"+args.hf_name, split="test_env")  

if __name__ == "__main__":
    main()