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
    parser.add_argument("--dim_x", type=int, default=6, help="Room width (default: 6)")
    parser.add_argument("--dim_y", type=int, default=6, help="Room height (default: 6)")
    parser.add_argument("--num_boxes", type=int, default=1, help="Number of boxes (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max_steps", type=int, default=10, help="Maximum steps per episode (default: 10)")
    parser.add_argument("--search_depth", type=int, default=30, help="Maximum search depth for BFS (default: 30)")
    parser.add_argument("--len_horizon", type=int, default=1, help="Length of horizon (default: 1)")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'], 
                       help="Template prefix to use (default: qwen-instruct)")
    parser.add_argument("--train_size", type=int, default=1000, help="Number of train instances to generate (default: 1000)")
    parser.add_argument("--test_size", type=int, default=50, help="Number of test instances to generate (default: 50)")
    parser.add_argument("--num_test_envs", type=int, default=20, help="Number of test environments(default: 20)")
    parser.add_argument("--output", type=str, default='./data/sokoban', help="Output directory (default: ./data/sokoban)")
    args = parser.parse_args()

    # Extract arguments
    dim_x = args.dim_x
    dim_y = args.dim_y
    num_boxes = args.num_boxes
    seed = args.seed
    max_steps = args.max_steps
    search_depth = args.search_depth
    prefix = args.prefix
    len_horizon = args.len_horizon
    instances = []
    env = SokobanEnv(dim_room=(dim_x, dim_y), num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth)

    # Create training instances
    train_instances = []
    for seed_train in range(seed, seed + args.train_size):
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
                'gt_action': action_sequence
            })
    
            obs, reward, done, info = env.step(action_sequence[0])

    # Create test instances 
    test_instances = []
    for seed_test in range(seed + args.train_size, seed + args.train_size + args.test_size):
        obs = env.reset(seed=seed_test)
        gt_action_sequence = get_shortest_action_path(env.room_fixed, env.room_state, MAX_DEPTH=search_depth)
        if gt_action_sequence is None:
            print(f"No action sequence found for seed {seed_test}")
            continue

        if len_horizon > len(gt_action_sequence):
            print(f"No enough actions for seed {seed_train}, skip")
            continue
        
        for i in range(len(gt_action_sequence) - len_horizon + 1):
            instruction = templates[prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs, len_horizon=len_horizon))
            action_sequence = gt_action_sequence[i:i+len_horizon]
            test_instances.append({
                'instruction': instruction,
                'gt_action': action_sequence
            })
    
            obs, reward, done, info = env.step(action_sequence[0])
    
    # Create test instances for each test environment
    test_env_instances = []
    for seed_test_env in range(seed+args.train_size+args.test_size, seed+args.train_size+args.test_size+args.num_test_envs):
        obs = env.reset(seed=seed_test_env)
        instruction = templates[prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=obs, len_horizon=len_horizon))
        test_env_instances.append(instruction)

    def _create_instance(idx, instance):
        prompt_formatted = instance['instruction']
        gt_action = instance['gt_action']

        return {
            "data_source": "sokoban",
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": gt_action},
            "extra_info": {"split": "train", "index": idx}
        }
    
    def _create_test_env_instance(idx, instance):
        prompt_formatted = instance

        return {
            "data_source": "sokoban",
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": -1},
            "extra_info": {"split": "test", "index": idx}
        }

    train_dataset = Dataset.from_list([_create_instance(i, train_instances[i]) for i in range(len(train_instances))])
    test_dataset = Dataset.from_list([_create_instance(i, test_instances[i]) for i in range(len(test_instances))])
    test_env_dataset = Dataset.from_list([_create_test_env_instance(args.seed + i, test_env_instances[i-args.train_size-args.test_size]) for i in range(args.train_size+args.test_size, args.train_size+args.test_size+args.num_test_envs)])
    
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
    train_dataset.push_to_hub("Xiaofeng77/reil_sokoban", split="train")
    test_dataset.push_to_hub("Xiaofeng77/reil_sokoban", split="test")
    test_env_dataset.push_to_hub("Xiaofeng77/reil_sokoban", split="test_env")

if __name__ == "__main__":
    main()