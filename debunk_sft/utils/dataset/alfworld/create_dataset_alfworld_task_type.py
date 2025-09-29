from datasets import Dataset
import os
import argparse
import yaml
from reil.env.alfworld.env import ALFWorldTW, load_config_file
from reil.env.alfworld.config import ALFWorldConfig

os.environ['ALFWORLD_DATA'] = "YOUR_ALFWORLD_DATA"

PROMPT_TEMPLATE = {
    "llm_system_prompt": (
        "You are an household agent designed to interact with a simulated household environment to solve household tasks step by step. "
        "In this environment, you can interact with objects and receptacles to solve the task."
        "After you execute an action, you will receive a textual feedback from the environment."
    ),
    "llm_action_prompt": (
        "Specify the next action the agent should take to progress toward the task goal, following these guidelines:\n\n"
        "1. Object and Receptacle References: Use specific identifiers:\n"
        "   - [obj id] for objects (e.g., apple 1).\n"
        "   - [recep id] for receptacles (e.g., countertop 1).\n"
        "2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:\n"
        "Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]\n"
    )
}

INSTRUCTION_PROMPT = PROMPT_TEMPLATE['llm_system_prompt']  + "Here is the task:\n{history}" + PROMPT_TEMPLATE['llm_action_prompt']
templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

# Task types mapping
TASK_TYPES = {
    "pick_and_place": 1,
    "look_at_obj_in_light": 2,
    "pick_clean_then_place_in_recep": 3,
    "pick_heat_then_place_in_recep": 4,
    "pick_cool_then_place_in_recep": 5,
    "pick_two_obj_and_place": 6
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file", type=str, default="./thirdparty/alfworld/configs/base_config.yaml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--horizon", type=int, default=30, help="Maximum number of steps per episode (default: 30)")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'], 
                       help="Template prefix to use (default: qwen-instruct)")
    parser.add_argument("--output", type=str, default="./data/alfworld_task_type", help="Path to the output directory")
    parser.add_argument("--hf_repo", type=str, default="YOUR_HF_REPO", help="Hugging Face repo to push to")
    
    # Task-specific counts for training set
    parser.add_argument("--train_pick_and_place", type=int, default=100, 
                        help="Number of pick_and_place trajectories for training")
    parser.add_argument("--train_look_at_obj_in_light", type=int, default=0, 
                        help="Number of look_at_obj_in_light trajectories for training")
    parser.add_argument("--train_pick_clean_then_place_in_recep", type=int, default=0, 
                        help="Number of pick_clean_then_place_in_recep trajectories for training")
    parser.add_argument("--train_pick_heat_then_place_in_recep", type=int, default=0, 
                        help="Number of pick_heat_then_place_in_recep trajectories for training")
    parser.add_argument("--train_pick_cool_then_place_in_recep", type=int, default=0, 
                        help="Number of pick_cool_then_place_in_recep trajectories for training")
    parser.add_argument("--train_pick_two_obj_and_place", type=int, default=0, 
                        help="Number of pick_two_obj_and_place trajectories for training")
    
    # Task-specific counts for test set
    parser.add_argument("--test_pick_and_place", type=int, default=25, 
                        help="Number of pick_and_place trajectories for testing")
    parser.add_argument("--test_look_at_obj_in_light", type=int, default=25, 
                        help="Number of look_at_obj_in_light trajectories for testing")
    parser.add_argument("--test_pick_clean_then_place_in_recep", type=int, default=25, 
                        help="Number of pick_clean_then_place_in_recep trajectories for testing")
    parser.add_argument("--test_pick_heat_then_place_in_recep", type=int, default=25, 
                        help="Number of pick_heat_then_place_in_recep trajectories for testing")
    parser.add_argument("--test_pick_cool_then_place_in_recep", type=int, default=25, 
                        help="Number of pick_cool_then_place_in_recep trajectories for testing")
    parser.add_argument("--test_pick_two_obj_and_place", type=int, default=25, 
                        help="Number of pick_two_obj_and_place trajectories for testing")
    
    args = parser.parse_args()

    config = ALFWorldConfig()
    config.config_path = args.config_file
    config.train_eval = 'train'
    config.render_mode = 'collect'
    # setup environment
    env = ALFWorldTW(aw_config=config)

    # Create dictionaries to track the number of trajectories collected for each task type
    train_task_counts = {
        "pick_and_place": 0,
        "look_at_obj_in_light": 0,
        "pick_clean_then_place_in_recep": 0,
        "pick_heat_then_place_in_recep": 0,
        "pick_cool_then_place_in_recep": 0,
        "pick_two_obj_and_place": 0
    }
    
    test_task_counts = {
        "pick_and_place": 0,
        "look_at_obj_in_light": 0,
        "pick_clean_then_place_in_recep": 0,
        "pick_heat_then_place_in_recep": 0,
        "pick_cool_then_place_in_recep": 0,
        "pick_two_obj_and_place": 0
    }
    
    # Get the target counts for each task type
    train_task_targets = {
        "pick_and_place": args.train_pick_and_place,
        "look_at_obj_in_light": args.train_look_at_obj_in_light,
        "pick_clean_then_place_in_recep": args.train_pick_clean_then_place_in_recep,
        "pick_heat_then_place_in_recep": args.train_pick_heat_then_place_in_recep,
        "pick_cool_then_place_in_recep": args.train_pick_cool_then_place_in_recep,
        "pick_two_obj_and_place": args.train_pick_two_obj_and_place
    }
    
    test_task_targets = {
        "pick_and_place": args.test_pick_and_place,
        "look_at_obj_in_light": args.test_look_at_obj_in_light,
        "pick_clean_then_place_in_recep": args.test_pick_clean_then_place_in_recep,
        "pick_heat_then_place_in_recep": args.test_pick_heat_then_place_in_recep,
        "pick_cool_then_place_in_recep": args.test_pick_cool_then_place_in_recep,
        "pick_two_obj_and_place": args.test_pick_two_obj_and_place
    }
    
    train_instances = []
    test_instances = []
    # Track unique game files that have been generated
    generated_games = set()
    
    # Calculate total number of trajectories needed
    total_train_target = sum(train_task_targets.values())
    total_test_target = sum(test_task_targets.values())
    
    # interact
    seed = args.seed
    obs, infos = env.reset(seed=seed)
    
    # Function to check if we've collected enough trajectories
    def is_collection_complete():
        train_complete = all(train_task_counts[task] >= train_task_targets[task] for task in train_task_counts)
        test_complete = all(test_task_counts[task] >= test_task_targets[task] for task in test_task_counts)
        return train_complete and test_complete
    
    # Continue collecting until we have enough trajectories of each type
    while not is_collection_complete():
        game_file = '/'.join(infos['extra.gamefile'][0].split('/')[-3:-1])
        print(f"Processing game file: {game_file}")
        task_type = env.get_task_type()
        is_needed = train_task_counts[task_type] < train_task_targets[task_type] or test_task_counts[task_type] < test_task_targets[task_type]
        if game_file in generated_games or not is_needed:
            obs, infos = env.reset()
            continue
            
        generated_games.add(game_file)
        
        for step in range(args.horizon):
            expert_action = infos['extra.expert_plan'][0]
            obs, scores, dones, infos = env.step(expert_action)
            
            if step == args.horizon-1 or infos['won'][0]:
                if infos['won'][0]:
                    task_type = env.get_task_type()
                    s_a_history = env.get_s_a_history()
                    
                    # Check if we need more trajectories of this task type
                    for_train = train_task_counts[task_type] < train_task_targets[task_type]
                    for_test = test_task_counts[task_type] < test_task_targets[task_type]
                    
                    if for_train:
                        # Add to training set
                        train_instances.extend({
                            "data_source": task_type,
                            "state": templates[args.prefix].format(prompt=INSTRUCTION_PROMPT.format(history=item["state"])),
                            "expert_action": item["action"],
                            "seed": seed,
                            "game_file": game_file
                        } for item in s_a_history)
                        train_task_counts[task_type] += 1
                        print(f"Added 1 {task_type} trajectory to training set. Total: {train_task_counts[task_type]}/{train_task_targets[task_type]}")
                    elif for_test:
                        # Add to test set
                        test_instances.extend({
                            "data_source": task_type,
                            "state": templates[args.prefix].format(prompt=INSTRUCTION_PROMPT.format(history=item["state"])),
                            "expert_action": item["action"],
                            "seed": seed,
                            "game_file": game_file
                        } for item in s_a_history)
                        test_task_counts[task_type] += 1
                        print(f"Added 1 {task_type} trajectory to test set. Total: {test_task_counts[task_type]}/{test_task_targets[task_type]}")
                
                obs, infos = env.reset()
                break
    
    # Print final collection statistics
    print("\nFinal collection statistics:")
    print("Training set:")
    for task_type, count in train_task_counts.items():
        print(f"  {task_type}: {count}/{train_task_targets[task_type]}")
    
    print("\nTest set:")
    for task_type, count in test_task_counts.items():
        print(f"  {task_type}: {count}/{test_task_targets[task_type]}")

    def _create_instance(idx, instance):
        data_source = instance['data_source']
        prompt_formatted = instance['state']
        expert_action = instance['expert_action']
        game_file = instance['game_file']
        seed = instance['seed']
        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "household",
            "reward_model": {"style": "rule", "ground_truth": expert_action},
            "extra_info": {"split": "train", "index": idx, "game_file": game_file, "seed": seed}
        }
    
    train_dataset = Dataset.from_list([_create_instance(i, train_instances[i]) for i in range(len(train_instances))])
    test_dataset = Dataset.from_list([_create_instance(i, test_instances[i]) for i in range(len(test_instances))])

    def make_map_fn(split):
        def process_fn(example, idx):
            # Add split information to each example
            example['extra_info']['split'] = split
            if split == 'train':
                example['id'] = f"train_{idx}"
            elif split == 'test':
                example['id'] = f"test_{idx}"

            return example
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    os.makedirs(args.output, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

    # Push to hub
    if args.hf_repo and args.hf_repo != "YOUR_HF_REPO":
        train_dataset.push_to_hub(args.hf_repo, split="train")
        test_dataset.push_to_hub(args.hf_repo, split="test")

if __name__ == "__main__":
    main()
