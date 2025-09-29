from datasets import Dataset
import os
import argparse
import yaml
from debunk_sft.env.alfworld.env import ALFWorldTW, load_config_file
from debunk_sft.env.alfworld.config import ALFWorldConfig

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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_size", type=int, default=200, help="Number of train instances to generate (default: 200)")
    parser.add_argument("--test_size", type=int, default=40, help="Number of test instances to generate (default: 40)")
    parser.add_argument("--horizon", type=int, default=30, help="Number of test instances to generate (default: 50)")
    parser.add_argument("--config_file", type=str, default="./thirdparty/alfworld/configs/base_config.yaml", help="Path to the config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'], 
                       help="Template prefix to use (default: qwen-instruct)")
    parser.add_argument("--output", type=str, default="./data/alfworld", help="Path to the output directory")
    parser.add_argument("--hf_repo", type=str, default="YOUR_HF_REPO", help="Hugging Face repo to push to")
    
    args = parser.parse_args()

    config = ALFWorldConfig()
    config.train_eval = 'train'
    config.render_mode = 'complete'
    # setup environment
    env = ALFWorldTW(aw_config=config)

    train_instances = []
    test_instances = []
    # Track unique game files that have been generated
    generated_games = set()
    # interact
    obs, infos = env.reset(seed=args.seed)
    for seed in range(args.seed+1, args.seed + args.train_size + args.test_size+1):
        game_file = '/'.join(infos['extra.gamefile'][0].split('/')[-3:-1])
        print(game_file)
        if game_file in generated_games:
            obs, infos = env.reset(seed=seed)
            continue
        generated_games.add(game_file)
        for step in range(args.horizon):
            # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
            # admissible_commands = list(infos['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
            expert_action = infos['extra.expert_plan'][0]
            # step
            obs, scores, dones, infos = env.step(expert_action)
            # print("Action: {}, Obs: {}".format(expert_action[0], obs))
            if step == args.horizon-1 or infos['won'][0]:
                if infos['won'][0]:
                    data_source = env.get_task_type()
                    s_a_history = env.get_s_a_history()
                    if seed > args.seed + args.train_size:
                        test_instances.extend({
                            "data_source": data_source,
                            "state": templates[args.prefix].format(prompt=INSTRUCTION_PROMPT.format(history=item["state"])),
                            "expert_action": item["action"],
                            "seed": seed,
                            "game_file": game_file
                        } for item in s_a_history)
                    else:
                        train_instances.extend({
                            "data_source": data_source,
                            "state": templates[args.prefix].format(prompt=INSTRUCTION_PROMPT.format(history=item["state"])),
                            "expert_action": item["action"],
                            "seed": seed,
                            "game_file": game_file
                        } for item in s_a_history)
                obs, infos = env.reset(seed=seed)

                break

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
            example['extra_info']['split'] = split  # Currently in your code
            if split == 'train':
                # Apply training-specific transformations
                example['id'] = f"train_{idx}"
            elif split == 'test':
                # Apply test-specific transformations
                example['id'] = f"test_{idx}"

            return example
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    os.makedirs(args.output, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))

    if args.hf_repo and args.hf_repo != "YOUR_HF_REPO":
        train_dataset.push_to_hub(args.hf_repo, split="train")
        test_dataset.push_to_hub(args.hf_repo, split="test")
if __name__ == "__main__":

    main()

