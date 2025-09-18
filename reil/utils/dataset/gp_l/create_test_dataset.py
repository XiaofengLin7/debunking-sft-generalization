"""
This script is used to create the test split of dataset for gp-l domain.
It creates multiple test datasets with different configurations:
- gp_l_5cards: 5 cards with J,Q,K as 10
- gp_l_large: 4 cards with at least one number from 14-19
- gp_l_face_card_as_regular: 4 cards with J,Q,K as 11,12,13 respectively
- gp_l_all_12: 4 cards with J,Q,K all as 12
- gp_l_fake: 4 cards with inconsistent prompt vs actual values
"""
import random
import json
import numpy as np
from typing import List, Dict, Any, Optional
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
from datasets import Dataset, Features, Value
from tqdm import tqdm

# Import face card mappings from configuration file
from .face_card_configs import FACE_CARD_MAPPINGS

Q_GeneralPoint_EQN_L = """
[Task Description]
You are an expert {target_number} points card game player. You will receive a set of {num_cards} cards.
Note that {face_card_msg}, and each card must be used once.
Your goal is to output a formula that evaluates to {target_number} using numbers from the cards and operators such as '+', '-', '*', '/', '(', ')', and '='.

[Input]
Cards: {cards}

[Output]
{{
  "cards": [x, y, z, w], where {face_card_msg},
  "number": [a, b, c, d], where a, b, c, and d are the numbers on the cards,
  "formula": 'an equation that equals {target_number}',
}}

"""



def generate_cards(num_cards=4, treat_face_cards_as_10=True, ood=False, largest_card=13):
    if not ood:
        cards_num = [random.randint(1, largest_card) for _ in range(num_cards)]
    else:
        cards_num = [random.randint(1, largest_card) for _ in range(num_cards - 1)] + [random.randint(11, largest_card)]
        # shuffle the cards
    random.shuffle(cards_num)
    
    if treat_face_cards_as_10:
        display_card_nums = [min(num, 10) if num <= 13 else num for num in cards_num]
    else:
        display_card_nums = cards_num

    cards_str = [card_num_to_str(num) for num in cards_num]
    return cards_str, display_card_nums

def card_num_to_str(num: int) -> str:
    face_cards = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    return face_cards.get(num, str(num))

def card_str_to_num(card: str) -> int:
    assert card in ['A', 'J', 'Q', 'K', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    face_cards = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    if card in face_cards:
        return face_cards[card]
    else:
        return int(card)

def solve_game(digits: List[int], target: int) -> List[str]:
    digilen = len(digits)
    exprlen = 2 * digilen - 1
    digiperm = sorted(set(permutations(digits)))
    random.shuffle(digiperm)
    opcomb = list(product('+-*/', repeat=digilen-1))
    brackets = ([()] + [(x, y)
                        for x in range(0, exprlen, 2)
                        for y in range(x+4, exprlen+2, 2)
                        if (x, y) != (0, exprlen+1)]
                + [(0, 3+1, 4+2, 7+3)])
    
    solutions = []
    for d in digiperm:
        for ops in opcomb:
            if '/' in ops:
                d2 = [('F(%s)' % i) for i in d]
            else:
                d2 = d
            ex = list(chain.from_iterable(zip_longest(d2, ops, fillvalue='')))
            for b in brackets:
                exp = ex[::]
                for insertpoint, bracket in zip(b, '()'*(len(b)//2)):
                    exp.insert(insertpoint, bracket)
                txt = ''.join(str(i) for i in exp)
                try:
                    num = eval(txt)
                except ZeroDivisionError:
                    continue
                if num == target:
                    if '/' in ops:
                        exp = [(term if not term.startswith('F(') else term[2:-1])
                            for term in exp]
                    ans = ''.join(str(i) for i in exp).rstrip()
                    solutions.append(ans)

    return solutions

def generate_task(task_id: int, 
                  target: int, 
                  num_cards: int = 4, 
                  data_source: str = "gp-l",
                  treat_face_cards_as_10: bool = True, 
                  seed: Optional[int] = None,
                  ood: bool = False,
                  largest_card: int = 13,
                  is_fake: bool = False) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    if is_fake:
        ## prompt and reward calculation are inconsistent to test shortcut learning
        largest_card = 13
        target = 24
        ood = True
        num_cards = 4
        face_card_msg = "'J', 'Q', and 'K' count as '10'" if not treat_face_cards_as_10 \
                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively"
    else:
        face_card_msg = "'J', 'Q', and 'K' count as '10'" if treat_face_cards_as_10 \
                        else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively"
    
    cards_str, display_card_nums = generate_cards(num_cards=num_cards, 
                                                  treat_face_cards_as_10=treat_face_cards_as_10, 
                                                  largest_card=largest_card, 
                                                  ood=ood)
    
    solutions = solve_game(display_card_nums, target)

    attempts = 0
    while not solutions and attempts < 50:
        cards_str, display_card_nums = generate_cards(num_cards=num_cards, 
                                                  treat_face_cards_as_10=treat_face_cards_as_10, 
                                                  largest_card=largest_card, 
                                                  ood=ood)
        solutions = solve_game(display_card_nums, target)
        attempts += 1

    if not solutions:
        raise ValueError(f"No solution found for task {task_id} with target {target} and {num_cards} cards")
    
    solution = random.choice(solutions)
    # Convert cards_str to proper JSON format with double quotes
    cards_json = json.dumps(cards_str)
    formatted_solution = f"{{\n\"cards\": {cards_json},\n \"number\": {display_card_nums},\n \"formula\": \"{solution}\"\n}}"

    task_prompt = Q_GeneralPoint_EQN_L.format(
        target_number=target,
        face_card_msg=face_card_msg,
        cards=cards_str,
        num_cards=num_cards,
    )

    extra_info = {
        "index": task_id,
        "cards": cards_str,  # Keep as list, features will handle the schema
        "display_cards": display_card_nums,  # Keep as list, features will handle the schema
        "solution": solution,
        "target": target,
        "treat_face_cards_as_10": treat_face_cards_as_10,
    }
    sft_instance = {
        "data_source": data_source,
        "question": task_prompt,
        "answer": formatted_solution,
        "extra_info": extra_info
    }
    rl_instance = {
        "data_source": data_source,
        "question": [{"role": "user", "content": task_prompt}],
        "extra_info": extra_info
    }
    return sft_instance, rl_instance

# Enhanced task generation with flexible face card mapping support
def generate_task_with_mapping(task_id: int, 
                              target: int, 
                              num_cards: int = 4,
                              face_card_mapping: str = "all_10",
                              data_source: str = "gp-l",
                              seed: Optional[int] = None,
                              ood: bool = False,
                              largest_card: int = 13,
                              fake_prompt: bool = False) -> Dict[str, Any]:
    """
    Generate a task with flexible face card mapping support.
    
    Args:
        fake_prompt: If True, the prompt will show different face card values than actual
    """
    if seed is not None:
        random.seed(seed)
    
    # Get the mapping configuration
    if face_card_mapping not in FACE_CARD_MAPPINGS:
        raise ValueError(f"Unknown face card mapping: {face_card_mapping}")
    
    mapping = FACE_CARD_MAPPINGS[face_card_mapping]
    
    # Generate face card message for prompt
    # if fake_prompt:
    #     # For fake prompt, always show J=11, Q=12, K=13 in prompt
    #     face_card_msg = "'J' counts as '11', 'Q' counts as '12', and 'K' counts as '13'"
    # else:
    #     # Generate real face card message based on mapping
    if len(set(mapping.values())) == 1:
        value = list(mapping.values())[0]
        face_card_msg = f"'J', 'Q', and 'K' all count as '{value}'"
    else:
        parts = [f"'{k}' counts as '{v}'" for k, v in mapping.items()]
        face_card_msg = f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    # Generate cards
    cards_str, display_card_nums = generate_cards_with_mapping(
        num_cards=num_cards, 
        face_card_mapping=mapping,
        largest_card=largest_card, 
        ood=ood,
        fake_values=fake_prompt
    )
    
    solutions = solve_game(display_card_nums, target)

    attempts = 0
    while not solutions and attempts < 50:
        cards_str, display_card_nums = generate_cards_with_mapping(
            num_cards=num_cards, 
            face_card_mapping=mapping,
            largest_card=largest_card, 
            ood=ood,
            fake_values=fake_prompt
        )
        solutions = solve_game(display_card_nums, target)
        attempts += 1

    if not solutions:
        raise ValueError(f"No solution found for task {task_id} with target {target} and {num_cards} cards")
    
    solution = random.choice(solutions)
    cards_json = json.dumps(cards_str)
    formatted_solution = f"{{\n\"cards\": {cards_json},\n \"number\": {display_card_nums},\n \"formula\": \"{solution}\"\n}}"

    task_prompt = Q_GeneralPoint_EQN_L.format(
        target_number=target,
        face_card_msg=face_card_msg,
        cards=cards_str,
        num_cards=num_cards,
    )

    extra_info = {
        "index": task_id,
        "cards": cards_str,
        "display_cards": display_card_nums,
        "solution": solutions,
        "target": target,
        "face_card_mapping": face_card_mapping,
    }
    
    sft_instance = {
        "data_source": data_source,
        "question": task_prompt,
        "answer": formatted_solution,
        "extra_info": extra_info
    }
    
    rl_instance = {
        "data_source": data_source,
        "question": [{"role": "user", "content": task_prompt}],
        "extra_info": extra_info
    }
    
    return sft_instance, rl_instance

def generate_cards_with_mapping(num_cards: int = 4, 
                               face_card_mapping: Dict[str, int] = None,
                               largest_card: int = 13, 
                               ood: bool = False,
                               fake_values: bool = False) -> tuple:
    """
    Generate cards with flexible face card mapping.
    
    Args:
        fake_values: If True, use 10 for all face cards regardless of mapping
    """
    if face_card_mapping is None:
        face_card_mapping = {"J": 10, "Q": 10, "K": 10}
    
    if not ood:
        cards_num = [random.randint(1, largest_card) for _ in range(num_cards)]
    else:
        # For OOD, ensure we have at least one card from the target range
        if largest_card >= 14:
            # For large cards (>= 14), ensure at least one card from 14-largest_card
            cards_num = [random.randint(1, largest_card) for _ in range(num_cards - 1)] + [random.randint(14, largest_card)]
        elif largest_card >= 11:
            # For face cards (11-13), ensure at least one face card
            cards_num = [random.randint(1, largest_card) for _ in range(num_cards - 1)] + [random.randint(11, largest_card)]
        else:
            # If largest_card < 11, just generate normal cards
            cards_num = [random.randint(1, largest_card) for _ in range(num_cards)]
    
    random.shuffle(cards_num)
    
    # Convert to display values using face card mapping
    display_card_nums = []
    cards_str = []
    
    for num in cards_num:
        card_str = card_num_to_str(num)
        cards_str.append(card_str)
        
        # Map face cards to numbers
        if card_str in ['J', 'Q', 'K']:
            if fake_values:
                # For fake values, always use 10 regardless of mapping
                display_card_nums.append(10)
            else:
                display_card_nums.append(face_card_mapping[card_str])
        elif card_str == 'A':
            display_card_nums.append(1)
        else:
            display_card_nums.append(int(card_str))
    
    return cards_str, display_card_nums

# Test dataset generation functions
def generate_test_5cards(num_tasks: int = 500) -> tuple:
    """Generate test dataset with 5 cards where J,Q,K count as 10."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_5cards"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=5,
                face_card_mapping="all_10",
                data_source="gp_l_5cards",
                seed=42 + task_id,
                ood=False,
                largest_card=13
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_large(num_tasks: int = 500) -> tuple:
    """Generate test dataset with 4 cards, at least one from 14-19."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_large"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_10",
                data_source="gp_l_large",
                seed=42 + task_id,
                ood=True,  # Ensure we get large numbers
                largest_card=19
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_face_card_as_regular(num_tasks: int = 500) -> tuple:
    """Generate test dataset where J,Q,K are 11,12,13 respectively."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_face_card_as_regular"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="mixed_11_12_13",
                data_source="gp_l_face_card_as_regular",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_all_7(num_tasks: int = 500) -> tuple:
    """Generate test dataset where J,Q,K all count as 7."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_all_7"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_7",
                data_source="gp_l_all_7",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_all_5_fake(num_tasks: int = 500) -> tuple:
    """Generate test dataset where J,Q,K all count as 5."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_all_5_fake"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_5",
                data_source="gp_l_all_5_fake",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13,
                fake_prompt=True
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_all_5(num_tasks: int = 500) -> tuple:
    """Generate test dataset where J,Q,K all count as 5."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_all_5"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_5",
                data_source="gp_l_all_5",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_all_12(num_tasks: int = 500) -> tuple:
    """Generate test dataset where J,Q,K all count as 12."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_all_12"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_12",
                data_source="gp_l_all_12",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_fake(num_tasks: int = 500) -> tuple:
    """Generate fake test dataset where prompt says J,Q,K are 11,12,13 but actual values are 10."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_fake"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="mixed_11_12_13",  # This is just for generation, won't be used for actual values
                data_source="gp_l_fake",
                seed=42 + task_id,
                ood=True,  # Ensure we get face cards
                largest_card=13,
                fake_prompt=True  # This makes all face cards actually worth 10
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def generate_test_id(num_tasks: int = 500) -> tuple:
    """Generate in-distribution test dataset."""
    sft_datapoints = []
    rl_datapoints = []
    
    for task_id in tqdm(range(num_tasks), desc="Generating gp_l_id"):
        try:
            sft_instance, rl_instance = generate_task_with_mapping(
                task_id=task_id,
                target=24,
                num_cards=4,
                face_card_mapping="all_10",  # This is just for generation, won't be used for actual values
                data_source="gp_l_id",
                seed=42 + task_id,
                ood=True, 
                largest_card=13,
                fake_prompt=False 
            )
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue
    
    return sft_datapoints, rl_datapoints

def get_test_dataset_features():
    """Get the dataset features schema for test datasets."""
    return Features({
        'data_source': Value('string'),
        'question': [{'content': Value(dtype='string', id=None), 
                      'role': Value(dtype='string', id=None)}],
        'extra_info': {
            'index': Value('int64'),
            'cards': [Value('string')],
            'display_cards': [Value('int64')],
            'solution': [Value('string')],
            'target': Value('int64'),
            'face_card_mapping': Value('string')
        }
    })

def main():
    """Generate all test datasets."""
    dataset_id = "Xiaofeng77/gp-l-only-20k-mixed"  
    num_tasks = 500
    
    print("Generating test datasets for gp-l domain...")
    
    # Generate all test datasets
    test_datasets = {
        # "5cards": generate_test_5cards(num_tasks),
        # "large": generate_test_large(num_tasks),
        # "face_card_as_regular": generate_test_face_card_as_regular(num_tasks),
        # "all_12": generate_test_all_12(num_tasks),
        # "fake": generate_test_fake(num_tasks)
        # "id": generate_test_id(num_tasks),
        # "all_7": generate_test_all_7(num_tasks),
        # "all_5": generate_test_all_5(num_tasks),
        "all_5_fake": generate_test_all_5_fake(num_tasks),
    }
    
    features = get_test_dataset_features()
    
    # Save each dataset
    for dataset_name, (sft_data, rl_data) in test_datasets.items():
        print(f"Saving {dataset_name} dataset with {len(sft_data)} instances...")
        
        # Create datasets
        sft_dataset = Dataset.from_list(sft_data)
        rl_dataset = Dataset.from_list(rl_data, features=features)
        
        # Save to parquet files
        sft_dataset.to_parquet(f"./data/gp-l-only/10k-mixed/sft/test_{dataset_name}.parquet")
        rl_dataset.to_parquet(f"./data/gp-l-only/10k-mixed/rl/test_{dataset_name}.parquet")
        sft_dataset.to_parquet(f"./data/gp-l-only/20k-mixed/sft/test_{dataset_name}.parquet")
        rl_dataset.to_parquet(f"./data/gp-l-only/20k-mixed/rl/test_{dataset_name}.parquet")
        
        # Push to hub with appropriate split name
        rl_dataset.push_to_hub(dataset_id, split=f"test_{dataset_name}")
    
    print("All test datasets generated and saved successfully!")


if __name__ == "__main__":
    main()