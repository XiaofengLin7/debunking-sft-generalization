"""
This script is used to create the test split of dataset for gp-l domain.
"""
import random
import json
import numpy as np
from typing import List, Dict, Any, Optional
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
from datasets import Dataset
from tqdm import tqdm

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

def main():
    dataset_id = "Xiaofeng77/gp-l-only-10k"  # Change this to your HF namespace
    num_tasks = 500
    sft_datapoints = []
    rl_datapoints = []

    for task_id in tqdm(range(num_tasks)):
        try:
            sft_instance, rl_instance = generate_task(task_id, target=24, num_cards=5, 
                                                    treat_face_cards_as_10=True, 
                                                    seed=42 + task_id, 
                                                    data_source="gp-l-5cards",
                                                    ood=True,
                                                    largest_card=13,
                                                    is_fake=False)
            sft_datapoints.append(sft_instance)
            rl_datapoints.append(rl_instance)
        except ValueError as e:
            print(f"Skipping task {task_id}: {e}")
            continue

    # Define features to match existing dataset schema
    from datasets import Features, Value
    
    features = Features({
        'data_source': Value('string'),
        'question': [{'content': Value(dtype='string', id=None), 
                      'role': Value(dtype='string', id=None)}],
        'extra_info': {
            'index': Value('int64'),
            'cards': [Value('string')],  # Fixed length of 4
            'display_cards': [Value('int64')],  # Fixed length of 4
            'solution': Value('string'),
            'target': Value('int64'),
            'treat_face_cards_as_10': Value('bool')
        }
    })
    
    sft_dataset = Dataset.from_list(sft_datapoints)
    rl_dataset = Dataset.from_list(rl_datapoints, features=features)

    sft_dataset.to_parquet("./data/gp-l-only/sft/test_5cards.parquet")
    rl_dataset.to_parquet("./data/gp-l-only/rl/test_5cards.parquet")
    rl_dataset.push_to_hub(dataset_id, split="test_5cards")


if __name__ == "__main__":
    main()