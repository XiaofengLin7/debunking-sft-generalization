import random
import json
from typing import List, Dict, Any, Optional
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
from datasets import Dataset
from tqdm import tqdm

Q_GeneralPoint_EQN_L = """
[Task Description]
You are an expert {target_number} points card game player. You will receive a set of 4 cards.
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

def generate_cards(num_cards=4, treat_face_cards_as_10=True):
    cards_num = [random.randint(1, 13) for _ in range(num_cards)]
    if treat_face_cards_as_10:
        display_card_nums = [min(num, 10) for num in cards_num]
    else:
        display_card_nums = cards_num

    cards_str = [card_num_to_str(num) for num in cards_num]
    return cards_str, display_card_nums

def card_num_to_str(num: int) -> str:
    face_cards = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
    return face_cards.get(num, str(num))

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

def generate_task(task_id: int, target: int, num_cards: int = 4, 
                  treat_face_cards_as_10: bool = True, seed: Optional[int] = None) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    
    cards_str, display_card_nums = generate_cards(num_cards, treat_face_cards_as_10)
    solutions = solve_game(display_card_nums, target)

    attempts = 0
    while not solutions and attempts < 100:
        cards_str, display_card_nums = generate_cards(
            num_cards, treat_face_cards_as_10, 
        )
        solutions = solve_game(display_card_nums, target)
        attempts += 1

    if not solutions:
        raise ValueError(f"No solution found for task {task_id} with target {target} and {num_cards} cards")
    
    solution = random.choice(solutions)
    # Convert cards_str to proper JSON format with double quotes
    cards_json = json.dumps(cards_str)
    formatted_solution = f"{{\n\"cards\": {cards_json},\n \"number\": {display_card_nums},\n \"formula\": \"{solution}\"\n}}"
    face_card_msg = "'J', 'Q', and 'K' count as '10'." if treat_face_cards_as_10 \
                                            else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively."
    
    task_prompt = Q_GeneralPoint_EQN_L.format(
        target_number=target,
        face_card_msg=face_card_msg,
        cards=cards_str,
    )

    meta_info = {
        "task_id": task_id,
        "cards": cards_str,
        "display_cards": display_card_nums,
        "solution": solution,
        "target": target,
        "treat_face_cards_as_10": treat_face_cards_as_10,
    }

    return {
        "question": task_prompt,
        "answer": formatted_solution,
        "meta_info": meta_info
    }

def main():
    dataset_id = "Xiaofeng77/gp-l-only"  # Change this to your HF namespace
    num_tasks = 5000
    datapoints = []

    for task_id in tqdm(range(num_tasks)):
        task = generate_task(task_id, target=24, num_cards=4, treat_face_cards_as_10=False, seed=42 + task_id)
        # print(task["question"])
        # print(task["answer"])
        # print("-"*100)
        datapoints.append(task)

    dataset = Dataset.from_list(datapoints)

    dataset.to_parquet("./data/gp-l-only/test.parquet")
    dataset.push_to_hub(dataset_id, split="test")


if __name__ == "__main__":
    main()