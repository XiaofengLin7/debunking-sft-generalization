import re
import json
from reil.utils.dataset.gp_l.create_test_dataset_gp_l import solve_game
from collections import Counter
from typing import Optional

# some predefined patterns
RE_PATTERN_DICT = {
    "cards": r'"cards": \[([^\]]+)\]',
    "number": r'"number": \[([^\]]+)\]',
    "answer": r'"cards": \[([^\]]+)\]',
    "cards_remained": r'"cards_remained": \[([^\]]+)\]',
    "action": r'"action": "(.*?)"',
    "formula": r'"formula": "(.*?)"',
    "current observation": r'"current observation": "(.*?)"',
    "current instruction": r'"current instruction": "(.*?)"',
}

REWARD_FN = {
    "CORRECT_SOLUTION": 5,
    "PARTIAL_SOLUTION": 1,
    "NO_SOLUTION": -1,
    "INCORRECT_VISION": -1.5,
    "INCORRECT_NUMBER": -2,
    "ILLEGAL_FORMULA": -3,
}


def re_match(text: str, pattern: str):
    
    try:
        output_dict = json.loads(text)
        pred = output_dict[pattern]
    except:
        try:
            pattern_re = re.search(RE_PATTERN_DICT[pattern], text)
            pred = pattern_re.group(1)
            # print("Pred in try 2:", pred)
            # handle cases for list
            if 'cards' in pattern:
                try:
                    pred = list(map(int, pattern_re.group(1).split(', ')))
                except:
                    pred = '[' + pred + ']'
            
        except:
            pred = "None"
    
    return pred

def robust_str_to_list(list_like_str: str, num_cards: int):
    try:
        list_like_str = list_like_str.replace('[', '').replace(']', '').replace('\'', '').replace('\"', '').replace(' ', '').split(',')
        assert len(list_like_str) == num_cards
    except:
        list_like_str = []
    return list_like_str

def calculate_rewards(  card_nums: list[int],
                    current_formula: str,
                    target_points: int,
                    recognized_cards: list[str],
                    translated_number: list[str],   
                    gt_cards: list[str],
                    solutions: Optional[list] = None,
                    ) -> int:
    """
    card_nums: list of nums in the ground truth cards
    current_formula: formula string
    solutions: list of solutions
    target_points: target points
    recognized_cards: list of cards in the recognized cards
    translated_number: list of numbers in the translated number
    gt_cards: list of cards in the ground truth cards
    """
    reward = 0
    sorted_recognized_cards = sorted(recognized_cards)
    sorted_gt_cards = sorted(gt_cards)
    sorted_card_nums = sorted(card_nums)
    # convert translated_number to int
    try:
        translated_number = [int(num) for num in translated_number]
    except:
        # return REWARD_FN["INCORRECT_VISION"]
        pass
    sorted_translated_number = sorted(translated_number)
    # print(sorted_recognized_cards, sorted_gt_cards, sorted_card_nums, sorted_translated_number)
    # if sorted_recognized_cards != sorted_gt_cards or sorted_card_nums != sorted_translated_number:
    #     reward += REWARD_FN["INCORRECT_VISION"]
    #     return reward
    # Function to get token type
    def get_token_type(token):
        if token in '+-*/':
            return 'operator'
        elif token == '(':
            return 'open_paren'
        elif token == ')':
            return 'close_paren'
        elif re.match(r'\d+', token):
            return 'number'
        else:
            return 'unknown'
        # Extract tokens
    current_formula = current_formula.replace(' ', '')
    tokens = re.findall(r'\d+|[+\-*/()]', current_formula)
    tokens_str = ''.join(tokens)

    # print(tokens_str, current_formula)
    if tokens_str != current_formula:
        # There are illegal characters in current_formula
        reward += REWARD_FN["ILLEGAL_FORMULA"]
        return reward
    prev_token_type = None
    paren_stack = []
    for token in tokens:
        token_type = get_token_type(token)
        if token_type == 'number':
            if prev_token_type in [None, 'operator', 'open_paren']:
                pass
            else:
                # invalid sequence
                reward += REWARD_FN["ILLEGAL_FORMULA"]
                return reward
        elif token_type == 'operator':
            if prev_token_type in ['number', 'close_paren']:
                pass
            else:
                # invalid sequence
                reward += REWARD_FN["ILLEGAL_FORMULA"]
                return reward
        elif token_type == 'open_paren':
            if prev_token_type in [None, 'operator', 'open_paren']:
                paren_stack.append(token)
            else:
                # invalid sequence
                reward += REWARD_FN["ILLEGAL_FORMULA"]
                return reward
        elif token_type == 'close_paren':
            if prev_token_type in ['number', 'close_paren']:
                if paren_stack:
                    paren_stack.pop()
                else:
                    # invalid sequence 
                    reward += REWARD_FN["ILLEGAL_FORMULA"]
                    return reward
            else:
                # invalid sequence
                reward += REWARD_FN["ILLEGAL_FORMULA"]
                return reward
        else:
            reward += REWARD_FN["ILLEGAL_FORMULA"]
            return reward
        prev_token_type = token_type

    # Extract numbers from current_formula
    numbers_in_formula = re.findall(r'\d+', current_formula)
    numbers_in_formula_int = [int(n) for n in numbers_in_formula]

    # Count numbers
    card_nums_counts = Counter(card_nums)
    numbers_in_formula_counts = Counter(numbers_in_formula_int)
    # Check for invalid numbers
    invalid_numbers = [num for num in numbers_in_formula_counts if num not in card_nums_counts]
    overused_numbers = [num for num in numbers_in_formula_counts if numbers_in_formula_counts[num] > card_nums_counts[num]]
    if invalid_numbers:
        reward += REWARD_FN["INCORRECT_NUMBER"]
    elif overused_numbers:
        reward += REWARD_FN["INCORRECT_NUMBER"]
    
    underused_numbers = [num for num in card_nums_counts if card_nums_counts[num] > numbers_in_formula_counts[num]]
    if underused_numbers:
        reward += REWARD_FN["INCORRECT_NUMBER"]
    # if currently reward is 0
    if reward == 0:
        try:
            if eval(current_formula) == target_points:
                reward += REWARD_FN["CORRECT_SOLUTION"]
                return reward
        except Exception as e:
            if solutions and any(sol.startswith(current_formula) for sol in solutions):
                reward += REWARD_FN["PARTIAL_SOLUTION"]
                return reward
    else:
        return reward
    
    # Now we check cases with formula that use valid numbers but still no solution
    register = ""
    for token in tokens:
        # print(token)
        prev_register = register
        register += token
        # check if register is in any solution
        if solutions and any(sol.startswith(register) for sol in solutions):
            pass
        else:
            if prev_register == "":
                reward += REWARD_FN["NO_SOLUTION"]
            else:
                reward += REWARD_FN["NO_SOLUTION"]
            break

    return reward

def score_gp_l(solution_str: str, meta_info: dict):

    current_formula = re_match(solution_str, "formula")
    recognized_cards = re_match(solution_str, 'cards')
    translated_number = re_match(solution_str, 'number')
    # print(recognized_cards, translated_number)
    try:
        current_formula = current_formula.split('=')[0]
    except:
        pass
    
    num_cards = len(meta_info["cards"])
    digits = meta_info["display_cards"]
    target = meta_info["target"]

    solutions = solve_game(digits, target)
    
    recognized_cards = robust_str_to_list(recognized_cards, num_cards)
    translated_number = robust_str_to_list(translated_number, num_cards)
    # print(recognized_cards, translated_number)
    # print(meta_info["cards"])


    reward = calculate_rewards(card_nums=digits, 
                               current_formula=current_formula, 
                               solutions=solutions, 
                               target_points=target, 
                               recognized_cards=recognized_cards, 
                               translated_number=translated_number, 
                               gt_cards=meta_info["cards"])
    return reward

def score_gp_l_wo_sol(solution_str: str, meta_info: dict):
    current_formula = re_match(solution_str, "formula")
    recognized_cards = re_match(solution_str, 'cards')
    translated_number = re_match(solution_str, 'number')
    # print(recognized_cards, translated_number)
    try:
        current_formula = current_formula.split('=')[0]
    except:
        pass
    
    num_cards = len(meta_info["cards"])
    digits = meta_info["display_cards"]
    target = meta_info["target"]
    
    recognized_cards = robust_str_to_list(recognized_cards, num_cards)
    translated_number = robust_str_to_list(translated_number, num_cards)
    # print(recognized_cards, translated_number)
    # print(meta_info["cards"])

    reward = calculate_rewards(card_nums=digits, 
                               current_formula=current_formula, 
                               target_points=target, 
                               recognized_cards=recognized_cards, 
                               translated_number=translated_number, 
                               gt_cards=meta_info["cards"])
    return reward


if __name__ == "__main__":
    solution_str = """
    {
        "cards": ["A', "2", "3", "4"],
        "number": [1, 2, 3, 4],
        "formula": "1+2+3*4",
    }
    """
    meta_info = {
        "cards": ["K", "10", "3", "4"],
        "display_cards": [1, 2, 3, 4],
        "target": 24,
    }
    reward = score_gp_l(solution_str, meta_info)
    print(reward)

    meta_info = {
        "cards": ["K", "10", "3", "4", "7"],
        "display_cards": [10, 10, 3, 4, 7],
        "target": 39,
    }
    solution_str = """
    {
        "cards": ["K", "10", "3", "4", "7"],
        "number": [10, 10, 3, 4, 7],
        "formula": "10+10+3*4+7",
    }
    """
    reward = score_gp_l_wo_sol(solution_str, meta_info)
    print(reward)