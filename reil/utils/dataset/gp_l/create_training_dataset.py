"""
This script is used to create the training split of dataset for gp-l domain.
It supports different face card mappings for training data diversity.
"""
import random
import json
from typing import List, Dict, Any, Optional, Tuple
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
from datasets import Dataset, Features, Value
from tqdm import tqdm

# Import face card mappings from configuration file
from .face_card_configs import FACE_CARD_MAPPINGS, TRAINING_PRESETS

TASK_TEMPLATE = """
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

class FaceCardMapper:
    """Handles face card to number mapping with flexible configurations."""
    
    def __init__(self, mapping_name: str):
        if mapping_name not in FACE_CARD_MAPPINGS:
            raise ValueError(f"Unknown mapping: {mapping_name}. Available: {list(FACE_CARD_MAPPINGS.keys())}")
        
        self.mapping_name = mapping_name
        self.mapping = FACE_CARD_MAPPINGS[mapping_name]
        
    def get_face_card_message(self) -> str:
        """Generate human-readable description of the mapping."""
        if len(set(self.mapping.values())) == 1:
            value = list(self.mapping.values())[0]
            return f"'J', 'Q', and 'K' all count as '{value}'"
        else:
            parts = [f"'{k}' counts as '{v}'" for k, v in self.mapping.items()]
            return f"{', '.join(parts[:-1])}, and {parts[-1]}"
    
    def map_face_card(self, card: str) -> int:
        """Map a face card to its numeric value."""
        if card in ['J', 'Q', 'K']:
            if card not in self.mapping:
                raise ValueError(f"Card {card} not in mapping {self.mapping_name} (available: {list(self.mapping.keys())})")
            return self.mapping[card]
        elif card == 'A':
            return 1
        else:
            return int(card)
    
    def get_max_card_value(self) -> int:
        """Get the maximum possible card value for this mapping."""
        return max(self.mapping.values())

class CardGameGenerator:
    """Generates card game instances with solvable puzzles."""
    
    def __init__(self, face_mapper: FaceCardMapper):
        self.face_mapper = face_mapper
        
    def generate_cards(self, num_cards: int = 4, ood: bool = False, largest_card: int = 13) -> Tuple[List[str], List[int]]:
        """Generate cards with their display values."""
        if not ood:
            cards_num = [random.randint(1, largest_card) for _ in range(num_cards)]
        else:
            # For OOD, ensure we have at least one face card (J, Q, K) but be more careful
            # Only generate face cards if largest_card >= 11
            if largest_card >= 11:
                cards_num = [random.randint(1, largest_card) for _ in range(num_cards - 1)] + [random.randint(11, min(13, largest_card))]
            else:
                # If largest_card < 11, just generate normal cards
                cards_num = [random.randint(1, largest_card) for _ in range(num_cards)]
        
        random.shuffle(cards_num)
        
        # Convert to display values using face card mapping
        display_card_nums = []
        cards_str = []
        
        for num in cards_num:
            card_str = self._num_to_card_str(num)
            cards_str.append(card_str)
            try:
                display_card_nums.append(self.face_mapper.map_face_card(card_str))
            except ValueError as e:
                # If we can't map a face card, skip this card combination
                raise ValueError(f"Cannot map card {card_str} (value {num}) with mapping {self.face_mapper.mapping_name}: {e}")
        
        return cards_str, display_card_nums
    
    def _num_to_card_str(self, num: int) -> str:
        """Convert numeric value to card string representation."""
        face_cards = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}
        return face_cards.get(num, str(num))
    
    def solve_game(self, digits: List[int], target: int) -> List[str]:
        """Find all solutions for the given digits and target."""
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

class DatasetGenerator:
    """Main class for generating training datasets with different face card mappings."""
    
    def __init__(self, target: int = 24, num_cards: int = 4, data_source: str = "gp-l"):
        self.target = target
        self.num_cards = num_cards
        self.data_source = data_source
        
    def generate_task(self, task_id: int, face_mapper: FaceCardMapper, 
                     seed: Optional[int] = None, ood: bool = False, 
                     largest_card: int = 13) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a single task instance."""
        if seed is not None:
            random.seed(seed)
        
        game_gen = CardGameGenerator(face_mapper)
        cards_str, display_card_nums = game_gen.generate_cards(
            num_cards=self.num_cards, ood=ood, largest_card=largest_card
        )
        
        solutions = game_gen.solve_game(display_card_nums, self.target)
        
        # Retry if no solution found
        attempts = 0
        while not solutions and attempts < 50:
            cards_str, display_card_nums = game_gen.generate_cards(
                num_cards=self.num_cards, ood=ood, largest_card=largest_card
            )
            solutions = game_gen.solve_game(display_card_nums, self.target)
            attempts += 1
        
        if not solutions:
            raise ValueError(f"No solution found for task {task_id}")
        
        solution = random.choice(solutions)
        cards_json = json.dumps(cards_str)
        formatted_solution = f"{{\n\"cards\": {cards_json},\n \"number\": {display_card_nums},\n \"formula\": \"{solution}\"\n}}"
        
        task_prompt = TASK_TEMPLATE.format(
            target_number=self.target,
            face_card_msg=face_mapper.get_face_card_message(),
            cards=cards_str,
            num_cards=self.num_cards,
        )
        
        extra_info = {
            "index": task_id,
            "cards": cards_str,
            "display_cards": display_card_nums,
            "solution": solutions,
            "target": self.target,
            "face_card_mapping": face_mapper.mapping_name,
        }
        
        sft_instance = {
            "data_source": self.data_source+f"-{face_mapper.mapping_name}",
            "question": task_prompt,
            "answer": formatted_solution,
            "extra_info": extra_info
        }
        
        rl_instance = {
            "data_source": self.data_source+f"-{face_mapper.mapping_name}",
            "question": [{"role": "user", "content": task_prompt}],
            "extra_info": extra_info
        }
        
        return sft_instance, rl_instance
    
    def generate_dataset(self, mapping_names: List[str], tasks_per_mapping: int = 1000,
                        ood: bool = False, largest_card: int = 13) -> Tuple[List[Dict], List[Dict]]:
        """Generate dataset with multiple face card mappings."""
        sft_datapoints = []
        rl_datapoints = []
        
        for mapping_name in mapping_names:
            print(f"Generating {tasks_per_mapping} tasks with mapping: {mapping_name}")
            face_mapper = FaceCardMapper(mapping_name)
            
            for task_id in tqdm(range(tasks_per_mapping), desc=f"Mapping: {mapping_name}"):
                try:
                    sft_instance, rl_instance = self.generate_task(
                        task_id=len(sft_datapoints),
                        face_mapper=face_mapper,
                        seed=42 + len(sft_datapoints),
                        ood=ood,
                        largest_card=largest_card
                    )
                    sft_datapoints.append(sft_instance)
                    rl_datapoints.append(rl_instance)
                except ValueError as e:
                    print(f"Skipping task: {e}")
                    continue
        
        return sft_datapoints, rl_datapoints

def get_dataset_features():
    """Get the dataset features schema."""
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
    """Main function to generate training dataset."""
    # Configuration
    dataset_id = "Xiaofeng77/gp-l-only-20k-mixed"
    tasks_per_mapping = 2500  # Adjust as needed
    ood = True
    largest_card = 13
    
    # Select which mappings to use for training
    # You can choose from: "basic", "diverse", "extensive", or "custom"
    # Or specify your own list of mapping names
    preset_name = "diverse"  # Change this to use different presets
    training_mappings = TRAINING_PRESETS[preset_name]
    
    # Alternatively, you can specify custom mappings:
    # training_mappings = ["all_10", "mixed_9_10_11", "sequential_8_9_10"]
    
    print(f"Available mappings: {list(FACE_CARD_MAPPINGS.keys())}")
    print(f"Selected for training: {training_mappings}")
    
    # Generate dataset
    generator = DatasetGenerator(target=24, num_cards=4, data_source="gp-l")
    sft_datapoints, rl_datapoints = generator.generate_dataset(
        mapping_names=training_mappings,
        tasks_per_mapping=tasks_per_mapping,
        ood=ood,
        largest_card=largest_card
    )
    
    print(f"Generated {len(sft_datapoints)} SFT instances and {len(rl_datapoints)} RL instances")
    
    # Create datasets
    features = get_dataset_features()
    sft_dataset = Dataset.from_list(sft_datapoints)
    rl_dataset = Dataset.from_list(rl_datapoints, features=features)
    
    # Save datasets
    sft_dataset.to_parquet("./data/gp-l-only/mixed/sft/train.parquet")
    rl_dataset.to_parquet("./data/gp-l-only/mixed/rl/train.parquet")
    
    # Push to hub
    rl_dataset.push_to_hub(dataset_id, split="train")
    print("Dataset saved and pushed to hub successfully!")

if __name__ == "__main__":
    main()
