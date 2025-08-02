import gymnasium as gym
import random
from itertools import permutations, product, chain, zip_longest
from fractions import Fraction as F
from reil.env.utils.prompts import Q_GeneralPoint_EQN_L
from reil.utils.dataset.create_dataset_gp_l import card_num_to_str, card_str_to_num
from reil.utils.reward_score.gp_l import re_match, robust_str_to_list, calculate_rewards, REWARD_FN 
from reil.env.gp_l.config import GPLEnvConfig

class GPLEnv(gym.Env):
    def __init__(self, config):
        # Initialize environment with configuration
        self.config = config
        self.target = config.target
        self.num_cards = config.num_cards
        self.treat_face_cards_as_10 = config.treat_face_cards_as_10
        self.ood = config.ood
        self.face_card_msg = "'J', 'Q', and 'K' count as '10'" if self.treat_face_cards_as_10 \
                            else "'J', 'Q', and 'K' count as '11', '12', and '13' respectively"
        
    def solve(self, digits: list):
        """
            Code obtained from here: https://rosettacode.org/wiki/24_game/Solve#Python
            This function takes a list of 4 digits and returns
            True if a solution exists, False otherwise.
            If true, we also save the solution.
        """
        digilen = len(digits)
        # length of an exp without brackets
        exprlen = 2 * digilen - 1
        # permute all the digits
        # added shuffle to avoid always the same solution
        digiperm = sorted(set(permutations(digits)))
        random.shuffle(digiperm)
        # All the possible operator combinations
        opcomb = list(product('+-*/', repeat=digilen-1))
        # All the bracket insertion points:
        brackets = ([()] + [(x, y)
                            for x in range(0, exprlen, 2)
                            for y in range(x+4, exprlen+2, 2)
                            if (x, y) != (0, exprlen+1)]
                    + [(0, 3+1, 4+2, 7+3)])  # double brackets case
        self.solution = []
        for d in digiperm:
            for ops in opcomb:
                if '/' in ops:
                    d2 = [('F(%s)' % i) for i in d]  # Use Fractions for accuracy
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
                    if num == self.target:
                        if '/' in ops:
                            exp = [(term if not term.startswith('F(') else term[2:-1])
                                for term in exp]
                        ans = ''.join(str(i) for i in exp).rstrip()
                        self.solution.append(ans)
        if len(self.solution) > 0:
            return True
        else:
            return False

    def _generate_cards(self):
        if not self.ood:
            cards_num = [random.randint(1, 13) for _ in range(self.num_cards)]
        else:
            cards_num = [random.randint(1, 13) for _ in range(self.num_cards - 1)] + [random.randint(11, 13)]
            # shuffle the cards
            random.shuffle(cards_num)

        if self.treat_face_cards_as_10:
            cards_num = [min(x, 10) for x in cards_num]
        
        cards_str = [card_num_to_str(num) for num in cards_num]

        return cards_str, cards_num
    
    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        Args:
            seed: Optional seed for reproducibility
        Returns:
            Initial observation (rendered environment state)
        """
        super().reset(seed=seed)
        random.seed(seed)
        self.cards, self.cards_num = self._generate_cards()
        while not self.solve(self.cards_num):
            self.cards, self.cards_num = self._generate_cards()

        
    
    def step(self, action):
        """
        Execute one step in the environment.
        Args:
            action: Action to take (string for text-based actions)
        Returns:
            Tuple of (observation, reward, done, info)
            - observation: rendered environment state
            - reward: float reward for this step
            - done: boolean indicating if episode is finished
            - info: dict with additional information (must include 'success' key)
        """
        terminated, reward, info = False, 0, {}
        current_formula = re_match(action, 'formula')
        recognized_cards = re_match(action, 'cards')
        translated_number = re_match(action, 'number')
        try:
            current_formula = current_formula.split('=')[0]
        except:
            pass
        
        num_cards = len(self.cards)
        digits = self.cards_num
        target = self.target
        solutions = self.solution
        recognized_cards = robust_str_to_list(recognized_cards, num_cards)
        translated_number = robust_str_to_list(translated_number, num_cards)
        
        # print(recognized_cards, translated_number, current_formula)
        reward = calculate_rewards(card_nums=digits, 
                                   current_formula=current_formula, 
                                   solutions=solutions, 
                                   target_points=target, 
                                   recognized_cards=recognized_cards, 
                                   translated_number=translated_number, 
                                   gt_cards=self.cards)

        if reward == max(REWARD_FN.values()):
            terminated = True
        info = {"success": terminated}
        return self.render(), reward, terminated, info
    
    def render(self):
        """
        Render the current environment state.
        Returns:
            String representation of the environment state
        """
        return Q_GeneralPoint_EQN_L.format(target_number=self.target, 
        face_card_msg=self.face_card_msg, cards=self.cards, cards_num=self.cards_num)
    
    def close(self):
        """
        Clean up environment resources.
        """
        super().close()

class GPLEnvFaceCardsAs10(GPLEnv):
    """
    GP-L environment with face cards always treated as 10 when calculating rewards, but rendered as 11, 12, and 13 respectively
    to test if there is any shortcut learning.
    """
    def __init__(self, config):
        self.config = config
        self.target = config.target
        self.num_cards = config.num_cards
        # treat cards as 10 by default is False
        self.treat_face_cards_as_10 = True  
        self.ood = True
        self.face_card_msg = "'J', 'Q', and 'K' count as '11', '12', and '13' respectively"
    
    def _generate_cards(self):
        if not self.ood:
            cards_num = [random.randint(1, 13) for _ in range(self.num_cards)]
        else:
            cards_num = [random.randint(1, 13) for _ in range(self.num_cards - 1)] + [random.randint(11, 13)]
            # shuffle the cards
            random.shuffle(cards_num)
        cards_str = [card_num_to_str(num) for num in cards_num]

        # treat face cards as 10 to calculate rewards
        cards_num = [min(x, 10) for x in cards_num]

        return cards_str, cards_num
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        random.seed(seed)
        self.cards, self.cards_num = self._generate_cards()
        # covert self.cards to number treating face cards as 11, 12 and 13
        self.real_cards_num = [card_str_to_num(card) for card in self.cards]
        while not self.solve(self.real_cards_num) or not self.solve(self.cards_num):
            self.cards, self.cards_num = self._generate_cards()
            self.real_cards_num = [card_str_to_num(card) for card in self.cards]

    





if __name__ == "__main__":
    print("Testing card_str_to_num function:")
    print(f"card_str_to_num('A') = {card_str_to_num('A')}")
    print(f"card_str_to_num('J') = {card_str_to_num('J')}")
    print(f"card_str_to_num('10') = {card_str_to_num('10')}")
    
    config = GPLEnvConfig()
    env = GPLEnv(config)
    env.reset(seed=42)
    action = """
    {
        "cards": ["10", "5", "10", "A"],
        "number": [10, 5, 10,   1],
        "formula": "(10 + 10 + 5 -1)",
    }
    """
    print(env.render())
    print(env.step(action))
    config1 = GPLEnvConfig(treat_face_cards_as_10=True, ood=True)
    env1 = GPLEnvFaceCardsAs10(config1)
    env1.reset(seed=42)

    print(env1.solution)

    action1 = """
    {
        "cards": ["2", "7", "Q", "2"],
        "number": [2, 7, 10,   2],
        "formula": "(10/2+7)*2",
    }
    """
    print(env1.step(action1))


