from ragen.env.sokoban.env import SokobanEnv
import numpy as np
from ragen.utils import NoLoggerWarnings
from ragen.env.sokoban.room_utils import generate_room
from ragen.utils import set_seed
from typing import List
import torch
from transformers import AutoTokenizer
from ragen.env.base import BaseEnv
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import copy
from .config import SokobanEnvConfig

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
Decide the next action:\
"""

CARDINAL_INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Answers:
<answer> North </answer> | <answer> West </answer> | <answer> South </answer> | <answer> East </answer>

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Current Observation]:
{observation}
Decide the next action:\
"""

EMOJI_INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Answers:
<answer> ⬆️ </answer> | <answer> ⬇️ </answer> | <answer> ⬅️ </answer> | <answer> ➡️ </answer>

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Current Observation]:
{observation}
Decide the next action:\
"""

EMPTY_INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Current Observation]:
{observation}
Decide the next action:\
"""
templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}

class SokobanEnvReil(SokobanEnv):
    def __init__(self, config=None):
        self.config = config or SokobanEnvConfig()
        self.search_depth = self.config.search_depth
        self.dim_room = self.config.dim_room
        self.num_boxes = self.config.num_boxes
        self.prefix = self.config.prefix or 'qwen-instruct'
        super().__init__(
            dim_room=self.dim_room,
            max_steps=self.config.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )

        
    def step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).
        """
        assert not self.success()

        if action == self.INVALID_ACTION:
            return self.render(), 0, False, {"action_is_effective": False}
        prev_player_position = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action, observation_mode='tiny_rgb_array')
        
        # # NOTE re-define reward for sokoban
        # reward = -1 # format reward
        # if done:
        #     reward = 1 # success reward
            
        obs = self.render(mode='complete')
        info = {"action_is_effective": not np.array_equal(prev_player_position, self.player_position),
                "success": self.success()}
        return obs, reward, done, info
    
    def reset(self, mode='complete', seed=None):
        self._reset_tracking_variables()
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(mode, next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
            return self.render(mode)
        
    def render(self, mode='complete'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array', 'text', 'complete']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img

        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
        if mode == 'complete':
            map = self.render(mode='tiny_rgb_array')
            return templates[self.prefix].format(prompt=INSTRUCTION_TEMPLATE.format(observation=map))
        
        if mode == 'text':
            # Get map dimensions from room_state
            height, width = len(room_state), len(room_state[0])
            
            elements = {
                'size': f"({height}, {width})",
                'walls': [],
                'targets': [],
                'boxes_on_targets': [],
                'boxes': [],
                'player': None,
                'player_on_target': None
            }
            
            for i, row in enumerate(room_state):
                for j, cell in enumerate(row):
                    pos = (i, j)
                    if cell == 0:
                        elements['walls'].append(pos)
                    elif cell == 2:
                        elements['targets'].append(pos)
                    elif cell == 3:
                        elements['boxes_on_targets'].append(pos)
                    elif cell == 4:
                        elements['boxes'].append(pos)
                    elif cell == 5:
                        elements['player'] = pos
                    elif cell == 6:
                        elements['player_on_target'] = pos
            
            description = []
            description.append(f"map size: {elements['size']}")
            for key, positions in elements.items():
                if positions and key != 'size':  # Skip empty lists, None values, and already handled size
                    description.append(f"{key.replace('_', ' ')}: {positions}")
            
            description.append("rest are floors.")

            return "\n".join(description)
    
    
    def copy(self):
        new_self = SokobanEnvReil(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    
    def close(self):
        self.render_cache = None
        super(SokobanEnvReil, self).close()


class SokobanEnvReilCardinal(SokobanEnvReil):
    """
    Variant of SokobanEnvReil with cardinal direction action space:
    - 1: North (corresponds to Up in original)
    - 2: South (corresponds to Down in original) 
    - 3: West (corresponds to Left in original)
    - 4: East (corresponds to Right in original)
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        # Override the ACTION_LOOKUP to use cardinal directions
        self.ACTION_LOOKUP = {
            0: "None",
            1: "North",
            2: "South", 
            3: "West",
            4: "East",
        }
    
    def extract_action(self, text):
        """
        Extract action from text for cardinal directions.
        - 0: Still (Invalid Action)
        - 1: North
        - 2: South
        - 3: West
        - 4: East
        """
        import re
        DIRECTION_MAP = {"North": 1, "South": 2, "West": 3, "East": 4}
        # Pattern to match cardinal directions or numbers
        pattern = r'^\s*(([1-4])\s*\((north|west|south|east)\)|(north|west|south|east)|([1-4]))\s*$'
        match = re.fullmatch(pattern, text.strip(), flags=re.IGNORECASE | re.X)
        
        if not match:
            return self.INVALID_ACTION
        
        if match.group(2):   
            return int(match.group(2))
        elif match.group(4): 
            return DIRECTION_MAP[match.group(4).capitalize()]
        elif match.group(5): 
            return int(match.group(5))
        
        return self.INVALID_ACTION
    
    def render(self, mode='complete'):
        if mode == 'complete':
            map = self.render(mode='tiny_rgb_array')
            return templates[self.prefix].format(prompt=CARDINAL_INSTRUCTION_TEMPLATE.format(observation=map))
        else:
            return super().render(mode)
    
    
    def copy(self):
        new_self = SokobanEnvReilCardinal(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    
    def close(self):
        self.render_cache = None
        super(SokobanEnvReilCardinal, self).close()
    
class SokobanEnvReilEmoji(SokobanEnvReil):
    """
    Variant of SokobanEnvReil with emoji action space:
    - 1: ⬆️
    - 2: ⬇️
    - 3: ⬅️
    - 4: ➡️
    """
    def __init__(self, config=None):
        super().__init__(config)
        # Override the ACTION_LOOKUP to use cardinal directions
        self.ACTION_LOOKUP = {
            0: "None",
            1: "⬆️",
            2: "⬇️", 
            3: "⬅️",
            4: "➡️",
        }
    
    def render(self, mode='complete'):
        if mode == 'complete':
            map = self.render(mode='tiny_rgb_array')
            return templates[self.prefix].format(prompt=EMOJI_INSTRUCTION_TEMPLATE.format(observation=map))
        else:
            return super().render(mode)
        
    def copy(self):
        new_self = SokobanEnvReilEmoji(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    
    def close(self):
        self.render_cache = None
        super(SokobanEnvReilEmoji, self).close()

class SokobanEnvReilEmpty(SokobanEnvReil):
    """
    Variant of SokobanEnvReil with empty action instruction:
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.ACTION_LOOKUP = {
            0: "None",
            1: "Up",
            2: "Down",
            3: "Left",
            4: "Right",
        }
    def render(self, mode='complete'):
        if mode == 'complete':
            map = self.render(mode='tiny_rgb_array')
            return templates[self.prefix].format(prompt=EMPTY_INSTRUCTION_TEMPLATE.format(observation=map))
        else:
            return super().render(mode)
        
    def copy(self):
        new_self = SokobanEnvReilEmpty(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    
    def close(self):
        self.render_cache = None
        super(SokobanEnvReilEmpty, self).close()
    
if __name__ == "__main__":
    env = SokobanEnvReil(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=30, prefix='base')
    print(env.reset(mode='complete', seed=1010))
