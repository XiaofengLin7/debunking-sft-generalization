from ragen.env.sokoban.env import SokobanEnv
import numpy as np
from ragen.utils import NoLoggerWarnings
from ragen.env.sokoban.room_utils import generate_room
from ragen.utils import set_seed
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer
from ragen.env.base import BaseEnv
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import copy
from .config import SokobanEnvConfig
from enum import Enum

class ActionFormat(Enum):
    """Enum for different action formats."""
    BASE = "base"
    CARDINAL = "cardinal" 
    EMOJI = "emoji"
    EMPTY = "empty"
    NUMERICAL = "numerical"
    ALPHABETICAL = "alphabetical"
    RANDOM = "random"

class InstructionTemplates:
    """Centralized template management with caching."""
    
    BASE_INSTRUCTION = """You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
# Wall | _ Floor | O Target | X Box | P You | √ = Box on Target | S = You on Target

Rules:
1. Push boxes (can't pull).
2. Avoid walls (#).

{action_section}

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0


[Current Observation]:
{observation}
Decide the next action:\\"""

    ACTION_SECTIONS = {
        ActionFormat.BASE: "Answers:\n<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>",
        ActionFormat.CARDINAL: "Answers:\n<answer> North </answer> | <answer> South </answer> | <answer> West </answer> | <answer> East </answer>",
        ActionFormat.EMOJI: "Answers:\n<answer> ⬆️ </answer> | <answer> ⬇️ </answer> | <answer> ⬅️ </answer> | <answer> ➡️ </answer>",
        ActionFormat.EMPTY: "",
        ActionFormat.NUMERICAL: "Answers:\n<answer> 1 </answer> | <answer> 2 </answer> | <answer> 3 </answer> | <answer> 4 </answer>\nwhere 1 is Up, 2 is Down, 3 is Left, 4 is Right.",
        ActionFormat.ALPHABETICAL: "Answers:\n<answer> A </answer> | <answer> B </answer> | <answer> C </answer> | <answer> D </answer>\nwhere A is Up, B is Down, C is Left, D is Right.",
        ActionFormat.RANDOM: "Answers:\n<answer> * </answer> | <answer> & </answer> | <answer> 1 </answer> | <answer> M </answer>\nwhere * is Up, & is Down, 1 is Left, M is Right."
    }

    CHAT_TEMPLATES = {
        'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
        'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
    }


class SokobanEnvReil(SokobanEnv):
    """Optimized Sokoban environment with configurable action formats."""
    
    # Class-level constants for better performance
    ACTION_LOOKUPS = {
        ActionFormat.BASE: {0: "None", 1: "Up", 2: "Down", 3: "Left", 4: "Right"},
        ActionFormat.CARDINAL: {0: "None", 1: "North", 2: "South", 3: "West", 4: "East"},
        ActionFormat.EMOJI: {0: "None", 1: "⬆️", 2: "⬇️", 3: "⬅️", 4: "➡️"},
        ActionFormat.EMPTY: {0: "None", 1: "Up", 2: "Down", 3: "Left", 4: "Right"},
        ActionFormat.NUMERICAL: {0: "None", 1: "1", 2: "2", 3: "3", 4: "4"},
        ActionFormat.ALPHABETICAL: {0: "None", 1: "A", 2: "B", 3: "C", 4: "D"},
        ActionFormat.RANDOM: {0: "None", 1: "*", 2: "&", 3: "1", 4: "M"}
    }

    def __init__(self, config=None, action_format: ActionFormat = ActionFormat.BASE):
        self.config = config or SokobanEnvConfig()
        self.action_format = action_format
        self.search_depth = self.config.search_depth
        self.dim_room = self.config.dim_room
        self.num_boxes = self.config.num_boxes
        self.prefix = self.config.prefix or 'qwen-instruct'
        
        # Set action lookup based on format
        self.ACTION_LOOKUP = self.ACTION_LOOKUPS[action_format]
        
        self.action_section = InstructionTemplates.ACTION_SECTIONS[action_format]
        
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
        
        obs = self.render(mode='complete')
        info = {
            "action_is_effective": not np.array_equal(prev_player_position, self.player_position),
            "action_is_valid": action != self.INVALID_ACTION,
            "success": self.success()
        }
        return obs, reward, done, info
    
    def reset(self, mode='complete', seed=None):
        """Reset environment with improved error handling."""
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
                print(f"[SOKOBAN] Runtime Error/Warning: {e}")
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(mode, next_seed)
            
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0
            self.action_sequence = action_sequence
            return self.render(mode)


    def render(self, mode='complete'):
        """Optimized render method with caching."""
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array', 'text', 'complete']

        if mode == 'rgb_array':
            return self.get_image(mode, scale=1)

        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        if mode == 'complete':
            map_str = self.render(mode='tiny_rgb_array')
            prompt = InstructionTemplates.BASE_INSTRUCTION.format(action_section=self.action_section, observation=map_str)
            return InstructionTemplates.CHAT_TEMPLATES[self.prefix].format(prompt=prompt)
        
        # Handle other modes
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
        
        if mode == 'text':
            return self._render_text_description(room_state)

    def _render_text_description(self, room_state):
        """Generate text description of the game state."""
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
        
        description = [f"map size: {elements['size']}"]
        for key, positions in elements.items():
            if positions and key != 'size':
                description.append(f"{key.replace('_', ' ')}: {positions}")
        
        description.append("rest are floors.")
        return "\n".join(description)
    
    def copy(self):
        """Optimized copy method without deep copying simple objects."""
        new_self = self.__class__(config=self.config, action_format=self.action_format)
        
        # Copy numpy arrays efficiently
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        
        # Copy simple attributes
        new_self.reward = self.reward
        
        # Only deep copy if _valid_actions exists and is complex
        if hasattr(self, '_valid_actions') and self._valid_actions:
            new_self._valid_actions = copy.deepcopy(self._valid_actions)
            
        return new_self
    
    def close(self):
        """Clean up resources."""
        self.render_cache = None
        super().close()

# Factory function to create environments with different action formats
def create_sokoban_env(action_format: ActionFormat, config=None):
    """Factory function to create Sokoban environments with different action formats."""
    return SokobanEnvReil(config=config, action_format=action_format)

# Backward compatibility classes
class SokobanEnvReilCardinal(SokobanEnvReil):
    """Backward compatibility wrapper for cardinal directions."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.CARDINAL)

class SokobanEnvReilEmoji(SokobanEnvReil):
    """Backward compatibility wrapper for emoji actions."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.EMOJI)

class SokobanEnvReilEmpty(SokobanEnvReil):
    """Backward compatibility wrapper for empty instruction."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.EMPTY)

class SokobanEnvReilNumerical(SokobanEnvReil):
    """Backward compatibility wrapper for numerical actions."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.NUMERICAL)

class SokobanEnvReilAlphabetical(SokobanEnvReil):
    """Backward compatibility wrapper for alphabetical actions."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.ALPHABETICAL)

class SokobanEnvReilRandom(SokobanEnvReil):
    """Backward compatibility wrapper for random actions."""
    def __init__(self, config=None):
        super().__init__(config=config, action_format=ActionFormat.RANDOM)

if __name__ == "__main__":
    # Example usage showing improved flexibility
    config = SokobanEnvConfig()
    config.dim_room = (6, 6)
    config.num_boxes = 1
    config.max_steps = 100
    config.search_depth = 100
    
    # Test different action formats
    for action_format in ActionFormat:
        print(f"\nTesting {action_format.value} format:")
        env = SokobanEnvReil(config, action_format)
        obs = env.reset(seed=42)
        print(obs)
        # print(f"Action lookup: {env.ACTION_LOOKUP}")
        print(f"Sample observation length: {len(obs)}")
        env.close()
