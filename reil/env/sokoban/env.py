from ragen.env.sokoban.env import SokobanEnv
import numpy as np
from ragen.utils import NoLoggerWarnings
from ragen.env.sokoban.room_utils import generate_room
from ragen.utils import set_seed

class SokobanEnvText(SokobanEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, mode='text', seed=None):
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
        
    def render(self, mode='text'):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array', 'text']

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
        
if __name__ == "__main__":
    env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=30)
    print(env.reset(mode='text', seed=1010))
