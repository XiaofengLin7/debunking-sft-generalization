from ragen.env.sokoban.env import SokobanEnv
import numpy as np

class SokobanEnv(SokobanEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
            elements = {
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
            for key, positions in elements.items():
                if positions:  # Skip empty lists and None values
                    if key == 'player' or key == 'player_on_target':
                        description.append(f"{key.replace('_', ' ')}: {positions}")
                    else:
                        description.append(f"{key.replace('_', ' ')}: {positions}")
            
            return "\n ".join(description)
        
if __name__ == "__main__":
    env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=100, search_depth=30)
    print(env.reset(mode='text', seed=1010))
