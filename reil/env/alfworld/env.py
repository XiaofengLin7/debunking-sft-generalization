from thirdparty.alfworld.alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
from typing import List, Dict


def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    query = base_query
    query += f"\nHere is the task:\n{start_info}"
    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'

    return query

class EnvironmentHistory:
    """
    History of the environment.
    Adapted from https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/env_history.py
    """
    def __init__(self, start_info,  history: List[Dict[str, str]] = []) -> None:
        self._start_info: str = start_info
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        return self._is_exhausted

    def reset(self) -> None:
        self._history = []

    def __str__(self) -> str:
        s: str = self._start_info + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'act: {item["value"]}'
            elif item['label'] == 'observation':
                s += f'obs: {item["value"]}'
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                s += '\n'
        return s

def process_ob(ob):
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


class ALFWorldTW(AlfredTWEnv):
    def __init__(self, config, train_eval='train'):
        assert config['env']['type'] == 'AlfredTWEnv', "ALFWorldTW only supports AlfredTWEnv"
        super().__init__(config, train_eval)
        self.env = self.init_env(batch_size=1)
        self.task_type = None

    def get_history(self):
        return str(self.history)

    def get_s_a_history(self):
        s_a_history = []
        cur_s = self.history._start_info+'\n'
        for i, item in enumerate(self.history._history):
            if item['label'] == 'action':
                s_a_history.append(
                    {
                        "state": cur_s,
                        "action": item["value"]
                    }
                )
                cur_s += f'act: {item["value"]}'
            elif item['label'] == 'observation':
                cur_s += f'obs: {item["value"]}'
            if i != len(self.history._history) - 1:
                cur_s += '\n'

        return s_a_history
    def get_task_type(self):
        return self.task_type

    def step(self, action):
        obs, scores, dones, infos = self.env.step(action)
        obs = process_ob(obs[0])
        if not any(cmd in action[0] for cmd in ["look", "examine", "inventory"]):
            self.history.add(label='action', value=action[0])
            self.history.add(label='observation', value=obs)
        return obs, scores, dones, infos
    

    def reset(
        self,
        seed=42,
    ):
        self.env.seed(seed)
        obs, infos = self.env.reset()
        start_info = '\n'.join(obs[0].split('\n\n')[1:])
        self.history = EnvironmentHistory(
            start_info=start_info,
            history=[]
        )
        if infos["extra.gamefile"][0] is not None:
            if "pick_and_place" in infos["extra.gamefile"][0]:
                self.task_type = "pick_and_place"
            elif "pick_two_obj_and_place" in infos["extra.gamefile"][0]:
                self.task_type = "pick_two_obj_and_place"
            elif "look_at_obj_in_light" in infos["extra.gamefile"][0]:
                self.task_type = "look_at_obj_in_light"
            elif "pick_heat_then_place_in_recep" in infos["extra.gamefile"][0]:
                self.task_type = "pick_heat_then_place_in_recep"
            elif "pick_cool_then_place_in_recep" in infos["extra.gamefile"][0]:
                self.task_type = "pick_cool_then_place_in_recep"
            elif "pick_clean_then_place_in_recep" in infos["extra.gamefile"][0]:
                self.task_type = "pick_clean_then_place_in_recep"
        else:
            self.task_type = None
        return obs, infos
    