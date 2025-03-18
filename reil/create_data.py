# create behavior cloning dataset for sokoban agent
# adapted from ragen/sft/utils/generate_sft_verl_sokoban.py

from ragen.env.sokoban import SokobanEnv
from ragen.env.sokoban.room_utils import get_shortest_action_path

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


[Cumulative Observations]:
{observation}
Decide the next action:\
"""

######################################## different language model input templates ########################################

qwen_instruction_template = """\
<|im_start|>system
You are a helpful assistant. <|im_end|>
<|im_start|>user
{prompt}
Always output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra test. Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>\
"""
qwen_observation_template = """\
<|im_start|>user
After you take this action, the observation is: 
{observation}
<|im_end|>
<|im_start|>assistant
<think>\
"""
qwen_response_template = """\
</think> <answer> {action} </answer> <|im_end|>
"""

system_message = "You are a helpful assistant."
instruction_message = "{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra test. Strictly follow this format."
observation_message = "After you take this action, the observation is: \n{observation}\n"
response_message = "<answer> {action} </answer>"

# NOTE in message format, <think> token is not included here

base_instruction_template = """\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.
User: {prompt}
Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>
Assistant: 
<think>\
"""
base_observation_template = """\
User: After you take this action, the observation is: {observation}
Assistant: 
<think>\
"""
base_response_template = """\
</think> <answer> {action} </answer>
"""

templates = {
    'qwen-instruct': {
        'instruction': qwen_instruction_template,
        'observation': qwen_observation_template,
        'response': qwen_response_template
    },
    'message': {
        'system': system_message,
        'instruction': instruction_message,
        'observation': observation_message,
        'response': response_message
    },
    'base': {
        'instruction': base_instruction_template,
        'observation': base_observation_template,
        'response': base_response_template
    }
}

def main():
    # Sokoban environment settings
    dim_x = 6
    dim_y = 6
    num_boxes = 1

    # generate normal difficulty dataset
    max_steps = 10
    search_depth = 30

    env = SokobanEnv(dim_room=(dim_x, dim_y), num_boxes=num_boxes, max_steps=max_steps, search_depth=search_depth)
    gt_action_sequence = get_shortest_action_path(env.room_fixed, env.room_state, MAX_DEPTH=search_depth)
    assert gt_action_sequence, f"No action sequence found." # for seed {seed}
    for action in gt_action_sequence:
        obs, reward, done, info = env.step(action)
        print(obs)

if __name__ == "__main__":
    main()