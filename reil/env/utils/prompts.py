SOKOBAN_INSTRUCTION_TEMPLATE = """You are a Sokoban solver.

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

ALFWORLD_PROMPT_TEMPLATE = {
    "llm_system_prompt": (
        "You are an household agent designed to interact with a simulated household environment to solve household tasks step by step. "
        "In this environment, you can interact with objects and receptacles to solve the task."
        "After you execute an action, you will receive a textual feedback from the environment."
    ),
    "llm_action_prompt": (
        "Specify the next action the agent should take to progress toward the task goal, following these guidelines:\n\n"
        "1. Object and Receptacle References: Use specific identifiers:\n"
        "   - [obj id] for objects (e.g., apple 1).\n"
        "   - [recep id] for receptacles (e.g., countertop 1).\n"
        "2. Action Validity: Follow the exact format below. Any deviation renders the action invalid:\n"
        "Valid actions: go to [recep id], take [obj id] from [recep id], put [obj id] in/on [recep id], open/close [recep id], use [obj id], heat/cool/clean [obj id] with [recep id]\n"
    )
}

ALFWORLD_INSTRUCTION_PROMPT = ALFWORLD_PROMPT_TEMPLATE['llm_system_prompt']  + "Here is the task:\n{history}" + ALFWORLD_PROMPT_TEMPLATE['llm_action_prompt']

templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}
