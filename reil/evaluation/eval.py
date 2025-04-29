import hydra
from reil.trainer.llm_agent.agent_proxy import VllmWrapperWg, LLMAgentProxy
from transformers import AutoTokenizer
import os
from reil.env.sokoban.env import SokobanEnvReil

@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    run_eval(config)

def run_eval(config):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)

    env = SokobanEnvReil(dim_room=(config.env.dim_x, config.env.dim_y), num_boxes=config.env.num_boxes, max_steps=config.env.max_steps, search_depth=config.env.search_depth)
    obs = env.reset(seed=42)
    for i in range(config.max_turn):
        print(f"Turn {i+1}")
    

if __name__ == "__main__":
    main()
