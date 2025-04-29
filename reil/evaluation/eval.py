import hydra
from reil.trainer.llm_agent.agent_proxy import VllmWrapperWg, LLMAgentProxy
from transformers import AutoTokenizer
import os
@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    run_eval(config)

def run_eval(config):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)

    for i in range(config.max_turn):
        print(f"Turn {i+1}")
    

if __name__ == "__main__":
    main()
