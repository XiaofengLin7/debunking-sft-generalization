import hydra
from reil.trainer.llm_agent.agent_proxy import VllmWrapperWg, HFWrapperWg, LLMAgentProxy
from reil.trainer.llm_agent.es_manager import EnvStateManager
from reil.trainer.llm_agent.ctx_manager import NaiveContextManager

from transformers import AutoTokenizer
import os
from typing import List, Dict
from verl import DataProto
from pprint import pprint
from tqdm import tqdm
@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    run_eval(config)

def run_eval(config):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ['ALFWORLD_DATA'] = "/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    actor_wg = VllmWrapperWg(config, tokenizer)
    # actor_wg = HFWrapperWg(module=None, config=config, tokenizer=tokenizer)
    proxy = LLMAgentProxy(config, actor_wg, tokenizer)
    es_manager = EnvStateManager(config, mode="val")
    ctx_manager = NaiveContextManager(config, tokenizer, processor=None, mode="val")
    import time
    start_time = time.time()
    env_outputs = es_manager.reset()
    end_time = time.time()
    print(f"Loading envs takes: {end_time - start_time} seconds")
    meta_info = {
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
        'recompute_log_prob': False,
        'do_sample': False,
        'validate': True,
    }
    
    start_time = time.time()
    # rollouts = proxy.rollout()
    for i in tqdm(range(config.agent_proxy.max_turn)):
        lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
        lm_inputs.meta_info = meta_info 
        lm_outputs: DataProto = proxy.generate_sequences(lm_inputs)
        env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
        env_outputs: List[Dict] = es_manager.step(env_inputs)
        if len(env_outputs) == 0: # all finished
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    rollout_states = es_manager.get_rollout_states() 
    rollouts = ctx_manager.formulate_rollouts(rollout_states)
    pprint(rollouts.meta_info)

if __name__ == "__main__":
    main()
