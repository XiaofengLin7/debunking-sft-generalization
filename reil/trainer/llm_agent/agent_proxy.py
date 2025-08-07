"""
Adapted from https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/llm_agent/agent_proxy.py
"""

from vllm import LLM, SamplingParams
from typing import List, Dict, Union, Any
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
from verl.workers.rollout.hf_rollout import HFRollout
from torch import nn
from reil.trainer.llm_agent.es_manager import EnvStateManager
from reil.trainer.llm_agent.ctx_manager import NaiveContextManager
import time
from tqdm import tqdm
from pprint import pprint
import torch
from tensordict import TensorDict

class Config:
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)
	
	def get(self, key: str, default: Any = None) -> Any:
		return getattr(self, key, default)

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer, model_path: Union[str, None]=None):
		if model_path is None:
			model_path = config.actor_rollout_ref.model.path
		self.config = config
		self.tokenizer = tokenizer
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_path,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			# tensor_parallel_size=config.evaluator.n_gpus_per_node,
		)
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def load_checkpoint(self, checkpoint_path: str):

		ro_config = self.config.actor_rollout_ref.rollout
		self.llm = LLM(
			checkpoint_path,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		attention_mask = lm_inputs.batch['attention_mask']

		# print(f"lm_inputs.batch: {lm_inputs.batch}")
		pad_token_id = lm_inputs.meta_info['pad_token_id']
		
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params, use_tqdm=False)
		texts = [output.outputs[0].text for output in outputs] 
		# print(f"texts: {texts[0]}")
		prompt_token_ids = [torch.tensor(output.prompt_token_ids, dtype=input_ids.dtype) for output in outputs]
		response_token_ids = [torch.tensor(output.outputs[0].token_ids, dtype=input_ids.dtype) for output in outputs]
		
		prompts_padded = torch.nn.utils.rnn.pad_sequence(prompt_token_ids, batch_first=True, padding_value=pad_token_id, padding_side="right")
		responses_padded = torch.nn.utils.rnn.pad_sequence(response_token_ids, batch_first=True, padding_value=pad_token_id, padding_side="right")
		print(f"responses_padded.shape: {responses_padded.shape}")

		input_ids = torch.cat([prompts_padded, responses_padded], dim=1)
		input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id, padding_side="right")

		attention_mask = torch.where(input_ids_padded != pad_token_id, 1, 0)
		
		batch_size = len(input_texts)
		batch = TensorDict(
			{
				'prompts': prompts_padded,
				'responses': responses_padded,
				'input_ids': input_ids_padded,
				'attention_mask': attention_mask,
			},
			batch_size=batch_size
		)
		lm_outputs = DataProto()
		lm_outputs.batch = batch

		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			# 'data_source': lm_inputs.non_tensor_batch['data_source'] if 'data_source' in lm_inputs.non_tensor_batch else None,
			# 'env_ids': lm_inputs.non_tensor_batch['env_ids'] if 'env_ids' in lm_inputs.non_tensor_batch else None,
			# 'group_ids': lm_inputs.non_tensor_batch['group_ids'] if 'group_ids' in lm_inputs.non_tensor_batch else None
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		if 'env_ids' in lm_inputs.non_tensor_batch.keys():
			lm_outputs.non_tensor_batch['env_ids'] = lm_inputs.non_tensor_batch['env_ids']
		if 'group_ids' in lm_inputs.non_tensor_batch.keys():
			lm_outputs.non_tensor_batch['group_ids'] = lm_inputs.non_tensor_batch['group_ids']
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class HFWrapperWg:
	def __init__(self, config, tokenizer, module: Union[nn.Module, None] = None):
		if module is None:
			self.module = AutoModelForCausalLM.from_pretrained(config.actor_rollout_ref.model.path, device_map="cuda")
		else:
			self.module = module
		self.config = config
		self.tokenizer = tokenizer
		self.rollout_config = Config(
			micro_batch_size=config.es_manager.val.env_groups,
			response_length=config.actor_rollout_ref.rollout.response_length,
			do_sample=config.actor_rollout_ref.rollout.do_sample,
			temperature=config.actor_rollout_ref.rollout.val_kwargs.temperature,
			top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
			top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
		)
		self.llm = HFRollout(self.module, self.rollout_config)
	
	def load_checkpoint(self, checkpoint_path: str):
		self.module = AutoModelForCausalLM.from_pretrained(checkpoint_path, device_map="cuda")
		# Recreate HFRollout with the new module
		self.llm = HFRollout(self.module, self.rollout_config)

	def generate_sequences(self, lm_inputs: DataProto):
		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
		lm_inputs.batch['input_ids'] = inputs.input_ids
		lm_inputs.batch['attention_mask'] = inputs.attention_mask
		position_ids = inputs.attention_mask.cumsum(dim=-1)
		lm_inputs.batch['position_ids'] = position_ids

		lm_outputs = self.llm.generate_sequences(lm_inputs)
		lm_outputs.non_tensor_batch = {
			'env_ids': lm_inputs.non_tensor_batch['env_ids'] if 'env_ids' in lm_inputs.non_tensor_batch else None,
			'group_ids': lm_inputs.non_tensor_batch['group_ids'] if 'group_ids' in lm_inputs.non_tensor_batch else None
		}
		lm_outputs.meta_info = lm_inputs.meta_info
		return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer
		self.val_ctx_manager = NaiveContextManager(config, tokenizer, processor=None, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, HFWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs
	
	def set_actor_wg(self, actor_wg):
		self.actor_wg = actor_wg
	
	def rollout(self):
		start_time = time.time()
		env_outputs = self.val_es_manager.reset(seed=self.config.evaluator.seed)
		end_time = time.time()
		print(f"Loading envs takes: {end_time - start_time} seconds")
		meta_info = {
			'eos_token_id': self.tokenizer.eos_token_id,
			'pad_token_id': self.tokenizer.pad_token_id,
			'recompute_log_prob': False,
			'do_sample': self.config.actor_rollout_ref.rollout.do_sample,
			'validate': True,
		}
		for _ in tqdm(range(self.config.agent_proxy.max_turn), desc="Agent turns"):
			lm_inputs: DataProto = self.val_ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
			lm_inputs.meta_info = meta_info 
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = self.val_ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = self.val_es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
		rollout_states = self.val_es_manager.get_rollout_states() 
		rollouts = self.val_ctx_manager.formulate_rollouts(rollout_states)
		return rollouts
