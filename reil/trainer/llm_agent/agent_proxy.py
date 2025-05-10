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
from types import SimpleNamespace

class Config:
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)
	
	def get(self, key: str, default: Any = None) -> Any:
		return getattr(self, key, default)

class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
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
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class HFWrapperWg:
	def __init__(self, config, tokenizer, module: Union[nn.Module, None] = None):
		if module is None:
			module = AutoModelForCausalLM.from_pretrained(config.actor_rollout_ref.model.path, device_map="cuda")
		self.config = config
		self.tokenizer = tokenizer
		HFRolloutConfig = Config(
			micro_batch_size=config.es_manager.val.env_groups,
			response_length=config.actor_rollout_ref.rollout.response_length,
			do_sample=config.actor_rollout_ref.rollout.do_sample,
			temperature=config.actor_rollout_ref.rollout.val_kwargs.temperature,
			top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
			top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k
		)
		self.llm = HFRollout(module, HFRolloutConfig)

	def generate_sequences(self, lm_inputs: DataProto):
		# input_ids = lm_inputs.batch['input_ids']
		# input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		# input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		# inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False)
		# lm_inputs.batch['input_ids'] = inputs.input_ids
		# lm_inputs.batch['attention_mask'] = inputs.attention_mask
		# position_ids = inputs.attention_mask.cumsum(dim=-1)
		# lm_inputs.batch['position_ids'] = position_ids

		lm_outputs = self.llm.generate_sequences(lm_inputs)
		lm_outputs.non_tensor_batch = {
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
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
