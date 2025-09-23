import hydra
import ray
import time
import contextlib
import gc
import wandb
from typing import List, Dict, Optional
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from enum import Enum
from verl import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from reil.trainer.llm_agent.agent_proxy import VllmWrapperWg, HFWrapperWg, LLMAgentProxy
from reil.trainer.llm_agent.es_manager import EnvStateManager
from reil.trainer.llm_agent.ctx_manager import NaiveContextManager
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils.tracking import Tracking
from omegaconf import OmegaConf
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import torch
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from reil.trainer.main_ppo import get_custom_reward_fn
import numpy as np
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    destroy_distributed_environment,
)
from verl.utils.tokenizer import hf_tokenizer
from verl.utils.fs import copy_to_local
#TODO: code needed to be optimized
@ray.remote(num_gpus=1)
def _kl_worker_remote(eval_ckpt: str,
                      ref_model_path: str,
                      trust_remote_code: bool,
                      temperature: float,
                      input_ids_cpu,
                      attention_mask_cpu,
                      responses_cpu,
                      position_ids_cpu,
                      micro_bs: int | None = None):
    import torch
    from transformers import AutoModelForCausalLM

    def _logprobs_from_logits(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logprobs = torch.log_softmax(logits, dim=-1)
        return torch.gather(logprobs, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def _compute_lp(model, input_ids, attention_mask, position_ids, responses, temperature):
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        responses = responses.to(device)
        if position_ids is None:
            position_ids = attention_mask.cumsum(dim=-1)
        else:
            position_ids = position_ids.to(device)
        response_length = responses.size(1)
        outputs = []
        batch_size = input_ids.size(0)
        use_mb = micro_bs is not None and isinstance(micro_bs, int) and micro_bs > 0 and micro_bs < batch_size
        if use_mb:
            for start in tqdm(range(0, batch_size, micro_bs), desc="Computing log-probs"):
                end = min(start + micro_bs, batch_size)
                ids = input_ids[start:end]
                am = attention_mask[start:end]
                pos = position_ids[start:end]
                rsp = responses[start:end]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids,
                                   attention_mask=am,
                                   position_ids=pos,
                                   use_cache=False).logits
                    logits = logits.div_(temperature)
                    logits = logits[:, -response_length - 1:-1, :]
                    lp = _logprobs_from_logits(logits, rsp)
                outputs.append(lp.detach().to('cpu'))
                del ids, am, pos, rsp, logits, lp
                torch.cuda.empty_cache()
            return torch.cat(outputs, dim=0)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids,
                               use_cache=False).logits
                logits = logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]
                log_probs = _logprobs_from_logits(logits, responses)
            return log_probs.detach().to('cpu')

    device = torch.device('cuda')
    # Load models sequentially to reduce peak memory
    eval_model = AutoModelForCausalLM.from_pretrained(
        eval_ckpt,
        trust_remote_code=trust_remote_code,
        attn_implementation='flash_attention_2'
    ).to(device)
    eval_model.eval()
    eval_lp = _compute_lp(eval_model, input_ids_cpu, attention_mask_cpu, position_ids_cpu, responses_cpu, temperature)
    del eval_model
    torch.cuda.empty_cache()

    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_path,
        trust_remote_code=trust_remote_code,
        attn_implementation='flash_attention_2'
    ).to(device)
    ref_model.eval()
    ref_lp = _compute_lp(ref_model, input_ids_cpu, attention_mask_cpu, position_ids_cpu, responses_cpu, temperature)
    del ref_model
    torch.cuda.empty_cache()

    response_length = responses_cpu.size(1)
    response_mask = attention_mask_cpu[:, -response_length:]
    kld = eval_lp - ref_lp
    kld = (kld * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1e-9)
    kld = torch.mean(kld, dim=0).item()
    return float(kld)

class Role(Enum):
    """
    Define roles for different components in the evaluation system
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2

class CheckpointEvaluator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=True)
        # Initialize environment and context managers
        self.es_manager = EnvStateManager(config, mode="val")
        self.ctx_manager = NaiveContextManager(config, self.tokenizer, processor=None, mode="val")
        # Initialize logger
        self.logger = Tracking(
            project_name=config.evaluator.project_name,
            experiment_name=config.evaluator.experiment_name,
            default_backend=config.evaluator.logger,
            config=OmegaConf.to_container(config, resolve=True)
        )
        if self.config.data.get('val_score_files', None):
            self.val_score_dataset = RLHFDataset(parquet_files=config.data.val_score_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        image_key=config.data.get('image_key', 'images'),
                                        max_prompt_length=config.data.max_prompt_length,
                                        chat_template=config.data.get('chat_template', False),
                                        filter_prompts=True,
                                        return_raw_chat=config.data.get('return_raw_chat', False),
                                        truncation=config.data.get('truncation', 'error'),
                                        filter_overlong_prompts=config.data.get('filter_overlong_prompts', False))
            self.val_score_dataloader = StatefulDataLoader(
                dataset=self.val_score_dataset,
                batch_size=len(self.val_score_dataset),
                num_workers=8,
                shuffle=True,
                drop_last=False,
                collate_fn=collate_fn)
            reward_manager_name = self.config.reward_model.get("reward_manager", "naive")
            if reward_manager_name == 'naive':
                from verl.workers.reward_manager import NaiveRewardManager
                reward_manager_cls = NaiveRewardManager
            elif reward_manager_name == 'prime':
                from verl.workers.reward_manager import PrimeRewardManager
                reward_manager_cls = PrimeRewardManager
            elif reward_manager_name == 'complete':
                from reil.workers.reward_manager import CompleteRewardManager
                reward_manager_cls = CompleteRewardManager
            elif reward_manager_name == 'gp_l':
                from reil.workers.reward_manager import GPLRewardManager
                reward_manager_cls = GPLRewardManager
            else:
                raise NotImplementedError
            compute_score = get_custom_reward_fn(self.config)
            
            self.val_reward_fn = reward_manager_cls(tokenizer=self.tokenizer, num_examine=1, compute_score=compute_score)
            self.MAX_REWARD = 5 if reward_manager_name == 'gp_l' else 1
        
        self.is_lora = self.config.evaluator.is_lora
        
        self._init_checkpoint_dirs()
        
        self.init_checkpoint_type()

        if self.is_fsdp:
            self.role_worker_mapping = {
                Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            }
            global_pool_id = 'global_pool'
            resource_pool_spec = {
                global_pool_id: [config.evaluator.n_gpus_per_node] * config.evaluator.nnodes,
            }
            # Initialize resource pool manager
            self.resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec,
                mapping={Role.ActorRollout: global_pool_id}
            )
            
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'false',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN',
            }
        })
        self.current_ckpt_path: Optional[str] = None
        self.last_val_batch: Optional[DataProto] = None
            
    def _init_checkpoint_dirs(self):
        """Initialize checkpoint directories for evaluation"""
        # Get checkpoint directory from config
        checkpoint_dir = Path(self.config.evaluator.checkpoint_dir)
        checkpoint_patterns = ["global_step_*", "checkpoint-*"]
        found_pattern = None
        
        for pattern in checkpoint_patterns:
            if list(checkpoint_dir.glob(pattern)):
                found_pattern = pattern
                break
                
        if not found_pattern:
            raise ValueError(f"No checkpoints found in {checkpoint_dir} with patterns {checkpoint_patterns}")
            
        print(f"Found checkpoints following pattern: {found_pattern}")
        
        # Use the appropriate pattern to find checkpoints
        self.checkpoint_dirs = sorted([
            d for d in checkpoint_dir.glob(found_pattern)
            if d.is_dir()
        ], key=lambda x: int(x.name.split('_')[-1] if found_pattern == "global_step_*" else x.name.split('-')[-1]))
            
        if not self.checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
        print("Checkpoint directories found:")
        for d in self.checkpoint_dirs:
            print(f"  {d}")

        print(f"Found {len(self.checkpoint_dirs)} checkpoints to evaluate")   

    def init_checkpoint_type(self):
        """
        Determine the checkpoint format by checking for the presence of an actor directory.
        Sets self.is_fsdp to True for FSDP format, False for HuggingFace format.
        """
        self.is_fsdp = (self.checkpoint_dirs[0] / "actor").exists()
        if self.is_fsdp:
            print("Using FSDP checkpoint")
        else:
            print("Using HuggingFace checkpoint")

    def cleanup_llm(self):
        """Clean up LLM instance and free GPU memory"""
        if hasattr(self, 'actor_rollout_wg'):
            if isinstance(self.actor_rollout_wg, VllmWrapperWg):
                if not self.is_lora and hasattr(self.actor_rollout_wg, 'llm'):
                    # Delete the LLM instance
                    destroy_model_parallel()
                    destroy_distributed_environment()
                    self.actor_rollout_wg.llm.llm_engine.engine_core.shutdown()
                    del self.actor_rollout_wg.llm
                    gc.collect()
                    # ray.shutdown()
            elif isinstance(self.actor_rollout_wg, HFWrapperWg):
                if hasattr(self.actor_rollout_wg, 'llm') and hasattr(self.actor_rollout_wg.llm, 'module'):
                    # Move model to CPU and delete
                    del self.actor_rollout_wg.llm.module

            torch.cuda.empty_cache()
            

    def init_workers(self):
        """Initialize resource pools and worker groups"""
        if self.is_fsdp:
            self.resource_pool_manager.create_resource_pool()
            
            # Create actor-rollout worker group
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role='actor_rollout'
            )
            
            # Create worker group with proper initialization
            # worker_dict_cls = create_colocated_worker_cls(class_dict={'actor_rollout': actor_rollout_cls})
            self.actor_rollout_wg = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=actor_rollout_cls)
            # ).spawn(prefix_set={'actor_rollout'})['actor_rollout']
            
            # Initialize the model first
            self.actor_rollout_wg.init_model()
        else:
            # self.actor_rollout_wg = HFWrapperWg(self.config, self.tokenizer)
            self.actor_rollout_wg = VllmWrapperWg(self.config, self.tokenizer)


    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint into initialized workers"""
        if checkpoint_path:
            # Ensure previous engines/models are cleaned up to free GPU memory
            self.cleanup_llm()
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Loading checkpoint from {checkpoint_path}")
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)
            # Record current checkpoint path for external KL
            self.current_ckpt_path = checkpoint_path

    
    def generate_sequences(self, lm_inputs: DataProto):
        """
        Generate sequences using the actor rollout worker group.
        """
        if isinstance(self.actor_rollout_wg, RayWorkerGroup):
            padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_rollout_wg.world_size)
            padded_lm_outputs = self.actor_rollout_wg.generate_sequences(padded_lm_inputs)
            lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
            lm_outputs.meta_info = lm_inputs.meta_info
            lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
        elif isinstance(self.actor_rollout_wg, (HFWrapperWg, VllmWrapperWg)):
            lm_outputs = self.actor_rollout_wg.generate_sequences(lm_inputs)
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_rollout_wg)}")

        return lm_outputs

    @torch.no_grad()
    def compute_log_probs(self, model, data: DataProto) -> torch.Tensor:
        """Compute per-token log-probs for given responses using the specified model (no generation). HF model only for now.

        Args:
            model: a causal LM instance with forward(input_ids, attention_mask, position_ids) -> logits
            data: DataProto containing 'input_ids', 'attention_mask', 'position_ids', 'responses'

        Returns:
            torch.Tensor on CPU of shape (batch_size, response_len)
        """
        # Teacher-forced log-probs under the specified model (CUDA) with micro-batching
        device = torch.device('cuda')
        model = model.to(device, dtype=torch.bfloat16)
        input_ids = data.batch['input_ids'].to(device)
        attention_mask = data.batch['attention_mask'].to(device)
        if 'position_ids' in data.batch:
            position_ids_full = data.batch['position_ids'].to(device)
        else:
            position_ids_full = attention_mask.cumsum(dim=-1)
        responses_full = data.batch['responses'].to(device)
        response_length = responses_full.size(1)

        # Determine micro-batch size
        kl_micro_bs = getattr(self.config.evaluator, 'kl_micro_batch_size', 16)
        batch_size = input_ids.size(0)
        if not isinstance(kl_micro_bs, int):
            kl_micro_bs = int(kl_micro_bs) if kl_micro_bs else 0
        use_micro_batch = kl_micro_bs and kl_micro_bs > 0 and kl_micro_bs < batch_size

        outputs = []
        if use_micro_batch:
            for start in range(0, batch_size, kl_micro_bs):
                end = min(start + kl_micro_bs, batch_size)
                ids = input_ids[start:end]
                am = attention_mask[start:end]
                pos = position_ids_full[start:end]
                rsp = responses_full[start:end]
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids,
                                   attention_mask=am,
                                   position_ids=pos,
                                   use_cache=False).logits
                    logits = logits.div_(self.config.actor_rollout_ref.rollout.val_kwargs.temperature)
                    logits = logits[:, -response_length - 1:-1, :]
                    chunk_log_probs = logprobs_from_logits(logits, rsp)
                outputs.append(chunk_log_probs.detach().to('cpu'))
                del ids, am, pos, rsp, logits, chunk_log_probs
                torch.cuda.empty_cache()
            log_probs = torch.cat(outputs, dim=0)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(input_ids=input_ids,
                               attention_mask=attention_mask,
                               position_ids=position_ids_full,
                               use_cache=False).logits
                logits = logits.div_(self.config.actor_rollout_ref.rollout.val_kwargs.temperature)
                logits = logits[:, -response_length - 1:-1, :]
                log_probs = logprobs_from_logits(logits, responses_full)
            log_probs = log_probs.detach().to('cpu')

        return log_probs

    def evaluate_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Evaluate a single checkpoint"""
        
        start_time = time.time()
        
        # Reset environments
        env_outputs = self.es_manager.reset(seed=self.config.evaluator.seed)
        print(f"Loading envs takes: {time.time() - start_time} seconds")
        
        # Set up meta info for generation
        meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            'validate': True,
        }
        
        # Run evaluation loop
        start_time = time.time()
        for i in tqdm(range(self.config.agent_proxy.max_turn)):
            # Get language model inputs
            lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
            lm_inputs.meta_info = meta_info
            # Generate sequences
            lm_outputs: DataProto = self.generate_sequences(lm_inputs)
            
            # Process environment inputs and outputs
            env_inputs: List[Dict] = self.ctx_manager.get_env_inputs(lm_outputs)
            env_outputs: List[Dict] = self.es_manager.step(env_inputs)
            
            if len(env_outputs) == 0:  # all finished
                break
                
        print(f"Evaluation time: {time.time() - start_time} seconds")
        
        # Get final results
        rollout_states = self.es_manager.get_rollout_states()
        self.maybe_log_rollout_states(rollout_states)
        rollouts = self.ctx_manager.formulate_rollouts(rollout_states)
        
        return rollouts.meta_info['metrics']
    
    def maybe_log_rollout_states(self, rollout_states, n_samples=4):
        """
        Input:
        rollout_states: List[Dict]
        rollout_states[i]= {
            'env_id': env_id,
            'history': List[List[Dict]],
            'group_id': group_id,
            'tag': environment_type,
            'penalty': penalty,
            'metrics': metrics,
        }
        rollout_states[i]['history'][j] = {
            'state': state,
            'action_left': action_left,
            'metrics': metrics,
            'llm_raw_response': llm_raw_response if the task is not finished,
            'llm_response': parsed llm response,
            'info': info returned by the environment
        }
        Return:
        - randomly sample 4(in default) rollout_states' last two states in history to log
        """
        import numpy as np
        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(rollout_states)
        rollout_states = rollout_states[:n_samples]
        columns = ["step"] + sum([[f"state_{i+1}", f"responses_{i+1}"] for i in range(n_samples)], [])
        if not hasattr(self, 'generations_table') and 'wandb' in self.logger.logger:
            self.generations_table = wandb.Table(columns=columns)
        states = []
        responses = []
        for rollout_state in rollout_states:
            state_info = rollout_state['history'][-2] if len(rollout_state['history']) > 1 else rollout_state['history'][-1]
            state_n_response = state_info['state'] + state_info['llm_raw_response'] if 'llm_raw_response' in state_info else state_info['state']
            print(state_n_response)
            # log texts to wandb
            if 'wandb' in self.logger.logger:
                states.append(state_info['state'])
                responses.append(state_info['llm_raw_response'] if 'llm_raw_response' in state_info else "")
            else:
                # pass for other loggers for now.
                pass
                # self.logger.log({"generations": state_n_response}, step=self.step)
                
        samples = list(zip(states, responses))
        print(samples)
        row_data = []
        row_data.append(self.step)
        for sample in samples:
            row_data.extend(sample)
        if 'wandb' in self.logger.logger:
            new_table = wandb.Table(columns=columns, data=self.generations_table.data)
            new_table.add_data(*row_data)
            self.logger.log({"generations": new_table}, step=self.step)
            self.generations_table = new_table
            # TODO: add other loggers
                            
    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        # the val_score_dataloader has only one batch in default.
        for test_data in self.val_score_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            test_output_gen_batch = self.generate_sequences(test_gen_batch)
            print('validation generation end')
            print(test_output_gen_batch.batch['responses'].shape)
            # Store generated outputs
            if 'response_texts' in test_output_gen_batch.non_tensor_batch.keys():
                output_texts = test_output_gen_batch.non_tensor_batch['response_texts']
            else:
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # Save for external KL (especially for vLLM path)
            self.last_val_batch = test_batch
            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        # self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        data_source_acc = {}
        data_source_valid = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_acc[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_acc[data_source].append(reward_tensor[i].item()==self.MAX_REWARD)
            # print(f"data_source: {data_source}, reward: {reward_tensor[i].item()}")
            if "gp_l" in data_source:
                if data_source not in data_source_valid:
                    data_source_valid[data_source] = []
                data_source_valid[data_source].append(reward_tensor[i].item()>=-1)
        
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/test_acc/{data_source}'] = np.mean(accs)

        if data_source_valid is not None:
            for data_source, valid in data_source_valid.items():
                metric_dict[f'val/test_valid/{data_source}'] = np.mean(valid)

        return metric_dict

    def compute_external_kl(self) -> Optional[float]:
        """Compute KL using an HF model loaded from the current checkpoint path.

        Offloads computation to a Ray remote worker (1 GPU) so the worker
        process owns and tears down its CUDA context independently.
        """
        if self.current_ckpt_path is None or self.last_val_batch is None:
            return None
        try:
            batch = self.last_val_batch
            input_ids = batch.batch['input_ids'].cpu()
            attention_mask = batch.batch['attention_mask'].cpu()
            responses = batch.batch['responses'].cpu()
            position_ids = batch.batch.get('position_ids', None)
            if position_ids is not None:
                position_ids = position_ids.cpu()

            temperature = float(self.config.actor_rollout_ref.rollout.val_kwargs.temperature)
            ref_model_path = self.config.actor_rollout_ref.model.path

            kl_micro_bs = getattr(self.config.evaluator, 'kl_micro_batch_size', 256)
            future = _kl_worker_remote.remote(
                self.current_ckpt_path,
                ref_model_path,
                bool(self.trust_remote_code),
                temperature,
                input_ids,
                attention_mask,
                responses,
                position_ids,
                int(kl_micro_bs) if isinstance(kl_micro_bs, int) or (isinstance(kl_micro_bs, str) and kl_micro_bs.isdigit()) else 4,
            )
            kld = ray.get(future)
            return float(kld)
        except Exception as e:
            print(f"External KL computation failed (ray): {e}")
            return None
    
    def eval(self, checkpoint_dir: str):
        """Evaluate multiple checkpoints in a directory"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Find all checkpoint directories
        # Check if checkpoints follow global_step_* or checkpoint-* pattern
        checkpoint_patterns = ["global_step_*", "checkpoint-*"]
        found_pattern = None
        
        for pattern in checkpoint_patterns:
            if list(checkpoint_dir.glob(pattern)):
                found_pattern = pattern
                break
                
        if not found_pattern:
            raise ValueError(f"No checkpoints found in {checkpoint_dir} with patterns {checkpoint_patterns}")
            
        print(f"Found checkpoints following pattern: {found_pattern}")
        
        # Use the appropriate pattern to find checkpoints
        checkpoint_dirs = sorted([
            d for d in checkpoint_dir.glob(found_pattern)
            if d.is_dir()
        ], key=lambda x: int(x.name.split('_')[-1] if found_pattern == "global_step_*" else x.name.split('-')[-1]))
        
        # Filter checkpoints based on resume_step if provided
        if hasattr(self.config.evaluator, 'resume_step') and self.config.evaluator.resume_step > 0:
            resume_step = self.config.evaluator.resume_step
            checkpoint_dirs = [d for d in checkpoint_dirs if int(d.name.split('_')[-1] if found_pattern == "global_step_*" else d.name.split('-')[-1]) >= resume_step]
            print(f"Resuming evaluation from step {resume_step}")

        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        print("Checkpoint directories found:")
        for d in checkpoint_dirs:
            print(f"  {d}")

        print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")  
        if self.config.evaluator.get('eval_base', False):
            self.step = 0
            if self.config.data.get('val_score_files', None):
                # single turn
                metrics = self._validate()
                # metrics.update(self.evaluate_checkpoint())
                # If KL wasn't computed internally (e.g., vLLM), compute it externally now
                # external_kld = self.compute_external_kl()
                # if external_kld is not None:
                #     metrics['val/kl'] = external_kld
            else:
                # multi turn
                metrics = self.evaluate_checkpoint()
            self.logger.log(data=metrics, step=self.step)
            
            print(f"Checkpoint {step} metrics:")
            pprint(metrics)
        
        self.cleanup_llm()
        # Evaluate each checkpoint
        for ckpt_dir in checkpoint_dirs:
            step = int(ckpt_dir.name.split('_')[-1] if found_pattern == "global_step_*" else ckpt_dir.name.split('-')[-1])
            print(f"\nEvaluating checkpoint at step {step}")
            self.step = step
            
            # Get actor checkpoint path
            if self.is_fsdp:
                actor_ckpt_path = ckpt_dir / "actor"
            else:
                actor_ckpt_path = ckpt_dir

            if not actor_ckpt_path.exists():
                print(f"Skipping {ckpt_dir} - no checkpoint found")
                continue
            
            # Evaluate checkpoint
            
            self.load_checkpoint(str(actor_ckpt_path))
            if self.config.data.get('val_score_files', None):
                # single turn
                metrics = self._validate()
                # metrics.update(self.evaluate_checkpoint())
                # If KL wasn't computed internally (e.g., vLLM), compute it externally now
                # external_kld = self.compute_external_kl()
                # if external_kld is not None:
                #     metrics['val/kl'] = external_kld
            else:
                # multi turn
                metrics = self.evaluate_checkpoint()
            self.cleanup_llm()
            
            # Log metrics using Tracking
            self.logger.log(data=metrics, step=self.step)
            
            print(f"Checkpoint {step} metrics:")
            pprint(metrics)

    def close(self):
        if self.is_fsdp:
            ray.shutdown()
        else:
            self.cleanup_llm()
            torch.distributed.barrier()

@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    evaluator = CheckpointEvaluator(config)
    evaluator.init_workers()
    
    evaluator.eval(config.evaluator.checkpoint_dir)
    evaluator.close()

if __name__ == "__main__":
    main() 