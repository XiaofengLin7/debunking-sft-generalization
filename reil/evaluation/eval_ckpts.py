import hydra
import ray
import time
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
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
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
                                        filter_overlong_prompts=config.data.filter_overlong_prompts)
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
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN',
                    'VLLM_LOGGING_LEVEL': 'WARN',
                    'VLLM_ATTENTION_BACKEND': 'XFORMERS',
                    
                }
            })
            
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
                if hasattr(self.actor_rollout_wg, 'llm'):
                    # Delete the LLM instance
                    del self.actor_rollout_wg.llm
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
            self.cleanup_llm()
            print(f"Loading checkpoint from {checkpoint_path}")
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)
            # print(self.actor_rollout_wg.module)
    
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
                
                self.logger.log({"generations": state_n_response}, step=self.step)
                
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
            # TODO: add feasible solution for vllm wrapper
            if 'response_texts' in test_output_gen_batch.non_tensor_batch.keys():
                output_texts = test_output_gen_batch.non_tensor_batch['response_texts']
            else:
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

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
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
                data_source_acc[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
            data_source_acc[data_source].append(reward_tensor[i].item()==self.MAX_REWARD)
            # print(f"data_source: {data_source}, reward: {reward_tensor[i].item()}")

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
        for data_source, accs in data_source_acc.items():
            metric_dict[f'val/test_acc/{data_source}'] = np.mean(accs)
        return metric_dict
    def evaluate_checkpoints(self, checkpoint_dir: str):
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
            else:
                # multi turn
                metrics = self.evaluate_checkpoint()
            
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
    evaluator.evaluate_checkpoints(config.evaluator.checkpoint_dir)
    evaluator.close()

if __name__ == "__main__":
    main() 