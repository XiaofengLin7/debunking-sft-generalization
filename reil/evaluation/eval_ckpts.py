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
            
        # Find all checkpoint directories
        self.checkpoint_dirs = sorted([
            d for d in checkpoint_dir.glob("global_step_*")
            if d.is_dir()
        ], key=lambda x: int(x.name.split('_')[-1]))
        
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



    def _init_workers(self):
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
            pass


    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint into initialized workers"""
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)
    
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
        else:
            raise ValueError(f"Unsupported actor worker type: {type(self.actor_rollout_wg)}")

        return lm_outputs

    def evaluate_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Evaluate a single checkpoint"""
        # Initialize workers
        # Load checkpoint if provided
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        
        start_time = time.time()
        
        # Reset environments
        env_outputs = self.es_manager.reset()
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
        rollouts = self.ctx_manager.formulate_rollouts(rollout_states)
        
        return rollouts.meta_info['metrics']

    def evaluate_checkpoints(self, checkpoint_dir: str):
        """Evaluate multiple checkpoints in a directory"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Find all checkpoint directories
        checkpoint_dirs = sorted([
            d for d in checkpoint_dir.glob("global_step_*")
            if d.is_dir()
        ], key=lambda x: int(x.name.split('_')[-1]))
        
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")

        print("Checkpoint directories found:")
        for d in checkpoint_dirs:
            print(f"  {d}")

        print(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")        
        
        # Evaluate each checkpoint
        for ckpt_dir in checkpoint_dirs:
            step = int(ckpt_dir.name.split('_')[-1])
            print(f"\nEvaluating checkpoint at step {step}")
            
            # Get actor checkpoint path
            actor_ckpt_path = ckpt_dir / "actor"
            if not actor_ckpt_path.exists():
                print(f"Skipping {ckpt_dir} - no actor checkpoint found")
                continue
            
            # Evaluate checkpoint
            metrics = self.evaluate_checkpoint(str(actor_ckpt_path))
            
            # Log metrics using Tracking
            self.logger.log(data=metrics, step=step)
            
            print(f"Checkpoint {step} metrics:")
            pprint(metrics)

    def close(self):
        if self.is_fsdp:
            ray.shutdown()
        else:
            pass

@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    evaluator = CheckpointEvaluator(config)
    evaluator._init_workers()
    evaluator.evaluate_checkpoints(config.evaluator.checkpoint_dir)
    evaluator.close()

if __name__ == "__main__":
    main() 