import hydra
import ray
import time
import wandb
from typing import List, Dict, Optional
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from enum import Enum
from verl import DataProto
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from reil.trainer.llm_agent.agent_proxy import VllmWrapperWg, HFWrapperWg, LLMAgentProxy
from reil.trainer.llm_agent.es_manager import EnvStateManager
from reil.trainer.llm_agent.ctx_manager import NaiveContextManager
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

class Role(Enum):
    """
    Define roles for different components in the evaluation system
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2

@dataclass
class ResourcePoolManager:
    """
    Resource pool manager for evaluation, similar to ray_trainer.py
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )

class CheckpointEvaluator:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
        
        # Initialize resource pool manager
        self.resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=config.resource_pool_spec,
            mapping={Role.ActorRollout: "actor_rollout"}
        )
        
        # Initialize environment and context managers
        self.es_manager = EnvStateManager(config, mode="val")
        self.ctx_manager = NaiveContextManager(config, self.tokenizer, processor=None, mode="val")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
            
        # Initialize wandb if enabled
        if config.get('wandb', {}).get('enabled', False):
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.name,
                config=config
            )

    def _init_workers(self, checkpoint_path: Optional[str] = None):
        """Initialize resource pools and worker groups with optional checkpoint loading"""
        self.resource_pool_manager.create_resource_pool()
        
        # Create actor-rollout worker group
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=VllmWrapperWg,
            config=self.config.actor_rollout_ref,
            role='actor_rollout'
        )
        
        # Create worker group
        worker_dict_cls = create_colocated_worker_cls({'actor_rollout': actor_rollout_cls})
        self.actor_rollout_wg = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=worker_dict_cls
        ).spawn(prefix_set={'actor_rollout'})['actor_rollout']
        
        # Initialize the model
        self.actor_rollout_wg.init_model()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)

    def evaluate_checkpoint(self, checkpoint_path: Optional[str] = None) -> Dict:
        """Evaluate a single checkpoint"""
        # Initialize workers with checkpoint
        self._init_workers(checkpoint_path)
        
        start_time = time.time()
        
        # Reset environments
        env_outputs = self.es_manager.reset()
        print(f"Loading envs takes: {time.time() - start_time} seconds")
        
        # Set up meta info for generation
        meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'recompute_log_prob': False,
            'do_sample': False,
            'validate': True,
        }
        
        # Run evaluation loop
        start_time = time.time()
        for i in tqdm(range(self.config.agent_proxy.max_turn)):
            # Get language model inputs
            lm_inputs: DataProto = self.ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
            lm_inputs.meta_info = meta_info
            
            # Generate sequences
            lm_outputs: DataProto = self.actor_rollout_wg.generate_sequences(lm_inputs)
            
            # Process environment inputs and outputs
            env_inputs: List[Dict] = self.ctx_manager.get_env_inputs(lm_outputs)
            env_outputs: List[Dict] = self.es_manager.step(env_inputs)
            
            if len(env_outputs) == 0:  # all finished
                break
                
        print(f"Evaluation time: {time.time() - start_time} seconds")
        
        # Get final results
        rollout_states = self.es_manager.get_rollout_states()
        rollouts = self.ctx_manager.formulate_rollouts(rollout_states)
        
        # Clean up Ray resources
        ray.shutdown()
        
        return rollouts.meta_info

    def evaluate_checkpoints(self, checkpoint_dir: str):
        """Evaluate multiple checkpoints in a directory"""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Find all checkpoint directories
        checkpoint_dirs = sorted([
            d for d in checkpoint_dir.glob("global_step_*")
            if d.is_dir()
        ])
        
        if not checkpoint_dirs:
            raise ValueError(f"No checkpoints found in {checkpoint_dir}")
        
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
            
            # Log metrics
            if self.config.get('wandb', {}).get('enabled', False):
                wandb.log({
                    **metrics,
                    'checkpoint_step': step
                })
            
            print(f"Checkpoint {step} metrics:")
            pprint(metrics)

@hydra.main(config_path="../trainer/config", config_name="evaluation.yaml")
def main(config):
    evaluator = CheckpointEvaluator(config)
    evaluator.evaluate_checkpoints(config.checkpoint_dir)

if __name__ == "__main__":
    main() 