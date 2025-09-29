# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
# from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from debunk_sft.trainer.llm_agent.agent_proxy import LLMAgentProxy, HFWrapperWg
from debunk_sft.trainer.fsdp_sft_trainer import FSDPSFTTrainer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


class FSDPContrastiveTrainer(FSDPSFTTrainer):
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, pos_train_dataset: Dataset, neg_train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader(pos_train_dataset, neg_train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        
        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
            
        self.init_agent_proxy()


    def _build_dataloader(self, pos_train_dataset, neg_train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.pos_train_dataset, self.neg_train_dataset, self.val_dataset = pos_train_dataset, neg_train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.pos_train_sampler = DistributedSampler(self.pos_train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.neg_train_sampler = DistributedSampler(self.neg_train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.pos_train_dataloader = DataLoader(
            dataset=self.pos_train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.pos_train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )
        self.neg_train_dataloader = DataLoader(
            dataset=self.neg_train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.neg_train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.pos_train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        # elif self.config.optim.lr_scheduler == "wsd":
        #     self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")
        
    def _compute_contrastive_loss_and_backward(self, pos_batch: TensorDict, neg_batch: TensorDict, do_backward=True):
        pass
        
    def training_step(self, pos_batch: TensorDict, neg_batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        pos_micro_batches = pos_batch.split(self.config.data.micro_batch_size_per_gpu)
        neg_micro_batches = neg_batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(pos_micro_batches)
        step_loss = 0
        step_loss_pos = 0
        step_loss_neg = 0
        step_gap = 0
        gamma = self.config.algorithm.contrastive_loss.gamma
        ratio = self.config.algorithm.contrastive_loss.ratio
        
        for k, (pos_mb, neg_mb) in enumerate(zip(pos_micro_batches, neg_micro_batches)):
            # ------------------------------------------------------------
            # 1. Forward passes (build two CE losses with graphs)
            # ------------------------------------------------------------
            loss_pos = self._compute_loss_and_backward(pos_mb, do_backward=False)
            loss_neg = self._compute_loss_and_backward(neg_mb, do_backward=False)

            # ------------------------------------------------------------
            # 2. Margin + logistic loss for THIS pair
            #    Divide by n_mb so the sum over pairs is an average
            # ------------------------------------------------------------
            gap      =  loss_pos - ratio * loss_neg + gamma
            mb_loss  = torch.nn.functional.softplus(gap) / n_micro_batches

            step_loss += mb_loss.detach().item()
            step_loss_pos += loss_pos.detach().item() / n_micro_batches
            step_loss_neg += loss_neg.detach().item() / n_micro_batches
            step_gap += (loss_pos.detach().item() - loss_neg.detach().item()) / n_micro_batches 
            # ------------------------------------------------------------
            # 3. Back‑prop right away, release the graph
            # ------------------------------------------------------------
            if k != n_micro_batches - 1:
                with self.fsdp_model.no_sync():
                    mb_loss.backward()
            else:
                mb_loss.backward()


        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        step_loss_pos = torch.tensor(step_loss_pos).cuda()
        torch.distributed.all_reduce(step_loss_pos, op=torch.distributed.ReduceOp.AVG)
        step_loss_neg = torch.tensor(step_loss_neg).cuda()
        torch.distributed.all_reduce(step_loss_neg, op=torch.distributed.ReduceOp.AVG)
        step_gap = torch.tensor(step_gap).cuda()
        torch.distributed.all_reduce(step_gap, op=torch.distributed.ReduceOp.AVG)
        return {"train/loss": step_loss.detach().item(), "train/loss_pos": step_loss_pos.detach().item(), "train/loss_neg": step_loss_neg.detach().item(), "train/gap": step_gap.detach().item(), "train/lr(1e-3)": lr * 1e3}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.pos_train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.pos_train_sampler.set_epoch(epoch=epoch)
            self.neg_train_sampler.set_epoch(epoch=epoch)
            for pos_data, neg_data in tqdm(
                zip(self.pos_train_dataloader, self.neg_train_dataloader),
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            ):
                global_step += 1
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA is not available")
                if torch.cuda.current_device() != rank:
                    raise RuntimeError(f"Device mismatch: current={torch.cuda.current_device()}, expected={rank}")

                local_rank = int(os.environ["LOCAL_RANK"])

                pos_data = TensorDict(pos_data, batch_size=self.config.data.train_batch_size).cuda(device=local_rank)
                neg_data = TensorDict(neg_data, batch_size=self.config.data.train_batch_size).cuda(device=local_rank)

                metric = self.training_step(pos_data, neg_data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda(device=local_rank)
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    # if self.config.trainer.policy_eval and self.config.model.lora_rank == 0:
                    if self.config.trainer.policy_eval:
                        actor_wg = HFWrapperWg(self.config, self.tokenizer, module=self.fsdp_model)
                        self.proxy.set_actor_wg(actor_wg)
                        rollouts = self.proxy.rollout()

                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": avg_val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        tracking.log(data=rollouts.meta_info['metrics'], step=global_step)
                    
                    torch.distributed.barrier()

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

            # validation
            val_losses = []
            for data in self.val_dataloader:
                data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda(device=local_rank)
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)

            # if self.config.trainer.policy_eval and self.config.model.lora_rank == 0:
            if self.config.trainer.policy_eval:
                actor_wg = HFWrapperWg(self.config, self.tokenizer, module=self.fsdp_model)
                self.proxy.set_actor_wg(actor_wg)
                rollouts = self.proxy.rollout()    
            
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
                if self.config.trainer.policy_eval:
                    tracking.log(data=rollouts.meta_info['metrics'], step=global_step)
            
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)




@hydra.main(config_path="config", config_name="contrastive_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    pos_train_dataset = create_sft_dataset(config.data.pos_train_files, config.data, tokenizer)
    neg_train_dataset = create_sft_dataset(config.data.neg_train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPContrastiveTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, pos_train_dataset=pos_train_dataset, neg_train_dataset=neg_train_dataset, val_dataset=val_dataset)

    trainer.fit()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()