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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from tqdm import tqdm

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import Role
from verl.trainer.ppo.ray_trainer import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.trainer.ppo.ray_trainer import compute_response_mask
import torch
from verl.utils.torch_functional import masked_mean
from collections import defaultdict
from reil.utils.dataset.rg_dataset import prepare_reasoning_gym_dataset
from reasoning_gym.utils import extract_answer
from reil.utils.reward_score.reward import reward_registry
WorkerType = Type[Worker]

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last



class ReilPPOTrainer(RayPPOTrainer):
    def __init__(self,
                config,
                tokenizer,
                role_worker_mapping: dict[Role, WorkerType],
                resource_pool_manager: ResourcePoolManager,
                ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                processor=None,
                reward_fn=None,
                val_reward_fn=None,
                ):
        # determine if we are using reasoning gym
        if config.data.type == 'reasoning_gym':
            # construct dataset first
            self.train_dataset, self.val_dataset = prepare_reasoning_gym_dataset(config.data.reasoning_gym, tokenizer)
            # construct reward function
            self.reward_functions = []
            if hasattr(config, "reward") and hasattr(config.reward, "secondary_rewards"):
                for func_config in config.reward.secondary_rewards:
                    func_name = func_config.name
                    scaling_factor = func_config.get("scaling_factor", 1.0)
                    func = reward_registry.get(func_name)
                    if func:
                        # Store both function and its arguments
                        self.reward_functions.append(
                            {
                                "function": func,
                                "name": func_name,
                                "scaling_factor": scaling_factor,
                                "kwargs": func_config.get("kwargs", {}),
                            }
                        )

            reward_fn = lambda data: self._score_output(data, num_examine=0, is_val=False)
            val_reward_fn = lambda data: self._score_output(data, num_examine=1, is_val=True)
        else:
            # TODO: we have to make sure the batch size is divisible by the dp size
            self.train_dataset = RLHFDataset(parquet_files=config.data.train_files,
                                            tokenizer=tokenizer,
                                            processor=processor,
                                            prompt_key=config.data.prompt_key,
                                            image_key=config.data.get('image_key', 'images'),
                                            max_prompt_length=config.data.max_prompt_length,
                                            chat_template=config.data.get('chat_template', False),
                                            filter_prompts=True,
                                            return_raw_chat=config.data.get('return_raw_chat', False),
                                            truncation=config.data.get('truncation', 'error'),
                                            filter_overlong_prompts=config.data.filter_overlong_prompts)
            
            assert self.train_dataset.truncation == config.data.get(
                'truncation', 'error'
            ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {config.data.get("truncation", "error")}'
            
            self.val_dataset = RLHFDataset(parquet_files=config.data.val_files,
                                        tokenizer=tokenizer,
                                        processor=processor,
                                        prompt_key=config.data.prompt_key,
                                        image_key=config.data.get('image_key', 'images'),
                                        max_prompt_length=config.data.max_prompt_length,
                                        chat_template=config.data.get('chat_template', False),
                                        filter_prompts=True,
                                        return_raw_chat=config.data.get('return_raw_chat', False),
                                        truncation=config.data.get('truncation', 'error'),
                                        filter_overlong_prompts=config.data.filter_overlong_prompts)
            assert self.val_dataset.truncation == config.data.get(
                'truncation', 'error'
            ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {config.data.get("truncation", "error")}'
            
        super().__init__(config=config, 
                         tokenizer=tokenizer, 
                         role_worker_mapping=role_worker_mapping, 
                         resource_pool_manager=resource_pool_manager, 
                         ray_worker_group_cls=ray_worker_group_cls, 
                         processor=processor, 
                         reward_fn=reward_fn, 
                         val_reward_fn=val_reward_fn)

        # assert torch.cuda.is_available(), 'cuda must be available on driver'


    def init_agent_proxy(self):
        if self.config.trainer.policy_eval and self.config.data.type != 'reasoning_gym':
            from reil.trainer.llm_agent.agent_proxy import LLMAgentProxy
            self.agent_proxy = LLMAgentProxy(self.config, self.actor_rollout_wg, self.tokenizer)

    def _create_dataloader(self):
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn)
        
        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

            
    def _validate_on_env(self):
        rollouts = self.agent_proxy.rollout()
        return rollouts.meta_info['metrics']

    def _score_output(self, data: DataProto, num_examine: int = 0, is_val: bool = False) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        num_printed = 0
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]  # tokenized prompts
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = prompt_str + response_str

            index = data_item.non_tensor_batch["index"]
            correctness_score = self._compute_correctness_score(
                solution_str=response_str,
                index=index,
                is_val=is_val
            )
            if self.config.reward.use_accuracy:
                reward_components = {"correctness": correctness_score}
                total_reward = correctness_score
            else:
                reward_components = {}
                total_reward = 0

            for reward_fn in self.reward_functions:
                func = reward_fn["function"]
                name = reward_fn["name"]
                scaling_factor = reward_fn["scaling_factor"]
                kwargs = reward_fn["kwargs"]
                if name == "cosine":
                    is_correct = correctness_score == 1.0
                    reward = func(response_str, scaling_factor, is_correct=is_correct, **kwargs)
                elif name == "length":
                    reward = func(response_str, scaling_factor, correctness_score=correctness_score, **kwargs)
                else:
                    reward = func(response_str, scaling_factor, **kwargs)
                reward_components[name] = reward
                total_reward += reward

            reward_tensor[i, valid_response_length - 1] = total_reward

            if num_printed < num_examine:
                components = ", ".join([f"{k}={v:.2f}" for k, v in reward_components.items()])
                print(f"(score={total_reward}, seq={sequences_str}, response={response_str})")
                print(f"reward={total_reward:.2f} ({components})")
                num_printed += 1

        return reward_tensor

    def _compute_correctness_score(self, solution_str: str, index: int, is_val: bool = False) -> float:
        found_answer = extract_answer(solution_str, tag_name="answer")
        if is_val:
            data = self.val_dataset.data
        else:
            data = self.train_dataset.data

        entry = data[index]
        if is_val:
            if self.val_dataset.experiment:
                experiment = self.val_dataset.experiment
                return experiment.score_answer_with_id(found_answer, entry["metadata"]["entry_id"])
            else:
                return data.score_answer(found_answer, entry=entry)
        else:
            if self.train_dataset.experiment:   
                experiment = self.train_dataset.experiment
                return experiment.score_answer_with_id(found_answer, entry["metadata"]["entry_id"])
            else:
                return data.score_answer(found_answer, entry=entry)
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.policy_eval:
                val_env_metrics = self._validate_on_env()
                pprint(f'Initial validation metrics on envs: {val_env_metrics}')
                logger.log(data=val_env_metrics, step=self.global_steps)

            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  advantage=self.config.algorithm.advantage,
                                                  positive_advantage_weight=self.config.algorithm.positive_advantage_weight)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if self.config.trainer.policy_eval:
                                with _timer('policy_eval', timing_raw):
                                    val_env_metrics = self._validate_on_env()
                            if is_last_step:
                                last_val_metrics = val_metrics
                                if self.config.trainer.policy_eval:
                                    last_val_env_metrics = val_env_metrics
                        metrics.update(val_metrics)
                        if self.config.trainer.policy_eval:
                            metrics.update(val_env_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                        logger.log(data={"checkpoint_saved": self.global_steps}, step=self.global_steps)
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    if self.config.trainer.policy_eval:
                        pprint(f'Final validation metrics on envs: {last_val_env_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
