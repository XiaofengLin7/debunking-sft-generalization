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
from ragen.llm_agent.generation import LLMGenerationManager, GenerationConfig
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
                val_env=None,
                env_class=None):
        super().__init__(config=config, 
                         tokenizer=tokenizer, 
                         role_worker_mapping=role_worker_mapping, 
                         resource_pool_manager=resource_pool_manager, 
                         ray_worker_group_cls=ray_worker_group_cls, 
                         processor=processor, 
                         reward_fn=reward_fn, 
                         val_reward_fn=val_reward_fn)

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.val_env = val_env
        self.env_class = env_class
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.get('truncation', 'error'),
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.train_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
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

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation=self.config.data.get('truncation', 'error'),
                                       filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        assert self.val_dataset.truncation == self.config.data.get(
            'truncation', 'error'
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_fn)
        
        self.val_env_dataset = RLHFDataset(parquet_files=self.config.data.val_env_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        
        self.val_env_dataloader = StatefulDataLoader(dataset=self.val_env_dataset,
                                             batch_size=self.config.data.val_env_batch_size,
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

            
    def _validate_on_env(self, logger):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        # Initialize global metric storage
        global_token_scores = []
        global_metrics = {}
        metrics = defaultdict(list)
        # breakpoint()

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            logging=self.config.logging,
            num_gpus=self.config.trainer.n_gpus_per_node,
            no_think_rl=self.config.algorithm.no_think_rl,
            state_masking=self.config.actor_rollout_ref.actor.state_masking,
            start_state_marker=self.config.algorithm.state_masking.start_state_marker,
            end_state_marker=self.config.algorithm.state_masking.end_state_marker,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            env_class=self.env_class,
            config=gen_config,
            logger = logger,
            is_validation = True,
        )

        envs = [self.val_env.copy() for _ in range(self.config.data.val_env_batch_size)] # do not repeat
        # envs = [self.val_env.copy() for _ in range(self.config.data.val_batch_size * self.config.actor_rollout_ref.rollout.n_agent)]
        val_global_steps = 1

        for batch_dict in self.val_env_dataloader:
            timing_raw = {}
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

            env_seeds = [i['index'] for i in test_batch.non_tensor_batch['extra_info']]
            print("env_seeds:", env_seeds)
            for env, seed in zip(envs, env_seeds):
                env.reset(seed=seed)
            
            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }
            with _timer('step', timing_raw):
                first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                output_dir = (f"{self.config.logging.log_image_dir}/"
                                f"{self.config.trainer.experiment_name}/"
                                f"validation_{self.global_steps}/"
                                f"step_{val_global_steps}")
                with _timer('gen', timing_raw):
                    final_gen_batch_output = generation_manager.run_llm_loop(
                        gen_batch=test_gen_batch,
                        envs=envs,
                        initial_input_ids=first_input_ids,
                        output_dir=output_dir,
                        global_steps=self.global_steps,
                    )
                with torch.no_grad():
                    output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                    final_gen_batch_output = final_gen_batch_output.union(output)

                test_batch.non_tensor_batch['reward'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['reward'][idx] = env.reward
                
                test_batch.non_tensor_batch['total_env'] = np.array([1 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['finished_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['success_env'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['traj_length'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['valid_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action'] = np.array([0 for _ in range(len(envs))], dtype=object)
                test_batch.non_tensor_batch['effective_action_ratio'] = np.array([0 for _ in range(len(envs))], dtype=object)
                for idx, env in enumerate(envs):
                    test_batch.non_tensor_batch['finished_env'][idx] = int(env.finished())
                    test_batch.non_tensor_batch['success_env'][idx] = int(env.success())
                    tracking_vars = env.get_tracking_variables()
                    test_batch.non_tensor_batch['traj_length'][idx] = len(tracking_vars['actions'])
                    test_batch.non_tensor_batch['valid_action'][idx] = sum(1 for x in tracking_vars['actions_valid'] if x is not None)
                    test_batch.non_tensor_batch['effective_action'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None)
                    test_batch.non_tensor_batch['effective_action_ratio'][idx] = sum(1 for x in tracking_vars['actions_effective'] if x is not None) / len(tracking_vars['actions'])

                # action metrics
                metrics['total_env'].append(test_batch.non_tensor_batch['total_env'])
                metrics['finished_env'].append(test_batch.non_tensor_batch['finished_env'])
                metrics['success_env'].append(test_batch.non_tensor_batch['success_env'])
                metrics['traj_length'].append(test_batch.non_tensor_batch['traj_length'])
                metrics['valid_action'].append(test_batch.non_tensor_batch['valid_action'])
                metrics['effective_action'].append(test_batch.non_tensor_batch['effective_action'])
                metrics['effective_action_ratio'].append(test_batch.non_tensor_batch['effective_action_ratio'])

                # Accumulate batch metrics into global storage
                global_token_scores.append(test_batch.non_tensor_batch['reward'])


        global_scores = np.concatenate(global_token_scores, axis=0)
        global_metrics = {
            'global_score/mean': float(global_scores.mean()),
            'global_score/max': float(global_scores.max()),
            'global_score/min': float(global_scores.min()),
            'global_score/std': float(global_scores.std()),
            'validate_metric/total_env': int(np.array(metrics['total_env'], dtype=np.int16).sum()),
            'validate_metric/finished_env': int(np.array(metrics['finished_env'], dtype=np.int16).sum()),
            'validate_metric/success_env': int(np.array(metrics['success_env'], dtype=np.int16).sum()),
            'validate_metric/traj_length': float(np.array(metrics['traj_length'], dtype=np.int16).mean()),
            'validate_metric/valid_action': float(np.array(metrics['valid_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action': float(np.array(metrics['effective_action'], dtype=np.int16).mean()),
            'validate_metric/effective_action_ratio': float(np.array(metrics['effective_action_ratio'], dtype=np.float32).mean()),
        }
        print("global_metrics", global_metrics)

        return global_metrics
    
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
            if self.config.trainer.is_rl_validation:
                val_env_metrics = self._validate_on_env(logger)
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
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

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
                            if self.config.trainer.is_rl_validation:
                                val_env_metrics = self._validate_on_env(logger)
                            if is_last_step:
                                last_val_metrics = val_metrics
                                if self.config.trainer.is_rl_validation:
                                    last_val_env_metrics = val_env_metrics
                        metrics.update(val_metrics)
                        if self.config.trainer.is_rl_validation:
                            metrics.update(val_env_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

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
                    if self.config.trainer.is_rl_validation:
                        pprint(f'Final validation metrics on envs: {last_val_env_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
