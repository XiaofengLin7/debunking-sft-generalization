#!/bin/bash

set -x


DATA_DIR="./data/alfworld_task_type"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/sft/alfworld-1.5b-pick_n_place-sft-qwen-2.5-base-full-sft/global_step_90"
BETA=0.005
KL_COEF=0.001
CONTEXT_LENGTH=1024
# EXPERIMENT_NAME="pretrain-alfworld-1.5b-${BETA}beta-${KL_COEF}kl-pick_n_place_$(date +%m-%d)"
EXPERIMENT_NAME="alfworld-1.5b-${BETA}beta-${KL_COEF}kl-pick_n_place_05-07"
# EXPERIMENT_NAME="alfworld-1.5b-${BETA}beta-${KL_COEF}kl-2025-04-28"
# EXPERIMENT_NAME="1.5b-${BETA}beta-${KL_COEF}kl-2025-04-20"
ROLLOUT_TP_SIZE=1
N_GPUS=4
export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m reil.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=32 \
data.max_prompt_length=2000 \
data.max_response_length=$CONTEXT_LENGTH \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
actor_rollout_ref.actor.entropy_coeff=${BETA} \
actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=64 \
algorithm.use_kl_in_reward=True \
algorithm.kl_ctrl.kl_coef=${KL_COEF} \
trainer.logger=['wandb'] \
+trainer.val_only=False \
trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=500 \
trainer.test_freq=50 \
trainer.project_name=REIL \
trainer.resume_mode=auto \
trainer.log_val_generations=4 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=checkpoints/ds543/REIL/${EXPERIMENT_NAME} \
trainer.total_epochs=1500 \
trainer.policy_eval=True \
reward_model.reward_manager=complete \
es_manager.val.env_groups=100 \
es_manager.val.group_size=1 \
es_manager.val.env_configs.tags=["ALFWorld"] \
es_manager.val.env_configs.n_groups=[100] \
custom_reward_function.path=./reil/utils/reward_score/alfworld.py \
custom_reward_function.name=compute_score 2>&1 | tee alfworld_1.5b.log
