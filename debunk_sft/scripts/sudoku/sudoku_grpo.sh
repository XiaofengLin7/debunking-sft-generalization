#!/bin/bash

set -x

# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/sft/alfworld-1.5b-pick_n_place-sft-qwen-2.5-base-full-sft/global_step_90"
# Default values
BETA=0.005
KL_COEF=0.001

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --beta)
      BETA="$2"
      shift # past argument
      shift # past value
      ;;
    --kl)
      KL_COEF="$2"
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--beta BETA] [--kl KL_COEF]"
      exit 0
      ;;
    *)    # unknown option
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CONTEXT_LENGTH=4096
EXPERIMENT_NAME="sudoku-1.5b-${BETA}beta-${KL_COEF}kl-$(date +%m-%d)"
# EXPERIMENT_NAME="alfworld-1.5b-${BETA}beta-${KL_COEF}kl-pick_n_place_05-07"
# EXPERIMENT_NAME="alfworld-1.5b-${BETA}beta-${KL_COEF}kl-2025-04-28"
# EXPERIMENT_NAME="1.5b-${BETA}beta-${KL_COEF}kl-2025-04-20"
ROLLOUT_TP_SIZE=1
N_GPUS=4
BATCH_SIZE=256
export VLLM_USE_V1=1

python3 -m reil.trainer.main_ppo \
data.type=reasoning_gym \
+data.reasoning_gym.train.datasets.mini_sudoku.weight=1.0 \
+data.reasoning_gym.train.datasets.mini_sudoku.config.min_empty=8 \
+data.reasoning_gym.train.datasets.mini_sudoku.config.max_empty=12 \
+data.reasoning_gym.val.datasets.sudoku.weight=1.0 \
+data.reasoning_gym.val.datasets.sudoku.config.min_empty=30 \
+data.reasoning_gym.val.datasets.sudoku.config.max_empty=50 \
data.train_batch_size=$BATCH_SIZE \
data.max_prompt_length=1000 \
data.max_response_length=$CONTEXT_LENGTH \
algorithm.adv_estimator=grpo \
algorithm.use_kl_in_reward=True \
algorithm.kl_ctrl.kl_coef=${KL_COEF} \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
actor_rollout_ref.actor.entropy_coeff=${BETA} \
actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
actor_rollout_ref.rollout.n=8 \
trainer.logger=['wandb'] \
+trainer.val_only=False \
trainer.val_before_train=True \
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
trainer.total_epochs=100 \
trainer.policy_eval=False 2>&1 | tee sudoku_1.5b.log
