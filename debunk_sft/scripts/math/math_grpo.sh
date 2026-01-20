#!/bin/bash
set -x

# ---------------------------
# GRPO Training for Math
# ---------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

DATA_DIR="${ROOT_DIR}/data/math/diverse/rl"
# BASE_MODEL should be a Hugging Face repo ID, e.g. Qwen/Qwen2.5-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B

BETA=0.000      # Entropy coefficient
KL_COEF=0.001   # KL coefficient

# MATH-Perturb evaluation data paths
MATH_PERTURB_SIMPLE="${ROOT_DIR}/data/math_perturb/simple.parquet"
MATH_PERTURB_HARD="${ROOT_DIR}/data/math_perturb/hard.parquet"

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --beta)
      BETA="$2"
      shift
      shift
      ;;
    --kl)
      KL_COEF="$2"
      shift
      shift
      ;;
    --model)
      BASE_MODEL="$2"
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--beta BETA] [--kl KL_COEF] [--model BASE_MODEL]"
      exit 0
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CONTEXT_LENGTH=4096
BATCH_SIZE=128
EXPERIMENT_NAME="math-grpo-${BETA}beta-${KL_COEF}kl-$(date +%m-%d)"
ROLLOUT_TP_SIZE=1
N_GPUS=4

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# Evaluation data paths for validation
EVAL_DATA="['${MATH_PERTURB_SIMPLE}', '${MATH_PERTURB_HARD}']"

python3 -m debunk_sft.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/train.parquet \
data.train_batch_size=$BATCH_SIZE \
data.val_batch_size=32 \
data.max_prompt_length=2048 \
data.max_response_length=$CONTEXT_LENGTH \
data.chat_template=True \
algorithm.adv_estimator=grpo \
algorithm.use_kl_in_reward=True \
algorithm.kl_ctrl.kl_coef=${KL_COEF} \
actor_rollout_ref.model.path="$BASE_MODEL" \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=128 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
actor_rollout_ref.actor.entropy_coeff=${BETA} \
actor_rollout_ref.rollout.log_prob_micro_batch_size=16 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
actor_rollout_ref.rollout.n=8 \
actor_rollout_ref.rollout.val_kwargs.temperature=0 \
actor_rollout_ref.rollout.val_kwargs.do_sample=False \
trainer.logger="['console', 'wandb']" \
+trainer.val_only=False \
trainer.val_before_train=True \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=20 \
trainer.project_name=REIL \
trainer.resume_mode=auto \
trainer.log_val_generations=4 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=checkpoints/math/grpo/${EXPERIMENT_NAME} \
trainer.total_epochs=100 \
trainer.policy_eval=False \
evaluator.init_envs=False \
agent_proxy.max_turn=1 \
agent_proxy.parse_response=False \
agent_proxy.chat_template=True \
reward_model.reward_manager=complete \
custom_reward_function.path="${ROOT_DIR}/debunk_sft/utils/reward_score/math.py" \
custom_reward_function.name=compute_score_math 2>&1 | tee checkpoints/math/grpo/${EXPERIMENT_NAME}.log
