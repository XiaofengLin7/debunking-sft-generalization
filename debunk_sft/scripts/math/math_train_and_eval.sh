#!/bin/bash

set -x

# ---------------------------
# Training (SFT) for Math
# ---------------------------
N_GPUS=4
DATA_DIR="./data/math/original/sft"
# BASE_MODEL should be a Hugging Face repo ID, e.g. Qwen/Qwen2.5-1.5B
BASE_MODEL=Qwen/Qwen2.5-1.5B
LEARNING_RATE=1e-5
SFT_TYPE="standard"  # "aft", "dft", "standard"
AFT_POWER=1.0
ANCHOR_ENABLED=False
ANCHOR_COEF=0
KL_ENABLED=False
KL_COEF=0
EXPERIMENT_NAME="qwen-2.5-1.5b-math-${SFT_TYPE}-lr-${LEARNING_RATE}-$(date +%m-%d)"

# Resolve repo root so paths work regardless of current working directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)

# MATH-Perturb evaluation data paths
MATH_PERTURB_SIMPLE="${ROOT_DIR}/data/math_perturb/simple.parquet"
MATH_PERTURB_HARD="${ROOT_DIR}/data/math_perturb/hard.parquet"
EVAL_DATA="['${MATH_PERTURB_SIMPLE}', '${MATH_PERTURB_HARD}']"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m debunk_sft.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/hendrycks_math_train.parquet \
    data.val_files=$DATA_DIR/hendrycks_math_test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.max_length=8192 \
    data.train_batch_size=128 \
    data.chat_template=True \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    trainer.sft_type=$SFT_TYPE \
    trainer.aft_power=$AFT_POWER \
    trainer.anchor_regularization.enabled=${ANCHOR_ENABLED} \
    trainer.anchor_regularization.l2_anchor_coeff=${ANCHOR_COEF} \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/math/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=10 \
    trainer.val_before_train=False \
    trainer.kl_regularization.enabled=${KL_ENABLED} \
    trainer.kl_regularization.kl_coef=${KL_COEF} \
    trainer.default_hdfs_dir=null "$@" | tee checkpoints/math/sft/${EXPERIMENT_NAME}_train.log

# ---------------------------
# Evaluation on MATH-Perturb
# ---------------------------
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derive checkpoint directory from EXPERIMENT_NAME
CHECKPOINT_DIR=$(realpath "${ROOT_DIR}/checkpoints/math/sft/${EXPERIMENT_NAME}")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
# Only resolve local paths; keep HF repo IDs as-is.
if [[ "$BASE_MODEL" == /* || "$BASE_MODEL" == ./* ]]; then
  BASE_MODEL=$(realpath "$BASE_MODEL")
fi
PROJECT_NAME="REIL"
EVAL_EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"

# Evaluation data paths


echo "Running evaluation with the following configuration:"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EVAL_EXPERIMENT_NAME"
echo "Evaluation data: $EVAL_DATA"

python -m debunk_sft.evaluation.eval_ckpts \
    +data.val_score_files="$EVAL_DATA" \
    data.prompt_key=prompt \
    +data.chat_template=True \
    data.max_prompt_length=2048 \
    +data.filter_overlong_prompts=False \
    data.max_response_length=4096 \
    evaluator.checkpoint_dir="$CHECKPOINT_DIR" \
    evaluator.project_name="$PROJECT_NAME" \
    evaluator.experiment_name="$EVAL_EXPERIMENT_NAME" \
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=0 \
    evaluator.is_lora=False \
    evaluator.init_envs=False \
    +evaluator.eval_base=True \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    agent_proxy.max_turn=1 \
    agent_proxy.parse_response=False \
    agent_proxy.chat_template=True \
    reward_model.reward_manager=complete \
    custom_reward_function.name=compute_score_math \
    custom_reward_function.path="${ROOT_DIR}/debunk_sft/utils/reward_score/math.py"
