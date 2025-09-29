#!/bin/bash

set -x

# ---------------------------
# Training (SFT)
# ---------------------------
N_GPUS=4
DATA_DIR=YOUR_SOKOBAN_DATA_DIR
BASE_MODEL=YOUR_BASE_MODEL
LEARNING_RATE=1e-5
SFT_TYPE="standard" # "aft", "dft", "standard"
AFT_POWER=1.0
ANCHOR_ENABLED=False
ANCHOR_COEF=0
KL_ENABLED=False
KL_COEF=0
EXPERIMENT_NAME="qwen-2.5-1.5b-non-diverse-cot-sokoban-${SFT_TYPE}-lr-${LEARNING_RATE}-anchor-${ANCHOR_COEF}-$(date +%m-%d)"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m debunk_sft.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=5000 \
    data.train_batch_size=128 \
    data.chat_template=False \
    data.max_response_length=4096 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    es_manager.val.env_groups=900 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'SimpleSokobanEmoji', 'FakeSokobanEmoji', 'SimpleSokobanCardinal', 'FakeSokobanCardinal', 'SimpleSokobanEmpty', 'TwoBoxesSokoban']" \
    es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100,100]" \
    agent_proxy.max_turn=30 \
    trainer.sft_type=$SFT_TYPE \
    trainer.aft_power=$AFT_POWER \
    trainer.anchor_regularization.enabled=${ANCHOR_ENABLED} \
    trainer.anchor_regularization.l2_anchor_coeff=${ANCHOR_COEF} \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds310/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.kl_regularization.enabled=${KL_ENABLED} \
    trainer.kl_regularization.kl_coef=${KL_COEF} \
    trainer.default_hdfs_dir=null "$@" | tee checkpoints/ds310/sft/${EXPERIMENT_NAME}_train.log

# ---------------------------
# Evaluation
# ---------------------------
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Derive checkpoint directory from EXPERIMENT_NAME
CHECKPOINT_DIR=$(realpath "checkpoints/ds310/sft/${EXPERIMENT_NAME}")
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
BASE_MODEL=$(realpath $BASE_MODEL)
PROJECT_NAME="REIL"
EVAL_EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"

# Evaluation settings

echo "Running evaluation with the following configuration:"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EVAL_EXPERIMENT_NAME"
echo "GPUs per node: $EVAL_N_GPUS"

python -m debunk_sft.evaluation.eval_ckpts \
    evaluator.checkpoint_dir=$CHECKPOINT_DIR \
    evaluator.project_name=$PROJECT_NAME \
    evaluator.experiment_name=$EVAL_EXPERIMENT_NAME \
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=0 \
    evaluator.is_lora=False \
    data.max_response_length=4096 \
    data.max_prompt_length=1024 \
    es_manager.val.env_groups=800 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'TwoBoxesSokoban', 'SimpleSokobanRandom', 'SimpleSokobanNumerical', 'SimpleSokobanAlphabetical', 'FakeSokobanNumerical']" \
    es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100]" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    agent_proxy.max_turn=30 \
    reward_model.reward_manager=complete \
    custom_reward_function.name=compute_score_with_action_sequence \
    custom_reward_function.path=./debunk_sft/utils/reward_score/sokoban.py


