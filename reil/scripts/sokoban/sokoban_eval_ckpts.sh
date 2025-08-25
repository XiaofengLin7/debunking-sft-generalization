#!/bin/bash

set -x
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# Model and checkpoint settings
BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323  # Base model path
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/REIL/sokoban-1.5b-0.0075beta-0.001kl-2025-05-01  # Directory containing checkpoints to evaluate
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/contrastive/sokoban-1.5b-contrastive-qwen-2.5-base-full-sft-05-26
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/sokoban-1.5b-full-sft-lr-1e-5-06-17
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/REIL/sokoban-qwen3-1.7b-0.000beta-0.000kl-08-10-grpo
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/sft/ultradiverse_sokoban-8b-full-sft-lr-1e-5-08-17
CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/sft/super_random-sokoban-1.5b-standard-lr-5e-6-08-25
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging

# Evaluation settings
N_GPUS=4
# Print configuration
echo "Running evaluation with the following configuration:"
# echo "Model path: $BASE_MODEL"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPUs per node: $N_GPUS"


# Run evaluation
python -m reil.evaluation.eval_ckpts \
    evaluator.checkpoint_dir=$CHECKPOINT_DIR \
    evaluator.project_name=$PROJECT_NAME \
    evaluator.experiment_name=$EXPERIMENT_NAME \
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=0 \
    evaluator.is_lora=False \
    es_manager.val.env_groups=1200 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'SimpleSokobanEmoji', 'FakeSokobanEmoji', 'SimpleSokobanCardinal', 'FakeSokobanCardinal', 'SimpleSokobanEmpty', 'TwoBoxesSokoban', 'SimpleSokobanNumerical', 'SimpleSokobanAlphabetical', 'SimpleSokobanRandom']" \
    es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100,100,100,100,100]" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    agent_proxy.max_turn=30 \