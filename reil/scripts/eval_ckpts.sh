#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate reil || exit 1

set -x

export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"

# Model and checkpoint settings
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"  # Base model path
CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/REIL/alfworld-1.5b-0.005beta-0.001kl-pick_n_place_05-07  # Directory containing checkpoints to evaluate
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging

# Evaluation settings
N_GPUS=4                      # Number of GPUs per node

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
    evaluator.n_gpus_per_node=$N_GPUS 