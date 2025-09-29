#!/bin/bash

set -x

# Model and checkpoint settings
# BASE_MODEL=YOUR_BASE_MODEL  # Base model path
BASE_MODEL=YOUR_BASE_MODEL
CHECKPOINT_DIR=YOUR_CHECKPOINT_DIR
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging
TEST_DATA=YOUR_DATA
# Evaluation settings
N_GPUS=4
# CUDA_VISIBLE_DEVICES=0
TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Print configuration
echo "Running evaluation with the following configuration:"
# echo "Model path: $BASE_MODEL"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPUs per node: $N_GPUS"
# CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# Run evaluation
python -m reil.evaluation.eval_ckpts \
    evaluator.checkpoint_dir=$CHECKPOINT_DIR \
    evaluator.project_name=$PROJECT_NAME \
    evaluator.experiment_name=$EXPERIMENT_NAME \
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
    custom_reward_function.path=YOUR_REPO/reil/utils/reward_score/sokoban.py  
    
