#!/bin/bash

set -x

# Model and checkpoint settings
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659  # Base model path
BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/REIL/sokoban-1.5b-0.0075beta-0.001kl-2025-05-01  # Directory containing checkpoints to evaluate
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/contrastive/sokoban-1.5b-contrastive-qwen-2.5-base-full-sft-05-26
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/sokoban-1.5b-full-sft-lr-1e-5-06-17
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/REIL/sokoban-qwen3-1.7b-0.000beta-0.000kl-08-10-grpo
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/sft/ultradiverse_sokoban-8b-full-sft-lr-1e-5-08-17
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/rjs-sokoban-1.5b-standard-lr-1e-5-09-08
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/sokoban-1.5b-standard-lr-1e-5-09-06
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/sft/cot-sokoban-1.5b-standard-lr-1e-5-09-04
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/sokoban-1.5b-standard-lr-1-kl-1e-5-09-14
# CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/llama-3.1-8b-instruct-non-diverse-cot-sokoban-standard-lr-1e-5-anchor-0-09-16
CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/qwen-2.5-7b-non-diverse-cot-sokoban-standard-lr-1e-5-anchor-0-09-17
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging
TEST_DATA=/usr3/graduate/xfl/lab/REIL/data/sokoban_one_horizon_large_envs/train.parquet
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
    custom_reward_function.path=/usr3/graduate/xfl/lab/REIL/reil/utils/reward_score/sokoban.py  
    
