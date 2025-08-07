#!/bin/bash

set -x
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# Model and checkpoint settings
CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/gp-l-1.5b-full-sft-lr-1e-5-08-01
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging
test_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/rl/test.parquet
test_5cards_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/rl/test_5cards.parquet
test_fake_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/rl/test_fake.parquet
test_large_card_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/rl/test_large_card.parquet
TEST_DATA="['${test_data_path}', '${test_5cards_data_path}', '${test_fake_data_path}', '${test_large_card_data_path}']"
# Evaluation settings
N_GPUS=4                      # Number of GPUs per node
export TOKENIZERS_PARALLELISM=false
# Print configuration
echo "Running evaluation with the following configuration:"
# echo "Model path: $BASE_MODEL"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPUs per node: $N_GPUS"


# Run evaluation
python -m reil.evaluation.eval_ckpts \
    +data.val_score_files="$TEST_DATA" \
    +data.prompt_key=question \
    +data.chat_template=True \
    data.max_prompt_length=1024 \
    +data.filter_overlong_prompts=True \
    evaluator.checkpoint_dir=$CHECKPOINT_DIR \
    evaluator.project_name=$PROJECT_NAME \
    evaluator.experiment_name=$EXPERIMENT_NAME \
    evaluator.n_gpus_per_node=$N_GPUS \
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=100 \
    es_manager.val.env_groups=768 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['GP-L', 'GP-L-FACE-CARDS-AS-REGULAR', 'GP-L-FACE-CARDS-AS-10']" \
    es_manager.val.env_configs.n_groups="[256, 256, 256]" \
    agent_proxy.max_turn=1 \
    agent_proxy.parse_response=False \
    agent_proxy.chat_template=True \
    +reward_model.reward_manager=gp_l