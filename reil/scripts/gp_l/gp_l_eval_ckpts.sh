#!/bin/bash

set -x
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# Model and checkpoint settings
CHECKPOINT_DIR=/usr3/graduate/xfl/lab/REIL/checkpoints/ds543/sft/gp-l-10k-non-mixed-8b-lora-32-lr-1e-5-08-20
BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
CHECKPOINT_NAME=$(basename $CHECKPOINT_DIR)  # Extract the last segment of the path
PROJECT_NAME="REIL"      # Project name for logging
EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"    # Experiment name for logging
test_5cards_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_5cards.parquet
test_fake_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_fake.parquet
test_large_card_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_large.parquet
test_face_card_as_regular_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_face_card_as_regular.parquet
test_all_12_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_all_12.parquet
test_id_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_id.parquet
test_all_5_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_all_5.parquet
test_all_7_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/20k-mixed/rl/test_all_7.parquet
TEST_DATA="['${test_5cards_data_path}', '${test_fake_data_path}', '${test_large_card_data_path}', '${test_face_card_as_regular_data_path}', '${test_all_12_data_path}', '${test_id_data_path}', '${test_all_5_data_path}', '${test_all_7_data_path}']"
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
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=0 \
    evaluator.is_lora=True \
    es_manager.val.env_groups=768 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['GP-L', 'GP-L-FACE-CARDS-AS-REGULAR', 'GP-L-FACE-CARDS-AS-10']" \
    es_manager.val.env_configs.n_groups="[256, 256, 256]" \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    agent_proxy.max_turn=1 \
    agent_proxy.parse_response=False \
    agent_proxy.chat_template=True \
    +reward_model.reward_manager=gp_l