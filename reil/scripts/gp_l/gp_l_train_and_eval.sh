#!/bin/bash

set -x

# ---------------------------
# Training (SFT)
# ---------------------------
N_GPUS=4
DATA_DIR="./data/gp-l-only/10k-non-mixed/cot-sft"
# BASE_MODEL=YOUR_BASE_MODEL
BASE_MODEL=YOUR_BASE_MODEL
LEARNING_RATE=1e-5
EXPERIMENT_NAME="qwen-2.5-7b-non-diverse-cot-gp-l-lr-${LEARNING_RATE}-$(date +%m-%d)"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test_id.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.chat_template=True \
    data.max_length=5000 \
    data.train_batch_size=128 \
    data.max_response_length=4096 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=1 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds543/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=5 \
    trainer.default_hdfs_dir=null \
    trainer.val_before_train=False \
    reward_model.reward_manager=gp_l "$@" | tee checkpoints/ds543/sft/${EXPERIMENT_NAME}_train.log

# ---------------------------
# Evaluation
# ---------------------------
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export TOKENIZERS_PARALLELISM=false

# Derive checkpoint directory from the EXPERIMENT_NAME above
CHECKPOINT_DIR=$(realpath checkspoints 2>/dev/null || true)
CHECKPOINT_DIR=$(realpath "checkpoints/ds543/sft/${EXPERIMENT_NAME}")
BASE_MODEL=$(realpath $BASE_MODEL)
CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
PROJECT_NAME="REIL"
EVAL_EXPERIMENT_NAME="eval_${CHECKPOINT_NAME}"

# Test data for GP-L evaluation
test_5cards_data_path=YOUR_DATA
test_fake_data_path=YOUR_DATA
test_large_card_data_path=YOUR_DATA
test_face_card_as_regular_data_path=YOUR_DATA
test_all_12_data_path=YOUR_DATA
test_id_data_path=YOUR_DATA
test_all_5_data_path=YOUR_DATA
test_all_7_data_path=YOUR_DATA
test_all_5_fake_data_path=YOUR_DATA
TEST_DATA="['${test_5cards_data_path}', '${test_fake_data_path}', '${test_large_card_data_path}', '${test_face_card_as_regular_data_path}', '${test_all_12_data_path}', '${test_id_data_path}', '${test_all_5_data_path}', '${test_all_7_data_path}', '${test_all_5_fake_data_path}']"


echo "Running evaluation with the following configuration:"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Project name: $PROJECT_NAME"
echo "Experiment name: $EVAL_EXPERIMENT_NAME"
echo "GPUs per node: $EVAL_N_GPUS"

python -m reil.evaluation.eval_ckpts \
    +data.val_score_files="$TEST_DATA" \
    data.prompt_key=question \
    +data.chat_template=True \
    data.max_prompt_length=1024 \
    +data.filter_overlong_prompts=False \
    data.max_response_length=8192 \
    evaluator.checkpoint_dir="$CHECKPOINT_DIR" \
    evaluator.project_name="$PROJECT_NAME" \
    evaluator.experiment_name="$EVAL_EXPERIMENT_NAME" \
    evaluator.logger="['console', 'wandb']" \
    evaluator.resume_step=0 \
    evaluator.is_lora=False \
    es_manager.val.env_groups=768 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['GP-L', 'GP-L-FACE-CARDS-AS-REGULAR', 'GP-L-FACE-CARDS-AS-10']" \
    es_manager.val.env_configs.n_groups="[256, 256, 256]" \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    agent_proxy.max_turn=1 \
    agent_proxy.parse_response=False \
    agent_proxy.chat_template=True \
    reward_model.reward_manager=gp_l


