#!/bin/bash

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=4
# test_data_path=./data/gp-l-only/mixed/sft/test.parquet
# test_5cards_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/10k-non-mixed/sft/test_5cards.parquet
# test_fake_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/10k-non-mixed/sft/test_fake.parquet
# test_large_card_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/10k-non-mixed/sft/test_large.parquet
# test_face_card_as_regular_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/10k-non-mixed/sft/test_face_card_as_regular.parquet
# test_all_12_data_path=/usr3/graduate/xfl/lab/REIL/data/gp-l-only/10k-non-mixed/sft/test_all_12.parquet
# TEST_DATA="['${test_5cards_data_path}', '${test_fake_data_path}', '${test_large_card_data_path}', '${test_face_card_as_regular_data_path}', '${test_all_12_data_path}']"
DATA_DIR="./data/gp-l-only/10k-non-mixed/sft"
BASE_MODEL=./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
LEARNING_RATE=1e-5
EXPERIMENT_NAME="gp-l-10k-non-mixed-8b-full-sft-lr-${LEARNING_RATE}-$(date +%m-%d)"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train-10k.parquet \
    data.val_files=$DATA_DIR/test_id.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.chat_template=True \
    data.max_length=2048 \
    data.train_batch_size=64 \
    data.max_response_length=200 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=4 \
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
    reward_model.reward_manager=gp_l $@ | tee checkpoints/ds543/sft/${EXPERIMENT_NAME}_train.log