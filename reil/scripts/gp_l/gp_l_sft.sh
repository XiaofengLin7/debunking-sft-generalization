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
DATA_DIR="./data/gp-l-only/10k-non-mixed/cot-sft"
# BASE_MODEL=/usr3/graduate/xfl/lab/REIL/models/rlft/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796"
LEARNING_RATE=1e-5
EXPERIMENT_NAME="qwen-2.5-7b-non-diverse-gp-l-non-diverse-cot-lr-${LEARNING_RATE}-$(date +%m-%d)"

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
    reward_model.reward_manager=gp_l $@ | tee checkpoints/ds543/sft/${EXPERIMENT_NAME}_train.log