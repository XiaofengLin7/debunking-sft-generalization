#!/bin/bash

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=4

# DATA_DIR="./data/sokoban_one_horizon_large_envs/sft"
DATA_DIR="./data/sokoban_one_horizon_large_envs/ultradiverse/sft"
BASE_MODEL="./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LEARNING_RATE=1e-5
EXPERIMENT_NAME="ultradiverse-sokoban-3b-lora-32-$(date +%m-%d)"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m debunk_sft.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.train_batch_size=128 \
    data.chat_template=False \
    data.max_response_length=200 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    es_manager.val.env_groups=1200 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'SimpleSokobanEmoji', 'FakeSokobanEmoji', 'SimpleSokobanCardinal', 'FakeSokobanCardinal', 'SimpleSokobanEmpty', 'TwoBoxesSokoban', 'SimpleSokobanNumerical', 'SimpleSokobanAlphabetical', 'SimpleSokobanRandom']" \
    es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100,100,100,100,100]" \
    agent_proxy.max_turn=30 \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds310/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=30 \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    model.lora_rank=32\
    model.lora_alpha=16 \
    model.target_modules=all-linear $@ 2>&1 | tee checkpoints/ds310/sft/${EXPERIMENT_NAME}_train.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj]