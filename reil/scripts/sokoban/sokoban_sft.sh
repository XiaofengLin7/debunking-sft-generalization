#!/bin/bash

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=4

# DATA_DIR="./data/sokoban_one_horizon_large_envs/sft"
DATA_DIR="./data/sokoban_one_horizon_large_envs/mixed/sft"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
BASE_MODEL="./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
LEARNING_RATE=1e-5
SFT_TYPE="aft" # "aft", "dft", "standard"
AFT_POWER=5
EXPERIMENT_NAME="mixed-sokoban-8b-${SFT_TYPE}-power-${AFT_POWER}-lr-${LEARNING_RATE}-$(date +%m-%d)"


export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.train_batch_size=64 \
    data.chat_template=False \
    data.max_response_length=200 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=4 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    es_manager.val.env_groups=900 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'SimpleSokobanEmoji', 'FakeSokobanEmoji', 'SimpleSokobanCardinal', 'FakeSokobanCardinal', 'SimpleSokobanEmpty', 'TwoBoxesSokoban']" \
    es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100,100]" \
    agent_proxy.max_turn=30 \
    trainer.sft_type=$SFT_TYPE \
    trainer.aft_power=$AFT_POWER \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds543/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.default_hdfs_dir=null $@ | tee checkpoints/ds543/sft/${EXPERIMENT_NAME}_train.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
#     trainer.test_freq=10 \
#     trainer.save_freq=10 \