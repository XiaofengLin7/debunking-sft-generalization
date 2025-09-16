#!/bin/bash

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=2
# DATA_DIR="./data/sokoban-answer-only/sft"
DATA_DIR="./data/sokoban_one_horizon_large_envs/cot-sft"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
# BASE_MODEL="./models/rlft/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
LEARNING_RATE=1e-5
SFT_TYPE="standard" # "aft", "dft", "standard"
AFT_POWER=1.0
ANCHOR_COEF=0.5
KL_COEF=0 
EXPERIMENT_NAME="cot-sokoban-1.5b-${SFT_TYPE}-lr-${KL_COEF}-kl-${LEARNING_RATE}-anchor-${ANCHOR_COEF}-$(date +%m-%d)"

CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=5000 \
    data.train_batch_size=32 \
    data.chat_template=False \
    data.max_response_length=4096 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=2 \
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
    trainer.anchor_regularization.enabled=True \
    trainer.anchor_regularization.l2_anchor_coeff=${ANCHOR_COEF} \
    trainer.policy_eval=False \
    trainer.project_name=REIL \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds310/sft/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=5 \
    trainer.val_before_train=False \
    trainer.kl_regularization.enabled=False \
    trainer.kl_regularization.kl_coef=${KL_COEF} \
    trainer.default_hdfs_dir=null $@ | tee checkpoints/ds310/sft/${EXPERIMENT_NAME}_train.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \
#     trainer.test_freq=10 \
#     trainer.save_freq=10 \