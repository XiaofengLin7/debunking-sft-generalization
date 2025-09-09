#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate reil || exit 1

set -x

# checkpoint_dir="./checkpoints/ds543/REIL/alfworld-1.5b-0.005beta-0.001kl-pick_n_place_05-07/global_step_1500/actor"
checkpoint_dir="/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/REIL/sokoban-1.5b-0.000beta-0.000kl-08-04-grpo/global_step_3000/actor"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
target_dir="./checkpoints/ds310/REIL/sokoban-1.5b-0.000beta-0.000kl-08-04-grpo/huggingface"

python3 -m reil.utils.models.model_merger --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir 