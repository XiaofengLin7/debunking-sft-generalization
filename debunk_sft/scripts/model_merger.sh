#!/bin/bash

set -x

# checkpoint_dir="YOUR_CHECKPOINT_DIR/actor"
checkpoint_dir="YOUR_CHECKPOINT_DIR/actor"
BASE_MODEL="YOUR_BASE_MODEL"
target_dir="YOUR_TARGET_DIR"

python3 -m debunk_sft.utils.models.model_merger --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir 