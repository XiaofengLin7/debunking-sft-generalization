#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate reil || exit 1

set -x


DATA_DIR="./data/sokoban_one_horizon_large_envs"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
N_GPUS=4
accelerate launch --config_file reil/trainer/config/fsdp.yaml --num_processes $N_GPUS -m reil.trainer.fsdp_dpo_trainer \
    actor_rollout_ref.model.path=$BASE_MODEL \
    dataset.name=Xiaofeng77/reil_sokoban_preference \
    output_dir=checkpoints/ds310/dpo_model \
    dpo.learning_rate=1e-4 \
    dpo.batch_size=16 \
    dpo.num_epochs=200 \
    dpo.gradient_accumulation_steps=4 \
    dpo.max_grad_norm=1.0 \
    dpo.beta=0.1 \
    dpo.logging_steps=1 \
    dpo.save_steps=200 \
    dpo.eval_steps=2 \
    es_manager.val.env_groups=512 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['LargerSokoban','SimpleSokoban']" \
    es_manager.val.env_configs.n_groups="[256,256]" \
    evaluator.policy_eval=False \
    evaluator.experiment_name=sokoban_dpo_1.5b
