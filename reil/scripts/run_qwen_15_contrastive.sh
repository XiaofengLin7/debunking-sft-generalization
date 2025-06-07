
eval "$(conda shell.bash hook)"
conda activate reil || exit 1

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=4

# DATA_DIR="./data/alfworld_task_type/sft"
DATA_DIR="./data/sokoban_one_horizon_large_envs/constrastive"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
ratio=1
# EXPERIMENT_NAME="alfworld-1.5b-pick_n_place-sft-qwen-2.5-base-full-sft"
EXPERIMENT_NAME="sokoban-1.5b-contrastive-qwen-2.5-base-full-sft-$(date +%m-%d)-ratio-$ratio"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_contrastive_trainer \
    data.pos_train_files=$DATA_DIR/train.parquet \
    data.neg_train_files=$DATA_DIR/train_negative.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.train_batch_size=256 \
    data.max_response_length=200 \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    es_manager.val.env_groups=512 \
    es_manager.val.group_size=1 \
    es_manager.val.env_configs.tags="['LargerSokoban','SimpleSokoban']" \
    es_manager.val.env_configs.n_groups="[256,256]" \
    agent_proxy.max_turn=14 \
    algorithm.contrastive_loss.gamma=0.5 \
    algorithm.contrastive_loss.ratio=$ratio \
    trainer.policy_eval=True \
    trainer.project_name=REIL-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds310/contrastive/$EXPERIMENT_NAME \
    trainer.logger="['console', 'wandb']" \
    trainer.total_epochs=30 \
    trainer.default_hdfs_dir=null $@ | tee checkpoints/ds310/contrastive/${EXPERIMENT_NAME}_train.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \