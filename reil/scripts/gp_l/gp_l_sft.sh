eval "$(conda shell.bash hook)"
conda activate verl-test || exit 1

set -x

# Shift the arguments so $@ refers to the rest
shift 2
N_GPUS=4

# DATA_DIR="./data/alfworld_task_type/sft"
# DATA_DIR="./data/sokoban_one_horizon_large_envs/sft"
DATA_DIR="./data/gp-l-only/sft"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# EXPERIMENT_NAME="alfworld-1.5b-pick_n_place-sft-qwen-2.5-base-full-sft"
LEARNING_RATE=1e-5
EXPERIMENT_NAME="gp-l-1.5b-full-sft-lr-${LEARNING_RATE}-$(date +%m-%d)"

export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export ALFWORLD_DATA="/projectnb/replearn/xfl/Retriever/src/envs/alf_world/data_storage"

torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS \
     -m reil.trainer.fsdp_sft_trainer \
    data.train_files=$DATA_DIR/train-10k.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.prompt_key=question \
    data.response_key=answer \
    data.chat_template=True \
    data.max_length=2048 \
    data.train_batch_size=512 \
    data.max_response_length=200 \
    optim.lr=$LEARNING_RATE \
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$BASE_MODEL \
    model.fsdp_config.cpu_offload=False \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    trainer.policy_eval=False \
    trainer.project_name=REIL-sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=checkpoints/ds310/sft/$EXPERIMENT_NAME \
    trainer.logger="['console']" \
    trainer.total_epochs=30 \
    trainer.default_hdfs_dir=null $@ | tee checkpoints/ds310/sft/${EXPERIMENT_NAME}_train.log