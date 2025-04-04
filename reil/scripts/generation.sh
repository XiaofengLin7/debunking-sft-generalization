
eval "$(conda shell.bash hook)"
conda activate reil || exit 1

set -x

DATA_DIR="./data/sokoban"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
MODEL_PATH="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# MODEL_PATH="./checkpoints/REIL/sokoban-rl-exp-1.5b-0.01beta-logic-with-kl/huggingface"
# EXPERIMENT_NAME="sokoban-rl-exp-1.5b-0.01beta-logic-with-kl"
EXPERIMENT_NAME="base_model_0.5b"
# ROLLOUT_TP_SIZE=1
N_GPUS=2
# export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m reil.trainer.main_generation \
data.path=$DATA_DIR/test.parquet \
model.path=$MODEL_PATH \
data.output_path="./data/sokoban/${EXPERIMENT_NAME}/test_generation.parquet" \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \