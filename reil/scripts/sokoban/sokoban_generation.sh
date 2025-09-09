# BASE_MODEL="./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
# BASE_MODEL="./checkpoints/ds310/REIL/sokoban-1.5b-0.000beta-0.000kl-08-04-grpo/huggingface"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# BASE_MODEL="./models/rlft/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
DATASETS="Xiaofeng77/sokoban_one_horizon_large_envs"
OUTPUT_DIR="./results"
NUM_GENERATION=16
N_GPUS=1
BATCH_SIZE=256
python -m reil.evaluation.generation \
    --model_path $BASE_MODEL \
    --datasets $DATASETS \
    --output_dir $OUTPUT_DIR \
    --num_generation $NUM_GENERATION \
    --num_gpus $N_GPUS \
    --batch_size $BATCH_SIZE \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --split "train" \
    --max_tokens 4096 \
    --rejection_sampling 
