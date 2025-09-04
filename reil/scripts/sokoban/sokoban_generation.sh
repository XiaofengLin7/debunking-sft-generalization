BASE_MODEL="./models/rlft/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DATASETS="Xiaofeng77/sokoban_one_horizon_large_envs"
OUTPUT_DIR="./results"
NUM_GENERATION=3
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
