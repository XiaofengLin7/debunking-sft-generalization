VERIFIER_PATH="./models/rlft/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
FILE_PATH="./results/small_sokoban-train-temp_1.0-top_p_1.0-top_k_-1.jsonl"
BATCH_SIZE=512
NUM_GENERATIONS=7

python -m reil.evaluation.reason_eval \
    --verifier_path $VERIFIER_PATH \
    --file_path $FILE_PATH \
    --batch_size $BATCH_SIZE \
    --num_generations $NUM_GENERATIONS