
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"

LORA_MODEL="./checkpoints/sft/sokoban-1.5b-sft-qwen-2.5-1.5b-base/global_step_300"

OUTPUT_PATH="./checkpoints/sft/sokoban-1.5b-sft-qwen-2.5-1.5b-base-merged"

python3 -m reil.utils.models.merge_lora \
    --base_model_name $BASE_MODEL \
    --lora_model_path $LORA_MODEL \
    --output_path $OUTPUT_PATH
