
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"

LORA_MODEL="./checkpoints/ds543/sft/alfworld-1.5b-sft-qwen-2.5-1.5b-base-lora-32/global_step_120"

OUTPUT_PATH="./checkpoints/ds543/sft/alfworld-1.5b-sft-qwen-2.5-1.5b-base-lora-32/huggingface"

CUDA_VISIBLE_DEVICES=3

python3 -m reil.utils.models.merge_lora \
    --base_model_name $BASE_MODEL \
    --lora_model_path $LORA_MODEL \
    --output_path $OUTPUT_PATH \
    --hf_upload_path "Xiaofeng77/alfworld-1.5b-lora"
