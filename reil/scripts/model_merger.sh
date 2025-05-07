

checkpoint_dir="./checkpoints/ds543/REIL/sokoban-1.5b-0.0075beta-0.001kl-2025-05-01/global_step_4000/actor"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
target_dir="./checkpoints/ds543/REIL/sokoban-1.5b-0.0075beta-0.001kl-2025-05-01/huggingface"

python3 -m reil.utils.models.model_merger --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir \
    --hf_upload_path "Xiaofeng77/sokoban-1.5b"