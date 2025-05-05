

checkpoint_dir="./checkpoints/REIL/alfworld-1.5b-0.005beta-0.0015kl-2025-04-28/global_step_2400/actor"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
target_dir="./checkpoints/ds543/REIL/alfworld-1.5b-0.005beta-0.0015kl-2025-04-28/huggingface"

python3 -m reil.utils.models.model_merger --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir \
    --hf_upload_path "Xiaofeng77/alfworld-1.5b"