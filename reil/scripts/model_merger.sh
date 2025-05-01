

checkpoint_dir="./checkpoints/REIL/exp-1.5b-0.005beta-logic-with-kl-0.001-1024-ctx-one-horizon/global_step_5000/actor"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
target_dir="./checkpoints/REIL/exp-1.5b-0.005beta-logic-with-kl-0.001-1024-ctx-one-horizon/huggingface"

python3 -m reil.utils.models.model_merger --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir \