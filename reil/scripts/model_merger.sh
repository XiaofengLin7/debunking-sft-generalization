

checkpoint_dir="./checkpoints/REIL/sokoban-rl-exp-1.5b-0.01beta-logic-with-kl/global_step_1500/actor"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
target_dir="./checkpoints/REIL/sokoban-rl-exp-1.5b-0.01beta-logic-with-kl/huggingface"

python3 ./thirdparty/verl/scripts/model_merger.py --backend fsdp \
    --hf_model_path $BASE_MODEL \
    --local_dir $checkpoint_dir \
    --target_dir $target_dir