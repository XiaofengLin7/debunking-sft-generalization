set -x

DATA_DIR="./data/sokoban"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
EXPERIMENT_NAME="sokoban-rl-exp-0.5b"
ROLLOUT_TP_SIZE=1
N_GPUS=2

python3 -m reil.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=256 \
data.val_batch_size=32 \
data.max_prompt_length=400 \
data.max_response_length=400 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size=8 \
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
critic.optim.lr=1e-5 \
critic.model.path=$BASE_MODEL \
critic.ppo_micro_batch_size=8 \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=100 \
trainer.test_freq=100 \
trainer.project_name=REIL \
trainer.resume_mode=disabled \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.total_epochs=500 \
custom_reward_function.path=./reil/utils/reward_score/sokoban.py \
custom_reward_function.name=compute_score 2>&1 | tee verl_demo.log