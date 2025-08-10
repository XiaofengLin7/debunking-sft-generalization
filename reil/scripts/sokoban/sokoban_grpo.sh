#!/bin/bash
set -x


DATA_DIR="./data/sokoban_one_horizon_large_envs"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-3B/snapshots/3aab1f1954e9cc14eb9509a215f9e5ca08227a9b"
# BASE_MODEL="./models/rlft/models--deepseek-ai--DeepSeek-R1-0528-Qwen3-8B/snapshots/6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
# BASE_MODEL="./models/rlft/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
BASE_MODEL="./models/rlft/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/sft/sokoban-1.5b-sft-qwen-2.5-base-full-sft-05-15/global_step_180"
BETA=0.000
KL_COEF=0.000

# Parse named arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --beta)
      BETA="$2"
      shift # past argument
      shift # past value
      ;;
    --kl)
      KL_COEF="$2"
      shift
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--beta BETA] [--kl KL_COEF]"
      exit 0
      ;;
    *)    # unknown option
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

CONTEXT_LENGTH=1024
BATCH_SIZE=256
EXPERIMENT_NAME="sokoban-qwen3-1.7b-${BETA}beta-${KL_COEF}kl-$(date +%m-%d)-grpo"
#EXPERIMENT_NAME="small_sokoban-1.5b-${BETA}beta-${KL_COEF}kl-06-18-grpo"
ROLLOUT_TP_SIZE=1
N_GPUS=4
export VLLM_USE_V1=1

python3 -m reil.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files=$DATA_DIR/test.parquet \
data.train_batch_size=$BATCH_SIZE \
data.val_batch_size=32 \
data.max_prompt_length=1000 \
data.max_response_length=$CONTEXT_LENGTH \
algorithm.adv_estimator=grpo \
algorithm.use_kl_in_reward=True \
algorithm.kl_ctrl.kl_coef=${KL_COEF} \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
actor_rollout_ref.actor.entropy_coeff=${BETA} \
actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
actor_rollout_ref.rollout.n=8 \
trainer.logger=['wandb'] \
+trainer.val_only=False \
trainer.val_before_train=True \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=500 \
trainer.test_freq=50 \
trainer.project_name=REIL \
trainer.resume_mode=auto \
trainer.log_val_generations=4 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=checkpoints/ds310/REIL/${EXPERIMENT_NAME} \
trainer.total_epochs=500 \
trainer.policy_eval=True \
reward_model.reward_manager=complete \
custom_reward_function.path=./reil/utils/reward_score/sokoban.py \
es_manager.val.env_groups=900 \
es_manager.val.group_size=1 \
es_manager.val.env_configs.tags="['SimpleSokoban', 'LargerSokoban', 'ComplexSokoban', 'SimpleSokobanEmoji', 'FakeSokobanEmoji', 'SimpleSokobanCardinal', 'FakeSokobanCardinal', 'SimpleSokobanEmpty', 'TwoBoxesSokoban']" \
es_manager.val.env_configs.n_groups="[100,100,100,100,100,100,100,100,100]" \
agent_proxy.max_turn=30 \
custom_reward_function.name=compute_score_with_action_sequence 2>&1 | tee sokoban_1.5b.log
