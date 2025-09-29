#!/bin/bash
set -x


DATA_DIR="./data/gp-l-only/10k-non-mixed/rl"
# BASE_MODEL=YOUR_BASE_MODEL
BASE_MODEL=YOUR_BASE_MODEL

# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/gp-l-1.5b-full-sft-lr-1e-5-08-01/global_step_95"
# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/ds310/sft/gp-l-1.5b-full-sft-lr-1e-5-08-01/global_step_570"
# BASE_MODEL="/usr3/graduate/xfl/lab/REIL/checkpoints/sft/sokoban-1.5b-sft-qwen-2.5-base-full-sft-05-15/global_step_180"

test_5cards_data_path=YOUR_DATA
test_fake_data_path=YOUR_DATA
test_large_card_data_path=YOUR_DATA
test_face_card_as_regular_data_path=YOUR_DATA
test_all_12_data_path=YOUR_DATA
test_id_data_path=YOUR_DATA
test_all_5_data_path=YOUR_DATA
test_all_7_data_path=YOUR_DATA
test_all_5_fake_data_path=YOUR_DATA
TEST_DATA="['${test_5cards_data_path}', '${test_fake_data_path}', '${test_large_card_data_path}', '${test_face_card_as_regular_data_path}', '${test_all_12_data_path}', '${test_id_data_path}', '${test_all_5_data_path}', '${test_all_7_data_path}', '${test_all_5_fake_data_path}']"

BETA=0
KL_COEF=0

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

CONTEXT_LENGTH=4096
BATCH_SIZE=256
EXPERIMENT_NAME="gp-l-1.5b-${BETA}beta-${KL_COEF}kl-$(date +%m-%d)-grpo"
#EXPERIMENT_NAME="small_sokoban-1.5b-${BETA}beta-${KL_COEF}kl-06-18-grpo"
ROLLOUT_TP_SIZE=1
N_GPUS=4
export VLLM_USE_V1=1

python3 -m reil.trainer.main_ppo \
data.train_files=$DATA_DIR/train.parquet \
data.val_files="$TEST_DATA" \
data.train_batch_size=$BATCH_SIZE \
data.val_batch_size=32 \
data.max_prompt_length=1000 \
data.prompt_key=question \
data.max_response_length=$CONTEXT_LENGTH \
data.chat_template=True \
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
trainer.save_freq=100 \
trainer.test_freq=50 \
trainer.project_name=REIL \
trainer.resume_mode=auto \
trainer.log_val_generations=4 \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=checkpoints/ds310/REIL/${EXPERIMENT_NAME} \
trainer.total_epochs=2000 \
trainer.policy_eval=False \
reward_model.reward_manager=gp_l 2>&1 | tee gp-l-1.5b.log


# \
# es_manager.val.env_groups=1000 \
# es_manager.val.group_size=1 \
# es_manager.val.env_configs.tags="['GP-L',  'MediumGP-L', 'HardGP-L', 'GP-L-FACE-CARDS-AS-REGULAR', 'GP-L-FACE-CARDS-AS-10']" \
# es_manager.val.env_configs.n_groups="[200, 200, 200, 200, 200]" \
# agent_proxy.max_turn=1 \
# agent_proxy.parse_response=False \
# agent_proxy.chat_template=True