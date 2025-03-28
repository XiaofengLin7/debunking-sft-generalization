from verl.workers.reward_manager import NaiveRewardManager
from reil.utils.reward_score.sokoban import compute_score
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.workers.fsdp_workers import ActorRolloutRefWorker
import ray
import hydra
@ray.remote(num_cpus=1)
def test_naive_reward_manager(config):
    local_path = "./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    data_dir = "./data/sokoban/test.parquet"
    
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)
    actor_path = "./checkpoints/REIL/sokoban-rl-exp-0.5b-format-reward/global_step_3000/actor"
    val_reward_fn = NaiveRewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)
    n_gpus_per_node = 0
    # load data
    val_dataset = RLHFDataset(parquet_files=data_dir,
                                tokenizer=tokenizer,
                                processor=processor,
                                prompt_key="prompt",
                                image_key="images",
                                max_prompt_length=512,
                                filter_prompts=True,
                                return_raw_chat=False,
                                truncation='error',
                                filter_overlong_prompts=False)
    
    val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)
    
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role='rollout')
    resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * 1)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    wg.init_model()
    wg.load_checkpoint(actor_path, del_local_after_load=False)
    for batch_dict in val_dataloader:
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        gen_batch = batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=['raw_prompt_ids'],
        )

        print(gen_batch)
        # TODO: do inference

@hydra.main(config_path='config', config_name='generation', version_base=None)
def main(config):
    test_naive_reward_manager(config)

if __name__ == "__main__":
    main()
    
    