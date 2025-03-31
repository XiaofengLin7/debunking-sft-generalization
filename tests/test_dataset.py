from reil.utils.reward_score.sokoban import compute_score
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl import DataProto

def test_dataset():
    local_path = "./models/rlft/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    data_dir = "./data/sokoban/test.parquet"
    
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)
    val_dataset = RLHFDataset(parquet_files=data_dir,
                                tokenizer=tokenizer,
                                processor=processor,
                                prompt_key="prompt",
                                image_key="images",
                                max_prompt_length=512,
                                filter_prompts=True,
                                return_raw_chat=True,
                                truncation='error',
                                filter_overlong_prompts=False,
                                add_generation_prompt=False)
    
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        # Validation datasets are sent to inference engines as a whole batch,
        # which will schedule the memory themselves.
        batch_size=len(val_dataset),
        num_workers=8,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn)
    
    for batch_dict in val_dataloader:
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        gen_batch = batch.pop(
            batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            non_tensor_batch_keys=['raw_prompt_ids'],
        )
        print(gen_batch)
        raw_prompt_ids = gen_batch.non_tensor_batch['raw_prompt_ids']
        print(tokenizer.decode(raw_prompt_ids[0], skip_special_tokens=False))

if __name__ == "__main__":
    test_dataset()

