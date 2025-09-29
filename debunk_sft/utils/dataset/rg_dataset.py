"""
Adapted from http://github.com/open-thought/reasoning-gym/blob/main/examples/veRL/grpo_train.py#L117
"""

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Optional, Tuple, Literal
import verl.utils.torch_functional as verl_F
from torch.utils.data import Dataset
from verl.utils.model import compute_position_id_with_mask
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.composite import CompositeDataset, DatasetSpec
import reasoning_gym
from omegaconf import OmegaConf
import torch
import numpy as np
response_template = """\
<think></think> <answer> {answer} </answer>
"""
class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        q = row_dict["question"]

        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})

        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["data_source"] = "reasoning_gym"
        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        row_dict["raw_prompt"] = chat
        row_dict["index"] = index
        return row_dict

    def update_experiment_difficulty(self, dataset_name: str, method: Literal["increment", "decrement"]):
        """Update the difficulty of the underlying dataset."""
        if self.experiment is None:
            raise ValueError("Cannot update difficulty: dataset is not a CurriculumExperiment")
        if method not in ["increment", "decrement"]:
            raise ValueError("Invalid method: must be 'increment' or 'decrement'")
        self.experiment.score_board.clear(dataset_name)
        self.experiment.update_difficulty(dataset_name, method)
        self.data = self.experiment.composite
        return True

    def aggregate(self, last_n: Optional[int] = None):
        """Aggregate scores from the underlying experiment"""
        if self.experiment is None:
            raise ValueError("Cannot aggregate scores: dataset is not a CurriculumExperiment")

        results = self.experiment.score_board.aggregate(last_n=last_n)
        output_results = {}

        for key, value in results.items():
            output_results[key] = {}
            scores = value.scores
            first_key = list(scores.keys())[0]
            output_results[key]["results"] = np.mean(scores[first_key])
            output_results[key]["total_samples"] = value.total_scores
        return output_results

class ReasoningGymSFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_length = max_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        """
        follow implementation in verl/utils/dataset/sft_dataset.py and ReasoningGymDataset
        """
        row_dict = self.data[index].copy()
        q = row_dict["question"]
        a = row_dict["answer"]
        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})

        prompt_chat_str = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        response = response_template.format(answer=a)
        # response_chat_str = response + self.tokenizer.eos_token
        response_chat_str = response
        # tokenize
        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False, padding_side='left')
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = self.tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False, padding_side='left')
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length left padding
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((padded_input_ids, input_ids))
            attention_mask = torch.cat((padded_attention_mask, attention_mask))

            padded_prompt_ids = torch.ones(size=(self.max_length - prompt_length,),
                                          dtype=prompt_ids.dtype) * self.tokenizer.pad_token_id
            padded_prompt_attention_mask = torch.zeros(size=(self.max_length - prompt_length,), dtype=prompt_attention_mask.dtype)

            prompt_ids = torch.cat((padded_prompt_ids, prompt_ids))
            prompt_attention_mask = torch.cat((padded_prompt_attention_mask, prompt_attention_mask))

        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')
        
        position_ids = compute_position_id_with_mask(attention_mask)
        prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)
        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            'prompt_ids': prompt_ids,
            'prompt_attention_mask': prompt_attention_mask,
            'prompt_position_ids': prompt_position_ids,
            'index': index,
        }



def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    max_prompt_length: int = 2048,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    """
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    else:
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    
def make_sft_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    max_length: int = 2048,
) -> ReasoningGymSFTDataset:
    if isinstance(data_source, Experiment):
        return ReasoningGymSFTDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_length=max_length,
            truncation="error",
        )
    else:
        return ReasoningGymSFTDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_length=max_length,
            truncation="error",
        )

def prepare_reasoning_gym_dataset(config, tokenizer) -> Tuple[ReasoningGymDataset, ReasoningGymDataset]:
    developer_prompt_setting = config.developer_prompt
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS[developer_prompt_setting]
    train_data_specs = [
        DatasetSpec(
            name=name,
            weight=ds.weight,
            config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
        )
        for name, ds in config.train.datasets.items()
    ]
    train_size = config.train.dataset_size
    train_seed = config.train.seed
    train_data_source = reasoning_gym.create_dataset("composite", seed=train_seed, size=train_size, datasets=train_data_specs)
    
    val_data_specs = [
        DatasetSpec(
            name=name,
            weight=ds.weight,
            config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
        )
        for name, ds in config.val.datasets.items()
    ]
    val_size = config.val.dataset_size
    val_seed = config.val.seed
    val_data_source = reasoning_gym.create_dataset("composite", seed=val_seed, size=val_size, datasets=val_data_specs)
    
    train_dataset = make_dataset(tokenizer, train_data_source, developer_prompt, max_prompt_length=config.max_prompt_length)
    val_dataset = make_dataset(tokenizer, val_data_source, developer_prompt, max_prompt_length=config.max_prompt_length)
    return train_dataset, val_dataset

def prepare_reasoning_gym_sft_dataset(config, tokenizer) -> Tuple[ReasoningGymSFTDataset, ReasoningGymSFTDataset]:
    developer_prompt_setting = config.developer_prompt
    developer_prompt = reasoning_gym.utils.SYSTEM_PROMPTS[developer_prompt_setting]
    train_data_specs = [
        DatasetSpec(
            name=name,
            weight=ds.weight,
            config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
        )
        for name, ds in config.train.datasets.items()
    ]
    train_size = config.train.dataset_size
    train_seed = config.train.seed
    train_data_source = reasoning_gym.create_dataset("composite", seed=train_seed, size=train_size, datasets=train_data_specs)
    
    val_data_specs = [
        DatasetSpec(
            name=name,
            weight=ds.weight,
            config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
        )
        for name, ds in config.val.datasets.items()
    ]
    val_size = config.val.dataset_size
    val_seed = config.val.seed
    val_data_source = reasoning_gym.create_dataset("composite", seed=val_seed, size=val_size, datasets=val_data_specs)
    
    train_dataset = make_sft_dataset(tokenizer, train_data_source, developer_prompt, max_length=config.max_length)
    val_dataset = make_sft_dataset(tokenizer, val_data_source, developer_prompt, max_length=config.max_length)
    return train_dataset, val_dataset