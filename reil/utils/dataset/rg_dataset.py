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