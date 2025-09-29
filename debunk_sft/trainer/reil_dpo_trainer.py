import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from typing import Union
from datasets import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin
from trl.trainer.dpo_trainer import maybe_extract_prompt, maybe_apply_chat_template
from trl.trainer.dpo_trainer import PartialState
from typing import Optional, Union, Callable
from torch import nn
from transformers import PreTrainedModel
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader
# from torch.utils.data.dataloader import default_collate
from trl.trainer.dpo_trainer import EvalLoopOutput
from trl.trainer.dpo_trainer import TrainerCallback
from trl.trainer.dpo_trainer import DataCollator
import torch
from debunk_sft.trainer.llm_agent.agent_proxy import LLMAgentProxy
from debunk_sft.trainer.llm_agent.agent_proxy import HFWrapperWg
import time

class ReilDPOTrainer(DPOTrainer):
    def __init__(self,         
                 model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
                args: Optional[DPOConfig] = None,
                data_collator: Optional[DataCollator] = None,
                train_dataset: Optional[Dataset] = None,
                eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
                processing_class: Optional[
                    Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
                ] = None,
                model_init: Optional[Callable[[], PreTrainedModel]] = None,
                compute_metrics: Optional[Callable[[EvalLoopOutput], dict]] = None,
                callbacks: Optional[list[TrainerCallback]] = None,
                optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
                peft_config: Optional[dict] = None,
                eval_config: Optional[dict] = None):

        super().__init__(model, ref_model, args, data_collator, train_dataset, eval_dataset, processing_class, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics, peft_config)
        self.policy_eval = eval_config.evaluator.policy_eval
        self.actor_wg = HFWrapperWg(eval_config, processing_class, self.model)
        self.proxy = LLMAgentProxy(eval_config, self.actor_wg, processing_class)
        # Add this before training

                
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        processing_class: Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin],
        args: DPOConfig,
        dataset_name: str,
    ) -> Union[Dataset, IterableDataset]:
        # Build the kwargs for the `map` function
        map_kwargs = {"writer_batch_size": 10}
        if isinstance(dataset, Dataset):  # IterableDataset does not support num_proc
            map_kwargs["num_proc"] = args.dataset_num_proc

        with PartialState().main_process_first():
            # Extract prompt if needed
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Extracting prompt in {dataset_name} dataset"
            dataset = dataset.map(maybe_extract_prompt, **map_kwargs)

            # Apply the chat template if needed (TODO: not needed for REIL for now)
            # if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
            #     map_kwargs["desc"] = f"Applying chat template to {dataset_name} dataset"
            # dataset = dataset.map(
            #     maybe_apply_chat_template, fn_kwargs={"tokenizer": processing_class, "tools": args.tools}, **map_kwargs
            # )

            # Tokenize the dataset
            if isinstance(dataset, Dataset):  # `IterableDataset.map` does not support `desc`
                map_kwargs["desc"] = f"Tokenizing {dataset_name} dataset"

            dataset = dataset.map(
                self.tokenize_row if not self.is_vision_model else self.process_row,
                remove_columns=["chosen", "rejected"],
                fn_kwargs={
                    "processing_class": processing_class,
                    "max_prompt_length": args.max_prompt_length,
                    "max_completion_length": args.max_completion_length,
                    # for enc-dec, we add the special tokens ([bos_token] + prompt + [eos_token]; completion + [eos_token])
                    "add_special_tokens": False,
                },
                **map_kwargs,
            )

        return dataset
    
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None
    ):
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate)
        # args = self.args
        # model = self._wrap_model(self.model, training=False, dataloader=None)

        # if len(self.accelerator._models) == 0 and model is self.model:
        #     start_time = time.time()
        #     model = (
        #         self.accelerator.prepare(model)
        #         if self.is_deepspeed_enabled or (self.is_fsdp_enabled and self.accelerator.mixed_precision != "fp8")
        #         else self.accelerator.prepare_model(model, evaluation_mode=True)
        #     )
        #     self.model_preparation_time = round(time.time() - start_time, 4)

        #     if self.is_fsdp_enabled:
        #         self.model = model

        #     # for the rest of this function `model` is the outside model, whether it was wrapped or not
        #     if model is not self.model:
        #         self.model_wrapped = model

        #     # backward compatibility
        #     if self.is_deepspeed_enabled:
        #         self.deepspeed = self.model_wrapped

        # # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # # while ``train`` is running, cast it to the right dtype first and then put on device
        # if not self.is_in_train:
        #     if args.fp16_full_eval:
        #         model = model.to(dtype=torch.float16, device=args.device)
        #     elif args.bf16_full_eval:
        #         model = model.to(dtype=torch.bfloat16, device=args.device)
        
        if self.policy_eval and self.state.global_step % self.args.eval_steps == 0:  
            rollouts = self.proxy.rollout()
            self.log(rollouts.meta_info['metrics'], start_time=None)


@hydra.main(version_base=None, config_path="config", config_name="dpo_trainer")
def main(config: DictConfig):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.actor_rollout_ref.model.path)
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    # print("Model config:", model.config)
    # print("Embedding shape:", model.model.embed_tokens.weight.shape)
    # print(tokenizer.vocab_size)
    # Load dataset
    dataset = load_dataset(config.dataset.name, split="train")
    
    # Configure DPO training
    training_args = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.dpo.learning_rate,
        per_device_train_batch_size=config.dpo.batch_size,
        num_train_epochs=config.dpo.num_epochs,
        gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
        max_grad_norm=config.dpo.max_grad_norm,
        beta=config.dpo.beta,
        logging_steps=config.dpo.logging_steps,
        save_steps=config.dpo.save_steps,
        eval_steps=config.dpo.eval_steps,
        # eval_strategy="steps",
        run_name=config.evaluator.experiment_name,
        # apply_chat_template=config.dataset.apply_chat_template,
    )

    
    
    # Initialize trainer
    trainer = ReilDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        eval_config=config
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()