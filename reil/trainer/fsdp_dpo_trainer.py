import hydra
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

@hydra.main(version_base=None, config_path="config", config_name="dpo_trainer")
def main(config: DictConfig):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    
    # Load dataset
    dataset = load_dataset(config.dataset.name, split="train")
    
    # Configure DPO training
    training_args = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_grad_norm=config.max_grad_norm,
        beta=config.beta,
    )
    
    # Initialize trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()