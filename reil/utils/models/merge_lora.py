"""
Used to merge the LoRA model with the base model to get a full model.
Adapted from ragen/sft/utils/merge_lora.py
"""


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse
import os

def merge_lora_model(
    base_model_name: str,
    lora_model_path: str,
    output_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Load a LoRA model, merge it with the base model, and save the merged model.
    
    Args:
        base_model_name (str): Hugging Face model name or path to base model
        lora_model_path (str): Path to the LoRA model
        output_path (str): Path to save the merged model
        device (str): Device to load the model on ("cuda" or "cpu")
    """
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    print(f"Loading LoRA adapter: {lora_model_path}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    print("Merging LoRA parameters with base model...")
    model = model.merge_and_unload()

    
    print(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)
    
    # Save tokenizer if it exists with the LoRA model
    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_model_path)
        tokenizer.save_pretrained(output_path)
        print("Tokenizer saved successfully")
    except Exception as e:
        print(f"No tokenizer found in LoRA path, attempting to save base model tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.save_pretrained(output_path)
            print("Base model tokenizer saved successfully")
        except Exception as e:
            print(f"Warning: Could not save tokenizer: {str(e)}")
    
    print("Merge completed successfully!")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--lora_model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=False)
    merge_lora_model(args.base_model_name, args.lora_model_path, args.output_path)
