#!/usr/bin/env python3
"""
Merge a LoRA adapter with a base model for use with vLLM.

vLLM doesn't natively support LoRA adapters, so we need to merge them first.
This script merges a PEFT LoRA adapter with its base model and saves the merged model.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig


def merge_lora_adapter(base_model_path: str, lora_adapter_path: str, output_path: str):
    """
    Merge a LoRA adapter with its base model.

    Args:
        base_model_path: Path to the base model
        lora_adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
    """
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    # Load the LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        lora_adapter_path,
        torch_dtype=torch.float16,
    )

    print("Merging LoRA adapter with base model...")
    # Merge the adapter with the base model
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    # Save the merged model
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # Also save the tokenizer (copy from base model or adapter if it has one)
    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"✓ Successfully merged and saved model to: {output_path}")
    print(f"  You can now use this merged model with vLLM")


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter with a base model for vLLM"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Path to the base model (e.g., models/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        required=True,
        help="Path to the LoRA adapter (e.g., models/Qwen2.5-3B-OpenThoughts-LoRA)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the merged model (e.g., models/Qwen2.5-3B-OpenThoughts-Merged)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for the merged model (default: float16)"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    # Verify paths exist
    if not os.path.exists(args.base_model):
        raise FileNotFoundError(f"Base model path does not exist: {args.base_model}")
    if not os.path.exists(args.lora_adapter):
        raise FileNotFoundError(f"LoRA adapter path does not exist: {args.lora_adapter}")

    merge_lora_adapter(args.base_model, args.lora_adapter, args.output)


if __name__ == "__main__":
    main()

