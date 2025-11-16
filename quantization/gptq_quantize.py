#!/usr/bin/env python3
"""
GPTQ Quantization Script
Alternative quantization using AutoGPTQ which is more compatible with custom architectures

Usage:
    python gptq_quantize.py --model_path /path/to/model
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset


def validate_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    return path


def prepare_calibration_data(dataset_name: str, tokenizer, n_samples: int = 128, max_length: int = 2048):
    """Prepare calibration dataset for GPTQ"""
    print(f"Loading calibration dataset: {dataset_name}")
    
    if dataset_name == "open_platypus":
        dataset = load_dataset("garage-bAInd/Open-Platypus", split="train")
        text_column = "instruction"
    elif dataset_name == "c4":
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_column = "text"
    else:
        dataset = load_dataset(dataset_name, split="train")
        text_column = list(dataset.features.keys())[0]
    
    examples = []
    for i, example in enumerate(dataset):
        if i >= n_samples:
            break
        text = example[text_column] if isinstance(example[text_column], str) else str(example[text_column])
        examples.append(text)
    
    print(f"Prepared {len(examples)} calibration samples")
    return examples


def quantize_model(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    dataset: str = "open_platypus",
    n_samples: int = 128,
    max_length: int = 2048,
):
    """Quantize model using GPTQ"""
    
    print(f"\n{'='*60}")
    print(f"GPTQ Quantization")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Bits: {bits}, Group Size: {group_size}")
    print(f"{'='*60}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare calibration data
    calibration_data = prepare_calibration_data(dataset, tokenizer, n_samples, max_length)
    
    # Configure quantization
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,  # Disable for better compatibility
        sym=True,
        true_sequential=True,
    )
    
    print(f"\nLoading model for quantization...")
    print("This may take a few minutes...")
    
    try:
        # Load model with GPTQ
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=quantize_config,
            trust_remote_code=True,
            device_map="auto",
        )
        
        print("\nRunning GPTQ quantization...")
        print("This will take some time depending on model size...")
        
        # Quantize
        model.quantize(
            calibration_data,
            batch_size=1,
        )
        
        # Save quantized model
        print(f"\nSaving quantized model to: {output_dir}")
        model.save_quantized(output_dir, use_safetensors=True)
        tokenizer.save_pretrained(output_dir)
        
        print("\n" + "="*60)
        print("Quantization completed successfully!")
        print(f"Quantized model saved to: {output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR during quantization: {e}")
        print("\nTrying alternative approach with manual model loading...")
        
        # Fallback: Load with transformers first
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Try to wrap with GPTQ
        from auto_gptq.modeling._base import BaseGPTQForCausalLM
        
        print("Attempting manual GPTQ wrapping...")
        raise NotImplementedError("Manual GPTQ wrapping not yet implemented for custom architectures")


def main():
    parser = argparse.ArgumentParser(description="GPTQ Quantization for custom models")
    
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to model")
    parser.add_argument("--output_suffix", "-o", type=str, default="gptq-4bit", help="Output directory suffix")
    parser.add_argument("--bits", "-b", type=int, default=4, choices=[2, 3, 4, 8], help="Quantization bits")
    parser.add_argument("--group_size", "-g", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--dataset", "-d", type=str, default="open_platypus", help="Calibration dataset")
    parser.add_argument("--n_samples", "-n", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length")
    
    args = parser.parse_args()
    
    try:
        model_path = validate_model_path(args.model_path)
        output_dir = model_path / args.output_suffix
        output_dir.mkdir(exist_ok=True)
        
        quantize_model(
            model_path=str(model_path),
            output_dir=str(output_dir),
            bits=args.bits,
            group_size=args.group_size,
            dataset=args.dataset,
            n_samples=args.n_samples,
            max_length=args.max_length,
        )
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
