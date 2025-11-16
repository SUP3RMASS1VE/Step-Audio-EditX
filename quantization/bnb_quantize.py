#!/usr/bin/env python3
"""
BitsAndBytes Quantization Script
Simple 4-bit/8-bit quantization using bitsandbytes (most compatible with custom models)

Usage:
    python bnb_quantize.py --model_path /path/to/model --bits 4
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def validate_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    return path


def quantize_and_save_model(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    compute_dtype: str = "float16",
):
    """
    Load model with bitsandbytes quantization and save it
    """
    
    print(f"\n{'='*60}")
    print(f"BitsAndBytes {bits}-bit Quantization")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Bits: {bits}")
    print(f"Compute dtype: {compute_dtype}")
    print(f"{'='*60}\n")
    
    # Configure quantization
    if bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif bits == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )
    else:
        raise ValueError(f"Unsupported bits: {bits}. Use 4 or 8.")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    
    print(f"\nLoading model with {bits}-bit quantization...")
    print("This may take a few minutes...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=getattr(torch, compute_dtype),
        )
        
        print(f"\nModel loaded successfully with {bits}-bit quantization!")
        
        # Save the quantized model
        print(f"\nSaving quantized model to: {output_dir}")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
        
        # Save quantization config
        import json
        config_path = Path(output_dir) / "quantization_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "bits": bits,
                "compute_dtype": compute_dtype,
                "quantization_method": "bitsandbytes",
                "load_in_4bit": bits == 4,
                "load_in_8bit": bits == 8,
                "bnb_4bit_compute_dtype": compute_dtype if bits == 4 else None,
                "bnb_4bit_use_double_quant": True if bits == 4 else None,
                "bnb_4bit_quant_type": "nf4" if bits == 4 else None,
            }, f, indent=2)
        
        print("\n" + "="*60)
        print("Quantization completed successfully!")
        print(f"Quantized model saved to: {output_dir}")
        print("\nTo load this model:")
        print(f"  from transformers import AutoModelForCausalLM, BitsAndBytesConfig")
        print(f"  config = BitsAndBytesConfig(load_in_{bits}bit=True)")
        print(f"  model = AutoModelForCausalLM.from_pretrained(")
        print(f"      '{output_dir}',")
        print(f"      quantization_config=config,")
        print(f"      trust_remote_code=True,")
        print(f"      device_map='auto'")
        print(f"  )")
        print("="*60)
        
        # Print memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"  Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
    except Exception as e:
        print(f"\nERROR during quantization: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="BitsAndBytes Quantization")
    
    parser.add_argument("--model_path", "-m", type=str, required=True, help="Path to model")
    parser.add_argument("--output_suffix", "-o", type=str, default=None, help="Output directory suffix (default: bnb-{bits}bit)")
    parser.add_argument("--bits", "-b", type=int, default=4, choices=[4, 8], help="Quantization bits (4 or 8)")
    parser.add_argument("--compute_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Compute dtype")
    
    args = parser.parse_args()
    
    try:
        model_path = validate_model_path(args.model_path)
        
        # Set default output suffix if not provided
        output_suffix = args.output_suffix or f"bnb-{args.bits}bit"
        output_dir = model_path / output_suffix
        output_dir.mkdir(exist_ok=True)
        
        quantize_and_save_model(
            model_path=str(model_path),
            output_dir=str(output_dir),
            bits=args.bits,
            compute_dtype=args.compute_dtype,
        )
        
    except Exception as e:
        print(f"\nExecution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
