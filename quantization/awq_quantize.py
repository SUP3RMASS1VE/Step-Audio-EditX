#!/usr/bin/env python3
"""
AWQ Quantization Script
Performs AWQ quantization processing using llmcompressor

Usage:
    python awq_quantize.py --model_path /path/to/model [other options]

"""

import argparse
import sys
import time
import os
from pathlib import Path
from typing import Optional, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor import oneshot

# Environment settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TORCH_FX_DISABLE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["HF_HUB_DISABLE_CUSTOM_CODE"] = "0"   # ← allow custom model code


# ---------------------------------------------------------------------
# PATH VALIDATION
# ---------------------------------------------------------------------

def validate_model_path(model_path: str) -> Path:
    path = Path(model_path)

    if not path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    expected_files = ['config.json']
    model_files = ['pytorch_model.bin', 'model.safetensors']

    has_config = any((path / f).exists() for f in expected_files)
    has_model = any((path / f).exists() for f in model_files) or any(
        list(path.glob('pytorch_model-*.bin')) or list(path.glob('model-*.safetensors'))
    )

    if not (has_config and has_model):
        print(f"WARNING: Model path may be incomplete. Found: {list(path.iterdir())[:10]}")

    return path


# ---------------------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------------------

def create_output_directory(model_path: Path, suffix: str = "awq-4bit") -> Path:
    output_dir = model_path / suffix
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir


# ---------------------------------------------------------------------
# AWQ RECIPE BUILDER
# ---------------------------------------------------------------------

def get_awq_recipe(
    scheme: str,
    group_size: int = 128,
    ignore_layers: Optional[list] = None
) -> list:

    if ignore_layers is None:
        ignore_layers = [
            "lm_head",
            "embed_tokens",
            "model.embed_tokens",
            "model.norm",
            "norm",
            "output",
            "classifier"
        ]

    if not scheme:
        awq_modifier = AWQModifier(
            offload_device=torch.device("cpu"),
            scheme=None,
            ignore=ignore_layers,
            targets=["Linear"],
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "symmetric": True,
                        "strategy": "group",
                        "group_size": group_size
                    }
                }
            }
        )
    else:
        awq_modifier = AWQModifier(
            scheme=scheme,
            ignore=ignore_layers,
            targets=["Linear"]
        )

    return [
        SmoothQuantModifier(smoothing_strength=0.5),
        awq_modifier,
    ]


# ---------------------------------------------------------------------
# MAIN QUANTIZATION FUNCTION
# ---------------------------------------------------------------------

def quantize_model(
    model_path: str,
    output_dir: str,
    scheme: str,
    dataset: str = "open_platypus",
    max_seq_length: int = 2048,
    num_calibration_samples: int = 512,
    group_size: int = 128,
    ignore_layers: Optional[list] = None,
    device: Optional[str] = None
) -> None:

    # ---------------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------------

    print(f"Preparing calibration dataset: {dataset}")

    try:
        if '/' in dataset:
            if dataset.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset, split='train')
            elif dataset.endswith('.csv'):
                dataset = load_dataset('csv', data_files=dataset, split='train')
            elif dataset.endswith('.parquet'):
                dataset = load_dataset('parquet', data_files=dataset, split='train')
            else:
                dataset = load_dataset(dataset, split='train')

            print(f"Loaded dataset with {len(dataset)} samples")
        else:
            print(f"Dataset '{dataset}' does not contain '/', assuming HuggingFace dataset name")
    except Exception as e:
        print(f"Dataset load failed: {e}")
        print("Falling back to using dataset string")
        pass

    # ---------------------------------------------------------------
    # DEVICE SETUP
    # ---------------------------------------------------------------

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ---------------------------------------------------------------
    # CREATE RECIPE
    # ---------------------------------------------------------------

    recipe = get_awq_recipe(scheme, group_size, ignore_layers)

    # ---------------------------------------------------------------
    # LOAD MODEL WITH CUSTOM CODE ENABLED
    # ---------------------------------------------------------------

    print("\n=== Loading model with trust_remote_code=True ===")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Model successfully loaded.\n")

    # ---------------------------------------------------------------
    # RUN ONE-SHOT QUANTIZATION
    # ---------------------------------------------------------------

    try:
        print("Running AWQ quantization…")

        start = time.time()

        oneshot(
            model=model,                   # IMPORTANT: pass model object, NOT path
            tokenizer=tokenizer,
            dataset=dataset,
            recipe=recipe,
            output_dir=output_dir,
            max_seq_length=max_seq_length,
            num_calibration_samples=num_calibration_samples,
            precision="auto",
            clear_sparse_session=True,
            save_compressed=True,
        )

        end = time.time()
        print(f"\nQuantization completed in {end - start:.2f} seconds!")
        print(f"Saved quantized model to: {output_dir}")

    except Exception as e:
        print(f"\nERROR during quantization: {e}")
        raise


# ---------------------------------------------------------------------
# MEMORY ESTIMATION
# ---------------------------------------------------------------------

def estimate_memory_requirements(model_path: str) -> Dict[str, float]:
    try:
        path = Path(model_path)
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        size_gb = size / (1024**3)

        return {
            "model_size_gb": size_gb,
            "estimated_peak_memory_gb": size_gb * 2.5,
            "quantized_size_gb": size_gb * 0.3,
        }

    except Exception as e:
        print(f"Memory estimation failed: {e}")
        return {}


# ---------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", "-m", type=str, required=True)
    parser.add_argument("--output_suffix", "-o", type=str, default="awq-4bit")
    parser.add_argument("--dataset", "-d", type=str, default="open_platypus")
    parser.add_argument("--scheme", "-s", type=str, default="", choices=["W4A16_ASYM", "W4A16_SYM", "W8A16"])
    parser.add_argument("--group_size", "-g", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_calibration_samples", type=int, default=512)
    parser.add_argument("--ignore_layers", type=str, nargs="+", default=["lm_head"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    try:
        model_path = validate_model_path(args.model_path)
        output_dir = create_output_directory(model_path, args.output_suffix)

        memory = estimate_memory_requirements(str(model_path))
        print("Memory estimation:", memory)

        if args.dry_run:
            print("Dry run complete.")
            return

        quantize_model(
            model_path=str(model_path),
            output_dir=str(output_dir),
            scheme=args.scheme,
            dataset=args.dataset,
            max_seq_length=args.max_seq_length,
            num_calibration_samples=args.num_calibration_samples,
            group_size=args.group_size,
            ignore_layers=args.ignore_layers,
            device=None if args.device == "auto" else args.device
        )

    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
