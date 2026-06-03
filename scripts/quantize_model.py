#!/usr/bin/env python3
"""CLI script to quantize a model using any supported method."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import QuantConfig
from src.quantize import quantize_model, QUANT_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description="Quantize an LLM for AMD MI300X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPTQ 4-bit
  python scripts/quantize_model.py --config configs/llama3-8b-gptq.yml

  # AWQ 4-bit
  python scripts/quantize_model.py --config configs/mixtral-awq.yml

  # CLI args
  python scripts/quantize_model.py --model meta-llama/Meta-Llama-3-8B --method gptq --bits 4
        """,
    )
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="HuggingFace model ID or path")
    parser.add_argument("--method", choices=QUANT_REGISTRY.keys(), default="gptq")
    parser.add_argument("--bits", type=int, default=4, choices=[3, 4, 8])
    parser.add_argument("--output", type=str, default="./quantized", help="Output directory")
    parser.add_argument("--calibration-data", type=str, help="Path to calibration text file")
    parser.add_argument("--num-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build config
    if args.config:
        config = QuantConfig.from_yaml(args.config)
    elif args.model:
        config = QuantConfig(
            model=args.model,
            method=args.method,
            bits=args.bits,
            output_dir=args.output,
            calibration_data=args.calibration_data,
            num_calibration_samples=args.num_samples,
            gpu_id=args.gpu_id,
            verbose=args.verbose,
        )
    else:
        parser.error("Either --config or --model is required")

    # Run
    output_path = quantize_model(config, output_dir=args.output if not args.config else None)
    print(f"\n✅ Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
