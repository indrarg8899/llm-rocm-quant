#!/usr/bin/env python3
"""CLI script to evaluate perplexity of a quantized model."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval import evaluate_perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate model perplexity on WikiText-2")
    parser.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path (default: same as model)")
    parser.add_argument("--dataset", default="wikitext", help="Dataset name")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model: {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    result = evaluate_perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        stride=args.stride,
        seq_len=args.seq_len,
        device=device,
    )

    print(f"\n{'='*40}")
    print(f"Model:           {result.model_name}")
    print(f"Perplexity:      {result.perplexity:.4f}")
    print(f"Eval time:       {result.eval_time_sec:.1f}s")
    print(f"Tokens/sec:      {result.tokens_per_sec:.1f}")
    print(f"Total tokens:    {result.total_tokens:,}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
