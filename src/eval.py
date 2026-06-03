"""Perplexity evaluation on WikiText-2.

Measures language modeling quality of quantized models against FP16 baselines.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result container."""
    model_name: str
    method: str
    bits: int
    perplexity: float
    perplexity_delta: float  # vs FP16 baseline
    eval_time_sec: float
    tokens_per_sec: float
    total_tokens: int


def evaluate_perplexity(
    model: nn.Module,
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    stride: int = 512,
    seq_len: int = 2048,
    device: Optional[str] = None,
) -> EvalResult:
    """Compute perplexity on WikiText-2.

    Args:
        model: Loaded model (quantized or FP16).
        tokenizer: Corresponding tokenizer.
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration.
        split: Dataset split.
        stride: Stride for sliding window.
        seq_len: Sequence length for evaluation.
        device: Target device.

    Returns:
        EvalResult with perplexity and metadata.
    """
    if device is None:
        device = next(model.parameters()).device

    logger.info("Loading %s/%s split=%s", dataset_name, dataset_config, split)
    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split)
    encodings = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    total_tokens = input_ids.numel()

    logger.info("Evaluating %d tokens with stride=%d, seq_len=%d", total_tokens, stride, seq_len)

    model.eval()
    nlls = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for begin_loc in range(0, total_tokens, stride):
            end_loc = min(begin_loc + seq_len, total_tokens)
            trg_len = end_loc - begin_loc if end_loc < total_tokens else seq_len
            input_chunk = input_ids[:, begin_loc:end_loc]

            target_ids = input_chunk.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
            nlls.append(neg_log_likelihood.item())

    eval_time = time.perf_counter() - t0
    avg_nll = sum(nlls) / len(nlls) if nlls else float("inf")
    ppl = math.exp(avg_nll)
    tps = total_tokens / eval_time if eval_time > 0 else 0

    return EvalResult(
        model_name=tokenizer.name_or_path,
        method="eval",
        bits=16,
        perplexity=round(ppl, 4),
        perplexity_delta=0.0,
        eval_time_sec=round(eval_time, 2),
        tokens_per_sec=round(tps, 1),
        total_tokens=total_tokens,
    )


def compare_perplexity(
    fp16_result: EvalResult,
    quantized_result: EvalResult,
) -> str:
    """Format comparison report."""
    delta = quantized_result.perplexity - fp16_result.perplexity
    pct = (delta / fp16_result.perplexity) * 100

    report = [
        "=== Perplexity Comparison ===",
        f"  FP16 baseline:  {fp16_result.perplexity:.4f}",
        f"  Quantized:      {quantized_result.perplexity:.4f}",
        f"  Delta:          +{delta:.4f} ({pct:+.2f}%)",
        f"  FP16 speed:     {fp16_result.tokens_per_sec:.1f} tok/s",
        f"  Quantized speed:{quantized_result.tokens_per_sec:.1f} tok/s",
        f"  Speedup:        {quantized_result.tokens_per_sec / fp16_result.tokens_per_sec:.2f}x",
    ]
    return "\n".join(report)
