"""Inference throughput benchmark for quantized models.

Measures tokens/sec, latency, and memory usage across batch sizes on ROCm/MI300X.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark results for a single configuration."""
    model_name: str
    method: str
    bits: int
    batch_size: int
    prompt_len: int
    gen_len: int
    tokens_per_sec: float
    latency_ms: float
    memory_peak_gb: float
    memory_allocated_gb: float
    throughput_samples_per_sec: float
    results: list[dict] = field(default_factory=list)


def benchmark_generation(
    model: nn.Module,
    tokenizer,
    method: str = "unknown",
    bits: int = 16,
    prompts: Optional[list[str]] = None,
    batch_sizes: Optional[list[int]] = None,
    gen_lengths: Optional[list[int]] = None,
    warmup: int = 3,
    device: Optional[str] = None,
) -> list[BenchmarkResult]:
    """Run generation benchmark across batch sizes and lengths.

    Args:
        model: Model to benchmark.
        tokenizer: Corresponding tokenizer.
        method: Quantization method name.
        bits: Bit-width used.
        prompts: List of prompt strings. Default: synthetic prompts.
        batch_sizes: Batch sizes to test.
        gen_lengths: Generation lengths to test.
        warmup: Number of warmup iterations.
        device: Target device.

    Returns:
        List of BenchmarkResult for each (batch_size, gen_length) combo.
    """
    if prompts is None:
        prompts = [
            "Explain quantum computing in simple terms:",
            "Write a Python function to sort a list:",
            "What are the key differences between ROCm and CUDA?",
            "Describe the architecture of a transformer model:",
            "Summarize the history of artificial intelligence:",
        ]
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    if gen_lengths is None:
        gen_lengths = [128, 512, 1024]
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for bs in batch_sizes:
        for gl in gen_lengths:
            # Prepare inputs
            texts = (prompts * ((bs // len(prompts)) + 1))[:bs]
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Warmup
            logger.info("Benchmark: bs=%d, gen_len=%d (warmup=%d)", bs, gl, warmup)
            for _ in range(warmup):
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=min(gl, 32))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)

            # Measure
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=gl, do_sample=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
            total_new = new_tokens * bs
            tps = total_new / elapsed

            mem_peak = 0.0
            mem_alloc = 0.0
            if torch.cuda.is_available():
                mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)

            r = BenchmarkResult(
                model_name=tokenizer.name_or_path,
                method=method,
                bits=bits,
                batch_size=bs,
                prompt_len=inputs["input_ids"].shape[1],
                gen_len=gl,
                tokens_per_sec=round(tps, 1),
                latency_ms=round((elapsed / bs) * 1000, 1),
                memory_peak_gb=round(mem_peak, 2),
                memory_allocated_gb=round(mem_alloc, 2),
                throughput_samples_per_sec=round(bs / elapsed, 2),
            )
            results.append(r)
            logger.info(
                "  bs=%d gl=%d → %.1f tok/s, %.1fms latency, %.2f GB peak",
                bs, gl, tps, r.latency_ms, mem_peak,
            )

    return results


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Format results as a readable table."""
    header = f"{'BS':>4} {'GenLen':>6} {'tok/s':>8} {'Latency':>10} {'Peak GB':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.batch_size:>4} {r.gen_len:>6} {r.tokens_per_sec:>8.1f} "
            f"{r.latency_ms:>8.1f}ms {r.memory_peak_gb:>8.2f}"
        )
    return "\n".join(lines)
