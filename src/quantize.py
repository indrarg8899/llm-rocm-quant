"""Main quantization orchestrator.

Supports GPTQ, AWQ, and SmoothQuant methods with ROCm/MI300X optimizations.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

from .config import QuantConfig
from .gptq import GPTQQuantizer
from .awq import AWQQuantizer
from .smoothquant import SmoothQuantizer

logger = logging.getLogger(__name__)

QUANT_REGISTRY: dict[str, type] = {
    "gptq": GPTQQuantizer,
    "awq": AWQQuantizer,
    "smoothquant": SmoothQuantizer,
}


def quantize_model(
    config: QuantConfig,
    output_dir: Optional[str] = None,
) -> Path:
    """Run quantization pipeline end-to-end.

    Args:
        config: Validated quantization configuration.
        output_dir: Override output path from config.

    Returns:
        Path to saved quantized model.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(
        "Starting quantization: method=%s model=%s bits=%d",
        config.method,
        config.model,
        config.bits,
    )
    t0 = time.perf_counter()

    # --- Load model --------------------------------------------------------
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # --- Get calibration data ----------------------------------------------
    calib_data = _load_calibration(config)

    # --- Quantize ----------------------------------------------------------
    quantizer_cls = QUANT_REGISTRY[config.method]
    quantizer = quantizer_cls(config)
    model = quantizer.quantize(model, calib_data)

    # --- Save --------------------------------------------------------------
    dest = Path(output_dir or config.output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dest, safe_serialization=True)
    tokenizer.save_pretrained(dest)

    elapsed = time.perf_counter() - t0
    logger.info("Quantized model saved to %s in %.1fs", dest, elapsed)
    return dest


def _load_calibration(config: QuantConfig) -> list[str]:
    """Load or download calibration dataset."""
    if config.calibration_data:
        path = Path(config.calibration_data)
        if path.exists():
            with open(path) as f:
                return [line.strip() for line in f if line.strip()]

    # Fallback: WikiText-2 subset
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
        texts = [t for t in ds["text"] if len(t.strip()) > 20]
        return texts[: config.num_calibration_samples]
    except Exception as exc:
        logger.warning("Failed to load WikiText-2 (%s), using dummy calib", exc)
        return [
            "The quick brown fox jumps over the lazy dog. " * 10
            for _ in range(config.num_calibration_samples)
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize an LLM")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--method", choices=QUANT_REGISTRY.keys(), default="gptq")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--output", default="./quantized")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = QuantConfig(model=args.model, method=args.method, bits=args.bits, output_dir=args.output)
    quantize_model(cfg)
