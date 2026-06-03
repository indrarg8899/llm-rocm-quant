"""Model format conversion utilities.

Supports converting quantized models to GGUF and ONNX formats for
deployment on various runtimes.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def convert_to_gguf(
    model_path: str,
    output_path: Optional[str] = None,
    quant_type: str = "q4_k_m",
    tokenizer_path: Optional[str] = None,
) -> Path:
    """Convert a HF model directory to GGUF format using llama.cpp tools.

    Args:
        model_path: Path to quantized HF model directory.
        output_path: Output GGUF file path.
        quant_type: GGUF quantization type (q4_0, q4_k_m, q5_k_m, etc).
        tokenizer_path: Optional separate tokenizer path.

    Returns:
        Path to generated GGUF file.
    """
    src = Path(model_path)
    if output_path is None:
        output_path = str(src / f"{src.name}.gguf")
    out = Path(output_path)

    logger.info("Converting %s → GGUF (%s)", src, quant_type)

    # Try llama.cpp convert script
    convert_script = _find_llama_cpp_convert()
    if convert_script:
        cmd = [
            sys.executable, str(convert_script),
            "--outfile", str(out),
            "--outtype", quant_type,
            str(src),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"GGUF conversion failed:\n{result.stderr}")
        logger.info("GGUF saved to %s", out)
        return out

    raise FileNotFoundError(
        "llama.cpp convert script not found. Install llama.cpp and set "
        "LLAMA_CPP_DIR environment variable."
    )


def convert_to_onnx(
    model: nn.Module,
    tokenizer,
    output_path: str,
    opset: int = 17,
    input_shape: tuple = (1, 512),
    dynamic_axes: bool = True,
) -> Path:
    """Export model to ONNX format.

    Args:
        model: PyTorch model.
        tokenizer: Corresponding tokenizer.
        output_path: Output ONNX file path.
        opset: ONNX opset version.
        input_shape: Fixed input shape for tracing.
        dynamic_axes: Enable dynamic batch/seq_len axes.

    Returns:
        Path to generated ONNX file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting to ONNX: opset=%d, dynamic=%s", opset, dynamic_axes)

    model.eval()
    device = next(model.parameters()).device

    dummy_input = torch.zeros(*input_shape, dtype=torch.long, device=device)

    axes = {}
    if dynamic_axes:
        axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
        }

    torch.onnx.export(
        model,
        (dummy_input, torch.ones_like(dummy_input)),
        str(out),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=axes if dynamic_axes else None,
    )

    # Validate
    import onnx
    onnx_model = onnx.load(str(out))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model saved to %s", out)
    return out


def convert_to_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 512),
) -> Path:
    """Export to TorchScript for production deployment."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros(*input_shape, dtype=torch.long, device=device)

    traced = torch.jit.trace(model, (example, torch.ones_like(example)))
    traced.save(str(out))
    logger.info("TorchScript model saved to %s", out)
    return out


def _find_llama_cpp_convert() -> Optional[Path]:
    """Locate llama.cpp conversion script."""
    # Check env
    env_path = os.environ.get("LLAMA_CPP_DIR")
    if env_path:
        script = Path(env_path) / "convert_hf_to_gguf.py"
        if script.exists():
            return script

    # Check common paths
    for candidate in [
        Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path("/usr/local/bin/convert_hf_to_gguf.py"),
    ]:
        if candidate.exists():
            return candidate

    return None
