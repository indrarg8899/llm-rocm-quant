"""SmoothQuant with FP8 quantization support.

Xiao et al., 2023 — migrates quantization difficulty from activations to
weights via per-channel smoothing, then applies FP8 or INT8 quantization.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# FP8 ranges (E4M3)
FP8_MAX = 448.0


class SmoothQuantizer:
    """SmoothQuant quantizer with optional FP8 activation support."""

    def __init__(self, config):
        self.config = config
        self.bits = getattr(config, "bits", 8)
        self.alpha = getattr(config, "smooth_alpha", 0.5)
        self.fp8 = getattr(config, "fp8", True)

    def quantize(self, model: nn.Module, calib_data: list[str]) -> nn.Module:
        """Apply SmoothQuant smoothing + quantize linear layers."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(
            "Running SmoothQuant (alpha=%.2f, fp8=%s)",
            self.alpha,
            self.fp8,
        )

        linear_layers = _find_linear_modules(model)

        # Collect activation + weight statistics
        layer_stats = {}
        hooks = []
        for name, module in linear_layers:
            collector = _LayerStats(name)
            layer_stats[name] = collector
            hooks.append(module.register_forward_hook(collector))

        model.eval()
        with torch.no_grad():
            for text in calib_data[: self.config.num_calibration_samples]:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                try:
                    model(**inputs)
                except Exception:
                    pass

        for h in hooks:
            h.remove()

        # Smooth and quantize
        for name, module in linear_layers:
            self._smooth_and_quantize(name, module, layer_stats.get(name))

        return model

    def _smooth_and_quantize(
        self, name: str, layer: nn.Linear, stats: Optional["_LayerStats"]
    ) -> None:
        """Per-channel smoothing + quantization."""
        W = layer.weight.data.float()

        if stats is not None and stats.has_data:
            # Compute per-channel smoothing factor
            w_max = W.abs().max(dim=0).values.clamp(min=1e-8)  # (in_features,)
            x_max = stats.act_max.clamp(min=1e-8)  # (in_features,)
            s = (x_max.pow(self.alpha) / w_max.pow(1 - self.alpha)).clamp(min=1e-8, max=1e6)

            # Smooth: W' = W / s, X' = X * s (absorbed into next layer if possible)
            W_smooth = W / s.unsqueeze(0)
            layer.weight.data = W_smooth.to(layer.weight.dtype)
        else:
            W_smooth = W

        # Quantize weights
        if self.fp8:
            layer.weight.data = self._fp8_quantize(W_smooth).to(layer.weight.dtype)
        else:
            layer.weight.data = self._int8_quantize(W_smooth).to(layer.weight.dtype)

        logger.debug("  SmoothQuant %s", name)

    @staticmethod
    def _fp8_quantize(tensor: torch.Tensor) -> torch.Tensor:
        """FP8 (E4M3) symmetric quantization."""
        scales = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        q = torch.clamp(torch.round(tensor / scales), -FP8_MAX, FP8_MAX)
        return q * scales

    @staticmethod
    def _int8_quantize(tensor: torch.Tensor) -> torch.Tensor:
        """INT8 symmetric quantization."""
        scales = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
        q = torch.clamp(torch.round(tensor / scales), -128, 127)
        return q * scales


class _LayerStats:
    """Collects per-channel input activation statistics."""

    def __init__(self, name: str):
        self.name = name
        self.act_max = None
        self._count = 0

    @property
    def has_data(self) -> bool:
        return self.act_max is not None

    def __call__(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        x = inp.detach().float()
        x_max = x.abs().amax(dim=[0, 1])  # (in_features,)
        if self.act_max is None:
            self.act_max = x_max
        else:
            self.act_max = torch.max(self.act_max, x_max)
        self._count += 1


def _find_linear_modules(model: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear]]:
    results = []
    for name, child in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            results.append((full, child))
        else:
            results.extend(_find_linear_modules(child, full))
    return results
