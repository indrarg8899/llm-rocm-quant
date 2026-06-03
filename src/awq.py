"""AWQ (Activation-Aware Weight Quantization) implementation.

Lin et al., 2024 — preserves accuracy by protecting salient weight channels
identified from activation magnitudes.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AWQQuantizer:
    """Activation-aware weight quantizer with automatic search."""

    def __init__(self, config):
        self.config = config
        self.bits = config.bits
        self.group_size = getattr(config, "group_size", 128)
        self.n_search = getattr(config, "n_search", 20)
        self.alpha_min = getattr(config, "alpha_min", 0.0)
        self.alpha_max = getattr(config, "alpha_max", 1.0)

    def quantize(self, model: nn.Module, calib_data: list[str]) -> nn.Module:
        """Apply AWQ quantization to model."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.config.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Running AWQ with %d-bit, group_size=%d", self.bits, self.group_size)

        linear_modules = _find_linear_modules(model)

        # Collect activation statistics
        act_stats = {}
        hooks = []
        for name, module in linear_modules:
            stats = _ActivationCollector()
            act_stats[name] = stats
            h = module.register_forward_hook(stats)
            hooks.append(h)

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

        # Quantize with optimal scaling
        for name, module in linear_modules:
            self._quantize_layer(name, module, act_stats.get(name))

        return model

    def _quantize_layer(self, name: str, layer: nn.Linear, stats) -> None:
        """AWQ quantization with per-channel scale search."""
        W = layer.weight.data.float()
        n_channels = W.shape[0]
        qmin, qmax = -(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1

        if stats is None or stats.input_scales is None:
            W_q = self._uniform_quantize(W, qmin, qmax)
            layer.weight.data = W_q.to(layer.weight.dtype)
            return

        # Per-channel input scales
        channel_scales = stats.input_scales.to(W.device).float().clamp(min=1e-6)

        best_scale = 1.0
        best_loss = float("inf")

        # Search for optimal global scale
        for alpha in torch.linspace(self.alpha_min, self.alpha_max, self.n_search):
            s = channel_scales.pow(alpha)
            s = s / s.mean()
            W_scaled = W * s.unsqueeze(1)
            W_q = self._uniform_quantize(W_scaled, qmin, qmax)
            W_dequant = W_q / s.unsqueeze(1)

            # Measure quantization error
            err = (W - W_dequant).pow(2).mean().item()
            if err < best_loss:
                best_loss = err
                best_scale = alpha.item()

        s = channel_scales.pow(best_scale)
        s = s / s.mean()
        W_scaled = W * s.unsqueeze(1)
        W_q = self._uniform_quantize(W_scaled, qmin, qmax)

        layer.weight.data = W_q.to(layer.weight.dtype)
        logger.debug("  AWQ %s: best_alpha=%.3f, loss=%.6f", name, best_scale, best_loss)

    def _uniform_quantize(self, tensor: torch.Tensor, qmin: int, qmax: int) -> torch.Tensor:
        """Group-wise symmetric quantization."""
        groups = tensor.reshape(-1, self.group_size)
        scales = groups.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(groups / scales), qmin, qmax)
        return (q * scales).reshape(tensor.shape)


class _ActivationCollector:
    """Hook that records input activation channel-wise L2 norms."""

    def __init__(self):
        self.input_scales = None
        self._acc = None
        self._count = 0

    def __call__(self, module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.detach().float()
        # Channel-wise L2 norm
        norms = inp.pow(2).mean(dim=[0, 1])  # shape: (in_features,)
        if self._acc is None:
            self._acc = norms
        else:
            self._acc += norms
        self._count += 1
        if self._count == 16:  # average after enough samples
            self.input_scales = (self._acc / self._count).sqrt()


def _find_linear_modules(model: nn.Module, prefix: str = "") -> list[tuple[str, nn.Linear]]:
    results = []
    for name, child in model.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            results.append((full, child))
        else:
            results.extend(_find_linear_modules(child, full))
    return results
