"""Quantization configuration dataclasses."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class QuantConfig:
    """Unified configuration for all quantization methods.

    Attributes:
        model: HuggingFace model ID or local path.
        method: Quantization method (gptq, awq, smoothquant).
        bits: Bit-width (3, 4, 8).
        group_size: Quantization group size (default 128).
        output_dir: Output directory for quantized model.
        calibration_data: Path to calibration text file.
        num_calibration_samples: Number of calibration samples.
        gpu_id: GPU device ID.
        smooth_alpha: SmoothQuant smoothing parameter (0.0-1.0).
        fp8: Enable FP8 activations (SmoothQuant only).
        act_order: Enable activation ordering (GPTQ only).
        percdamp: Hessian dampening percentage (GPTQ only).
        blocksize: Block size for GPTQ updates.
        n_search: Number of alpha search points (AWQ only).
        alpha_min: Minimum search alpha (AWQ only).
        alpha_max: Maximum search alpha (AWQ only).
        num_gpus: Number of GPUs for tensor parallel.
        verbose: Enable verbose logging.
    """
    model: str
    method: str = "gptq"
    bits: int = 4
    group_size: int = 128
    output_dir: str = "./quantized"
    calibration_data: Optional[str] = None
    num_calibration_samples: int = 128
    gpu_id: int = 0
    smooth_alpha: float = 0.5
    fp8: bool = True
    act_order: bool = True
    percdamp: float = 0.01
    blocksize: int = 128
    n_search: int = 20
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    num_gpus: int = 1
    verbose: bool = False

    def __post_init__(self):
        assert self.method in ("gptq", "awq", "smoothquant"), f"Unknown method: {self.method}"
        assert self.bits in (3, 4, 8), f"Unsupported bit-width: {self.bits}"
        assert self.group_size > 0 and (self.group_size & (self.group_size - 1)) == 0, \
            "group_size must be a power of 2"
        if self.method == "smoothquant" and self.bits == 4:
            raise ValueError("SmoothQuant supports 8-bit and FP8 only")

    @classmethod
    def from_yaml(cls, path: str) -> "QuantConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Map YAML keys to dataclass fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered)

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    model: str
    method: str = "gptq"
    bits: int = 4
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 16])
    gen_lengths: list[int] = field(default_factory=lambda: [128, 512, 1024])
    warmup: int = 3
    output_file: Optional[str] = None
