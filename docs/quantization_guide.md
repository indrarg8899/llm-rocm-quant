# Quantization Guide

## Overview

LLM-ROCm-Quant supports three quantization methods, each optimized for AMD MI300X GPUs with ROCm 6.x.

---

## GPTQ (Generalized Post-Training Quantization)

**Reference:** Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"

### How it works

1. **Hessian estimation** — Accumulate approximate inverse Hessian H⁻¹ from calibration data using the empirical Fisher information.
2. **Layer-wise quantization** — Process each linear layer independently, quantizing weight columns in blocks.
3. **Error compensation** — After quantizing a block, the quantization error is propagated to remaining columns via the Hessian, minimizing overall reconstruction error.

### Configuration

```yaml
method: gptq
bits: 4          # 3 or 4
group_size: 128  # per-channel quantization group
act_order: true  # activation-ordered quantization
percdamp: 0.01   # Hessian dampening (0.01 = 1%)
blocksize: 128   # block size for iterative updates
```

### When to use

- Best for pure text generation models (LLaMA, Mistral, Phi)
- Supports 3-bit and 4-bit
- Fast quantization (minutes on MI300X)
- Minimal accuracy loss at 4-bit

---

## AWQ (Activation-Aware Weight Quantization)

**Reference:** Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"

### How it works

1. **Saliency identification** — Run calibration data, measure per-channel activation magnitudes. Channels with larger activations are more "salient."
2. **Scaling search** — Find per-channel scaling factors that protect salient channels from quantization error. Search over α ∈ [0, 1] to minimize reconstruction loss.
3. **Uniform quantization** — Apply group-wise symmetric quantization on the scaled weights.

### Configuration

```yaml
method: awq
bits: 4          # 3 or 4
group_size: 128
n_search: 20     # number of alpha candidates
alpha_min: 0.0
alpha_max: 1.0
```

### When to use

- Best when accuracy is critical (smaller perplexity delta vs FP16)
- Particularly effective for MoE models (Mixtral)
- Slightly slower quantization due to alpha search

---

## SmoothQuant with FP8

**Reference:** Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"

### How it works

1. **Per-channel smoothing** — Compute per-channel smoothing factor s based on activation and weight magnitudes: sⱼ = (max(|xⱼ|)^α) / (max(|Wⱼ|)^(1-α))
2. **Migration** — Smooth input activations: W' = W/s, X' = X·s (absorbed into next layer)
3. **FP8 quantization** — Apply FP8 (E4M3) symmetric quantization to smoothed weights. Range: ±448, 3-bit exponent.

### Configuration

```yaml
method: smoothquant
bits: 8           # 8-bit INT or FP8
smooth_alpha: 0.5  # smoothing strength (0.0=no smoothing, 1.0=full)
fp8: true          # use FP8 vs INT8
```

### When to use

- Best for 8-bit precision requirements
- FP8 on MI300X achieves near-FP16 accuracy
- Good for encoder models (BERT, RoBERTa) and LLMs

---

## ROCm Optimization Tips

- **Flash Attention:** Enable via `TORCH_ROCM_FLASH_ATTENTION=1` for faster inference
- **Memory:** Set `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` to reduce fragmentation
- **Multi-GPU:** Use `num_gpus > 1` for tensor-parallel quantization of large models
- **Kernel fusion:** ROCm fuses fused attention kernels automatically with PyTorch 2.4+
