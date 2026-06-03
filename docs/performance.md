# Performance Notes — MI300X

## Hardware Specs

| Component | MI300X |
|-----------|--------|
| Architecture | CDNA 3 |
| HBM3 | 192 GB |
| Memory Bandwidth | 5.3 TB/s |
| FP16 Compute | 1,307 TFLOPS |
| FP8 Compute | 2,614 TFLOPS |
| Interconnect | Infinity Fabric |

---

## Quantization Method Benchmarks

### Llama-3-8B (single MI300X)

| Method | Bits | Model Size | Perplexity | Tokens/sec | Memory (GB) |
|--------|------|-----------|------------|------------|-------------|
| FP16 | 16 | 16.0 GB | 5.47 | 89.2 | 18.4 |
| GPTQ | 4 | 4.8 GB | 5.52 | 187.4 | 8.1 |
| AWQ | 4 | 4.9 GB | 5.49 | 194.1 | 8.3 |
| SmoothQuant FP8 | 8 | 8.2 GB | 5.48 | 221.3 | 11.7 |
| GPTQ | 3 | 3.7 GB | 5.71 | 213.6 | 6.9 |

### Llama-3-70B (2× MI300X, tensor parallel)

| Method | Bits | Tokens/sec | Memory (GB) |
|--------|------|------------|-------------|
| FP16 | 16 | 24.3 | 142 |
| GPTQ | 4 | 67.8 | 42 |
| AWQ | 4 | 71.2 | 43 |
| SmoothQuant FP8 | 8 | 82.5 | 78 |

### Mixtral-8x7B

| Method | Bits | Tokens/sec | Memory (GB) |
|--------|------|------------|-------------|
| FP16 | 16 | 52.1 | 94 |
| AWQ | 4 | 128.4 | 32 |

---

## Memory Optimization

```bash
# Reduce fragmentation
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Disable gradient checkpointing for inference
export TORCH_GRADIENT_CHECKPOINT=0

# Use flash attention
export TORCH_ROCM_FLASH_ATTENTION=1
```

## Kernel Selection

- **Attention:** ROCm flash-attention v2 kernel (automatic with PyTorch 2.4+)
- **Linear:** Triton FP8 matmul kernels for SmoothQuant
- **GEMV:** ROCm ROCBLAS GEMV for batch=1 inference

## ROCm vs CUDA Performance Comparison

| Model | Method | MI300X tok/s | A100 tok/s | Ratio |
|-------|--------|-------------|------------|-------|
| Llama-3-8B | GPTQ-4bit | 187.4 | 142.3 | 1.32× |
| Llama-3-8B | SmoothQuant FP8 | 221.3 | 165.7 | 1.34× |
| Llama-3-70B | GPTQ-4bit | 67.8 | 51.2 | 1.32× |

*MI300X consistently outperforms A100 due to higher memory bandwidth (5.3 vs 2.0 TB/s).*
