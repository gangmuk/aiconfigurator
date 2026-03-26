# Benchmark Data: Pre-Collected Kernel Microbenchmarks

This directory contains **kernel-level GPU latency measurements** used by AIConfigurator to estimate end-to-end LLM inference performance without running actual models.

## How It Was Collected

Each operation (GEMM, attention, MoE, communication) was profiled **in isolation** on real hardware using the `collector/` module:

- **3 warmup iterations** to stabilize GPU clocks
- **6 timed iterations** with CUDA graph capture (eager fallback for complex ops)
- Latency reported in **milliseconds**
- No end-to-end model runs -- pure kernel microbenchmarks

The collector is version-aware: each backend (TRT-LLM, SGLang, vLLM) has versioned collector modules that match specific framework APIs.

## Directory Structure

```
data/
├── <system>/                    # GPU type (h200_sxm, h100_sxm, gb200, ...)
│   ├── <backend>/<version>/     # Framework + version (trtllm/1.2.0rc5, sglang/0.5.9, ...)
│   │   ├── gemm_perf.txt
│   │   ├── context_attention_perf.txt
│   │   ├── generation_attention_perf.txt
│   │   ├── moe_perf.txt
│   │   ├── custom_allreduce_perf.txt
│   │   └── ...
│   └── nccl/<version>/          # NCCL collectives (shared across backends)
│       └── nccl_perf.txt
```

**9 systems**: a100_sxm, b200_sxm, b60, gb200, gb300, h100_sxm, h200_sxm, l40s, rtxpro6000_blackwell_server

**3 backends**: trtllm (TensorRT-LLM), sglang, vllm -- each with multiple versions

## File Reference

### Core Operations

| File | What | Key Parameters | Typical Rows |
|------|------|----------------|-------------|
| `gemm_perf.txt` | Matrix multiply | `gemm_dtype` (float16/fp8/fp8_block/int4_wo/int8_wo/sq), `m` (batch 1-8192), `n`, `k` (dims 64-65536) | ~54K |
| `context_attention_perf.txt` | Prefill-phase attention | `batch_size` (1-256), `isl` (16-16384), `num_heads`, `num_key_value_heads`, `attn_dtype`, `kv_cache_dtype`, `window_size` | ~47K |
| `generation_attention_perf.txt` | Decode-phase attention | Same as above + `step` (KV cache depth: 1, 3, 7, 15, ..., 16383) | ~30K |
| `moe_perf.txt` | Mixture of Experts | `moe_dtype`, `num_tokens` (1-65536), `topk` (2/8), `num_experts` (8/128/256), `moe_tp_size`, `moe_ep_size`, `distribution` | ~64K |
| `custom_allreduce_perf.txt` | Intra-node TP all-reduce | `num_gpus` (2/4/8), `message_size` (128B-33MB) | ~70 |

### Communication

| File | What | Key Parameters |
|------|------|----------------|
| `nccl_perf.txt` | NCCL collectives | `op_name` (all_reduce/all_gather/reduce_scatter/alltoall), `num_gpus`, `message_size` (256B-64MB) |

### Specialized Operations

| File | What | When Present |
|------|------|-------------|
| `context_mla_perf.txt` / `generation_mla_perf.txt` | Multi-head Latent Attention (DeepSeek) | Models with MLA |
| `mla_bmm_perf.txt` | MLA batch matrix multiply | Models with MLA |
| `mamba2_perf.txt` | Mamba2 state-space ops | GB200 only (NemotronH) |
| `scale_matrix_perf.txt` / `computescale_perf.txt` | FP8 quantization scaling | FP8-capable systems |
| `wideep_*.txt` (12 files) | DeepSeek WideEP operations | SGLang only |

## MoE Token Distribution

The MoE benchmarks include **three load imbalance patterns** in the `distribution` column:

- `balanced` -- uniform routing (theoretical best case)
- `power_law_1.01` -- mild Zipfian skew (observed in DeepSeek-V3-R1)
- `power_law_1.2` -- heavier skew (observed in Qwen3-235B)

The measured latency reflects the **bottleneck EP group** (the one receiving the most tokens).

## Important Notes

- **Git LFS**: All `.txt` files are tracked by Git LFS. Run `git lfs pull` after cloning.
- **No end-to-end inference**: These are isolated kernel benchmarks. Scheduling, memory allocation, and inter-op overhead are modeled analytically by the configurator.
- **Latency unit**: All `latency` columns are in **milliseconds**.
- **Interpolation**: The configurator interpolates/extrapolates between measured data points for parameter combinations not directly profiled.
- **INCOMPLETE.txt**: Marker files indicating a data collection run did not finish. These directories may have partial data.
- **Power data**: Optional. Most files show `0.0 W` unless `--measure_power` was used during collection.
- **File format**: CSV with header row. Parseable with `pandas.read_csv()`.
