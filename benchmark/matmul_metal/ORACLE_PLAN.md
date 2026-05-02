# Metal GEMM Benchmark Oracle Plan

## Scope
Coordinate rerunnable benchmark/oracle coverage for TileLang Metal GEMM versus a
stable MPS baseline. This plan does **not** introduce MPSGraph (it remains a
forbidden external token per `test_metal_internal_scaffolding.py`); the oracle
baseline is `torch.mm` on MPS.

## Target Shapes

### Standard square/large GEMM (production-like)
| M    | N    | K    | Notes                           |
|------|------|------|---------------------------------|
| 1024 | 1024 | 1024 | Small square sanity check       |
| 2048 | 2048 | 2048 | Mid-size square                 |
| 4096 | 4096 | 4096 | Large square (default in bench) |
| 8192 | 8192 | 8192 | XL square - promotion gate      |

### LLM-weight rectangular GEMM (attention/FFN-like)
| M    | N     | K    | Notes                       |
|------|-------|------|-----------------------------|
| 1    | 4096  | 4096 | Token-generation decode     |
| 1    | 11008 | 4096 | LLaMA-7B FFN up-proj decode |
| 128  | 4096  | 4096 | Small-batch prefill         |
| 128  | 11008 | 4096 | Small-batch FFN             |
| 2048 | 4096  | 4096 | Medium-batch prefill        |
| 2048 | 11008 | 4096 | Medium-batch FFN            |

### Sweep configs for autotuning data
Block configs to sweep (already in `benchmark_matmul_metal.py`):
```
(16, 16, 16)
(32, 32, 16)
(32, 32, 32)
(64, 64, 32)
```
Future additions:
```
(64, 64, 16)
(128, 128, 32)
(128, 128, 64)
```

## Exact Rerunnable Commands

### 1. Default large square benchmark
```bash
cd /Users/work/Documents/tilelang_exp/tilelang_metal_gemm_upstream_rebase
python3 benchmark/matmul_metal/benchmark_matmul_metal.py --m 4096 --n 4096 --k 4096 --warmup 10 --repeats 100
```

### 2. Sweep all block configs on a given shape
```bash
python3 benchmark/matmul_metal/benchmark_matmul_metal.py --m 4096 --n 4096 --k 4096 --sweep --warmup 10 --repeats 100 --output-json benchmark/matmul_metal/results/metal_gemm_4096.json
```

### 3. Decode-weight rectangular shape
```bash
python3 benchmark/matmul_metal/benchmark_matmul_metal.py --m 128 --n 11008 --k 4096 --warmup 20 --repeats 200 --block-config 64,64,32 --block-config 128,64,32 --output-json benchmark/matmul_metal/results/metal_gemm_rectangular.json
```

### 4. Correctness oracle (pytest, runtime-validated on MPS)
```bash
python3 -m pytest testing/python/metal/test_metal_gemm_v2.py -q
```

### 5. Linux codegen oracle (no Metal runtime required)
```bash
python3 -m pytest testing/python/metal/test_metal_gemm_v2_linux.py -q
```

### 6. Internal scaffolding runtime + source-boundary oracle
```bash
python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py -q
```

### 7. Opt-in micro-benchmark hooks (small/scaled/component)
```bash
TILELANG_RUN_METAL_SMALL_BENCH=1 \
  python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py::test_small_synthetic_runtime_benchmarks_opt_in -q -s

TILELANG_RUN_METAL_SCALED_BENCH=1 \
  python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py::test_scaled_synthetic_runtime_benchmarks_opt_in -q -s

TILELANG_RUN_METAL_COMPONENT_BENCH=1 \
  python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py::test_component_synthetic_runtime_benchmarks_opt_in -q -s
```

## Preserved Artifacts

After each benchmark run, the following should be captured by CI/monitor scripts:

1. **Console stdout** - exact TFLOPS numbers and best-config summary.
2. **JSON artifact**: write with `--output-json benchmark/matmul_metal/results/<name>.json`
   - Fields: `m, n, k, warmup, repeats, sweep, block_configs, torch_tflops, best_config, best_tilelang_tflops, configs, timestamp, commit`
3. **Generated Metal source** (for codegen audits):
   ```python
   artifact = tilelang.lower(kernel, target="metal -supports_simdgroup=True")
   artifact.kernel_source  # save to file for diff auditing
   ```
4. **pytest JUnit XML** (if running via CI):
   ```bash
   python3 -m pytest testing/python/metal/test_metal_gemm_v2.py -q --junitxml=results/metal_gemm_v2.xml
   ```

## Promotion Criteria

A TileLang Metal GEMM config is **promotable** from experimental to recommended when:

1. **Correctness**: `test_metal_gemm_v2.py` passes for the target shape at `atol=1e-2` (fp16) or `atol=1e-4` (fp32).
2. **Source boundary**: `test_metal_gemm_v2_linux.py` emits `simdgroup_multiply_accumulate`, `simdgroup_load`, `simdgroup_store`, and no forbidden tokens (`cooperative`, `mpp`, `mpsgraph`, `cuda`, etc.).
3. **Performance gate**: TileLang achieves >= 80% of `torch.mm` MPS TFLOPS on Apple Silicon for the target shape, averaged over >= 100 repeats with 10+ warmup.
4. **Composed/target-shape improvement**: For non-square shapes (e.g., M=1, N=11008, K=4096), a promoted config must show composed improvement over the baseline square-tuned config (i.e., a shape-aware tuned config beats the generic 64x64x32 config).

## Gaps and Next Steps

| Gap | Impact | Proposed Action |
|-----|--------|-----------------|
| No MPSGraph baseline | MPSGraph is explicitly forbidden in source; `torch.mm` is the only viable oracle today | Keep `torch.mm` as oracle; document the MPSGraph restriction |
| Limited non-square block tuning | Decode-like shapes likely underperform | Use repeated `--block-config` values on rectangular shapes and persist the JSON results |
| No fp8/fp4 GEMM benchmark | Packed quant probes are scalar-only | Defer until native fp8/fp4 storage is supported; keep uint8 boundary probes as correctness oracle only |
| No FlashQLA/GDN full benchmark | Only 8x8 and 16x16 component probes exist | Add `benchmark_flashqla_metal.py` with target chunk/key/value dims once kernels are stable |

## Verification

Rerunnable end-to-end verification:
```bash
cd /Users/work/Documents/tilelang_exp/tilelang_metal_gemm_upstream_rebase
python3 -m pytest testing/python/metal/test_metal_gemm_v2.py testing/python/metal/test_metal_gemm_v2_linux.py -q
python3 benchmark/matmul_metal/benchmark_matmul_metal.py --m 4096 --n 4096 --k 4096 --sweep --warmup 10 --repeats 100 --output-json benchmark/matmul_metal/results/metal_gemm_4096.json
```
