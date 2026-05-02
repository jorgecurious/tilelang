# Metal Internal Runtime Coverage

This document summarizes internal-only Metal backend coverage for scalar lowering, simdgroup helpers, packed quantization probes, and GDN/attention-style tiled kernels. It does not add public `T.rt`/`T.rv` aliases and does not introduce model checkpoints, production integration, MPP/cooperative lowering, MPSGraph, CUDA, or native fp8/fp6/fp4 Metal storage.

## Runtime-validated coverage

- Packed quant matmul: `M=16,N=32,K=64`, synthetic packed `uint8` fp8 activations, packed `uint8` fp4 weights, `uint8` e8m0 activation/weight scales, and fp32 output. MPS output is compared against a CPU decode/reference matmul.
- GDN/attention-style staged component probe: `chunk=16,key_dim=16,value_dim=16`, deterministic synthetic fp32 tensors, staged 8x8 KKT score accumulation over two key-dimension slices, scalar gate/causal triangular mask, and tiled W/U accumulation. MPS output is compared against Torch reference.
- Smaller runtime probes remain in `testing/python/metal/test_metal_internal_scaffolding.py` and related focused Metal tests.

## Source-boundary-only / fail-closed coverage

- Native Metal fp8/fp6/fp4 storage remains intentionally unsupported and fail-closed; component probes keep the packed `uint8` boundary.
- RegisterTile/RowVector helpers remain internal under `tilelang.tileop`; no public language aliases are added.
- RegisterTile layout/transpose misuse is covered by negative tests that fail before Metal source generation, preserving helper-internal source boundaries.
- Component probes assert that forbidden external backend tokens (`cooperative`, `mpp`, `mpsgraph`, `cuda`, etc.) are absent from generated Metal source.

## Known blockers and deferrals

- This coverage is correctness/scaffolding only; it does not optimize component-scale performance.
- Any future MPP/cooperative support claim needs an explicit capability gate, layout/permutation proof tests, source-boundary preservation checks, and MPS runtime correctness coverage first.
- Packed quant matmul uses scalar per-output decode/accumulation rather than native fp8/fp4 tensor storage or a production quantized GEMM lowering.
- GDN/attention-style coverage remains synthetic and chunk-local; no checkpoint-bound integration or full production recurrent/chunked scheduler is included.

## Future native sub-byte storage contract

Native fp8/fp6/fp4 Metal storage remains intentionally unsupported until all of these prerequisites exist together:

- Type mapping: CodeGenMetal must define exact Metal storage/scalar types for each supported TileLang dtype.
- Storage layout: tests must prove byte/nibble/bit packing, vector lane layout, and alignment for global, threadgroup, and register boundaries.
- Packing semantics: encode/decode behavior must match CPU references for signed zeros, saturation, scale bytes, and unsupported bit patterns.
- Target capability: native storage lowering must be gated by target metadata rather than selected by dtype presence alone.
- Source-boundary checks: packed `uint8` fallback kernels must remain free of accidental native `float8`, `float6`, or `float4` Metal tokens until native storage is enabled.
- Runtime correctness: focused MPS tests must compare native-storage kernels against CPU/Torch references before any support claim is enabled.

## Verification hooks

- Default focused tests: `python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py -q`
- Opt-in component timing hook: `TILELANG_RUN_METAL_COMPONENT_BENCH=1 python3 -m pytest testing/python/metal/test_metal_internal_scaffolding.py::test_component_synthetic_runtime_benchmarks_opt_in -q -s`
