import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

import torch

import tilelang
import tilelang.language as T

logging.getLogger("tilelang").setLevel(logging.WARNING)

BLOCK_CONFIGS = [
    (16, 16, 16),
    (32, 32, 16),
    (32, 32, 32),
    (64, 64, 32),
]


@tilelang.jit
def matmul_simdgroup(M, N, K, block_M=64, block_N=64, block_K=32, dtype=T.float16, accum_dtype=T.float32):

    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


def _tflops(M, N, K, seconds):
    return 2.0 * M * N * K / seconds / 1e12


def _bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - t0) / repeats


def _git_commit():
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _config_result(block_config, tilelang_tflops, ref_tflops, error=None):
    block_M, block_N, block_K = block_config
    result = {
        "block_m": block_M,
        "block_n": block_N,
        "block_k": block_K,
        "tilelang_tflops": tilelang_tflops,
        "ratio_pct": None if tilelang_tflops is None else tilelang_tflops / ref_tflops * 100,
        "status": "failed" if error else "ok",
    }
    if error:
        result["error"] = str(error)
    return result


def _write_json(path, result):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_block_config(value):
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("block config must have the form M,N,K")
    try:
        config = tuple(int(part) for part in parts)
    except ValueError as err:
        raise argparse.ArgumentTypeError("block config values must be integers") from err
    if any(part <= 0 for part in config):
        raise argparse.ArgumentTypeError("block config values must be positive")
    return config


def _selected_configs(args):
    if args.block_config:
        return args.block_config
    return BLOCK_CONFIGS if args.sweep else [(64, 64, 32)]


def run_benchmark(args):
    M, N, K = args.m, args.n, args.k
    for name, value in (("m", M), ("n", N), ("k", K), ("repeats", args.repeats)):
        if value <= 0:
            raise SystemExit(f"--{name} must be a positive integer, got {value}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be non-negative, got {args.warmup}")

    print(f"torch:    {torch.__version__}")
    print(f"tilelang: {tilelang.__version__}")
    print(f"MPS:      {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("Metal GEMM benchmark requires PyTorch MPS support")
    print(f"M={M}, N={N}, K={K}, warmup={args.warmup}, repeats={args.repeats}")
    print()

    ref_tflops = bench_torch_mps(M, N, K, args.warmup, args.repeats)
    print(f"PyTorch MPS (torch.mm fp16): {ref_tflops:.1f} TFLOPS")
    print()

    configs = _selected_configs(args)

    print(f"{'block (M,N,K)':>16s} | {'TileLang':>14s} | {'Ratio':>6s}")
    print("-" * 44)

    results = []
    best = None
    for config in configs:
        bM, bN, bK = config
        try:
            tl = bench_tilelang(M, N, K, bM, bN, bK, args.warmup, args.repeats)
            ratio = tl / ref_tflops * 100
            if best is None or tl > best[1]:
                best = (config, tl)
            results.append(_config_result(config, tl, ref_tflops))
            print(f"{f'({bM},{bN},{bK})':>16s} | {tl:>10.1f} TFLOPS | {ratio:>5.0f}%")
        except Exception as e:
            results.append(_config_result(config, None, ref_tflops, error=e))
            print(f"{f'({bM},{bN},{bK})':>16s} | {'FAILED':>14s} | {e}")

    if args.sweep:
        print()
        print(f"Best config: {None if best is None else best[0]}")
        print(f"Best TFlops: {0.0 if best is None else best[1]:.1f}")
        print(f"Reference TFlops (PyTorch MPS): {ref_tflops:.1f}")

    result = {
        "m": M,
        "n": N,
        "k": K,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sweep": args.sweep,
        "block_configs": [list(config) for config in configs],
        "torch_tflops": ref_tflops,
        "best_config": None if best is None else list(best[0]),
        "best_tilelang_tflops": None if best is None else best[1],
        "configs": results,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if args.output_json:
        _write_json(args.output_json, result)
        print()
        print(f"Wrote JSON results to {args.output_json}")
    return result


def bench_torch_mps(M, N, K, warmup, repeats):
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    avg_s = _bench(lambda: torch.mm(a, b), warmup, repeats)
    return _tflops(M, N, K, avg_s)


def bench_tilelang(M, N, K, block_M, block_N, block_K, warmup, repeats):
    kernel = matmul_simdgroup(M, N, K, block_M, block_N, block_K)
    a = torch.randn(M, K, dtype=torch.float16, device="mps")
    b = torch.randn(K, N, dtype=torch.float16, device="mps")
    c = torch.zeros(M, N, dtype=torch.float32, device="mps")
    avg_s = _bench(lambda: kernel(a, b, c), warmup, repeats)
    return _tflops(M, N, K, avg_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metal GEMM Benchmark (simdgroup)")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sweep", action="store_true", help="Sweep all block configs instead of using default (64,64,32)")
    parser.add_argument(
        "--block-config",
        action="append",
        type=_parse_block_config,
        help="Custom block config as M,N,K. May be passed multiple times; overrides --sweep.",
    )
    parser.add_argument("--output-json", help="Write machine-readable benchmark results to this path")
    args = parser.parse_args()
    run_benchmark(args)
