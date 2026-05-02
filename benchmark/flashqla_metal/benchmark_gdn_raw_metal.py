import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _ensure_repo_on_path():
    repo_root = str(_repo_root())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(_repo_root()), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _load_probe_module():
    _ensure_repo_on_path()
    probe_path = _repo_root() / "testing" / "python" / "metal" / "test_metal_internal_scaffolding.py"
    spec = importlib.util.spec_from_file_location("metal_internal_scaffolding", probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path, result):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _bench(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
        torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
        torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats


def _torch_gdn_component_ref(k, v, beta, g_cum):
    scores = k @ k.T
    row_idx = torch.arange(16, device=k.device).view(16, 1)
    col_idx = torch.arange(16, device=k.device).view(1, 16)
    causal = col_idx < row_idx
    gated = scores * torch.exp(g_cum.view(16, 1) - g_cum.view(1, 16))
    a_pre = torch.where(causal, gated, torch.zeros_like(gated))
    k_scaled = k * (beta * torch.exp(g_cum)).unsqueeze(1)
    v_scaled = v * beta.unsqueeze(1)
    return a_pre, a_pre @ k_scaled, a_pre @ v_scaled


def _validate(args):
    for name, value in (("warmup", args.warmup), ("repeats", args.repeats)):
        if value < 0:
            raise SystemExit(f"--{name} must be non-negative, got {value}")
    if args.repeats == 0:
        raise SystemExit("--repeats must be a positive integer")


def run_benchmark(args):
    _validate(args)
    print(f"MPS: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("Raw GDN Metal benchmark requires PyTorch MPS support")

    _ensure_repo_on_path()
    import tilelang

    probes = _load_probe_module()
    kernel = tilelang.compile(probes._make_flashqla_gdn_raw_forward_probe(), target="metal")
    k, v, beta, g_cum = probes._flashqla_gdn_component_synthetic_inputs()
    k_mps, v_mps = k.to("mps"), v.to("mps")
    beta_mps, g_cum_mps = beta.to("mps"), g_cum.to("mps")
    a_pre = torch.empty((16, 16), dtype=torch.float32, device="mps")
    w = torch.empty((16, 16), dtype=torch.float32, device="mps")
    u = torch.empty((16, 16), dtype=torch.float32, device="mps")

    def run_raw_forward():
        kernel(k_mps, v_mps, beta_mps, g_cum_mps, a_pre, w, u)

    def run_torch_ref():
        _torch_gdn_component_ref(k_mps, v_mps, beta_mps, g_cum_mps)

    run_raw_forward()
    torch.mps.synchronize()
    ref_a, ref_w, ref_u = probes._flashqla_gdn_component_ref(k, v, beta, g_cum)
    assert torch.allclose(a_pre.cpu(), ref_a, atol=1e-4, rtol=1e-5)
    assert torch.allclose(w.cpu(), ref_w, atol=1e-4, rtol=1e-5)
    assert torch.allclose(u.cpu(), ref_u, atol=1e-4, rtol=1e-5)

    raw_forward_ms = _bench(run_raw_forward, args.warmup, args.repeats)
    torch_ref_ms = _bench(run_torch_ref, args.warmup, args.repeats)
    result = {
        "name": "flashqla_gdn_raw_forward_16x16",
        "chunk": 16,
        "key_dim": 16,
        "value_dim": 16,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_forward_ms": raw_forward_ms,
        "torch_ref_ms": torch_ref_ms,
        "speedup_vs_torch_ref": torch_ref_ms / raw_forward_ms if raw_forward_ms > 0 else None,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(f"raw_forward: {raw_forward_ms:.4f} ms")
    print(f"torch_ref:   {torch_ref_ms:.4f} ms")
    print(f"speedup:     {result['speedup_vs_torch_ref']:.2f}x")
    if args.output_json:
        _write_json(args.output_json, result)
        print(f"Wrote JSON results to {args.output_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic raw-fragment Metal GDN benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output-json", help="Write machine-readable benchmark results to this path")
    run_benchmark(parser.parse_args())
