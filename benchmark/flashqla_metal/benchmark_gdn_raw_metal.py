import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

COMPONENTS = ("raw-kkt", "raw-forward", "raw-forward-outputs", "raw-forward-outputs-32")
METAL_TARGET = "metal -supports_simdgroup=True"


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
    chunk = k.shape[0]
    scores = k @ k.T
    row_idx = torch.arange(chunk, device=k.device).view(chunk, 1)
    col_idx = torch.arange(chunk, device=k.device).view(1, chunk)
    causal = col_idx < row_idx
    gated = scores * torch.exp(g_cum.view(chunk, 1) - g_cum.view(1, chunk))
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


def _selected_components(args):
    requested = getattr(args, "component", None) or ["raw-forward"]
    if "all" in requested:
        return list(COMPONENTS)
    return requested


def _run_raw_kkt(tilelang, probes, args):
    kernel = tilelang.compile(probes._make_flashqla_gdn_raw_kkt_probe(), target=METAL_TARGET)
    row_k = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 17.0
    col_k = (torch.arange(64, dtype=torch.float32).reshape(8, 8).flip(1) - 10.0) / 19.0
    row_k_mps, col_k_mps = row_k.to("mps"), col_k.to("mps")
    scores = torch.empty((8, 8), dtype=torch.float32, device="mps")

    def run_raw_kkt():
        kernel(row_k_mps, col_k_mps, scores)

    def run_torch_ref():
        row_k_mps @ col_k_mps.T

    run_raw_kkt()
    torch.mps.synchronize()
    ref = row_k @ col_k.T
    assert torch.allclose(scores.cpu(), ref, atol=1e-5, rtol=1e-5)

    raw_ms = _bench(run_raw_kkt, args.warmup, args.repeats)
    torch_ref_ms = _bench(run_torch_ref, args.warmup, args.repeats)
    return {
        "name": "flashqla_gdn_raw_kkt_8x8",
        "component": "raw-kkt",
        "rows": 8,
        "cols": 8,
        "key_dim": 8,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_ms": raw_ms,
        "torch_ref_ms": torch_ref_ms,
        "speedup_vs_torch_ref": torch_ref_ms / raw_ms if raw_ms > 0 else None,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_raw_forward(tilelang, probes, args):
    kernel = tilelang.compile(probes._make_flashqla_gdn_raw_forward_probe(), target=METAL_TARGET)
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

    raw_ms = _bench(run_raw_forward, args.warmup, args.repeats)
    torch_ref_ms = _bench(run_torch_ref, args.warmup, args.repeats)
    return {
        "name": "flashqla_gdn_raw_forward_16x16",
        "component": "raw-forward",
        "chunk": 16,
        "key_dim": 16,
        "value_dim": 16,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_ms": raw_ms,
        "torch_ref_ms": torch_ref_ms,
        "speedup_vs_torch_ref": torch_ref_ms / raw_ms if raw_ms > 0 else None,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_raw_forward_outputs(tilelang, probes, args):
    kernel = tilelang.compile(probes._make_flashqla_gdn_raw_forward_outputs_probe(), target=METAL_TARGET)
    k, v, beta, g_cum = probes._flashqla_gdn_component_synthetic_inputs()
    k_mps, v_mps = k.to("mps"), v.to("mps")
    beta_mps, g_cum_mps = beta.to("mps"), g_cum.to("mps")
    w = torch.empty((16, 16), dtype=torch.float32, device="mps")
    u = torch.empty((16, 16), dtype=torch.float32, device="mps")

    def run_raw_forward_outputs():
        kernel(k_mps, v_mps, beta_mps, g_cum_mps, w, u)

    def run_torch_ref():
        _torch_gdn_component_ref(k_mps, v_mps, beta_mps, g_cum_mps)

    run_raw_forward_outputs()
    torch.mps.synchronize()
    _, ref_w, ref_u = probes._flashqla_gdn_component_ref(k, v, beta, g_cum)
    assert torch.allclose(w.cpu(), ref_w, atol=1e-4, rtol=1e-5)
    assert torch.allclose(u.cpu(), ref_u, atol=1e-4, rtol=1e-5)

    raw_ms = _bench(run_raw_forward_outputs, args.warmup, args.repeats)
    torch_ref_ms = _bench(run_torch_ref, args.warmup, args.repeats)
    return {
        "name": "flashqla_gdn_raw_forward_outputs_16x16",
        "component": "raw-forward-outputs",
        "chunk": 16,
        "key_dim": 16,
        "value_dim": 16,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_ms": raw_ms,
        "torch_ref_ms": torch_ref_ms,
        "speedup_vs_torch_ref": torch_ref_ms / raw_ms if raw_ms > 0 else None,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _run_raw_forward_outputs32(tilelang, probes, args):
    kernel = tilelang.compile(probes._make_flashqla_gdn_raw_forward_outputs32_probe(), target=METAL_TARGET)
    k, v, beta, g_cum = probes._flashqla_gdn_component32_synthetic_inputs()
    k_mps, v_mps = k.to("mps"), v.to("mps")
    beta_mps, g_cum_mps = beta.to("mps"), g_cum.to("mps")
    w = torch.empty((32, 16), dtype=torch.float32, device="mps")
    u = torch.empty((32, 16), dtype=torch.float32, device="mps")

    def run_raw_forward_outputs32():
        kernel(k_mps, v_mps, beta_mps, g_cum_mps, w, u)

    def run_torch_ref():
        _torch_gdn_component_ref(k_mps, v_mps, beta_mps, g_cum_mps)

    run_raw_forward_outputs32()
    torch.mps.synchronize()
    _, ref_w, ref_u = probes._flashqla_gdn_component_ref(k, v, beta, g_cum)
    assert torch.allclose(w.cpu(), ref_w, atol=1e-4, rtol=1e-5)
    assert torch.allclose(u.cpu(), ref_u, atol=1e-4, rtol=1e-5)

    raw_ms = _bench(run_raw_forward_outputs32, args.warmup, args.repeats)
    torch_ref_ms = _bench(run_torch_ref, args.warmup, args.repeats)
    return {
        "name": "flashqla_gdn_raw_forward_outputs_32x16",
        "component": "raw-forward-outputs-32",
        "chunk": 32,
        "key_dim": 16,
        "value_dim": 16,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "raw_ms": raw_ms,
        "torch_ref_ms": torch_ref_ms,
        "speedup_vs_torch_ref": torch_ref_ms / raw_ms if raw_ms > 0 else None,
        "commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_benchmark(args):
    _validate(args)
    print(f"MPS: {torch.backends.mps.is_available()}")
    if not torch.backends.mps.is_available():
        raise SystemExit("Raw GDN Metal benchmark requires PyTorch MPS support")

    _ensure_repo_on_path()
    import tilelang

    probes = _load_probe_module()
    runners = {
        "raw-kkt": _run_raw_kkt,
        "raw-forward": _run_raw_forward,
        "raw-forward-outputs": _run_raw_forward_outputs,
        "raw-forward-outputs-32": _run_raw_forward_outputs32,
    }
    components = _selected_components(args)
    component_results = []
    for component in components:
        print(f"=== {component} ===")
        result = runners[component](tilelang, probes, args)
        component_results.append(result)
        print(f"raw:       {result['raw_ms']:.4f} ms")
        print(f"torch_ref: {result['torch_ref_ms']:.4f} ms")
        print(f"speedup:   {result['speedup_vs_torch_ref']:.2f}x")
        print()

    if len(component_results) == 1:
        result = component_results[0]
    else:
        result = {
            "name": "flashqla_gdn_raw_component_suite",
            "components": component_results,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "commit": _git_commit(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    if args.output_json:
        _write_json(args.output_json, result)
        print(f"Wrote JSON results to {args.output_json}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic raw-fragment Metal GDN benchmark")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--component",
        action="append",
        choices=(*COMPONENTS, "all"),
        help="Component to benchmark. May be passed multiple times; use 'all' for the component suite.",
    )
    parser.add_argument("--output-json", help="Write machine-readable benchmark results to this path")
    run_benchmark(parser.parse_args())
