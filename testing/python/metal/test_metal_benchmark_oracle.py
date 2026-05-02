import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types


def _load_benchmark_module(monkeypatch):
    fake_tilelang = types.ModuleType("tilelang")
    fake_tilelang.__version__ = "test"
    fake_tilelang.jit = lambda fn: fn
    fake_language = types.ModuleType("tilelang.language")
    fake_language.float16 = "float16"
    fake_language.float32 = "float32"
    fake_language.prim_func = lambda fn: fn
    fake_tilelang.language = fake_language
    fake_torch = types.SimpleNamespace(
        __version__="test",
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "tilelang", fake_tilelang)
    monkeypatch.setitem(sys.modules, "tilelang.language", fake_language)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    repo_root = Path(__file__).resolve().parents[3]
    benchmark_path = repo_root / "benchmark" / "matmul_metal" / "benchmark_matmul_metal.py"
    spec = importlib.util.spec_from_file_location("benchmark_matmul_metal", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metal_benchmark_writes_json_results(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)
    output_json = tmp_path / "results" / "matmul.json"
    args = argparse.Namespace(
        m=128,
        n=256,
        k=64,
        warmup=1,
        repeats=2,
        sweep=True,
        output_json=str(output_json),
    )

    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "BLOCK_CONFIGS", [(16, 16, 16), (32, 32, 16)])
    monkeypatch.setattr(benchmark, "bench_torch_mps", lambda *args: 4.0)

    def bench_tilelang(_M, _N, _K, block_M, _block_N, _block_K, _warmup, _repeats):
        if block_M == 32:
            raise RuntimeError("compile failed")
        return 2.0

    monkeypatch.setattr(benchmark, "bench_tilelang", bench_tilelang)

    result = benchmark.run_benchmark(args)
    written = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["best_config"] == [16, 16, 16]
    assert written["m"] == 128
    assert written["n"] == 256
    assert written["k"] == 64
    assert written["warmup"] == 1
    assert written["repeats"] == 2
    assert written["sweep"] is True
    assert written["torch_tflops"] == 4.0
    assert written["best_tilelang_tflops"] == 2.0
    assert written["commit"] == "abc123"
    assert written["configs"][0] == {
        "block_m": 16,
        "block_n": 16,
        "block_k": 16,
        "tilelang_tflops": 2.0,
        "ratio_pct": 50.0,
        "status": "ok",
    }
    assert written["configs"][1]["status"] == "failed"
    assert written["configs"][1]["tilelang_tflops"] is None
    assert written["configs"][1]["ratio_pct"] is None
    assert written["configs"][1]["error"] == "compile failed"


def test_metal_benchmark_validates_inputs_before_running(monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)
    args = argparse.Namespace(
        m=128,
        n=128,
        k=128,
        warmup=0,
        repeats=0,
        sweep=False,
        output_json=None,
    )
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)

    try:
        benchmark.run_benchmark(args)
    except SystemExit as err:
        assert "--repeats must be a positive integer" in str(err)
    else:
        raise AssertionError("expected invalid repeats to raise SystemExit")
