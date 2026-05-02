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
        shape=None,
        block_config=None,
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
    assert written["block_configs"] == [[16, 16, 16], [32, 32, 16]]
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
        shape=None,
        block_config=None,
        output_json=None,
    )
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)

    try:
        benchmark.run_benchmark(args)
    except SystemExit as err:
        assert "--repeats must be a positive integer" in str(err)
    else:
        raise AssertionError("expected invalid repeats to raise SystemExit")


def test_metal_benchmark_uses_custom_block_configs(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)
    output_json = tmp_path / "custom.json"
    args = argparse.Namespace(
        m=256,
        n=256,
        k=128,
        warmup=0,
        repeats=1,
        sweep=True,
        shape=None,
        block_config=[(64, 64, 16), (128, 64, 32)],
        output_json=str(output_json),
    )

    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "bench_torch_mps", lambda *args: 8.0)

    seen_configs = []

    def bench_tilelang(_M, _N, _K, block_M, block_N, block_K, _warmup, _repeats):
        seen_configs.append((block_M, block_N, block_K))
        return 4.0 if block_M == 64 else 5.0

    monkeypatch.setattr(benchmark, "bench_tilelang", bench_tilelang)

    result = benchmark.run_benchmark(args)
    written = json.loads(output_json.read_text(encoding="utf-8"))

    assert seen_configs == [(64, 64, 16), (128, 64, 32)]
    assert result["best_config"] == [128, 64, 32]
    assert written["block_configs"] == [[64, 64, 16], [128, 64, 32]]


def test_metal_benchmark_parses_block_config(monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)

    assert benchmark._parse_block_config("64,128,32") == (64, 128, 32)
    try:
        benchmark._parse_block_config("64,128")
    except argparse.ArgumentTypeError as err:
        assert "form M,N,K" in str(err)
    else:
        raise AssertionError("expected malformed block config to raise")


def test_metal_benchmark_writes_multi_shape_json(tmp_path, monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)
    output_json = tmp_path / "suite.json"
    args = argparse.Namespace(
        m=4096,
        n=4096,
        k=4096,
        shape=[(128, 256, 64), (256, 512, 128)],
        warmup=1,
        repeats=2,
        sweep=False,
        block_config=[(64, 64, 32)],
        output_json=str(output_json),
    )

    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark, "bench_torch_mps", lambda M, _N, _K, _warmup, _repeats: M / 64)
    monkeypatch.setattr(benchmark, "bench_tilelang", lambda M, _N, _K, *_args: M / 128)

    result = benchmark.run_benchmark(args)
    written = json.loads(output_json.read_text(encoding="utf-8"))

    assert result["runs"][0]["m"] == 128
    assert result["runs"][1]["m"] == 256
    assert written["warmup"] == 1
    assert written["repeats"] == 2
    assert written["block_configs"] == [[64, 64, 32]]
    assert written["commit"] == "abc123"
    assert len(written["runs"]) == 2
    assert written["runs"][0]["torch_tflops"] == 2.0
    assert written["runs"][0]["best_tilelang_tflops"] == 1.0
    assert written["runs"][1]["torch_tflops"] == 4.0
    assert written["runs"][1]["best_tilelang_tflops"] == 2.0


def test_metal_benchmark_parses_shape(monkeypatch):
    benchmark = _load_benchmark_module(monkeypatch)

    assert benchmark._parse_shape("128,4096,4096") == (128, 4096, 4096)
    try:
        benchmark._parse_shape("128,4096")
    except argparse.ArgumentTypeError as err:
        assert "form M,N,K" in str(err)
    else:
        raise AssertionError("expected malformed shape to raise")
