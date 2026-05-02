import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[3]
    benchmark_path = repo_root / "benchmark" / "flashqla_metal" / "benchmark_gdn_raw_metal.py"
    spec = importlib.util.spec_from_file_location("benchmark_gdn_raw_metal", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gdn_benchmark_writes_json(tmp_path):
    benchmark = _load_benchmark_module()
    output_json = tmp_path / "results" / "gdn.json"
    result = {
        "name": "flashqla_gdn_raw_forward_16x16",
        "raw_forward_ms": 0.25,
        "torch_ref_ms": 0.5,
    }

    benchmark._write_json(output_json, result)

    assert json.loads(output_json.read_text(encoding="utf-8")) == result


def test_gdn_benchmark_validates_repeats():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(warmup=0, repeats=0, output_json=None)

    try:
        benchmark._validate(args)
    except SystemExit as err:
        assert "--repeats must be a positive integer" in str(err)
    else:
        raise AssertionError("expected invalid repeats to raise SystemExit")


def test_gdn_benchmark_selects_default_component():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(warmup=0, repeats=1, component=None, output_json=None)

    assert benchmark._selected_components(args) == ["raw-forward"]


def test_gdn_benchmark_selects_all_components():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(warmup=0, repeats=1, component=["all"], output_json=None)

    assert benchmark._selected_components(args) == [
        "raw-kkt",
        "raw-forward",
        "raw-forward-outputs",
        "raw-forward-outputs-32",
    ]


def test_gdn_benchmark_preserves_requested_component_order():
    benchmark = _load_benchmark_module()
    args = argparse.Namespace(
        warmup=0,
        repeats=1,
        component=["raw-forward-outputs-32", "raw-forward-outputs", "raw-forward", "raw-kkt"],
        output_json=None,
    )

    assert benchmark._selected_components(args) == [
        "raw-forward-outputs-32",
        "raw-forward-outputs",
        "raw-forward",
        "raw-kkt",
    ]


def test_gdn_benchmark_metadata_uses_stable_fields(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark.time, "gmtime", lambda: "fixed-gmtime")
    monkeypatch.setattr(benchmark.time, "strftime", lambda fmt, value: f"{fmt}:{value}")
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)

    metadata = benchmark._benchmark_metadata(SimpleNamespace(__version__="1.2.3"))

    assert metadata == {
        "schema_version": 1,
        "target": benchmark.METAL_TARGET,
        "torch_version": benchmark.torch.__version__,
        "tilelang_version": "1.2.3",
        "mps_available": False,
        "commit": "abc123",
        "timestamp": "%Y-%m-%dT%H:%M:%SZ:fixed-gmtime",
    }


def test_gdn_benchmark_metadata_allows_missing_tilelang_version():
    benchmark = _load_benchmark_module()

    assert benchmark._tilelang_version(SimpleNamespace()) is None


def test_gdn_benchmark_suite_json_has_requested_components_and_metadata(monkeypatch):
    benchmark = _load_benchmark_module()
    fake_tilelang = SimpleNamespace(__version__="9.9.9")
    args = argparse.Namespace(
        warmup=0,
        repeats=1,
        component=["raw-kkt", "raw-forward"],
        output_json=None,
    )

    monkeypatch.setitem(__import__("sys").modules, "tilelang", fake_tilelang)
    monkeypatch.setattr(benchmark, "_load_probe_module", lambda: SimpleNamespace())
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(benchmark.time, "gmtime", lambda: "fixed-gmtime")
    monkeypatch.setattr(benchmark.time, "strftime", lambda fmt, value: f"{fmt}:{value}")

    def fake_runner(_tilelang, _probes, runner_args):
        return {
            **benchmark._benchmark_metadata(_tilelang),
            "name": f"fake_{runner_args.repeats}",
            "component": "fake",
            "warmup": runner_args.warmup,
            "repeats": runner_args.repeats,
            "raw_ms": 1.0,
            "torch_ref_ms": 2.0,
            "speedup_vs_torch_ref": 2.0,
        }

    monkeypatch.setattr(benchmark, "_run_raw_kkt", fake_runner)
    monkeypatch.setattr(benchmark, "_run_raw_forward", fake_runner)

    result = benchmark.run_benchmark(args)

    assert result["schema_version"] == 1
    assert result["target"] == benchmark.METAL_TARGET
    assert result["torch_version"] == benchmark.torch.__version__
    assert result["tilelang_version"] == "9.9.9"
    assert result["mps_available"] is True
    assert result["components_requested"] == ["raw-kkt", "raw-forward"]
    assert [component["schema_version"] for component in result["components"]] == [1, 1]
