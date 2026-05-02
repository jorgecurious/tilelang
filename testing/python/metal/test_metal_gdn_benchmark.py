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


def _fake_component_result(benchmark, component="raw-forward", **overrides):
    if component == "raw-kkt":
        dimensions = {"rows": 8, "cols": 8, "key_dim": 8}
    else:
        dimensions = {"chunk": 16, "key_dim": 16, "value_dim": 16}
    result = {
        **benchmark._benchmark_metadata(SimpleNamespace(__version__="1.2.3")),
        **dimensions,
        "name": f"fake_{component}",
        "component": component,
        "warmup": 0,
        "repeats": 1,
        "raw_ms": 1.0,
        "torch_ref_ms": 2.0,
        "speedup_vs_torch_ref": 2.0,
    }
    result.update(overrides)
    return result


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


def test_gdn_benchmark_validates_single_component_result_schema(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")

    result = _fake_component_result(benchmark, component="raw-kkt")

    assert benchmark._validate_benchmark_result_schema(result) is result


def test_gdn_benchmark_rejects_missing_component_result_field(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    result = _fake_component_result(benchmark, component="raw-forward")
    del result["value_dim"]

    try:
        benchmark._validate_benchmark_result_schema(result)
    except ValueError as err:
        assert "raw-forward result missing required fields: value_dim" in str(err)
    else:
        raise AssertionError("expected missing value_dim to raise ValueError")


def test_gdn_benchmark_validates_suite_result_schema(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    components = [
        _fake_component_result(benchmark, component="raw-forward-outputs-32"),
        _fake_component_result(benchmark, component="raw-kkt"),
    ]
    result = {
        **benchmark._benchmark_metadata(SimpleNamespace(__version__="1.2.3")),
        "name": "flashqla_gdn_raw_component_suite",
        "components_requested": ["raw-forward-outputs-32", "raw-kkt"],
        "components": components,
        "warmup": 0,
        "repeats": 1,
    }

    assert benchmark._validate_benchmark_result_schema(result) is result
    assert result["components"] is components
    assert [component["component"] for component in result["components"]] == ["raw-forward-outputs-32", "raw-kkt"]


def test_gdn_benchmark_rejects_suite_component_order_mismatch(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    result = {
        **benchmark._benchmark_metadata(SimpleNamespace(__version__="1.2.3")),
        "name": "flashqla_gdn_raw_component_suite",
        "components_requested": ["raw-kkt", "raw-forward"],
        "components": [
            _fake_component_result(benchmark, component="raw-forward"),
            _fake_component_result(benchmark, component="raw-kkt"),
        ],
        "warmup": 0,
        "repeats": 1,
    }

    try:
        benchmark._validate_benchmark_result_schema(result)
    except ValueError as err:
        assert "suite component order does not match components_requested" in str(err)
    else:
        raise AssertionError("expected component order mismatch to raise ValueError")


def test_gdn_benchmark_rejects_missing_suite_field(monkeypatch):
    benchmark = _load_benchmark_module()
    monkeypatch.setattr(benchmark.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(benchmark, "_git_commit", lambda: "abc123")
    result = {
        **benchmark._benchmark_metadata(SimpleNamespace(__version__="1.2.3")),
        "name": "flashqla_gdn_raw_component_suite",
        "components": [_fake_component_result(benchmark, component="raw-forward")],
        "warmup": 0,
        "repeats": 1,
    }

    try:
        benchmark._validate_benchmark_result_schema(result)
    except ValueError as err:
        assert "suite result missing required fields: components_requested" in str(err)
    else:
        raise AssertionError("expected missing components_requested to raise ValueError")


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

    def fake_raw_kkt_runner(_tilelang, _probes, runner_args):
        return _fake_component_result(
            benchmark,
            component="raw-kkt",
            warmup=runner_args.warmup,
            repeats=runner_args.repeats,
            tilelang_version=_tilelang.__version__,
            mps_available=True,
        )

    def fake_raw_forward_runner(_tilelang, _probes, runner_args):
        return _fake_component_result(
            benchmark,
            component="raw-forward",
            warmup=runner_args.warmup,
            repeats=runner_args.repeats,
            tilelang_version=_tilelang.__version__,
            mps_available=True,
        )

    monkeypatch.setattr(benchmark, "_run_raw_kkt", fake_raw_kkt_runner)
    monkeypatch.setattr(benchmark, "_run_raw_forward", fake_raw_forward_runner)

    result = benchmark.run_benchmark(args)

    assert result["schema_version"] == 1
    assert result["target"] == benchmark.METAL_TARGET
    assert result["torch_version"] == benchmark.torch.__version__
    assert result["tilelang_version"] == "9.9.9"
    assert result["mps_available"] is True
    assert result["components_requested"] == ["raw-kkt", "raw-forward"]
    assert [component["schema_version"] for component in result["components"]] == [1, 1]
    assert [component["component"] for component in result["components"]] == ["raw-kkt", "raw-forward"]
