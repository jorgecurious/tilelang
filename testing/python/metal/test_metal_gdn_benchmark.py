import argparse
import importlib.util
import json
from pathlib import Path


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
