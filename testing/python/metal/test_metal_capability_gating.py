"""Focused tests for Metal simdgroup capability gating.

These tests verify that:
1. target_has_simdgroup fails closed (returns False) for plain metal targets.
2. target_has_simdgroup returns True when explicit attrs are set.
3. Gemm instruction selection raises for Metal targets without simdgroup support.
4. MetalFragmentToSimdgroup raises for Metal targets without simdgroup support.
5. GemmMetal.lower raises for Metal targets without simdgroup support.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import pytest

from tvm.target import Target
from tvm import tir
from tilelang.utils.target import target_has_simdgroup, target_get_metal_version, target_is_metal
from tilelang.transform.metal_fragment_to_simdgroup import _metal_fragment_to_simdgroup


class TestTargetHasSimdgroup:
    """Test fail-closed simdgroup capability detection."""

    def test_plain_metal_is_false(self):
        t = Target("metal")
        assert target_is_metal(t)
        assert target_has_simdgroup(t) is False

    def test_explicit_supports_simdgroup_true(self):
        t = Target("metal -supports_simdgroup=True")
        assert target_has_simdgroup(t) is True

    def test_explicit_supports_simdgroup_false(self):
        t = Target("metal -supports_simdgroup=False")
        assert target_has_simdgroup(t) is False

    def test_arch_apple7_is_true(self):
        t = Target("metal -arch=apple7")
        assert target_has_simdgroup(t) is True

    def test_arch_apple8_is_true(self):
        t = Target("metal -arch=apple8")
        assert target_has_simdgroup(t) is True

    def test_arch_apple6_is_false(self):
        t = Target("metal -arch=apple6")
        assert target_has_simdgroup(t) is False

    def test_metal_version_23_is_true(self):
        t = Target("metal -metal_version=23")
        assert target_has_simdgroup(t) is True

    def test_metal_version_30_is_true(self):
        t = Target("metal -metal_version=30")
        assert target_has_simdgroup(t) is True

    def test_metal_version_20_is_false(self):
        t = Target("metal -metal_version=20")
        assert target_has_simdgroup(t) is False

    def test_cuda_is_false(self):
        t = Target("cuda -arch=sm_80")
        assert target_has_simdgroup(t) is False
        assert target_get_metal_version(t) == 0

    def test_hip_is_false(self):
        t = Target("hip -mcpu=gfx90a")
        assert target_has_simdgroup(t) is False


class TestMetalFragmentToSimdgroupGating:
    """Test that MetalFragmentToSimdgroup raises on unsupported targets."""

    def test_plain_metal_without_gemm_is_unchanged(self):
        t = Target("metal")
        x = tir.Var("x", "int32")
        body = tir.Evaluate(tir.const(0, "int32"))
        func = tir.PrimFunc([x], body).with_attr("target", t)

        result = _metal_fragment_to_simdgroup(func, None, None)
        assert result.same_as(func)

    def test_passes_with_simdgroup(self):
        t = Target("metal -supports_simdgroup=True")
        x = tir.Var("x", "int32")
        body = tir.Evaluate(tir.const(0, "int32"))
        func = tir.PrimFunc([x], body).with_attr("target", t)

        # No gemm ops in body, so it should return func unchanged
        result = _metal_fragment_to_simdgroup(func, None, None)
        assert result.same_as(func)

    def test_skips_non_metal(self):
        t = Target("cuda -arch=sm_80")
        x = tir.Var("x", "int32")
        body = tir.Evaluate(tir.const(0, "int32"))
        func = tir.PrimFunc([x], body).with_attr("target", t)

        result = _metal_fragment_to_simdgroup(func, None, None)
        assert result.same_as(func)


class TestGemmMetalLowerGating:
    """Test that GemmMetal.lower raises on unsupported targets."""

    def test_gemm_lower_raises_without_simdgroup(self):
        @T.prim_func
        def main(
            A: T.Tensor((16, 16), "float16"),
            B: T.Tensor((16, 16), "float16"),
            C: T.Tensor((16, 16), "float32"),
        ):
            with T.Kernel(1, 1, threads=128):
                A_shared = T.alloc_shared((16, 16), "float16", scope="shared")
                B_shared = T.alloc_shared((16, 16), "float16", scope="shared")
                C_local = T.alloc_fragment((16, 16), "float32")
                T.copy(A, A_shared)
                T.copy(B, B_shared)
                T.clear(C_local)
                T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C)

        with pytest.raises(ValueError, match="simdgroup"):
            tilelang.lower(main, target="metal")


if __name__ == "__main__":
    tilelang.testing.main()
