# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Bonsai t5 LUT-GEMM prefill kernel.

Verifies that bonsai_t5_qmm_lut (paired-trit μ=2 table-dot) produces output
numerically identical to the existing bonsai_t5_qmm (dequant + simdgroup MMA)
for the same random t5-packed weights / scales / activations.

Covered:
  - group_size 64 and 128 (partial last byte of each K-group = 3/4 trits)
  - multiple K-groups per row (partial-byte path taken in every group)
  - M / N shapes that exercise edge tiles (M,N not multiples of 32)
  - both fp16 and bf16 activation dtypes

Skips when the native extension (or the lut symbol) is unavailable.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

import omlx.custom_kernels.bonsai.fast as bf
from tools.repack_ternary_t5 import pack_t5

# Relative tolerance: fp32 accumulate with fp16-rounded dequant elements; the
# LUT path keeps the pair sums in fp32. Differences are fp16 rounding only,
# well inside 1e-2 relative.
RTOL = 1e-2
ATOL = 1e-2

SHAPES_GS = [
    # (M, K, N, group_size)
    (8, 128, 16, 128),     # exactly one K-group
    (33, 256, 12, 128),    # two groups, edge tiles (M/N not % 32 == 0)
    (16, 384, 32, 128),    # three groups: partial last byte every group
    (8, 128, 16, 64),
    (33, 192, 12, 64),
    (16, 256, 32, 64),
]


def _t5_tensors(M, K, N, group_size, dtype, seed):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(M, K)).astype(np.float16)).astype(dtype)
    n_groups = K // group_size
    quants = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
    w = mx.array(pack_t5(quants, group_size))
    scales = mx.array(
        rng.normal(size=(N, n_groups)).astype(np.float16)
    ).astype(dtype)
    return x, w, scales


@pytest.mark.parametrize(("M", "K", "N", "group_size"), SHAPES_GS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_lut_matches_reference(M, K, N, group_size, dtype):
    if not (bf.has_native()
            and all(bf.has_symbol(s) for s in
                    ("bonsai_t5_qmm_lut", "bonsai_t5_qmm_nomul", "bonsai_t5_qmm_steel"))):
        pytest.skip("t5 qmm native extensions not available")
    x, w, scales = _t5_tensors(M, K, N, group_size, dtype, seed=42)
    ref = bf.bonsai_t5_qmm(x, w, scales)
    lut = bf.bonsai_t5_qmm_lut(x, w, scales)
    nom = bf.bonsai_t5_qmm_nomul(x, w, scales)
    stl = bf.bonsai_t5_qmm_steel(x, w, scales)
    mx.eval(ref, lut, nom, stl)
    assert lut.shape == ref.shape == (M, N)
    # mlx-native comparison (np.asarray on fp16 hits a numpy>=2 buffer-format
    # quirk with mlx 0.32; mx.allclose avoids the interop entirely).
    tol = ATOL + RTOL * mx.abs(ref)
    assert mx.all(mx.less(mx.abs(lut - ref), tol)).item(), (
        f"lut vs reference mismatch (M={M}, K={K}, N={N}, gs={group_size}, {dtype}): "
        f"max abs {mx.max(mx.abs(lut - ref)).item():.3e}"
    )
    assert mx.all(mx.less(mx.abs(nom - ref), tol)).item(), (
        f"nomul vs reference mismatch (M={M}, K={K}, N={N}, gs={group_size}, {dtype}): "
        f"max abs {mx.max(mx.abs(nom - ref)).item():.3e}"
    )
    assert mx.all(mx.less(mx.abs(stl - ref), tol)).item(), (
        f"steel vs reference mismatch (M={M}, K={K}, N={N}, gs={group_size}, {dtype}): "
        f"max abs {mx.max(mx.abs(stl - ref)).item():.3e}"
    )


def test_lut_raises_without_native(monkeypatch):
    """Fallback contract: RuntimeError when the native symbol is missing."""
    monkeypatch.setattr(bf, "_ext", None)
    with pytest.raises(RuntimeError, match="bonsai_t5_qmm_lut"):
        bf.bonsai_t5_qmm_lut(mx.zeros((2, 64)), mx.zeros((2, 13), mx.uint8),
                             mx.ones((2, 1)))
