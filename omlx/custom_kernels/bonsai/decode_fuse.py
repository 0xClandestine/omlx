# SPDX-License-Identifier: Apache-2.0
"""Fused single-token decode kernels for Bonsai ternary models.

Maple-style fused dispatches (see deepgrove/maple-preview-2bit-mlx): single
token decode on MLX is bounded by its serial dispatch chain (~150 us per
kernel at M=1 on M4-class silicon), not by the math. These kernels collapse
elementwise chains into one Metal dispatch:

  * ``fused_add_rms_norm`` — residual add + RMSNorm (1 dispatch, replaces 2-4)
  * ``fused_qk_norm_rope`` — per-head q/k RMSNorm + partial RoPE (1 dispatch,
    replaces q_norm + k_norm + 2 rope calls)

Both are implemented as JIT kernels via ``mx.fast.metal_kernel`` (mlx 0.32+),
created lazily per (dtype, shape) and cached. Numerically they reproduce the
same op chains: the sum is rounded once (like a dtype add), the norm reads the
rounded stream with an fp32 weight multiply (reference RMSNorm semantics), and
the RoPE rotation follows the non-traditional pairing (i, i + dims/2).
"""

from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

# Simdgroups in the add+rmsnorm kernel; N = DIM must divide n_threads * PT.
_ADD_RMS_THREADS = 256
_ADD_RMS_SGS = 8  # 256 / 32

_add_rms_cache: dict[tuple, object] = {}
_qk_rope_cache: dict[tuple, object] = {}


def _add_rms_kernel(dtype: mx.Dtype, dim: int, eps: float):
    key = (str(dtype), dim, f"{eps:.6e}")
    k = _add_rms_cache.get(key)
    if k is not None:
        return k
    if dim % _ADD_RMS_THREADS:
        raise ValueError(f"fused_add_rms_norm: dim {dim} not divisible by 256")
    pt = dim // _ADD_RMS_THREADS
    source = r"""
        uint tid = thread_position_in_grid.x;   // 0..255 (grid.x == total threads)
        constexpr uint N = DIM;
        constexpr uint PT = N / 256u;
        float hb[PT];
        float ss = 0.0f;
        for (uint i = 0u; i < PT; ++i) {
            uint j = tid * PT + i;
            float v = (float)x[j] + (float)r[j];
            T_ vb = (T_)v;              // one rounding, same as a dtype add
            h_out[j] = vb;
            hb[i] = (float)vb;          // norm sees the rounded stream
            ss += hb[i] * hb[i];
        }
        ss = simd_sum(ss);
        threadgroup float sums[8];
        uint sg = tid / 32u;
        uint lane = tid % 32u;
        if (lane == 0u) sums[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0.0f;
        for (uint i = 0u; i < 8u; ++i) tot += sums[i];
        float scale = metal::rsqrt(tot / (float)N + EPS_);
        for (uint i = 0u; i < PT; ++i) {
            uint j = tid * PT + i;
            hn_out[j] = (T_)(hb[i] * scale * (float)w[j]);
        }
    """.replace("EPS_", f"{eps:.10e}f")
    _nm = "".join(c if (c.isalnum() or c == '_') else '_' for c in str(key))
    k = mx.fast.metal_kernel(
        name=f"omlx_fused_add_rms_{_nm}",
        input_names=["x", "r", "w"],
        output_names=["h_out", "hn_out"],
        source=source,
    )
    _add_rms_cache[key] = k
    return k


def fused_add_rms_norm(
    h: mx.array,
    r: mx.array,
    w: mx.array,
    eps: float,
) -> tuple[mx.array, mx.array]:
    """h_out = round(h + r); hn = rmsnorm(h_out) with fp32 weight multiply.

    One dispatch instead of add + rms_norm (+ optional astype round-trips).
    """
    dim = h.shape[-1]
    kern = _add_rms_kernel(h.dtype, dim, eps)
    hi = h.reshape(-1)
    out_h, out_hn = kern(
        inputs=[hi, r.reshape(-1), w.reshape(-1)],
        template=[("T_", h.dtype), ("DIM", dim)],
        grid=(_ADD_RMS_THREADS, 1, 1),
        threadgroup=(_ADD_RMS_THREADS, 1, 1),
        output_shapes=[hi.shape, hi.shape],
        output_dtypes=[h.dtype, h.dtype],
    )
    return out_h.reshape(h.shape), out_hn.reshape(h.shape)


def _qk_rope_kernel(dtype: mx.Dtype, head_dim: int, rope_dim: int):
    key = (str(dtype), head_dim, rope_dim)
    k = _qk_rope_cache.get(key)
    if k is not None:
        return k
    if head_dim % 32:
        raise ValueError(f"fused_qk_norm_rope: head_dim {head_dim} not divisible by 32")
    if rope_dim % 2:
        raise ValueError(f"fused_qk_norm_rope: rope_dim {rope_dim} must be even")
    source = r"""
        uint t = thread_position_in_grid.x;    // grid.x == total threads
        uint head = t / 32u;
        uint lane = t % 32u;
        constexpr int per_lane = HEAD_DIM / 32;
        const device T_* xh = x + head * HEAD_DIM;
        const device T_* wh = w + head * HEAD_DIM;
        device T_* oh = out + head * HEAD_DIM;
        float ss = 0.0f;
        for (int i = 0; i < per_lane; ++i) {
            float v = (float)xh[lane * per_lane + i];
            ss += v * v;
        }
        ss = simd_sum(ss);
        float pos = pos_eps[0];
        float eps = pos_eps[1];
        float scale = metal::rsqrt(ss / (float)HEAD_DIM + eps);
        for (int i = 0; i < per_lane; ++i) {
            int j = lane * per_lane + i;
            float v = (float)xh[j] * scale * (float)wh[j];
            if (ROPE_DIM > 0 && j < ROPE_DIM) {
                constexpr int rhalf = ROPE_DIM / 2;
                int p = j < rhalf ? j : j - rhalf;
                float theta = pos * inv_freq[p];
                float c = metal::cos(theta);
                float s = metal::sin(theta);
                int j2 = j < rhalf ? j + rhalf : j - rhalf;
                float u = (float)xh[j2] * scale * (float)wh[j2];
                v = j < rhalf ? (v * c - u * s) : (v * c + u * s);
            }
            oh[j] = (T_)v;
        }
    """
    _nm = "".join(c if (c.isalnum() or c == '_') else '_' for c in str(key[0]))
    k = mx.fast.metal_kernel(
        name=f"omlx_fused_qk_rope_{_nm}_{head_dim}_{rope_dim}",
        input_names=["x", "w", "inv_freq", "pos_eps"],
        output_names=["out"],
        source=source,
    )
    _qk_rope_cache[key] = k
    return k


def fused_qk_norm_rope(
    qk: mx.array,
    w: mx.array,
    rope_base: float,
    rope_dim: int,
    pos: float,
    eps: float,
) -> mx.array:
    """Per-head RMSNorm + partial non-traditional RoPE in one dispatch.

    qk: (n_heads, head_dim) — q and k heads stacked (q block first).
    w:  (n_heads, head_dim) per-head norm weights.
    Returns (n_heads, head_dim), normed and rotated.
    """
    n_heads, head_dim = qk.shape
    # kernel is monomorphic in the activation dtype — cast weights defensively
    w = w.astype(qk.dtype)
    rhalf = rope_dim // 2
    inv_freq = rope_base ** (-mx.arange(rhalf, dtype=mx.float32) / rhalf)
    pos_eps = mx.array([float(pos), eps], dtype=mx.float32)
    kern = _qk_rope_kernel(qk.dtype, head_dim, rope_dim)
    flat = qk.reshape(-1)
    out = kern(
        inputs=[flat, w.reshape(-1), inv_freq, pos_eps],
        template=[("T_", qk.dtype), ("HEAD_DIM", head_dim), ("ROPE_DIM", rope_dim)],
        grid=(32 * n_heads, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[flat.shape],
        output_dtypes=[qk.dtype],
    )[0]
    return out.reshape(qk.shape)
