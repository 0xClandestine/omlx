"""Bonsai 1-bit AFFINE weight loading patch (uint32 mlx-native format).

Complements ``bonsai_t5_load`` (which handles the base-3 uint8 *ternary* format)
and ``bonsai_qmv`` (fast decode for already-constructed 1-/2-bit layers).

The gap this closes
-------------------
Standard ``prism-ml/Bonsai-27B-mlx-1bit`` weights are ordinary mlx affine
quantization with ``bits=1`` (uint32-packed weight + fp16 scales/biases,
group_size=128).  Loading such a model does::

    nn.quantize(model, group_size=128, bits=1)   # build QuantizedLinear/Embedding
      -> layer.to_quantized(...) -> mx.quantize(w, bits=1)
      -> ValueError: bits 1 not supported (stock mlx supports 2,3,4,5,6,8)

so the model dies at *construction*, before any weight is loaded and before the
bonsai decode kernels are ever reached.  ``bonsai_qmv`` can execute a bits=1
layer but cannot create one; ``bonsai_t5_load`` only relaxes ``load_weights``.

Two patches (mirroring the style of ``bonsai_t5_load``)
------------------------------------------------------
Patch 1 - ``mx.quantize``
  For ``bits==1`` (affine) return correctly *shaped* zero placeholders
  (uint32 weight (N,K/32), fp16 scales/biases (N,K/group_size)) instead of
  invoking the unsupported kernel.  The real weights overwrite these during
  ``load_weights``; placeholder values are never used.  All other bit-widths
  delegate to the original C function unchanged.

Patch 2 - ``mx.quantized_matmul``
  qwen3_5 / mlx_vlm call ``mx.quantized_matmul`` directly.  For uint32 bits=1
  affine weights: decode (M<=5) routes to the native ``bonsai_q1`` Metal kernel;
  prefill (M>5) dequantizes (unpack 32 bits/uint32, w = scale*q + bias) and uses
  ``mx.matmul``.  Composes with an already-installed t5 wrapper: non-matching
  calls fall through to whatever ``quantized_matmul`` was patched before us.

Apply once via ``apply_bonsai_affine1_load_patch()`` before mlx_vlm / mlx_lm
load().  Idempotent; ``remove_*`` restores originals.
"""
from __future__ import annotations

import logging

import mlx.core as mx

logger = logging.getLogger(__name__)

_original_quantize = None
_original_dequantize = None
_prev_quantized_matmul = None   # may itself be the t5 wrapper
_patch_active = False

_MAX_DECODE_M = 5


def _placeholder_quantize(w, group_size: int = 64, bits: int = 4, mode: str = "affine",
                          **kwargs):
    """mx.quantize wrapper: synthesize bits=1 affine placeholders; else delegate."""
    if bits == 1 and mode == "affine" and w.ndim == 2:
        N, K = w.shape
        n_groups = K // group_size
        packed = mx.zeros((N, K // 32), dtype=mx.uint32)      # 32 one-bit vals / uint32
        scales = mx.zeros((N, n_groups), dtype=mx.float16)
        biases = mx.zeros((N, n_groups), dtype=mx.float16)
        return packed, scales, biases
    return _original_quantize(w, group_size=group_size, bits=bits, mode=mode, **kwargs)


def _dequant_affine1(w, scales, biases, group_size, dtype):
    """Unpack uint32 bits=1 affine weight -> float (*lead, K). LSB-first, mlx layout.

    Handles arbitrary leading dims (2D linear weights AND 3D+ embedding gathers,
    where token indices add batch/seq leading axes).
    """
    lead = tuple(w.shape[:-1])
    packed_cols = w.shape[-1]
    K = packed_cols * 32
    n_groups = scales.shape[-1]
    shifts = mx.arange(32, dtype=mx.uint32)
    q = (w[..., None] >> shifts) & mx.array(1, dtype=mx.uint32)   # (*lead, packed_cols, 32)
    q = q.reshape(*lead, n_groups, group_size).astype(dtype)
    sc = scales.astype(dtype).reshape(*lead, n_groups, 1)
    bi = biases.astype(dtype).reshape(*lead, n_groups, 1)
    return (q * sc + bi).reshape(*lead, K)


def _dequantize_affine1(w, scales=None, biases=None, group_size: int = 64,
                        bits: int = 4, mode: str = "affine", **kwargs):
    """mx.dequantize wrapper: unpack uint32 bits=1 affine manually; else delegate.

    Stock mlx has no compiled ``affine_dequantize_*_b_1`` Metal kernel, so any
    dequantize at bits=1 (e.g. QuantizedEmbedding lookup during prefill) throws.
    Handle it with bit-shift unpacking instead.
    """
    if (bits == 1 and mode == "affine" and w is not None
            and w.dtype == mx.uint32 and w.ndim >= 2
            and scales is not None and biases is not None):
        return _dequant_affine1(w, scales, biases, group_size, scales.dtype)
    return _original_dequantize(
        w, scales, biases, group_size=group_size, bits=bits, mode=mode, **kwargs
    )


def _affine1_quantized_matmul(x, w, scales, biases, *, transpose: bool = True,
                              bits: int = 4, group_size: int = 64, **kwargs):
    """quantized_matmul wrapper for uint32 bits=1 affine weights."""
    is_affine1 = (bits == 1 and w.dtype == mx.uint32
                  and scales is not None and biases is not None)
    if not is_affine1:
        return _prev_quantized_matmul(
            x, w, scales, biases, transpose=transpose, bits=bits,
            group_size=group_size, **kwargs)

    M = x.shape[-2] if x.ndim >= 2 else 1

    # Decode: native bonsai 1-bit kernel (the whole point of the PR).
    if M <= _MAX_DECODE_M and transpose:
        from omlx.custom_kernels.bonsai.fast import (
            bonsai_q1_affine_qmv, bonsai_qmv_wide, has_native, _use_qmv_wide,
        )
        if has_native():
            sc = scales.astype(x.dtype)
            bi = biases.astype(x.dtype)
            if _use_qmv_wide(1, M):
                return bonsai_qmv_wide(x, w, sc, bi, bits=1)
            return bonsai_q1_affine_qmv(x, w, sc, bi)

    # Prefill (or non-transpose): dequantize + matmul.
    weight_fp = _dequant_affine1(w, scales, biases, group_size, x.dtype)
    return x @ weight_fp.T if transpose else x @ weight_fp


def apply_bonsai_affine1_load_patch() -> bool:
    """Patch mx.quantize + mx.quantized_matmul for uint32 bits=1 affine models."""
    global _original_quantize, _original_dequantize
    global _prev_quantized_matmul, _patch_active
    if _patch_active:
        return False
    # If the underlying mlx already supports bits=1 natively (e.g. the PrismML
    # mlx fork), no Python shim is needed — let native kernels handle everything.
    try:
        _t = mx.random.normal((32, 128)).astype(mx.float16)
        mx.eval(mx.quantize(_t, group_size=64, bits=1)[0])
        logger.info("bonsai_affine1_load: native bits=1 present; patch not needed.")
        return False
    except Exception:
        pass
    _original_quantize = mx.quantize
    mx.quantize = _placeholder_quantize
    _original_dequantize = mx.dequantize
    mx.dequantize = _dequantize_affine1
    _prev_quantized_matmul = mx.quantized_matmul   # keep any t5 wrapper in the chain
    mx.quantized_matmul = _affine1_quantized_matmul
    _patch_active = True
    logger.info(
        "bonsai_affine1_load: mx.quantize + mx.dequantize + mx.quantized_matmul "
        "patched for uint32 bits=1 affine weights."
    )
    return True


def remove_bonsai_affine1_load_patch() -> None:
    global _original_quantize, _original_dequantize
    global _prev_quantized_matmul, _patch_active
    if not _patch_active:
        return
    if _original_quantize is not None:
        mx.quantize = _original_quantize
        _original_quantize = None
    if _original_dequantize is not None:
        mx.dequantize = _original_dequantize
        _original_dequantize = None
    if _prev_quantized_matmul is not None:
        mx.quantized_matmul = _prev_quantized_matmul
        _prev_quantized_matmul = None
    _patch_active = False


def is_patch_active() -> bool:
    return _patch_active
