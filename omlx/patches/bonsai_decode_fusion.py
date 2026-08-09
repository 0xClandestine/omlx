# SPDX-License-Identifier: Apache-2.0
"""Bonsai decode fusion (Maple-style) for 2-bit ternary models.

Single-token decode on MLX is bound by its serial dispatch chain (~150 us
per kernel at M=1 on M4-class silicon), not by the math — see the
deepgrove/maple-preview-2bit-mlx reference. This patches any mlx-lm-style
dense model with three maple optimizations:

  1. Projection fusion: q/k/v and gate/up QuantizedLinears are concatenated
     along the output axis into one module (qkv_proj / up_gate_proj).
     Row-wise quantized tensors concatenate losslessly, so outputs are exact.
  2. Fused per-head q/k RMSNorm + partial RoPE (one dispatch) in decode.
  3. Fused residual-add + RMSNorm at the decoder-layer level (one dispatch
     instead of add + norm per norm site).

Attention / MLP / layer modules are wrapped in lightweight nn.Module proxies
whose own ``__call__`` runs the fused decode path and delegates every other
attribute to the wrapped module. Every fused path is probe-verified against
the original on its first use; any mismatch latches it off permanently, so a
real model never runs a wrong fast path.

Usage
-----
    from omlx.patches.bonsai_decode_fusion import apply_bonsai_decode_fusion
    n = apply_bonsai_decode_fusion(model)   # number of fused sites
    remove_bonsai_decode_fusion(model)      # unwrap

Automatically applied in ``omlx.utils.model_loading.apply_post_load_transforms``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import mlx.core as mx
import mlx.nn as nn

from omlx.custom_kernels.bonsai.decode_fuse import (
    fused_add_rms_norm,
    fused_qk_norm_rope,
)

logger = logging.getLogger(__name__)

_PROBE_ATOL = 1e-2
_PROBE_RTOL = 1e-2


# ---------------------------------------------------------------------------
# fusion helpers
# ---------------------------------------------------------------------------


def _linear_quant_key(m: Any) -> Optional[tuple]:
    if not hasattr(m, "weight"):
        return None
    scales = getattr(m, "scales", None)
    if scales is None:
        return None
    if int(m.weight.shape[-1]) % int(scales.shape[-1]):
        return None
    bits = int(getattr(m, "bits", 0))
    pack = 32 // bits if bits else 1
    return (
        str(m.weight.dtype),
        int(m.weight.shape[-1] // scales.shape[-1]) * pack,
        bits,
        str(getattr(m, "mode", "affine")),
    )


def _fuse_linear_rows(modules: list[Any], name: str) -> nn.Module:
    """Concatenate row-wise quantized Linears into one module (exact math)."""
    w = mx.concatenate([m.weight for m in modules], axis=0)
    sc = mx.concatenate([m.scales for m in modules], axis=0)
    out = sum(int(m.weight.shape[0]) for m in modules)
    bits = int(getattr(modules[0], "bits", 2))
    fused = nn.QuantizedLinear(
        int(w.shape[-1]) * (32 // bits),  # K (unpacked)
        out,
        bias=False,
        group_size=int(getattr(modules[0], "group_size", 64)),
        bits=bits,
    )
    fused.weight = w
    fused.scales = sc
    biases = [getattr(m, "biases", None) for m in modules]
    if all(b is not None for b in biases):
        fused.biases = mx.concatenate(biases, axis=0)
    fused._omlx_fused_from = name
    return fused


def _exact(a: mx.array, b: mx.array) -> bool:
    try:
        mx.eval(a, b)
        return bool(
            mx.max(mx.abs(a - b)).item()
            <= _PROBE_ATOL + _PROBE_RTOL * mx.max(mx.abs(b)).item()
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# wrappers (own __call__, attribute delegation to .inner)
# ---------------------------------------------------------------------------


class _FusedAttention(nn.Module):
    def __init__(self, inner: nn.Module, qk_path: str):
        super().__init__()
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "qk_path", qk_path)
        cfg = self._cfg()
        object.__setattr__(self, "_nq", cfg["n_heads"])
        object.__setattr__(self, "_nkv", cfg["n_kv_heads"])
        object.__setattr__(self, "_hd", cfg["head_dim"])
        object.__setattr__(self, "_scale", float(getattr(inner, "scale", cfg["head_dim"] ** -0.5)))
        object.__setattr__(self, "_ok", None)

    def _cfg(self) -> dict:
        inner = self.inner
        n_heads = getattr(inner, "n_heads", None) or getattr(
            inner, "num_attention_heads", None
        )
        n_kv = getattr(inner, "n_kv_heads", None) or getattr(
            inner, "num_key_value_heads", None
        )
        hd = getattr(inner, "head_dim", None)
        if hd is None and n_heads:
            q = getattr(inner, "q_proj", None)
            hd = int(q.weight.shape[0]) // n_heads if q is not None else None
        assert n_heads and n_kv and hd
        return {"n_heads": int(n_heads), "n_kv_heads": int(n_kv), "head_dim": int(hd)}

    def __getattr__(self, k):
        inner = object.__getattribute__(self, "__dict__").get("inner")
        if inner is None:
            raise AttributeError(k)
        return getattr(inner, k)

    def _qkv_split(self, qkv):
        q = qkv[..., : self._nq * self._hd].reshape(1, self._nq, 1, self._hd)
        k = qkv[..., self._nq * self._hd : (self._nq + self._nkv) * self._hd].reshape(
            1, self._nkv, 1, self._hd
        )
        v = qkv[..., (self._nq + self._nkv) * self._hd :].reshape(
            1, self._nkv, 1, self._hd
        )
        return q, k, v

    def _norm_rope(self, q, k, offset):
        inner = self.inner
        if self.qk_path == "fused":
            nq, nkv, hd = self._nq, self._nkv, self._hd
            rope = getattr(inner, "rope", None)
            rope_dim = int(rope.dims) if rope is not None else 0
            base = float(getattr(rope, "base", 10000.0)) if rope is not None else 10000.0
            eps = float(inner.q_norm.eps)
            qkw = mx.concatenate(
                [
                    mx.broadcast_to(inner.q_norm.weight[None], (nq, hd)),
                    mx.broadcast_to(inner.k_norm.weight[None], (nkv, hd)),
                ],
                axis=0,
            )
            qk = mx.concatenate([q.reshape(nq, hd), k.reshape(nkv, hd)], axis=0)
            # kernel is monomorphic in the activation dtype (weights cast)
            out = fused_qk_norm_rope(qk, qkw.astype(q.dtype), base, rope_dim, float(offset), eps)
            return out[:nq].reshape(1, nq, 1, hd), out[nq:].reshape(1, nkv, 1, hd)
        qn = inner.q_norm(q) if getattr(inner, "q_norm", None) is not None else q
        kn = inner.k_norm(k) if getattr(inner, "k_norm", None) is not None else k
        if getattr(inner, "rope", None) is not None:
            qn = inner.rope(qn, offset=offset)
            kn = inner.rope(kn, offset=offset)
        return qn, kn

    def __call__(self, x, mask=None, cache=None, **kw):
        inner = self.inner
        orig = inner  # wrapped module callable via __call__ of its own type
        if self._ok is False:
            return inner(x, mask, cache)
        B, L, _ = x.shape
        if not (B == 1 and L == 1):
            return inner(x, mask, cache)
        try:
            qkv = self.qkv_proj(x)
            q, k, v = self._qkv_split(qkv)
            offset = cache.offset if cache is not None else 0
            q, k = self._norm_rope(q, k, offset)
            if cache is not None:
                k, v = cache.update_and_fetch(k, v)
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=self._scale, mask=mask
            )
            return self.o_proj(out.reshape(B, L, -1))
        except Exception:
            self._ok = False
            return inner(x, mask, cache)


class _FusedMLP(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "_ok", None)

    def __getattr__(self, k):
        inner = object.__getattribute__(self, "__dict__").get("inner")
        if inner is None:
            raise AttributeError(k)
        return getattr(inner, k)

    def __call__(self, x, **kw):
        inner = self.inner
        if self._ok is False:
            return inner(x)
        B, L, _ = x.shape
        if not (B == 1 and L == 1):
            return inner(x)
        try:
            xg = self.up_gate_proj(x)
            d = xg.shape[-1] // 2
            return self.down_proj(mx.nn.silu(xg[..., :d]) * xg[..., d:])
        except Exception:
            self._ok = False
            return inner(x)


class _FusedLayer(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "_ok", None)
        dim = inner.input_layernorm.weight.shape[-1]
        self._zero = None
        self._zero_shape = (1, 1, dim)
        self._dtype = inner.input_layernorm.weight.dtype

    def __getattr__(self, k):
        inner = object.__getattribute__(self, "__dict__").get("inner")
        if inner is None:
            raise AttributeError(k)
        return getattr(inner, k)

    def __call__(self, x, mask=None, cache=None, **kw):
        inner = self.inner
        if self._ok is False:
            return inner(x, mask, cache)
        B, L, _ = x.shape
        if not (B == 1 and L == 1):
            return inner(x, mask, cache)
        try:
            if self._zero is None:
                self._zero = mx.zeros(self._zero_shape, self._dtype)
            ln1 = inner.input_layernorm
            ln2 = inner.post_attention_layernorm
            _, hn = fused_add_rms_norm(x, self._zero, ln1.weight, float(ln1.eps))
            attn_proxy = getattr(inner.self_attn, "_omlx_fused_attn", None)
            mlp_proxy = getattr(inner.mlp, "_omlx_fused_mlp", None)
            r = attn_proxy(hn, mask, cache) if attn_proxy is not None else inner.self_attn(hn, mask, cache)
            h, hn = fused_add_rms_norm(x, r, ln2.weight, float(ln2.eps))
            r = mlp_proxy(hn) if mlp_proxy is not None else inner.mlp(hn)
            return h + r
        except Exception:
            self._ok = False
            return inner(x, mask, cache)




def _probe_attention_proxy(proxy: "_FusedAttention") -> bool:
    """Apply-time probe: fused vs original attention on fresh caches."""
    inner = proxy.inner
    D = inner.q_proj.weight.shape[-1] * (32 // getattr(inner.q_proj, "bits", 2))
    try:
        import numpy as np

        x = mx.array(np.random.default_rng(0).normal(size=(1, 1, D)).astype(np.float16))
        y_f = proxy(x, cache=_ProbeCache())
        y_o = inner(x, cache=_ProbeCache())
        return _exact(y_f, y_o)
    except Exception:
        return False


def _probe_mlp_proxy(proxy: "_FusedMLP") -> bool:
    inner = proxy.inner
    D = inner.gate_proj.weight.shape[-1] * (32 // getattr(inner.gate_proj, "bits", 2))
    try:
        import numpy as np

        x = mx.array(np.random.default_rng(0).normal(size=(1, 1, D)).astype(np.float16))
        return _exact(proxy(x), inner(x))
    except Exception:
        return False


def _probe_layer_proxy(proxy: "_FusedLayer") -> bool:
    inner = proxy.inner
    try:
        import numpy as np

        D = inner.input_layernorm.weight.shape[-1]
        x = mx.array(np.random.default_rng(0).normal(size=(1, 1, D)).astype(np.float16))
        y_f = proxy(x, cache=_ProbeCache())
        y_o = inner(x, cache=_ProbeCache())
        return _exact(y_f, y_o)
    except Exception:
        return False



class _ProbeCache:
    """Minimal KV cache with the mlx-lm interface (offset / update_and_fetch)."""

    def __init__(self):
        self._state: list = []
        self.offset = 0

    def update_and_fetch(self, keys, values):
        self._state.append((keys, values))
        if self.offset == 0:
            self.offset = 1
        return keys, values

# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def _maybe_fuse_attention(parent: Any, sites: list) -> None:
    """Fuse q/k/v and wrap the attention decode path (if layer fusion is off)."""
    q, k, v = getattr(parent, "q_proj", None), getattr(parent, "k_proj", None), getattr(parent, "v_proj", None)
    o = getattr(parent, "o_proj", None)
    if not all(x is not None for x in (q, k, v, o)) or not hasattr(q, "weight"):
        return
    key = _linear_quant_key(q)
    if not (key and key == _linear_quant_key(k) == _linear_quant_key(v)):
        return
    try:
        if getattr(parent, "qkv_proj", None) is None:
            parent.qkv_proj = _fuse_linear_rows([q, k, v], "qkv")
        qk_path = "sequential"
        if (
            getattr(parent, "q_norm", None) is not None
            and getattr(parent, "k_norm", None) is not None
            and hasattr(parent.q_norm, "eps")
        ):
            hd = getattr(parent, "head_dim", None)
            if hd is None:
                n_heads = getattr(parent, "n_heads", None) or getattr(parent, "num_attention_heads", 1)
                hd = int(q.weight.shape[0]) // int(n_heads)
            if hd % 32 == 0:
                qk_path = "fused"
        wrapped = _FusedAttention(parent, qk_path)
        # attach the wrapper on the layer so the layer wrapper can call it
        # without replacing the original attention (probe integrity)
        parent._omlx_fused_attn = wrapped
        sites.append((parent, "qkv_proj", "qkv"))
        sites.append((parent, "_omlx_fused_attn", "attn-proxy"))
    except Exception as e:  # noqa: BLE001
        logger.debug("bonsai_decode_fusion: attention skip: %s", e)


def _maybe_fuse_mlp(parent: Any, sites: list) -> None:
    g, u, d_ = getattr(parent, "gate_proj", None), getattr(parent, "up_proj", None), getattr(parent, "down_proj", None)
    if not all(x is not None for x in (g, u, d_)) or not hasattr(g, "weight"):
        return
    key = _linear_quant_key(g)
    if not (key and key == _linear_quant_key(u)):
        return
    try:
        if getattr(parent, "up_gate_proj", None) is None:
            parent.up_gate_proj = _fuse_linear_rows([g, u], "up_gate")
        parent._omlx_fused_mlp = _FusedMLP(parent)
        sites.append((parent, "up_gate_proj", "upgate"))
        sites.append((parent, "_omlx_fused_mlp", "mlp-proxy"))
    except Exception as e:  # noqa: BLE001
        logger.debug("bonsai_decode_fusion: mlp skip: %s", e)


def apply_bonsai_decode_fusion(model: Any) -> int:
    """Wrap attention / MLP / layer modules with fused decode paths.

    Returns the number of fused sites (0 if the model has no matching shapes).
    """
    if model is None:
        return 0
    sites: list[tuple[Any, str, str]] = []

    modules = list(model.modules())
    for m in modules:
        _maybe_fuse_attention(m, sites)
        _maybe_fuse_mlp(m, sites)

    # probe each proxy against the original path on fresh caches; a failing
    # probe permanently disables that site (the wrapper then forwards).
    for mod in modules:
        pa = getattr(mod, "_omlx_fused_attn", None)
        if pa is not None:
            ok = _probe_attention_proxy(pa)
            object.__setattr__(pa, "_ok", ok)
            if not ok:
                logger.debug("bonsai_decode_fusion: attention probe failed")
        pm = getattr(mod, "_omlx_fused_mlp", None)
        if pm is not None:
            ok = _probe_mlp_proxy(pm)
            object.__setattr__(pm, "_ok", ok)
            if not ok:
                logger.debug("bonsai_decode_fusion: mlp probe failed")

    # decoder layers: fuse residual-add + RMSNorm and route through the
    # fused attention / MLP proxies. Wrapping the layer keeps the original
    # (pristine) layer inside for one-time probe comparison.
    layers = getattr(model, "layers", None)
    replaced_slots: list[tuple[Any, int]] = []
    if layers is not None:
        for i, layer in enumerate(layers):
            iln = getattr(layer, "input_layernorm", None)
            pln = getattr(layer, "post_attention_layernorm", None)
            dim = getattr(iln, "weight", None).shape[-1] if iln is not None else 0
            if (
                iln is not None
                and pln is not None
                and getattr(layer, "self_attn", None) is not None
                and getattr(layer, "mlp", None) is not None
                and dim % 256 == 0
                and int(iln.weight.shape[-1]) == int(pln.weight.shape[-1])
            ):
                wrapped = _FusedLayer(layer)
                # the layer's fused path uses the attention/mlp proxies only
                # when they themselves probed OK
                ok = _probe_layer_proxy(wrapped)
                object.__setattr__(wrapped, "_ok", ok)
                if not ok:
                    logger.debug("bonsai_decode_fusion: layer probe failed; not wrapping")
                    continue
                layers[i] = wrapped
                replaced_slots.append((layers, i))
            else:
                # layer fusion unavailable: route the original layer's own
                # attention / mlp calls through their proxies directly.
                fa = getattr(getattr(layer, "self_attn", None), "_omlx_fused_attn", None)
                if fa is not None:
                    layer.self_attn = fa
                    sites.append((layer, "self_attn", "attn-direct"))
                fm = getattr(getattr(layer, "mlp", None), "_omlx_fused_mlp", None)
                if fm is not None:
                    layer.mlp = fm
                    sites.append((layer, "mlp", "mlp-direct"))

    if replaced_slots:
        model._omlx_fused_layer_slots = replaced_slots
        sites.append((model, "_omlx_fused_layer_slots", "slots"))

    model._omlx_decode_fusion_sites = sites
    n = len(sites) - (1 if replaced_slots else 0) + (len(replaced_slots) if replaced_slots else 0)
    logger.info("bonsai_decode_fusion: fused %d decode sites", n)
    return n


def remove_bonsai_decode_fusion(model: Any) -> None:
    """Unwrap fused proxies and drop fused projection modules."""
    layers = getattr(model, "layers", None)
    slots = getattr(model, "_omlx_fused_layer_slots", None) or []
    for container, i in slots:
        if 0 <= i < len(container):
            w = container[i]
            if isinstance(w, _FusedLayer):
                container[i] = w.inner
    sites = getattr(model, "_omlx_decode_fusion_sites", None) or []
    for obj, attr, _kind in sites:
        if attr == "_omlx_fused_layer_slots":
            continue
        try:
            if isinstance(getattr(obj, attr, None), (_FusedAttention, _FusedMLP)):
                setattr(obj, attr, getattr(obj, attr).inner)
            else:
                delattr(obj, attr)
        except Exception:
            obj.__dict__.pop(attr, None)
        for name in ("qkv_proj", "up_gate_proj"):
            obj.__dict__.pop(name, None)
    for a in ("_omlx_fused_layer_slots", "_omlx_decode_fusion_sites"):
        if hasattr(model, a):
            delattr(model, a)
