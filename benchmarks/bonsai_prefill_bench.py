#!/usr/bin/env python3
"""Bonsai t5 prefill GEMM benchmark: dequant+MMA vs paired-trit LUT.

Compares bonsai_t5_qmm (simdgroup MMA) against bonsai_t5_qmm_lut (μ=2
table-dot) at realistic prefill shapes: M ∈ {128, 512, 2048}, K=N=4096,
group_size=128 (plus a gs=64 row).

Usage
-----
    python benchmarks/bonsai_prefill_bench.py [--M 128,512,2048]
                                              [--iters 20] [--warmup 3]
                                              [--dtype fp16] [--csv]

Prints a markdown table with wall-clock per call, tokens/sec (M rows per call
at one token each), and fp16-GFLOP-equivalent throughput
(2*M*N*K ops counted at full precision — the reference works on fp16 tiles).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

# `python benchmarks/x.py` puts benchmarks/ on sys.path, not the repo root:
# insert the repo root so `import omlx...` resolves regardless of invocation.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

try:
    import omlx.custom_kernels.bonsai.fast as bf
    _NATIVE = bf.has_native()
except ImportError as e:
    print(f"warning: omlx bonsai fast import failed: {e}", file=sys.stderr)
    bf = None  # type: ignore[assignment]
    _NATIVE = False

try:
    from tools.repack_ternary_t5 import pack_t5
    _HAS_T5_REPACK = True
except ImportError:
    _HAS_T5_REPACK = False

DTYPE_MAP = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}


def make_t5_tensors(M, N, K, group_size, dtype, seed=0):
    """Random t5-packed weights + scales + activations (shared by both fns)."""
    rng = np.random.default_rng(seed)
    x = mx.array(rng.normal(size=(M, K)).astype(np.float16)).astype(dtype)
    quants = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
    w = mx.array(pack_t5(quants, group_size))
    n_groups = K // group_size
    scales = mx.array(rng.normal(size=(N, n_groups)).astype(np.float16)).astype(dtype)
    return x, w, scales


def time_fn(fn, warmup: int, iters: int) -> float:
    """Mean wall time in seconds over `iters` iterations (compile warmed up)."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters


@dataclass
class Row:
    M: int
    N: int
    K: int
    gs: int
    variant: str
    ms: float
    tok_s: float
    gflops: float
    ratio: str = ""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", default="128,512,2048", help="comma-separated M values")
    ap.add_argument("--K", default="4096")
    ap.add_argument("--N", default="4096")
    ap.add_argument("--gs", default="128,64")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPE_MAP))
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    if bf is None or not _NATIVE:
        print("bonsai native extension unavailable — rebuild with "
              "OMLX_WITH_CUSTOM_KERNEL=1", file=sys.stderr)
        sys.exit(1)
    for sym in ("bonsai_t5_qmm", "bonsai_t5_qmm_lut", "bonsai_t5_qmm_nomul"):
        if not bf.has_symbol(sym):
            print(f"missing native symbol {sym}", file=sys.stderr)
            sys.exit(1)

    dtype = DTYPE_MAP[args.dtype]
    rows: list[Row] = []
    for gs in (int(v) for v in args.gs.split(",")):
        for M in (int(v) for v in args.M.split(",")):
            K = int(args.K); N = int(args.N)
            if K % gs != 0:
                continue
            x, w, scales = make_t5_tensors(M, N, K, gs, dtype)
            mx.eval(x, w, scales)

            results = {}
            for name in ("bonsai_t5_qmm", "bonsai_t5_qmm_lut", "bonsai_t5_qmm_nomul"):
                fn = getattr(bf, name)
                results[name] = time_fn(lambda: fn(x, w, scales), args.warmup, args.iters)

            ms_ref = results["bonsai_t5_qmm"] * 1e3
            ms_lut = results["bonsai_t5_qmm_lut"] * 1e3
            ms_nom = results["bonsai_t5_qmm_nomul"] * 1e3
            tok_s = lambda ms: 1000.0 * M / ms
            gflop = lambda ms: 2.0 * M * N * K / (ms * 1e-3) / 1e9
            ratio_lut = f"{ms_lut / ms_ref:.2f}x" if ms_ref > 0 else ""
            ratio_nom = f"{ms_nom / ms_ref:.2f}x" if ms_ref > 0 else ""
            for name, ms, ratio in (
                ("qmm (MMA)", ms_ref, ""),
                ("qmm_lut (μ2)", ms_lut, ratio_lut),
                ("qmm_nomul (μ1)", ms_nom, ratio_nom),
            ):
                rows.append(Row(M, N, K, gs, name, ms, tok_s(ms), gflop(ms), ratio))

    if args.csv:
        print("M,N,K,gs,variant,ms,tokens_per_sec,gflops,ratio_vs_mma")
        for r in rows:
            print(f"{r.M},{r.N},{r.K},{r.gs},{r.variant},{r.ms:.4f},"
                  f"{r.tok_s:.1f},{r.gflops:.1f},{r.ratio or ''}")
        return

    print(f"| M | N | K | gs | variant | ms/call | tokens/s | fp16 GFLOPS | vs MMA |")
    print(f"|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for r in rows:
        print(f"| {r.M} | {r.N} | {r.K} | {r.gs} | {r.variant} | "
              f"{r.ms:.3f} | {r.tok_s:.0f} | {r.gflops:.0f} | {r.ratio} |")


if __name__ == "__main__":
    main()
