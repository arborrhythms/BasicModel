#!/usr/bin/env python3
"""Benchmark codebook lookup as a function of vocabulary size.

The IR-mode parallel forward path is dominated by PerceptualSpace
codebook quantization (nearest-neighbor over the lexicon). MM_20M
uses a 4096-entry byte codebook; MM_5M_IR uses a 1,000,000-entry
word lexicon. The ratio shows up directly in `(B*N, V, D)` distance
computation cost.

Run on the CUDA training box to confirm the bottleneck transfers,
and to measure how much an FAISS / HNSW / two-stage prefix lookup
would buy.

Usage:
    cd basicmodel
    .venv/bin/python test/bench_codebook_lookup.py
    .venv/bin/python test/bench_codebook_lookup.py --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))


def bench_distance(B, N, V, D, device, n_iter=10, warmup=2):
    """Time the brute-force `(B*N, V) = ||x - codebook||^2` distance."""
    x = torch.randn(B * N, D, device=device)
    cb = torch.randn(V, D, device=device)
    # Warmup
    for _ in range(warmup):
        d = (x.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)
        if device == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        d = (x.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)
        idx = d.argmin(dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed, d.shape


def bench_matmul_distance(B, N, V, D, device, n_iter=10, warmup=2):
    """Same lookup via matmul shortcut: ||x||^2 + ||cb||^2 - 2*x@cb.T.

    Avoids the (B*N, V, D) outer-product allocation; only the
    (B*N, V) score tensor is materialized.
    """
    x = torch.randn(B * N, D, device=device)
    cb = torch.randn(V, D, device=device)
    cb_norm = (cb * cb).sum(-1)
    for _ in range(warmup):
        x_norm = (x * x).sum(-1, keepdim=True)
        d = x_norm + cb_norm.unsqueeze(0) - 2.0 * x @ cb.T
        if device == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        x_norm = (x * x).sum(-1, keepdim=True)
        d = x_norm + cb_norm.unsqueeze(0) - 2.0 * x @ cb.T
        idx = d.argmin(dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed, d.shape


def bench_pole_aligned(B, N, V, D, device, n_iter=10, warmup=2):
    """Antipode-aware matmul score: ``(a @ b.T).abs()``.

    Drop-in for *bivector-encoded* lookup sites (NEG ⇔ antipode).
    NOT a drop-in for lexicon / token codebooks (where each row is
    an independent vector); this bench measures the kernel cost of
    the matmul shortcut as a reference number.
    """
    x = torch.randn(B * N, D, device=device)
    cb = torch.randn(V, D, device=device)
    for _ in range(warmup):
        s = (x @ cb.T).abs()
        if device == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        s = (x @ cb.T).abs()
        idx = s.argmax(dim=-1)
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed, s.shape


def bench_wrapped_chunked(B, N, V, D, device, n_iter=10, warmup=2,
                          chunk=8192):
    """Torus-distance lookup with chunked V to bound peak memory.

    PerceptualSpace uses ``_wrapped_mse_score`` (torus distance with
    ``remainder(x-y+1, 2)-1``). The wrap precludes the matmul
    shortcut, but chunking the V axis bounds the intermediate to
    ``(B*N, chunk, D)`` regardless of full V -- which is what blows
    up the broadcast form at V=1M.
    """
    x = torch.randn(B * N, D, device=device)
    cb = torch.randn(V, D, device=device)
    def _wrap(t):
        return torch.remainder(t + 1.0, 2.0) - 1.0
    def _step():
        scores = torch.empty(B * N, V, device=device)
        for s in range(0, V, chunk):
            e = min(s + chunk, V)
            delta = _wrap(x.unsqueeze(1) - cb[s:e].unsqueeze(0))
            scores[:, s:e] = -delta.square().mean(dim=-1)
        return scores.argmax(dim=-1)
    for _ in range(warmup):
        _step()
        if device == "cuda":
            torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _step()
        if device == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_iter
    return elapsed, (B * N, V)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu",
                   choices=("cpu", "cuda", "mps"))
    p.add_argument("--B", type=int, default=128, help="batch size")
    p.add_argument("--N", type=int, default=32,
                   help="positions per row (sentence length)")
    p.add_argument("--D", type=int, default=6,
                   help="embedding dim (PerceptualSpace.nDim)")
    p.add_argument("--n-iter", type=int, default=10)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: --device cuda but CUDA not available; falling back to cpu")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("WARN: --device mps but MPS not available; falling back to cpu")
        args.device = "cpu"

    print(f"device={args.device} B={args.B} N={args.N} D={args.D} "
          f"n_iter={args.n_iter}")
    print(f"{'V':>10} {'bcast (s)':>10} {'matmul (s)':>11} "
          f"{'pole (s)':>10} {'wrap-chunk (s)':>15} {'mm spd':>8} {'pl spd':>8} {'wc spd':>8}")
    print("-" * 90)
    for V in (4096, 65536, 262144, 1_000_000):
        try:
            t_bcast, _ = bench_distance(args.B, args.N, V, args.D,
                                         args.device, n_iter=args.n_iter)
        except RuntimeError as e:
            t_bcast = float("nan")
            print(f"{V:>10}  broadcast OOM/error: {e}")
            continue
        try:
            t_mm, _ = bench_matmul_distance(args.B, args.N, V, args.D,
                                             args.device, n_iter=args.n_iter)
        except RuntimeError as e:
            t_mm = float("nan")
        try:
            t_pl, _ = bench_pole_aligned(args.B, args.N, V, args.D,
                                          args.device, n_iter=args.n_iter)
        except RuntimeError as e:
            t_pl = float("nan")
        try:
            t_wc, _ = bench_wrapped_chunked(args.B, args.N, V, args.D,
                                             args.device, n_iter=args.n_iter)
        except RuntimeError as e:
            t_wc = float("nan")
        r_mm = (t_bcast / t_mm) if (t_mm and t_mm == t_mm) else float("nan")
        r_pl = (t_bcast / t_pl) if (t_pl and t_pl == t_pl) else float("nan")
        r_wc = (t_bcast / t_wc) if (t_wc and t_wc == t_wc) else float("nan")
        print(f"{V:>10} {t_bcast:>10.4f} {t_mm:>11.4f} {t_pl:>10.4f} "
              f"{t_wc:>15.4f} {r_mm:>7.1f}x {r_pl:>7.1f}x {r_wc:>7.1f}x")

    print()
    print("Columns:")
    print("  bcast      -- (B*N, V, D) outer-product form currently used by")
    print("                Spaces.py:647 (Basis._snap_content) and")
    print("                Spaces.py:821 / 2014 via _wrapped_mse_score.")
    print("                Allocates a V*B*N*D float intermediate.")
    print("  matmul     -- ||x-cb||^2 = ||x|| + ||cb|| - 2*x @ cb.T form.")
    print("                Euclidean only (does not match torus wrap).")
    print("  pole       -- _pole_aligned_score: (a @ b.T).abs(). Antipode-")
    print("                aware; for *bivector-encoded* lookup only (NEG")
    print("                ⇔ antipode). NOT correct for lexicon / token")
    print("                codebooks where each row is independent.")
    print("  wrap-chunk -- torus-distance form (matches _wrapped_mse_score")
    print("                semantics) with V-axis chunked to bound peak mem.")
    print("                Same semantics as bcast, V*chunk peak instead of")
    print("                V*B*N peak. Drop-in for the lexicon hot path.")


if __name__ == "__main__":
    main()
