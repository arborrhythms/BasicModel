#!/usr/bin/env python3
"""Bench torus vs unit-ball Lexicon lookup paths.

Compares forward-only top-k lookup wall-clock at the project's
working shapes (V up to 10^6, D = 6, B = 128). Four families:

  * **torus**         -- legacy ``Lexicon.distance_sq`` broadcast form
                         (wrap + subtract + square + sum). Topk over the
                         full ``[B, V]`` distance matrix.
  * **torus cplx**    -- complex-log-domain encoding ``W_c = exp(i*pi*W)``,
                         then chord-squared distance via one complex
                         matmul + ``.real`` + topk. Argmin-equivalent to
                         the wrap-distance form; the wrap is built into
                         the unit-circle representation. Codebook
                         lifted once per opt step (paired with the
                         ball variants' precomputed ``W_norm2``);
                         query lifted per step.
  * **ball L2**       -- ``Lexicon.topk_l2`` / ``topk_l2_chunked``.
                         Plain unit-ball Euclidean L2; one real matmul
                         + bias subtract.
  * **ball RP**       -- ``Lexicon.topk_rp`` / ``topk_rp_chunked``.
                         Projective unit ball ``RP^D``; one real matmul
                         + abs + bias subtract.

Run on CUDA for representative numbers; CPU works as a smoke test.

Usage:
    .venv/bin/python test/bench_lexicon_paths.py
    .venv/bin/python test/bench_lexicon_paths.py --device cuda
    .venv/bin/python test/bench_lexicon_paths.py --B 64 --D 6 --k 8
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Layers import Lexicon, topk_l2_chunked, topk_rp_chunked


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _time(fn, *, n_iter, warmup, device):
    for _ in range(warmup):
        fn()
        _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
        _sync(device)
    return (time.perf_counter() - t0) / n_iter


def bench_torus_broadcast(x, W, k):
    # Legacy wrapped-MSE form. distance_sq broadcasts to [B, V, D]
    # internally; this is the actual cost path the codebase used to take.
    def step():
        d = Lexicon.distance_sq(x.unsqueeze(1), W.unsqueeze(0))
        return d.topk(k, dim=-1, largest=False).indices
    return step


def bench_torus_chunked(x, W, k, *, chunk_size):
    # Chunked wrap-distance lookup; what we'd write to bound memory.
    def step():
        B = x.shape[0]
        V = W.shape[0]
        best = torch.full((B, k), float("inf"), device=x.device, dtype=x.dtype)
        idx = torch.zeros((B, k), device=x.device, dtype=torch.long)
        for s in range(0, V, chunk_size):
            e = min(s + chunk_size, V)
            d = Lexicon.distance_sq(x.unsqueeze(1), W[s:e].unsqueeze(0))
            local_k = min(k, e - s)
            d_k, i_k = torch.topk(d, k=local_k, dim=-1, largest=False)
            i_k = i_k + s
            merged_d = torch.cat([best, d_k], dim=-1)
            merged_i = torch.cat([idx, i_k], dim=-1)
            best, pos = torch.topk(merged_d, k=k, dim=-1, largest=False)
            idx = merged_i.gather(-1, pos)
        return idx
    return step


def bench_torus_complex(x_real, W_c, k):
    # Complex-encoding torus distance: argmin chord-squared
    # ||tilde a - tilde W_i||^2 == argmax Re<tilde a, tilde W_i>.
    # Wrap is absorbed into the unit-circle representation.
    pi = math.pi

    def step():
        x_c = torch.exp(1j * pi * x_real.to(W_c.dtype))     # [B, D] complex
        scores = (x_c @ W_c.conj().T).real                  # [B, V] real
        return scores.topk(k, dim=-1, largest=True).indices
    return step


def bench_torus_complex_chunked(x_real, W_c, k, *, chunk_size):
    pi = math.pi

    def step():
        V = W_c.shape[0]
        x_c = torch.exp(1j * pi * x_real.to(W_c.dtype))
        best_scores = None
        best_idx = None
        for s in range(0, V, chunk_size):
            e = min(s + chunk_size, V)
            scores_c = (x_c @ W_c[s:e].conj().T).real        # [B, Vc]
            local_k = min(k, e - s)
            sc_k, ix_k = torch.topk(
                scores_c, k=local_k, dim=-1, largest=True)
            ix_k = ix_k + s
            if best_scores is None:
                best_scores = sc_k
                best_idx = ix_k
            else:
                merged_s = torch.cat([best_scores, sc_k], dim=-1)
                merged_i = torch.cat([best_idx, ix_k], dim=-1)
                best_scores, pos = torch.topk(
                    merged_s, k=k, dim=-1, largest=True)
                best_idx = merged_i.gather(-1, pos)
        return best_idx
    return step


def bench_ball_l2(x, W, W_norm2, k):
    def step():
        return Lexicon.topk_l2(x, W, W_norm2, k=k)
    return step


def bench_ball_l2_chunked(x, W, W_norm2, k, *, chunk_size):
    def step():
        return topk_l2_chunked(x, W, W_norm2, k=k, chunk_size=chunk_size)
    return step


def bench_ball_rp(x, W, W_norm2, k):
    def step():
        return Lexicon.topk_rp(x, W, W_norm2, k=k)
    return step


def bench_ball_rp_chunked(x, W, W_norm2, k, *, chunk_size):
    def step():
        return topk_rp_chunked(x, W, W_norm2, k=k, chunk_size=chunk_size)
    return step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu",
                   choices=("cpu", "cuda", "mps"))
    p.add_argument("--B", type=int, default=128)
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--n-iter", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=131_072)
    p.add_argument("--V-list", type=int, nargs="+",
                   default=[4096, 65536, 262144, 1_000_000])
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARN: cuda unavailable; falling back to cpu")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("WARN: mps unavailable; falling back to cpu")
        args.device = "cpu"

    device = args.device
    print(f"device={device} B={args.B} D={args.D} k={args.k} "
          f"n_iter={args.n_iter} warmup={args.warmup} "
          f"chunk_size={args.chunk_size}")
    print()

    header = (
        f"{'V':>10}  "
        f"{'torus bcast':>12}  {'torus chunk':>12}  "
        f"{'torus cplx':>11}  {'cplx chunk':>11}  "
        f"{'ball L2':>9}  {'ball L2 ch':>11}  "
        f"{'ball RP':>9}  {'ball RP ch':>11}    "
        f"{'cplx/tor':>9}  {'L2/tor':>8}  {'RP/tor':>8}"
    )
    print(header)
    print("-" * len(header))

    for V in args.V_list:
        # Real codebook: random rows projected into the unit ball, then
        # also wrap-canonical -- so both torus and ball forms see legal
        # inputs.
        torch.manual_seed(0)
        W = torch.rand(V, args.D, device=device) * 2 - 1   # [-1, 1)
        W_ball = Lexicon.project_unit_ball(W)
        W_norm2 = W_ball.square().sum(dim=-1).contiguous()
        # Complex-log lift of the (torus) codebook -- precomputed once
        # per "opt step", paired with the ball variants' W_norm2.
        W_complex = torch.exp(
            1j * math.pi * W.to(torch.complex64)
        ).contiguous()
        x = torch.rand(args.B, args.D, device=device) * 2 - 1
        x_ball = Lexicon.project_unit_ball(x)

        results = {}
        for name, fn in [
            ("torus_bcast",
                bench_torus_broadcast(x, W, args.k)),
            ("torus_chunk",
                bench_torus_chunked(x, W, args.k,
                                    chunk_size=args.chunk_size)),
            ("torus_cplx",
                bench_torus_complex(x, W_complex, args.k)),
            ("torus_cplx_ch",
                bench_torus_complex_chunked(
                    x, W_complex, args.k, chunk_size=args.chunk_size)),
            ("ball_l2",
                bench_ball_l2(x_ball, W_ball, W_norm2, args.k)),
            ("ball_l2_ch",
                bench_ball_l2_chunked(x_ball, W_ball, W_norm2, args.k,
                                      chunk_size=args.chunk_size)),
            ("ball_rp",
                bench_ball_rp(x_ball, W_ball, W_norm2, args.k)),
            ("ball_rp_ch",
                bench_ball_rp_chunked(x_ball, W_ball, W_norm2, args.k,
                                      chunk_size=args.chunk_size)),
        ]:
            try:
                t = _time(fn, n_iter=args.n_iter, warmup=args.warmup,
                          device=device)
            except RuntimeError as exc:
                t = float("nan")
                print(f"  {name} OOM / error at V={V}: {exc}")
            results[name] = t

        t_torus = results["torus_bcast"]
        def _spd(name):
            v = results.get(name, float("nan"))
            return (t_torus / v) if (v == v and v > 0) else float("nan")
        cplx_speedup = _spd("torus_cplx")
        l2_speedup = _spd("ball_l2")
        rp_speedup = _spd("ball_rp")

        print(
            f"{V:>10}  "
            f"{results['torus_bcast']:>11.4f}s  "
            f"{results['torus_chunk']:>11.4f}s  "
            f"{results['torus_cplx']:>10.4f}s  "
            f"{results['torus_cplx_ch']:>10.4f}s  "
            f"{results['ball_l2']:>8.4f}s  "
            f"{results['ball_l2_ch']:>10.4f}s  "
            f"{results['ball_rp']:>8.4f}s  "
            f"{results['ball_rp_ch']:>10.4f}s    "
            f"{cplx_speedup:>8.1f}x  {l2_speedup:>7.1f}x  {rp_speedup:>7.1f}x"
        )

    print()
    print("Columns:")
    print("  torus bcast    -- Lexicon.distance_sq broadcast form, full [B, V] topk.")
    print("                    Wrap + subtract + square + sum + topk.")
    print("  torus chunk    -- Same, chunked over V to bound peak memory.")
    print("  torus cplx     -- Complex-log encoding W_c = exp(i*pi*W);")
    print("                    chord-distance via Re(<x_c, W_c>). One")
    print("                    complex matmul + .real + topk. Codebook lift")
    print("                    precomputed (once per opt step); query lifted")
    print("                    per step. Argmin-equivalent to torus distance;")
    print("                    wrap is implicit in the unit-circle encoding.")
    print("  cplx chunk     -- Same, chunked over V.")
    print("  ball L2        -- Lexicon.topk_l2 (plain unit-ball L2 matmul).")
    print("  ball L2 ch     -- topk_l2_chunked.")
    print("  ball RP        -- Lexicon.topk_rp (projective unit ball matmul + abs).")
    print("  ball RP ch     -- topk_rp_chunked.")
    print("  cplx/tor       -- speedup of complex-encoding torus over torus broadcast.")
    print("  L2/tor         -- speedup of plain ball L2 over torus broadcast.")
    print("  RP/tor         -- speedup of projective ball over torus broadcast.")


if __name__ == "__main__":
    main()
