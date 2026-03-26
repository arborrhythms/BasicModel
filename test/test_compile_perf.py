#!/usr/bin/env python3
"""Benchmark torch.compile backends against eager execution.

Runs a small model doing large matrix operations (matmul, softmax, layer-norm)
through multiple forward+backward passes, comparing:
  - eager (no compilation)
  - util.compile() default backend
  - torch.compile(backend='eager')   (dynamo tracing, no codegen)

Usage:
    cd basicmodel
    .venv/bin/python test/test_compile_perf.py
"""

import os
import sys
import time
import copy

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import compile as util_compile, TheDevice


class MatMulModel(nn.Module):
    """Model that chains large matrix ops: linear → softmax → linear → layernorm."""
    def __init__(self, dim=512, depth=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.gelu(x)
        x = self.norm(x)
        x = F.softmax(x, dim=-1)
        return x


def bench(model, x, target, n_iters=50, warmup=5, label=""):
    """Time n_iters of forward+backward+step, return median ms/iter."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    times = []
    for i in range(warmup + n_iters):
        if TheDevice.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        y = model(x)
        loss = F.mse_loss(y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if TheDevice.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    best = times[0]
    worst = times[-1]
    print(f"  {label:30s}  median={median:7.2f}ms  mean={mean:7.2f}ms  "
          f"best={best:7.2f}ms  worst={worst:7.2f}ms")
    return median


def main():
    print(f"Device: {TheDevice}")
    print(f"PyTorch: {torch.__version__}")

    dim = 512
    batch = 64
    x = torch.randn(batch, dim, device=TheDevice.get())
    target = torch.randn(batch, dim, device=TheDevice.get())

    # --- Eager baseline ---
    model_eager = MatMulModel(dim=dim).to(TheDevice.get())
    torch.manual_seed(42)
    # Save initial weights so all variants start from same state
    init_state = copy.deepcopy(model_eager.state_dict())

    print(f"\nBenchmark: {batch}x{dim} matmul model, forward+backward+step")
    print("-" * 80)

    eager_ms = bench(model_eager, x, target, label="eager (no compile)")

    # --- util.compile() default ---
    model_default = MatMulModel(dim=dim).to(TheDevice.get())
    model_default.load_state_dict(init_state)
    torch._dynamo.reset()
    model_default = util_compile(model_default, verbose=True)
    default_ms = bench(model_default, x, target, label="util.compile (default)")

    # --- torch.compile(backend='eager') ---
    model_dynamo_eager = MatMulModel(dim=dim).to(TheDevice.get())
    model_dynamo_eager.load_state_dict(init_state)
    torch._dynamo.reset()
    try:
        model_dynamo_eager = torch.compile(model_dynamo_eager, backend='eager')
        print("  [compiled with backend='eager']")
        dynamo_eager_ms = bench(model_dynamo_eager, x, target, label="torch.compile(eager)")
    except Exception as e:
        print(f"  torch.compile(backend='eager') failed: {e}")
        dynamo_eager_ms = None

    # --- Summary ---
    print("-" * 80)
    print("Summary (vs eager baseline):")
    print(f"  eager:                {eager_ms:7.2f} ms  (baseline)")
    speedup = eager_ms / default_ms if default_ms else 0
    print(f"  util.compile:         {default_ms:7.2f} ms  ({speedup:.2f}x)")
    if dynamo_eager_ms:
        speedup = eager_ms / dynamo_eager_ms
        print(f"  compile(eager):       {dynamo_eager_ms:7.2f} ms  ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
