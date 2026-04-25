#!/usr/bin/env python3
"""Benchmark training performance under different configurations.

Runs a short training workload (2 training epochs on MNIST with ergodic model)
and reports wall-clock time.  Use from the Makefile:

    make bench          # run benchmark
    make bench-all      # compare env configurations

Or standalone:

    cd basicmodel
    .venv/bin/python test/bench_training.py
"""

import os
import sys
import time
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

import Models

os.chdir(_project / "bin")

import torch
from util import ProjectPaths

# ---------------------------------------------------------------------------
# Configuration: 3 training epochs keeps the benchmark short but representative
# ---------------------------------------------------------------------------
BENCH_EPOCHS = 3
BENCH_XML = os.path.join(ProjectPaths.DATA_DIR, "ergodic-only.xml")


def bench_training():
    """Run a short training pass and return (elapsed_seconds, accuracy)."""
    # Suppress print noise during benchmark
    import io
    from contextlib import redirect_stdout, redirect_stderr

    cfg = Models.BaseModel.load_config(BENCH_XML)
    train = cfg.get("training", {})
    dataset = train.get("dataset", "xor")
    data = Models.Data()
    data.load(dataset)

    m = Models.BasicModel()
    cfg = m.create_from_config(BENCH_XML, data=data)
    train = cfg["training"]

    # Override to short run
    num_epochs = BENCH_EPOCHS
    batch_size = train["batchSize"]
    lr = train["learningRate"]

    buf = io.StringIO()
    t0 = time.monotonic()
    with redirect_stdout(buf), redirect_stderr(buf):
        m.run(1, num_epochs, batch_size, lr=lr)
    elapsed = time.monotonic() - t0

    # Extract final accuracy from captured output
    lines = buf.getvalue().splitlines()
    accuracy = None
    for line in reversed(lines):
        if "Test Accuracy:" in line:
            accuracy = line.strip()
            break

    return elapsed, accuracy


def main():
    print(f"Device:     {Models.TheDevice}")
    print(f"Config:     {os.path.basename(BENCH_XML)}")
    print(f"Epochs:     {BENCH_EPOCHS}")
    print(f"PyTorch:    {torch.__version__}")
    print(f"Threads:    {torch.get_num_threads()}")
    omp = os.environ.get("OMP_NUM_THREADS", "(unset)")
    kmp = os.environ.get("KMP_DUPLICATE_LIB_OK", "(unset)")
    dyld = os.environ.get("DYLD_LIBRARY_PATH", "(unset)")
    print(f"OMP_NUM_THREADS:      {omp}")
    print(f"KMP_DUPLICATE_LIB_OK: {kmp}")
    print(f"DYLD_LIBRARY_PATH:    {dyld}")
    print("-" * 50)

    elapsed, accuracy = bench_training()

    print(f"Time:       {elapsed:.1f}s")
    print(f"Per-epoch:  {elapsed / BENCH_EPOCHS:.1f}s")
    print(f"Last:       {accuracy}")


if __name__ == "__main__":
    main()
