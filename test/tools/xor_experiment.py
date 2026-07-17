#!/usr/bin/env python3
"""XOR convergence experiment: explore alpha schedules * LR * ergodic.

Goal: drive both output and reconstruction losses to zero.

Sweeps:
  - Alpha schedules: none (alpha=0), fast linear, exponential decay, cosine
  - Learning rates: 0.01, 0.005, 0.001
  - Ergodic: on/off

Writes results to output/xor_experiment_results.txt and prints a summary table.
"""

import os, sys, math, copy, time, io
from contextlib import redirect_stdout

_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

# ---------------------------------------------------------------------------
# Alpha schedule factories
# ---------------------------------------------------------------------------

def alpha_none(epoch, numEpochs):
    """No annealing -- pure exploitation from the start."""
    return 0.0

def alpha_linear(epoch, numEpochs):
    """Original linear 1->0."""
    return 1.0 - epoch / max(1, numEpochs - 1)

def alpha_fast_linear(epoch, numEpochs):
    """Fast linear: reaches 0 at 10% of training."""
    warmup = max(1, numEpochs // 10)
    return max(0.0, 1.0 - epoch / warmup)

def alpha_very_fast_linear(epoch, numEpochs):
    """Very fast linear: reaches 0 at 5% of training."""
    warmup = max(1, numEpochs // 20)
    return max(0.0, 1.0 - epoch / warmup)

def alpha_exponential(epoch, numEpochs):
    """Exponential decay with half-life = 5% of training."""
    half_life = max(1, numEpochs // 20)
    return math.exp(-0.693 * epoch / half_life)

def alpha_cosine(epoch, numEpochs):
    """Cosine annealing to 0 over first 20% of training."""
    warmup = max(1, numEpochs // 5)
    if epoch >= warmup:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * epoch / warmup))

ALPHA_SCHEDULES = {
    "none":       alpha_none,
    "fast":       alpha_fast_linear,
    "very_fast":  alpha_very_fast_linear,
    "exp":        alpha_exponential,
    "cosine":     alpha_cosine,
    "linear":     alpha_linear,
}

# ---------------------------------------------------------------------------
# Patched run method
# ---------------------------------------------------------------------------

def run_with_schedule(model, numEpochs, batchSize, lr, alpha_fn):
    """Run training with a custom alpha schedule, return final losses and accuracy."""
    trainLosses = [[], []]
    testLosses  = [[], []]
    accuracy    = []
    best_out    = float('inf')
    best_recon  = float('inf')
    ping_pong_count = 0
    prev_out    = None
    optimizer   = model.getOptimizer(lr)

    for epoch in range(numEpochs):
        alpha = alpha_fn(epoch, numEpochs)
        model.set_sigma(alpha)

        if epoch != 0:
            outErr, inErr, allOut, lastIn = model.runEpoch(
                optimizer=optimizer, batchSize=batchSize, split="train")
            trainLosses[0].append(outErr)
            trainLosses[1].append(inErr)

            if outErr < best_out:
                best_out = outErr
            if inErr < best_recon:
                best_recon = inErr

            # Detect ping-pong: loss increases by >50% from previous
            if prev_out is not None and outErr > prev_out * 1.5 and prev_out > 0.001:
                ping_pong_count += 1
            prev_out = outErr

        # Test evaluation
        outErr_t, inErr_t, allOut, lastIn = model.runEpoch(
            batchSize=batchSize, split="test")
        testLosses[0].append(outErr_t)
        testLosses[1].append(inErr_t)

        if allOut.dim() == 1:
            predicted = (allOut > 0.5).long()
            actual = (model.outputSpace.getTestOutput().squeeze() > 0.5).long()
        else:
            _, predicted = torch.max(allOut, 1)
            _, actual = torch.max(model.outputSpace.getTestOutput(), 1)
        total = predicted.size(0)
        correct = (predicted == actual).sum().item()
        accuracy.append(correct / total)

        model.inputSpace.shuffle()

        # Early stop if both losses tiny
        if len(trainLosses[0]) > 0 and trainLosses[0][-1] < 1e-6 and trainLosses[1][-1] < 1e-6:
            break

    final_out   = trainLosses[0][-1] if trainLosses[0] else float('inf')
    final_recon = trainLosses[1][-1] if trainLosses[1] else float('inf')
    final_acc   = accuracy[-1] if accuracy else 0.0

    return {
        "final_out":    final_out,
        "final_recon":  final_recon,
        "best_out":     best_out,
        "best_recon":   best_recon,
        "final_acc":    final_acc,
        "ping_pong":    ping_pong_count,
        "epochs_run":   len(trainLosses[0]),
        "trainLosses":  trainLosses,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

import tempfile, xml.etree.ElementTree as ET

def create_model(ergodic=False):
    """Create a fresh XOR model from XOR_exact.xml with ergodic override.

    Uses create_from_config to handle LM model type dimension setup properly.
    Writes a temp XML with the ergodic flag patched.
    """
    xml_path = os.path.join(_PROJECT, "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Patch ergodic flag
    erg_elem = root.find("architecture/ergodic")
    if erg_elem is None:
        erg_elem = ET.SubElement(root.find("architecture"), "ergodic")
    erg_elem.text = "true" if ergodic else "false"

    # Ensure autoload=false so we start fresh
    auto_elem = root.find("architecture/autoload")
    if auto_elem is None:
        auto_elem = ET.SubElement(root.find("architecture"), "autoload")
    auto_elem.text = "false"

    # Write temp XML
    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True)
    tmp.close()

    Models.TheData.load("xor")
    m = Models.BasicModel()
    m.create_from_config(tmp.name, data=Models.TheData)
    os.unlink(tmp.name)
    return m


def run_experiment():
    """Run the full sweep and print results."""
    NUM_EPOCHS = 500
    BATCH_SIZE = 10

    lrs = [0.01, 0.005, 0.001]
    schedules = ["none", "very_fast", "fast", "exp", "cosine", "linear"]
    ergodic_opts = [False, True]

    results = []
    total = len(lrs) * len(schedules) * len(ergodic_opts)
    run_idx = 0

    for erg in ergodic_opts:
        for lr in lrs:
            for sched_name in schedules:
                run_idx += 1
                label = f"erg={erg} lr={lr} alpha={sched_name}"
                print(f"\n{'='*60}")
                print(f"[{run_idx}/{total}] {label}")
                print(f"{'='*60}")

                m = create_model(ergodic=erg)
                alpha_fn = ALPHA_SCHEDULES[sched_name]

                t0 = time.time()
                # Suppress per-epoch output
                buf = io.StringIO()
                with redirect_stdout(buf):
                    r = run_with_schedule(m, NUM_EPOCHS, BATCH_SIZE, lr, alpha_fn)
                elapsed = time.time() - t0

                r["label"] = label
                r["ergodic"] = erg
                r["lr"] = lr
                r["schedule"] = sched_name
                r["time"] = elapsed
                results.append(r)

                print(f"  final: out={r['final_out']:.6f}  recon={r['final_recon']:.6f}  "
                      f"acc={r['final_acc']*100:.1f}%  ping_pong={r['ping_pong']}  "
                      f"best_out={r['best_out']:.6f}  best_recon={r['best_recon']:.6f}  "
                      f"({elapsed:.1f}s)")

    # Summary table
    print("\n" + "="*100)
    print(f"{'Config':<40} {'Out':>10} {'Recon':>10} {'BestOut':>10} {'BestRec':>10} {'Acc':>6} {'PP':>4}")
    print("-"*100)

    # Sort by final_out + final_recon
    results.sort(key=lambda r: r["final_out"] + r["final_recon"])

    for r in results:
        print(f"{r['label']:<40} {r['final_out']:>10.6f} {r['final_recon']:>10.6f} "
              f"{r['best_out']:>10.6f} {r['best_recon']:>10.6f} "
              f"{r['final_acc']*100:>5.1f}% {r['ping_pong']:>4d}")

    # Write to file
    out_dir = os.path.join(_PROJECT, "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "xor_experiment_results.txt")
    with open(out_path, "w") as f:
        f.write(f"{'Config':<40} {'Out':>10} {'Recon':>10} {'BestOut':>10} {'BestRec':>10} {'Acc':>6} {'PP':>4}\n")
        f.write("-"*100 + "\n")
        for r in results:
            f.write(f"{r['label']:<40} {r['final_out']:>10.6f} {r['final_recon']:>10.6f} "
                    f"{r['best_out']:>10.6f} {r['best_recon']:>10.6f} "
                    f"{r['final_acc']*100:>5.1f}% {r['ping_pong']:>4d}\n")
    print(f"\nResults saved to {out_path}")

    # Highlight best
    best = results[0]
    print(f"\n*** BEST: {best['label']}")
    print(f"    final output={best['final_out']:.6f}  recon={best['final_recon']:.6f}  acc={best['final_acc']*100:.1f}%")


if __name__ == "__main__":
    run_experiment()
