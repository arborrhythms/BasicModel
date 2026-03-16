#!/usr/bin/env python3
"""XOR convergence experiment round 2: focused on the dual-optimizer ping-pong.

Key insight from round 1: alpha has zero effect in non-ergodic mode (layers
hardcode bias=1, temp=0). Ergodic mode destabilizes completely. The bottleneck
is the separate forward/reverse optimizers fighting over shared parameters.

This round tests:
  A) Combined loss: single optimizer, loss = output_loss + λ * recon_loss
  B) Asymmetric LR: forward optimizer at higher LR, reverse at lower LR
  C) LR decay via scheduler
  D) Gradient clipping
  E) Ergodic with combined loss (alpha actually matters when ergodic=True)
"""

import os, sys, math, copy, time, io
from contextlib import redirect_stdout

_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from BasicModel import BasicModel, BasicModelFactory, BaseModel, TheData
import tempfile, xml.etree.ElementTree as ET


def create_model(ergodic=False):
    """Create a fresh XOR model from XOR_exact.xml with ergodic override."""
    xml_path = os.path.join(_PROJECT, "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    erg_elem = root.find("architecture/ergodic")
    if erg_elem is None:
        erg_elem = ET.SubElement(root.find("architecture"), "ergodic")
    erg_elem.text = "true" if ergodic else "false"
    auto_elem = root.find("architecture/autoload")
    if auto_elem is None:
        auto_elem = ET.SubElement(root.find("architecture"), "autoload")
    auto_elem.text = "false"

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True)
    tmp.close()
    TheData.load("xor")
    m = BasicModel()
    m.create_from_config(tmp.name, data=TheData)
    os.unlink(tmp.name)
    return m


# ---------------------------------------------------------------------------
# Alpha schedules (only matter for ergodic=True)
# ---------------------------------------------------------------------------

def alpha_none(epoch, N): return 0.0
def alpha_very_fast(epoch, N):
    warmup = max(1, N // 20)
    return max(0.0, 1.0 - epoch / warmup)
def alpha_exp(epoch, N):
    half_life = max(1, N // 20)
    return math.exp(-0.693 * epoch / half_life)


# ---------------------------------------------------------------------------
# Custom training loops
# ---------------------------------------------------------------------------

def run_combined_loss(model, numEpochs, batchSize, lr, recon_weight=1.0,
                      alpha_fn=alpha_none, grad_clip=0.0, lr_decay=1.0):
    """Single optimizer: loss = output_loss + recon_weight * reconstruction_loss.

    This eliminates the dual-optimizer ping-pong by making both losses cooperate.
    """
    trainLosses = [[], []]
    accuracy = []
    ping_pong = 0
    prev_out = None
    best_out = float('inf')
    best_recon = float('inf')

    optimizer = model.getOptimizer(lr=lr)
    if lr_decay < 1.0:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    else:
        scheduler = None

    criterionOutput, criterionInput = model._getLossFn()

    for epoch in range(numEpochs):
        alpha = alpha_fn(epoch, numEpochs)
        model.setAlpha(alpha)

        if epoch == 0:
            # Eval only on first epoch
            model.train(False)
            test_input, test_output = model.inputSpace.getTestData()
            with torch.no_grad():
                for i in range(0, len(test_input), batchSize):
                    ib = test_input[i:i+batchSize]
                    ob = test_output[i:i+batchSize]
                    it = model.inputSpace.prepInput(ib)
                    ot = model.outputSpace.prepOutput(ob)
                    pred, end = model.forward(it)
            continue

        # Training
        model.train(True)
        train_input, train_output = model.inputSpace.getTrainData()
        nBatches = (len(train_input) + batchSize - 1) // batchSize
        epoch_out = 0.0
        epoch_recon = 0.0

        for i in range(0, len(train_input), batchSize):
            ib = train_input[i:i+batchSize]
            ob = train_output[i:i+batchSize]
            it = model.inputSpace.prepInput(ib)
            ot = model.outputSpace.prepOutput(ob)

            optimizer.zero_grad()

            # Forward
            outputPred, end_state = model.forward(it)
            lossOut = criterionOutput(outputPred.squeeze(), ot.squeeze())

            # Reverse (reconstruction)
            if model.reversible:
                reconstructed, start_state = model.reverse(end_state)
                lossIn = criterionInput(start_state, end_state.detach())
                total_loss = lossOut + recon_weight * lossIn
            else:
                lossIn = torch.tensor(0.0)
                total_loss = lossOut

            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if model.ergodic:
                model.paramUpdate()
            optimizer.step()

            epoch_out = lossOut.item()
            epoch_recon = lossIn.item() if model.reversible else 0.0

        trainLosses[0].append(epoch_out)
        trainLosses[1].append(epoch_recon)
        if epoch_out < best_out: best_out = epoch_out
        if epoch_recon < best_recon: best_recon = epoch_recon

        if prev_out is not None and epoch_out > prev_out * 1.5 and prev_out > 0.001:
            ping_pong += 1
        prev_out = epoch_out

        if scheduler:
            scheduler.step()

        # Test accuracy every 50 epochs
        if epoch % 50 == 0 or epoch == numEpochs - 1:
            model.train(False)
            test_input, test_output = model.inputSpace.getTestData()
            with torch.no_grad():
                allOut = []
                for i in range(0, len(test_input), batchSize):
                    ib = test_input[i:i+batchSize]
                    ob = test_output[i:i+batchSize]
                    it = model.inputSpace.prepInput(ib)
                    pred, _ = model.forward(it)
                    allOut.append(pred.squeeze())
                allOut = torch.cat(allOut, dim=0)
                predicted = (allOut > 0.5).long()
                actual = (model.outputSpace.getTestOutput().squeeze() > 0.5).long()
                acc = (predicted == actual).sum().item() / predicted.size(0)
                accuracy.append(acc)

        model.inputSpace.shuffle()

        # Early exit
        if epoch_out < 1e-6 and epoch_recon < 1e-6:
            break

    final_out = trainLosses[0][-1] if trainLosses[0] else float('inf')
    final_recon = trainLosses[1][-1] if trainLosses[1] else float('inf')
    final_acc = accuracy[-1] if accuracy else 0.0

    return {
        "final_out": final_out, "final_recon": final_recon,
        "best_out": best_out, "best_recon": best_recon,
        "final_acc": final_acc, "ping_pong": ping_pong,
        "epochs_run": len(trainLosses[0]),
    }


def run_asymmetric_lr(model, numEpochs, batchSize, lr_fwd, lr_rev,
                      alpha_fn=alpha_none, grad_clip=0.0):
    """Two optimizers but with asymmetric learning rates.

    Lower reverse LR should reduce interference with the forward pass.
    """
    trainLosses = [[], []]
    accuracy = []
    ping_pong = 0
    prev_out = None
    best_out = float('inf')
    best_recon = float('inf')

    criterionOutput, criterionInput = model._getLossFn()

    for epoch in range(numEpochs):
        alpha = alpha_fn(epoch, numEpochs)
        model.setAlpha(alpha)

        if epoch == 0:
            continue

        model.train(True)
        opt_fwd = model.getOptimizer(lr=lr_fwd)
        opt_rev = model.getOptimizer(lr=lr_rev)

        train_input, train_output = model.inputSpace.getTrainData()

        for i in range(0, len(train_input), batchSize):
            ib = train_input[i:i+batchSize]
            ob = train_output[i:i+batchSize]
            it = model.inputSpace.prepInput(ib)
            ot = model.outputSpace.prepOutput(ob)

            # Forward
            opt_fwd.zero_grad()
            outputPred, end_state = model.forward(it)
            lossOut = criterionOutput(outputPred.squeeze(), ot.squeeze())
            lossOut.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if model.ergodic:
                model.paramUpdate()
            opt_fwd.step()

            # Reverse
            if model.reversible:
                opt_rev.zero_grad()
                reconstructed, start_state = model.reverse(end_state.detach())
                lossIn = criterionInput(start_state, end_state.detach())
                lossIn.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if model.ergodic:
                    model.paramUpdate()
                opt_rev.step()
                epoch_recon = lossIn.item()
            else:
                epoch_recon = 0.0

            epoch_out = lossOut.item()

        trainLosses[0].append(epoch_out)
        trainLosses[1].append(epoch_recon)
        if epoch_out < best_out: best_out = epoch_out
        if epoch_recon < best_recon: best_recon = epoch_recon

        if prev_out is not None and epoch_out > prev_out * 1.5 and prev_out > 0.001:
            ping_pong += 1
        prev_out = epoch_out

        model.inputSpace.shuffle()

        if epoch_out < 1e-6 and epoch_recon < 1e-6:
            break

    # Final accuracy
    model.train(False)
    test_input, test_output = model.inputSpace.getTestData()
    with torch.no_grad():
        allOut = []
        for i in range(0, len(test_input), batchSize):
            ib = test_input[i:i+batchSize]
            it = model.inputSpace.prepInput(ib)
            pred, _ = model.forward(it)
            allOut.append(pred.squeeze())
        allOut = torch.cat(allOut, dim=0)
        predicted = (allOut > 0.5).long()
        actual = (model.outputSpace.getTestOutput().squeeze() > 0.5).long()
        final_acc = (predicted == actual).sum().item() / predicted.size(0)

    final_out = trainLosses[0][-1] if trainLosses[0] else float('inf')
    final_recon = trainLosses[1][-1] if trainLosses[1] else float('inf')

    return {
        "final_out": final_out, "final_recon": final_recon,
        "best_out": best_out, "best_recon": best_recon,
        "final_acc": final_acc, "ping_pong": ping_pong,
        "epochs_run": len(trainLosses[0]),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    NUM_EPOCHS = 1000
    BS = 10

    configs = []

    # --- A) Combined loss, non-ergodic ---
    for lr in [0.01, 0.005, 0.001]:
        for rw in [0.1, 0.5, 1.0, 2.0]:
            configs.append({
                "label": f"combined lr={lr} λ={rw}",
                "fn": "combined", "ergodic": False,
                "lr": lr, "recon_weight": rw, "alpha_fn": alpha_none,
                "grad_clip": 0.0, "lr_decay": 1.0,
            })

    # --- B) Combined loss + grad clip ---
    for lr in [0.01, 0.005]:
        for gc in [1.0, 0.5]:
            configs.append({
                "label": f"combined+clip lr={lr} gc={gc}",
                "fn": "combined", "ergodic": False,
                "lr": lr, "recon_weight": 1.0, "alpha_fn": alpha_none,
                "grad_clip": gc, "lr_decay": 1.0,
            })

    # --- C) Combined loss + LR decay ---
    for lr in [0.01, 0.005]:
        for decay in [0.999, 0.998, 0.995]:
            configs.append({
                "label": f"combined+decay lr={lr} d={decay}",
                "fn": "combined", "ergodic": False,
                "lr": lr, "recon_weight": 1.0, "alpha_fn": alpha_none,
                "grad_clip": 0.0, "lr_decay": decay,
            })

    # --- D) Asymmetric LR (non-ergodic) ---
    for lr_f in [0.01, 0.005]:
        for lr_r in [0.001, 0.0005, 0.0001]:
            configs.append({
                "label": f"asym fwd={lr_f} rev={lr_r}",
                "fn": "asymmetric", "ergodic": False,
                "lr_fwd": lr_f, "lr_rev": lr_r, "alpha_fn": alpha_none,
                "grad_clip": 0.0,
            })

    # --- E) Ergodic + combined loss + fast alpha ---
    for lr in [0.01, 0.005, 0.001]:
        for afn_name, afn in [("none", alpha_none), ("vfast", alpha_very_fast), ("exp", alpha_exp)]:
            configs.append({
                "label": f"erg+comb lr={lr} α={afn_name}",
                "fn": "combined", "ergodic": True,
                "lr": lr, "recon_weight": 1.0, "alpha_fn": afn,
                "grad_clip": 0.0, "lr_decay": 1.0,
            })

    results = []
    total = len(configs)

    for idx, cfg in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] {cfg['label']}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        m = create_model(ergodic=cfg["ergodic"])

        t0 = time.time()
        buf = io.StringIO()
        with redirect_stdout(buf):
            if cfg["fn"] == "combined":
                r = run_combined_loss(
                    m, NUM_EPOCHS, BS, cfg["lr"],
                    recon_weight=cfg["recon_weight"],
                    alpha_fn=cfg["alpha_fn"],
                    grad_clip=cfg["grad_clip"],
                    lr_decay=cfg["lr_decay"],
                )
            elif cfg["fn"] == "asymmetric":
                r = run_asymmetric_lr(
                    m, NUM_EPOCHS, BS, cfg["lr_fwd"], cfg["lr_rev"],
                    alpha_fn=cfg["alpha_fn"],
                    grad_clip=cfg["grad_clip"],
                )
        elapsed = time.time() - t0

        r["label"] = cfg["label"]
        results.append(r)

        print(f"  final: out={r['final_out']:.6f}  recon={r['final_recon']:.6f}  "
              f"acc={r['final_acc']*100:.1f}%  pp={r['ping_pong']}  "
              f"best_out={r['best_out']:.6f}  best_recon={r['best_recon']:.6f}  "
              f"({elapsed:.1f}s)")

    # Summary
    print("\n" + "="*110)
    print(f"{'Config':<45} {'Out':>10} {'Recon':>10} {'BestOut':>10} {'BestRec':>10} {'Acc':>6} {'PP':>4}")
    print("-"*110)

    results.sort(key=lambda r: r["final_out"] + r["final_recon"])

    for r in results:
        print(f"{r['label']:<45} {r['final_out']:>10.6f} {r['final_recon']:>10.6f} "
              f"{r['best_out']:>10.6f} {r['best_recon']:>10.6f} "
              f"{r['final_acc']*100:>5.1f}% {r['ping_pong']:>4d}")

    # Save
    out_path = os.path.join(_PROJECT, "output", "xor_experiment2_results.txt")
    with open(out_path, "w") as f:
        f.write(f"{'Config':<45} {'Out':>10} {'Recon':>10} {'BestOut':>10} {'BestRec':>10} {'Acc':>6} {'PP':>4}\n")
        f.write("-"*110 + "\n")
        for r in results:
            f.write(f"{r['label']:<45} {r['final_out']:>10.6f} {r['final_recon']:>10.6f} "
                    f"{r['best_out']:>10.6f} {r['best_recon']:>10.6f} "
                    f"{r['final_acc']*100:>5.1f}% {r['ping_pong']:>4d}\n")
    print(f"\nResults saved to {out_path}")

    # Top 5
    print("\n*** TOP 5 ***")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. {r['label']}: out={r['final_out']:.6f} recon={r['final_recon']:.6f} acc={r['final_acc']*100:.1f}%")


if __name__ == "__main__":
    run_experiment()
