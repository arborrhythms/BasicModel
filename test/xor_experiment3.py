#!/usr/bin/env python3
"""XOR experiment 3: Deep dive into reconstruction loss plateau.

Key findings so far:
  - Combined loss drives output → 0 reliably
  - Reconstruction plateaus ~0.036-0.050
  - Ergodic mode broken (50% accuracy)
  - Forward/reverse layers are separate (pi1/pi2, sigma1/sigma2)
  - Optimizers recreated every epoch (no momentum history!)

This round investigates:
  1. Loss curve shape — is recon stuck from start or converging then plateau?
  2. Persistent optimizer (keep Adam state across epochs)
  3. Higher recon weight to force more reconstruction learning
  4. Separate forward-only phase then joint training
  5. Much higher epochs (5000) to see if recon eventually moves
"""

import os, sys, math, time, io
from contextlib import redirect_stdout, nullcontext

_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
from BasicModel import BasicModel, BasicModelFactory, BaseModel, TheData
import tempfile, xml.etree.ElementTree as ET


def create_model(ergodic=False):
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


def run_detailed(model, numEpochs, batchSize, lr, recon_weight=1.0,
                 persistent_opt=False, phase_split=0):
    """Combined loss training with detailed loss curve tracking.

    Args:
        persistent_opt: keep optimizer across epochs (preserve Adam momentum)
        phase_split: if >0, train forward-only for this many epochs first
    """
    model.set_sigma(0.0)
    criterionOutput, criterionInput = model._getLossFn()

    if persistent_opt:
        optimizer = model.getOptimizer(lr=lr)

    out_curve = []
    recon_curve = []
    acc_curve = []

    for epoch in range(numEpochs):
        if epoch == 0:
            continue

        model.train(True)
        if not persistent_opt:
            optimizer = model.getOptimizer(lr=lr)

        train_input, train_output = model.inputSpace.getTrainData()

        # In phase_split mode: forward-only for first N epochs
        use_recon = (epoch >= phase_split)
        current_rw = recon_weight if use_recon else 0.0

        for i in range(0, len(train_input), batchSize):
            ib = train_input[i:i+batchSize]
            ob = train_output[i:i+batchSize]
            it = model.inputSpace.prepInput(ib)
            ot = model.outputSpace.prepOutput(ob)

            optimizer.zero_grad()
            outputPred, end_state = model.forward(it)
            lossOut = criterionOutput(outputPred.squeeze(), ot.squeeze())

            if model.reversible and use_recon:
                reconstructed, start_state = model.reverse(end_state)
                lossIn = criterionInput(start_state, end_state.detach())
                total_loss = lossOut + current_rw * lossIn
            else:
                lossIn = torch.tensor(0.0)
                total_loss = lossOut

            total_loss.backward()
            optimizer.step()

            e_out = lossOut.item()
            e_recon = lossIn.item() if use_recon else 0.0

        out_curve.append(e_out)
        recon_curve.append(e_recon)

        # Accuracy every 100 epochs
        if epoch % 100 == 0 or epoch == numEpochs - 1:
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
                acc = (predicted == actual).sum().item() / predicted.size(0)
                acc_curve.append((epoch, acc))

        model.inputSpace.shuffle()

    return {
        "out_curve": out_curve,
        "recon_curve": recon_curve,
        "acc_curve": acc_curve,
        "final_out": out_curve[-1] if out_curve else float('inf'),
        "final_recon": recon_curve[-1] if recon_curve else float('inf'),
        "final_acc": acc_curve[-1][1] if acc_curve else 0.0,
    }


def run_experiment():
    BS = 10

    configs = [
        # 1. Baseline: fresh optimizer each epoch (current behavior)
        {"label": "baseline fresh-opt 1000ep",
         "epochs": 1000, "lr": 0.005, "rw": 1.0, "persist": False, "phase": 0},

        # 2. Persistent optimizer (keep momentum)
        {"label": "persistent-opt 1000ep",
         "epochs": 1000, "lr": 0.005, "rw": 1.0, "persist": True, "phase": 0},

        # 3. Persistent + longer training
        {"label": "persistent-opt 3000ep",
         "epochs": 3000, "lr": 0.005, "rw": 1.0, "persist": True, "phase": 0},

        # 4. Persistent + higher recon weight
        {"label": "persist rw=5.0 3000ep",
         "epochs": 3000, "lr": 0.005, "rw": 5.0, "persist": True, "phase": 0},

        # 5. Persistent + much higher recon weight
        {"label": "persist rw=10.0 3000ep",
         "epochs": 3000, "lr": 0.005, "rw": 10.0, "persist": True, "phase": 0},

        # 6. Phase split: forward-only then joint
        {"label": "persist phase=200 3000ep",
         "epochs": 3000, "lr": 0.005, "rw": 1.0, "persist": True, "phase": 200},

        # 7. Lower LR persistent
        {"label": "persist lr=0.001 3000ep",
         "epochs": 3000, "lr": 0.001, "rw": 1.0, "persist": True, "phase": 0},

        # 8. Higher LR persistent
        {"label": "persist lr=0.01 3000ep",
         "epochs": 3000, "lr": 0.01, "rw": 1.0, "persist": True, "phase": 0},

        # 9. Very long run
        {"label": "persist lr=0.005 5000ep",
         "epochs": 5000, "lr": 0.005, "rw": 1.0, "persist": True, "phase": 0},

        # 10. recon_only: weight=0 on output, high on recon
        {"label": "persist rw=1 out_w=0 (recon only) 1000ep",
         "epochs": 1000, "lr": 0.005, "rw": 1.0, "persist": True, "phase": 0,
         "recon_only": True},
    ]

    results = []
    for idx, cfg in enumerate(configs):
        label = cfg["label"]
        recon_only = cfg.get("recon_only", False)
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(configs)}] {label}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        m = create_model(ergodic=False)

        t0 = time.time()
        buf = io.StringIO()
        with redirect_stdout(buf):
            r = run_detailed(m, cfg["epochs"], BS, cfg["lr"],
                           recon_weight=cfg["rw"],
                           persistent_opt=cfg["persist"],
                           phase_split=cfg["phase"])
        elapsed = time.time() - t0

        r["label"] = label
        results.append(r)

        # Print loss at key epochs
        milestones = [10, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
        print(f"  Loss trajectory (output / recon):")
        for ep in milestones:
            if ep < len(r["out_curve"]):
                print(f"    ep {ep:>5d}: out={r['out_curve'][ep-1]:.6f}  recon={r['recon_curve'][ep-1]:.6f}")
        print(f"  FINAL: out={r['final_out']:.6f}  recon={r['final_recon']:.6f}  "
              f"acc={r['final_acc']*100:.1f}%  ({elapsed:.1f}s)")

    # Summary
    print("\n" + "="*90)
    print(f"{'Config':<40} {'Final Out':>10} {'Final Recon':>12} {'Acc':>6}")
    print("-"*90)
    for r in results:
        print(f"{r['label']:<40} {r['final_out']:>10.6f} {r['final_recon']:>12.6f} "
              f"{r['final_acc']*100:>5.1f}%")

    # Save
    out_path = os.path.join(_PROJECT, "output", "xor_experiment3_results.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{'Config':<40} {'Final Out':>10} {'Final Recon':>12} {'Acc':>6}\n")
        f.write("-"*90 + "\n")
        for r in results:
            f.write(f"{r['label']:<40} {r['final_out']:>10.6f} {r['final_recon']:>12.6f} "
                    f"{r['final_acc']*100:>5.1f}%\n")
            # Write loss curve samples
            for ep in [10, 50, 100, 200, 500, 1000, 2000, 3000, 5000]:
                if ep < len(r["out_curve"]):
                    f.write(f"  ep {ep:>5d}: {r['out_curve'][ep-1]:.6f} / {r['recon_curve'][ep-1]:.6f}\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
