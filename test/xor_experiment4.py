#!/usr/bin/env python3
"""XOR experiment 4: Debug ergodic mode + test wider bottleneck for recon.

Part A: Why is ergodic always 50%?
  - Log alpha/bias/var values during training
  - Test ergodic with alpha=0 permanently (should behave like non-ergodic)
  - Check if paramUpdate() is destabilizing

Part B: Can widening the symbolic bottleneck reduce recon loss?
  - nSymbols=3 (current) -> 4, 6, 8
  - Test if recon loss drops with more symbols
"""

import os, sys, math, time, io
from contextlib import redirect_stdout, nullcontext

_BIN = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_BIN)
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.optim as optim
import tempfile, xml.etree.ElementTree as ET


def create_model(ergodic=False, nSymbols=3, nConcepts=3):
    xml_path = os.path.join(_PROJECT, "data", "XOR_exact.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Patch ergodic
    erg_elem = root.find("architecture/ergodic")
    if erg_elem is None:
        erg_elem = ET.SubElement(root.find("architecture"), "ergodic")
    erg_elem.text = "true" if ergodic else "false"

    # Patch autoload
    auto_elem = root.find("architecture/autoload")
    if auto_elem is None:
        auto_elem = ET.SubElement(root.find("architecture"), "autoload")
    auto_elem.text = "false"

    # Patch nSymbols
    sym_nvec = root.find("SymbolicSpace/nVectors")
    if sym_nvec is not None:
        sym_nvec.text = str(nSymbols)

    # Patch nConcepts
    con_nvec = root.find("ConceptualSpace/nVectors")
    if con_nvec is not None:
        con_nvec.text = str(nConcepts)

    tmp = tempfile.NamedTemporaryFile(mode="wb", suffix=".xml", delete=False)
    tree.write(tmp, xml_declaration=True)
    tmp.close()
    Models.TheData.load("xor")
    m = Models.BasicModel()
    m.create_from_config(tmp.name, data=Models.TheData)
    os.unlink(tmp.name)
    return m


def run_combined(model, numEpochs, batchSize, lr, recon_weight=1.0,
                 alpha_fn=None, skip_param_update=False):
    """Combined loss training. alpha_fn(epoch, N) -> alpha value."""
    model.set_sigma(0.0)
    criterionOutput, criterionInput = model._getLossFn()
    optimizer = model.getOptimizer(lr=lr)

    out_curve = []
    recon_curve = []

    for epoch in range(numEpochs):
        if alpha_fn is not None:
            alpha = alpha_fn(epoch, numEpochs)
            model.set_sigma(alpha)

        if epoch == 0:
            continue

        model.train(True)
        train_input, train_output = model.inputSpace.getTrainData()

        for i in range(0, len(train_input), batchSize):
            ib = train_input[i:i+batchSize]
            ob = train_output[i:i+batchSize]
            it = model.inputSpace.prepInput(ib)
            ot = model.outputSpace.prepOutput(ob)

            optimizer.zero_grad()
            outputPred, end_state = model.forward(it)
            lossOut = criterionOutput(outputPred.squeeze(), ot.squeeze())

            if model.reversible:
                reconstructed, start_state = model.reverse(end_state)
                lossIn = criterionInput(start_state, end_state.detach())
                total_loss = lossOut + recon_weight * lossIn
            else:
                lossIn = torch.tensor(0.0)
                total_loss = lossOut

            total_loss.backward()
            if model.ergodic and not skip_param_update:
                model.paramUpdate()
            optimizer.step()

        out_curve.append(lossOut.item())
        recon_curve.append(lossIn.item() if model.reversible else 0.0)
        model.inputSpace.shuffle()

    # Final accuracy
    model.train(False)
    model.set_sigma(0.0)
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

    return {
        "final_out": out_curve[-1] if out_curve else float('inf'),
        "final_recon": recon_curve[-1] if recon_curve else float('inf'),
        "final_acc": acc,
        "out_curve": out_curve,
        "recon_curve": recon_curve,
    }


def run_experiment():
    BS = 10
    results = []

    # ===== Part A: Debug ergodic =====
    print("\n" + "="*60)
    print("PART A: DEBUGGING ERGODIC MODE")
    print("="*60)

    ergodic_configs = [
        # Baseline non-ergodic
        {"label": "non-ergodic (baseline)",
         "ergodic": False, "alpha_fn": None, "skip_pu": False, "lr": 0.005},

        # Ergodic but alpha=0 always (should behave like non-ergodic?)
        {"label": "ergodic alpha=0 always",
         "ergodic": True, "alpha_fn": lambda e, N: 0.0, "skip_pu": False, "lr": 0.005},

        # Ergodic, alpha=0, skip paramUpdate
        {"label": "ergodic alpha=0 skip-paramUpdate",
         "ergodic": True, "alpha_fn": lambda e, N: 0.0, "skip_pu": True, "lr": 0.005},

        # Ergodic, alpha=0, skip paramUpdate, lower LR
        {"label": "ergodic alpha=0 skip-pu lr=0.001",
         "ergodic": True, "alpha_fn": lambda e, N: 0.0, "skip_pu": True, "lr": 0.001},

        # Ergodic, very fast alpha decay
        {"label": "ergodic vfast-alpha skip-pu",
         "ergodic": True,
         "alpha_fn": lambda e, N: max(0.0, 1.0 - e / max(1, N // 20)),
         "skip_pu": True, "lr": 0.005},

        # Ergodic, very fast alpha decay WITH paramUpdate
        {"label": "ergodic vfast-alpha WITH pu",
         "ergodic": True,
         "alpha_fn": lambda e, N: max(0.0, 1.0 - e / max(1, N // 20)),
         "skip_pu": False, "lr": 0.005},
    ]

    for idx, cfg in enumerate(ergodic_configs):
        label = cfg["label"]
        print(f"\n[A{idx+1}] {label}")

        m = create_model(ergodic=cfg["ergodic"])

        # Check what getOptimizer returns for ergodic
        if cfg["ergodic"]:
            opt = m.getOptimizer(lr=cfg["lr"])
            n_params = sum(p.numel() for group in opt.param_groups for p in group['params'])
            print(f"  Optimizer param count: {n_params}")

        buf = io.StringIO()
        with redirect_stdout(buf):
            r = run_combined(m, 1000, BS, cfg["lr"],
                           alpha_fn=cfg["alpha_fn"],
                           skip_param_update=cfg["skip_pu"])
        r["label"] = label
        results.append(r)

        # Show trajectory
        for ep in [10, 50, 100, 200, 500]:
            if ep < len(r["out_curve"]):
                print(f"  ep {ep:>5d}: out={r['out_curve'][ep-1]:.6f}  recon={r['recon_curve'][ep-1]:.6f}")
        print(f"  FINAL: out={r['final_out']:.6f}  recon={r['final_recon']:.6f}  acc={r['final_acc']*100:.1f}%")

    # ===== Part B: Wider bottleneck =====
    print("\n" + "="*60)
    print("PART B: WIDER SYMBOLIC BOTTLENECK")
    print("="*60)

    for nSym in [3, 4, 6, 8]:
        nCon = nSym  # keep conceptual and symbolic same width
        label = f"nSym={nSym} nCon={nCon}"
        print(f"\n[B] {label}")

        m = create_model(ergodic=False, nSymbols=nSym, nConcepts=nCon)

        buf = io.StringIO()
        with redirect_stdout(buf):
            r = run_combined(m, 2000, BS, 0.005)
        r["label"] = label
        results.append(r)

        for ep in [100, 500, 1000, 2000]:
            if ep < len(r["out_curve"]):
                print(f"  ep {ep:>5d}: out={r['out_curve'][ep-1]:.6f}  recon={r['recon_curve'][ep-1]:.6f}")
        print(f"  FINAL: out={r['final_out']:.6f}  recon={r['final_recon']:.6f}  acc={r['final_acc']*100:.1f}%")

    # Summary
    print("\n" + "="*90)
    print(f"{'Config':<45} {'Final Out':>10} {'Final Recon':>12} {'Acc':>6}")
    print("-"*90)
    for r in results:
        print(f"{r['label']:<45} {r['final_out']:>10.6f} {r['final_recon']:>12.6f} "
              f"{r['final_acc']*100:>5.1f}%")

    # Save
    out_path = os.path.join(_PROJECT, "output", "xor_experiment4_results.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(f"{r['label']}: out={r['final_out']:.6f} recon={r['final_recon']:.6f} acc={r['final_acc']*100:.1f}%\n")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
