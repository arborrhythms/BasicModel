"""Measure untrained MentalModel over a declared seed range, without selection.

Each seed gets a fresh bounded process. Every outcome, including a failure,
is retained; this is a reproducibility measurement, not a passing-seed picker.
Run from basicmodel with PYTHONPATH=bin:test and the existing virtualenv.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

os.environ.setdefault("MODEL_COMPILE", "eager")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("BASIC_AUTOLOAD", "false")


def measure(seed, out):
    import numpy as np
    import torch
    import Language
    from test_hierarchical import TestBackwardCompat

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rounds = []
    original = Language.BinaryStructuredReductionLayer.forward

    def observed(self, x, *args, **kwargs):
        lengths = kwargs.get("lengths")
        rounds.append({"shape": list(x.shape),
                       "lengths": lengths.tolist() if lengths is not None else None,
                       "input_abs_max": float(x.detach().abs().max())})
        return original(self, x, *args, **kwargs)

    Language.BinaryStructuredReductionLayer.forward = observed
    report = {"seed": seed, "rounds": rounds}
    try:
        result = TestBackwardCompat()._mentalmodel_forward()
        report["outputs_finite"] = all(bool(torch.isfinite(t).all())
                                       for t in result if torch.is_tensor(t))
        report["status"] = "passed" if report["outputs_finite"] else "failed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        Language.BinaryStructuredReductionLayer.forward = original
        out.write_text(json.dumps(report, indent=2) + "\n")
    return report["status"] == "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is not None:
        raise SystemExit(0 if measure(args.seed, args.out) else 1)

    from bounded_tests import run_guarded, source_snapshot
    args.out.mkdir(parents=True, exist_ok=True)
    source = source_snapshot(Path.cwd())
    report = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source": source, "seeds": list(range(args.count)), "runs": []}
    for seed in report["seeds"]:
        result_file = args.out / f"seed-{seed:02d}.json"
        result = run_guarded(
            [sys.executable, __file__, "--seed", str(seed), "--out", str(result_file)],
            cwd=Path.cwd(), env=os.environ.copy(),
            log_path=args.out / f"seed-{seed:02d}.log",
            memory_bytes=8 * 2**30, timeout=120)
        case = json.loads(result_file.read_text()) if result_file.exists() else {"seed": seed, "status": "failed"}
        report["runs"].append({"result": case, "process": result})
        (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"seed {seed}: {case['status']}", flush=True)
    report["source_unchanged"] = source_snapshot(Path.cwd()) == source
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return all(run["result"]["status"] == "passed" for run in report["runs"])


if __name__ == "__main__":
    main()
