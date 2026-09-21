"""Repeat item 10's three fixed workloads in fresh, bounded CPU processes.

Run from BasicModel using the existing virtualenv. --out must be a fresh
directory; prior results are never overwritten. A parity null is a result,
not a process failure or a reason to select another seed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "test"))
from bounded_tests import run_guarded, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(ROOT / p) for p in ("bin", "test")),
               BASICMODEL_DEVICE="cpu", MODEL_COMPILE="eager", BASIC_AUTOLOAD="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("BASIC_SEED", None)
    source = source_snapshot(ROOT)
    manifest = {"validated_source": source, "seed": 42, "completed": [],
                "probe_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE / name for name in
                                          ("probe.py", "parity.py", "parity.xml", "compare.py",
                                           "run_measurements.py"))}}
    for name in ("baseline", "packed", "single", "comparison"):
        command = [sys.executable, str(HERE / "probe.py"), "--out", str(out / f"{name}.json")]
        if name in ("packed", "single"):
            command += ["--config", str(HERE / "parity.xml"), "--parity", name]
        elif name == "comparison":
            command = [sys.executable, str(HERE / "compare.py"), str(out)]
        result = run_guarded(command, cwd=ROOT, env=env, log_path=out / f"{name}.log",
                             memory_bytes=8 * 2**30, timeout=600)
        (out / f"{name}-process.json").write_text(json.dumps(result, indent=2) + "\n")
        if result["exit_code"] != 0:
            raise SystemExit(result["exit_code"])
        manifest["completed"].append(name)
        manifest["source_unchanged"] = source_snapshot(ROOT) == source
        (out / "measurement-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if not manifest["source_unchanged"]:
            raise RuntimeError("Source changed during the measurement")
        print(f"{name}: completed", flush=True)


if __name__ == "__main__":
    main()
