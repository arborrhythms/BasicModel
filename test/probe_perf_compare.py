"""Side-by-side perf: MM_5M_IR (useGrammar=none) vs LM_5M
(grammar=all, subsymbolicOrder=3) under pure IR mode.

Each config runs in its own subprocess so module-level singletons
(TheData, TheXMLConfig, TheGrammar) start fresh.  Forces CPU + small
max-docs so the run finishes in seconds.
"""
import os, subprocess, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = str(PROJECT / ".venv" / "bin" / "python")

ENV = {
    **os.environ,
    "BASICMODEL_DEVICE": "cpu",
    "BASIC_MAX_DOCS": "200",
    "BASIC_NUM_SHARDS": "1",
    "BASIC_NUM_EPOCHS": "1",
    "PYTHONPATH": str(PROJECT / "bin"),
}


def time_one(config_path, label):
    print(f"\n========== {label} ==========\n  config: {config_path}")
    t0 = time.monotonic()
    p = subprocess.run(
        [PYTHON, str(PROJECT / "bin" / "Models.py"), config_path],
        env=ENV,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.monotonic() - t0
    out = p.stdout + p.stderr
    # Quick parse: look for "Sentences processed", batch counts, etc.
    batches = out.count("batch = ")
    epochs = out.count("Epoch [")
    rc = p.returncode
    print(f"  exit={rc}  elapsed={elapsed:.2f}s  batches~{batches}  epochs~{epochs}")
    if rc != 0:
        print(f"  --- last 20 lines of output ---")
        for line in out.splitlines()[-20:]:
            print(f"    {line}")
    return label, elapsed, batches, rc


results = []
for label, cfg in [
    ("MM_5M_IR (useGrammar=none, IR)",
     str(PROJECT / "data" / "MM_5M_IR.xml")),
    ("LM_5M (grammar=all, subsymbolicOrder=3, IR)",
     str(PROJECT / "data" / "LM_5M.xml")),
]:
    results.append(time_one(cfg, label))

print(f"\n========== SUMMARY (IR mode, 200 docs, batchSize=128, CPU) ==========")
for label, elapsed, batches, rc in results:
    status = "OK" if rc == 0 else f"FAIL (rc={rc})"
    print(f"  {label}")
    print(f"    {status}  wall={elapsed:.2f}s  batches~{batches}")
