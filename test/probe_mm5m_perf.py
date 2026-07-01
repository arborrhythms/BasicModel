"""Perf probe: how many sentences/sec does MM_20M achieve on a small
slice of fineweb? Compares the perf-critical paths against the user's
historical baseline (30-40 sent/sec in serial AR mode).

Override knobs honoured by Models.py:
    BASIC_MAX_DOCS    — cap docs per shard
    BASIC_NUM_SHARDS  — cap shards loaded
    BASIC_NUM_EPOCHS  — cap epochs

Forces CPU so we get reproducible numbers across runs without
worrying about device warm-up.
"""
import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "bin"))
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ["BASIC_MAX_DOCS"] = "50"
os.environ["BASIC_NUM_SHARDS"] = "1"
os.environ["BASIC_NUM_EPOCHS"] = "1"
# Disable autoload so we don't pick up an old checkpoint.
os.environ["BASIC_NO_COMPILE"] = "1"

CONFIG_PATH = str(PROJECT / "data" / "MM_20M_legacy.xml")

from util import init_config, ProjectPaths, TheXMLConfig
from data import TheData
from Models import BaseModel, ModelFactory


t0 = time.monotonic()
results = ModelFactory.run(CONFIG_PATH)
elapsed = time.monotonic() - t0

# Best-effort sentence count: TheData.train_input length × num_epochs.
sentences = (
    len(TheData.train_input) if hasattr(TheData, 'train_input') else 0
) * int(os.environ["BASIC_NUM_EPOCHS"])
rate = sentences / elapsed if elapsed > 0 and sentences > 0 else 0.0

print(f"\n===== MM_20M perf probe =====")
print(f"Elapsed: {elapsed:.2f}s")
print(f"Sentences processed: ~{sentences}")
print(f"Sentences/sec: ~{rate:.1f}")
print(f"User's historical baseline: 30-40 sent/sec (serial AR)")
