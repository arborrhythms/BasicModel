"""Verify the perf-critical paths in MM_20M's forward are bypassed:
  1. SymbolSpace.compose returns immediately on useGrammar='none'.
  2. Chart._chart_inside is never invoked.
  3. No MereologicalTree is constructed.
  4. _apply_codebook_pos_seed never runs.

Wraps the relevant methods with counters and runs a single forward pass.
"""
import os, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "bin"))
os.environ["BASICMODEL_DEVICE"] = "cpu"
os.environ["BASIC_MAX_DOCS"] = "10"
os.environ["BASIC_NUM_SHARDS"] = "1"

CONFIG_PATH = str(PROJECT / "data" / "MM_20M.xml")

from util import init_config, ProjectPaths, TheXMLConfig
from data import TheData
from Models import BaseModel
import Language

init_config(path=CONFIG_PATH,
            defaults_path=str(PROJECT / "data" / "model.xml"))

cfg = TheXMLConfig.data
arch = cfg.get("architecture", {})
dat = arch.get("data", {})
TheData.load(dat.get("dataset"), num_shards=1, max_docs=10,
             shard_dir=dat.get("shardDir"), dat=dat)

m, _ = BaseModel.from_config(CONFIG_PATH, data=TheData)

ss = getattr(m, 'symbolSpace', None)
print(f"symbolSpace: {type(ss).__name__ if ss else None}")
print(f"useGrammar: {m.useGrammar}")
print(f"mereological_tree: {getattr(ss, 'mereological_tree', None)}")
print(f"_grammar_is_default_only: {getattr(ss, '_grammar_is_default_only', None)}")

# Wrap to count.
counts = {'compose': 0, 'chart_inside': 0, 'seed': 0}

orig_compose = ss.compose
def _wcompose(*a, **kw):
    counts['compose'] += 1
    return orig_compose(*a, **kw)
ss.compose = _wcompose

chart = ss.chart
orig_inside = chart._chart_inside
def _winside(*a, **kw):
    counts['chart_inside'] += 1
    return orig_inside(*a, **kw)
chart._chart_inside = _winside

orig_seed = chart._apply_codebook_pos_seed
def _wseed(*a, **kw):
    counts['seed'] += 1
    return orig_seed(*a, **kw)
chart._apply_codebook_pos_seed = _wseed

import torch
batch_texts = TheData.train_input[:4]
batch = torch.stack([TheData.stringTensor(t) for t in batch_texts]).unsqueeze(1).float()
print(f"\nRunning one forward, batch shape {batch.shape}")
m.eval()
with torch.no_grad():
    try:
        m(batch)
    except Exception as e:
        print(f"forward raised: {type(e).__name__}: {e}")

print(f"\nCounts after one forward:")
for k, v in counts.items():
    print(f"  {k}: {v}")
print()
print("Expected for MM_20M (useGrammar='none'):")
print("  compose: > 0  (called by ChartCompose pipeline stage)")
print("  chart_inside: 0  (short-circuited by useGrammar guard)")
print("  seed: 0  (only fires inside _chart_inside)")
