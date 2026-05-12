"""Probe: build the model, call chart.compose() once with a small batch
of basic sentences, and let the IndexError bubble up so we can see
exactly which line in the chart's inside pass fails."""
import os, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "bin"))
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

from util import init_config, ProjectPaths, TheXMLConfig
from data import TheData
from Models import BaseModel

CONFIG_PATH = str(PROJECT / "data" / "POS_smoke.xml")

defaults_path = str(PROJECT / "data" / "model.xml")
init_config(path=CONFIG_PATH, defaults_path=defaults_path)

cfg = TheXMLConfig.data
arch = cfg.get("architecture", {})
dat = arch.get("data", {})

dataset = dat.get("dataset")
TheData.load(
    dataset,
    num_shards=1,
    max_docs=10000,
    shard_dir=dat.get("shardDir"),
    dat=dat,
)

m, _ = BaseModel.from_config(CONFIG_PATH, data=TheData)

# Set up a tiny manual batch to feed through forward — the chart fires
# inside model.forward via the Sequential pipeline. Use the data's first
# 4 train sentences encoded as byte tensors.
import torch

batch_texts = TheData.train_input[:4]
print(f"Batch sentences: {batch_texts}")

# stringTensor → [inputLength] int8 byte tensor
batch = torch.stack([TheData.stringTensor(t) for t in batch_texts])
batch = batch.unsqueeze(1)  # [B, 1, inputLength]
print(f"Batch tensor shape: {batch.shape}")

m.eval()
ws = m.wordSpace
print(f"wordSpace: {type(ws).__name__}")
print(f"chart: {type(ws.chart).__name__}, w_max={ws.chart.w_max}")
print(f"chart router_kind: {ws.chart.router_kind}")
print(f"useGrammar: {m.useGrammar}")

# Monkey-patch ChartCompose to re-raise instead of swallowing.
from Models import ChartCompose
_orig_forward = ChartCompose.forward

def _noisy_forward(self, subspace):
    ws = self._word_space
    if ws is None or subspace is None:
        return subspace
    if hasattr(subspace, 'is_empty') and subspace.is_empty():
        return subspace
    data = subspace.materialize() if hasattr(subspace, 'materialize') else None
    print(f"  [ChartCompose] data shape: {None if data is None else tuple(data.shape)}")
    if data is None:
        return subspace
    ws.compose(data, subspace=subspace)
    return subspace

ChartCompose.forward = _noisy_forward

import traceback
try:
    out = m(batch.to(torch.float32))
    print("forward OK")
except Exception:
    traceback.print_exc()
