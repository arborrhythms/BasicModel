"""Probe what the chart and codebook actually know about POS after
training.  Loads POS_smoke.xml, runs the model on the training set
once, and dumps:

  1. Per-token POS distribution from chart._chart_pos for the lex
     row (first row of the simplex).
  2. Codebook.category_ids for the symbolic codebook.
  3. The argmax POS for each test sentence's leaf positions.
"""
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

TheData.load(dat.get("dataset"), dat=dat)
m, _ = BaseModel.from_config(CONFIG_PATH, data=TheData)

import torch
m.eval()

# Train a tiny bit so the lex_cat_scorer has nontrivial weights.
m.run(numTrials=1, numEpochs=20, batchSize=16, lr=0.005)

ss = m.symbolicSpace
chart = ss.chart
print("\n===== Chart category names =====")
print(chart._category_names)

print("\n===== Codebook category_ids (per atom) =====")
sym_sub = m.wholeSpace.subspace
what = getattr(sym_sub, 'what', None)
if what is not None and getattr(what, 'category_ids', None) is not None:
    ids = what.category_ids
    print(f"shape: {tuple(ids.shape)}, distinct values: {set(ids.tolist())}")
    print(f"first 16 entries: {ids[:16].tolist()}")
else:
    print("No category_ids on the symbolic codebook.")

# Now run a single batch through the model and inspect chart._chart_pos
test_texts = ["cats chase mice", "cats are happy", "big cats sleep", "cats and dogs"]
batch = torch.stack([TheData.stringTensor(t) for t in test_texts]).unsqueeze(1).float()
print(f"\nBatch shape: {batch.shape}")

with torch.no_grad():
    m(batch)

if chart._chart_pos is None:
    print("chart._chart_pos is None")
else:
    print(f"\nchart._chart_pos shape: {tuple(chart._chart_pos.shape)}")
    print("chart_pos[b, i, i+1, :] gives the POS distribution at lexical position (i, i+1)")
    cat_names = chart._category_names or []
    for b, txt in enumerate(test_texts):
        # Tokenize via the same lexer the model uses.
        from util import parse
        toks = [t for t, _ in parse(txt, lex='words')]
        print(f"\n--- Sentence {b}: {txt!r} ---")
        print(f"  Tokens: {toks}")
        for i in range(min(len(toks), chart._chart_pos.shape[1] - 1)):
            row = chart._chart_pos[b, i, i + 1, :]
            top_k = torch.topk(row, k=min(3, row.numel()))
            top_str = ", ".join(
                f"{cat_names[idx.item()] if idx.item() < len(cat_names) else '?'}={val.item():.3f}"
                for val, idx in zip(top_k.values, top_k.indices))
            print(f"    pos {i} ({toks[i]!r}): {top_str}")
