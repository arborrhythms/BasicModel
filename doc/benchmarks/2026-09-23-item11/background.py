"""Production snap regression, on either the signed baseline or paired source."""
import argparse
import json
from pathlib import Path
import torch
from test_cs_sparse_weights import _cs, _mint_row

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', required=True, type=Path)
parser.add_argument('--signed-baseline', action='store_true')
args = parser.parse_args()
report = []
for parts in (2, 4, 8):
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    for col in range(parts):
        cs.add_concept_edge(row, col, 1.)
    snap = cs.cs_snap_order0(torch.zeros(2, 8, 8))
    _, field = cs.cs_forward_content(snap, cs.similarity_codebook.getW())
    if args.signed_baseline:
        positive = ((field[row] + 1) / 2).detach()
    else:
        from ConceptEvidence import symbols
        positive = symbols(field)[row, :, 0].detach()
    report.append(dict(parts=parts, presence=positive.tolist(), expected=0.))
args.out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
raise SystemExit(0 if all(max(r['presence']) == 0 for r in report) else 1)
