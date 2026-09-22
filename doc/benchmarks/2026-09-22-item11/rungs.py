"""Actual pyramid depth/weak-part measurements; old-hop formula as reference."""
import json
import sys
from pathlib import Path
import torch
from test_cs_sparse_weights import _cs, _mint_row


def measure():
    torch.set_num_threads(1)
    records = {}
    for pi in (False, True):
        cs = _cs(nS=128, order=4)
        cs.conceptual_pi = pi
        a0 = torch.full((cs._order_caps()[0], 1), -1.)
        a0[0] = 1.
        rows, previous = [], 0
        for order in range(1, 5):
            r = _mint_row(cs, order, order + 100)
            for source in (previous, *range(1, 8)):
                cs.add_concept_edge(r, source, weight=1.)
            previous = r
            rows.append(r)
        _, a = cs.cs_forward_content(a0, torch.zeros(128, 8))
        records[str(pi).lower()] = {'rows': rows, 'signed_activation': a[rows, 0].tolist(),
                                    'presence': ((a[rows, 0] + 1) / 2).tolist()}
    old = torch.tensor(1.)
    table = []
    for _ in range(4):
        old = torch.tanh(old)
        table.append(float(old))
    records['old_hop_finding_10'] = {'revision': 'd4dc385', 'one_active_zero_background': table}
    cs = _cs(nS=128, order=1)
    r = _mint_row(cs, 1, 101)
    for col in range(32):
        cs.add_concept_edge(r, col, weight=1.)
    _, a = cs.cs_forward_content(torch.full((64, 1), -.998), torch.zeros(128, 8))
    records['weak'] = {'K': 32, 'strongest_presence': .001,
                       'union_presence': float((a[r, 0] + 1) / 2),
                       'excess_bound': .031}
    return records


if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps(measure(), indent=2) + '\n')
