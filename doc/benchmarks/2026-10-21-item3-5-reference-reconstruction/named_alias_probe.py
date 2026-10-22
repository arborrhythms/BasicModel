"""Read-only diagnosis of Claude's same-close named-object question.

This is a fixture probe on the frozen source, not a new gate or learning
run. Order and participant positions are explicitly supplied as §13 allows.
The ordinary bind departure selects the second site's next-ranked option.
"""
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT/'bin'), str(ROOT/'test')]
from arrival_capture import plain
from ClauseRow import Clause
from Meaning import ConceptualMeaning
from ReferenceContext import SituationFrame
from ReferenceReconstruction import cost
from test_medium_term_memory import keep, snapshot, unchanged
from test_reference_arrivals import world, named

source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert source == json.loads((HERE/'frozen-source.json').read_text())['source']
folder = HERE/'named-alias-probe'
folder.mkdir(exist_ok=False)
torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=np.random.get_state()), folder/'initial-rng.pt')
resolver, store, atoms, words, _ = world()
with torch.no_grad():
    resolver.components.owner.similarity_codebook.W[
        resolver.components.owner._csw_row_of(words['x'])].copy_(atoms[0])


def site():
    node = named(atoms[0], words['x'])
    return replace(node, point=atoms[0], meaning=ConceptualMeaning.from_description(atoms[0]),
        composition=replace(node.composition, point=atoms[0]),
        reference_request=dict(node.reference_request, mode='bind',
                               content=atoms[0], cue=atoms[0]))


torch.save(dict(parameters=resolver.components.owner.state_dict(), store=store.state_dict()),
           folder/'initial-model.pt')
first, second = site(), site()
clause = Clause(ConceptualMeaning.from_description(atoms[0]), point=atoms[0],
    children=(first, second), refs=(('clause', 0), -1, ('clause', 1)))
before, pool = snapshot(resolver), dict(resolver.components.owner.mtm.entries)
same = resolver.preview(clause, (), departure=1)
same_cost = cost(resolver, [same], atoms[:1])
unchanged(resolver, before)
assert resolver.components.owner.mtm.entries == pool
row, introduction = keep(resolver, site())
frame = SituationFrame.from_row(store.row(row))
before, pool = snapshot(resolver), dict(resolver.components.owner.mtm.entries)
later = resolver.preview(site(), (frame,))
later_cost = cost(resolver, [later], atoms[:1])
unchanged(resolver, before)
assert resolver.components.owner.mtm.entries == pool
result = dict(source=source, fixture_order=1, description=atoms[0],
    same_close=dict(selected=same['items'][1]['selected'], query=same['items'][1]['content'],
        record=same['individuals'][same['items'][1]['selected']], decisions=same['decisions'],
        uncovered_share=same_cost, weighted_cost=same_cost*.1),
    next_close=dict(selected=later['items'][0]['selected'], query=later['items'][0]['content'],
        record=later['individuals'][later['items'][0]['selected']], decisions=later['decisions'],
        uncovered_share=later_cost, weighted_cost=later_cost*.1),
    preview_and_cost_leave_rows_bank_and_pool_unchanged=True)
(folder/'result.json').write_text(json.dumps(plain(result), indent=2)+'\n')
print('same close',float(same_cost[0]*.1),'next close',float(later_cost[0]*.1))
