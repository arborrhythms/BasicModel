"""Inspect the proposed derived sources without changing the store contract."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import torch

from Layers import LTM0
from Meaning import ConceptualMeaning
from ThoughtReferences import open_slots

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
store = LTM0(4, capacity=8)
base = ConceptualMeaning.from_description(torch.eye(4)[0])
rows = []
for mode, polarity in (('assertive', True), ('assertive', False), ('interrogative', True)):
    row = store.append_meaning(replace(base, mode=mode, polarity=polarity),
        trust=.8, evidence=(.8, .2), order=1, sentence_index=len(store))
    reading = store.raw_meaning_of(row)
    rows.append(dict(mode=reading.mode, polarity=reading.polarity,
        open_slots=open_slots(reading), c_plus=float(store.c_plus[row]),
        c_minus=float(store.c_minus[row]), source_trust=float(store.source_trust[row]),
        role_refs=reading.role_refs, bindings=reading.bindings, scope=reading.scope,
        slots=reading.roles.tolist(), role_mask=reading.role_mask.tolist()))

store.witness_adjacency(0, (2, 3, 2))
before = [(list(pair), sorted(list(witnesses))) for pair, witnesses in store._adjacent_postings.items()]
state = store.state_dict()
extras = store.semantic_extras()
for key in ('adjacent_left', 'adjacent_right', 'adjacent_address', 'adjacent_position'):
    state.pop(key)
restored = LTM0(4, capacity=8)
restored.load_state_dict(state)
restored.load_semantic_extras(extras)
restored.reindex_meanings()
after = [(list(pair), sorted(list(witnesses))) for pair, witnesses in restored._adjacent_postings.items()]

result = dict(source=source, diagnostic_only=True, source_changed=False,
    independently_stored_modes_and_polarities=rows,
    adjacency_before=before, adjacency_after_omission_and_reindex=after,
    note='No test was changed or run by this probe; the proposed sources are inspected on isolated rows.')
(HERE/'cull-preflight.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
