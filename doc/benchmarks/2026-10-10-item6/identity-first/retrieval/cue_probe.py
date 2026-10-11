"""Replay the captured cue at the order-reader boundary, not a full sentence.

The first failed warm-up retained only its traceback. This probe uses the
first native invalid cue captured by the corrected run and keeps that
distinction explicit. No training, reference replacement or seeding occurs.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import torch
import MemoryIndex
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from Taxonomy import concept_reference


source = Path(sys.argv[1])
destination = Path(sys.argv[2])
saved = torch.load(source, map_location='cpu', weights_only=True)
cue = ConceptualMeaning(saved['roles'], saved['role_mask'], **saved['metadata'])
baseline = '06f6e2b88811a107fbf7dff505cc8b2af189c34b'
old = subprocess.check_output(['git', 'show', baseline + ':bin/MemoryIndex.py'], text=True)
tree = ast.parse(old)
node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'cued_rows')
scope = dict(vars(MemoryIndex))
exec(compile(textwrap.dedent('\n'.join(old.splitlines()[node.lineno-1:node.end_lineno])),
             baseline + ':cued_rows', 'exec'), scope)


def probe(method):
    store = TernaryTruthStore(cue.roles.shape[-1], capacity=2)
    calls, unfolded = [], []
    def code_row(reference):
        calls.append(reference)
        return concept_reference(reference)[1]
    def unfold(value, allowance, **kwargs):
        unfolded.append(dict(norm=float(value.norm()), order=kwargs['order']))
        return (), 0, False
    store.configure_leaf_index(code_row=code_row, order_of=lambda _: 0, unfold=unfold)
    try:
        result = method(store, cue)
        return dict(error=None, reader_calls=calls, unfolded=unfolded,
                    candidates=len(result['value']))
    except Exception as error:
        return dict(error=f'{type(error).__name__}: {error}', reader_calls=calls, unfolded=unfolded)


result = dict(captured_from=str(source), baseline=baseline, references=cue.role_refs,
    old=probe(scope['cued_rows']), corrected=probe(MemoryIndex.LeafCodeIndex.cued_rows),
    scope='Order-reader boundary replay on the corrected run capture; not a retained context from the first failed run, and not an end-to-end replay.')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
