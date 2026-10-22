"""Replay the retained failed gate and inspect its missing frame metadata.

The production source and gate are untouched. This diagnostic restores the
failed gate's captured RNG, executes its unchanged assertions, and then
compares the bare frame with the production SituationAddress read.
"""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent/'2026-10-21-item3-5-reference-reconstruction'),
               str(ROOT/'bin'), str(ROOT/'test')]
from arrival_capture import plain, restore
from ReferenceContext import SituationAddress, SituationFrame
from test_medium_term_memory import snapshot, unchanged
from test_reference_reconstruction import test_named_token_keeps_its_content_before_and_after_the_close, sentence_cost

source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
assert source == json.loads((HERE/'frozen-source.json').read_text())['source']
assert not (HERE/'frame-probe.json').exists()
capture = HERE/'scoped-state/test_named_token_keeps_its_content_before_and_after_the_close'
restore(torch.load(capture/'initial-rng.pt', weights_only=False))
try:
    test_named_token_keeps_its_content_before_and_after_the_close()
except AssertionError as error:
    trace = error.__traceback__
    while trace.tb_next is not None:
        trace = trace.tb_next
    held = trace.tb_frame.f_locals
    failure = dict(line=trace.tb_lineno, message=str(error), function=trace.tb_frame.f_code.co_name)
else:
    raise RuntimeError('The retained failure did not replay')
resolver, store, owner = held['resolver'], held['store'], held['owner']
row, written, kept, token = (held[key] for key in ('row', 'written', 'kept', 'token'))
address = SituationAddress(store.occurrence_of(row),
    symbols=written.get(('symbols', id(kept)), ()),
    activations=written.get(('activations', id(kept)), ()))
frame = address.read(store)
before, pool = snapshot(resolver), dict(owner.mtm.entries)
complete = resolver.preview(held['site'](0), (frame,))
registered_cost = sentence_cost(resolver, [complete]).total(objective='reconstruction')
unchanged(resolver, before)
assert owner.mtm.entries == pool
same = held['same']
result = dict(source=source, replay_of=str(capture.relative_to(ROOT)), retained_failure=failure,
    bare_frame=asdict(SituationFrame.from_row(store.row(row))),
    production_frame=asdict(frame), token=token,
    token_entry=vars(owner.mtm.entries[token]),
    same_sentence=dict(selected=same['items'][1]['selected'],
        features=same['items'][1]['features'], decision=same['decisions'][1],
        reference_cost=sentence_cost(resolver, [same]).total(objective='reconstruction')),
    bare_next_sentence=dict(selected=held['later']['items'][0]['selected'],
        candidates=held['later']['items'][0]['candidates'], records=held['later']['records']),
    complete_next_sentence=dict(selected=complete['items'][0]['selected'],
        features=complete['items'][0]['features'], decision=complete['decisions'][0],
        reference_cost=registered_cost, records=complete['records']),
    complete_frame_selects_introduction=complete['items'][0]['selected'] == token,
    reads_leave_rows_bank_and_mtm_unchanged=True,
    diagnostic_only=True, original_gate_remains_failed=True)
(HERE/'frame-probe.json').write_text(json.dumps(plain(result), indent=2)+'\n')
print('retained failure:', failure)
print('bare frame activations:',len(result['bare_frame']['activations']))
print('production frame activations:',len(frame.activations))
print('complete frame selects introduction:',result['complete_frame_selects_introduction'])
print('same / next registered cost:',plain(result['same_sentence']['reference_cost']),plain(registered_cost))
