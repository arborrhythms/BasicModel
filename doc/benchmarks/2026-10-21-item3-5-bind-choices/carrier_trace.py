"""Observe the lexical/content boundary on the exact retained bootstrap draw."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def trace_carrier(request, monkeypatch):
    if request.node.name != 'test_both_training_trials_share_bootstrap_snapshot':
        return
    from Models import BasicModel
    from arrival_capture import plain
    original = BasicModel._program_entries

    def capture(model, *args, **kwargs):
        entries = original(model, *args, **kwargs)
        owner = model._concept_owner()
        record = dict(admit=kwargs.get('admit'), trial=getattr(model, '_sentence_trial', None),
            entries=[None if entry is None else dict(
                concept_ids=entry.concept_ids, word_ids=entry.word_ids,
                word_rows=entry.word_rows, rows=entry.rows,
                reference_ids=entry.reference_ids,
                words=[owner.word_surface_for_row(int(row)) for row in entry.word_rows],
                leaves=entry.leaves) for entry in entries])
        with (Path(os.environ['BOOTSTRAP_CAPTURE'])/'carrier.jsonl').open('a') as output:
            output.write(json.dumps(plain(record))+'\n')
        return entries
    monkeypatch.setattr(BasicModel, '_program_entries', capture)
