"""Replay a retained failed initial state and observe its returned row kind."""
import json
import os
from pathlib import Path
import random
import numpy as np
import pytest
import torch

@pytest.fixture(autouse=True)
def relative_end_probe(request, monkeypatch):
    if request.node.name != 'test_one_native_sentence_owns_only_its_end_state_after_the_boundary':
        return
    saved = torch.load(os.environ['END_STATE_REPLAY'],weights_only=False,map_location='cpu')
    torch.set_rng_state(saved['torch'])
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    from Models import BasicModel
    original = BasicModel._commit_sentence
    def commit(model, *args, **kwargs):
        result = original(model, *args, **kwargs)
        store = model.symbolSpace.ltm_store
        rows=[]
        for fields in model._sentence_fields.values():
            for field in fields:
                if field is not None:
                    rows.append(dict(row_id=field.row_id,location=store.location(field.row_id),
                        occurrence_index=store.index_of_row(field.row_id),
                        row_count=len(store), retained_slots=store.slots[:len(store)].tolist(),
                        meaning_roles=field.meaning.roles.tolist()))
        Path(os.environ['END_STATE_REPORT']).write_text(json.dumps(rows,indent=2)+'\n')
        return result
    monkeypatch.setattr(BasicModel,'_commit_sentence',commit)
