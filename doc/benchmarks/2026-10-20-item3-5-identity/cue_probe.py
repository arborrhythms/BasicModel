"""Record a failing live cube read; do not change its result or the seed."""
import inspect
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent


def value(item):
    if torch.is_tensor(item):
        return item.detach().cpu().tolist()
    if isinstance(item, (tuple, list)):
        return [value(x) for x in item]
    if isinstance(item, dict):
        return {str(k): value(v) for k, v in item.items()}
    if item is None or isinstance(item, (bool, int, float, str)):
        return item
    return repr(item)


@pytest.fixture(autouse=True)
def record_cue(monkeypatch, request):
    from ReferenceClose import ReferenceClose
    original = ReferenceClose.coordinates

    def observe(resolver, point):
        try:
            return original(resolver, point)
        except ValueError:
            frames = []
            for frame in inspect.stack():
                if frame.function == 'preview':
                    frames.append(dict(function=frame.function,
                        request=value(frame.frame.f_locals.get('request'))))
                if frame.function == '_sentence_observation':
                    local = frame.frame.f_locals
                    entries = local.get('entries', ())
                    b = local.get('b', 0)
                    program = entries[b] if len(entries) > b else None
                    if program is not None:
                        frames.append(dict(function=frame.function,
                            program={key:value(getattr(program,key,None)) for key in
                                ('actions','leaves','word_ids','concept_ids','reference_ids',
                                 'operation_values','lexical_forms')}))
            rows = []
            store = resolver.store
            for row in range(len(store)):
                candidate = store.point_of_row(int(store.row_ids[row]))
                if candidate is not None and torch.equal(candidate,point):
                    rows.append(value(store.row(row)))
            (ROOT/'live-cube-failure.json').write_text(json.dumps(dict(
                nodeid=request.node.nodeid, initial_seed=torch.initial_seed(),
                point=value(point), rows=rows, frames=frames), indent=2)+'\n')
            raise

    monkeypatch.setattr(ReferenceClose,'coordinates',observe)
