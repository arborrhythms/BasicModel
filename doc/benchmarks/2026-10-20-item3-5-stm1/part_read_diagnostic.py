"""Read-only symbol-type evidence for the unchanged decomposition certificate."""
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def capture_part_types(monkeypatch):
    from Queries import ThoughtLTMCapability
    from Spaces import _concept_alloc_of
    original = ThoughtLTMCapability._with_components

    def read(capability, result, **kwargs):
        store = object.__getattribute__(capability, '_ThoughtLTMCapability__store')()
        owner = store._concept_owner()
        alloc = _concept_alloc_of(owner)
        pos = owner._model.languageSpace.parts_of_speech
        record = dict(
            roots=[frame['occurrence'] for frame in result.get('frames', ())],
            parts={str(i): store.parts.refs[i].tolist()
                   for i in store.parts.indices().tolist()},
            symbols={str(i): dict(types=sorted(alloc.symbol_types.get(i, ())),
                order=alloc.order_of(i), no_object=owner.interpret.no_object(i),
                words=owner.definitions.words(i),
                counts=dict(pos.counts.get(i, {})),
                word_counts={str(w): dict(pos.counts.get(w, {}))
                             for w in owner.definitions.words(i)})
                     for i in alloc.placement})
        path = Path(os.environ['PART_READ_DIAGNOSTIC'])
        path.parent.mkdir(parents=True, exist_ok=True)
        result = original(capability, result, **kwargs)
        record['components'] = {str(ref): dict(kind=value['kind'], operands=value['operands'],
            roles=value['meaning'].roles.tolist()) for ref, value in result['components'].items()}
        record['incomplete'] = result['incomplete']
        with path.open('a') as output:
            output.write(json.dumps(record) + '\n')
        return result

    monkeypatch.setattr(ThoughtLTMCapability, '_with_components', read)
