"""Read-only diagnostics for the native-42 regressions, with RNG replay."""
import json
import os
from pathlib import Path
import random
import traceback

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def regression_diagnostic(monkeypatch, request):
    if not any(name in request.node.nodeid for name in (
        'both_training_trials_share_bootstrap_snapshot',
        'real_model_forgets_at_document_boundary',
        'forced_ordinary_pending_premise_in_both_orders')):
        return
    directory = Path(os.environ['REGRESSION_DIAGNOSTIC'])
    directory.mkdir(parents=True, exist_ok=True)
    name = request.node.name
    if 'both_training_trials' in name:
        replay = os.environ.get('RECONSTRUCTION_REPLAY')
        if replay:
            rng = torch.load(replay, weights_only=False)
            torch.set_rng_state(rng['torch'])
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
        else:
            snapshot = directory/f'{name}-initial-rng.pt'
            if snapshot.exists():
                raise RuntimeError('refusing to overwrite a diagnostic initialization')
            torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                            numpy=np.random.get_state()), snapshot)

    def clean(value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def record(value):
        with (directory/f'{name}.jsonl').open('a') as output:
            output.write(json.dumps(clean(value))+'\n')

    if 'both_training_trials' in name:
        from Models import BasicModel
        cost = BasicModel._sentence_path_cost

        def path_cost(model, state, sid, active):
            result = cost(model, state, sid, active)
            observation = result[2]
            understanding = observation['record']
            terms = {term: dict(value=value, weight=weight, category=category)
                     for term, value, weight, _, category
                     in model._sentence_cost_registry.terms()
                     if category == 'reconstruction'}
            record(dict(sentence=sid, active=active, terms=terms,
                expected=understanding.word_valid.sum(-1),
                decoded=model._last_decoder_trace[1],
                decoder_comparison=model._last_decoder_comparison,
                field_cost=model._last_field_cost,
                root=understanding.root, words=understanding.word_values,
                word_valid=understanding.word_valid))
            return result
        monkeypatch.setattr(BasicModel, '_sentence_path_cost', path_cost)
    elif 'real_model_forgets' in name:
        from Layers import TernaryTruthStore
        append = TernaryTruthStore.append_meaning

        def append_meaning(store, meaning, **kwargs):
            before = len(store)
            try:
                return append(store, meaning, **kwargs)
            finally:
                record(dict(before=before, after=len(store),
                    kind=kwargs.get('kind', 'fact'), bindings=meaning.bindings,
                    callers=[f'{frame.name}:{frame.lineno}'
                             for frame in traceback.extract_stack(limit=8)[:-1]]))
        monkeypatch.setattr(TernaryTruthStore, 'append_meaning', append_meaning)
    else:
        from ClauseRow import ClauseRows
        from ThoughtReferences import bindings
        read = ClauseRows.read_reference_bindings
        seen = set()

        def read_bindings(store, index, meaning, **kwargs):
            result = read(store, index, meaning, **kwargs)
            names = bindings(result).get('_forward_references')
            key = index, result.role_refs, names, store.parts.revision
            if names and key not in seen:
                seen.add(key)
                from Spaces import _concept_alloc_of
                owner = store._concept_owner()
                alloc = _concept_alloc_of(owner)
                record(dict(index=index, before=meaning.metadata(),
                    after=result.metadata(),
                    parts=[dict(symbol=int(i), **store.parts.row(int(i)))
                           for i in store.parts.indices()],
                    singletons=sorted(alloc.singletons),
                    definitions={str(word): owner.definitions.objects(word)
                                 for word in owner.definitions.word_ids}))
            return result
        monkeypatch.setattr(ClauseRows, 'read_reference_bindings', read_bindings)
