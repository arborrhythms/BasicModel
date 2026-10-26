"""Occurrence addresses keep the retained Part arguments' participant evidence."""
from chunk_fixtures import context as chunk_context
import pytest
import torch

from test_bind_thought import execute, native, occurrence, site


@pytest.mark.parametrize('kind,other', ((0, 1), (1, 2), (2, 3), (3, 0)))
def test_occurrence_candidates_use_retained_part_participants(monkeypatch, kind, other):
    f = native(monkeypatch)
    target, _ = occurrence(f, f.atoms[kind], f.atoms[4])
    distractor, _ = occurrence(f, f.atoms[other], f.atoms[5])
    address = f.identities[target]
    assert address == f.store.occurrence_of(target)
    assert not f.cs.components.individual_ids
    before = f.store.parts.refs.clone(), f.store.slots.clone(), f.cs.similarity_codebook.W.clone()
    records = f.resolver.accessible(chunk_context(f.resolver, f.frames))
    candidates = f.resolver.individuals(records)
    assert address in candidates and f.identities[distractor] in candidates
    assert f.resolver.prediction(address, f.atoms[4], 0) > 0
    assert f.resolver.prediction(f.identities[distractor], f.atoms[4], 0) == 0
    context = next(context for location, context in records.values() if location == target)
    assert context['identity'] == address and context['predicate'] == f.predicates[tuple(f.atoms[4].tolist())]
    result = execute(f, site(f, f.atoms[kind]*0, predicate=f.atoms[4]))
    assert result.value.role_refs[0] == address
    for actual, expected in zip((f.store.parts.refs, f.store.slots, f.cs.similarity_codebook.W), before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
