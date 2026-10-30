"""The reading-round candidate-table probe, retained as a regression."""
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from Meaning import ConceptualMeaning
from Queries import GrammaticalThoughtRegistry
from ThoughtReferences import validate_local_references


@pytest.mark.parametrize('held', [('constituent', 0), (('nested', ('constituent', 0)),)])
def test_candidate_table_probe(held):
    child = ConceptualMeaning.from_description(torch.full((8,), .25))
    parent = replace(ConceptualMeaning.from_description(torch.full((8,), .5)),
        bindings={'held': held, 'durable': ('sym', 2), '_evidence_pair': (1., 0.)},
        constituents=(child,))
    validate_local_references(parent)
    owner = SimpleNamespace(_reference=lambda identity: ('sym', 1),
        _payload=lambda reference: torch.full((8,), .5),
        _slot_for_operand=lambda role: 0,
        _is_description_reference=lambda reference: reference == ('ltm', 'probe', 1))
    operation = SimpleNamespace(semantic_id='query', operand_roles=('argin1',))
    descriptor = SimpleNamespace(domain='ltm', open_operands=False,
                                kinds_for=lambda roles: ('description',))
    formed = GrammaticalThoughtRegistry._form_candidate(owner, operation, descriptor,
        {'argin1': (parent.roles[0], ('ltm', 'probe', 1))}, source=parent)
    validate_local_references(formed)
    assert formed.constituents == ()
    assert dict(formed.bindings) == {'durable': ('sym', 2), '_evidence_pair': (1., 0.),
                                   '_open_references': ()}
    assert dict(parent.bindings)['held'] == held and parent.constituents == (child,)
    # An empty read supplies neither its request nor an unrelated child.
    from ThoughtStream import materialize
    from Queries import ThoughtResult
    result = materialize(ThoughtResult('query', 'ltm', 'set', 'retrieval',
                                     formed, MappingProxyType({})))
    assert result.evidence.get('meaning') is None


@pytest.mark.parametrize('slot', (0, 2))
@pytest.mark.parametrize('kind', ('concept', 'meaning'))
def test_candidate_fixed_operand_skips_an_occupied_open_placeholder(slot, kind):
    """Native capture: a chain's missing argin2 still held a full vector."""
    refs = [('sym', 22), ('sym', 7), ('sym', 24)]
    refs[slot] = None
    pending = ConceptualMeaning(torch.full((3, 8), .25), torch.ones(3, dtype=torch.bool),
        role_refs=tuple(refs), bindings={'_chain_view': True,
            '_open_references': (('referent', slot),)})
    refs[slot] = ('sym', 31)
    known = replace(pending, roles=torch.full((3, 8), .75), role_refs=tuple(refs),
        bindings={'_chain_view': True, '_open_references': ()})
    owner = SimpleNamespace(_slot_for_operand=lambda role: 0 if role == 'argin1' else 2)
    role = 'argin1' if slot == 0 else 'argin2'
    assert GrammaticalThoughtRegistry._candidate_operand(owner, (pending,), role, kind) is None
    value, reference = GrammaticalThoughtRegistry._candidate_operand(
        owner, (pending, known), role, kind)
    assert reference == ('sym', 31)
    torch.testing.assert_close(value, known.roles[slot], rtol=0, atol=0)


@pytest.mark.parametrize('open_role', ['argin1', 'argin2'])
def test_candidate_from_closed_chain_retains_its_declared_hole_at_the_part_writer(open_role):
    """Native ccc capture: isPart inherited _chain_view and empty open slots."""
    from ThoughtReferences import open_slots
    from ThoughtStream import write
    source = ConceptualMeaning(torch.full((3, 8), .25), torch.ones(3, dtype=torch.bool),
        role_refs=(('sym', 23), ('sym', 7), ('sym', 24)),
        bindings={'_chain_view': True, '_open_references': (), '_evidence_pair': (.05, 0.)})
    owner = SimpleNamespace(_reference=lambda identity: ('sym', 7),
        _payload=lambda reference: source.roles[1],
        _slot_for_operand=lambda role: 0 if role == 'argin1' else 2)
    operation = SimpleNamespace(semantic_id='isPart', operand_roles=('argin1', 'argin2'))
    from Queries import THOUGHT_EXECUTORS
    operands = {role: (source.roles[slot], source.role_refs[slot])
                for role, slot in (('argin1', 0), ('argin2', 2)) if role != open_role}
    formed = GrammaticalThoughtRegistry._form_candidate(owner, operation, THOUGHT_EXECUTORS['isPart'],
        operands, source=source, open_roles=(open_role,))
    missing = owner._slot_for_operand(open_role)
    assert open_slots(formed) == (('referent', missing),)
    def signature(value, **kwargs):
        if not bool(value.role_mask[1]):
            raise ValueError('a point has no relation')
        return SimpleNamespace(operation=operation)
    written = []
    def store_clause(clause, **kwargs):
        written.append((clause, kwargs))
        return 0
    store = SimpleNamespace(_validate_local_references=validate_local_references, write_clause=store_clause)
    model = SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store),
        grammatical_thoughts=SimpleNamespace(signature_for=signature))
    assert write(model, formed, row=0) == 0
    clause, metadata = written[0]
    assert metadata['kind'] == 'question' and clause.refs[missing] == ('clause', 0)
    assert open_slots(clause.children[0].meaning) == (('referent', 0),)
    # The source remains a closed occurrence; opening a candidate is no edit.
    assert not open_slots(source)
