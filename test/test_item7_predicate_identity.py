"""AK: closing and thought use the same row-free predicate identity and point."""
import pytest
import torch

from ClauseRow import predicate_identity
from reading_fixtures import finish_reading
from Language import Grammar
from Queries import GrammaticalThoughtRegistry
from Understanding import AnswerProgram
from test_cs_symbol_table import _cs
from test_item7_acceptance import SentenceFixture
from test_item7_closing_identities import inventory


@pytest.mark.parametrize('relation', ['part', 'implies'])
@pytest.mark.parametrize('component', ['reference', 'point'])
def test_closing_and_registry_share_predicate(monkeypatch, relation, component):
    f = SentenceFixture(monkeypatch)
    tree = (('part', 'cat', 'animal') if relation == 'part' else
            ('implies', ('lift', 'cat', 'runs'), ('lift', 'dog', 'sleeps')))
    row = f.store.write_clause(f.clause(tree))
    stored = f.store.meaning_of(row)
    reference = f.registry.clause_reference(relation)
    if component == 'reference':
        assert stored.role_refs[1] == reference == ('sym', predicate_identity(relation))
    else:
        assert torch.equal(stored.roles[1], f.registry._payload(reference))
    if relation == 'part':
        question = f.registry.form('part', ('sym', f.noun('cat')), ('sym', f.noun('animal')))
        assert stored.role_refs == question.role_refs
        assert torch.equal(stored.roles[1], question.roles[1])
        assert f.registry.signature_for(question).operation.semantic_id == 'part'


def test_registry_allocates_no_inventory_rows_for_part_or_implies(monkeypatch):
    f = SentenceFixture(monkeypatch)
    for relation in ('part', 'implies'):
        reference = f.registry.clause_reference(relation)
        assert f.cs._csw_row_of(reference[1]) is None
        assert reference[1] not in f.cs._concept_allocator.placement
    assert 'grammatical-vp:conceptual-taxonomy:part' not in f.cs._frozen_named
    assert 'clause:implies' not in f.cs._frozen_named


def test_full_inventory_still_installs_and_forms_part_question():
    cs = _cs()
    operands = []
    while True:
        try:
            cid = cs.new_concept()
            cs._csw_concept_row(0, cid)
            operands.append(('sym', cid))
        except RuntimeError as error:
            assert 'concept inventory exhausted' in str(error)
            break
    assert len(operands) >= 2
    grammar = Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    before = inventory(cs)
    names = dict(getattr(cs, '_frozen_named', {}))
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    question = registry.form('part', *operands[:2])
    assert question.role_refs == (operands[0], ('sym', predicate_identity('part')), operands[1])
    assert registry.signature_for(question).operation.semantic_id == 'part'
    assert inventory(cs) == before
    assert getattr(cs, '_frozen_named', {}) == names


@pytest.mark.parametrize('relation', ['part', 'implies'])
@pytest.mark.parametrize('use_registry', [False, True])
def test_first_three_slot_predicate_occurrence(monkeypatch, relation, use_registry):
    f = SentenceFixture(monkeypatch)
    left = f.store.write_clause(f.clause(('lift', 'cat', 'runs')))
    right = f.store.write_clause(f.clause(('lift', 'dog', 'sleeps')))
    reference = f.registry.clause_reference(relation)
    point = f.registry._payload(reference)
    leaves = torch.stack((f.store.slots[left, 0], point, f.store.slots[right, 0]))
    refs = torch.tensor([int(f.store.row_ids[left]), reference[1], int(f.store.row_ids[right])])
    entry = AnswerProgram(rows=torch.full((3,), -1), word_rows=torch.full((3,), -1),
        word_ids=torch.full((3,), -1), activations=torch.ones(3), leaves=leaves,
        actions=torch.tensor([[0, -1, 0], [0, -1, 1], [0, -1, 2]]),
        targets=torch.tensor([-1]), end_state=leaves, concept_ids=refs)
    before = dict(f.cs._csw_rows)
    clause = finish_reading(f.language, entry, registry=f.registry if use_registry else None)
    row = f.store.write_clause(clause)
    expected = f.store.REL_PARTOF if relation == 'part' else f.store.REL_IMPLIES
    assert f.store.rel_type[row] == expected
    assert f.store.refs[row].tolist() == refs.tolist()
    assert int(f.store.refs[row, 1]) == predicate_identity(relation)
    assert torch.equal(f.store.slots[row, 1], point)
    assert f.cs._csw_rows == before
    assert f.cs._csw_row_of(reference[1]) is None
