from ConceptHierarchy import refresh
"""Operation authority is measured from kept inputs and uses, never optimizer steps."""
from types import SimpleNamespace
from dataclasses import replace
import copy
import math
import pytest
import torch
from ClauseRow import Clause, Composition
from Meaning import ConceptualMeaning
from definition_fixtures import agreeing_views, learned_operation
from test_clause_acceptance import SentenceFixture
from test_review13_subspaces import fixture as paired_fixture
from ConceptTrust import positive_length


def through_operation(clause, name):
    return replace(clause, composition=replace(clause.composition,
        operation=name, formal_name=name))








def operation_fixture(monkeypatch):
    owner, bank, _, alloc = paired_fixture(meaning_pairs=4, capacity=160)
    f = SentenceFixture(monkeypatch, cs=owner, native_cube=False)
    words = [owner.interpret.lookup_word([part], (), form=form)
             for part, form in ((4, 'dax'), (7, 'wug'))]
    for i, word in enumerate(words):
        row=owner._csw_row_of(word)
        bank.mereology.placement[row]=torch.eye(bank.mereology.meaning_pairs)[i]
        bank.mereology.placement_witnesses[row]=1
    agreeing_views(owner, *words, count=10)
    speech = f.language.parts_of_speech
    leaves = tuple(Composition(bank.lookup_rows(owner._csw_row_of(word)),
        reference=word, word_id=word, order=0) for word in words)
    composition = Composition(leaves[0].point, leaves, operation='lift', formal_name='lift')
    clause = Clause(ConceptualMeaning.from_description(composition.point),
                    composition=composition, point=composition.point)
    index = next(i for i, rule in enumerate(f.binary) if rule.method_name == 'lift')
    program = SimpleNamespace(word_ids=torch.tensor(words),
        actions=torch.tensor([[0,-1,0],[0,-1,1],[1,index,-1]]))
    return f, words, clause, program




def test_shared_word_teaches_another_argument_without_dividing_existing_authority(monkeypatch):
    f, words, clause, program = operation_fixture(monkeypatch)
    speech, parts = f.language.parts_of_speech, f.store.parts
    speech.observe_kept(program, f.binary, f.unary)
    refresh(speech.owner)
    address = f.store.write_clause(clause, evidence=(.8,.6))
    assert isinstance(address, int)
    before = f.store.trust[:len(f.store)].clone()
    prior_seat = positive_length(f.cs, speech.seats['lift'].concept)
    revision = speech.revision
    learned_operation(speech, 'verb', (words[0],), (), count=1)
    assert speech.revision != revision
    measure = speech.operation('lift')
    radius = -math.expm1(-1/3)
    assert measure['uses'] == 1 and measure['definiteness'] == pytest.approx((radius, radius))
    assert positive_length(f.cs, speech.seats['lift'].concept) == prior_seat
    torch.testing.assert_close(f.store.trust[:len(f.store)], before)






def test_empty_occupants_count_uses_without_inventing_a_class(monkeypatch):
    f, _, _, program = operation_fixture(monkeypatch)
    program.word_ids = None
    program.leaves = torch.zeros(2, f.cs.nDim)
    speech = f.language.parts_of_speech
    for _ in range(3):
        speech.observe_kept(program, f.binary, f.unary)
        refresh(speech.owner)
    assert all(not speech.members(arg) for arg in speech.arguments)
    assert speech.operation('lift')['definiteness'] == (0.,0.)
    assert speech.operation('lift')['uses'] == 3
    assert positive_length(f.cs, speech.seats['lift'].concept) == 0.


def test_argument_confidence_uses_member_agreement_instead_of_competing_roles():
    from argument_fixtures import argument_inventory
    owner, speech, (a,b,c) = argument_inventory(('lift_argin1','lift_argin2','verb_argin1'), count=3)
    learned_operation(speech, 'lift', a, c, count=3)
    learned_operation(speech, 'lift', b, c, count=1)
    learned_operation(speech, 'verb', b, count=1)
    # The fourth, orthogonal member disagrees; its other role is independent.
    expected = .75 * -math.expm1(-4/3)
    assert float(speech.radii()[speech.index['lift_argin1']]) == pytest.approx(expected)
    evidence = -math.expm1(-4/3)
    assert speech.operation('lift')['definiteness'] == pytest.approx((evidence,evidence))
    assert 0 < positive_length(owner, speech.seats['lift'].concept) <= evidence


def test_curriculum_measurements_match_hand_counts_and_lexical_report(monkeypatch):
    from IdentityRows import mint
    from ConceptTrust import observe
    from corpus_lessons import lexical_report
    f, words, _, _ = operation_fixture(monkeypatch)
    owner = f.cs
    third = owner.interpret.lookup_word([4,7], (), form='blick')
    # Two of three words have ten views; the new third has none.
    a, b = mint(owner, torch.eye(owner.nDim)[3]), mint(owner, torch.eye(owner.nDim)[4])
    for _ in range(2):
        observe(owner, a, torch.eye(owner.nDim)[3])
    owner.interpret.define(words[0], a)
    owner.interpret.define(words[1], b)
    speech = f.language.parts_of_speech
    learned_operation(speech, 'lift', (words[0],), (words[1],), count=3)
    measured = owner.trust_measurements()
    assert measured['admitted_words'] == 3 and measured['stable_words'] == 2
    assert measured['stable_word_fraction'] == pytest.approx(2/3)
    assert measured['named_words'] == 2 and measured['denoting_words'] == 1
    assert measured['denoting_word_fraction'] == .5
    assert measured['operations']['lift']['uses'] == 3
    assert measured['operations']['lift']['definiteness'] == pytest.approx((-math.expm1(-1.),)*2)
    assert 'trust' not in measured['operations']['lift']
    assert positive_length(owner, speech.seats['lift'].concept) > 0
    model = SimpleNamespace(_concept_owner=lambda: owner, languageSpace=f.language)
    report = lexical_report(model, ['dax','wug','blick'])
    assert report['trust_measurements'] == measured
    assert report['forms']['blick']['word'] == third


def test_curriculum_readiness_requires_a_measured_radius_not_the_operand_fallback():
    owner = paired_fixture(meaning_pairs=0)[0]
    word = owner.interpret.lookup_word([4], (), form='dax')
    assert positive_length(owner, word) == 1.
    measured = owner.trust_measurements()
    assert measured['admitted_words'] == 1
    assert measured['stable_word_fraction'] == 0.
    assert measured['named_words'] == 0 and measured['denoting_word_fraction'] == 0.


def test_bind_menu_excludes_unknown_result_order_and_preserves_legal_referents(monkeypatch):
    from Memory import STMChunks
    from ReferenceContext import prepare_operands
    from Meaning import ConceptualMeaning
    f = SentenceFixture(monkeypatch)
    rule = next(rule for rule in f.binary if rule.method_name == 'bind')
    mention = torch.eye(8)[0].reshape(1,1,8)
    def menu(state, order):
        return prepare_operands(mention.expand(1,2,8), torch.full((1,2),-1),
            torch.zeros(1,2,dtype=torch.long), torch.zeros(1,2,dtype=torch.long),
            orders=torch.full((1,2),order), rules=(rule,), unary_rules=(),
            active=torch.tensor([[True,False]]), chunk_state=state, bank=None, live=None)
    empty = STMChunks.empty(1,8,8)
    assert not menu(empty,-1)['binary_valid'].any()
    assert menu(empty,0)['binary_valid'].sum() == 1  # Valid dummy retains order zero.
    known,_ = STMChunks.deposit(empty,mention[:,0],torch.tensor([True]),identity=42,order=2)
    selected = menu(known,-1)
    assert selected['binary_valid'].sum() == 1  # A real referent supplies its own order.
    assert selected['binary_refs'][...,1][selected['binary_valid']].tolist() == [42]
    with pytest.raises(ValueError,match='order must be a nonnegative integer'):
        Clause(ConceptualMeaning.from_description(mention[0,0]),order=-1)
