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


def test_first_composition_keeps_its_operation_when_another_merges_poles(monkeypatch):
    import Language
    from ClauseRow import attach_clause_index
    from Memory import LTM0
    from PartsOfSpeech import PartsOfSpeech, formal_arguments
    owner,bank,_,alloc = paired_fixture(meaning_pairs=4, capacity=160)
    grammar = Language.Grammar()
    grammar.load_from_grammar_file('complete.grammar')
    speech = PartsOfSpeech(formal_arguments(grammar.rules_upward))
    store = LTM0(owner.nDim,capacity=64)
    attach_clause_index(SimpleNamespace(languageSpace=SimpleNamespace(parts_of_speech=speech)),store,owner)
    words = [owner.interpret.lookup_word([part],(),form=form) for part,form in ((4,'dax'),(7,'wug'))]
    agreeing_views(owner,*words)
    leaves = tuple(Composition(bank.lookup_rows(owner._csw_row_of(word)),reference=word,word_id=word) for word in words)
    term = Composition(leaves[0].point,leaves,operation='disjunction',formal_name='disjunction')
    clause = Clause(ConceptualMeaning.from_description(term.point),composition=term,point=term.point)
    parts = store.parts
    first = store.write_clause(clause, evidence=(.8,.2))
    count = len(parts)
    second = store.write_clause(through_operation(clause, 'conjunction'), evidence=(.3,.7))
    row = store.location(first)[1]
    assert second == first and len(parts) == count
    assert parts.grammar_operation[row] == speech.operations.index('disjunction')
    assert (float(parts.truth_plus[row]),float(parts.truth_minus[row])) == pytest.approx((.8,.7))


def test_naming_clears_zero_use_operation_and_later_composition_cannot_restore_it(monkeypatch):
    from IdentityRows import mint
    f, words, clause, _ = operation_fixture(monkeypatch)
    owner, parts = f.cs, f.store.parts
    word = words[1]
    obj = mint(owner, torch.eye(owner.nDim)[3])
    agreeing_views(owner, obj, count=3)
    left = Composition(owner.similarity_codebook.lookup_rows(owner._csw_row_of(obj)), reference=obj, order=0)
    clause = replace(clause, composition=replace(clause.composition,
        children=(left, clause.composition.children[1])))
    clause = through_operation(clause, 'union')
    address = f.store.write_clause(clause, evidence=(.8,.2))
    row = f.store.location(address)[1]
    assert parts.find(obj,word) == row
    assert f.language.parts_of_speech.operation('union')['uses'] == 0
    assert float(parts.evidence(row)[0]) == 0.
    assert owner.interpret.define(word,obj,evidence=(.8,.2)) == row
    assert parts.grammar_operation[row] == -1
    scale = min(positive_length(owner,obj),positive_length(owner,word))
    torch.testing.assert_close(torch.stack(parts.evidence(row)),torch.tensor([.8,.2])*scale)
    assert owner.definitions.objects(word) == (obj,)
    # Once named, a complete clause may compress to an LTM0 point. Exercise
    # the later write of this same pair at its relation writer explicitly.
    assert parts.add(obj,word,truth_plus=.7,truth_minus=.1,
        grammar_operation=f.language.parts_of_speech.operations.index('union')) == row
    assert parts.grammar_operation[row] == -1
    assert owner.interpret.define(word,obj,evidence=(.9,.2)) == row
    assert parts.grammar_operation[row] == -1


def test_attachment_without_occupations_preserves_owner_catalogue_and_effective_pairs(monkeypatch):
    from ClauseRow import attach_clause_index
    from PartsOfSpeech import PartsOfSpeech
    f, _, clause, program = operation_fixture(monkeypatch)
    speech,parts = f.language.parts_of_speech,f.store.parts
    speech.observe_kept(program,f.binary,f.unary)
    refresh(speech.owner)
    address = f.store.write_clause(clause,evidence=(.8,.2))
    parts.evidence(f.store.location(address)[1])
    before = parts.effective_plus.clone(),parts.effective_minus.clone()
    catalogue,owner = parts.operation_catalog,parts._occupation_owner()
    attach_clause_index(SimpleNamespace(),f.store,f.cs)
    assert parts.operation_catalog == catalogue and parts._occupation_owner() is owner
    assert f.store._occupation_owner() is owner
    parts.refresh_effective()
    torch.testing.assert_close(parts.effective_plus,before[0])
    torch.testing.assert_close(parts.effective_minus,before[1])
    with pytest.raises(ValueError,match='catalogue'):
        parts.configure_operations(PartsOfSpeech(('different_argin1',)))


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


@pytest.mark.parametrize('uses', [1, 3])
def test_kept_lift_uses_cap_only_the_effective_pair(monkeypatch, uses):
    f, words, clause, program = operation_fixture(monkeypatch)
    speech, parts = f.language.parts_of_speech, f.store.parts
    for _ in range(uses):
        address = f.store.write_clause(clause, evidence=(.8,.6))
        speech.observe_kept(program, f.binary, f.unary)
        refresh(speech.owner)
    row = f.store.location(address)[1]
    assert parts.grammar_operation[row] == speech.operations.index('lift')
    measured = speech.operation('lift')
    assert measured['uses'] == uses
    expected_trust = -math.expm1(-uses/3)
    assert measured['definiteness'] == pytest.approx((expected_trust, expected_trust))
    assert measured['trust'] == pytest.approx(expected_trust)
    endpoints = min(positive_length(f.cs, word) for word in words)
    torch.testing.assert_close(torch.stack(parts.evidence(row)),
                               torch.tensor([.8,.6])*min(endpoints, expected_trust))
    assert (float(parts.truth_plus[row]), float(parts.truth_minus[row])) == pytest.approx((.8,.6))


def test_shared_word_teaches_another_argument_without_dividing_existing_authority(monkeypatch):
    f, words, clause, program = operation_fixture(monkeypatch)
    speech, parts = f.language.parts_of_speech, f.store.parts
    speech.observe_kept(program, f.binary, f.unary)
    refresh(speech.owner)
    address = f.store.write_clause(clause, evidence=(.8,.6))
    row = f.store.location(address)[1]
    before = torch.stack(parts.evidence(row)).clone()
    revision = speech.revision
    learned_operation(speech, 'verb', (words[0],), (), count=1)
    assert speech.revision != revision
    measure = speech.operation('lift')
    radius = -math.expm1(-1/3)
    assert measure['uses'] == 1 and measure['definiteness'] == pytest.approx((radius, radius))
    assert measure['trust'] == pytest.approx(radius)
    torch.testing.assert_close(torch.stack(parts.evidence(row)), before)


def test_fixed_operations_and_identity_mechanics_stay_at_one(monkeypatch):
    f, words, _, _ = operation_fixture(monkeypatch)
    speech = f.language.parts_of_speech
    for operation in ('part','whole','equal','generic','implies','assertPart','interpret','sign','bind'):
        assert speech.operation(operation)['trust'] == 1.
    address = f.store.write_clause(Clause(ConceptualMeaning.from_description(torch.zeros(f.cs.nDim)),
        relation='part', refs=(words[0], -1, words[1])), evidence=(.8,.6))
    row = f.store.location(address)[1]
    assert f.store.parts.grammar_operation[row] == speech.operations.index('part')
    assert speech.operation('part')['trust'] == 1.
    scale = min(positive_length(f.cs, word) for word in words)
    torch.testing.assert_close(torch.stack(f.store.parts.evidence(row)), torch.tensor([.8,.6])*scale)


def test_operation_counts_catalogue_column_and_pair_round_trip(monkeypatch):
    f, _, clause, program = operation_fixture(monkeypatch)
    speech, parts = f.language.parts_of_speech, f.store.parts
    for _ in range(3):
        speech.observe_kept(program, f.binary, f.unary)
        refresh(speech.owner)
    address = f.store.write_clause(clause, evidence=(.8,.6)); row = f.store.location(address)[1]
    expected = torch.stack(parts.evidence(row)).clone()
    speech_state, parts_state = copy.deepcopy(speech.state_dict()), copy.deepcopy(parts.state_dict())
    concept_state = copy.deepcopy(f.cs.similarity_codebook.state_dict())
    pointers = (parts.grammar_operation.data_ptr(), parts.truth_plus.data_ptr())
    parts.begin_epoch()
    speech.observe_kept(program, f.binary, f.unary)
    refresh(speech.owner)
    parts.begin_epoch()
    assert speech.operation('lift')['uses'] == 4  # Epoch restoration never unlearns uses.
    assert parts.grammar_operation[row] == speech.operations.index('lift')
    speech.load_state_dict(speech_state); parts.load_state_dict(parts_state)
    f.cs.similarity_codebook.load_state_dict(concept_state)
    assert speech.operation('lift')['uses'] == 3
    assert pointers == (parts.grammar_operation.data_ptr(), parts.truth_plus.data_ptr())
    torch.testing.assert_close(torch.stack(parts.evidence(row)), expected)
    assert parts.operation_catalog == speech.operations
    bad = copy.deepcopy(parts_state); bad['grammar_operation'][row] = len(speech.operations)
    with pytest.raises(ValueError, match='catalogue index'):
        parts.load_state_dict(bad)


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
    assert speech.operation('lift')['trust'] == 0.


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
    assert speech.operation('lift')['trust'] == pytest.approx(evidence)


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
    assert measured['operations']['lift']['trust'] == pytest.approx(-math.expm1(-1.))
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
