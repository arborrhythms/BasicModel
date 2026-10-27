"""Selected grammar roles resolve one form across persistent concept orders."""
from types import SimpleNamespace

import pytest
import torch

import Language
import Spaces
from order_fixtures import force_order
from Models import BasicModel
from test_cs_sparse_weights import _cs


def vocabulary():
    cs = _cs(nS=256, order=3)
    word, _ = cs.interpret_word([7], [], key='cat')
    from reading_fixtures import supply_object
    obj = supply_object(cs, word)
    ids = [obj]
    for order in range(1, 3):
        cid = cs.new_concept()
        if ids:
            cs.add_part(cid, ('sym', ids[-1]))
        cs._csw_concept_row(order, cid)
        cs.interpret.define(word, cid)
        cs.bind_word_concept('cat', cid)
        ids.append(cid)
    row = cs._csw_row_of(ids[0])
    return cs, ids, row


@pytest.mark.parametrize('context, order', [('event', 0), ('particular', 1),
    ('name', 1), ('pronoun', 1), ('kind', 2)])
def test_selected_grammar_context_selects_order(context, order):
    cs, ids, row = vocabulary()
    grammar = Language.Grammar()
    grammar.configure({'compose': {'rule': {
        '_': 'lower_argout1 = lower.forward(lower_argin1, lower_argin2)',
        'reference': 'argin2'}}})
    # This fixture tests order resolution without determiner admission.
    rules = [r._replace(determiner_mode=None, reference_modes=(('argin2', context),))
             for r in grammar.rules if r.method_name == 'lower']
    language = SimpleNamespace(_compose_binary_rules=rules, _compose_unary_rules=[])
    actions = torch.tensor([[0, 0, 0], [0, 0, 1], [1, 0, -1]])
    got, orders = Language.LanguageSpace.resolve_lexical_references(
        language, cs, torch.tensor([-1, row]), torch.tensor([-1, ids[0]]), actions,
        leaf_orders=torch.tensor([0, order+1 if order else 0]))
    # The source kind survives the member-order request. Closing chooses
    # the token's identity independently of this derivation carrier.
    assert got.tolist() == [-1, ids[0] if order == 1 else ids[order]]
    assert orders.tolist() == [-1, order]
    assert cs.word_concepts('cat') == tuple(sorted([cs.definitions.word(form='cat'), *ids]))


def test_capture_owns_grammar_resolution_without_changing_reconstruction():
    cs, ids, row = vocabulary()
    rule = SimpleNamespace(reference_roles=('argin2',))
    language = SimpleNamespace(_compose_binary_rules=[rule], _compose_unary_rules=[])
    language.resolve_lexical_references = lambda *args, **kwargs: Language.LanguageSpace.resolve_lexical_references(language, *args, **kwargs)
    host = SimpleNamespace(languageSpace=language, _concept_owner=lambda: cs,
                           symbolSpace=SimpleNamespace(_word_reference_orders=torch.tensor([[0, 1]])),
                           _reconstruction_stack=lambda: None)
    leaves = torch.randn(1, 2, 8)
    actions = torch.tensor([[[0, 0, 0], [0, 0, 1], [1, 0, -1]]])
    entry, = BasicModel._program_entries(host,
        (torch.tensor([[0, 1]]), actions, torch.zeros(1, 3, dtype=torch.long),
         torch.tensor([[-1, -1, 0]])),
        leaves, torch.tensor([[-1, row]]), torch.tensor([[-1, row]]),
        torch.ones(1, 2), torch.zeros(1, 3, 8), concept_ids=torch.tensor([[-1, ids[0]]]))
    assert entry.reference_ids.tolist() == [-1, ids[0]]
    assert entry.reference_orders.tolist() == [-1, 1]
    torch.testing.assert_close(entry.leaves, leaves[0])
    assert entry.concept_ids.tolist() == [-1, ids[0]]
    assert entry.detached().reference_ids.tolist() == [-1, ids[0]]


def test_order_three_relation_over_kinds_is_read_from_its_stamp():
    """Operand positions and the configured storage rung impose no order ceiling."""
    from ReferenceContext import reference_requests, denoted_order
    cs = _cs(nS=256, order=1)
    word, _ = cs.interpret_word([7], [], key='statement')
    from reading_fixtures import supply_object
    raw = supply_object(cs, word)
    individual = cs.new_concept()
    cs.add_part(individual, ('sym', raw))
    kinds = [cs.new_concept(), cs.new_concept()]
    for kind in kinds:
        cs.add_part(kind, ('sym', individual))
    relation = cs.new_concept()
    for kind in kinds:
        cs.add_part(relation, ('sym', kind))
    assert [cs._concept_source_order(kind) for kind in kinds] == [2, 2]
    assert cs._concept_source_order(relation) == 3
    cs._csw_concept_row(3, relation)
    cs.interpret.define(word, relation)
    cs.bind_word_concept('statement', relation)
    grammar = Language.Grammar()
    grammar.configure({'compose': {'rule': {
        '_': 'generic_argout1 = generic.forward(generic_argin1)',
        'reference': 'argin1'}}})
    rule = next(rule for rule in grammar.rules if rule.method_name == 'generic')
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    actions = torch.tensor([[0, -1, 0], [2, 0, -1]])
    stamps = torch.tensor([cs._concept_source_order(relation)])
    refs, orders = Language.LanguageSpace.resolve_lexical_references(language, cs,
        torch.tensor([cs._csw_row_of(raw)]), torch.tensor([relation]), actions, leaf_orders=stamps)
    assert refs.tolist() == [relation] and orders.tolist() == [3]
    assert reference_requests(language, actions, leaf_orders=stamps) == {0: (3, 'generic')}
    assert denoted_order(rule, (3,), (), None) == 3
    assert denoted_order(rule, (17,), (), None) == 17
    # A binding operation retains its mode at the same high order. An
    # individual with identical content is not that relation's identity.
    from ReferenceContext import ReferenceBank, prepare_operands
    value = torch.tensor([[[.2, .3, .4, .5]]])
    bank = ReferenceBank(torch.tensor([[relation, individual]]), value.expand(1, 2, 4),
        torch.ones(1, 2, dtype=torch.bool), torch.tensor([[True, False]]),
        value[:, 0], torch.tensor([True]), orders=torch.tensor([[3, 1]]))
    binder = rule._replace(reference_modes=(('argin1', 'pronoun'),))
    live = (value, torch.tensor([[raw]]), stamps[None], torch.tensor([[False]]), torch.tensor([[0]]))
    proposals = prepare_operands(value, torch.tensor([[raw]]), torch.tensor([[0]]), torch.tensor([[0]]),
        orders=stamps[None], rules=(), unary_rules=(binder,), bank=bank, live=live,
        active=torch.tensor([[True]]))
    valid = proposals['unary_valid'][0, 0]
    assert set(proposals['unary_refs'][0, 0, valid, 0].tolist()) == {relation}
    # Choosing a member cannot keep the containing relation's own identity.
    # Its same-content kind candidate has a different order stamp.
    bank = bank._replace(ids=torch.tensor([[relation, kinds[0]]]),
                         orders=torch.tensor([[3, 2]]))
    lower = binder._replace(determiner_mode='bind', reference_modes=(('argin1', 'bind'),))
    proposals = prepare_operands(value, torch.tensor([[relation]]), torch.tensor([[0]]),
        torch.tensor([[0]]), orders=stamps[None], rules=(), unary_rules=(lower,),
        bank=bank, live=live, active=torch.tensor([[True]]))
    valid = proposals['unary_valid'][0, 0]
    assert set(proposals['unary_refs'][0, 0, valid, 0].tolist()) == {kinds[0]}


def test_form_sets_survive_checkpoint_and_missing_order_stays_unknown():
    from test_structural_checkpoint import _model_with
    cs, ids, row = vocabulary()
    source = _model_with(cs, SimpleNamespace())
    source.symbolSpace = torch.nn.Module()
    source.symbolSpace.ltm_store = cs.definitions._store()
    saved = source._collect_structural_extras()
    restored = _cs(nS=256, order=3)
    target = _model_with(restored, SimpleNamespace())
    target.symbolSpace = torch.nn.Module()
    target.symbolSpace.ltm_store = restored.definitions._store()
    target.symbolSpace.ltm_store.load_state_dict(source.symbolSpace.ltm_store.state_dict())
    target._restore_structural_extras(saved)
    assert restored.word_concepts('cat') == tuple(sorted([restored.definitions.word(form='cat'), *ids]))
    assert restored.resolve_word_concept('cat', order=3) is None
    assert restored.resolve_word_concept('cat', order=1) == ids[1]


def test_resolved_reference_supplies_meaning_but_keeps_input_provenance(monkeypatch):
    from dataclasses import replace
    from test_selected_relation_meaning import _program_owner
    cs, grammar, registry, language, leaves, program, a, b = _program_owner(monkeypatch)
    original = program()
    resolved = replace(original, reference_ids=torch.tensor([b[1], b[1]]),
                       reference_orders=torch.tensor([1, 1]))
    meaning = language.program_meaning(resolved, registry)
    assert meaning.role_refs[0] == b
    torch.testing.assert_close(meaning.roles[0], registry._payload(b))
    torch.testing.assert_close(resolved.leaves, original.leaves)
    torch.testing.assert_close(resolved.concept_ids, original.concept_ids)


def test_ambiguous_particular_requires_a_carried_reference():
    cs, ids, row = vocabulary()
    other = cs.new_concept()
    cs.add_part(other, ('sym', ids[0]))
    cs._csw_concept_row(1, other)
    cs.bind_word_concept('cat', other)
    assert cs.resolve_word_concept('cat', order=1) is None
    assert cs.resolve_word_concept('cat', order=1, previous=other) == other


@pytest.mark.parametrize('ambiguous', (False, True))
def test_order_zero_reference_agrees_before_and_after_admission(ambiguous):
    from Spaces import _concept_alloc_of
    cs = _cs(nS=256, order=3)
    word, _ = cs.interpret_word([7], [], key='token')
    if ambiguous:
        for _ in range(2):
            obj = cs.new_concept()
            force_order(cs, obj, 0)
            cs._csw_concept_row(0, obj)
            cs.interpret.define(word, obj)
            cs.bind_word_concept('token', obj)
    grammar = Language.Grammar()
    grammar.configure({'compose': {'rule': {
        '_': 'generic_argout1 = generic.forward(generic_argin1)',
        'reference': 'argin1'}}})
    rule = next(r for r in grammar.rules if r.method_name == 'generic')
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    bank, next_id = cs.similarity_codebook.W.clone(), _concept_alloc_of(cs).next_id
    results = [Language.LanguageSpace.resolve_lexical_references(language, cs,
        torch.tensor([cs._csw_row_of(word)]), torch.tensor([word]),
        torch.tensor([[0, -1, 0], [2, 0, -1]]), leaf_orders=torch.tensor([0]), admit=admit)
        for admit in (False, True)]
    for refs, orders in results:
        assert refs.tolist() == [-1 if ambiguous else word]
        assert orders.tolist() == [0]
    assert _concept_alloc_of(cs).next_id == next_id
    torch.testing.assert_close(cs.similarity_codebook.W, bank, rtol=0, atol=0)
