"""Final operators update: independent switches and catalogue §5–§6."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch


def test_centroid_moves_without_losing_identity_and_caps_containment():
    from SymbolCentroid import place
    forms = {0: torch.tensor([1., 0., 0.]), 1: torch.tensor([1., 1., 0.]),
             2: torch.tensor([0., 0., 1.]), 3: torch.tensor([1., 0., 0.])}
    parts = {0: {'a'}, 1: {'a', 'b'}, 2: {'c'}, 3: {'a'}}
    pairs = {(0, 2): {(11, 0), (12, 0)}, (1, 3): {(13, 0)}}
    codes, audit = place(forms, parts, pairs)
    assert audit['before']['violating_pairs'] > 0
    assert audit['after']['violating_pairs'] == audit['below_lower'] == 0
    assert audit['identity_recovered'] and 0 < audit['share_moved'] < 1
    assert audit['cosine_after'] < 1
    assert torch.equal(codes[0], codes[3])
    assert not any(value.requires_grad for value in codes.values())
    off, audit = place(forms, parts, pairs, enabled=False)
    assert audit['moved'] == 0
    for row in forms:
        torch.testing.assert_close(off[row], forms[row], rtol=0, atol=0)


def test_adjacent_wholes_rewitness_compact_and_checkpoint():
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    store = TernaryTruthStore(4, capacity=4)
    value = ConceptualMeaning.from_description(torch.ones(4))
    discarded = store.append_meaning(value, document_key='gone')
    store.origin[discarded] = store.ORIGIN_USER
    row = store.append_meaning(value, document_key='kept')
    for _ in range(400):
        store.witness_adjacency(row, [2, 3, 2])
    address = int(store.address_keys[row])
    assert store._adjacent_postings == {(2, 3): {(address, 0)}, (3, 2): {(address, 1)}}
    store.witness_adjacency(discarded, [7, 8])
    store.clear_origin(store.ORIGIN_USER)
    assert store.index_of_row(address) == 0 and (7, 8) not in store._adjacent_postings
    loaded = TernaryTruthStore(4, capacity=4)
    loaded.load_state_dict(store.state_dict())
    assert loaded._adjacent_postings == store._adjacent_postings
    store.witness_adjacency(0, [2, 4])
    assert store._adjacent_postings == {(2, 4): {(address, 0)}}


def test_membership_priming_shares_degree_budget_and_is_detached(monkeypatch):
    import MereologicalCodes
    from util import TheXMLConfig
    from Layers import TernaryTruthStore
    from Meaning import ConceptualMeaning
    store = TernaryTruthStore(4, capacity=4)
    store.append_meaning(ConceptualMeaning.from_description(torch.ones(4)))
    owner = SimpleNamespace(_closed_clause_store=lambda: store)
    monkeypatch.setattr(MereologicalCodes, 'occurrence_memberships', lambda *_: {0: {0}, 2: {0}})
    monkeypatch.setitem(TheXMLConfig.data.setdefault('ConceptualSpace', {}), 'membershipPriming', True)
    surface = torch.tensor([[4., 1., 1.]])
    before = torch.get_rng_state().clone()
    edges = (torch.tensor([0]), torch.tensor([1]), torch.tensor([2.]))
    assert MereologicalCodes.diffuse_memberships(owner, surface, edges, .3, torch.tensor([True]))
    torch.testing.assert_close(surface, torch.tensor([[3.25, 1.6, 1.15]]))
    assert torch.equal(before, torch.get_rng_state()) and not surface.requires_grad
    assert surface.sum() == pytest.approx(6.)
    monkeypatch.setitem(TheXMLConfig.data['ConceptualSpace'], 'membershipPriming', False)
    old = surface.clone()
    assert not MereologicalCodes.diffuse_memberships(owner, surface, edges, .3, torch.tensor([True]))
    torch.testing.assert_close(old, surface)


def test_bilattice_rule_survives_operator_rename():
    from Language import ConjunctionLayer, DisjunctionLayer, NotLayer
    for cls in (ConjunctionLayer, DisjunctionLayer):
        op = cls(); op.meaning_layout = (2, 6)
        a = torch.tensor([[.4, .6, .1, .3, .5, .8]])
        b = a.flip(-1)
        expected = op.compose(a, b)
        op.rule_name = 'opaque'
        torch.testing.assert_close(op.compose(a, b), expected)
    op = NotLayer(); op.meaning_layout = (2, 6); op.rule_name = 'opaque'
    torch.testing.assert_close(op(op(a)), a)


def test_meaning_width_zero_disables_bootstrap_without_changing_form(monkeypatch):
    from test_mm_xor import _fresh_model
    from util import TheXMLConfig
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(['hello world', 'loving there']))
        derived = model._concept_owner().similarity_codebook.mereology
        assert derived.occurrence_terms()
        from MereologicalCodes import MereologicalCodes
        monkeypatch.setitem(TheXMLConfig.data['ConceptualSpace'], 'meaningWidth', 0)
        off = MereologicalCodes(derived.owner, torch.zeros(1, derived.code_width),
            percept_width=derived.percept_width, percept_event_width=derived.percept_event_width)
        assert off.context_width == 0 and off.occurrence_terms() == {}
        rows = sorted(derived.occurrence_terms())
        derived._context = None; derived._form_context = None
        active, absent = derived.derive(rows), off.derive(rows)
        torch.testing.assert_close(active[:, :104], absent[:, :104])
        assert not absent[:, 104:].any() and active[:, 104:].any()
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_determiner_marker_has_no_content_and_binding_uses_an_address():
    from Language import LowerLayer, GenericLayer
    from ReferenceContext import ReferenceBank, resolve_operand
    kind = torch.tensor([[[.2, .6, .9]]])
    op = LowerLayer(3, 3)
    torch.testing.assert_close(op.compose(torch.ones_like(kind), kind), kind)
    torch.testing.assert_close(op.compose(-torch.ones_like(kind), kind), kind)
    torch.testing.assert_close(GenericLayer().compose(kind), kind)
    bank = ReferenceBank(torch.tensor([[-7264]]), kind.clone(), torch.tensor([[True]]),
                         torch.tensor([[False]]), kind[:, 0], torch.tensor([False]))
    live = (torch.empty(1, 0, 3), *(torch.empty(1, 0, dtype=torch.long) for _ in range(2)),
            torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0, dtype=torch.long))
    args = dict(bank=bank, live=live, active=torch.tensor([[True]]))
    fresh = resolve_operand(kind, torch.tensor([[21]]), torch.tensor([[0]]), mode='mint', **args)
    bound = resolve_operand(kind, torch.tensor([[21]]), torch.tensor([[0]]), mode='bind', **args)
    assert int(fresh[1]) == -1 and int(bound[1]) == -7264
    assert fresh[3].all() and bound[3].all()
    torch.testing.assert_close(bound[0], kind)


def test_determiner_mints_distinct_occurrence_and_universal_stays_high(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', ('lower', 'a', 'cat'), 'sleeps'))
    a = f.store.write_clause(clause, document_key='first', sentence_index=1)
    b = f.store.write_clause(clause, document_key='second', sentence_index=1)
    assert f.store.refs[a, 0] != f.store.refs[b, 0]
    first = f.store.index_of_row(int(f.store.refs[a, 0]))
    second = f.store.index_of_row(int(f.store.refs[b, 0]))
    assert first is not None and second is not None
    torch.testing.assert_close(f.store.slots[first], f.store.slots[second])
    universal = next(r for r in f.grammar.rules_upward if r.determiner_mode == 'kind')
    assert universal.order_delta == 0 and dict(universal.reference_orders)['I2'] == 2
    renamed = universal._replace(method_name='opaque')
    assert renamed.determiner_mode == 'kind'


def test_relative_slot_names_its_row_instead_of_an_operand():
    from ClauseScope import ClauseScope
    from References import address_code
    buffer = torch.ones(1, 2, 8)
    scope = torch.tensor([[[ClauseScope.RELATIVE | ClauseScope.LOCAL, -26], [0, -1]]])
    result = ClauseScope.publish_name((buffer, torch.tensor([1])), scope, torch.tensor([3]))
    torch.testing.assert_close(result[0][0, 0], address_code(torch.tensor(-26), 8, like=buffer))
    assert not torch.equal(result[0][0, 0], buffer[0, 0])
    torch.testing.assert_close(result[0][0, 1], buffer[0, 1])


def test_definition_sentence_joins_identities_beside_the_interpret_definition(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from ClauseRow import predicate_identity
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('equal', 'bachelor', 'unmarried-man'))
    first = f.store.write_clause(clause, document_key='definition', sentence_index=1)
    assert len(f.store) == 2 and f.store.refs[0].tolist() == f.store.refs[1, [2, 1, 0]].tolist()
    assert int(f.store.refs[first, 1]) == predicate_identity('part')
    assert f.noun('bachelor') != f.noun('unmarried-man')
    assert f.cs.definitions.objects(f.words['bachelor'][0]) == (f.noun('bachelor'),)
    assert f.registry.operation_spec('equal').predicate_identity == 'operation:equal'
    import Queries
    assert not hasattr(Queries, 'GrammaticalQueryRegistry')


def test_parameterized_form_map_cannot_read_or_normalize_the_meaning_complement():
    from Language import LiftLayer, _BinaryGrammarOpAdapter
    layer = LiftLayer(8, 8)
    layer.meaning_layout = (4, 8)
    op = _BinaryGrammarOpAdapter(layer)
    a = torch.tensor([[.1, .2, .3, .4, .6, .7, .8, .9]])
    b = a.flip(-1)
    actual = op(a, b)
    padded = [torch.cat((v[:, :4], torch.zeros_like(v[:, 4:])), -1) for v in (a, b)]
    expected = layer.compose(*padded)[:, :4]
    torch.testing.assert_close(actual[:, :4], expected)
    torch.testing.assert_close(actual[:, 4:], a[:, 4:])
    changed = op(torch.cat((a[:, :4], a[:, 4:] * 7), -1), b)
    torch.testing.assert_close(changed[:, :4], actual[:, :4])
    torch.testing.assert_close(changed[:, 4:], a[:, 4:] * 7)


def test_determiner_kept_closing_is_the_only_mint_and_lowers_once(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import finish_reading
    from Language import LanguageSpace
    f = SentenceFixture(monkeypatch)
    program = f.program(('lift', ('lower', 'a', 'cat'), 'sleeps'))
    before = len(f.cs.definitions._store())
    refs, orders = LanguageSpace.resolve_lexical_references(f.language, f.cs,
        program.word_rows, program.concept_ids, program.actions, admit=True)
    assert len(f.cs.definitions._store()) == before
    # The selected mint carries no existing column. Dictionary capture may
    # not substitute the noun's concept for that individual choice.
    assert refs[1] == -1 and orders[1] == 1
    reading = replace(program, reference_ids=refs, reference_orders=orders,
                      leaf_orders=torch.tensor([0, 2, 0]))
    clause = finish_reading(f.language, reading, registry=f.registry)
    assert clause.order == clause.children[0].order == 1
    row = f.store.write_clause(clause, document_key='determiner', sentence_index=1)
    assert f.store.index_of_row(int(f.store.refs[row, 0])) is not None


def test_determiner_reverse_uses_a_marker_witness():
    from Language import LowerLayer
    value, marker = torch.tensor([[.2, .4]]), torch.tensor([[.9, .1]])
    left, right = LowerLayer().reverse(value, marker=marker)
    torch.testing.assert_close(left, marker)
    torch.testing.assert_close(right, value)
    torch.testing.assert_close(LowerLayer().compose(left, right), value)
