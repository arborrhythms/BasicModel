"""Final operators update: independent switches and catalogue §5–§6."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch






def test_location_priming_shares_excess_budget_and_is_detached():
    from Priming import propagate
    from test_priming_diffusion import owner as fixture_owner, directions
    owner = fixture_owner(directions([0,10,80], dtype=torch.float32))
    surface = torch.tensor([[4., 1., 1.]])
    before = torch.get_rng_state().clone()
    propagate(owner, surface, torch.tensor([True]))
    torch.testing.assert_close(surface, torch.tensor([[3.1, 1.9, 1.]]))
    assert torch.equal(before, torch.get_rng_state()) and not surface.requires_grad
    assert surface.sum() == pytest.approx(6.)


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
        assert derived.context_audit()
        from MereologicalCodes import MereologicalCodes
        monkeypatch.setitem(TheXMLConfig.data['ConceptualSpace'], 'meaningWidth', 0)
        off = MereologicalCodes(derived.owner, torch.zeros_like(derived.owner.similarity_codebook.W),
            percept_width=derived.percept_width, percept_event_width=derived.percept_event_width,
            identity_width=derived.identity_width, property_width=derived.property_width)
        assert off.context_width == 0 and off.context_audit() == []
        rows = [record['row'] for record in derived.context_audit()]
        derived.begin_forward()
        active, absent = derived.derive(rows), off.derive(rows)
        width = derived.percept_event_width
        torch.testing.assert_close(active[:, :width], absent[:, :width])
        assert not absent[:, width:].any() and active[:, width:].any()
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


def test_determined_sentences_keep_distinct_occurrences_and_universal_stays_high(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('lift', ('lower', 'a', 'cat'), 'sleeps'))
    writes_a, writes_b = {}, {}
    a = f.store.write_clause(clause, document_key='first', sentence_index=1, written_rows=writes_a)
    b = f.store.write_clause(clause, document_key='second', sentence_index=1, written_rows=writes_b)
    assert id(clause.children[0]) not in writes_a and id(clause.children[0]) not in writes_b
    assert f.store.occurrence_of(a) != f.store.occurrence_of(b)
    torch.testing.assert_close(f.store.slots[a], f.store.slots[b])
    assert len(f.store) == 2
    universal = next(r for r in f.grammar.rules_upward if r.determiner_mode == 'kind')
    from ReferenceContext import binding_spec
    assert not hasattr(universal, 'order_delta') and binding_spec(universal, 'argin2', 2) == (1, 'kind')
    renamed = universal._replace(method_name='opaque')
    assert renamed.determiner_mode == 'kind'


def test_definition_sentence_joins_identities_beside_the_interpret_definition(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('equal', 'bachelor', 'unmarried-man'))
    first = f.store.write_clause(clause, document_key='definition', sentence_index=1)
    assert len(f.store) == 0 and first[0] == 'sym'
    row = f.store.row(first)
    assert row['refs'].tolist() == [f.noun('bachelor'), f.noun('unmarried-man')]
    assert row['readings'] == ('equality',)
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
    changed = op(torch.cat((a[:, :4], a[:, 4:] * .7), -1), b)
    torch.testing.assert_close(changed[:, :4], actual[:, :4])
    torch.testing.assert_close(changed[:, 4:], a[:, 4:] * .7)


def test_determiner_lowering_keeps_order_without_minting_a_composing_argument(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from reading_fixtures import finish_reading
    from Language import LanguageSpace
    f = SentenceFixture(monkeypatch)
    program = f.program(('lift', ('lower', 'a', 'cat'), 'sleeps'))
    before = len(f.cs.definitions._store())
    refs, orders = LanguageSpace.resolve_lexical_references(f.language, f.cs,
        program.word_rows, program.concept_ids, program.actions,
        leaf_orders=torch.tensor([0, 2, 0]), admit=True)
    assert len(f.cs.definitions._store()) == before
    # The source kind remains available to the close's inline lowering;
    # the member-order request does not allocate an individual column.
    assert refs[1] == program.concept_ids[1] and orders[1] == 1
    reading = replace(program, reference_ids=refs, reference_orders=orders,
                      leaf_orders=torch.tensor([0, 2, 0]))
    clause = finish_reading(f.language, reading, registry=f.registry)
    assert clause.order == 0 and clause.children[0].order == 1
    written = {}
    row = f.store.write_clause(clause, document_key='determiner', sentence_index=1,
                               written_rows=written)
    assert id(clause.children[0]) not in written
    assert f.store.row(row)['order'] == 0 and len(f.store) == 1
    assert not len(f.store.parts) - before
    root = f.store.raw_meaning_of(row).role_refs[0]
    assert root == ('sym', f.noun('cat'))
    assert not f.cs.components.individual_ids


def test_determiner_reverse_uses_a_marker_witness():
    from Language import LowerLayer
    value, marker = torch.tensor([[.2, .4]]), torch.tensor([[.9, .1]])
    left, right = LowerLayer().reverse(value, marker=marker)
    torch.testing.assert_close(left, marker)
    torch.testing.assert_close(right, value)
    torch.testing.assert_close(LowerLayer().compose(left, right), value)
