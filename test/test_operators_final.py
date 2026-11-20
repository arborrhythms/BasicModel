"""Final operators update: independent switches and catalogue §5–§6."""
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch






def test_predictor_priming_preserves_both_lanes_and_is_detached():
    from test_review13_subspaces import fixture
    owner,*_=fixture()
    owner._expected_knowing[0,:3]=torch.tensor([[.3,.7],[.9,.1],[0.,0.]])
    before=owner._expected_knowing.clone();rng=torch.get_rng_state().clone()
    result=owner.relevance_weights(batch=1)
    torch.testing.assert_close(result[0,:3],torch.tensor([1.7,1.9,1.]))
    assert not result.requires_grad
    assert torch.equal(before,owner._expected_knowing) and torch.equal(rng,torch.get_rng_state())


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


def test_meaning_width_zero_disables_bootstrap_without_changing_form(monkeypatch, eager_reading):
    from test_mm_xor import _fresh_model
    from util import TheXMLConfig
    from nomination_fixtures import readings_before_nomination
    model, _, _ = _fresh_model('data/XOR_grammar.xml')
    try:
        # The forms are words from the reading in which MTM nominates them.
        read = lambda: model.forward(model.inputSpace.prepInput(['hello world', 'loving there']))
        readings_before_nomination(model, read)
        with torch.no_grad():
            read()
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


def test_sentences_keep_distinct_occurrences(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    f.noun('cat', order=1)
    clause = f.clause(('change', ('lower_unary', 'cat'), 'sleeps'))
    writes_a, writes_b = {}, {}
    a = f.store.write_clause(clause,   written_rows=writes_a)
    b = f.store.write_clause(clause,   written_rows=writes_b)
    assert all(key not in writes_a and key not in writes_b
               for child in clause.children for key in (id(child),))
    assert f.store.occurrence_of(a) != f.store.occurrence_of(b)
    torch.testing.assert_close(f.store.slots[a], f.store.slots[b])
    assert len(f.store) == 2


def test_definition_sentence_joins_identities_beside_the_interpret_definition(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    f = SentenceFixture(monkeypatch)
    clause = f.clause(('part', 'bachelor', 'unmarried-man'))
    first = f.store.write_clause(clause, )
    assert len(f.store) == 0 and first[0] == 'sym'
    row = f.store.row(first)
    assert row['refs'].tolist() == [f.noun('bachelor'), f.noun('unmarried-man')]
    assert row['readings'] == ()  # one Part direction is not an equality
    assert f.noun('bachelor') != f.noun('unmarried-man')
    assert f.cs.definitions.objects(f.words['bachelor'][0]) == (f.noun('bachelor'),)
    assert f.registry.operation_spec('define').predicate_identity == 'operation:define'
    import Queries
    assert not hasattr(Queries, 'GrammaticalQueryRegistry')


def test_parameterized_form_map_cannot_read_or_normalize_the_meaning_complement():
    from Language import ChangeLayer, _BinaryGrammarOpAdapter
    layer = ChangeLayer(8, 8)
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


def test_unary_lowering_reads_existing_order_without_minting_a_composing_argument(monkeypatch):
    from test_clause_acceptance import SentenceFixture
    from Language import LanguageSpace
    f = SentenceFixture(monkeypatch)
    source = f.noun('cat', order=2)
    word = f.words['cat'][0]
    member = f.cs.new_concept(order=1)
    with torch.no_grad():
        f.cs.similarity_codebook.W[f.cs._concept_row_of(member)].sigmoid_()
    f.cs.interpret.define(word, member)
    program = f.program(('change', ('lower_unary', 'cat'), 'sleeps'))
    before = len(f.cs.definitions._store()), len(f.cs._concept_allocator.concepts)
    refs, orders = LanguageSpace.resolve_lexical_references(f.language, f.cs,
        program.word_rows, program.concept_ids, program.actions,
        leaf_orders=torch.tensor([2, 0]), admit=True)
    assert (len(f.cs.definitions._store()), len(f.cs._concept_allocator.concepts)) == before
    assert program.concept_ids[0] == source
    assert refs[0] == member and orders[0] == 1
    from chunk_fixtures import record
    from ClauseJournal import finish_clause
    reading, _, _ = record(f, ('change', ('lower_unary', 'cat'), 'sleeps'))
    clause = finish_clause(f.language, reading, registry=f.registry)
    assert clause.order == 0 and clause.children[0].order == 1
    written = {}
    row = f.store.write_clause(clause, written_rows=written)
    # The native lowered participant has its own occurrence. The removed
    # binary determiner's inline-only child is no longer this contract.
    child = written[id(clause.children[0])]
    assert f.store.row(row)['order'] == f.store.row(child)['order'] == 1
    assert len(f.store) == 2
    assert len(f.store.parts) == before[0]
    assert f.store.raw_meaning_of(child).role_refs[0] == ('sym', member)
    assert f.store.raw_meaning_of(row).role_refs[0] == f.store.occurrence_of(child)
    assert (len(f.cs.definitions._store()), len(f.cs._concept_allocator.concepts)) == before
    assert not f.cs.components.individual_ids


def test_determiner_reverse_uses_a_marker_witness():
    from Language import LowerLayer
    value, marker = torch.tensor([[.2, .4]]), torch.tensor([[.9, .1]])
    left, right = LowerLayer().reverse(value, marker=marker)
    torch.testing.assert_close(left, marker)
    torch.testing.assert_close(right, value)
    torch.testing.assert_close(LowerLayer().compose(left, right), value)
