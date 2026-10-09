"""Round 3a: testimony enters the pair; field order and both poles persist."""
from dataclasses import replace

import pytest
import torch

from ClauseRow import Clause
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning


def store_and_clause():
    store = TernaryTruthStore(4, capacity=8)
    identities = iter(range(10, 100))
    store.configure_clause_index(allocate=lambda _point: next(identities),
                                 concept_point=lambda _cid: torch.ones(4))
    point = torch.tensor([.2, .4, .6, .8])
    return store, Clause(ConceptualMeaning.from_description(point), point=point)


def test_source_trust_enters_the_positive_pole():
    store, clause = store_and_clause()
    row = store.write_clause(clause, trust=.8)
    assert store.row(row)['trust'] == pytest.approx(.8)
    assert store.row(row)['evidence'] == pytest.approx((.8, 0.))
    assert 'trust' not in store.state_dict()


def test_pair_retains_both_and_neither_until_explicit_signed_reassertion():
    store, clause = store_and_clause()
    both = store.write_clause(clause, trust=.8, evidence=(.6, .6))
    neither = store.write_clause(clause, trust=.8, evidence=(0., 0.), document_key='neither')
    assert store.row(both)['evidence'] == pytest.approx((.6, .6))
    assert store.row(neither)['evidence'] == (0., 0.)
    store.set_trust(both, -.3)
    store.set_trust(neither, -.3)
    assert store.row(both)['evidence'] == pytest.approx((0., .3))
    assert store.row(neither)['evidence'] == pytest.approx((0., .3))
    assert store.row(both)['trust'] == store.row(neither)['trust'] == pytest.approx(-.3)
    store.set_evidence(both, .9, .2)
    assert store.row(both)['trust'] == pytest.approx(.7)


def test_clause_order_and_evidence_are_stored_without_a_program():
    store, clause = store_and_clause()
    clause = replace(clause, order=3, evidence=(.75, .5))
    row = store.write_clause(clause)
    assert store.row(row)['order'] == 3
    assert store.row(row)['evidence'] == pytest.approx((.75, .5))
    restored = TernaryTruthStore(4, capacity=8)
    restored.load_state_dict(store.state_dict(), strict=True)
    assert restored.row(row)['order'] == 3
    assert restored.row(row)['trust'] == pytest.approx(.25)
    assert restored.row(row)['evidence'] == pytest.approx((.75, .5))
    assert not hasattr(restored, 'clause_derivation')


def test_row_compaction_preserves_each_order_and_pole():
    store, clause = store_and_clause()
    first = store.write_clause(replace(clause, order=1, evidence=(.2, .1)))
    store.set_origin(first, store.ORIGIN_USER)
    second = store.write_clause(replace(clause, order=3, evidence=(.6, .5)), document_key='second')
    occurrence = store.occurrence_of(second)
    assert store.clear_origin(store.ORIGIN_USER) == 1
    assert store.occurrence_of(0) == occurrence
    assert store.row(0)['order'] == 3
    assert store.row(0)['evidence'] == pytest.approx((.6, .5))
    assert store.row(0)['trust'] == pytest.approx(.1)


def test_embedded_identification_does_not_grant_source_authority():
    store, clause = store_and_clause()
    child = replace(clause, evidence=(.8, .4), order=2)
    parent = replace(clause, children=(child,), refs=(('clause', 0), -1, -1),
                     evidence=(.9, .3), order=2)
    row = store.write_clause(parent, trust=.7)
    assert row == 1 and store.row(0)['trust'] == pytest.approx(.4)
    assert store.row(0)['evidence'] == pytest.approx((.8, .4))
    assert store.row(1)['trust'] == pytest.approx(.7)
    assert store.row(1)['evidence'] == pytest.approx((.7, 0.))


def test_open_reading_keeps_both_poles_and_its_order(monkeypatch):
    from reading_fixtures import finish_reading
    from test_clause_acceptance import SentenceFixture
    fixture = SentenceFixture(monkeypatch)
    reading = replace(fixture.program(('sum', 'cat', 'animal')),
        leaf_orders=torch.tensor([3, 3]),
        leaf_evidence=torch.tensor([[.8, .7], [.6, 0.]]))
    clause = finish_reading(fixture.language, reading, registry=fixture.registry)
    row = fixture.store.write_clause(clause)
    assert fixture.store.row(row)['order'] == 3
    assert fixture.store.row(row)['evidence'] == pytest.approx((.6, .7))
    assert fixture.store.row(row)['trust'] == pytest.approx(-.1)


def test_index_unfolding_reads_the_stored_order():
    store, clause = store_and_clause()
    seen = []
    def unfold(value, limit, *, order, work=None):
        seen.append(order)
        return (1,), 1, True
    store.configure_leaf_index(unfold=unfold)
    row = store.write_clause(replace(clause, order=3), trust=.8)
    store.reindex_meanings()
    assert seen == [3, 3]
    assert store.leaf_terms(row, 0) == (1,)


def test_abstract_unfold_descends_each_sigma_rung():
    from types import SimpleNamespace
    from MemoryIndex import unfold_idea
    basis = torch.eye(3)
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(),
        reverse_inverses=lambda _ops: (),
        generate_policy_logits=lambda value: value.new_zeros(len(value), 1))
    calls = []
    def descend(code, order):
        calls.append((code, order))
        return (basis[code - 1],)
    result = unfold_idea(language, basis, basis[2], 8, activation=torch.ones(3) * 2,
                         order=2, order_of=int, sigma_inverse=descend)
    assert result['complete'] and result['codes'] == (2, 1, 0)
    assert calls == [(2, 2), (1, 1)]
    assert result['spent'] == 3


def test_old_scalar_checkpoint_does_not_invent_evidence_or_order():
    store, clause = store_and_clause()
    store.write_clause(clause, trust=.8)
    state = store.state_dict()
    state['trust'] = torch.full((store.capacity,), .8)  # legacy provenance
    for name in ('order', 'c_plus', 'c_minus'):
        del state[name]
    restored = TernaryTruthStore(4, capacity=8)
    with pytest.warns(UserWarning, match='unknown order'):
        restored.load_state_dict(state, strict=True)
    assert restored.row(0)['order'] == -1
    assert restored.row(0)['evidence'] == (0., 0.)
    assert restored.row(0)['trust'] == 0.


def test_native_word_reference_accepts_both_without_touching_its_signed_value():
    from types import SimpleNamespace
    from Language import SymbolSpace
    owner = SimpleNamespace()
    signed = torch.tensor([[[.2]]])
    result = SymbolSpace.commit_word_reference_slab(owner, torch.tensor([[1]]), signed,
        torch.tensor([[True]]), orders=torch.tensor([[3]]), evidence=torch.tensor([[[.8, .6]]]))
    torch.testing.assert_close(result, signed, rtol=0, atol=0)
    torch.testing.assert_close(owner._word_reference_evidence, torch.tensor([[[.8, .6]]]), rtol=0, atol=0)


def test_withdrawing_testimony_clears_its_pair():
    store, clause = store_and_clause()
    row = store.write_clause(replace(clause, evidence=(.8, .6)), trust=.9,
                             origin=store.ORIGIN_USER, kind='fact')
    store.withdraw_origin(store.ORIGIN_USER)
    assert store.row(row)['trust'] == 0.
    assert store.row(row)['evidence'] == (0., 0.)


def test_luminosity_view_reads_signed_testimony_from_the_pair():
    from Layers import TruthLayer
    store, clause = store_and_clause()
    store.write_clause(replace(clause, evidence=(.8, .6)), trust=-.3,
                       origin=store.ORIGIN_USER, kind='fact')
    light = TruthLayer(4, max_truths=8)
    light.attach_ltm(store)
    assert light.sync_from_ltm() == 1
    assert light._trusts == pytest.approx([-.3])
    before = light.luminosity()
    store.set_trust(0, .9)
    light.sync_from_ltm()
    assert before == pytest.approx(-.15)
    assert light.luminosity() == pytest.approx(.45)
    assert light._trusts == pytest.approx([.9])


def test_native_sentence_carries_paired_identification_to_its_row(tmp_path, monkeypatch):
    import util
    from Language import SymbolSpace
    from test_meronomy_ladder import _build_ladder_variant
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _build_ladder_variant(tmp_path, 'row_schema', [
        ('<architecture>', '<architecture><ltmConsolidation>true</ltmConsolidation>')])
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    model._install_unit_span_fn()
    model.reconstruct_in_loop = False
    model.loss.reconstruction_scale = 0.
    commit = SymbolSpace.commit_word_reference_slab
    calls = []
    def paired(owner, rows, activations, active, **kwargs):
        calls.append(rows.shape)
        kwargs['evidence'] = activations.new_tensor([.8, .6]).expand(*rows.shape, 2)
        return commit(owner, rows, activations, active, **kwargs)
    monkeypatch.setattr(SymbolSpace, 'commit_word_reference_slab', paired)
    try:
        inputs = model.inputSpace.prepPackedInput([['1 plus 2']])
        model.runBatch(train=False, batchSize=1, split='runtime',
                       batch_override=(inputs, torch.zeros(1, 1, 0)))
        field = model._sentence_fields[0][0]
        store = model.symbolSpace.ltm_store
        row = store.index_of_row(field.row_id)
        assert calls and row is not None
        assert field.evidence == pytest.approx((.8, .6))
        assert store.row(row)['evidence'] == pytest.approx((.8, .6))
        assert store.row(row)['order'] == field.order
        assert store.row(row)['trust'] == pytest.approx(.2)
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()
