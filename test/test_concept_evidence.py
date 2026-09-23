"""Live paired evidence: snap, scoped composition, reverse and persistence."""
import pytest
import torch

import Spaces
from ConceptEvidence import admit, corners, symbols, union
from test_cs_sparse_weights import _cs, _mint_row


def field(cs, batch=1, occurrences=1):
    return torch.zeros(cs._order_caps()[0], batch, occurrences, 2)


@pytest.mark.parametrize('pi', [False, True])
@pytest.mark.parametrize('parts', [2, 4, 8])
def test_production_snap_background_is_neither(pi, parts):
    cs = _cs()
    cs.conceptual_pi = pi
    row = _mint_row(cs, 1, 101)
    for part in range(parts):
        cs.add_concept_edge(row, part, 1., conjunctive=pi)
    snap = cs.cs_snap_order0(torch.zeros(2, 8, 8))
    _, result = cs.cs_forward_content(snap, cs.similarity_codebook.getW())
    assert torch.count_nonzero(result) == 0
    torch.testing.assert_close(corners(symbols(result)[row])[:, 3], torch.ones(2))


def test_snap_keeps_both_until_symbol_readout():
    cs = _cs()
    cs.concept_evidence_floor = .2
    with torch.no_grad():
        cs.similarity_codebook.getW()[0].fill_(1.)
    event = torch.tensor([[[1.] * 8, [-1.] * 8]])
    snap = cs.cs_snap_order0(event)
    torch.testing.assert_close(snap[0, 0], torch.tensor([[1., 0.], [0., 1.]]))
    torch.testing.assert_close(symbols(snap)[0, 0], torch.ones(2))
    torch.testing.assert_close(corners(symbols(snap))[0, 0], torch.tensor([0., 0., 1., 0.]))
    assert symbols(cs.cs_snap_order0(torch.full((1, 8, 8), .1)))[0].count_nonzero() == 0


@pytest.mark.parametrize('conjunctive', [False, True])
@pytest.mark.parametrize('pair', [(0., 0.), (1., 1.)])
def test_dual_folds_preserve_both_and_neither(conjunctive, pair):
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    for col in (0, 1):
        cs.add_concept_edge(row, col, 1., conjunctive=conjunctive)
    a = field(cs)
    a[:2, 0, 0] = torch.tensor(pair)
    _, result = cs.cs_forward_content(a, torch.zeros(64, 8))
    torch.testing.assert_close(result[row, 0, 0], torch.tensor(pair), atol=1e-6, rtol=0)


def test_unknown_definition_is_neither_even_with_zero_edges():
    cs = _cs()
    cs.conceptual_pi = True
    rows = [_mint_row(cs, 1, 101 + i) for i in range(3)]
    cs.add_concept_edge(rows[1], 0)
    cs.add_concept_edge(rows[2], 0, conjunctive=True)
    _, result = cs.cs_forward_content(torch.ones_like(field(cs)), torch.zeros(64, 8))
    assert result[rows].count_nonzero() == 0


def test_kind_both_member_and_absent_member_is_both():
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    for col in (0, 1):
        cs.add_concept_edge(row, col, 1.)
    a = field(cs)
    a[:2, 0, 0] = torch.tensor([[1., 1.], [0., 1.]])
    _, result = cs.cs_forward_content(a, torch.zeros(64, 8))
    torch.testing.assert_close(result[row, 0, 0], torch.ones(2))


def test_xor_combination_preserves_occurrence_and_observed_zero():
    """The OR and AND features are already grounded at one occurrence.

    This checks concept composition, not item 11a's tower learning gate.
    The final example omits B's observation: unknown is not an observed 0.
    """
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    cs.add_concept_edge(row, 1, 1., conjunctive=True, negated=True)
    a = field(cs, batch=5, occurrences=2)
    a[0, :4, 0] = torch.tensor([[0., 1.], [1., 0.], [1., 0.], [1., 0.]])
    a[1, :4, 0] = torch.tensor([[0., 1.], [0., 1.], [0., 1.], [1., 0.]])
    a[0, 4, 0] = torch.tensor([1., 0.])
    _, before = cs.cs_forward_content(a, torch.zeros(64, 8))
    torch.testing.assert_close(before[row, :4, 0, 0], torch.tensor([0., 1., 1., 0.]))
    assert before[row, 4, 0].count_nonzero() == 0
    a[:, :, 1] = 1.  # unrelated occurrence cannot refute the first
    _, after = cs.cs_forward_content(a, torch.zeros(64, 8))
    torch.testing.assert_close(after[row, :, 0], before[row, :, 0])


def test_reverse_keeps_counterevidence_and_its_occurrence():
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    cs.add_concept_edge(row, 1, 1., conjunctive=True, negated=True)
    query = torch.zeros(sum(cs._order_caps()), 1, 2, 2)
    query[row, 0, 1, 0] = .81
    result = cs.cs_reverse_presence(query)
    torch.testing.assert_close(result[0, 0, 1], torch.tensor([.9, 0.]))
    torch.testing.assert_close(result[1, 0, 1], torch.tensor([0., .9]))
    assert result[:, :, 0].count_nonzero() == 0


def test_noise_admission_and_count_accumulation_are_explicit():
    assert admit(torch.full((64,), .1), .2).count_nonzero() == 0
    for count in (8, 64):
        admitted = admit(torch.full((count,), .24), .2)
        assert float(union(admitted[:, 0], 0)) == pytest.approx(1 - .95 ** count, abs=1e-6)


def test_part_values_have_only_one_checkpoint_owner():
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, .7, negated=True)
    cs.add_concept_edge(row, 1, .8, conjunctive=True)
    store = Spaces._concept_alloc_of(cs).layer()
    store.ensure_context()[row, 0] = .4
    assert not any('concept_parts_layer' in name for name in cs.state_dict())
    assert any(parameter is store.values for parameter in cs.parameters())
    saved = store.parts_extras()
    store.conjunctive.values.data.zero_()
    store.where.zero_()
    store.load_parts_extras(saved)
    assert store.conjunctive.values.item() == pytest.approx(.8)
    assert store.where[row, 0] == pytest.approx(.4)


def test_checkpoint_preserves_both_and_neither_in_occurrence_field():
    from types import SimpleNamespace
    from test_structural_checkpoint import _model_with
    cs = _cs()
    evidence = torch.zeros(sum(cs._order_caps()), 2, 3, 2)
    evidence[0, 0, 1] = 1.
    cs.subspace._concept_activations = evidence
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs()
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    actual = restored.subspace._concept_activations
    torch.testing.assert_close(actual, evidence)
    assert float(corners(actual)[0, 0, 1, 2]) == 1.
    assert float(corners(actual)[0, 1, 1, 3]) == 1.


def test_discovered_symbol_remains_available_after_nonuse():
    from test_attention_promotion import _fixture, _observe, _pool_rows
    cs, rows = _fixture()
    a, b, context = [r for _, r in rows[:3]]
    cs.concept_mint_threshold = .15
    _observe(cs, {a: 1., context: 1.})
    _observe(cs, {b: 1., context: 1.})
    row = _pool_rows(cs)[0]
    _observe(cs, {a: 1., context: 1.})
    assert cs.promotion_pass()
    store = Spaces._concept_alloc_of(cs).layer()
    assert not store.managed[row] and store.participation[row] == 1
    for _ in range(30):
        _observe(cs, {})
    assert store.participation[row] == 1
    a0 = field(cs)
    a0[a, 0, 0, 0] = 1.
    _, result = cs.cs_forward_content(a0, cs.similarity_codebook.getW())
    assert result[row, 0, 0, 0] == 1


def test_alternatives_normalize_each_update_before_discovery():
    from test_attention_promotion import _fixture, _observe, _pool_rows
    cs, rows = _fixture()
    alternatives = [r for _, r in rows[:4]]
    context = rows[4][1]
    row = None
    for step in range(40):
        _observe(cs, {alternatives[step % 4]: .9, context: .9})
        if step:
            row = _pool_rows(cs)[0]
            weights = dict(cs.concept_weights(row))
            assert max(weights.values()) == pytest.approx(1.)
    assert set(weights) == set(alternatives)
    assert Spaces._concept_alloc_of(cs).layer().participation[row] > .8


def test_context_allocation_uses_taper_span_not_inventory():
    cs = _cs(nS=128)
    cs.nVectors = 1048576
    cs.outputShape = (8, 8)
    cs._promotion_enabled = True
    cs._ensure_concept_pool()
    store = Spaces._concept_alloc_of(cs).layer()
    assert store.where.shape == (15, 15)
    assert store.where.device.type == 'cpu'


def test_projected_updates_keep_nonnegative_parts_and_kind_maximum():
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    for part in range(3):
        cs.add_concept_edge(row, part, .1)
    store = Spaces._concept_alloc_of(cs).layer()
    store.assigned[row] = True
    with torch.no_grad():
        store.values.copy_(torch.tensor([-.1, .2, .4]))
    store.project_parts()
    torch.testing.assert_close(store.values, torch.tensor([0., .5, 1.]))
    with pytest.raises(ValueError, match='self-edge'):
        store.conjunctive.add_edge(row, row + store.nOutput + 1, 1.)


def test_signed_checkpoint_migrates_poles_and_optimizer_first_moments():
    from types import SimpleNamespace
    from test_structural_checkpoint import _model_with
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, .7)
    cs.add_concept_edge(row, 1, .8, conjunctive=True)
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    layer = saved['conceptual_spaces'][0]['allocator']['layers'][0]
    layer['nInput'] = layer['nOutput'] + 1
    layer['values'].neg_()
    parts = layer['parts']
    parts['version'] = 1
    parts['matrices']['where'] = dict(rows=[], cols=[], values=None)
    parts['matrices']['conjunctive']['values'].neg_()
    restored = _cs()
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    store = Spaces._concept_alloc_of(restored).layer()
    assert store._cols == [store.nOutput + 1]
    assert store.conjunctive._cols == [store.nOutput + 2]
    optimizer = torch.optim.Adam([store.values, store.conjunctive.values])
    for matrix in store.part_matrices():
        optimizer.state[matrix.values] = dict(exp_avg=torch.tensor([.3]), exp_avg_sq=torch.tensor([.4]))
    store.migrate_part_moments(optimizer)
    for matrix in store.part_matrices():
        assert optimizer.state[matrix.values]['exp_avg'].item() == pytest.approx(-.3)
        assert optimizer.state[matrix.values]['exp_avg_sq'].item() == pytest.approx(.4)
    store.migrate_part_moments(optimizer)  # migration is applied exactly once
    assert optimizer.state[store.values]['exp_avg'].item() == pytest.approx(-.3)


def test_restored_knowing_reads_symbols_without_a_percept_carrier():
    from types import SimpleNamespace
    import Language
    from test_structural_checkpoint import _model_with
    cs = _cs()
    evidence = torch.zeros(sum(cs._order_caps()), 1, 2, 2)
    evidence[0, 0, 0] = 1.
    cs.subspace._concept_activations = evidence
    saved = _model_with(cs, SimpleNamespace())._collect_structural_extras()
    restored = _cs()
    _model_with(restored, SimpleNamespace())._restore_structural_extras(saved)
    ss = SimpleNamespace(subspace=SimpleNamespace(what=None, nWhat=8),
                         _publish_symbol_snapshot=lambda event, *_a, **_kw: SimpleNamespace(event=event))
    leg = Language.SymbolSpace.forward_concept_to_symbol(ss, restored.subspace)
    torch.testing.assert_close(leg._symbol_evidence[0, 0], torch.ones(2))
    assert leg._symbol_evidence[1:].count_nonzero() == 0
    assert leg.event.shape == (1, 2 * sum(cs._order_caps()), 8)


def test_thought_transpose_keeps_negated_literal_in_a_separate_occurrence():
    from types import SimpleNamespace
    from AccessibleMind import apply_thought_effect
    from QueryWork import QueryWorkBudget
    cs = _cs()
    cs.conceptual_pi = True
    row = _mint_row(cs, 1, 101)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    cs.add_concept_edge(row, 1, 1., conjunctive=True, negated=True)
    before = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    before[1, 0, 0, 0] = 1.
    cs.subspace._concept_activations = before.clone()
    result = SimpleNamespace(semantic_id='quantize', evidence={'reference': ('row', row)})
    apply_thought_effect(SimpleNamespace(conceptualSpace=cs), result,
                         row=0, work=QueryWorkBudget(32))
    after = cs.subspace._concept_activations
    torch.testing.assert_close(after[:, :, :1], before)
    torch.testing.assert_close(after[1, 0, 1], torch.tensor([0., 1.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(symbols(after)[1, 0], torch.ones(2), atol=1e-6, rtol=0)
