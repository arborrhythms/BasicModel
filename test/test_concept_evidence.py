"""Live paired evidence: snap, scoped composition, reverse and persistence."""
import pytest
import math
import torch

import Spaces
from ConceptEvidence import corners, symbols, union
from basic_fixtures import order_zero_expected, published_presence
from test_concept_memberships import binary_features
from test_cs_sparse_weights import _cs, _mint_row, _mint_field
from symbol_fixtures import readable


def field(cs, batch=1, occurrences=1):
    return torch.zeros(cs._order_caps()[0], batch, occurrences, 2)


@pytest.mark.parametrize('parts', [2, 4, 8])
def test_production_snap_background_is_neither(parts):
    cs = _cs()
    row = _mint_field(cs, 101)
    # An order-zero concept reads basic percepts in the order-zero Sigma;
    # the concept store holds kinds above order zero only.
    for part in range(parts):
        with pytest.raises(ValueError, match='order-zero Sigma'):
            cs.add_concept_edge(row, part, 1.)
    native, extents = binary_features(cs, torch.zeros(2, 8, dtype=torch.long))
    snap = cs.cs_read_memberships(native, extents)
    _, result = cs.cs_forward_content(snap, cs.similarity_codebook.getW())
    assert torch.count_nonzero(result) == 0
    torch.testing.assert_close(corners(symbols(result)[row])[:, 3], torch.ones(2))


def test_membership_read_keeps_both_until_symbol_readout():
    cs = _cs()
    native, extents = binary_features(cs, torch.tensor([[49, 48]]))
    read = cs.cs_read_memberships(native, extents)
    field = cs._percept_field
    carrying = [field.keys.index(('bp', bp)) for bp in cs.basic_percepts.carrying('ws', 0)]
    # The property is present at the first event.
    torch.testing.assert_close(field.event_evidence([field.keys[i] for i in carrying]).amax(0)[0, 0, 0],
                               torch.tensor([1., 0.]))
    # Seen true and false in the extent, it is both across the rows of the
    # basic percepts that carry it: the row of 49 reads for where it lies,
    # the row of 48 against.
    basic = cs.basic_percepts
    for part, expected in ((49, [1., 0.]), (48, [0., 1.])):
        index = field.keys.index(('bp', basic.index[((part,), (0,))]))
        torch.testing.assert_close(field.values[index, 0, 0].detach(), torch.tensor(expected))
    torch.testing.assert_close(cs.order_zero_both(field, [0]), torch.ones(1, 1, 1))
    # The concept, and the symbol read from it, are one signed number:
    # for less against, never both.
    _, lanes = order_zero_expected(cs, 0, published_presence(cs))
    torch.testing.assert_close(read[0, 0, 0], lanes.to(read), atol=1e-5, rtol=1e-5)
    assert float(corners(symbols(read.detach()))[0, 0, 2]) == 0.


def test_kind_of_a_both_member_and_an_absent_member_is_one_signed_value():
    cs = _cs()
    row = _mint_row(cs, 1, 101)
    readable(cs, 0, 1)
    for col in (0, 1):
        cs.add_concept_edge(row, col, 1.)
    a = field(cs)
    a[:2, 0, 0] = torch.tensor([[1., 1.], [0., 1.]])
    _, result = cs.cs_forward_content(a, torch.zeros(64, 8))
    # A kind is one signed value, never both: the both member's value is
    # nothing and the absent member's is -1, summed.
    torch.testing.assert_close(result[row, 0, 0], torch.tensor([0., math.tanh(1.)]))






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
    from test_attention_promotion import _fixture, _close
    from FamiliarTier import observe_candidate
    cs, rows = _fixture()
    a, b = [r for _, r in rows[:2]]
    for _ in range(3):
        key = observe_candidate(cs, 1, {a: 1., b: 1.})
        _close(cs)
    cid = cs.mtm.promote_candidates(cs)[0]
    row = cs._concept_row_of(cid)
    for _ in range(cs.mtm.horizon+2):
        _close(cs)
    assert key not in cs.mtm.entries and cs.concept_id_at_row(row) == cid
    a0 = field(cs)
    a0[a, 0, 0, 0] = 1.
    _, result = cs.cs_forward_content(a0, cs.similarity_codebook.getW())
    # The Sigma reads the present member by its entry's weight.
    store = Spaces._concept_alloc_of(cs).layer()
    weight = sum(float(store.values[p].abs()) for p in cs.concept_entries(row, a))
    assert float(result[row, 0, 0, 0]) == pytest.approx(math.tanh(weight))



def test_candidate_alternatives_merge_sparse_support_before_discovery():
    from test_attention_promotion import _fixture, _close
    from FamiliarTier import observe_candidate, centroid
    cs, rows = _fixture()
    support = {row: .9 for _, row in rows[:4]}
    for _ in range(4):
        key = observe_candidate(cs, 1, support)
        _close(cs)
        entry = cs.mtm.entries[key]
        torch.testing.assert_close(cs.mtm.value_of(key), centroid(cs, entry.support))
    assert entry.count == 4 and set(entry.support) == set(support)
    assert entry.symbol is None



def test_context_allocation_is_bounded_independently_of_the_shared_inventory():
    from Memory import MediumTermMemory
    pool = MediumTermMemory(64, n_vectors=1048576, field_width=8)
    key = pool.open_candidate(order=0, when=0, value=torch.ones(8), support={1048575: 1.})
    assert pool.entries[key].support == {1048575: 1.}
    assert pool.transient_ids.shape == (64,)
    assert not any(torch.is_tensor(value) and value.ndim > 1 for value in pool.__dict__.values())





def test_discovered_and_end_definitions_keep_learned_exponent_scale():
    cs = _cs()
    discovered = _mint_row(cs, 1, 101)
    ended = _mint_row(cs, 1, 102)
    readable(cs, 0, 1)
    cs.add_concept_edge(discovered, 0, .4)
    cs.add_concept_edge(discovered, 1, 1.2)
    cs.add_concept_edge(ended, 0, .3)
    cs.add_concept_edge(ended, 1, 1.5)
    store = Spaces._concept_alloc_of(cs).layer()
    store.assigned[discovered] = True
    before = store.values.detach().clone()
    torch.testing.assert_close(store.values, before)




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




def test_concept_meaning_starts_zero_and_grows_without_changing_direction():
    from MeaningCodes import defined_code
    direction = torch.tensor([1., 2., 3., 4.])
    values = [defined_code(direction, count) for count in (0, 1, 4, 20)]
    assert not values[0].any()
    norms = [float(value.norm()) for value in values]
    assert norms == pytest.approx([0., .2, .5, 20/24])
    for value in values[1:]:
        torch.testing.assert_close(value / value.norm(), direction / direction.norm())
