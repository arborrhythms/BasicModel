"""Order-zero definitions read native features, independently of codes."""
import pytest
import torch

import Spaces
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs, _mint_row


def binary_features(cs, raw):
    primitive = PrimitiveProperties(2)
    primitive.teach(0, [48, 49], [0., 1.])
    primitive.teach(1, [48, 49], [1., 0.])
    cs.add_concept_feature(0, 'ws', 0, 1.)
    cs.add_concept_feature(0, 'ws', 1, -1.)
    b, length = raw.shape
    ix = torch.arange(length)
    spans = torch.stack((ix, ix + 1), -1)[None].expand(b, -1, -1)
    extents = torch.tensor([[[0, length]]]).expand(b, -1, -1)
    return (raw, spans, primitive, raw, spans), extents


def test_one_property_has_the_four_extent_corners():
    cs = _cs()
    native, extents = binary_features(cs, torch.tensor([[49, 49], [48, 48], [48, 49], [49, 48]]))
    actual = cs.cs_read_memberships(native, extents)
    torch.testing.assert_close(actual[0, :, 0], torch.tensor([[1., 0.], [0., 1.], [1., 1.], [1., 1.]]))


def test_unrelated_memberships_stay_exactly_zero_at_every_scope():
    cs = _cs()
    for length in (1, 8, 256, 1024):
        native, extents = binary_features(cs, torch.full((1, length), 65))
        actual = cs.cs_read_memberships(native, extents)
        assert actual.count_nonzero() == 0
        assert cs._cs_position_evidence.count_nonzero() == 0


def test_parameter_getter_does_not_offer_new_parts():
    cs = _cs()
    row = _mint_row(cs, 1, 100)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    matrix = Spaces._concept_alloc_of(cs).layer().conjunctive
    before = tuple(matrix._index)
    parameter = matrix.values
    cs.getParameters()
    assert tuple(matrix._index) == before
    assert matrix.values is parameter


def test_a_located_part_need_not_pervade_but_the_whole_property_must():
    cs = _cs()
    primitive = PrimitiveProperties(1)
    primitive.teach(0, [48, 49], [0., 1.])
    # A PS percept and a WS property have independent native addresses.
    cs.add_concept_feature(0, 'ps', 7, .5)
    cs.add_concept_feature(0, 'ws', 0, 1.)
    raw = torch.tensor([[49, 49], [49, 48], [48, 48]])
    part_ids = torch.tensor([[7, 8], [7, 8], [8, 8]])
    parts = torch.tensor([[[0, 1], [1, 2]]]).expand(3, -1, -1)
    whole = torch.tensor([[[0, 2]]]).expand(3, -1, -1)
    cs.cs_read_memberships((part_ids, parts, primitive, raw, whole), whole)
    # The part at [0,1] suffices at [0,2]; the property must hold at both
    # positions. Inspect this occurrence before the extent readout union.
    torch.testing.assert_close(cs._cs_position_evidence[0, :, 0, -1],
                               torch.tensor([[1., 0.], [0., 0.], [0., 1.]]))
    # A missing byte supplies neither pole, including a missing complement.
    raw[2, 1] = 0
    cs.cs_read_memberships((part_ids, parts, primitive, raw, whole), whole)
    assert cs._cs_position_evidence[0, 2, 0, -1].count_nonzero() == 0


@pytest.mark.parametrize('literal', [(7,), (7, 8)])
def test_part_containment_belongs_to_the_subject_extent(literal):
    cs = _cs()
    cs.add_concept_feature(0, 'ps', literal, 1.)
    raw = torch.full((1, 8), 65)
    ids = torch.tensor([[7, 8, 9, 9, 9, 9, 9, 9]])
    starts = torch.arange(8)
    positions = torch.stack((starts, starts + 1), -1)[None]
    extents = torch.tensor([[[0, 4], [4, 8]]])
    actual = cs.cs_read_memberships((ids, positions, None, raw, positions), extents)
    # Other positions inside a subject do not deny its present part.
    torch.testing.assert_close(actual[0, 0], torch.tensor([[1., 0.], [0., 1.]]))
    assert cs._cs_position_evidence[0, 0, 0, :, 1].count_nonzero() == 0
    assert cs._cs_position_evidence[0, 0, 1, :, 0].count_nonzero() == 0
    # Absence still requires complete observation of that subject.
    missing = cs.cs_read_memberships((ids[:, :6], positions[:, :6], None,
                                      raw, positions), extents)
    assert missing[0, 0, 1].count_nonzero() == 0
    if len(literal) > 1:
        separated = torch.tensor([[[0, 1], [1, 4]]])
        apart = cs.cs_read_memberships((ids, positions, None, raw, positions), separated)
        assert apart[0, 0, :, 0].count_nonzero() == 0
    empty = cs.cs_read_memberships((ids, positions, None, raw, positions), extents[:, :0])
    assert empty.shape[2] == cs._cs_position_evidence.shape[2] == 0


def test_raw_analysis_layout_clips_regions_to_observed_input():
    from types import SimpleNamespace
    raw = torch.zeros(4, 1, 4096, dtype=torch.long)
    for row, text in enumerate((b'hello world', b'bye', b'', b'a' * 600)):
        raw[row, 0, :len(text)] = torch.tensor(list(text), dtype=torch.long)
    positions, extents = Spaces.WholeSpace.concept_evidence_layout(
        SimpleNamespace(), raw, 8)
    expected = torch.zeros(4, 8, 2, dtype=torch.long)
    expected[0, 0] = torch.tensor([0, 11])
    expected[1, 0] = torch.tensor([0, 3])
    expected[3, :2] = torch.tensor([[0, 512], [512, 600]])
    torch.testing.assert_close(positions, expected)
    torch.testing.assert_close(extents[:, 0], torch.tensor([[0, 11], [0, 3], [0, 0], [0, 600]]))


def test_fractional_feature_weights_use_one_product_then_one_extent_union():
    cs = _cs()
    primitive = PrimitiveProperties(2)
    with torch.no_grad():
        primitive.members[:, 65] = torch.tensor([.25, .36])
    cs.add_concept_feature(0, 'ws', 0, .5)
    cs.add_concept_feature(0, 'ws', 1, -1.)
    raw = torch.tensor([[65, 65]])
    spans = torch.tensor([[[0, 1], [1, 2]]])
    extent = torch.tensor([[[0, 2]]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), extent)
    occurrence = torch.tensor([.25 ** .5 * .64, .75 ** .5 * .36])
    torch.testing.assert_close(cs._cs_position_evidence[0, 0, 0, :2], occurrence.expand(2, -1))
    torch.testing.assert_close(read[0, 0, 0], 1 - (1 - occurrence).square())


def test_feature_definitions_use_the_existing_sparsity_penalty_and_sidecar():
    cs = _cs()
    for feature, weight in enumerate((.8, -.6, .4)):
        cs.add_concept_feature(0, 'ws', feature, weight)
    store = Spaces._concept_alloc_of(cs).layer()
    penalty = cs.definition_sparsity_loss(1.)
    torch.testing.assert_close(penalty, torch.tensor(.4))
    penalty.backward()
    torch.testing.assert_close(store.features.values.grad, torch.tensor([0., 0., 1.]))
    saved = store.parts_extras()
    restored = _cs()
    restored._sparse_families(0)
    other = Spaces._concept_alloc_of(restored).layer()
    other.load_parts_extras(saved)
    assert other.features._index == store.features._index
    assert other.features.nInput == store.features.nInput
    torch.testing.assert_close(other.features.values, store.features.values)
    assert not any('features.values' in key for key in cs.state_dict())


def test_sentence_boundary_offers_candidates_and_getter_remains_pure():
    cs = _cs()
    row = _mint_row(cs, 1, 100)
    cs.add_concept_edge(row, 0, 1., conjunctive=True)
    cs.add_concept_feature(0, 'ws', 7, 1.)
    store = Spaces._concept_alloc_of(cs).layer()
    cs.Reset(hard=True)
    assert (row, store.nOutput + 1) in store.conjunctive._index
    assert (0, 4 * 7 + 3) in store.features._index
    before = [(tuple(m._index), m.values) for m in store.definition_matrices()]
    cs.getParameters()
    for (indices, parameter), matrix in zip(before, store.definition_matrices()):
        assert tuple(matrix._index) == indices and matrix.values is parameter


def test_pruning_a_feature_preserves_the_other_tower_at_the_same_address():
    cs = _cs()
    row = _mint_row(cs, 0, 100)
    cs.add_concept_feature(row, 'ps', 7, .5)
    cs.add_concept_feature(row, 'ws', 7, -.5)
    cs._prepare_part_learning()
    cs._drop_concept_edge(100, 7, side='part')
    matrix = Spaces._concept_alloc_of(cs).layer().features
    assert set(matrix._index) == {(row, 30), (row, 31)}
    cs._drop_concept_edge(100, 7, side='whole')
    assert matrix.nnz == 0 and matrix.values is None


def test_feature_growth_preserves_frozen_rows_and_optimizer_moments(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    frozen = _mint_row(cs, 0, 100)
    live = _mint_row(cs, 0, 101)
    cs.add_concept_feature(frozen, 'ws', 0, .5)
    cs.add_concept_feature(live, 'ws', 1, .5)
    cs.freeze_concept(100)
    optimizer = model.getOptimizer(lr=.001)
    matrix = Spaces._concept_alloc_of(cs).layer().features
    matrix.values.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    old = matrix.values
    moment = optimizer.state[old]['exp_avg'].clone()
    cs._prepare_part_learning()
    cs._maybe_rebuild_optimizer_for_csw()
    assert matrix.values is not old
    torch.testing.assert_close(optimizer.state[matrix.values]['exp_avg'][:2], moment)
    assert sum(p is matrix.values for group in optimizer.param_groups for p in group['params']) == 1
    assert not any(p is old for group in optimizer.param_groups for p in group['params'])
    matrix.values.sum().backward()
    assert float(matrix.values.grad[0]) == 0
    assert float(matrix.values.grad[1]) == 1


def test_distributed_codes_follow_features_at_the_boundary_only(tmp_path):
    from test_grounded_xor import grounded_model
    model, x = grounded_model(tmp_path)
    cs = model.conceptualSpaces[0]
    cs.add_concept_feature(0, 'ws', 0, 1.)
    ws = model.wholeSpaces[0]
    source = ws.subspace.what.getW()[0].detach()
    expected = torch.nn.functional.pad(source[:cs.nWhat], (0, max(0, cs.nWhat-len(source))))
    expected = torch.nn.functional.normalize(expected, dim=0)
    cs._refresh_feature_codes()
    torch.testing.assert_close(cs.similarity_codebook.getW()[0], expected)
    model.forward(x)
    before = cs._cs_last_a0.clone()
    with torch.no_grad():
        cs.similarity_codebook.getW().normal_()
    model.End()
    model.forward(x)
    torch.testing.assert_close(cs._cs_last_a0, before, atol=0, rtol=0)
    model.End()
