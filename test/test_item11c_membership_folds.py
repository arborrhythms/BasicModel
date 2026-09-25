"""Idempotent membership folds preserve evidence when a witness repeats."""
import torch

from ConceptEvidence import corners, union
from Layers import SparseLayer
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs, _mint_row


def test_required_parts_keep_support_and_counterevidence_separate():
    """A true and B false support both poles; negating B aligns them."""
    cs = _cs()
    _, store = cs._sparse_families(0)
    matrix = store.conjunctive
    offset = store.nOutput + 1
    for target, negative_b in ((2, False), (3, True)):
        matrix.add_edge(target, 0, 1.)
        matrix.add_edge(target, 1 + offset * negative_b, 1.)
    source = torch.zeros(store.nInput, 3)
    # Crisp evidence, graded evidence, and complete uncertainty.
    source[0] = torch.tensor([1., .8, 0.])
    source[offset + 1] = torch.tensor([1., .6, 0.])
    pair = torch.stack((matrix.fold_presence(source, conjunctive=True),
                        matrix.fold_presence(source, conjunctive=True, dual=True)), -1)
    torch.testing.assert_close(pair[2], torch.tensor([[1., 1.], [.8, .6], [0., 0.]]))
    torch.testing.assert_close(pair[3], torch.tensor([[1., 0.], [.6, 0.], [0., 0.]]))


def test_swapped_evidence_combination_is_distinct_from_a_both_pole_test():
    cs = _cs()
    _, store = cs._sparse_families(0)
    matrix = store.conjunctive
    offset = store.nOutput + 1
    matrix.add_edge(1, 0, 1.)
    matrix.add_edge(1, offset, 1.)
    pair = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]])
    source = torch.zeros(store.nInput, 4)
    source[0], source[offset] = pair.unbind(-1)
    combined = torch.stack((matrix.fold_presence(source, conjunctive=True),
                            matrix.fold_presence(source, conjunctive=True, dual=True)), -1)
    torch.testing.assert_close(combined[1], torch.tensor([[1., 1.], [1., 1.], [1., 1.], [0., 0.]]))
    torch.testing.assert_close(corners(pair)[:, 2], torch.tensor([0., 0., 1., 0.]))


def test_unknown_concept_does_not_refute_or_erase_a_known_conjunct():
    cs = _cs()
    _, store = cs._sparse_families(0)
    for source in (0, 1):
        store.conjunctive.add_edge(2, source, 1.)
    source = torch.zeros(store.nInput, 1)
    source[0] = .8
    read = store.conjunctive.fold_presence(source, conjunctive=True)
    torch.testing.assert_close(read[2], torch.tensor([.8]))
    source[store.nOutput + 2] = .6
    torch.testing.assert_close(store.conjunctive.fold_presence(source, conjunctive=True)[2],
                               torch.tensor([.8]))
    torch.testing.assert_close(store.conjunctive.fold_presence(source, conjunctive=True, dual=True)[2],
                               torch.tensor([.6]))


def test_seam_never_invents_counterevidence_from_an_absent_property():
    cs = _cs()
    primitive = PrimitiveProperties(2)
    primitive.teach(0, [48, 49], [0., 1.])
    primitive.teach(1, [48, 49], [1., 0.])
    cs.add_concept_feature(0, 'ws', 0, 1.)
    spans = torch.tensor([[[0, 1]]])
    raw = torch.tensor([[65]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    assert read.count_nonzero() == 0
    cs.add_concept_feature(0, 'ws', 1, -1.)
    raw = torch.tensor([[48]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    torch.testing.assert_close(read[0, 0, 0], torch.tensor([0., 1.]))


def test_conjunctive_edges_exist_only_within_order_zero():
    import pytest
    cs = _cs()
    cs.add_concept_edge(2, 0, 1., conjunctive=True)
    row = _mint_row(cs, 1, 100)
    with pytest.raises(ValueError, match='order.?0'):
        cs.add_concept_edge(row, 0, 1., conjunctive=True)


def test_union_is_max_including_empty_and_repeated_support():
    values = torch.full((3, 1024), .8, requires_grad=True)
    torch.testing.assert_close(union(values, dim=1), torch.full((3,), .8))
    assert union(values[:, :0], dim=1).count_nonzero() == 0
    union(values, dim=1).sum().backward()
    assert torch.isfinite(values.grad).all()
    torch.testing.assert_close(values.grad.sum(1), torch.ones(3))


def test_property_pervasion_is_min_over_observed_values_not_their_counts():
    properties = PrimitiveProperties(1)
    with torch.no_grad():
        properties.members[0, [48, 49]] = torch.tensor([.8, .6])
    counts = torch.zeros(4, 256)
    counts[0, 48] = 1
    counts[1, 48] = 1024
    counts[2, [48, 49]] = torch.tensor([40., 20.])
    torch.testing.assert_close(properties.on_counts(counts)[:, 0],
                               torch.tensor([.8, .8, .6, 0.]))
    torch.testing.assert_close(properties.on_counts(counts, conjunctive=False)[:, 0],
                               torch.tensor([.8, .8, .8, 0.]))


def test_property_alternatives_use_max_min_for_graded_primitive_inputs():
    properties = PrimitiveProperties(1)
    with torch.no_grad():
        properties.members[0, [48, 49]] = torch.tensor([.8, .6])
    mixture = torch.zeros(2, 256)
    mixture[0, [48, 49]] = torch.tensor([.5, .4])
    mixture[1, [48, 49]] = torch.tensor([.9, .3])
    torch.testing.assert_close(properties.from_primitives(mixture)[:, 0],
                               torch.tensor([.5, .8]))


def test_sparse_requirements_and_alternatives_are_weighted_min_and_max():
    layer = SparseLayer(3, 2, nonlinear=False)
    for target in range(2):
        for source in range(3):
            layer.add_edge(target, source, .5 if source == 0 else 1.)
    source = torch.tensor([[.25], [.6], [.6]], requires_grad=True)
    torch.testing.assert_close(layer.fold_presence(source, conjunctive=True),
                               torch.full((2, 1), .5))
    torch.testing.assert_close(layer.fold_presence(source), torch.full((2, 1), .6))
    layer.fold_presence(source, conjunctive=True).sum().backward()
    assert torch.isfinite(source.grad).all()
