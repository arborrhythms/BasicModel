"""Idempotent membership folds preserve evidence when a witness repeats."""
from basic_fixtures import feature as basic_feature, order_zero_expected, published_presence
import torch

from ConceptEvidence import corners, union
from Layers import SparseLayer
from PerceptProperties import PrimitiveProperties
from test_cs_sparse_weights import _cs, _mint_row








def test_seam_never_invents_counterevidence_from_an_absent_property():
    from basic_fixtures import with_own_codes
    cs = with_own_codes(_cs(), (48, 49, 65))
    primitive = PrimitiveProperties(2)
    primitive.teach(0, [48, 49], [0., 1.])
    primitive.teach(1, [48, 49], [1., 0.])
    basic_feature(cs, 0, 'ws', 0, 1., parts=(48, 49))
    spans = torch.tensor([[[0, 1]]])
    raw = torch.tensor([[65]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    assert read.count_nonzero() == 0
    basic_feature(cs, 0, 'ws', 1, -1., parts=(48, 49))
    raw = torch.tensor([[48]])
    read = cs.cs_read_memberships((raw, spans, primitive, raw, spans), spans)
    # The counterevidence is observed, at the basic percepts' rows: property
    # 0's row reads against (its part present, its whole seen false), and
    # the Not member's row reads for.
    basic, field = cs.basic_percepts, cs._percept_field
    slot = lambda prop: field.keys.index(('bp', basic.index[((48,), (prop,))]))
    torch.testing.assert_close(field.values[slot(0), 0, 0], torch.tensor([0., 1.]))
    torch.testing.assert_close(field.values[slot(1), 0, 0], torch.tensor([1., 0.]))
    # The concept is one signed number read by its column, for less against.
    _, lanes = order_zero_expected(cs, 0, published_presence(cs))
    torch.testing.assert_close(read[0, 0, 0], lanes.to(read), atol=1e-5, rtol=1e-5)




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


def test_sparse_sigma_reads_the_weighted_sum_of_options():
    # The store's read is a sparse Sigma over the Not's options (in place of
    # the weighted least and greatest): tanh of the weighted sum.
    import math
    from Layers import NotLayer3, sigma_read
    layer = SparseLayer(3, 2, nonlinear=False)
    for target in range(2):
        for source in range(3):
            layer.add_edge(target, source, .5 if source == 0 else 1.)
    lanes = torch.tensor([[[.25, 0.]], [[.6, 0.]], [[.6, 0.]]], requires_grad=True)
    value = sigma_read(layer, NotLayer3.options(lanes))
    torch.testing.assert_close(value, torch.full((2, 1), math.tanh(.5 * .25 + .6 + .6)))
    value.sum().backward()
    assert torch.isfinite(lanes.grad).all()


from types import SimpleNamespace

import torch

import Spaces

from Models import BasicModel

from test_cs_sparse_weights import _cs, _mint_row

from test_concept_memberships import binary_features



