"""Descent selects witnessed cases instead of inventing a sigma inverse."""
from basic_fixtures import feature as basic_feature
from symbol_fixtures import readable
import torch

from test_cs_sparse_weights import _cs, _mint_row


def test_descent_intersects_cases_with_the_present_field():
    cs = _cs()
    parent = _mint_row(cs, 1, 101)
    readable(cs, 0, 1)
    cs.add_concept_edge(parent, 0, 1.)
    cs.add_concept_edge(parent, 1, 1.)
    query = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    query[parent, 0, 0, 0] = .8
    observed = torch.zeros_like(query)
    observed[1, 0, 0, 0] = .7
    result = cs.cs_reverse_presence(query, observed=observed)
    torch.testing.assert_close(result[1, 0, 0], torch.tensor([.7, 0.]))
    assert result[0].count_nonzero() == 0
    observed.zero_()
    observed[2, 0, 0, 0] = 1.
    result = cs.cs_reverse_presence(query, observed=observed)
    assert result[:2].count_nonzero() == 0


def test_descent_without_a_field_chooses_one_case():
    cs = _cs()
    parent = _mint_row(cs, 1, 101)
    readable(cs, 0, 1)
    cs.add_concept_edge(parent, 0, .2)
    cs.add_concept_edge(parent, 1, .8)
    query = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    query[parent, 0, 0, 0] = 1.
    result = cs.cs_reverse_presence(query)
    torch.testing.assert_close(result[:2, 0, 0, 0], torch.tensor([0., 1.]))


def test_descent_keeps_the_source_pole_and_scope():
    cs = _cs()
    parent = _mint_row(cs, 1, 101)
    readable(cs, 0, 1)
    cs.add_concept_edge(parent, 0, 1., negated=True)
    query = torch.zeros(sum(cs._order_caps()), 1, 2, 2)
    query[parent, 0, 1, 0] = .8
    observed = torch.zeros_like(query)
    observed[0, 0, 1, 1] = .7
    result = cs.cs_reverse_presence(query, observed=observed)
    torch.testing.assert_close(result[0, 0, 1], torch.tensor([0., .7]))
    assert result[:, :, 0].count_nonzero() == 0


def test_percept_attribution_follows_signed_definitions_without_complements():
    cs = _cs()
    basic_feature(cs, 0, 'ps', 65, 1.)
    basic_feature(cs, 0, 'ws', 3, -1., parts=(65,))
    query = torch.zeros(sum(cs._order_caps()), 1, 1, 2)
    query[0, 0, 0] = torch.tensor([.8, .6])
    columns, values, spans = cs.cs_percept_attribution(query)
    # A concept's demand is divided over its written alternatives by weight.
    # The attributed total equals the demand, preserving each signed pole.
    assert columns.tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(values[:, 0, 0, 0], torch.tensor([.4, .3, .3, .4]))
    assert spans is None


from types import SimpleNamespace

import torch

import Spaces

from Models import BasicModel

from test_cs_sparse_weights import _cs, _mint_row

from test_concept_memberships import binary_features

