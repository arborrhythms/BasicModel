"""Clause admission has no pre-backprop learn-score gate."""
import pytest
from Spaces import ConceptualSpace

@pytest.mark.parametrize('name', ['_learn_score_children_in_codebook',
    '_learn_score_is_truth_obvious', '_learn_score_resolves_contradiction',
    '_maybe_learn_relation', '_route_learned_relation'])
def test_learn_score_components_are_deleted(name):
    assert not hasattr(ConceptualSpace, name)
