"""Retrieval postings follow a bounded Part read; generation stays separate."""
import copy
from types import SimpleNamespace

import torch

from Layers import LTM0
from Meaning import ConceptualMeaning
from MemoryIndex import unfold_idea






def test_unfold_candidates_are_the_current_priming_surface():
    language = SimpleNamespace(_generate_binary_ops=(), _generate_unary_ops=(), generate_policy_logits=lambda point,features=None: point.new_zeros(1, 1))
    basis = torch.eye(3)
    weights = torch.tensor([1., 2., 1.])
    active = lambda: weights
    cold = unfold_idea(language, basis, basis[0], 4, activation=active)
    hot = unfold_idea(language, basis, basis[1], 4, activation=active)
    assert cold['codes'] == () and not cold['complete']
    assert hot['codes'] == (1,) and hot['complete']
    weights[0] = 3.
    assert unfold_idea(language, basis, basis[0], 4, activation=active)['codes'] == (0,)
