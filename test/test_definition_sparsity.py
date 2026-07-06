"""<definitionSparsityScale> (snap contract sec 1.4 / sec 5, 2026-07-06): the
rank-ordered soft-then-hard L0 that keeps each concept's DEFINITION compact --
sort a concept's in-edge weights by |w|, exempt the top ``definitionFreeSize``
(genus + differentia), and shrink only the surplus ranks. A growth-PREVENTING
regularizer: it pulls marginal symbols toward zero, never adds any. These tests
lock the penalty math, that it excludes the EVERYTHING pole, that its gradient
lands only on the marginal ranks, and the default-off (lambda 0) no-op.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
from Layers import ConceptualAttentionLayer  # noqa: E402


def _row_layer(weights, pole_weight=None, n=6):
    """A square concept store with one concept (row 0) whose in-edges over the
    concept columns carry ``weights`` (col 0 is the forbidden self-edge, so we
    start at col 1); optionally a big edge to the EVERYTHING pole (last col)."""
    ly = ConceptualAttentionLayer.square(n)
    for k, w in enumerate(weights):
        ly.add_edge(0, 1 + k, weight=w)
    if pole_weight is not None:
        ly.add_edge(0, n, weight=pole_weight)    # col == nInput-1 = the pole
    return ly


def test_penalty_exempts_top_free_and_sums_marginal():
    ly = _row_layer([1.0, 0.9, 0.3, 0.1])
    p = ly.definition_sparsity_penalty(free_size=2)
    assert abs(float(p) - 0.4) < 1e-6, "marginal ranks 0.3 + 0.1, top-2 exempt"
    # free_size 0 sums everything; free_size >= row size gives nothing.
    assert abs(float(ly.definition_sparsity_penalty(free_size=0)) - 2.3) < 1e-6
    assert abs(float(ly.definition_sparsity_penalty(free_size=3)) - 0.1) < 1e-6


def test_penalty_excludes_everything_pole():
    without = float(_row_layer([1.0, 0.9, 0.3, 0.1]).definition_sparsity_penalty(2))
    withpole = float(
        _row_layer([1.0, 0.9, 0.3, 0.1], pole_weight=5.0)
        .definition_sparsity_penalty(2))
    assert abs(without - withpole) < 1e-6, "the pole is a standing axiom, uncounted"


def test_penalty_uses_absolute_value_signed_weights():
    # Exclusions (negative weights) count by magnitude, like inclusions.
    ly = _row_layer([1.0, -0.9, 0.3, -0.1])
    assert abs(float(ly.definition_sparsity_penalty(2)) - 0.4) < 1e-6


def test_gradient_lands_only_on_marginal_ranks():
    ly = _row_layer([1.0, 0.9, 0.3, 0.1], pole_weight=5.0)
    ly.definition_sparsity_penalty(free_size=2).backward()
    grad = {c: float(ly.values.grad[i]) for (r, c), i in ly._index.items()}
    assert grad[1] == 0.0 and grad[2] == 0.0, "top-2 (core) are unpenalized"
    assert grad[3] == 1.0 and grad[4] == 1.0, "marginal ranks are pulled down"
    assert grad[6] == 0.0, "the EVERYTHING pole is never penalized"


def test_penalty_none_when_empty():
    ly = ConceptualAttentionLayer.square(6)          # no edges
    assert ly.definition_sparsity_penalty(2) is None


def test_multi_row_sums_each_concept_independently():
    ly = ConceptualAttentionLayer.square(6)
    for c, w in [(1, 1.0), (2, 0.8), (3, 0.5)]:      # concept 0: marginal 0.5
        ly.add_edge(0, c, weight=w)
    for c, w in [(2, 1.0), (3, 0.9), (4, 0.2), (5, 0.2)]:  # concept 1: 0.2+0.2
        ly.add_edge(1, c, weight=w)
    assert abs(float(ly.definition_sparsity_penalty(2)) - (0.5 + 0.4)) < 1e-6


# -- ConceptualSpace wrapper: lambda gate + default-off ---------------------

def test_definition_sparsity_loss_lambda_gate():
    import types
    # A minimal host with just the allocator hook the loss method needs.
    from Spaces import ConceptualSpace
    cs = ConceptualSpace.__new__(ConceptualSpace)
    cs.definition_free_size = 2
    # lam <= 0 short-circuits BEFORE touching the store -> no-op.
    assert cs.definition_sparsity_loss(lam=0.0) is None
    assert cs.definition_sparsity_loss(lam=-1.0) is None
