# basicmodel/test/test_clarifications.py
"""Tests for EnsureConsistency + clarifying-question UX.

Plan reference: lazy-juggling-planet.md Phase C. Extends TruthLayer with
source/trust parallel lists, an opt-in structured consistency report, and
a templated ``suggest_clarifications()`` that surfaces part-of +
opposite-sign contradictions to the user.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

from Layers import TruthLayer  # noqa: E402


class _StubBasis:
    """Minimal Basis stub: returns part-scores the test explicitly sets.

    Tensors are compared by content (``torch.equal``) so views of the
    same underlying buffer still match after a round-trip through
    ``layer.truths[i]``.
    """

    monotonic = True

    def __init__(self):
        # list of (a_tensor, b_tensor, score_tensor).
        self.entries = []

    def set(self, a, b, score):
        self.entries.append(
            (a.detach().clone(), b.detach().clone(),
             torch.tensor(float(score)))
        )

    def part(self, a, b, monotonic=True):
        for ae, be, score in self.entries:
            if torch.equal(ae, a) and torch.equal(be, b):
                return score
        return torch.tensor(0.0)


def _make_layer(n_dim=8):
    return TruthLayer(nDim=n_dim, max_truths=16)


def test_consistency_default_scalar():
    """Back-compat: consistency() with no args returns a scalar."""
    layer = _make_layer()
    layer.truths[0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.count.fill_(1)
    out = layer.consistency()
    assert torch.is_tensor(out)
    assert out.ndim == 0


def test_consistency_report_part_opposite_sign():
    """Two truths: one is part of the other, signs opposite -> contradiction."""
    layer = _make_layer()
    # Truth 0: concept 0 positive pole hot (T)
    layer.truths[0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Truth 1: concept 0 negative pole hot (F) -- same concept, opposite sign
    layer.truths[1] = torch.tensor([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.count.fill_(2)

    basis = _StubBasis()
    basis.set(layer.truths[0], layer.truths[1], 0.8)  # i is part of j
    basis.set(layer.truths[1], layer.truths[0], 0.8)  # j is part of i

    score, contradictions = layer.consistency(basis=basis, return_report=True)
    assert score < 1.0
    assert len(contradictions) == 1
    i, j, desc = contradictions[0]
    assert i < j
    assert (i, j) == (0, 1)
    assert isinstance(desc, str) and desc


def test_consistency_report_no_contradiction_when_same_sign():
    """Part-of overlap with matching signs -> no contradiction."""
    layer = _make_layer()
    layer.truths[0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.truths[1] = torch.tensor([0.8, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.count.fill_(2)

    basis = _StubBasis()
    basis.set(layer.truths[0], layer.truths[1], 0.8)
    basis.set(layer.truths[1], layer.truths[0], 0.8)

    score, contradictions = layer.consistency(basis=basis, return_report=True)
    assert score == 1.0
    assert contradictions == []


def test_consistency_ignores_within_truth_both():
    """A single truth with co-active (T+, T-) is NOT a contradiction.

    Under bivector encoding it's a valid catuṣkoṭi BOTH state; 4-valued
    logic handles it without user intervention. The new report path
    intentionally does not flag within-truth coactivation.
    """
    layer = _make_layer()
    # Concept 3 has both poles hot: storage indices 6 and 7.
    layer.truths[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.8])
    layer.count.fill_(1)

    basis = _StubBasis()
    _, contradictions = layer.consistency(basis=basis, return_report=True)
    assert contradictions == []


def test_suggest_clarifications_template():
    """Template matches the exact plan wording for a clear contradiction."""
    layer = _make_layer()
    layer.truths[0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.truths[1] = torch.tensor([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.count.fill_(2)
    layer._sources = ["A is red", "A is not red"]
    layer._trusts = [0.9, 0.8]

    basis = _StubBasis()
    basis.set(layer.truths[0], layer.truths[1], 0.8)
    basis.set(layer.truths[1], layer.truths[0], 0.8)

    msgs = layer.suggest_clarifications(basis=basis)
    assert len(msgs) == 1
    expected = (
        "'A is red' (trust=0.9) and 'A is not red' (trust=0.8) "
        "appear to contradict — please revise to enable more rational thought."
    )
    assert msgs[0] == expected


def test_suggest_clarifications_missing_source_fallback():
    """Missing source string falls back to '(truth #i)' rather than raising."""
    layer = _make_layer()
    layer.truths[0] = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.truths[1] = torch.tensor([0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    layer.count.fill_(2)
    layer._sources = [None, "A is not red"]
    layer._trusts = [None, 0.8]

    basis = _StubBasis()
    basis.set(layer.truths[0], layer.truths[1], 0.8)
    basis.set(layer.truths[1], layer.truths[0], 0.8)

    msgs = layer.suggest_clarifications(basis=basis)
    assert len(msgs) == 1
    msg = msgs[0]
    assert "(truth #0)" in msg
    assert "trust=unknown" in msg
    assert "A is not red" in msg
    assert "please revise to enable more rational thought." in msg
