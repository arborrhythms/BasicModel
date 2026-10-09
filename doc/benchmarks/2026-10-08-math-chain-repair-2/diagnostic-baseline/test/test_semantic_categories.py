"""Grammatical categories from the semantic effect of operators (R4-sem 2).

User steering (2026-06-03): grammatical categories are distinguished by the
SEMANTIC EFFECT of their associated operators -- the operator's codebook
vector (shaped live by the soft superposition, R4-sem 1). A symbol's
category is recovered by clustering on the aggregate of its operators'
vectors. The payoff over structural participation (R4): two symbols whose
DIFFERENT operators have the SAME semantic effect unify into one category --
which structural slot-membership alone cannot do.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


# opX and opY have the SAME semantic-effect vector; opZ differs.
_OP_VECTORS = {
    "opX": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    "opY": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    "opZ": torch.tensor([0.0, 1.0, 0.0, 0.0]),
}
# a participates only in opX, b only in opY, c only in opZ.
_PARTICIPATION = {
    "a": {("opX", 0)},
    "b": {("opY", 0)},
    "c": {("opZ", 0)},
}


def test_semantic_signature_aggregates_operator_vectors():
    from semantic_categories import semantic_signature
    sig = semantic_signature({("opX", 0)}, _OP_VECTORS)
    assert torch.allclose(sig, _OP_VECTORS["opX"])


def test_semantic_recovery_unifies_same_effect_operators():
    """a (opX) and b (opY) unify -- their operators share a semantic effect;
    c (opZ) stays separate."""
    from semantic_categories import recover_semantic_categories
    cls = recover_semantic_categories(_PARTICIPATION, _OP_VECTORS)
    assert cls["a"] == cls["b"], cls
    assert cls["a"] != cls["c"], cls


def test_structural_participation_keeps_them_separate():
    """Contrast (the R4 gap): pure structural participation keeps a and b
    apart because they occupy DIFFERENT operator slots."""
    from participation import cluster_by_participation
    structural = cluster_by_participation(_PARTICIPATION)
    assert structural["a"] != structural["b"]


def test_threshold_controls_semantic_grouping():
    """A loose threshold groups merely-similar effects; a strict one splits
    them."""
    from semantic_categories import recover_semantic_categories
    ops = {
        "p": torch.tensor([1.0, 0.0, 0.0]),
        "q": torch.tensor([0.9, 0.1, 0.0]),   # near p
    }
    part = {"x": {("p", 0)}, "y": {("q", 0)}}
    assert recover_semantic_categories(part, ops, threshold=0.999)["x"] != \
        recover_semantic_categories(part, ops, threshold=0.999)["y"]
    loose = recover_semantic_categories(part, ops, threshold=0.9)
    assert loose["x"] == loose["y"]
