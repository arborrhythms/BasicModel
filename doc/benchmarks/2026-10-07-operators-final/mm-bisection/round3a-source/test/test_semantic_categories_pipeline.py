"""Semantic category grouping from explicit operator consequences/signatures.

The retired WholeSpace operator shaper supplied these inputs previously.
These are grouping mechanisms; chooser training is tested on LanguageSpace.
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


def test_consequence_vectors_categorize_by_operator_effect():
    """Equal consequence signatures cluster across argument positions."""
    from semantic_categories import recover_semantic_categories
    a = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    b = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    from Language import GRAMMAR_LAYER_CLASSES
    op_vectors = {name: GRAMMAR_LAYER_CLASSES[name]().compose(a, b).reshape(-1)
                  for name in ("conjunction", "disjunction")}
    participation = {
        "n1": {("conjunction", 0)},
        "n2": {("conjunction", 1)},   # same operator, other position
        "v1": {("disjunction", 0)},
    }
    cls = recover_semantic_categories(participation, op_vectors)
    assert cls["n1"] == cls["n2"], cls       # same operator effect
    assert cls["n1"] != cls["v1"], cls       # opposite effect


def test_recovers_on_transitional_grammar_participation():
    """Shared operator participation collapses order-variant categories."""
    from Language import Grammar, GRAMMAR_LAYER_CLASSES
    from participation import role_participation
    from semantic_categories import recover_semantic_categories

    g = Grammar()
    # The transitional POS-categoried baseline, archived as a fixture
    # (GrammarOpsPass §1: data/complete.grammar is now role-collapsed).
    g.load_from_grammar_file(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures",
        "transitional_pos.grammar"))
    part = role_participation(g)
    # Explicit distinct operator signatures isolate the category grouping;
    # no retired WholeSpace operator dictionary supplies them.
    names = sorted({m for entries in part.values() for m, _ in entries
                    if m in GRAMMAR_LAYER_CLASSES})
    basis = torch.eye(len(names))
    op_vectors = dict(zip(names, basis))
    cls = recover_semantic_categories(part, op_vectors, threshold=0.999)
    assert cls, "expected recovered categories"

    def cid(sym):
        return cls.get(sym)
    for fam in (("CONJ_L3", "CONJ_L4", "CONJ_L5"),
                ("DISJ_R3", "DISJ_R4", "DISJ_R5")):
        present = [s for s in fam if s in cls]
        assert present, fam
        assert len({cid(s) for s in present}) == 1, (fam, [cid(s) for s in present])
