"""Reverse PS synthesis + end-to-end round-trip (Phase 8).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("Reverse PS
Synthesis" + Execution-manifest E2E): reverse replays the surface from
ObjectSubSpace route metadata; the end-to-end round-trip is
analyze -> [op, X, Y] -> execute -> reverse -> emit -> surface, with the
marker word lists deleted (the connective is a LEARNED marker, not a
grammar token).
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


def _oss(cap=64, dim=4):
    from Language import ObjectSubSpace
    return ObjectSubSpace(percept_dim=dim, capacity=cap, batch=1)


def test_meronymic_reverse_replays_surface():
    """ObjectSubSpace route metadata (the analysis record) reconstructs the
    observed surface exactly."""
    from perceptual_analyzer import MeronymicAnalyzer
    surface = "the book has a cover"
    an = MeronymicAnalyzer()
    record = an.analyze(surface, _oss(), granularity="word")
    assert an.synthesize(record) == surface


def test_meronymic_synthesis_can_emit_spaces():
    """Synthesis places spaces without a word-tokenizer: the space is a real
    terminal in the record and is replayed."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    record = an.analyze("a b", _oss(), granularity="word")
    recon = an.synthesize(record)
    assert recon == "a b"
    assert " " in recon


def test_grammar_has_no_and_token():
    """The connective 'and' is NOT a grammar token anywhere (marker word
    lists were deleted) -- it can only enter via a learned marker."""
    from Language import Grammar
    g = Grammar()
    g.load_from_grammar_file("complete.grammar")
    for r in g.rules:
        toks = [c.strip() for c in str(r.lhs).split(",")] + list(r.rhs_symbols or ())
        assert "and" not in toks and "AND" not in toks


def test_e2e_round_trip_with_learned_marker():
    """End-to-end: analyze -> [op, X, Y] -> execute -> reverse -> emit ->
    surface, with 'and' a LEARNED marker (absorb/emit), not a grammar token."""
    from perceptual_analyzer import MeronymicAnalyzer
    from Language import GRAMMAR_LAYER_CLASSES
    surface = "cat and dog"
    an = MeronymicAnalyzer()

    # 1. analyze surface into terminals; 'and' is a surface terminal.
    record = an.analyze(surface, _oss(), granularity="word")
    assert "and" in [r["text"] for r in record]

    # 2. the connective is absorbed as conjunction's marker (learned from
    #    co-occurrence); content operands are cat / dog.
    conj = GRAMMAR_LAYER_CLASSES["conjunction"]()
    conj.absorb(left="cat", right="and", marker_id="and")

    # 3. operator-prefixed [conjunction, cat, dog] executes to a combined
    #    meaning (the operator combines, contributing no meaning of its own).
    cat_vec, dog_vec = torch.ones(4), torch.full((4,), 2.0)
    idea = conj.compose(cat_vec, dog_vec)
    assert idea.shape == cat_vec.shape

    # 4. reverse (SS analysis) recovers operands; emit replays the marker.
    # ADAPTED (2026-07-04 serial plan Task 1): the no-basis stub is
    # revoked -- recovery runs the mereology recommender against the
    # operand codebook and must RECOMPOSE faithfully.
    class _Shim:
        def __init__(self, W):
            self._W = W

        def getW(self):
            return self._W

    left_back, right_back = conj.generate(
        idea, basis=_Shim(torch.stack([cat_vec, dog_vec])))
    assert torch.equal(conj.compose(left_back, right_back),
                       idea.expand_as(left_back)
                       if left_back.shape != idea.shape else idea)
    assert conj.emit() == "and"

    # 5. PS meronymic synthesis reconstructs the surface from the
    #    operator-prefixed tree (marker placed at the T2 infix position).
    tree = ("op", conj, ("leaf", "cat"), ("leaf", "dog"))
    assert an.synthesize_tree(tree) == surface
