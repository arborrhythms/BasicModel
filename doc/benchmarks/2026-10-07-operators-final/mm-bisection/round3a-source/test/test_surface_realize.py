"""rewrite() bidirectional surface realizer (Stage D2). surface_morphology now
has BOTH directions: analyze (surface -> lemma+features, comprehension) and
realize (lemma+features -> surface, generation). Together they are the
bidirectional realizer; many surface forms collapse to ONE lexeme.
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import surface_morphology as sm


def _f(tense="PRESENT", aspect=None):
    return {"tense": tense, "aspect": aspect or []}


def test_realize_irregular():
    assert sm.realize("run", _f("PAST")) == "ran"
    assert sm.realize("run", _f("PRESENT", ["PROGRESSIVE"])) == "running"
    assert sm.realize("be", _f("PAST")) == "was"
    assert sm.realize("have", _f("PAST")) == "had"
    assert sm.realize("do", _f("PAST")) == "did"


def test_realize_regular_rules():
    assert sm.realize("walk", _f("PAST")) == "walked"
    assert sm.realize("stop", _f("PAST")) == "stopped"      # CVC doubling
    assert sm.realize("like", _f("PAST")) == "liked"        # e -> +d
    assert sm.realize("walk", _f("PRESENT", ["PROGRESSIVE"])) == "walking"
    assert sm.realize("make", _f("PRESENT", ["PROGRESSIVE"])) == "making"  # e-drop
    assert sm.realize("walk", _f("PRESENT")) == "walk"      # present simple = lemma


def test_bidirectional_roundtrip():
    """analyze(realize(lemma, feats)) recovers (lemma, feats) for covered forms."""
    cases = [
        ("run", _f("PAST")), ("run", _f("PRESENT", ["PROGRESSIVE"])),
        ("walk", _f("PAST")), ("stop", _f("PAST")),
        ("walk", _f("PRESENT", ["PROGRESSIVE"])),
    ]
    for lemma, feats in cases:
        surf = sm.realize(lemma, feats)
        lem2, feats2 = sm.analyze(surf)
        assert lem2 == lemma, f"{surf}: lemma {lem2} != {lemma}"
        assert feats2.get("tense") == feats["tense"], f"{surf}: tense mismatch"
        assert bool(feats2.get("aspect")) == bool(feats["aspect"]), f"{surf}: aspect"


def test_many_surfaces_one_lexeme():
    """ran / running / run all collapse to the single lexeme 'run' (the
    codebook is over lexemes, not surface forms)."""
    assert sm.analyze("ran")[0] == "run"
    assert sm.analyze("running")[0] == "run"
    assert sm.analyze("run")[0] == "run"
    # ... and was/were/being/is all collapse to 'be'
    for s in ("was", "being", "is"):
        assert sm.analyze(s)[0] == "be"
