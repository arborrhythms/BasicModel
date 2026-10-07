"""Phase 7 / Task 7.1: bin/surface_morphology.analyze (pure table; fixes the
-ed/-ing over-fire). doc/plans/2026-06-03-modality-architecture-plan.md."""

import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))

from surface_morphology import analyze


def test_verb_subsumes_surface_tense():
    assert analyze("ran")     == ("run",  {"tense": "PAST",    "aspect": []})
    assert analyze("running") == ("run",  {"tense": "PRESENT", "aspect": ["PROGRESSIVE"]})  # bare participle


def test_lemmatizer_overfire_fixed():
    assert analyze("seed")[0]  == "seed"     # NOT "se"
    assert analyze("freed")[0] == "freed"    # NOT "fre"


def test_plain_token_passthrough():
    assert analyze("cat") == ("cat", {})


def test_regular_inflection_strips():
    # Regular -ed / -ing inflections still lemmatize (the gate passes).
    assert analyze("walked")  == ("walk", {"tense": "PAST",    "aspect": []})
    assert analyze("walking") == ("walk", {"tense": "PRESENT", "aspect": ["PROGRESSIVE"]})
    assert analyze("stopped")[0] == "stop"   # CVC de-doubling


def test_features_are_role_neutral_dicts():
    # Non-verbs carry no features; verbs carry only tense/aspect (no POS).
    _lemma, feats = analyze("cat")
    assert feats == {}
    _lemma, feats = analyze("ran")
    assert set(feats) == {"tense", "aspect"}


if __name__ == "__main__":
    import unittest
    unittest.main()
