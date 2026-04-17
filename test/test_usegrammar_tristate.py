import pytest
from basicmodel.bin.util import parse_use_grammar


def test_parse_string_all():
    assert parse_use_grammar("all") == "all"


def test_parse_string_thoughtfree():
    assert parse_use_grammar("thoughtFree") == "thoughtFree"


def test_parse_string_none():
    assert parse_use_grammar("none") == "none"


def test_parse_legacy_boolean_true_no_thoughtfree():
    # Old schema: useGrammar=true, thought_free=false → "all"
    assert parse_use_grammar(True, thought_free=False) == "all"


def test_parse_legacy_boolean_true_thoughtfree():
    # Old schema: useGrammar=true, thought_free=true → "thoughtFree"
    assert parse_use_grammar(True, thought_free=True) == "thoughtFree"


def test_parse_legacy_boolean_false():
    # Old schema: useGrammar=false → "none"
    assert parse_use_grammar(False, thought_free=False) == "none"


def test_parse_invalid_string_raises():
    with pytest.raises(ValueError):
        parse_use_grammar("maybe")
