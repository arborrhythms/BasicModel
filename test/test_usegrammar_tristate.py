import pytest
from util import parse_use_grammar


def test_parse_string_all():
    assert parse_use_grammar("all") == "all"


def test_parse_string_thoughtfree():
    assert parse_use_grammar("thoughtFree") == "thoughtFree"


def test_parse_string_none():
    assert parse_use_grammar("none") == "none"


def test_parse_invalid_string_raises():
    with pytest.raises(ValueError):
        parse_use_grammar("maybe")


def test_parse_non_string_raises():
    with pytest.raises(ValueError):
        parse_use_grammar(True)
