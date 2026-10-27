"""The retired test-only formation ladder has no production entry point."""


def test_retired_conceptualize_api_is_absent():
    from Language import SymbolSubSpace
    assert not hasattr(SymbolSubSpace, 'conceptualize')
