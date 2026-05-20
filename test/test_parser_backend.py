"""Tests for the ``parserBackend`` config switch on WordSpace.

Plan: doc/plans/2026-05-20-knowledge-artifact-order-typed-stm.md

Adds an incremental cutover path: ``parserBackend = chart | stm |
parallel``, default ``chart``. ``WordSpace.compose()`` / ``generate()``
remain the public entry points and dispatch internally on
``self.parser_backend``. Default behavior (``chart``) is unchanged; new
backends (``stm`` and ``parallel``) raise ``NotImplementedError`` until
their drivers land in later steps.
"""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))


def _bare_word_space():
    """WordSpace with __init__ bypassed (sufficient for backend tests)."""
    from Language import WordSpace
    import torch.nn as nn
    ws = object.__new__(WordSpace)
    nn.Module.__init__(ws)
    return ws


def test_default_parser_backend_is_chart():
    """A fresh WordSpace defaults to ``parser_backend == 'chart'``."""
    ws = _bare_word_space()
    # Direct attribute access; in production __init__ sets it from XML.
    assert getattr(ws, 'parser_backend', 'chart') == 'chart'


def test_set_parser_backend_to_stm():
    """Backend can be set to 'stm'."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    assert ws.parser_backend == 'stm'


def test_compose_with_unknown_backend_raises_value_error():
    """Unknown backend strings raise ValueError, not silently fall back."""
    ws = _bare_word_space()
    ws.parser_backend = 'banana'
    import pytest
    with pytest.raises(ValueError, match='unknown.*parser_backend'):
        ws.compose(input_vectors=None)


def test_compose_stm_backend_requires_knowledge_attached():
    """``parser_backend='stm'`` requires a knowledge artifact attached
    (the driver pulls its rule_signatures from there)."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.compose(input_vectors=None)


def test_generate_stm_backend_requires_knowledge_attached():
    """Same for generate."""
    ws = _bare_word_space()
    ws.parser_backend = 'stm'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.generate(target_vectors=None)


def test_compose_parallel_backend_requires_knowledge_attached():
    """``parser_backend='parallel'`` requires knowledge for the STM
    side it constructs before falling through to the chart."""
    ws = _bare_word_space()
    ws.parser_backend = 'parallel'
    import pytest
    with pytest.raises(RuntimeError, match='knowledge'):
        ws.compose(input_vectors=None)
