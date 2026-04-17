"""Tests for the Stage 2 merged-loop scaffolding: WordSpace.done(),
WordSpace.reset_emit_state(), SymbolicSpace.empty_state().

The full mode-blind outer loop is a separate task (5.3). These tests only
verify the building blocks are in place.
"""
import inspect

from basicmodel.bin.Spaces import WordSpace, SymbolicSpace


def test_wordspace_exposes_done_method():
    assert callable(getattr(WordSpace, "done", None))


def test_wordspace_exposes_reset_emit_state():
    assert callable(getattr(WordSpace, "reset_emit_state", None))


def test_wordspace_exposes_note_emit():
    assert callable(getattr(WordSpace, "note_emit", None))


def test_symbolicspace_exposes_empty_state():
    assert callable(getattr(SymbolicSpace, "empty_state", None))


def test_wordspace_done_terminates_after_single_emit_without_parent():
    """Without _parent set, done() falls back to single-emit semantics
    (parallel-mode default)."""
    # We construct a bare object bypassing __init__ to avoid full model setup.
    ws = WordSpace.__new__(WordSpace)
    ws._emit_count = 0
    ws._n_percepts_consumed = 0
    ws._parent = None
    assert ws.done() is False
    ws.note_emit()
    assert ws.done() is True


def test_wordspace_done_serial_requires_nPercepts_emits():
    """With _parent.useGrammar=='all', done() needs nPercepts consumed emissions."""
    ws = WordSpace.__new__(WordSpace)
    ws._emit_count = 0
    ws._n_percepts_consumed = 0

    class FakeParent:
        useGrammar = "all"
        nPercepts = 3

    ws._parent = FakeParent()
    assert ws.done() is False
    ws.note_emit(consumed_percept=True)
    assert ws.done() is False
    ws.note_emit(consumed_percept=True)
    assert ws.done() is False
    ws.note_emit(consumed_percept=True)
    assert ws.done() is True


def test_wordspace_reset_emit_state_restores_zero():
    ws = WordSpace.__new__(WordSpace)
    ws._emit_count = 5
    ws._n_percepts_consumed = 5
    ws.reset_emit_state()
    assert ws._emit_count == 0
    assert ws._n_percepts_consumed == 0


def test_done_method_lives_on_wordspace_class():
    """Sanity: the implementation source references _parent.useGrammar."""
    src = inspect.getsource(WordSpace.done)
    assert "useGrammar" in src
    assert "all" in src
