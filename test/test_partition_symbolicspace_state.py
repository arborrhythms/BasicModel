"""Smoke tests that survived the Stage-4 active-payload retirement.

The Task 6.1 forward-contract tests in this file were deleted on
2026-05-29: they tested the legacy setW-writes-per-batch-shadow contract
that doc/plans/2026-05-21-active-payload-retirement.md §4 retired.
Under the new spec, ``forward()`` updates the SELECTION on ``_active``
and the prototype Parameter is stable, so the old assertions were no
longer meaningful. See git log for the deleted code.

What remains is the chunk-static modes sentinel (Task 7.1
characterization) which is independent of the deleted contract.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Spaces


# ---------------------------------------------------------------------------
# Task 7.1 -- characterization: chunking behavior must be unchanged by the
# grammar-dispatch refactor.  PerceptualSpace.chunk_static is a pure static
# method (no space state), so we can call it without constructing any Space
# chain.  Covers the raw and lexicon modes that survive the refactor.
# ---------------------------------------------------------------------------

def test_chunk_static_modes_smoke():
    """Smoke-test the supported chunking modes.  Task 7.2 removed the
    grammar-client call from PerceptualSpace.forward; ``chunk_static``
    itself was never in scope, so this is a belt-and-braces sentinel
    against accidental drift of the two-way (bpe | lexicon) switch.
    """
    lex = Spaces.PerceptualSpace.chunk_static(b"the cat sat", mode="lexicon")
    assert lex == [b"the", b"cat", b"sat"]
    bpe = Spaces.PerceptualSpace.chunk_static(b"abc", mode="bpe")
    assert bpe == [b"a", b"b", b"c"]
