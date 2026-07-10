"""Serial-plan Task 3 (NULL / sentence-end pathway).

The ``\x00`` NULL byte is the sentence TERMINATOR -- a real ASCII value the
lexer appends to every input (Spaces.py ``_terminate``) and the reconstruct
RECOGNIZES to bound the decoded length (Alec 2026-07-09: "the NULL terminator
is an ascii value that should be processed"). This is how sentence LENGTH
round-trips blind: the decode stops at the terminator, NOT at the padded slab
width, and content stamped past the terminator is dropped.

Padding-cost (Task 3.2) is already the eager skip-padding behavior
(``_n_trips = min(N_words, N_loop)``, Models.py ~8133; pinned by
test_per_word_ss_padding_noop / test_compile_static_loop): the loop slab stays
fixed but the body runs only for the columns with a valid word. No all-zeros
NULL word is added to the fold, so the reduce identity is untouched.

Full blind length round-trip (right words AND right count) on the fixed-length
xor sentences is covered by test_blind_decode::test_mm20m_xor_blind_roundtrip
(exact_match == 1.0). This file pins the VARIABLE-length piece xor does not
exercise: a terminator that cuts content stamped past sentence end.
"""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_BIN = os.path.join(os.path.dirname(_HERE), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import _render_token_buffer  # noqa: E402


def test_terminator_bounds_length_and_drops_content_past_end():
    """The ``\x00`` terminator recognizes sentence END: the render is bounded at
    the terminator offset and any content stamped PAST it is dropped -- the
    blind length round-trip that the fixed-length xor sentences cannot show."""
    # 'hi' @0, 'there' @3, TERMINATOR @8, then junk 'X' @9 (past the end).
    tokens = [("hi", 0), ("there", 3), ("\x00", 8), ("X", 9)]
    out = _render_token_buffer(tokens)
    # Bounded at the terminator (offset 8) -> the junk at 9 is dropped; the
    # terminator itself renders as \x00 at index 8 (callers rstrip it).
    assert out.rstrip("\x00") == "hi there", repr(out)
    assert "X" not in out, repr(out)


def test_no_terminator_uses_content_extent():
    """Without a terminator the render length is the content extent (max end),
    so the terminator is the ONLY length signal it honors -- not a silent cap."""
    tokens = [("hi", 0), ("there", 3)]
    assert _render_token_buffer(tokens).rstrip("\x00") == "hi there"


def test_positionless_tokens_break_at_terminator():
    """The offset-free (consecutive-join) render path also stops at the
    terminator (right count when positions are absent)."""
    tokens = [("hi", None), ("there", None), ("\x00", None), ("X", None)]
    out = _render_token_buffer(tokens)
    assert out.rstrip("\x00") == "hithere", repr(out)   # breaks at terminator
    assert "X" not in out, repr(out)                     # junk past end dropped


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
