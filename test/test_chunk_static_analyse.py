"""'analyse' chunking: the meronymic analyzer as the perceptual front end
(Phase R3-live, segmentation layer).

User steering (2026-06-03): the default perceptual operation is a
space-lexer (a ``boundary`` that treats word/space breaks as boundaries).
Within each whitespace-bounded run, bottom-up merge combines the bytes the
analyzer has LEARNED to merge. Cold -- the analyzer codebook holds only the
whole-input vector and every byte -- it can only emit byte terminals, so it
'initially fails' to reproduce word lexing; once every adjacent pair inside
a word is a learned merge, the run collapses to one word terminal and the
result reproduces today's space (lexicon) lexing. Whitespace is always a
hard boundary: merge never crosses a space.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import pytest
from Spaces import PerceptualSpace


def test_analyse_cold_is_byte_level_within_words():
    """No learned merges -> byte terminals within each word run (the
    'initially fails' state: not yet word lexing). Whitespace is dropped,
    mirroring lexicon's stream.split()."""
    assert PerceptualSpace.chunk_static(b"hi ox", "analyse") == [
        b"h", b"i", b"o", b"x"]


def test_analyse_learned_merges_recover_words():
    """A learned within-word merge collapses that run to a word terminal;
    runs without a learned merge stay byte-level."""
    units = PerceptualSpace.chunk_static(b"hi ox", "analyse",
                                         merges={(b"h", b"i")})
    assert units == [b"hi", b"o", b"x"]


def test_analyse_converges_to_lexicon():
    """With every within-word pair learned, analyse reproduces space
    (lexicon) lexing exactly -- 'reproduce the behavior we have now'."""
    merges = {(b"h", b"i"), (b"o", b"x")}
    assert (PerceptualSpace.chunk_static(b"hi ox", "analyse", merges)
            == PerceptualSpace.chunk_static(b"hi ox", "lexicon"))


def test_analyse_never_merges_across_whitespace():
    """Whitespace is a hard boundary: a learned pair that straddles a space
    never fires (the space-lexer split happens first)."""
    units = PerceptualSpace.chunk_static(b"hi ox", "analyse",
                                         merges={(b"i", b"o")})
    assert b"io" not in units
    assert b"".join(units) == b"hiox"


def test_analyse_iterates_merges_to_multibyte_words():
    """Bottom-up merge is iterative: chained learned pairs grow a multi-byte
    word from characters."""
    merges = {(b"t", b"h"), (b"th", b"e")}
    assert PerceptualSpace.chunk_static(b"the", "analyse", merges) == [b"the"]


def test_existing_modes_unchanged():
    """lexicon / bpe behavior is untouched; an unknown mode still raises."""
    assert PerceptualSpace.chunk_static(b"the quick fox", "lexicon") == [
        b"the", b"quick", b"fox"]
    assert PerceptualSpace.chunk_static(b"ab", "bpe") == [b"a", b"b"]
    with pytest.raises(ValueError):
        PerceptualSpace.chunk_static(b"x", "not_a_mode")
