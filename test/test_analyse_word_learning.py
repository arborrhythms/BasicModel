"""Bottom-up word learning for the analyse front end (Phase R3-live, D).

User steering (2026-06-03): words are learned from characters through
bottom-up merge. ``PerceptualSpace.learn_merges`` accumulates the merges
(BPE-style, space-bounded) the analyzer learns from a corpus; once the
within-word pairs are learned, ``chunk_static(..., 'analyse', merges)``
reproduces word (lexicon) lexing -- "reproduce the behavior we have now".
Cold (no learned merges) it stays byte-level ("initially fails").
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Spaces import PerceptualSpace


_CORPUS = [b"hello world", b"hello there", b"a new world", b"new hello"]


def test_learned_merges_converge_to_word_lexing():
    """After learning enough merges, analyse reproduces space (lexicon)
    lexing on the corpus words -- the converged target behavior."""
    merges = PerceptualSpace.learn_merges(_CORPUS, num_merges=64)
    for line in _CORPUS:
        assert (PerceptualSpace.chunk_static(line, "analyse", merges)
                == PerceptualSpace.chunk_static(line, "lexicon")), line


def test_cold_is_byte_level_before_learning():
    """With no learned merges the analyzer is byte-level (initially fails)."""
    assert PerceptualSpace.chunk_static(b"hello", "analyse") == [
        bytes([c]) for c in b"hello"]


def test_partial_learning_is_subword():
    """A single learned merge collapses one pair -> sub-word units, fewer
    than the byte count but not yet the whole word."""
    merges = PerceptualSpace.learn_merges([b"ab ab ab ab"], num_merges=1)
    units = PerceptualSpace.chunk_static(b"ab", "analyse", merges)
    assert units == [b"ab"]                       # the only pair learned
    units3 = PerceptualSpace.chunk_static(b"abc", "analyse", merges)
    assert units3 == [b"ab", b"c"]                # 'ab' merged, 'c' alone


def test_learning_respects_whitespace_boundaries():
    """Merges are learned WITHIN words only -- a cross-space pair is never
    learned (the space-lexer bounds the statistics)."""
    merges = PerceptualSpace.learn_merges([b"ax xa"] * 5, num_merges=8)
    # 'x x' across the space must never become a learned merge.
    assert (b"x", b"x") not in set(
        tuple(m) if not isinstance(m, tuple) else m for m in merges)
    # within-word pairs do get learned.
    assert PerceptualSpace.chunk_static(b"ax", "analyse", merges) == [b"ax"]
