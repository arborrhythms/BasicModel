"""Byte fallback must be UTF-8 exact (spec §9 carry-forward concern).

doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md
§9 "Carry-forward concerns": byte terminals should preserve the original
byte offsets and reconstruct the original surface, including non-ASCII text,
without mojibake or overlapping spans. The routed analyzer must therefore
carry the exact bytes of each terminal and reconstruct from bytes, never
interpolate ``decode(errors="replace")`` mojibake for a multi-byte glyph the
router split across byte terminals.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch


def _oss(cap=128, dim=4):
    from Language import ObjectSubSpace
    return ObjectSubSpace(percept_dim=dim, capacity=cap, batch=1)


def test_routed_byte_fallback_round_trips_non_ascii():
    """A non-ASCII surface with no percept store routes to byte terminals and
    reconstructs EXACTLY -- no mojibake, even when a multi-byte glyph is split
    across byte terminals."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()                 # no percept_lookup -> byte fallback
    for surface in ("café déjà", "naïve—oz", "日本語 test", "aéb"):
        oss = _oss()
        record = an.analyze_routed(surface, oss)
        assert an.synthesize(record) == surface, surface


def test_routed_byte_terminals_preserve_offsets_and_dont_overlap():
    """Terminal spans are contiguous byte offsets covering the whole surface
    with no overlap (binary-tiling guarantee), in UTF-8 byte units."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    surface = "café déjà"
    nbytes = len(surface.encode("utf-8"))
    record = an.analyze_routed(surface, _oss())
    spans = sorted((r["start"], r["end"]) for r in record)
    assert spans[0][0] == 0
    assert spans[-1][1] == nbytes
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        assert e0 == s1, (spans)            # contiguous, non-overlapping


def test_split_multibyte_glyph_has_no_replacement_char():
    """A byte terminal that is a partial UTF-8 sequence must NOT be recorded
    as the U+FFFD replacement char -- the exact bytes are preserved."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    record = an.analyze_routed("é", _oss())  # 0xC3 0xA9, likely split to bytes
    assert "�" not in an.synthesize(record)
    # Every terminal carries its exact source bytes summing to the surface.
    raws = b"".join(r["raw"] for r in sorted(record, key=lambda r: r["start"]))
    assert raws == "é".encode("utf-8")


def test_known_word_with_byte_fallback_mix_round_trips():
    """A mixed surface -- an ASCII known word plus a non-ASCII byte-fallback
    region -- still yields one stop terminal for the known word, every
    terminal carries its exact bytes, and the whole surface round-trips.
    (The word lexer splits non-ASCII words, so 'café' here is byte fallback;
    byte-exact reconstruction makes that lossless anyway.)"""
    from perceptual_analyzer import MeronymicAnalyzer
    known = {"book": (torch.ones(4), 7)}
    an = MeronymicAnalyzer(percept_lookup=lambda t: known.get(t))
    oss = _oss()
    record = an.analyze_routed("book café", oss)
    assert an.synthesize(record) == "book café"
    stop = [r for r in record if r["part_id"] == 7]
    assert len(stop) == 1 and stop[0]["text"] == "book"
    assert all("raw" in r for r in record)
