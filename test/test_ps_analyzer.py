"""MeronymicAnalyzer compatibility mode (PS forward analysis).

doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md ("Forward PS
Analysis", "Analyzer Contract"): the first analyzer mode uses stop /
whitespace boundary / uniform + byte fallback and reproduces the current
word-lexer terminal sequence. It writes durable spans to an ObjectSubSpace
and exposes a fixed-capacity terminal-stream view with endpoint-sum .where.
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


def _oss(cap=64, dim=4):
    from Language import ObjectSubSpace
    return ObjectSubSpace(percept_dim=dim, capacity=cap, batch=1)


def test_boundary_matches_word_lexer():
    """With whitespace-boundary segmentation the analyzer's spans match the
    current word lexer (util.parse lex='words')."""
    from perceptual_analyzer import MeronymicAnalyzer
    from util import parse
    surface = "the book has a cover"
    oss = _oss()
    an = MeronymicAnalyzer()
    an.analyze(surface, oss, granularity="word")

    expected = [(s, s + len(t.encode("utf-8"))) for (t, s) in parse(surface, "words")]
    got = [(oss.get(0, i)["span_start"], oss.get(0, i)["span_end"])
           for i in range(oss.depth(0))]
    assert got == expected, (got, expected)
    assert oss.depth(0) == len(expected)


def test_ps_analyzer_byte_fallback():
    """An unknown surface is covered by byte terminals (total fallback)."""
    from perceptual_analyzer import MeronymicAnalyzer
    oss = _oss()
    an = MeronymicAnalyzer()
    an.analyze("zq", oss, granularity="byte")
    assert oss.depth(0) == 2
    for i, (s, e) in enumerate([(0, 1), (1, 2)]):
        t = oss.get(0, i)
        assert t["part_id"] == -1            # byte fallback = unknown id
        assert t["span_start"] == s and t["span_end"] == e
        assert float(torch.linalg.vector_norm(t["vec"])) > 0


def test_ps_analyzer_prefers_known_word():
    """Once 'book' is a known percept, the analyzer emits ONE terminal for
    it instead of four byte terminals; an unknown word falls back to bytes."""
    from perceptual_analyzer import MeronymicAnalyzer
    known = {"book": (torch.ones(4), 11)}
    an = MeronymicAnalyzer(percept_lookup=lambda t: known.get(t))

    oss1 = _oss()
    an.analyze("book", oss1, granularity="auto")
    assert oss1.depth(0) == 1
    assert oss1.get(0, 0)["part_id"] == 11

    oss2 = _oss()
    an.analyze("xyz", oss2, granularity="auto")
    assert oss2.depth(0) == 3              # 3 byte terminals
    assert all(oss2.get(0, i)["part_id"] == -1 for i in range(3))


def test_terminal_stream_fixed_capacity():
    """The terminal view keeps fixed [1, cap, D] shape while len changes."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    oss_a = _oss(cap=64, dim=4)
    an.analyze("a b", oss_a, granularity="word")
    va = an.terminal_view(oss_a)
    oss_b = _oss(cap=64, dim=4)
    an.analyze("a b c d", oss_b, granularity="word")
    vb = an.terminal_view(oss_b)
    assert va["what"].shape == vb["what"].shape == (1, 64, 4)
    assert int(va["len"][0]) != int(vb["len"][0])
    # mask matches len
    assert int(va["mask"][0].sum()) == int(va["len"][0])


def test_terminal_stream_where_is_endpoint_sum():
    """Each terminal's .where is the endpoint-sum key of its source span."""
    from perceptual_analyzer import MeronymicAnalyzer
    an = MeronymicAnalyzer()
    surface = "hello world"
    oss = _oss()
    an.analyze(surface, oss, granularity="word")
    view = an.terminal_view(oss)
    n = int(view["len"][0])
    for i in range(n):
        s, e = an.where.decode(view["where"][0, i])
        assert (s, e) == (oss.get(0, i)["span_start"], oss.get(0, i)["span_end"])
