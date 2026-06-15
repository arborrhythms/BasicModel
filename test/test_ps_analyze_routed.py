"""Learned-route PS analysis wired into MeronymicAnalyzer (Phase R3).

The meronymic Viterbi router replaces the util.parse heuristic as the
analysis path: known words cohere into one stop terminal, unknown surface
stays byte terminals -- the same known-vs-byte cover as the compatibility
analyzer, now produced by the SHARED DP primitive. The analyzer's
terminal_view is the seam PartSpace.forward / the signal-router
dispatch consume, so routing here IS the PS-forward analysis path.
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


def test_routed_known_word_is_one_terminal():
    """A known word routes to ONE stop terminal (its percept id); an unknown
    word falls back to byte terminals -- via the learned DP, not util.parse."""
    from perceptual_analyzer import MeronymicAnalyzer
    known = {"book": (torch.ones(4), 11)}
    an = MeronymicAnalyzer(percept_lookup=lambda t: known.get(t))

    oss1 = _oss()
    an.analyze_routed("book", oss1)
    assert oss1.depth(0) == 1
    assert oss1.get(0, 0)["part_id"] == 11
    assert oss1.get(0, 0)["span_start"] == 0 and oss1.get(0, 0)["span_end"] == 4

    oss2 = _oss()
    an.analyze_routed("xyz", oss2)
    assert oss2.depth(0) == 3
    assert all(oss2.get(0, i)["part_id"] == -1 for i in range(3))


def test_routed_analysis_round_trips():
    """Reverse synthesis from the routed analysis record replays the
    surface exactly (markers replayed as their terminals)."""
    from perceptual_analyzer import MeronymicAnalyzer
    known = {"book": (torch.ones(4), 11)}
    an = MeronymicAnalyzer(percept_lookup=lambda t: known.get(t))
    oss = _oss()
    record = an.analyze_routed("book xyz", oss)
    assert an.synthesize(record) == "book xyz"
    # The known word is one stop terminal; the rest are byte terminals.
    parts = [r["text"] for r in sorted(record, key=lambda r: r["start"])]
    assert "book" in parts


def test_routed_terminal_view_consumable():
    """The routed analysis exposes the fixed-capacity terminal stream the
    PS-to-SS binding / PS.forward consume, with endpoint-sum .where."""
    from perceptual_analyzer import MeronymicAnalyzer
    known = {"book": (torch.ones(4), 11)}
    an = MeronymicAnalyzer(percept_lookup=lambda t: known.get(t))
    oss = _oss()
    an.analyze_routed("book", oss)
    view = an.terminal_view(oss)
    assert view["what"].shape == (1, 64, 4)
    assert int(view["len"][0]) == 1
    s, e = an.where.decode(view["where"][0, 0])
    assert (s, e) == (0, 4)
