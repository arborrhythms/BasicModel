"""Within-whole division: a type-run WHOLE that is not itself attested divides
by greedy longest-match tiling into attested standalone CONCEPTS.

doc/plans/2026-07-10-wholes-are-types-segmentation.md (T3). Whole != concept:
the whole is the maximal constant-type run (T2); the concepts are the attested
parts that tile it. An attested whole is preferred over its parts (longest
match); an unattested run with no complete attested tiling keeps the existing
byte fallback unchanged.

The live seam is ``WholeSpace.stage_analysis_spans`` ->
``Spaces._divide_spans_into_attested``; attestation comes from the peer
percept store and ``<WholeSpace><divideWithinWhole>`` defaults on.
"""

import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import torch














# -- LIVE seam: WholeSpace.stage_analysis_spans + the RadixLayer store ---------

_D = 8


def _whole_space(nS=32, **cfg_extra):
    """A real WholeSpace built from the test config (the
    test_mereology_word_binding fixture pattern); ``cfg_extra`` lands in
    the WholeSpace config section BEFORE construction (knob reads)."""
    import Models
    import Spaces
    from test_basicmodel import _populate_test_config
    nP = 4
    _populate_test_config(
        inputDim=_D, perceptDim=_D, conceptDim=_D, symbolDim=_D,
        wordDim=_D, outputDim=_D,
        nInput=nP, nPercepts=nP, nConcepts=nS, nSymbols=nS,
        nWords=nS, nOutput=nS, nWhere=0, nWhen=0,
    )
    Models.TheXMLConfig._data["WholeSpace"]["analysis"] = "meronomy"
    Models.TheXMLConfig._data["WholeSpace"].update(cfg_extra)
    return Spaces.WholeSpace([nP, _D], [nS, _D], [nS, _D])


def _radix_store(*attested):
    """A real RadixLayer percept store with ``attested`` byte entries."""
    from Layers import RadixLayer
    ps = RadixLayer(_D)
    for chunk in attested:
        ps.insert(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)
    return ps


def _unity(s, n=None):
    """[1, N] byte unity from a python str, optionally \\0-padded to n
    (the test_type_run_spans fixture)."""
    b = list(s.encode("ascii"))
    if n is not None:
        b = b + [0] * (n - len(b))
    return torch.tensor([b], dtype=torch.long)


def _live_spans(ws, s, n=None):
    """Non-padded (start, end) spans staged by the live method."""
    out = ws.stage_analysis_spans(_unity(s, n))
    return [tuple(sp) for sp in out[0].tolist() if sp != [0, 0]]


def test_live_default_on_divides_punct_run():
    """LIVE path, default knob (absent -> ON): the `")."` punct-whole of
    `"a). b"` divides into two concept spans when `)` and `.` are attested
    (store entries that have appeared as standalone type-runs)."""
    ws = _whole_space()
    assert ws.divide_within_whole is True          # default = live
    ws.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store(")", "."))
    # ")" and "." appear STANDALONE (their own 1-byte type-runs) first --
    # the plan's "independently attested" condition for 1-byte parts.
    _live_spans(ws, "x ) .")
    # T2 wholes: "a"(0,1)  ")."(1,3)  "b"(4,5); T3 divides the punct run.
    assert _live_spans(ws, "a). b") == [(0, 1), (1, 2), (2, 3), (4, 5)]


def test_live_attested_whole_stays_one_concept():
    """LIVE path: `"..."` attested as its own percept stays ONE span even
    though `"."` is also attested (whole preferred over parts); and with
    only `"."` attested it divides into three."""
    ws = _whole_space()
    ws.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store("...", "."))
    _live_spans(ws, "x .")                         # "." appears standalone
    assert _live_spans(ws, "a ...") == [(0, 1), (2, 5)]
    ws2 = _whole_space()
    ws2.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store("."))
    _live_spans(ws2, "x .")
    assert _live_spans(ws2, "a ...") == [(0, 1), (2, 3), (3, 4), (4, 5)]


def test_live_no_complete_tiling_keeps_span():
    """LIVE path: an unattested run with NO complete attested tiling keeps
    the undivided span (the existing fallback), exactly matching the
    knob-off spans."""
    ws_on = _whole_space()
    ws_on.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store(")"))          # "." unattested -> gap
    _live_spans(ws_on, "x ) .")
    ws_off = _whole_space(divideWithinWhole=False)
    assert ws_off.divide_within_whole is False
    ws_off.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store(")", "."))
    assert (_live_spans(ws_on, "a). b") == _live_spans(ws_off, "a). b")
            == [(0, 1), (1, 3), (4, 5)])


def test_live_seeded_bytes_alone_do_not_divide_words():
    """LIVE path: spell_out's lazily-seeded byte percepts do NOT attest --
    an unpromoted word must not re-divide per byte (`"hi"` stays one span
    even with "h"/"i" byte percepts in the store)."""
    ws = _whole_space()
    ws.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store("h", "i"))     # seeded-byte stand-ins
    assert _live_spans(ws, "hi ox") == [(0, 2), (3, 5)]


def test_live_knob_off_restores_pre_t3_spans():
    """Explicit <divideWithinWhole>false</> restores the undivided T2 spans
    even with every part attested."""
    ws = _whole_space(divideWithinWhole=False)
    ws.perceptualSpace_ref = types.SimpleNamespace(
        percept_store=_radix_store(")", ".", "a", "b"))
    assert _live_spans(ws, "a). b") == [(0, 1), (1, 3), (4, 5)]


def test_live_no_store_is_inert():
    """Without a reachable percept store the division is a no-op (byte-mode
    and lexicon-mode configs stage undivided spans unchanged)."""
    ws = _whole_space()
    assert ws.perceptualSpace_ref is None
    assert _live_spans(ws, "a). b") == [(0, 1), (1, 3), (4, 5)]


def test_xsd_accepts_divide_within_whole():
    """model.xsd validates a config carrying the new WholeSpace element."""
    try:
        import lxml.etree as ET
    except ImportError:
        import pytest
        pytest.skip("lxml not installed")
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    schema = ET.XMLSchema(ET.parse(os.path.join(data_dir, "model.xsd")))
    doc = ET.parse(os.path.join(data_dir, "model.xml"))
    ws = doc.find(".//WholeSpace")
    assert ws is not None
    el = ET.SubElement(ws, "divideWithinWhole")
    el.text = "false"
    assert schema.validate(doc), "\n".join(
        str(e.message) for e in schema.error_log)
