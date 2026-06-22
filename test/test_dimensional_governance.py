"""Dimensional-governance gates (doc/specs/2026-06-05-dimensional-governance.md)."""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Spaces

import torch

def test_sigma_pi_mode_resolves_and_aliases_butterfly():
    f = Spaces.Space.sigma_pi_mode  # staticmethod (raw value, str|bool) -> mode
    assert f("last") == "last"
    assert f("butterfly") == "butterfly"
    assert f("full") == "full"
    assert f(True) == "butterfly"      # legacy <butterfly>true</butterfly>
    assert f(False) == "last"          # legacy <butterfly>false</butterfly>
    assert f(None) == "last"           # absent


def test_full_sigma_pi_is_invertible_over_flat_slab():
    from Layers import SigmaLayer
    B, N, D = 2, 8, 6
    op = SigmaLayer(N * D, N * D, invertible=True, nonlinear=True,
                    stable=True)  # full: dense square over the flattened slab
    x = torch.randn(B, N, D).clamp(-0.5, 0.5)
    flat = x.reshape(B, N * D)
    y = op.forward(flat)
    x_rec = op.reverse(y).reshape(B, N, D)
    assert (x - x_rec).abs().max().item() < 1e-3


def _build(cfg_name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", cfg_name)
    init_config(path=p, defaults_path=os.path.join(os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_mm_5m_builds_and_forwards():
    import torch, Models
    m = _build("MM_20M.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    inp_items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()


def test_mm_5m_reconstructs():
    # Parallel ``_forward_per_stage`` returns last_reconstruction=None
    # (reconstruction is a runBatch-time loss term; the config carries
    # <reconstruct>concepts</reconstruct>). The spec's invertible-bridge
    # claim (sec.5: "one invertible matrix serves both legs") is gated here
    # directly: the SS fold (``pi`` post Pi/Sigma swap, rev. 2026-06-09)
    # butterfly round-trips, so the deep CS hub is recoverable from the
    # symbol distribution via pi.reverse.
    import torch, Models
    m = _build("MM_20M.xml"); m.eval()
    sig = m.wholeSpace.pi
    n = int(m.wholeSpace.inputShape[0])
    d = int(getattr(m.wholeSpace, "nOutputDim", 0) or m.wholeSpace.nDim)
    x = torch.randn(2, n, d).clamp(-0.5, 0.5)
    y = sig.forward(x)
    x_rec = sig.reverse(y)
    assert torch.isfinite(y).all() and torch.isfinite(x_rec).all()
    assert (x - x_rec).abs().max().item() < 1e-2


def test_serial_relaxes_symbol_dim_passthrough():
    # Phase-3 relax (pulled forward, doc/specs/2026-06-05 sec.4/sec.6): in
    # SERIAL mode the bounded-STM grammar fold bridges CS<->SS, so SS content
    # may differ from CS content (the small symbol code). validate_config
    # must NOT impose symbol_dim == concept_dim for a serial config.
    # MM_20M_grammar.xml has SS content 8 != CS content 1024, so from_config
    # must not raise the symbol_dim==concept pass-through ValueError.
    try:
        _build("MM_20M_grammar.xml")
    except ValueError as e:
        assert "symbol_dim" not in str(e), str(e)


def test_converted_grammars_load_role_collapsed():
    from Language import Grammar
    for g in ("xor.grammar", "default.grammar", "shamatha.grammar"):
        gr = Grammar()
        gr.load_from_grammar_file(g)
        assert len(gr.rules) > 0, g
        assert gr.ps_start_symbol == "U", (g, gr.ps_start_symbol)
        assert any("_O1" in (r.lhs or "") for r in gr.rules), g  # role form


# ---------------------------------------------------------------------------
# Task C1: IS->PS->CS->SS->OS handoff invariant is a FAIL-LOUD config error.
# (doc/specs/2026-06-05-dimensional-governance.md sec.4/sec.6;
#  doc/plans/2026-06-06-dimensional-governance-completion.md)
#
# validate_config (bin/Models.py) asserts adjacent-space_role handoff consistency on
# the flattened content slab / input side, consistent with the three existing
# relaxations (passthrough / serial fold / SS reshape):
#   PS->CS : pure reshape          (ps_slab == cs_slab, pre-existing)
#   CS->SS : INPUT-side equality   (SS.nInputDim == CS.nOutputDim, NEW in C1)
#   SS->OS : flattened-slab match  (SS.nOut*SS.nOutDim == OS.nIn*OS.nInDim, NEW)
# An inconsistent handoff must RAISE at BasicModel.from_config, NOT be silently
# absorbed by a downstream reshape/pad. These tests build a DELIBERATELY
# inconsistent config and assert it raises; the reference configs must still
# build (no false-positive rejection). Each test was authored to FAIL before
# the CS->SS / SS->OS checks existed (the mismatch reached the forward reshape
# and was absorbed) and to PASS after.
# ---------------------------------------------------------------------------
import re
import pytest

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


def _build_from_text(xml_text, stem):
    """Write ``xml_text`` to a temp config under ``data/`` (so relative model.xsd
    / sibling lookups resolve) and build it. Returns the model; raises whatever
    from_config raises. Cleans up the temp file."""
    import Models, Language
    from util import init_config
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    p = os.path.join(data_dir, f"_tmp_{stem}.xml")
    with open(p, "w") as fh:
        fh.write(xml_text)
    try:
        init_config(path=p, defaults_path=os.path.join(data_dir, "model.xml"))
        Language.TheGrammar._configured = False
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m, _ = Models.BasicModel.from_config(p)
        return m
    finally:
        if os.path.exists(p):
            os.remove(p)


def _ref_text(cfg_name):
    with open(os.path.join(os.path.dirname(_BIN), "data", cfg_name)) as fh:
        return fh.read()


def test_cs_ws_input_side_handoff_mismatch_raises():
    # Break the CS->SS INPUT-side handoff: MM_20M has CS.nOutputDim=1024 and
    # SS.nInputDim=1024 (equal). Force SS.nInputDim to a value that mismatches
    # CS.nOutputDim. validate_config must FAIL LOUD (the deep CS idea no longer
    # lines up with what SS claims to consume) rather than let the forward
    # reshape silently fit it.
    src = _ref_text("MM_20M.xml")
    broken = src.replace(
        "<nInputDim>1024</nInputDim>\n    <nVectors>65536</nVectors>\n"
        "    <nDim>1024</nDim>\n    <nOutput>1024</nOutput>",
        "<nInputDim>999</nInputDim>\n    <nVectors>65536</nVectors>\n"
        "    <nDim>1024</nDim>\n    <nOutput>1024</nOutput>")
    assert broken != src, "fixture edit did not apply (SS block changed?)"
    with pytest.raises(ValueError) as ei:
        _build_from_text(broken, "cs_ws_mismatch")
    msg = str(ei.value)
    assert "CS->WS handoff" in msg, msg
    assert "999" in msg and "1024" in msg, msg


def test_ws_os_flatten_handoff_mismatch_raises():
    # Break the SS->OS FLATTEN handoff: MM_20M's SS flattened output slab is
    # nOutput(1024)*nOutputDim(8)=8192, matched by OS.nInput(8)*nInputDim(1024)
    # =8192. Force OS.nInput=7 so the OS slab (7*1024=7168) no longer equals the
    # SS flattened slab. validate_config must FAIL LOUD; the SS->OS flatten is
    # not a place to silently drop/pad a slot.
    src = _ref_text("MM_20M.xml")
    broken = src.replace("<OutputSpace>\n    <nInput>8</nInput>",
                         "<OutputSpace>\n    <nInput>7</nInput>")
    assert broken != src, "fixture edit did not apply (OutputSpace block?)"
    with pytest.raises(ValueError) as ei:
        _build_from_text(broken, "ws_os_mismatch")
    msg = str(ei.value)
    assert "WS->OS handoff" in msg, msg
    assert "8192" in msg and "7168" in msg, msg


def test_reference_configs_still_build_no_false_positive():
    # The new CS->SS / SS->OS handoff checks must NOT reject the two reference
    # configs. MM_20M exercises the LEGITIMATE deep->wide SS reshape (SS.nInputDim
    # 1024 == CS.nOutputDim, SS emits a wide [1024,8] symbol slab flattened to
    # the OS 8192). XOR_exact is the no-reshape all-14 case. Both must build.
    for cfg in ("MM_20M.xml", "XOR_exact.xml"):
        m = _build(cfg)
        assert m is not None, cfg


# ---------------------------------------------------------------------------
# Task: the reconstruction REVERSE must round-trip a DEEP-CS config.
# (doc/specs/2026-06-05-dimensional-governance.md sec.2/sec.5)
#
# MM_20M_grammar is a SERIAL deep-CS config: PartSpace event width = 12
# (content 8 + band 4), ConceptualSpace event width = 1028 (content 1024 +
# band 4). The FORWARD PS->CS handoff is the wide->deep flat-slab reshape
# (ConceptualSpace.forward: content [B,1024,8] -> [B,8,1024], band re-padded;
# the constant content slab nOutput*content = 8192 is preserved). The
# RECONSTRUCTION REVERSE (``reverse``) seeds the
# terminal deep CS and walks PS.reverse; today the PS reverseBegin fell to a
# naive ``reshape(-1, PS_width=12)`` which is invalid for a CS-width tensor
# (1028 % 12 != 0) and raised. The fix is the INVERSE wide<->deep slab-reshape
# (the exact mirror of ConceptualSpace.forward) inside Space.reverseBegin, a
# NO-OP for equal-width configs (CS width == PS width).
#
# Authored to FAIL before the inverse regroup (the crash) and PASS after.
# Untrained model -> structural assertion only (completes, finite, recovered
# width is the PS/IS width 12, NOT the CS width 1028); no reconstruction VALUE
# accuracy is asserted.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _RUN_SLOW, reason="slow (~55s serial deep-CS reverse round-trip) -- set RUN_SLOW=1")
def test_deep_cs_reverse_round_trips_to_ps_width():
    import torch, Models
    m = _build("MM_20M_grammar.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    x = m.inputSpace.prepInput(inp_items)
    m.forward(x)

    cs = m.conceptualSpace
    ps_width = int(m.perceptualSpace.muxedSize)        # 12 (content 8 + band 4)
    cs_width = int(cs.muxedSize)                        # 1028 (content 1024 + band 4)
    assert ps_width != cs_width, (ps_width, cs_width)   # genuinely deep-CS

    # The terminal deep ConceptualSpace state is the STM snapshot (the design's
    # deep hub: N = STM depth, D = 1028). ``reverse`` is
    # documented to reverse "the terminal ConceptualSpace state"; runBatch seeds
    # it identically (cs.subspace.set_event(<STM snapshot>) then reverse). The
    # STM content slab equals the input slab (8 * 1024 == 1024 * 8 == 8192), so
    # the inverse wide<->deep regroup recovers the full IS-width [B, 1024, 12].
    snap = cs.stm.snapshot()
    assert snap is not None and snap.dim() == 3 and snap.shape[-1] == cs_width

    # (a) full deep CS (the whole STM): regroups to the input-space shape.
    cs.subspace.set_event(snap)
    r = m.reverse(cs.subspace)
    assert r is not None
    ev = r.materialize()
    assert ev is not None and ev.dim() == 3
    assert torch.isfinite(ev).all(), "reconstruction must be finite (fail-loud)"
    # Recovered WIDTH must be the PS/IS width, NOT the deep CS width.
    assert ev.shape[-1] == ps_width, (tuple(ev.shape), ps_width, cs_width)
    assert ev.shape[-1] != cs_width
    # The full deep CS slab (8*1024 content) regroups to the IS position count.
    assert ev.shape[1] == int(m.inputSpace.outputShape[0]), (
        tuple(ev.shape), m.inputSpace.outputShape)

    # (b) the production single-idea seed (runBatch serial path:
    # snapshot[:, -1:, :], one deep idea): a smaller but still PS/IS-width slab.
    m.forward(x)
    cs = m.conceptualSpace
    snap = cs.stm.snapshot()
    cs.subspace.set_event(snap[:, -1:, :])
    r2 = m.reverse(cs.subspace)
    assert r2 is not None
    ev2 = r2.materialize()
    assert ev2 is not None and ev2.dim() == 3
    assert torch.isfinite(ev2).all()
    assert ev2.shape[-1] == ps_width, (tuple(ev2.shape), ps_width)
    assert ev2.shape[-1] != cs_width
