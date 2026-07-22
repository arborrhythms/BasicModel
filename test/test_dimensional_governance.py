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
    m = _build("MM_20M_legacy.xml"); Models.TheData.load("xor")
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
    m = _build("MM_20M_legacy.xml"); m.eval()
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
    # The fixture (the pre-Task-6 MM_20M_grammar shape, see _DEEP_CS_SERIAL_XML
    # below) has SS content 8 != CS content 1024, so from_config must not
    # raise the symbol_dim==concept pass-through ValueError. (MM_20M_grammar
    # itself moved to equal widths with the 2026-07-04 meronomy switch.)
    try:
        _build_from_text(_DEEP_CS_SERIAL_XML, "serial_relax")
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
# Task C1: recurrent WS input and direct CS->OS geometry fail loud.
# (doc/specs/2026-06-05-dimensional-governance.md sec.4/sec.6;
#  doc/plans/2026-06-06-dimensional-governance-completion.md)
#
# validate_config (bin/Models.py) pins the two interfaces around the peer loop:
#   CS->WS : WS's recurrent input accepts a conceptual-width event; WS's
#            native output remains an independent perceptual peer.
#   CS->OS : the output head consumes terminal CS directly, with exact event
#            count and width. WholeSpace is not an intermediate producer.
# An inconsistent interface must RAISE at BasicModel.from_config, not be
# silently absorbed by a reshape/pad.
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


def test_cs_ws_recurrent_input_mismatch_raises():
    # Break WS's recurrent conceptual input: MM_20M has CS.nOutputDim=1024 and
    # WS.nInputDim=1024. Force WS.nInputDim to mismatch while leaving its
    # native peer output alone.
    src = _ref_text("MM_20M_legacy.xml")
    broken = src.replace(
        "<nInputDim>1024</nInputDim>\n    <nVectors>65536</nVectors>\n"
        "    <nDim>1024</nDim>\n    <nOutput>1024</nOutput>",
        "<nInputDim>999</nInputDim>\n    <nVectors>65536</nVectors>\n"
        "    <nDim>1024</nDim>\n    <nOutput>1024</nOutput>")
    assert broken != src, "fixture edit did not apply (SS block changed?)"
    with pytest.raises(ValueError) as ei:
        _build_from_text(broken, "cs_ws_mismatch")
    msg = str(ei.value)
    assert "CS->WS recurrent input" in msg, msg
    assert "999" in msg and "1024" in msg, msg


def test_cs_os_direct_handoff_mismatch_raises():
    # Break the direct terminal CS->OS interface. MM_20M emits CS [8,1024]
    # and OS consumes [8,1024]. Force OS.nInput=7; exact event geometry, not a
    # coincidentally equal flattened product, is the contract.
    src = _ref_text("MM_20M_legacy.xml")
    broken = src.replace("<OutputSpace>\n    <nInput>8</nInput>",
                         "<OutputSpace>\n    <nInput>7</nInput>")
    assert broken != src, "fixture edit did not apply (OutputSpace block?)"
    with pytest.raises(ValueError) as ei:
        _build_from_text(broken, "ws_os_mismatch")
    msg = str(ei.value)
    assert "CS->OS handoff" in msg, msg
    assert "8x1024" in msg and "7x1024" in msg, msg


def test_reference_configs_still_build_no_false_positive():
    # The recurrent WS input and direct CS->OS checks must not reject either
    # reference config. MM_20M carries a deep conceptual event; XOR_exact is
    # the equal-width case.
    for cfg in ("MM_20M_legacy.xml", "XOR_exact.xml"):
        m = _build(cfg)
        assert m is not None, cfg


# ---------------------------------------------------------------------------
# Task: the reconstruction REVERSE must round-trip a DEEP-CS config.
# (doc/specs/2026-06-05-dimensional-governance.md sec.2/sec.5)
#
# The fixture is a SERIAL deep-CS shape: PartSpace event width = 12
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
#
# Fixture note (Task 6, plan 2026-07-03-reconstruction-fidelity-execution.md):
# this shape WAS data/MM_20M_grammar.xml verbatim until the 2026-07-04
# meronomy/meronomy switch moved that config to equal PS/CS widths (the
# regroup no-op case). The deep-CS premise lives on here as the test's own
# inline fixture (the pre-switch grammar space blocks, byte analysis).
# ---------------------------------------------------------------------------
_DEEP_CS_SERIAL_XML = """<?xml version="1.0" ?>
<model xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:noNamespaceSchemaLocation="model.xsd">
  <architecture>
    <symbolicOrder>1</symbolicOrder>
    <subsymbolicOrder>3</subsymbolicOrder>
    <l1Lambda>0.01</l1Lambda>
    <sigmaPi>butterfly</sigmaPi>
    <data>
      <dataType>embedding</dataType>
      <dataset>xor</dataset>
    </data>
    <training>
      <numEpochs>1</numEpochs>
      <batchSize>64</batchSize>
      <learningRate>0.0005</learningRate>
      <reconstructionScale>0.1</reconstructionScale>
    </training>
  </architecture>
  <InputSpace>
    <nInput>1024</nInput>
    <nDim>12</nDim>
    <nVectors>256</nVectors>
    <nOutput>1024</nOutput>
  </InputSpace>
  <PartSpace>
    <nInput>1024</nInput>
    <nVectors>8</nVectors>
    <nDim>12</nDim>
    <nOutput>1024</nOutput>
    <invertible>true</invertible>
  </PartSpace>
  <ConceptualSpace>
    <nInput>1024</nInput>
    <nOutput>8</nOutput>
    <nDim>1028</nDim>
    <nVectors>8</nVectors>
    <invertible>true</invertible>
    <stmCapacity>8</stmCapacity>
  </ConceptualSpace>
  <WholeSpace>
    <analysis>byte</analysis>
    <butterfly>false</butterfly>
    <nInput>8</nInput>
    <nInputDim>1028</nInputDim>
    <nDim>8</nDim>
    <nOutputDim>8</nOutputDim>
    <nVectors>1000</nVectors>
    <nOutput>8</nOutput>
    <invertible>true</invertible>
  </WholeSpace>
  <OutputSpace>
    <nInput>8</nInput>
    <nOutput>1</nOutput>
    <nDim>4</nDim>
    <nVectors>1</nVectors>
  </OutputSpace>
  <SymbolSpace>
    <language>
      <grammar>complete.grammar</grammar>
    </language>
  </SymbolSpace>
</model>
"""


@pytest.mark.skipif(not _RUN_SLOW, reason="slow (~55s serial deep-CS reverse round-trip) -- set RUN_SLOW=1")
def test_deep_cs_reverse_round_trips_to_ps_width():
    import torch, Models
    m = _build_from_text(_DEEP_CS_SERIAL_XML, "deepcs_serial")
    Models.TheData.load("xor")
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
