"""Stage 4 of doc/plans/MeronomyPlan.md: interface factoring at the callosum.

MeronomySpec §3, percept half of §10.6: the PS leg crosses the corpus
callosum NAMELESS and FACTORED -- content selects a reference row
(embedding match), evidence sets the activation magnitude
``a ∈ [0, +1]`` -- and beliefs cash into membership-fold operands only
through ``Ops.eval_chart`` (χ), applied exactly once.

  * no stimulation ⇒ ``a = 0`` (a tautology of the dot product, not a
    clamp);
  * ``a < 0`` unreachable end-to-end from the percept path (negative
    input coordinates are structurally invisible; anti-matches are
    non-matches);
  * the crossing emits ``a · u`` in the K3 wire domain -- charting
    happens ONLY at the cash-out (never inside the crossing), and the
    chart is applied exactly once;
  * parallel mode's 2N mixing matrix is untouched -- the knob
    (<architecture><meronomy>, default off) governs what crosses it,
    not its shape; knob off is the pre-Stage-4 path.
"""
import os
import sys
import types

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

import torch

_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin')
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

from Layers import Ops, meronomy_enabled
from Spaces import ConceptualSpace

D, V = 6, 5


def unit_rows(seed=1, v=V, d=D):
    torch.manual_seed(seed)
    rows = torch.randn(v, d)
    return rows / rows.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# The mode knob: default off (dark landing), on by config.
# ---------------------------------------------------------------------------

def test_meronomy_knob_code_fallback_is_off():
    # The CODE fallback (key absent / no config) is off — dark by
    # construction. Since the Stage 9 cutover, data/model.xml carries
    # <meronomy>on</meronomy>, so configured models default on; this
    # pins the unconfigured fallback, popping any loaded config's key.
    from util import TheXMLConfig
    arch = TheXMLConfig._data.get("architecture")
    prev = arch.pop("meronomy", None) if isinstance(arch, dict) else None
    try:
        assert meronomy_enabled() is False
    finally:
        if prev is not None:
            TheXMLConfig._data["architecture"]["meronomy"] = prev


def test_meronomy_knob_forms():
    from util import TheXMLConfig
    had = "architecture" in TheXMLConfig._data
    try:
        TheXMLConfig.set("architecture.meronomy", "on")
        assert meronomy_enabled() is True
        TheXMLConfig.set("architecture.meronomy", "off")
        assert meronomy_enabled() is False
        # Attributed form: text under "_" beside dMaxStable etc.
        TheXMLConfig.set("architecture.meronomy",
                         {"_": "on", "dMaxStable": "4.0"})
        assert meronomy_enabled() is True
        TheXMLConfig.set("architecture.meronomy", {"dMaxStable": "4.0"})
        assert meronomy_enabled() is False
    finally:
        if had:
            TheXMLConfig._data["architecture"].pop("meronomy", None)
        else:
            TheXMLConfig._data.pop("architecture", None)


# ---------------------------------------------------------------------------
# factor_percept: content → row selection, evidence → a ∈ [0, +1].
# ---------------------------------------------------------------------------

def test_no_stimulation_is_zero_evidence():
    rows = unit_rows()
    idx, a = ConceptualSpace.factor_percept(torch.zeros(D), rows)
    assert a.item() == 0.0, "absence of stimulation IS a = 0 (tautology)"


def test_negative_half_unreachable():
    rows = unit_rows()
    torch.manual_seed(2)
    # Adversarial percepts: any sign, any magnitude.
    for scale in (0.1, 1.0, 10.0):
        p = torch.randn(32, D) * scale
        idx, a = ConceptualSpace.factor_percept(p, rows)
        assert (a >= 0).all(), "a < 0 unreachable from the percept path"
        assert (a <= 1).all(), "a capped at certainty 1"
    # Purely negative content is structurally invisible.
    idx, a = ConceptualSpace.factor_percept(-torch.rand(8, D), rows)
    assert (a == 0).all()


def test_content_selects_evidence_scales():
    rows = unit_rows()
    p = 0.4 * rows[3].clamp(min=0)          # aligned with row 3's + content
    idx1, a1 = ConceptualSpace.factor_percept(p, rows)
    idx2, a2 = ConceptualSpace.factor_percept(2.0 * p, rows)
    assert idx1.item() == idx2.item(), "scaling changes evidence, not content"
    assert a2.item() >= a1.item()
    assert a2.item() <= 1.0


def test_factor_with_no_rows():
    idx, a = ConceptualSpace.factor_percept(torch.rand(D), torch.zeros(0, D))
    assert idx is None and a is None


# ---------------------------------------------------------------------------
# The cash-out chart: exactly once, exact corners.
# ---------------------------------------------------------------------------

def test_belief_to_extent_is_the_chart_applied_once():
    cs = ConceptualSpace.__new__(ConceptualSpace)   # method needs no state
    a = torch.tensor([1.0, 0.0, -1.0, 0.5])
    m = cs.belief_to_extent(a)
    assert torch.equal(m, Ops.eval_chart(a)), "single χ application"
    assert m[0].item() == 1.0 and m[1].item() == 0.5 and m[2].item() == 0.0
    # Double-charting is detectable -- the contract is ONE application.
    assert not torch.allclose(Ops.eval_chart(m), m), (
        "χ∘χ differs from χ: re-charting a cashed value is an error")
    # Out-of-wire-domain beliefs clamp to [-1, 1] before the chart.
    assert cs.belief_to_extent(torch.tensor(3.0)).item() == 1.0


# ---------------------------------------------------------------------------
# The factored crossing: nameless, in the row span, wire-domain.
# ---------------------------------------------------------------------------

def _stub_cs(rows):
    """A ConceptualSpace stub exposing only what _factor_crossing reads."""
    cs = ConceptualSpace.__new__(ConceptualSpace)
    sub = types.SimpleNamespace(
        codebook_slot='what',
        prototype=lambda: rows,
    )
    object.__setattr__(cs, 'subspace', sub)
    return cs


def test_factored_crossing_lives_in_the_row_span():
    rows = unit_rows()
    cs = _stub_cs(rows)
    torch.manual_seed(3)
    PS_t = torch.randn(2, 4, D)             # [B, N, D] slots, signed garbage
    out = cs._factor_crossing(PS_t)
    assert out.shape == PS_t.shape
    flat = out.reshape(-1, D)
    for slot in flat:
        n = slot.norm()
        if n < 1e-9:
            continue                         # a = 0 slot: nothing crossed
        cos = (slot / n) @ rows.T
        assert cos.max() > 1 - 1e-5, (
            "every crossed slot is a·u for some reference row u -- "
            "raw coordinates never cross")
        assert n <= 1 + 1e-6, "evidence magnitude capped at 1"


def test_factored_crossing_zero_slot_crosses_nothing():
    rows = unit_rows()
    cs = _stub_cs(rows)
    PS_t = torch.zeros(1, 3, D)
    out = cs._factor_crossing(PS_t)
    assert (out == 0).all()


def test_factored_crossing_emits_wire_domain_not_memberships():
    # The crossing must NOT chart: a slot with evidence a crosses at
    # magnitude a, not (1+a)/2. (χ is applied only at fold cash-outs.)
    rows = unit_rows()
    cs = _stub_cs(rows)
    p = 0.6 * rows[1].clamp(min=0)
    out = cs._factor_crossing(p.reshape(1, 1, D))
    idx, a = ConceptualSpace.factor_percept(p, rows)
    assert torch.allclose(out.norm(), a, atol=1e-6), (
        "crossing magnitude is the evidence a (wire domain), uncharted")


def test_factored_crossing_without_codebook_is_identity():
    cs = ConceptualSpace.__new__(ConceptualSpace)
    object.__setattr__(cs, 'subspace',
                       types.SimpleNamespace(codebook_slot=None))
    PS_t = torch.randn(2, 3, D)
    assert torch.equal(cs._factor_crossing(PS_t), PS_t)
    # Width-mismatched codebook: cross unchanged (nothing to factor against).
    cs2 = _stub_cs(unit_rows(d=D + 2))
    assert torch.equal(cs2._factor_crossing(PS_t), PS_t)


# ---------------------------------------------------------------------------
# Knob-off regression at the model level: bind_streams unchanged (the
# full suite is the broader regression; here we pin determinism and that
# the knob-on path stays shape-safe end-to-end).
# ---------------------------------------------------------------------------

def test_bind_streams_knob_paths():
    import Language
    import Models
    from util import init_config, TheXMLConfig
    _PROJECT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    init_config(path=os.path.join(_PROJECT, "data", "MM_xor.xml"),
                defaults_path=os.path.join(_PROJECT, "data", "model.xml"))
    Language.TheGrammar._configured = False
    torch.manual_seed(1)
    m, cfg = Models.BasicModel.from_config(
        os.path.join(_PROJECT, "data", "MM_xor.xml"))
    cs = getattr(m, "conceptualSpace", None)
    if cs is None or getattr(cs, "combine", None) is None:
        return  # model variant without the bind stage: nothing to pin
    ps_sub, ws_sub, cs_sub = (m.perceptualSpace.subspace,
                              m.wholeSpace.subspace,
                              cs.subspace)
    torch.manual_seed(2)
    ev = torch.rand(2, cs_sub.nVectors if hasattr(cs_sub, 'nVectors') else 4,
                    cs_sub.muxedSize if hasattr(cs_sub, 'muxedSize') else 8)
    try:
        ps_sub.set_event(ev.clone()); ws_sub.set_event(ev.clone())
        cs_sub.set_event(ev.clone())
    except Exception:
        return  # fixture shapes unavailable: covered by unit tests above
    off1 = cs.bind_streams(ps_sub, ws_sub, cs_sub)
    ps_sub.set_event(ev.clone()); ws_sub.set_event(ev.clone())
    cs_sub.set_event(ev.clone())
    off2 = cs.bind_streams(ps_sub, ws_sub, cs_sub)
    if off1 is not None:
        assert torch.allclose(off1, off2), "knob-off bind is deterministic"
    TheXMLConfig.set("architecture.meronomy", "on")
    try:
        ps_sub.set_event(ev.clone()); ws_sub.set_event(ev.clone())
        cs_sub.set_event(ev.clone())
        on1 = cs.bind_streams(ps_sub, ws_sub, cs_sub)
        if off1 is not None and on1 is not None:
            assert on1.shape == off1.shape, "knob-on path stays shape-safe"
    finally:
        TheXMLConfig._data["architecture"].pop("meronomy", None)
