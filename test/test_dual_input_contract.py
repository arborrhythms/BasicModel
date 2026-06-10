"""Phase 0 contract tests -- analysis/synthesis dual-input plan
(doc/plans/2026-06-08-analysis-synthesis-dual-input.md, rev. 2026-06-09).

Orientation under test (the CORRECTED one):

  percepts_in -- the ATOM view  -> PS (bottom-up synthesis, Sigma/union)
  concepts_in -- the UNITY view -> SS (top-down analysis, Pi/intersection)

  InputSpace.forward(input) -> (percepts_in, concepts_in)
    percepts_in: the atom-stream SubSpace; content channel [B, N, 1]
    concepts_in: ONE width-N event [B, 1, N] (same values, unity view)

Knob-acceptance contracts (<synthesis> on PS, <analysis>/<lexer> on SS) land
with Phase 4, where the schema changes; they are deliberately not stubbed
here.
"""
import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path: sys.path.insert(0, _BIN)

import pytest
import torch


def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", name)
    init_config(path=p, defaults_path=os.path.join(
        os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def _staged_batch(m):
    import Models
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader))
    return m.inputSpace.prepInput(items)


def test_inputspace_forward_returns_dual_view():
    # Phase 1 contract: InputSpace.forward emits BOTH views of one source.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
    # Atom view: the lexed SubSpace; its content channel is [B, N, 1]
    # (N atoms of width-1 byte content; the where/when band is coordinate
    # metadata, not content).
    assert hasattr(percepts_in, "materialize"), (
        "percepts_in must be the atom-stream SubSpace")
    what = percepts_in.materialize(mode="what")
    assert what is not None and what.dim() == 3 and what.shape[-1] == 1, (
        "atom view content channel must be [B, N, 1], got "
        f"{None if what is None else tuple(what.shape)}")
    # Unity view: ONE width-N event.
    assert torch.is_tensor(concepts_in) and concepts_in.dim() == 3, (
        f"concepts_in must be a [B, 1, N] tensor, got {type(concepts_in)!r}")
    B, one, N = concepts_in.shape
    assert one == 1, (
        f"unity view must be a SINGLE event, got {tuple(concepts_in.shape)}")
    assert B == what.shape[0] and N == what.shape[1], (
        "the two views must present the SAME B and N "
        f"(atoms {tuple(what.shape)}, unity {tuple(concepts_in.shape)})")


def test_dual_views_share_values():
    # The two views are views of ONE presentation: same values, different
    # shape. (Analysis is non-altering; the unity view is the byte content
    # verbatim, not a transform of it.)
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
        what = percepts_in.materialize(mode="what")
    assert torch.equal(
        concepts_in.squeeze(1).to(torch.float32),
        what[..., 0].to(torch.float32)), (
        "unity-view values must equal the atom-view content channel")


def test_ss_stage0_consumes_unity():
    # Phase 2 contract: with an EMPTY recurrent CS (stage 0), a provided
    # unity drives the symbolic pass -- coarse region-mean evidence in the
    # standard SS output geometry. Different unities produce different
    # stage-0 symbols; the unity buffer itself is NEVER altered (analysis
    # is non-altering).
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    seed = m._empty_seed_ss
    # None == legacy path: empty in, empty out (no evidence, no symbols).
    out_none = ss.forward(seed, IS_concepts=None)
    assert out_none is seed and out_none.is_empty()
    u1 = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    u1_snapshot = u1.clone()
    out1 = ss.forward(seed, IS_concepts=u1)
    ev1 = out1.materialize()
    assert ev1 is not None and torch.isfinite(ev1).all()
    assert ev1.shape == (2, int(ss.inputShape[0]), int(ss.subspace.muxedSize)), (
        f"stage-0 evidence must land in the CS-aligned event geometry "
        f"(one narrow symbol event per concept slot), got {tuple(ev1.shape)}")
    assert float(ev1.abs().max()) > 0, "evidence must be non-trivial symbols"
    assert torch.equal(u1, u1_snapshot), (
        "analysis is NON-ALTERING: the unity buffer must be untouched")
    ev1 = ev1.detach().clone()
    u2 = (u1 + 64) % 256
    ev2 = ss.forward(seed, IS_concepts=u2).materialize()
    assert not torch.equal(ev1, ev2), (
        "stage-0 symbolic output must CHANGE when the unity changes")


def test_ss_rejects_concepts_with_nonempty_cs():
    # Phase 2 contract: input is read ONCE at stage 0. Supplying IS_concepts
    # alongside a live recurrent CS is the later repeated-injection knob and
    # must fail loudly (never a silent drop).
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    ss.forward(m._empty_seed_ss, IS_concepts=u)   # populates ss.subspace
    assert not ss.subspace.is_empty()
    with pytest.raises(NotImplementedError):
        ss.forward(ss.subspace, IS_concepts=u)


def test_model_forward_passes_unity_at_stage0():
    # Phase 2 wiring: the body hands the PARKED unity to SS at stage 0 only.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    ss = m.symbolicSpace
    real = ss.forward
    calls = []
    def _capture(sub, *a, **k):
        calls.append(k.get("IS_concepts", None))
        return real(sub, *a, **k)
    ss.forward = _capture
    try:
        with torch.no_grad():
            out = m.forward(x)[2]
    finally:
        ss.forward = real
    assert out is not None and torch.isfinite(out).all()
    assert len(calls) >= 1 and torch.is_tensor(calls[0]), (
        "stage 0 must receive the parked unity view")
    assert all(c is None for c in calls[1:]), (
        "stages after 0 read only the recurrent CS (input once)")


def test_full_forward_green_with_dual_view():
    # The orchestration shim threads the tuple; the model forward is intact
    # and the unity view is parked for Phase 2 (staged, unused).
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        out = m.forward(x)[2]
    assert out is not None and torch.isfinite(out).all()
    staged = getattr(m, "_staged_concepts_in", None)
    assert staged is not None and staged.dim() == 3 and staged.shape[1] == 1, (
        "the forward must park the unity view on _staged_concepts_in "
        "(Phase 1: staged, unused; Phase 2 consumes it at SS stage 0)")


def test_word_analysis_boundaries_shape_evidence():
    # Phase 4b contract: BOUNDARIES SHAPE THE EVIDENCE. With
    # <analysis>word, the whitespace-cut parts define the regions whose
    # coarse means become the stage-0 symbolic evidence -- replacing the
    # uniform-region pooling that remains the byte-mode default.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    # "hi ox" as byte codes, padded with the null sentinel.
    text = b"hi ox"
    u = torch.zeros(2, 1, 32, dtype=torch.int64)
    u[:, 0, :len(text)] = torch.tensor(list(text), dtype=torch.int64)
    u_snapshot = u.clone()
    # byte mode (default): uniform pooling.
    ss.analysis_mode = "byte"
    assert ss.stage_analysis_spans(u) is None
    ss._staged_analysis_spans = None
    ev_byte = ss.forward(m._empty_seed_ss, IS_concepts=u).materialize().clone()
    # word mode: parts are the whitespace-cut spans.
    ss.analysis_mode = "word"
    spans = ss.stage_analysis_spans(u)
    assert spans is not None and spans.shape == (2, 2, 2), (
        f"'hi ox' must cut into TWO parts per row, got "
        f"{None if spans is None else tuple(spans.shape)}")
    assert spans[0].tolist() == [[0, 2], [3, 5]]
    assert torch.equal(u, u_snapshot), "analysis must not alter the unity"
    ss._staged_analysis_spans = spans
    try:
        ev_word = ss.forward(
            m._empty_seed_ss, IS_concepts=u).materialize().clone()
    finally:
        ss._staged_analysis_spans = None
        ss.analysis_mode = "byte"
    assert not torch.equal(ev_byte, ev_word), (
        "word-cut evidence must differ from uniform-region evidence")
    # Cell 0 carries part 0's coarse mean: tanh(mean(b'hi') / 128).
    import math
    expected = math.tanh((ord("h") + ord("i")) / 2.0 / 128.0)
    got = float(ev_word[0, 0, 0])
    assert abs(got - expected) < 1e-5, (
        f"part-0 mean evidence: expected {expected:.6f}, got {got:.6f}")
    # Cells beyond the part count stay neutral (0), like null padding.
    assert float(ev_word[0, 1:, :].abs().max()) == 0.0


@pytest.mark.xfail(
    reason="Phase 7 (reconstruction recombination) not landed", strict=False)
def test_inputspace_reverse_two_branch():
    # Phase 7 contract: reverse recombines BOTH branches.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
        recon = m.inputSpace.reverse(percepts_in, concepts_in)
    assert recon is not None
