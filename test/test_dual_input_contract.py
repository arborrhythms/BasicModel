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
    from util import init_config, init_device
    # Run-time CPU pin: the env setdefault above only wins when this
    # module is imported before util (solo runs). In-suite, the default
    # device is whatever the process carries -- on real MPS the
    # painting-reverse F.interpolate hard-aborts the interpreter -- and
    # the conftest device guard unwinds this pin after the module.
    init_device("cpu")
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


@pytest.mark.slow
def test_inputspace_forward_returns_dual_view():
    # Phase 1 contract: InputSpace.forward emits BOTH views of one source.
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
    # Atom view: the lexed SubSpace; its content channel is [B, N, 1]
    # (N native identity codes; the where/when band is coordinate
    # metadata, not content).
    assert hasattr(percepts_in, "materialize"), (
        "percepts_in must be the atom-stream SubSpace")
    what = percepts_in.materialize(mode="what")
    assert what is not None and what.dim() == 3 and what.shape[-1] == m.perceptualSpace.percept_store.identity.width, (
        "atom view content channel must use the native identity width, got "
        f"{None if what is None else tuple(what.shape)}")
    # Unity view: ONE width-N event -- the DELIVERED raw surface itself
    # (IS is not the lexer, 2026-07-12: bytes to PS, one whole to WS).
    assert torch.is_tensor(concepts_in) and concepts_in.dim() == 3, (
        f"concepts_in must be a [B, 1, N] tensor, got {type(concepts_in)!r}")
    B, one, N = concepts_in.shape
    assert one == 1, (
        f"unity view must be a SINGLE event, got {tuple(concepts_in.shape)}")
    raw = getattr(percepts_in, "_raw_surface", None)
    assert torch.is_tensor(raw) and raw.dim() == 2, (
        "the atom view must carry the delivered raw surface for PS")
    assert B == raw.shape[0] and N == raw.shape[1], (
        "the unity must be the SAME presentation as the delivered bytes "
        f"(raw {tuple(raw.shape)}, unity {tuple(concepts_in.shape)})")


@pytest.mark.slow
def test_dual_views_share_values():
    # The two views are views of ONE presentation: same values, different
    # shape. (Analysis is non-altering; the unity view is the byte content
    # verbatim, not a transform of it.)
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, concepts_in = m.inputSpace.forward(x)
        what = percepts_in.materialize(mode="what")
    raw = getattr(percepts_in, "_raw_surface", None)
    assert torch.is_tensor(raw), "the delivered raw surface must be staged"
    assert torch.equal(
        concepts_in.squeeze(1).to(torch.float32),
        raw.to(torch.float32)), (
        "unity-view values must equal the DELIVERED bytes verbatim "
        "(one presentation, two views)")


@pytest.mark.slow
def test_ws_stage0_consumes_unity():
    # Phase 2 contract: with an EMPTY recurrent CS (stage 0), a provided
    # unity drives the symbolic pass -- coarse region-mean evidence in the
    # standard SS output geometry. Different unities produce different
    # stage-0 symbols; the unity buffer itself is NEVER altered (analysis
    # is non-altering).
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpace
    seed = m._empty_seed_ss
    # None == legacy path: empty in, empty out (no evidence, no symbols).
    out_none = ws.forward(None, cs_out=seed)
    assert out_none is seed and out_none.is_empty()
    u1 = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    u1_snapshot = u1.clone()
    out1 = ws.forward(u1, cs_out=seed)
    ev1 = out1.materialize()
    assert ev1 is not None and torch.isfinite(ev1).all()
    assert ev1.shape == (2, int(ws.inputShape[0]), int(ws.subspace.muxedSize)), (
        f"stage-0 evidence must land in the CS-aligned event geometry "
        f"(one narrow symbol event per concept slot), got {tuple(ev1.shape)}")
    # Property evidence is read from the canonical property basis; this
    # configuration may hold WS at zero. There is no byte-mean symbol snap.
    expected, membership = ws.compute_stage0_unity_event(u1)
    torch.testing.assert_close(ev1, expected, rtol=0, atol=0)
    assert torch.equal(u1, u1_snapshot), (
        "analysis is NON-ALTERING: the unity buffer must be untouched")
    ev1 = ev1.detach().clone()
    u2 = (u1 + 64) % 256
    ev2 = ws.forward(u2, cs_out=seed).materialize()
    expected2, _ = ws.compute_stage0_unity_event(u2)
    torch.testing.assert_close(ev2, expected2, rtol=0, atol=0)
    assert ws._stage0_indices is None


@pytest.mark.slow
def test_ws_routing_law_typed():
    # Serial migration (2026-07-11): ONE typed law -- a raw unity tensor
    # routes universe-primary even alongside a live carrier; passing the
    # carrier first (or in_sub=None) routes the carrier body.
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpace
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    ws.forward(u, cs_out=m._empty_seed_ss)   # populates ws.subspace
    assert not ws.subspace.is_empty()
    ws.forward(u, cs_out=ws.subspace)        # raw unity wins: universe
    assert getattr(ws, "_ws_routed_source", None) == "universe"
    ws.forward(ws.subspace)                  # carrier-first: body path
    assert getattr(ws, "_ws_routed_source", None) == "property-neutral"


@pytest.mark.slow
def test_model_forward_passes_unity_at_stage0():
    # Phase 2 wiring: the body hands the PARKED unity to the STAGE-0
    # WholeSpace only; later stages read the recurrent CS (input once).
    # The per-stage WholeSpaces are DISTINCT objects (``m.wholeSpaces[t]``,
    # ``body_stages[t]["ws"]``); the terminal ``m.wholeSpace`` is the LAST
    # stage, so at bindingDepth>1 (MM_20M ships sO=3, T=3) the capture
    # must hook EVERY stage's ws -- hooking only the terminal would miss the
    # stage-0 unity call entirely.
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    stage_ws = list(m.wholeSpaces)
    reals = [w.forward for w in stage_ws]
    calls = []  # (stage_idx, offered_unity, routed_source)
    def _mk(idx, real, w):
        def _capture(in_sub, *a, **k):
            out = real(in_sub, *a, **k)
            calls.append((idx, in_sub,
                          getattr(w, "_ws_routed_source", None)))
            return out
        return _capture
    for i, w in enumerate(stage_ws):
        w.forward = _mk(i, reals[i], w)
    try:
        with torch.no_grad():
            out = m.forward(x)[2]
    finally:
        for w, real in zip(stage_ws, reals):
            w.forward = real
    assert out is not None and torch.isfinite(out).all()
    # Serial migration (2026-07-11): stage 0 bootstraps with the unity
    # (universe stamp); t>0 carriers supply a neutral property field.
    stage0 = [(u, src) for (i, u, src) in calls if i == 0]
    later = [(u, src) for (i, u, src) in calls if i > 0]
    assert stage0 and torch.is_tensor(stage0[0][0]), (
        "stage 0 must be offered the parked unity view")
    assert all(src == "universe" for (_u, src) in stage0), (
        "stage 0 bootstraps from the universe")
    assert all(src == "property-neutral" for (_u, src) in later), (
        "t>0 carriers supply a neutral property field")


@pytest.mark.slow
def test_full_forward_green_with_dual_view():
    # The orchestration shim threads the tuple; the model forward is intact
    # and the unity view is parked for Phase 2 (staged, unused).
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        out = m.forward(x)[2]
    assert out is not None and torch.isfinite(out).all()
    staged = getattr(m, "_staged_concepts_in", None)
    assert staged is not None and staged.dim() == 3 and staged.shape[1] == 1, (
        "the forward must park the unity view on _staged_concepts_in "
        "(Phase 1: staged, unused; Phase 2 consumes it at SS stage 0)")




@pytest.mark.slow
def test_parallel_ws_quantize_fires():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpaces[0]
    basis = ws.subspace.what
    before = basis.getW().detach().clone()
    with torch.no_grad():
        out = m.forward(_staged_batch(m))[2]
    assert out is not None and torch.isfinite(out).all()
    assert not hasattr(ws, 'analysis_store')
    assert ws._stage0_indices is None
    torch.testing.assert_close(basis.getW(), before, rtol=0, atol=0)


@pytest.mark.slow
def test_ws_vq_asymmetric_flags():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    before = [ws.subspace.what.getW().detach().clone() for ws in m.wholeSpaces]
    m.train()
    try:
        m.forward(x)
        m.forward(x)
    finally:
        m.eval()
    for ws, rows in zip(m.wholeSpaces, before):
        assert not hasattr(ws, 'analysis_store')
        torch.testing.assert_close(ws.subspace.what.getW(), rows, rtol=0, atol=0)
        assert not any('analysis_store' in key for key in ws.state_dict())


@pytest.mark.slow
def test_ws_codebook_recon_gradient():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpace
    ws.train()
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    try:
        ws.forward(u, cs_out=m._empty_seed_ss)
    finally:
        ws.eval()
    assert not hasattr(ws, 'analysis_store')
    assert ws._stage0_recon_loss is None
    assert not any('analysis_store' in key for key, _ in ws.named_parameters())


@pytest.mark.slow
def test_ws_recon_term_reaches_pipeline_errors():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    m.train()
    try:
        m.forward(_staged_batch(m))
    finally:
        m.eval()
    terms = getattr(m.outputSpace.subspace.errors, '_terms', {})
    assert 'ws_codebook_recon' not in terms
    assert all(getattr(ws, '_stage0_recon_loss', None) is None for ws in m.wholeSpaces)


@pytest.mark.slow
def test_descriptor_roles_lf_coarse_tagging():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpace
    basis = ws.subspace.what
    before = basis.getW().detach().clone()
    ws.forward(torch.randint(0, 256, (2, 1, 512), dtype=torch.int64),
               cs_out=m._empty_seed_ss)
    assert not hasattr(ws, 'analysis_store')
    assert ws._stage0_indices is None
    assert basis.nVectors == m.nWholeProperties
    torch.testing.assert_close(basis.getW(), before, rtol=0, atol=0)


@pytest.mark.slow
def test_semantic_arrangement_mechanism():
    """Retirement certificate: WholeSpace owns properties, not a word dictionary."""
    m = _build("MM_20M_xor.xml")
    ws = m.wholeSpace
    assert not hasattr(ws, 'semantic_arrangement_weight')
    ws.train()
    try:
        ws.forward(torch.randint(0, 256, (2, 1, 512), dtype=torch.int64),
                   cs_out=m._empty_seed_ss)
    finally:
        ws.eval()
    assert ws._stage0_semantic_loss is None
    assert not hasattr(ws, 'analysis_store')


@pytest.mark.slow
def test_painting_reverse_blend():
    # Phase 7 contract (plan sec.6, rev. 2026-06-10): the reconstruction
    # recombination is PAINTING -- the Universal view paints the
    # background (each slot's coarse value over its contiguous region),
    # the Atomic view is AVERAGED in where it has support, and the
    # background remains alone where it doesn't (padding painted, not
    # halved). The concepts branch RIDES the SubSpace
    # (``_concepts_recon``) -- reverse stays single-arg per the
    # processing contract.
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        percepts_in, _ = m.inputSpace.forward(x)
    # Fabricate a small atomic reconstruction event: a few byte-valued
    # atoms up front, padding (zeros, no support) behind.
    B, T, Dv = 2, 32, 5
    atomic = torch.zeros(B, T, Dv)
    atomic[:, :5, 0] = torch.tensor([104.0, 105.0, 32.0, 111.0, 120.0])
    percepts_in.set_event(atomic.clone())
    # A flat universal background: every slot paints the value 128
    # (slot mean over W channels of 128s).
    K, W = 4, 16
    concepts = torch.full((B, K, W), 128.0)
    object.__setattr__(percepts_in, "_concepts_recon", concepts)
    with torch.no_grad():
        out = m.inputSpace.reverse(percepts_in)
        ev = out.materialize()
        painted = getattr(out, "_painted_event", None)
    # The EVENT stays the atomic reverse (the exact word/byte decode reads
    # it); the painted COMBINED surface rides the SubSpace.
    assert ev is not None and torch.allclose(ev, atomic, atol=1e-5), (
        "the event must remain the ATOMIC reverse (exactness)")
    assert painted is not None and painted.shape == atomic.shape, (
        "the painted surface must ride the SubSpace (_painted_event)")
    support = atomic.abs().sum(dim=-1, keepdim=True) > 0
    expected = torch.where(
        support, 0.5 * (atomic + 128.0),
        torch.full_like(atomic, 128.0))
    assert torch.allclose(painted, expected, atol=1e-4), (
        "painting must average atoms into the background at support and "
        "leave the background alone elsewhere")


@pytest.mark.slow
def test_model_reverse_threads_concepts_branch():
    # Phase 7 wiring: the stage-0 SS stream of the bind's exact inverse is
    # the conceptual reconstruction branch; _reverse_body stamps it onto
    # the returned sub (SubSpace-carried), and the model reverse() carries
    # it across the PS handoff into InputSpace.reverse. End-to-end reverse
    # stays finite.
    m = _build("MM_20M_xor.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        m.forward(x)
        body_sub = m._reverse_body(m._combine_last_cs_sub)
    assert getattr(body_sub, "_concepts_recon", None) is None, (
        "the property path has no independently snapped word reconstruction")
    with torch.no_grad():
        m.forward(x)  # re-bind (the first reverse consumed the state)
        recon = m.reverse(m._combine_last_cs_sub)
        ev = recon.materialize() if recon is not None else None
    assert ev is not None and torch.isfinite(ev).all(), (
        "the two-branch painting reverse must stay finite end-to-end")
    painted = getattr(m.inputSpace.subspace, "_painted_event", None)
    assert painted is None, "no separate background is synthesized by the property path"
