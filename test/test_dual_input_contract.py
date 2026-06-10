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
    # <analysis>word, the whitespace-cut parts define the PARTS whose
    # coarse means become the stage-0 symbolic evidence (part k ->
    # symbol slot k) -- replacing the uniform-region pooling that
    # remains the byte-mode default. The hand-checked means are
    # asserted on the threaded PRE-SNAP carrier (the evidence z_e
    # before the live SS codebook snap).
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
        z_pre = ss._stage0_z_pre_snap.detach().clone()
    finally:
        ss._staged_analysis_spans = None
        ss.analysis_mode = "byte"
    assert not torch.equal(ev_byte, ev_word), (
        "word-cut evidence must differ from uniform-region evidence")
    # Slot k carries part k's coarse mean on the pre-snap carrier:
    # tanh(mean(part bytes) / 128), broadcast across the carrier width.
    import math
    exp0 = math.tanh((ord("h") + ord("i")) / 2.0 / 128.0)
    exp1 = math.tanh((ord("o") + ord("x")) / 2.0 / 128.0)
    assert abs(float(z_pre[0, 0, 0]) - exp0) < 1e-5, (
        f"part-0 mean: expected {exp0:.6f}, got {float(z_pre[0, 0, 0]):.6f}")
    assert abs(float(z_pre[0, 1, 0]) - exp1) < 1e-5, (
        f"part-1 mean: expected {exp1:.6f}, got {float(z_pre[0, 1, 0]):.6f}")
    # Slots beyond the part count stay neutral (0), like null padding.
    assert float(z_pre[0, 2:, :].abs().max()) == 0.0


def test_parallel_ss_quantize_fires():
    # Plan-1 Task 1 acceptance (asymmetric-vq sec.7 task 8, DECISION
    # 2026-06-09): the SS codebook is LIVE in the parallel path --
    # Codebook.quantize() genuinely fires during a parallel forward
    # (it was a verified 0-call no-op before).
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    ss = m.symbolicSpace
    basis = ss.subspace.what
    real_q = basis.quantize
    calls = {"n": 0}
    def _counting(*a, **k):
        calls["n"] += 1
        return real_q(*a, **k)
    basis.quantize = _counting
    try:
        with torch.no_grad():
            out = m.forward(x)[2]
    finally:
        basis.quantize = real_q
    assert out is not None and torch.isfinite(out).all()
    assert calls["n"] >= 1, (
        "the SS codebook must QUANTIZE during a parallel forward "
        "(asymmetric-vq task 8); it never fired")
    assert ss._stage0_indices is not None, (
        "the snap must thread the selected indices "
        "(the future recon-gather leg reads them)")


def test_ss_vq_asymmetric_flags():
    # Plan-1 Task 2 (C-11/C-12): the SS VQ drops the standard crutches --
    # commitment weight 0 (STE carries the output->encoder leg) and the
    # in-forward EMA codebook update OFF (the input->codebook leg is the
    # recon gather). The FIRST training forward may legitimately mutate
    # the codebook ONCE: adopt-on-first-sight (Phase 5) data-initialises
    # VIRGIN rows from the stage-0 evidence in the eager stem. After
    # adoption has named the rows, further training forwards must be
    # bit-stable (no EMA, no drift).
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    ss = m.symbolicSpace
    vq = getattr(ss.subspace.what, "vq", None)
    assert vq is not None, "MM_20M SS must carry a customVQ codebook"
    assert vq.ema_update is False, "SS VQ EMA must be OFF (asymmetric C-11)"
    assert float(vq.commitment_weight) == 0.0, (
        "SS VQ commitment must be 0 (replaced by STE, asymmetric C-11)")
    m.train()
    try:
        m.forward(x)  # adoption + in-body role naming settle here
        before = vq.codebook.detach().clone()
        m.forward(x)
    finally:
        m.eval()
    after = vq.codebook.detach()
    assert torch.equal(before, after), (
        "with EMA off and adoption settled, a training-mode forward must "
        "NOT mutate the SS codebook in-forward (it trains only via the "
        "recon gradient)")


def test_ss_codebook_recon_gradient():
    # The asymmetric RECON leg (input -> codebook, asymmetric-vq sec.4):
    # the stage-0 snap emits an exact-gather reconstruction term whose
    # gradient lands on the SELECTED codebook rows only (the evidence is
    # detached; the argmin blocks the encoder leg). This is the EMA
    # replacement -- exact, not a running average.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    vq = ss.subspace.what.vq
    assert isinstance(vq.codebook, torch.nn.Parameter), (
        "the SS VQ codebook must be an nn.Parameter (it trains by the "
        "recon gradient)")
    ss.train()
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    try:
        ss.forward(m._empty_seed_ss, IS_concepts=u)
        recon = ss._stage0_recon_loss
    finally:
        ss.eval()
    assert recon is not None and recon.requires_grad, (
        "training-mode stage 0 must thread the recon term")
    if vq.codebook.grad is not None:
        vq.codebook.grad = None
    recon.backward()
    g = vq.codebook.grad
    assert g is not None and float(g.abs().sum()) > 0, (
        "the recon gradient must land on the codebook")
    # Plain gather: the gradient support is EXACTLY the selected rows.
    idx = ss._stage0_indices.reshape(-1)
    sel = torch.zeros(g.shape[0], dtype=torch.bool)
    sel[idx] = True
    assert float(g[~sel].abs().max()) == 0.0, (
        "gradient must touch ONLY the selected rows (plain gather; the "
        "argmin blocks every other path)")


def test_ss_recon_term_reaches_pipeline_errors():
    # The stage-0 recon term is threaded as an SS forward-local and lifted
    # onto the pipeline-chained error container by _forward_body, so the
    # training loss actually consumes it.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    m.train()
    try:
        m.forward(x)
    finally:
        m.eval()
    terms = getattr(m.outputSpace.subspace.errors, "_terms", {})
    assert "ss_codebook_recon" in terms, (
        f"the recon term must reach the pipeline error container; "
        f"got terms={list(terms)}")


def test_descriptor_roles_lf_coarse_tagging():
    # Phase 6 contract (plan sec.7): the SS generality codebook carries
    # per-row DESCRIPTOR ROLES (meaning-/term-general, LF-coarse) as
    # metadata in ONE codebook. The stage-0 evidence snap tags its
    # selected rows LF-COARSE (analysis outputs are the coarse
    # characterizations).
    from Spaces import Codebook
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    basis = ss.subspace.what
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    ss.forward(m._empty_seed_ss, IS_concepts=u)
    idx = ss._stage0_indices
    assert idx is not None
    for row in idx.reshape(-1)[:8].tolist():
        assert basis.get_descriptor_role(row) == Codebook.ROLE_LF_COARSE, (
            f"stage-0 snapped row {row} must be tagged LF_COARSE")
    # Roles are per-row metadata over the ONE codebook: a different role
    # can be assigned and read back (don-spyi / sgra-spyi are roles, not
    # separate codebooks).
    free_row = int(idx.reshape(-1).max()) + 1
    if free_row < int(basis.codebookSize):
        basis.set_descriptor_role(free_row, Codebook.ROLE_MEANING_GENERAL)
        assert (basis.get_descriptor_role(free_row)
                == Codebook.ROLE_MEANING_GENERAL)


def test_semantic_arrangement_mechanism():
    # Task 5 (C-13) contract: OFF by default (no term); with a weight set,
    # the post-sentence arrangement produces a loss whose gradient lands
    # ONLY on the activated rows (pode/antipode are detached). Semantic
    # PAYOFF is deliberately not asserted -- that is D's corpus gate
    # (asymmetric-vq sec.8: XOR cannot validate the semantic side).
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    vq = ss.subspace.what.vq
    u = torch.randint(0, 256, (2, 1, 512), dtype=torch.int64)
    # Off by default.
    assert float(getattr(ss, "semantic_arrangement_weight", 0.0)) == 0.0
    ss.train()
    try:
        ss.forward(m._empty_seed_ss, IS_concepts=u)
        assert ss._stage0_semantic_loss is None, (
            "semantic arrangement must be OFF by default")
        # Enable and re-run.
        ss.semantic_arrangement_weight = 0.5
        ss.forward(m._empty_seed_ss, IS_concepts=u)
        sem = ss._stage0_semantic_loss
    finally:
        ss.eval()
        ss.semantic_arrangement_weight = 0.0
    assert sem is not None and sem.requires_grad
    if vq.codebook.grad is not None:
        vq.codebook.grad = None
    sem.backward()
    g = vq.codebook.grad
    assert g is not None and float(g.abs().sum()) > 0
    idx = ss._stage0_indices.reshape(-1).unique()
    sel = torch.zeros(g.shape[0], dtype=torch.bool)
    sel[idx] = True
    assert float(g[~sel].abs().max()) == 0.0, (
        "the arrangement gradient must land ONLY on the activated rows "
        "(pode and antipode are detached)")


def test_painting_reverse_blend():
    # Phase 7 contract (plan sec.6, rev. 2026-06-10): the reconstruction
    # recombination is PAINTING -- the Universal view paints the
    # background (each slot's coarse value over its contiguous region),
    # the Atomic view is AVERAGED in where it has support, and the
    # background remains alone where it doesn't (padding painted, not
    # halved). The concepts branch RIDES the SubSpace
    # (``_concepts_recon``) -- reverse stays single-arg per the
    # processing contract.
    m = _build("MM_20M.xml")
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


def test_model_reverse_threads_concepts_branch():
    # Phase 7 wiring: the stage-0 SS stream of the bind's exact inverse is
    # the conceptual reconstruction branch; _reverse_body stamps it onto
    # the returned sub (SubSpace-carried), and the model reverse() carries
    # it across the PS handoff into InputSpace.reverse. End-to-end reverse
    # stays finite.
    m = _build("MM_20M.xml")
    x = _staged_batch(m)
    with torch.no_grad():
        m.forward(x)
        body_sub = m._reverse_body(m._combine_last_cs_sub)
    assert getattr(body_sub, "_concepts_recon", None) is not None, (
        "_reverse_body must stamp the stage-0 SS recon stream onto the "
        "returned sub")
    with torch.no_grad():
        m.forward(x)  # re-bind (the first reverse consumed the state)
        recon = m.reverse(m._combine_last_cs_sub)
        ev = recon.materialize() if recon is not None else None
    assert ev is not None and torch.isfinite(ev).all(), (
        "the two-branch painting reverse must stay finite end-to-end")
    painted = getattr(m.inputSpace.subspace, "_painted_event", None)
    assert painted is not None and torch.isfinite(painted).all(), (
        "the painted combined surface must ride the IS SubSpace")
