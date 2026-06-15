"""Step-1 contract tests -- symbolic-iteration codebook on the CS->SS leg
(doc/plans/2026-06-10-symbolic-iteration-codebook.md, Step 1; the
architectural statement is the "The <codebook> knob STAYS" section of
doc/plans/2026-06-08-analysis-synthesis-dual-input.md).

Semantics under test (parallel leg, t>0, <codebook>quantize</codebook>):

  * THE CODEBOOK REPLACES PI: the snap stands in for the Pi transform on
    the CS leg (selection-by-exclusion replaces computed intersection);
    the parallel fold AND the S-tier syntactic dispatch are bypassed.
  * ONE SYMBOL AT A TIME, APOHA: the emission frame carries the selected
    symbol's code in exactly ONE slot; the copart is ZEROS EVERYWHERE
    (anyapoha -- the universal appears through the exclusion of the
    other).
  * Value substitution is CORRECT here (symbolic iterations) -- unlike
    stage 0, where analysis does not alter the data.
  * Honest STE: a VIRGIN winner row (never adopted/named) must NOT be
    substituted -- the iteration stays continuous until the host-eager
    adopt-on-first-sight (stem) has named the row (the #13 lesson:
    value substitution against a virgin/random codebook poisons
    training).
  * The recon gather retargets to THIS leg's evidence: the winner row
    trains toward the concept code that selected it (EMA stays off;
    commitment stays 0).

An UNTRAINED model's CS slab is nearly slot-uniform (every slot snaps to
the same row), so these tests PLANT codebook rows / crafted CS events to
pin the contracts deterministically rather than relying on cold-forward
variety.
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
    # Force CPU regardless of import order: the env setdefault above is
    # too late when another test file already initialised the default
    # device (MPS) at util import.
    init_device("cpu")
    torch.manual_seed(0)
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


def _cs_view(m):
    """Drive one full forward and return the REAL persistent CS->SS view
    (``cs._subspaceForSS``) -- the t>0 leg's input geometry. The
    completed forward leaves a stale all-False AR valid_mask on the
    persistent subspaces; in-body t>0 passes run with the batch's live
    mask (None for IR/XOR), so it is cleared here."""
    x = _staged_batch(m)
    with torch.no_grad():
        m.forward(x)
    view = m.body_stages[0]["cs"]._subspaceForSS
    assert view is not None and not view.is_empty()
    view.valid_mask = None
    m.symbolicSpace.subspace.valid_mask = None
    return view


def _plant_rows(ss, rows, scale=0.9):
    """Write deterministic block-sign codes into codebook rows ``rows``
    (the analytic store, ``vq.codebook``), refresh the VQ's cached row
    norms, and tag the rows MEANING_GENERAL (simulating completed
    adoption). Returns the planted [len(rows), W] codes."""
    from Spaces import Codebook
    basis = ss.subspace.what
    vq = basis.vq
    W = int(ss.nDim)
    codes = []
    with torch.no_grad():
        for k, r in enumerate(rows):
            c = torch.zeros(vq.codebook.shape[-1],
                            dtype=vq.codebook.dtype)
            # distinct deterministic pattern per row: flip a different
            # 1/len(rows) stripe of an all-positive base
            c[:W] = scale
            lo = (W * k) // len(rows)
            hi = (W * (k + 1)) // len(rows)
            c[lo:hi] = -scale
            vq.codebook[r] = c
            codes.append(c[:W].clone())
        if hasattr(vq, "_b_norms_sq"):
            vq._b_norms_sq.copy_((vq.codebook.detach() ** 2).sum(dim=-1))
    basis.set_descriptor_role(
        torch.tensor(rows, dtype=torch.long), Codebook.ROLE_MEANING_GENERAL)
    return torch.stack(codes)


def _craft_event(view, W, slot_codes):
    """Write a crafted event onto the CS view: ``slot_codes`` is a list
    of (batch_row, slot, code[W]) placements; everything else zero."""
    ev0 = view.materialize()
    ev = torch.zeros_like(ev0)
    for b, n, c in slot_codes:
        ev[b, n, :W] = c
    view.set_event(ev)
    return ev


def test_csleg_pi_bypassed_under_quantize():
    # THE CODEBOOK REPLACES PI: under quantize the CS-leg forward must
    # not apply the pi transform -- the snap IS this iteration's
    # analysis. (The S-tier syntactic dispatch is bypassed by the same
    # predicate; the fold is the directly patchable surface.)
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    fold = getattr(ss, "pi", None)
    if fold is None:
        pytest.skip("MM_20M SS carries no parallel fold to bypass")
    real = fold.forward
    def _boom(*a, **k):
        raise AssertionError(
            "pi fold must be BYPASSED on the quantize CS leg "
            "(the codebook replaces Pi)")
    fold.forward = _boom
    try:
        with torch.no_grad():
            out = ss.forward(view)
    finally:
        fold.forward = real
    ev = out.materialize()
    assert ev is not None and torch.isfinite(ev).all()


def test_csleg_one_symbol_apoha_emission():
    # ONE SYMBOL AT A TIME with APOHA ZEROS: with adopted (planted) rows
    # and a CS event whose slot 5 carries row 20's code, the emission is
    # exactly that one slot carrying that one code; every other slot is
    # exactly zero, value AND copart.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    W = int(ss.nDim)
    N = int(ss.inputShape[0])
    codes = _plant_rows(ss, rows=[20, 21])
    ss.stage_symbolic_virgin_rows()
    B = view.materialize().shape[0]
    placements = [(b, 5 if b % 2 == 0 else 2, codes[b % 2]) for b in range(B)]
    _craft_event(view, W, placements)
    with torch.no_grad():
        out = ss.forward(view)
    ev = out.materialize()
    frame = ev.reshape(B, N, W)
    nonzero_slots = (frame.abs().sum(dim=-1) > 0)
    assert nonzero_slots.sum(dim=1).tolist() == [1] * B, (
        "apoha: exactly ONE live slot per batch row, got "
        f"{nonzero_slots.sum(dim=1).tolist()}")
    win_slot, win_row = ss._symbolic_emission
    vq = ss.subspace.what.vq
    for b, n, c in placements:
        assert int(win_slot[b]) == n, (
            f"row {b}: winner slot {int(win_slot[b])} != planted slot {n}")
        assert int(win_row[b]) == (20 if b % 2 == 0 else 21), (
            f"row {b}: winner row {int(win_row[b])} != planted row")
        assert torch.allclose(
            frame[b, n], vq.codebook[int(win_row[b])][:W].to(frame.dtype),
            atol=1e-5), (
            "the live slot must carry the WINNER ROW'S CODE "
            "(value substitution is correct on symbolic iterations)")


def test_csleg_virgin_fallback_stays_continuous():
    # Honest STE (the #13 lesson): with EVERY row virgin the iteration
    # must stay CONTINUOUS -- no codebook value substitution, no apoha
    # sparsity (a symbol that does not exist yet cannot be emitted).
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    W = int(ss.nDim)
    N = int(ss.inputShape[0])
    basis = ss.subspace.what
    roles = basis.ensure_descriptor_roles()
    with torch.no_grad():
        roles.zero_()                  # forget stage-0 tagging: all virgin
    ss.stage_symbolic_virgin_rows()
    assert bool(ss._staged_virgin_rows.all())
    ev0 = view.materialize()
    dense = torch.full_like(ev0, 0.3)
    view.set_event(dense)
    with torch.no_grad():
        out = ss.forward(view)
    ev = out.materialize()
    B = ev0.shape[0]
    frame = ev.reshape(B, N, W)
    live = (frame.abs().sum(dim=-1) > 0).sum(dim=1)
    assert int(live.min()) == N, (
        "virgin fallback must keep the CONTINUOUS carrier (all slots "
        f"dense), got live-slot counts {live.tolist()}")


def test_csleg_adoption_writes_tags_and_is_idempotent():
    # Adopt-on-first-sight, re-homed to the CS leg: a VIRGIN row adopts
    # the evidence vector that selects it, is tagged MEANING_GENERAL
    # (don-spyi: the concept-universal face -- distinct from the stage-0
    # analysis face, LF_COARSE), and a second pass is bit-stable.
    from Spaces import Codebook
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    basis = ss.subspace.what
    vq = basis.vq
    W = int(ss.nDim)
    roles = basis.ensure_descriptor_roles()
    target = 33
    assert roles[target] == Codebook.ROLE_UNASSIGNED, (
        "test premise: row 33 must start virgin")
    # Evidence near row 33's CURRENT value selects row 33 (L2 self-match)
    # but differs from it, so adoption must WRITE the evidence in.
    with torch.no_grad():
        evidence = vq.codebook[target][:W].clone() + 0.01
    _craft_event(view, W, [(0, 0, evidence)])
    ss.train()
    try:
        ss.adopt_symbolic_evidence(view)
    finally:
        ss.eval()
    assert roles[target] == Codebook.ROLE_MEANING_GENERAL, (
        "the adopted row must be tagged MEANING_GENERAL at adoption time")
    assert torch.allclose(
        vq.codebook[target][:W].detach(), evidence.to(vq.codebook.dtype),
        atol=1e-6), "the virgin row must ADOPT the evidence that selected it"
    snapshot = vq.codebook.detach().clone()
    ss.train()
    try:
        ss.adopt_symbolic_evidence(view)
    finally:
        ss.eval()
    assert torch.equal(snapshot, vq.codebook.detach()), (
        "second adoption pass must be bit-stable (adoption tags; tagged "
        "rows are no longer virgin)")


def test_csleg_recon_gather_lands_on_winner_rows():
    # The recon gather retargets to THIS leg: gradient support is exactly
    # the winner rows -- the argmax blocks the encoder leg, the evidence
    # is detached, EMA stays off.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    W = int(ss.nDim)
    codes = _plant_rows(ss, rows=[40, 41])
    ss.stage_symbolic_virgin_rows()
    B = view.materialize().shape[0]
    placements = [(b, 1, codes[b % 2]) for b in range(B)]
    _craft_event(view, W, placements)
    vq = ss.subspace.what.vq
    assert isinstance(vq.codebook, torch.nn.Parameter)
    ss.train()
    try:
        ss.forward(view)
        recon = getattr(ss, "_csleg_recon_loss", None)
    finally:
        ss.eval()
    assert recon is not None and recon.requires_grad, (
        "training-mode symbolic iteration must thread the CS-leg recon term")
    if vq.codebook.grad is not None:
        vq.codebook.grad = None
    recon.backward()
    g = vq.codebook.grad
    assert g is not None
    _, win_row = ss._symbolic_emission
    sel = torch.zeros(g.shape[0], dtype=torch.bool)
    sel[win_row.reshape(-1).cpu()] = True
    assert float(g[~sel].abs().max()) == 0.0, (
        "recon gradient must touch ONLY the winner rows")


def test_csleg_naming_indices_thread_full_frame():
    # The snap still NAMES every slot (indices thread for the narrow
    # output / downstream consumers) even though only one symbol emits.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    ss.stage_symbolic_virgin_rows()
    with torch.no_grad():
        ss.forward(view)
    idx = getattr(ss, "_naming_indices", None)
    assert idx is not None and idx.dim() == 2, (
        "per-slot naming indices must thread on the symbolic iteration")
    assert int(idx.shape[1]) == int(ss.inputShape[0])


@pytest.mark.xfail(
    reason="Step 4 of the 2026-06-10 symbolic-iteration plan: the narrow "
           "second-order emission ([nOutput, nOutputDim] = e.g. MM_20M "
           "[1024, 8] = 4-wide written-symbol ID + 2 where + 2 when, "
           "band included) and the REVERSE that keys the codebook by the "
           "ID to recreate the full concept representation are not built "
           "yet; the forward currently emits the value reshape of the "
           "apoha frame and the reverse re-applies the S-tier transform "
           "instead of keying.",
    strict=False)
def test_mm20m_second_order_reverse_keys_codebook():
    # THE 4-D SECOND-ORDER ACCEPTANCE (plan Step 4; MM_20M ships SS
    # quantize): the symbolic iteration emits narrow codes at the
    # CONFIGURED [nOutput, nOutputDim] geometry (the muxed frame,
    # band included -- MM_20M: [1024, 8]); the REVERSE keys the codebook
    # by the 4-wide symbol ID and recreates the full wide concept
    # representation -- "reconstruct by keying the codebook with
    # indices". Quantization is what allows CS to return ACTIVATION
    # VALUES ONLY -> second-order symbols.
    m = _build("MM_20M.xml")
    ss = m.symbolicSpace
    view = _cs_view(m)
    W = int(ss.nDim)
    codes = _plant_rows(ss, rows=[50, 51])
    ss.stage_symbolic_virgin_rows()
    B = view.materialize().shape[0]
    placements = [(b, 3, codes[b % 2]) for b in range(B)]
    _craft_event(view, W, placements)
    with torch.no_grad():
        out = ss.forward(view)
    narrow = out.materialize()
    # The narrow emission at the CONFIGURED output geometry (muxed:
    # nOutput rows of nOutputDim = what+where+when).
    assert narrow.dim() == 3, "narrow emission must be [B, nOutput, nOutputDim]"
    assert (int(narrow.shape[1]), int(narrow.shape[2])) == (
        int(ss.outputShape[0]), int(ss.nOutputDim)), (
        f"the symbolic iteration must emit the configured narrow geometry "
        f"[{int(ss.outputShape[0])}, {int(ss.nOutputDim)}], got "
        f"{tuple(narrow.shape[1:])}")
    win_slot, win_row = ss._symbolic_emission
    # REVERSE: recreate the full concept representation by KEYING the
    # codebook with the emitted symbol id.
    with torch.no_grad():
        rec = ss.reverse(out)
    wide = rec.materialize()
    N = int(ss.inputShape[0])
    frame = wide.reshape(B, N, -1)
    vq = ss.subspace.what.vq
    for b in range(B):
        s = int(win_slot[b]); r = int(win_row[b])
        n = min(int(frame.shape[-1]), W)
        row_code = vq.codebook[r][:n].to(frame.dtype)
        assert torch.allclose(frame[b, s, :n], row_code, atol=1e-4), (
            "the reverse must RECREATE the full concept code at the "
            "winner slot by keying the codebook with the emitted id")


def test_forward_body_lifts_csleg_recon():
    # Full-model wiring (subsymbolicOrder=2, parallel, SS quantize -- the
    # MM_symbolic_iter fixture; MM_20M ships order 1, whose t>0 leg
    # never runs in-body): the t>0 recon term reaches the pipeline error
    # container, and the stem's adopt-on-first-sight + virgin staging
    # run on the recurrent leg.
    m = _build("MM_symbolic_iter.xml")
    x = _staged_batch(m)
    m.train()
    try:
        m.forward(x)            # cold: CS views fill; adoption next stem
        m.forward(x)            # warm: stem adopted; symbolic emission live
    finally:
        m.eval()
    terms = getattr(m.outputSpace.subspace.errors, "_terms", {})
    assert "ss_codebook_recon" in terms, (
        f"the CS-leg recon term must reach the pipeline error container; "
        f"got terms={list(terms)}")
    # The stem staged the virgin mask on every SS stage. None is a
    # legitimate parked state (= every row virgin: the roles buffer is
    # only allocated by the first adoption/tagging); the contract is
    # that staging RAN, i.e. the attribute exists.
    for stage in m.body_stages:
        ss = stage["ss"] if "ss" in stage else None
        if ss is not None:
            assert hasattr(ss, "_staged_virgin_rows"), (
                "the stem must park the virgin-row staging for the body")
