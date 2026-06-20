import os, sys, warnings
import pytest
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
os.environ.setdefault("MODEL_COMPILE", "eager")
_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
if _BIN not in sys.path: sys.path.insert(0, _BIN)

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"

def _build(name):
    import Models, Language
    from util import init_config
    p = os.path.join(os.path.dirname(_BIN), "data", name)
    init_config(path=p, defaults_path=os.path.join(os.path.dirname(_BIN), "data", "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m

def test_reconstruct_enum_retired():
    # A1 (2026-06-09): the ``<reconstruct>`` enum was retired (schema element +
    # reconstructEnum removed). The combine now UNCONDITIONALLY mixes all three
    # streams (reconstruction is unconditionally from concepts), so the model
    # carries NEITHER the ``reconstruct`` enum string NOR the derived
    # ``perfect_reconstruction`` bool.
    m = _build("MM_20M.xml")
    assert not hasattr(m, "reconstruct"), (
        "the <reconstruct> enum was retired (A1); the model must not carry a "
        "self.reconstruct attribute")
    assert not hasattr(m, "perfect_reconstruction"), (
        "the derived perfect_reconstruction bool was retired with the enum")


def test_combine_square_roundtrip_exact():
    # 2-stream SLOT bind (C-10; geometry corrected 2026-06-10):
    # CS = ILL(stack[PS ; SS]) -- the streams stack along the VECTOR axis
    # (N slots each) and the carrier is the WHOLE flattened bind (no
    # augment); reverse is exact by construction. PS view = vectors
    # 0..N-1, SS view = vectors N..2N-1 of the STORED mix.
    import torch
    from Layers import ConceptualCombine
    B, Nv, D = 2, 4, 6
    c = ConceptualCombine(content_dim=D, n_vectors=Nv,
                          naive=False, sigma_pi_mode="full")
    ps, ws = (torch.randn(B, Nv, D).clamp(-0.5, 0.5) for _ in range(2))
    full = c.forward(ps, ws)
    assert full.shape == (B, c.carrier_dim)
    assert c.carrier_dim == 2 * Nv * D       # dense: no padding
    # The views are the slot-halves of the bind itself.
    ps_v, ws_v = c.views(full)
    assert ps_v.shape == (B, Nv, D) and ws_v.shape == (B, Nv, D)
    ps2, ss2 = c.reverse(full)
    assert ps2.shape == (B, Nv, D) and ss2.shape == (B, Nv, D)
    err = max((ps - ps2).abs().max(), (ws - ss2).abs().max())
    assert err < 1e-3, f"2-stream slot-bind round-trip err={err:.2e}"


def test_combine_square_roundtrip_exact_butterfly():
    # Butterfly is the production <sigmaPi> default; the cross-element
    # linear 2x2-LDU cascade must round-trip EXACTLY -- the 2-stream bind
    # (C-10) is a true bijection with NOTHING threaded alongside.
    #
    # LOAD-BEARING: this test perturbs the butterfly node parameters OFF
    # the identity init before round-tripping. At identity init the cascade
    # is trivially the identity and every coordinate (including any padded
    # tail) stays where it started, so the round-trip is vacuously exact and
    # would NOT catch the prior bug (cascade signal leaking into stripped
    # zero-pad coords that the zero-re-padding reverse cannot recover). With
    # perturbed weights the cascade genuinely mixes all M coordinates, so an
    # exact round-trip proves ConceptualCombine sizes the cascade at the
    # power-of-two width M (== layer N == M_total) -- nothing padded/stripped.
    import torch
    torch.manual_seed(0)
    from Layers import ConceptualCombine

    def _check(D, Nv, leading):
        c = ConceptualCombine(
            content_dim=D, n_vectors=Nv,
            naive=False, sigma_pi_mode="butterfly")
        assert c.sigma_pi_mode == "butterfly"
        gl = c.layer
        flat = 2 * Nv * D
        M = 1 << ((flat - 1).bit_length())  # next pow2 >= 2N*D
        # The cascade must be sized at M, with NO padding/stripping.
        assert gl.N == M and gl.M_total == M, (
            f"D={D},N={Nv}: cascade not sized at M=next_pow2(2ND); "
            f"N={gl.N} M_total={gl.M_total} M={M}")
        assert c.combine_padded == M
        assert c.carrier_dim == M
        # Perturb the per-level butterfly node params OFF identity so the
        # cascade actually mixes every coordinate (not a no-op).
        with torch.no_grad():
            for p in (gl.butterfly_L, gl.butterfly_d, gl.butterfly_U):
                p.add_(0.1 * torch.randn_like(p))
        # Verify the perturbation actually moved the cascade off identity:
        # a fresh random input must NOT be returned unchanged by forward.
        probe = torch.randn(*leading, Nv, D)
        full_p = c.forward(probe, probe)
        ps_vp, _ = c.views(full_p)
        with torch.no_grad():
            moved = float((ps_vp - probe).abs().max())
        assert moved > 1e-3, (
            f"D={D}: butterfly weights still at identity (moved={moved:.2e}); "
            "test would be vacuous")

        ps, ws = (
            torch.randn(*leading, Nv, D).clamp(-0.5, 0.5) for _ in range(2))
        full = c.forward(ps, ws)
        assert full.shape == (*leading, M), (
            f"the carrier must be the WHOLE bind (width M={M}), got "
            f"{tuple(full.shape)}")
        # Views are interface-shaped [.., N, D] slot-halves.
        ps_v, ws_v = c.views(full)
        assert ps_v.shape == (*leading, Nv, D)
        assert ws_v.shape == (*leading, Nv, D)
        ps2, ss2 = c.reverse(full)
        for t in (ps2, ss2):
            assert t.shape == (*leading, Nv, D)
        with torch.no_grad():
            err = float(max((ps - ps2).abs().max(),
                            (ws - ss2).abs().max()))
        assert err == err and err != float("inf"), "non-finite round-trip err"
        assert err < 1e-3, (
            f"D={D},N={Nv} leading={leading}: butterfly slot-bind "
            f"round-trip err={err:.2e} (perturbed weights)")
        return err

    # 2ND: 2*2*6=24 -> M=32; 2*2*8=32 -> M=32; 2*4*4=32 -> M=32.
    for D, Nv in ((6, 2), (8, 2), (4, 4)):
        _check(D, Nv, leading=(2,))
    # Multi-leading-dim [B, T, N, D] locks the shape contract:
    # ConceptualCombine must flatten leading dims itself (B=2, T=3).
    _check(6, 2, leading=(2, 3))
    _check(8, 2, leading=(2, 3))


def test_combine_square_roundtrip_exact_dense_large_leading():
    # Dense (full) path at a production-scale width with a multi-leading-dim
    # [B, T, D] input -- locks the dense shape contract the body feeds and
    # confirms the leading-dim flatten/restore is exact for the dense LDU.
    import torch
    torch.manual_seed(0)
    from Layers import ConceptualCombine
    B, T, Nv, D = 2, 3, 4, 64
    c = ConceptualCombine(content_dim=D, n_vectors=Nv,
                          naive=False, sigma_pi_mode="full")
    assert c.combine_padded == 2 * Nv * D and c.carrier_dim == 2 * Nv * D
    # Perturb the LDU off identity so the round-trip is non-trivial.
    with torch.no_grad():
        c.layer.raw_L.add_(0.05 * torch.randn_like(c.layer.raw_L))
        c.layer.raw_U.add_(0.05 * torch.randn_like(c.layer.raw_U))
    ps, ws = (torch.randn(B, T, Nv, D).clamp(-0.5, 0.5) for _ in range(2))
    full = c.forward(ps, ws)
    assert full.shape == (B, T, 2 * Nv * D)
    ps2, ss2 = c.reverse(full)
    for t in (ps2, ss2):
        assert t.shape == (B, T, Nv, D)
    with torch.no_grad():
        err = float(max((ps - ps2).abs().max(),
                        (ws - ss2).abs().max()))
    assert err < 1e-3, f"dense large [B,T,N,D] round-trip err={err:.2e}"


def test_mm5m_forward_finite_after_combine():
    # Step-2 regression (A4): the parallel conceptual-recurrence forward,
    # now driven by the per-stage ConceptualCombine, must still produce a
    # FINITE output on MM_20M. ``forward`` returns a 4-tuple
    # ``(input_state, sym_vectors, pred, None)`` (Models.py:7159); the head
    # prediction is index [2] (index [0] is the input embedding, not the CS
    # carrier).
    import torch
    m = _build("MM_20M.xml")
    import Models; Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
    # The actual terminal conceptual carrier (the last stage's combine output,
    # threaded as a forward-local) must be finite too.
    assert m._combine_carriers and m._combine_carriers[-1] is not None
    assert torch.isfinite(m._combine_carriers[-1]).all()


def test_mm5m_combine_carrier_roundtrip():
    # 2-stream bind round-trip (C-10, rev. 2026-06-09): each stage's
    # threaded carrier is the WHOLE bind ILL([PS_t || WS_t]); reverse is an
    # exact bijection with NOTHING threaded alongside (the augment machinery
    # is retired). Builds from the STOCK MM_20M.xml with no injected knob.
    import os, warnings, torch
    import Models, Language
    from util import init_config
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    p = os.path.join(data_dir, "MM_20M.xml")
    init_config(path=p, defaults_path=os.path.join(data_dir, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    # The retired enum leaves NO attribute behind.
    assert not hasattr(m, "perfect_reconstruction"), (
        "the <reconstruct> enum (and its derived perfect_reconstruction bool) "
        "was retired (A1)")
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    # forward() -> (input_state, sym_vectors, pred, None); [0] is the input
    # embedding (finiteness smoke check that the forward ran).
    input_state = m.forward(x)[0]
    assert torch.isfinite(input_state).all()
    T = len(m.body_stages)
    carriers = list(m._combine_carriers)
    assert len(carriers) == T and all(c is not None for c in carriers), (
        "every parallel stage must thread a (non-None) full-bind carrier")
    # The augment thread is GONE: the 2-stream bind needs nothing alongside.
    assert not hasattr(m, "_combine_augments"), (
        "the augment thread was retired with the 2-stream bind (C-10)")
    # The stage-0 snapshot IS the stage-0 carrier (the whole bind).
    fwd_cs0 = m._combine_fwd_cs0.detach()
    assert torch.equal(fwd_cs0, carriers[0].detach()), (
        "_combine_fwd_cs0 must snapshot the stage-0 full bind")
    combine0 = m.conceptualSpaces[0].combine
    M = int(combine0.carrier_dim)
    D = int(combine0.content_dim)
    Nv = int(combine0.n_vectors)
    assert fwd_cs0.shape[-1] == M, (
        f"the carrier must be the WHOLE flattened bind (width {M}), got "
        f"{tuple(fwd_cs0.shape)}")
    # The production parallel bind: ONE cascade over 2N*D = 16*nDim
    # (cross-slot reach; 2^14 exactly -- zero pad).
    assert M == 2 * Nv * D, (
        f"production bind must be unpadded (2N*D == M): "
        f"2*{Nv}*{D} != {M}")
    # The views are interface-shaped [B, N, D] slot-halves of the bind.
    with torch.no_grad():
        ps_v, ws_v = combine0.views(carriers[0].detach())
    assert ps_v.shape[-2] == Nv and ps_v.shape[-1] == D
    assert ws_v.shape[-2] == Nv and ws_v.shape[-1] == D
    # Exact per-stage inversion: ILL^{-1}(full_t) -> (PS_t, WS_t), finite,
    # and the t=0 PS-stream (the encoded input leg) must be NON-trivial --
    # it is the leg that decodes back to the input (PS owns reconstruction).
    with torch.no_grad():
        for t in range(T):
            ps_rec, ws_rec = m.conceptualSpaces[t].combine.reverse(
                carriers[t].detach())
            assert ps_rec.shape[-2:] == (Nv, D)
            assert ws_rec.shape[-2:] == (Nv, D)
            assert torch.isfinite(ps_rec).all() and torch.isfinite(ws_rec).all()
        ps0_rec, ss0_rec = combine0.reverse(carriers[0].detach())
        ps0_mag = float(ps0_rec.abs().max())
    assert ps0_mag > 1e-6, (
        f"stage 0 PS-stream (encoded input) must be non-trivial, "
        f"got max={ps0_mag:.2e}")
    # Bijection sanity at the production width: re-binding the recovered
    # streams reproduces the carrier to solve tolerance.
    with torch.no_grad():
        rebound = combine0.forward(ps0_rec, ss0_rec)
        err = float((rebound - carriers[0].detach()).abs().max())
    assert err < 1e-2, f"bind/unbind round-trip err={err:.2e}"
    # End-to-end concept-carrier reverse on the integrated body path must be
    # FINITE (exercises _reverse_body's bind-reverse + cs.reverse chain).
    recon_sub = m._reverse_body(m._combine_last_cs_sub)
    recon_ev = recon_sub.materialize()
    assert recon_ev is not None and torch.isfinite(recon_ev).all()


def _write_mm5m_3ep_cfg():
    """Write a 3-epoch copy of data/MM_20M.xml to /tmp and return its path.

    Three epochs over the (tiny) xor stream is enough to drive several
    runBatch forwards across sentence/document boundaries -- the regime
    where the A5 bug surfaced: per-batch STM data persisted on the space
    used to leave an oscillating ``requires_grad`` on the idea buffer that
    flipped a Dynamo guard (recompile) and crashed AOT functionalization.
    """
    import re
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    with open(os.path.join(data_dir, "MM_20M.xml")) as fh:
        xml = fh.read()
    xml2 = re.sub(r"<numEpochs>\d+</numEpochs>",
                  "<numEpochs>3</numEpochs>", xml)
    assert "<numEpochs>3</numEpochs>" in xml2, "epochs knob not found/rewritten"
    cfg = "/tmp/MM_5M_3ep.xml"
    with open(cfg, "w") as fh:
        fh.write(xml2)
    return cfg


@pytest.mark.skipif(not _RUN_SLOW, reason="slow (~33s no-recompile fullgraph gate) -- set RUN_SLOW=1")
def test_no_recompile_fullgraph():
    """A5 gate: the compiled step must compile ONCE across batches.

    Per-batch DATA must thread THROUGH the forward as tensors and never be
    persisted as accumulated state on a space/Layer. If the STM idea buffer
    (``_idea_buffer`` when a SymbolicSubSpace is attached, ``_fallback_buffer``
    standalone) is mutated in-place inside the compiled forward with
    grad-bearing per-batch data, its ``requires_grad`` oscillates across
    forwards, flips a Dynamo guard, and forces a recompile every batch --
    and under AOT autograd it crashes the runtime alias regeneration.

    Gate: run the model under ``MODEL_COMPILE=aot_eager`` with
    ``TORCH_LOGS=recompiles`` and assert the buffer name + the
    ``requires_grad mismatch`` guard string never appear in the output, and
    that the run did not crash.
    """
    import subprocess
    env = {**os.environ,
           "BASICMODEL_DEVICE": "cpu",
           "MODEL_COMPILE": "aot_eager",
           "KMP_DUPLICATE_LIB_OK": "TRUE",
           "TORCH_LOGS": "recompiles"}
    cfg = _write_mm5m_3ep_cfg()
    py = os.path.join(os.path.dirname(_BIN), ".venv", "bin", "python")
    out = subprocess.run(
        [py, os.path.join(_BIN, "Models.py"), cfg],
        env=env, capture_output=True, text=True)
    combined = out.stdout + out.stderr
    # The run must complete (the AOT alias-regeneration crash was the
    # buffer-as-graph-input symptom; a clean exit proves it's gone).
    assert out.returncode == 0, (
        f"compiled run crashed (rc={out.returncode}):\n{combined[-3000:]}")
    # No per-batch STM buffer may appear in a recompile guard.
    assert "_fallback_buffer" not in combined, (
        f"_fallback_buffer surfaced in a recompile guard:\n{combined[-3000:]}")
    assert "_idea_buffer" not in combined, (
        f"_idea_buffer surfaced in a recompile guard:\n{combined[-3000:]}")
    assert "requires_grad mismatch" not in combined, (
        f"a requires_grad guard flipped (recompile):\n{combined[-3000:]}")


# ``test_reconstruct_expectation_carrier_accumulates`` REMOVED (A1,
# 2026-06-09): it exercised ``<reconstruct>expectation</reconstruct>``, a mode
# of the now-retired ``reconstruct`` enum. The enum (schema element +
# reconstructEnum) is gone; reconstruction is unconditionally from concepts.


def test_combine_no_dropped_reverse():
    # The dropped-augment reverse was RETIRED with the 2-stream bind (C-10):
    # the carrier is the whole bind, so there is no augment to drop and no
    # approximate regime -- reverse is exact by construction.
    from Layers import ConceptualCombine
    c = ConceptualCombine(content_dim=6, naive=False, sigma_pi_mode="full")
    assert not hasattr(c, "reverse_dropped"), (
        "reverse_dropped must be retired (the 2-stream bind has no augment)")
    assert not hasattr(c, "aug_dim"), (
        "aug_dim must be retired (the carrier IS the whole bind)")


def test_callosum_glue_init_average():
    # The corpus callosum (2026-06-10): CS glues the bind's STACKED views
    # through a learned [2N, N] matrix over the slot axis. Initialised to
    # AVERAGING the two hemispheres; learnable thereafter.
    import torch
    from Layers import ConceptualCombine
    B, Nv, D = 2, 4, 6
    c = ConceptualCombine(content_dim=D, n_vectors=Nv,
                          naive=False, sigma_pi_mode="full")
    assert c.callosum.shape == (2 * Nv, Nv)
    assert c.callosum.requires_grad, "the callosum is a learned glue"
    ps, ws = (torch.randn(B, Nv, D).clamp(-0.5, 0.5) for _ in range(2))
    full = c.forward(ps, ws)
    glued = c.glue(full)
    assert glued.shape == (B, Nv, D)
    ps_v, ws_v = c.views(full)
    assert torch.allclose(glued, 0.5 * (ps_v + ws_v), atol=1e-6), (
        "at init the callosum must AVERAGE the two hemispheres")


def test_bind_contained_in_conceptual_space():
    # Processing contract (2026-06-10): the bind calculation lives ON
    # ConceptualSpace (bind_streams / unbind), and the carrier rides ON
    # the stage's SubSpace (_bind_carrier) -- not model-level state. The
    # body loop is an orchestrator only.
    import os, warnings, torch
    import Models, Language
    from util import init_config
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    p = os.path.join(data_dir, "MM_20M.xml")
    init_config(path=p, defaults_path=os.path.join(data_dir, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    with torch.no_grad():
        m.forward(x)
    cs0 = m.conceptualSpaces[0]
    # The carrier rides on the stage's SubSpace.
    carrier = getattr(cs0.subspace, "_bind_carrier", None)
    assert carrier is not None, (
        "the bind carrier must ride ON the stage SubSpace (_bind_carrier)")
    assert torch.equal(carrier.detach(), m._combine_carriers[0].detach()), (
        "the SubSpace-carried bind and the test handle must be the same")
    # unbind: the Space's exact inverse from SubSpace-carried state.
    with torch.no_grad():
        rec = cs0.unbind(cs0.subspace)
    assert rec is not None
    ps_rec, ws_rec = rec
    Nv = int(cs0.combine.n_vectors)
    D = int(cs0.combine.content_dim)
    assert ps_rec.shape[-2:] == (Nv, D) and ws_rec.shape[-2:] == (Nv, D)
    # The head-facing event is the corpus-callosum glue of THAT stage's
    # bind (+ the empty band at D == muxedSize), written by bind_streams
    # onto the FLOWING sub (the per-batch CS_sub handed downstream). At
    # subsymbolicOrder>1 the flowing sub is the LAST stage's (MM_20M ships
    # sO=3), so compare against the LAST stage's carrier+combine -- not
    # stage 0's (which only coincides with the flow when T==1).
    last = m._combine_last_cs_sub
    last_carrier = getattr(last, "_bind_carrier", None)
    assert last_carrier is not None, (
        "the flowing CS_sub must carry the bind too")
    cs_last = m.conceptualSpaces[len(m.body_stages) - 1]
    with torch.no_grad():
        glued = cs_last.combine.glue(last_carrier)
        ev = last.materialize()
    assert ev is not None and ev.shape[-2:] == (Nv, D)
    assert torch.allclose(ev, glued, atol=1e-4), (
        "the stage event must be the callosum glue of the bind")


# ---------------------------------------------------------------------------
# A6: <prediction>interSentence seeds the stage-0 conceptual carrier CS_{-1}
# from the SAME discourse predictor generate_sentence uses; <prediction>none
# (the default) keeps the empty seed. Plus the PS-not-IS invariant.
# ---------------------------------------------------------------------------
def _build_mm5m_with(prediction=None, sentence_prediction=True):
    """Build MM_20M from a /tmp copy, optionally injecting
    ``<prediction>...</prediction>`` under ``<architecture>`` and
    ``<sentencePrediction>...</sentencePrediction>`` under ``<training>``.

    MM_20M.xml ships with neither knob (so the discourse layer is absent and
    ``prediction_mode`` defaults to "none"). We inject under the SAME
    ``<architecture>`` anchor the rest of this module uses, so the build path
    is identical.
    """
    import os, tempfile, warnings
    import Models, Language
    from util import init_config
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    src = os.path.join(data_dir, "MM_20M.xml")
    with open(src) as fh:
        xml = fh.read()
    assert "<prediction>" not in xml and "<sentencePrediction>" not in xml, (
        "MM_20M.xml unexpectedly already sets prediction/sentencePrediction")
    if prediction is not None:
        xml = xml.replace(
            "<architecture>",
            "<architecture>\n    <prediction>%s</prediction>" % prediction,
            1)
        assert "<prediction>%s</prediction>" % prediction in xml
    if sentence_prediction:
        # Inject inside <training> (anywhere before </training>).
        xml = xml.replace(
            "</training>",
            "      <sentencePrediction>true</sentencePrediction>\n    </training>",
            1)
        assert "<sentencePrediction>true</sentencePrediction>" in xml
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir="/tmp")
    tmp.write(xml); tmp.close()
    p = tmp.name
    init_config(path=p, defaults_path=os.path.join(data_dir, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    return m


def test_intersentence_seed_used():
    # A6 Step 1: with <prediction>interSentence and a NON-EMPTY discourse AR
    # chain, the stage-0 conceptual carrier seed (CS_{-1}, the combine's
    # ``prev_cs_content``) is sourced from ``discourse.predict_next_end_state``
    # -- non-empty and finite -- AND that single predictor is the source (we
    # count its calls). With <prediction>none the seed is EMPTY (None) and the
    # predictor is NEVER invoked from the forward seed path.
    import torch, Models

    # --- interSentence: warm chain -> non-empty predicted seed -------------
    m = _build_mm5m_with(prediction="interSentence", sentence_prediction=True)
    assert m.prediction_mode == "interSentence"
    disc = m.symbolicSpace.discourse
    assert disc is not None and disc._inter_predictor is not None, (
        "sentencePrediction=true must build the discourse inter-predictor")
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    # Cold chain: predict() returns the degenerate (1, zeros[1,D]) -- the
    # forward must NOT treat a cold/empty chain as a real seed (back-compat),
    # so the stage-0 seed is still empty on the very first forward.
    assert not disc.get_stm_chain(n=1), "chain must start empty (cold start)"
    m.forward(x)
    assert getattr(m, "_intersentence_seed_payload", None) is None, (
        "cold-start (empty AR ring) must leave the stage-0 seed EMPTY")
    # Warm the AR ring: append one finite end-state so predict() is no longer
    # cold-start. The chain stores (depth, payload[depth, D], tet); D is the
    # inter-predictor's concept width (the muxed concept event width).
    D = int(disc._inter_predictor.concept_dim)
    B = int(x.shape[0]) if torch.is_tensor(x) and x.dim() >= 1 else 1
    disc.ensure_batch(B)
    disc.observe_stm_end_state(
        [1] * B, [torch.randn(1, D) for _ in range(B)])
    assert disc.get_stm_chain(n=1), "AR ring must be non-empty after observe"
    # Count predict_next_end_state calls across the next forward and capture
    # the value it returns, so we can assert the seed is SOURCED from it.
    real_predict = disc.predict_next_end_state
    calls = {"n": 0, "last": None}
    def _counting_predict(*a, **k):
        calls["n"] += 1
        out = real_predict(*a, **k)
        calls["last"] = out
        return out
    disc.predict_next_end_state = _counting_predict
    try:
        m.forward(x)
    finally:
        disc.predict_next_end_state = real_predict
    assert calls["n"] >= 1, (
        "interSentence forward must invoke discourse.predict_next_end_state")
    seed = getattr(m, "_intersentence_seed_payload", None)
    assert seed is not None, (
        "interSentence + warm chain must produce a NON-EMPTY stage-0 seed")
    assert torch.is_tensor(seed) and torch.isfinite(seed).all(), (
        "the predicted stage-0 seed must be a finite tensor")
    assert seed.abs().sum() > 0, "the predicted seed must be non-trivial (not zeros)"
    # Sourced-from-predict: the seed payload must equal the predicted
    # payload_hat the SINGLE predictor call returned (value identity).
    assert calls["last"] is not None
    _depth_hat, payload_hat = calls["last"]
    assert torch.allclose(seed, payload_hat), (
        "the stage-0 seed must be the SAME tensor predict_next_end_state "
        "returned (one seed source, no second predictor path)")

    # --- none (default mode): empty seed, predictor never seeds -----------
    m2 = _build_mm5m_with(prediction="none", sentence_prediction=True)
    assert m2.prediction_mode == "none"
    disc2 = m2.symbolicSpace.discourse
    assert disc2 is not None and disc2._inter_predictor is not None
    Models.TheData.load("xor")
    loader2 = m2.inputSpace.data.data_loader(split="train", num_streams=4)
    items2, _ = next(iter(loader2)); x2 = m2.inputSpace.prepInput(items2)
    # Warm THIS model's ring too, so the only difference from the branch
    # above is prediction_mode -- an empty seed here proves the gate.
    D2 = int(disc2._inter_predictor.concept_dim)
    B2 = int(x2.shape[0]) if torch.is_tensor(x2) and x2.dim() >= 1 else 1
    disc2.ensure_batch(B2)
    disc2.observe_stm_end_state(
        [1] * B2, [torch.randn(1, D2) for _ in range(B2)])
    n2 = {"n": 0}
    real_predict2 = disc2.predict_next_end_state
    def _count2(*a, **k):
        n2["n"] += 1
        return real_predict2(*a, **k)
    disc2.predict_next_end_state = _count2
    try:
        m2.forward(x2)
    finally:
        disc2.predict_next_end_state = real_predict2
    assert getattr(m2, "_intersentence_seed_payload", None) is None, (
        "prediction=none must keep the stage-0 seed EMPTY (byte-identical "
        "to today's empty-seed behaviour)")
    assert n2["n"] == 0, (
        "prediction=none must NOT invoke the inter-sentence predictor from "
        "the forward seed path")


def test_parallel_ps_called_once():
    # A6 Step 4 (PS-not-IS invariant): a single MM_20M forward calls
    # PartSpace.forward EXACTLY ONCE for stage-0 ingestion, and the
    # input it receives is the REDUCED percept carrier (the post-chunk
    # word-slots, N <= nOutput) -- never the raw InputSpace char buffer
    # (nInput = 8192). The subsymbolic substrate is single-pass; PS.pi(IS) is
    # the canonical stage-0 contribution.
    import torch, Models
    m = _build("MM_20M.xml")
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    nInput = int(m.inputSpace.outputShape[0])      # raw IS buffer length
    nOutput = int(m.perceptualSpace.outputShape[0])  # reduced percept slots
    assert nOutput < nInput, (
        "MM_20M chunks IS down: nOutput (%d) must be < nInput (%d)"
        % (nOutput, nInput))

    real_fwd = m.perceptualSpace.forward
    rec = {"n": 0, "shapes": []}
    def _counting_fwd(in_sub, *a, **k):
        rec["n"] += 1
        ev = in_sub.materialize() if in_sub is not None else None
        if ev is not None and torch.is_tensor(ev):
            rec["shapes"].append(tuple(ev.shape))
        else:
            rec["shapes"].append(None)
        return real_fwd(in_sub, *a, **k)
    m.perceptualSpace.forward = _counting_fwd
    try:
        m.forward(x)
    finally:
        m.perceptualSpace.forward = real_fwd

    assert rec["n"] == 1, (
        "PartSpace.forward must be called EXACTLY once per forward "
        "(single-pass subsymbolic stage-0 ingestion), got %d" % rec["n"])
    shp = rec["shapes"][0]
    assert shp is not None and len(shp) == 3, (
        "PS input must be a 3-D [B, N, D] percept slab, got %r" % (shp,))
    N_in = int(shp[1])
    assert N_in <= nOutput, (
        "PS input N (%d) must be the REDUCED percept slots (<= nOutput=%d), "
        "never the raw IS char buffer (nInput=%d)"
        % (N_in, nOutput, nInput))
    assert N_in < nInput, (
        "PS input N (%d) must NOT be the raw IS buffer (nInput=%d)"
        % (N_in, nInput))


def test_widening_ps_pi_sized_at_embedded_percept_width():
    # A widening PartSpace (nInputDim != nOutputDim: MM_20M's 5-wide
    # raw byte event -> 1024-wide embedded percept) must size ``pi`` -- and
    # the butterfly cascade -- at the EMBEDDED percept width, and
    # ``forwardBegin`` must reshape the embedded event to that same width
    # (``_fold_width``). The legacy nInputDim sizing reshaped the embedded
    # [B, 8, 1024] slab to width 5 (8192 % 5 != 0): the ``[4, -1, 5]``
    # reshape crash behind 7 suite failures. Non-widening configs have
    # nInputDim == nOutputDim, so ``_fold_width == nInputDim`` there (the
    # legacy sizing, unchanged). (The fold is ``ps.sigma`` post Pi/Sigma
    # swap, Phase 3.)
    m = _build("MM_20M.xml")
    ps = m.perceptualSpace
    assert int(ps._fold_width) == int(ps.nOutputDim) == 1024
    assert int(ps.butterflyN) == int(ps.outputShape[0]) * 1024


def test_mm5m_grammar_builds_and_forwards():
    # Phase B (B1): the SERIAL sibling (symbolicOrder>=1, role-collapsed
    # grammar) must build + forward FINITE under the new dims and A5's threaded
    # STM, with the bounded STM staying within capacity. A5 preserved serial
    # accumulation (the per-word fold threads through begin_forward's live
    # buffer), so no serial-fold reconciliation was needed -- this gate pins it.
    import torch, Models
    m = _build("MM_20M_grammar.xml"); Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    out = m.forward(x)[2]
    assert torch.isfinite(out).all()
    cap = int(m.conceptualSpace.stm.capacity)
    assert int(m.conceptualSpace.stm.size(0)) <= cap
