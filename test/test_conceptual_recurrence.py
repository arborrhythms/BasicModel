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

def test_perfect_reconstruction_flag_parsed():
    m = _build("MM_20M.xml")
    assert hasattr(m, "perfect_reconstruction")
    assert isinstance(m.perfect_reconstruction, bool)


def test_combine_square_roundtrip_exact():
    import torch
    from Layers import ConceptualCombine
    B, D = 2, 6
    c = ConceptualCombine(content_dim=D, naive=False, sigma_pi_mode="full")
    ps, ss, cs = (torch.randn(B, D).clamp(-0.5, 0.5) for _ in range(3))
    nxt, aug = c.forward(ps, ss, cs)
    # next_cs is the D-wide conceptual carrier; aug is the 2D augment.
    assert nxt.shape == (B, D)
    assert aug.shape == (B, 2 * D)
    ps2, ss2, cs2 = c.reverse(nxt, aug)
    err = max((ps - ps2).abs().max(),
              (ss - ss2).abs().max(),
              (cs - cs2).abs().max())
    assert err < 1e-3, f"perfect (aug-threaded) round-trip err={err:.2e}"


def test_combine_square_roundtrip_exact_butterfly():
    # Butterfly is the production <sigmaPi> default; the cross-element
    # linear 2x2-LDU cascade must round-trip EXACTLY (perfect regime).
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

    def _check(D, leading):
        c = ConceptualCombine(
            content_dim=D, naive=False, sigma_pi_mode="butterfly")
        assert c.sigma_pi_mode == "butterfly"
        gl = c.layer
        N3 = 3 * D
        M = 1 << ((N3 - 1).bit_length())  # next pow2 >= 3D
        # The cascade must be sized at M, with NO padding/stripping.
        assert gl.N == M and gl.M_total == M, (
            f"D={D}: cascade not sized at M=next_pow2(3D); "
            f"N={gl.N} M_total={gl.M_total} M={M}")
        assert c.combine_padded == M
        assert c.aug_dim == M - D
        # Perturb the per-level butterfly node params OFF identity so the
        # cascade actually mixes every coordinate (not a no-op).
        with torch.no_grad():
            for p in (gl.butterfly_L, gl.butterfly_d, gl.butterfly_U):
                p.add_(0.1 * torch.randn_like(p))
        # Verify the perturbation actually moved the cascade off identity:
        # a fresh random input must NOT be returned unchanged by forward.
        probe = torch.randn(*leading, D)
        nxt_p, aug_p = c.forward(probe, probe, probe)
        with torch.no_grad():
            moved = float((nxt_p - probe).abs().max())
        assert moved > 1e-3, (
            f"D={D}: butterfly weights still at identity (moved={moved:.2e}); "
            "test would be vacuous")

        ps, ss, cs = (
            torch.randn(*leading, D).clamp(-0.5, 0.5) for _ in range(3))
        nxt, aug = c.forward(ps, ss, cs)
        assert nxt.shape == (*leading, D)
        assert aug.shape == (*leading, M - D)
        ps2, ss2, cs2 = c.reverse(nxt, aug)
        for t in (ps2, ss2, cs2):
            assert t.shape == (*leading, D)
        with torch.no_grad():
            err = float(max((ps - ps2).abs().max(),
                            (ss - ss2).abs().max(),
                            (cs - cs2).abs().max()))
        assert err == err and err != float("inf"), "non-finite round-trip err"
        assert err < 1e-3, (
            f"D={D} leading={leading}: butterfly perfect round-trip "
            f"err={err:.2e} (perturbed weights)")
        # Dropped-augment reverse on the butterfly aug width (M - D): must be
        # finite and bounded (it discards the augment, so it is lossy).
        with torch.no_grad():
            pd, sd, cd = c.reverse_dropped(nxt)
            for t in (pd, sd, cd):
                assert t.shape == (*leading, D)
                assert torch.isfinite(t).all(), "butterfly dropped non-finite"
            drop_max = float(max(pd.abs().max(), sd.abs().max(),
                                 cd.abs().max()))
        assert drop_max < 50.0, (
            f"D={D}: butterfly dropped-aug exploded: max={drop_max:.2e}")
        return err

    # 3D=18 -> M=32, 3D=24 -> M=32, 3D=48 -> M=64. Single leading batch dim.
    for D in (6, 8, 16):
        _check(D, leading=(2,))
    # Multi-leading-dim [B, T, D] locks the A4 per-position shape contract:
    # ConceptualCombine must flatten leading dims itself (B=2, T=4).
    _check(D=6, leading=(2, 4))
    _check(D=8, leading=(2, 4))


def test_combine_square_roundtrip_exact_dense_large_leading():
    # Dense (full) path at a production-scale width with a multi-leading-dim
    # [B, T, D] input -- locks the dense shape contract A4 will feed and
    # confirms the leading-dim flatten/restore is exact for the dense LDU.
    import torch
    torch.manual_seed(0)
    from Layers import ConceptualCombine
    B, T, D = 2, 4, 128
    c = ConceptualCombine(content_dim=D, naive=False, sigma_pi_mode="full")
    assert c.combine_padded == 3 * D and c.aug_dim == 2 * D
    # Perturb the LDU off identity so the round-trip is non-trivial.
    with torch.no_grad():
        c.layer.raw_L.add_(0.05 * torch.randn_like(c.layer.raw_L))
        c.layer.raw_U.add_(0.05 * torch.randn_like(c.layer.raw_U))
    ps, ss, cs = (torch.randn(B, T, D).clamp(-0.5, 0.5) for _ in range(3))
    nxt, aug = c.forward(ps, ss, cs)
    assert nxt.shape == (B, T, D)
    assert aug.shape == (B, T, 2 * D)
    ps2, ss2, cs2 = c.reverse(nxt, aug)
    for t in (ps2, ss2, cs2):
        assert t.shape == (B, T, D)
    with torch.no_grad():
        err = float(max((ps - ps2).abs().max(),
                        (ss - ss2).abs().max(),
                        (cs - cs2).abs().max()))
    assert err < 1e-3, f"dense large-D [B,T,D] round-trip err={err:.2e}"


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


def test_mm5m_perfect_reconstruction():
    # Step-5 round-trip (A4): with <perfectReconstruction>true the per-stage
    # ConceptualCombine threads its augment from forward into reverse, so the
    # concept-carrier reverse reproduces the forward CS_0 to the LDU/cascade
    # solve tolerance. MM_20M.xml ships without the flag, so inject it into a
    # /tmp copy under <architecture> and build from there.
    import os, tempfile, warnings, torch
    import Models, Language
    from util import init_config
    data_dir = os.path.join(os.path.dirname(_BIN), "data")
    src = os.path.join(data_dir, "MM_20M.xml")
    with open(src) as fh:
        xml = fh.read()
    assert "<perfectReconstruction>" not in xml, (
        "MM_20M.xml unexpectedly already sets perfectReconstruction")
    # Insert the element immediately after the <architecture> open tag.
    xml2 = xml.replace(
        "<architecture>",
        "<architecture>\n    <perfectReconstruction>true</perfectReconstruction>",
        1)
    assert "<perfectReconstruction>true</perfectReconstruction>" in xml2
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, dir="/tmp")
    tmp.write(xml2); tmp.close()
    p = tmp.name
    init_config(path=p, defaults_path=os.path.join(data_dir, "model.xml"))
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(p)
    assert m.perfect_reconstruction is True
    Models.TheData.load("xor")
    loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
    items, _ = next(iter(loader)); x = m.inputSpace.prepInput(items)
    # forward() -> (input_state, sym_vectors, pred, None); [0] is the input
    # embedding (finiteness smoke check that the forward ran).
    input_state = m.forward(x)[0]
    assert torch.isfinite(input_state).all()
    # Snapshot the forward's stage-0 advanced carrier content (CS_0 ==
    # combine[0].forward output) and the threaded per-stage augments.
    fwd_cs0 = m._combine_fwd_cs0.detach().clone()
    augs = list(m._combine_augments)
    T = len(m.body_stages)
    assert len(augs) == T and all(a is not None for a in augs), (
        "every parallel stage must thread a (non-None) augment")
    # End-to-end concept-carrier reverse on the integrated body path must be
    # FINITE (exercises _reverse_body's combine-reverse + cs.reverse chain).
    recon_sub = m._reverse_body(m._combine_last_cs_sub)
    recon_ev = recon_sub.materialize()
    assert recon_ev is not None and torch.isfinite(recon_ev).all()
    # Reconstruct CS_0 by walking the SAME per-stage combine reverses the
    # body uses, threading the EXACT forward carriers + augments back
    # (perfect regime). The terminal carrier c_{T-1} unwinds:
    # combine[t].reverse(c_t, aug_t) -> (PS_t, SS_t, c_{t-1}); the CS-stream
    # output c_{t-1} feeds stage t-1. Stop after stage 1 so the recovered
    # CS-stream IS c_0 (== forward CS_0). Uses the stored ``_combine_carriers``
    # (the exact combine outputs, in forward position order) -- NOT the
    # post-processed ``forward(x)[0]`` -- so the next_cs<->aug pairing the
    # reverse relies on is exact.
    carriers = list(m._combine_carriers)
    assert len(carriers) == T and all(c is not None for c in carriers)
    D = int(m.conceptualSpaces[0].combine.content_dim)
    carrier = carriers[-1].detach().clone()   # c_{T-1} content
    for t in reversed(range(1, T)):
        _, _, carrier = m.conceptualSpaces[t].combine.reverse(
            carrier, augs[t].detach())
    err = float((carrier.detach() - fwd_cs0[..., :D]).abs().max())
    assert torch.isfinite(torch.as_tensor(err))
    # Bound: the per-stage square combine round-trip is exact to the LDU /
    # butterfly-cascade solve tolerance; threading the real forward augments
    # adds only that numerical slack. < 1e-2 is a defensible bound.
    assert err < 1e-2, f"perfect-reconstruction CS_0 round-trip err={err:.2e}"
    # --- Integration assertions (A4 review): the propagated walk above is
    # guaranteed by A3's per-stage bijection + forward prev_cs threading, so
    # on its own it adds little over A3. These assertions verify the FORWARD
    # WIRING itself, which is what _reverse_body actually relies on (it
    # re-inverts each stage against the stashed carrier, not a propagated one):
    #   (a) chained consistency -- combine[t].reverse(c_t, aug_t) CS-stream
    #       must equal the prior stashed carrier c_{t-1}; this is the assertion
    #       that fails if a future change breaks the forward c_{t-1} -> c_t
    #       prev_cs threading.
    #   (b) the t=0 CS-stream must recover the empty CS_-1 seed (~0).
    #   (c) the t=0 PS-stream (the pi-encoded input, alpha_ps live only at t=0)
    #       must be NON-trivial -- it is the leg that decodes back to the input.
    with torch.no_grad():
        for t in range(1, T):
            _, _, cs_rec = m.conceptualSpaces[t].combine.reverse(
                carriers[t].detach(), augs[t].detach())
            chain_err = float((cs_rec - carriers[t - 1].detach()).abs().max())
            assert chain_err < 1e-2, (
                f"stage {t}: combine.reverse(c_t) CS-stream != c_(t-1) "
                f"-- forward prev_cs threading broken (err={chain_err:.2e})")
        ps0_rec, _, cs0_rec = m.conceptualSpaces[0].combine.reverse(
            carriers[0].detach(), augs[0].detach())
        seed_mag = float(cs0_rec.abs().max())
        ps0_mag = float(ps0_rec.abs().max())
    assert seed_mag < 1e-2, (
        f"stage 0 CS-stream must recover the empty CS_-1 seed (~0), "
        f"got max={seed_mag:.2e}")
    assert ps0_mag > 1e-6, (
        f"stage 0 PS-stream (pi-encoded input) must be non-trivial, "
        f"got max={ps0_mag:.2e}")


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
    (``_idea_buffer`` when a WordSubSpace is attached, ``_fallback_buffer``
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


def test_combine_dropped_aug_exact_on_rank():
    # Dropped-augment reverse: aug is treated as the structured zero-pad.
    # It does NOT recover the inputs exactly (the 2D augment is discarded),
    # but the result must be FINITE and BOUNDED (no explosion). With a
    # near-identity init the surviving rank-D subspace is recovered well;
    # we assert finiteness + a generous bound rather than exact equality.
    import torch
    torch.manual_seed(0)
    from Layers import ConceptualCombine
    B, D = 2, 6
    c = ConceptualCombine(content_dim=D, naive=False, sigma_pi_mode="full")
    # Perturb the LDU off the identity so the dropped-aug reverse is a real
    # (non-trivial) projection, not just the identity.
    with torch.no_grad():
        c.layer.raw_L.add_(0.1 * torch.randn_like(c.layer.raw_L))
        c.layer.raw_U.add_(0.1 * torch.randn_like(c.layer.raw_U))
    ps, ss, cs = (torch.randn(B, D).clamp(-0.5, 0.5) for _ in range(3))
    nxt, aug = c.forward(ps, ss, cs)
    ps2, ss2, cs2 = c.reverse_dropped(nxt)
    for t in (ps2, ss2, cs2):
        assert t.shape == (B, D)
        assert torch.isfinite(t).all(), "dropped-aug reverse produced non-finite"
    # Bounded: the dropped-aug reconstruction stays on the same order of
    # magnitude as the inputs (clamped to <=0.5); assert it does not blow up.
    recon_max = max(ps2.abs().max(), ss2.abs().max(), cs2.abs().max())
    assert recon_max < 50.0, f"dropped-aug reverse exploded: max={recon_max:.2e}"
    # Sanity: dropping the 2D augment is lossy, so the error is generally
    # NON-zero (this is the approximate regime, not the exact one).
    err = max((ps - ps2).abs().max(),
              (ss - ss2).abs().max(),
              (cs - cs2).abs().max())
    assert torch.isfinite(torch.as_tensor(err))


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
    ``prediction_mode`` defaults to "none"). We inject under the SAME anchors
    ``test_mm5m_perfect_reconstruction`` uses for the perfectReconstruction
    flag, so the build path is identical to the rest of this module.
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
    disc = m.wordSubSpace.discourse
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
    disc2 = m2.wordSubSpace.discourse
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
    # PerceptualSpace.forward EXACTLY ONCE for stage-0 ingestion, and the
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
        "PerceptualSpace.forward must be called EXACTLY once per forward "
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


def test_mm5m_grammar_builds_and_forwards():
    # Phase B (B1): the SERIAL sibling (conceptualMode=serial, role-collapsed
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
