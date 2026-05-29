"""Stage 10: Per-stage CS pipeline + sigma migration.

Stage 10 of doc/plans/2026-05-27-perceptstore-meta-taxonomy-reentrancy.md.

Post-Stage-10 contract:

  * ``PerceptualSpace`` drops ``self.sigma``: PS is **pi-only**.
    ``PerceptualSpace.forward(x)`` body becomes ``return self.pi(x.materialize())``.
  * Each ``ConceptualSpace`` instance in ``self.conceptualSpaces``
    (length ``<conceptualOrder>``) gains **two owned SigmaLayers**:
      - ``self.sigma_in``  (incoming-contribution fold; fires in both modes)
      - ``self.sigma_cs``  (PARALLEL-mode residual-CS iteration kernel)
    Weights are Ramsified across stages — each stage has its own
    Parameter instances.
  * PARALLEL forward walks stages 0..T-1; each stage's ``sigma_cs`` is
    called with the prior stage's CS output as residual.
  * SERIAL forward per word walks stages 0..T-1; each stage's
    ``sigma_in`` folds the incoming. The per-stage ``SyntacticLayer``
    compose-round dispatch (replaces ``sigma_cs[k]`` in SERIAL mode) is
    a follow-up; SERIAL sigma_cs is allowed to remain dormant on this
    pass.
  * Reverse symmetric: walk stages T-1..0 with each stage's
    ``sigma_in.reverse`` / ``sigma_cs.reverse`` (PARALLEL).

The targeted gate uses ``MM_xor.xml`` (``<conceptualOrder>3</...>``,
``<conceptualMode>parallel</...>``) so the 3-stage Ramsified pipeline
is exercised end-to-end.
"""

import inspect
import os
import sys
import unittest
import warnings

import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_BIN = os.path.join(_PROJECT, "bin")
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)

import Models
import Language
from Layers import SigmaLayer
from util import init_config

_DATA_DIR = os.path.join(_PROJECT, "data")
_CONFIG_PARALLEL = os.path.join(_DATA_DIR, "MM_xor.xml")
_DEFAULTS = os.path.join(_DATA_DIR, "model.xml")


def _make_parallel_model():
    """Build the MM_xor model (3-stage parallel pipeline)."""
    init_config(path=_CONFIG_PARALLEL, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG_PARALLEL)
    Models.TheData.load("xor")
    m.eval()
    return m


def _one_input(model):
    loader = model.inputSpace.data.data_loader(
        split="train", num_streams=1)
    inp_items, _ = next(iter(loader))
    return model.inputSpace.prepInput(inp_items)


# ---------------------------------------------------------------------------
# 1. Structural — PerceptualSpace.sigma is retired
# ---------------------------------------------------------------------------


class TestPerceptualSpaceSigmaRetired(unittest.TestCase):
    """Stage 10: ``PerceptualSpace`` no longer owns ``self.sigma``."""

    def test_perceptual_space_has_no_sigma_attribute(self):
        m = _make_parallel_model()
        ps = m.perceptualSpace
        self.assertFalse(
            hasattr(ps, "sigma"),
            "Stage 10: PerceptualSpace.sigma must be retired; PS is "
            "pi-only. The sigma half migrates to ConceptualSpace per "
            "stage.")

    def test_perceptual_space_still_has_pi(self):
        m = _make_parallel_model()
        ps = m.perceptualSpace
        self.assertTrue(
            hasattr(ps, "pi"),
            "Stage 10: PerceptualSpace.pi survives — PS does pi-only.")


# ---------------------------------------------------------------------------
# 2. Structural — per-stage owned sigma_in + sigma_cs
# ---------------------------------------------------------------------------


class TestConceptualSpacesIsModuleList(unittest.TestCase):
    """``self.conceptualSpaces`` is a ``nn.ModuleList`` whose length
    equals ``<conceptualOrder>``."""

    def test_conceptualSpaces_is_module_list(self):
        m = _make_parallel_model()
        self.assertIsInstance(
            m.conceptualSpaces, torch.nn.ModuleList,
            "self.conceptualSpaces must be an nn.ModuleList.")

    def test_conceptualSpaces_length_equals_order(self):
        m = _make_parallel_model()
        T = int(m.conceptualOrder)
        # _create_per_stage's T = max(1, conceptualOrder)
        T_expected = max(1, T)
        self.assertEqual(
            len(m.conceptualSpaces), T_expected,
            f"self.conceptualSpaces length must equal conceptualOrder "
            f"({T_expected}); got {len(m.conceptualSpaces)}.")

    def test_mm_xor_has_three_stages(self):
        m = _make_parallel_model()
        self.assertEqual(
            len(m.conceptualSpaces), 3,
            "MM_xor.xml sets <conceptualOrder>3</...> so three CS stages "
            "must be constructed.")


class TestEachStageHasSigmaIn(unittest.TestCase):
    """Stage 10: each ``ConceptualSpace[k]`` has its own owned
    ``sigma_in`` (SigmaLayer)."""

    def test_each_stage_has_sigma_in(self):
        m = _make_parallel_model()
        for k, cs in enumerate(m.conceptualSpaces):
            self.assertTrue(
                hasattr(cs, "sigma_in"),
                f"ConceptualSpace[{k}].sigma_in must exist (Stage 10).")
            self.assertIsInstance(
                cs.sigma_in, SigmaLayer,
                f"ConceptualSpace[{k}].sigma_in must be a SigmaLayer; "
                f"got {type(cs.sigma_in).__name__}.")


class TestEachStageHasSigmaCS(unittest.TestCase):
    """Stage 10: each ``ConceptualSpace[k]`` has its own owned
    ``sigma_cs`` (SigmaLayer)."""

    def test_each_stage_has_sigma_cs(self):
        m = _make_parallel_model()
        for k, cs in enumerate(m.conceptualSpaces):
            self.assertTrue(
                hasattr(cs, "sigma_cs"),
                f"ConceptualSpace[{k}].sigma_cs must exist (Stage 10).")
            self.assertIsInstance(
                cs.sigma_cs, SigmaLayer,
                f"ConceptualSpace[{k}].sigma_cs must be a SigmaLayer; "
                f"got {type(cs.sigma_cs).__name__}.")


class TestSigmaParametersAreRamsified(unittest.TestCase):
    """Stage 10: the sigma_in / sigma_cs SigmaLayer Parameter instances
    differ across stages (Ramsified — each stage learns its own fold)."""

    def test_sigma_in_parameters_differ_across_stages(self):
        m = _make_parallel_model()
        sigma_in_ids = [
            id(p) for cs in m.conceptualSpaces
            for p in cs.sigma_in.parameters()
        ]
        # All parameter ids must be unique
        self.assertEqual(
            len(sigma_in_ids), len(set(sigma_in_ids)),
            "Each stage's sigma_in must own a distinct Parameter "
            "block (Ramsified).")

    def test_sigma_cs_parameters_differ_across_stages(self):
        m = _make_parallel_model()
        sigma_cs_ids = [
            id(p) for cs in m.conceptualSpaces
            for p in cs.sigma_cs.parameters()
        ]
        self.assertEqual(
            len(sigma_cs_ids), len(set(sigma_cs_ids)),
            "Each stage's sigma_cs must own a distinct Parameter block "
            "(Ramsified).")

    def test_sigma_in_and_sigma_cs_have_disjoint_params(self):
        """sigma_in and sigma_cs within the SAME stage are also
        independent layers — their Parameter ids must not overlap."""
        m = _make_parallel_model()
        for k, cs in enumerate(m.conceptualSpaces):
            ids_in = {id(p) for p in cs.sigma_in.parameters()}
            ids_cs = {id(p) for p in cs.sigma_cs.parameters()}
            self.assertEqual(
                ids_in & ids_cs, set(),
                f"Stage {k}: sigma_in and sigma_cs must own disjoint "
                f"Parameter sets.")


# ---------------------------------------------------------------------------
# 3. Behavioural — forward / reverse exercise the per-stage sigmas
# ---------------------------------------------------------------------------


class TestForwardBypassesSigmaInPerStage(unittest.TestCase):
    """Clean-stack STM (2026-05-29): PARALLEL forward DOES NOT invoke
    ``sigma_in.forward`` at any stage.

    The Stage-10 ``sigma_in(combined) + sigma_cs(prev)`` additive
    composition was replaced with per-stage tier attribution::

        stage 0      STM = primary    (PS event)
        stage k > 0  STM = sym        (SS event)

    sigma_in is bypassed in forward by design. It remains live on
    the *reverse* path (``ConceptualSpace.reverse`` still calls
    ``self.sigma_in.reverse`` to undo any prior-pass folds), but the
    training-time forward equation no longer touches it.
    """

    def test_parallel_forward_does_not_call_sigma_in(self):
        from unittest import mock
        m = _make_parallel_model()
        # Wrap each sigma_in.forward so we can detect invocations.
        spies = []
        for cs in m.conceptualSpaces:
            spy = mock.MagicMock(wraps=cs.sigma_in.forward)
            cs.sigma_in.forward = spy
            spies.append(spy)
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(x)
        for k, spy in enumerate(spies):
            self.assertEqual(
                spy.call_count, 0,
                f"Stage {k} sigma_in.forward must NOT be called during "
                f"clean-stack PARALLEL forward (folded = primary at "
                f"stage 0, folded = sym at k > 0).")


class TestForwardCallsSigmaCSForResidualStages(unittest.TestCase):
    """PARALLEL forward fires ``sigma_cs[k]`` for stages k > 0 (the
    residual lift from prior stage's CS output)."""

    def test_parallel_forward_calls_sigma_cs_for_residual_stages(self):
        from unittest import mock
        m = _make_parallel_model()
        spies = []
        for cs in m.conceptualSpaces:
            spy = mock.MagicMock(wraps=cs.sigma_cs.forward)
            cs.sigma_cs.forward = spy
            spies.append(spy)
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(x)
        # Stage 0 has no prior CS state, so sigma_cs[0] is not called
        # in the residual sense. Stages k > 0 must see sigma_cs fire.
        for k in range(1, len(spies)):
            self.assertGreater(
                spies[k].call_count, 0,
                f"Stage {k} sigma_cs.forward must be called at least "
                f"once during PARALLEL forward (residual-CS lift).")


# ---------------------------------------------------------------------------
# 4. Sigma_in shape: matches per-stage concept_dim
# ---------------------------------------------------------------------------


class TestSigmaShapes(unittest.TestCase):
    """sigma_in and sigma_cs are sized ``concept_dim → concept_dim``."""

    def test_sigma_in_io_dims_equal_concept_dim(self):
        m = _make_parallel_model()
        for k, cs in enumerate(m.conceptualSpaces):
            self.assertEqual(
                int(cs.sigma_in.nInput), int(cs.concept_dim),
                f"Stage {k} sigma_in.nInput must equal CS.concept_dim.")
            self.assertEqual(
                int(cs.sigma_in.nOutput), int(cs.concept_dim),
                f"Stage {k} sigma_in.nOutput must equal CS.concept_dim.")

    def test_sigma_cs_io_dims_equal_concept_dim(self):
        m = _make_parallel_model()
        for k, cs in enumerate(m.conceptualSpaces):
            self.assertEqual(
                int(cs.sigma_cs.nInput), int(cs.concept_dim),
                f"Stage {k} sigma_cs.nInput must equal CS.concept_dim.")
            self.assertEqual(
                int(cs.sigma_cs.nOutput), int(cs.concept_dim),
                f"Stage {k} sigma_cs.nOutput must equal CS.concept_dim.")


# ---------------------------------------------------------------------------
# 5. PARALLEL forward + reverse smoke
# ---------------------------------------------------------------------------


class TestParallelForwardReverseSmoke(unittest.TestCase):
    """End-to-end smoke: PARALLEL forward runs through the 3-stage
    pipeline and produces a valid head output; reverse runs back through
    all 3 stages without crashing."""

    def test_forward_runs(self):
        m = _make_parallel_model()
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                y = m.forward(x)
        self.assertIsNotNone(
            y, "PARALLEL forward must return a non-None output.")

    def test_reverse_runs_through_all_stages(self):
        """Reverse walks stages T-1..0 calling each stage's
        sigma_in.reverse without raising."""
        from unittest import mock
        m = _make_parallel_model()
        # Spy on sigma_in.reverse for all stages.
        spies_in = []
        for cs in m.conceptualSpaces:
            spy = mock.MagicMock(wraps=cs.sigma_in.reverse)
            cs.sigma_in.reverse = spy
            spies_in.append(spy)
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(x)
                # Stamp the terminal STM slot onto the CS subspace
                # (mirrors what the live caller in BasicModel._run_pipeline
                # does before _run_pipeline_rev_from_concepts).
                snap = m.conceptualSpace.stm.snapshot()
                self.assertIsNotNone(
                    snap,
                    "STM snapshot must be available after forward.")
                terminal_idea = snap[:, -1:, :]
                cs = m.conceptualSpace
                cs.subspace.set_event(terminal_idea)
                # Reverse from the terminal C-tier state
                rec = m._run_pipeline_rev_from_concepts(cs.subspace)
        self.assertIsNotNone(
            rec,
            "Reverse pipeline from concepts must return a non-None "
            "subspace.")
        # Each stage's sigma_in.reverse must have been called.
        for k, spy in enumerate(spies_in):
            self.assertGreater(
                spy.call_count, 0,
                f"Stage {k} sigma_in.reverse must be called during "
                f"reverse pipeline.")


# ---------------------------------------------------------------------------
# 6. Forward composition assertion: CS.forward applies sigma_in
# ---------------------------------------------------------------------------


class TestCSForwardBypassesSigmaIn(unittest.TestCase):
    """Clean-stack STM (2026-05-29): ``ConceptualSpace.forward`` does
    NOT apply ``self.sigma_in`` to the materialized PS+SS combination.

    The per-stage tier attribution (``folded = primary`` at stage 0,
    ``folded = sym`` at k > 0) writes straight to the STM slot ring
    with no sigma_in fold on the forward path.
    """

    def test_cs_forward_does_not_invoke_sigma_in_forward(self):
        from unittest import mock
        m = _make_parallel_model()
        # Use the FIRST CS stage. Stage 0's ``folded = primary``
        # invariant is the cleanest target for "CS.forward bypasses
        # sigma_in" -- higher stages use ``sym`` which depends on the
        # word_subspace plumbing.
        cs = m.conceptualSpaces[0]
        ps = m.perceptualSpace
        loader = m.inputSpace.data.data_loader(
            split="train", num_streams=1)
        inp_items, _ = next(iter(loader))
        x_input = m.inputSpace.prepInput(inp_items)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub = m.inputSpace.forward(x_input)
                ps_sub = ps.forward(in_sub)
                # Spy sigma_in.forward then call cs.forward.
                spy = mock.MagicMock(wraps=cs.sigma_in.forward)
                cs.sigma_in.forward = spy
                cs.stm.clear()
                cs.forward(ps_sub)
        self.assertEqual(
            spy.call_count, 0,
            "ConceptualSpace.forward must NOT invoke self.sigma_in.forward "
            "under clean-stack STM (folded = primary at stage 0; the "
            "Stage-10 sigma_in fold was removed by the 2026-05-29 "
            "experiment).")


# ---------------------------------------------------------------------------
# 7. PARALLEL reverse roundtrip exactness (Gap 1)
# ---------------------------------------------------------------------------


class TestParallelReverseRoundtripExactness(unittest.TestCase):
    """Stage 10 acceptance: ``PARALLEL forward + reverse roundtrip is
    exact across the full 3-stage pipeline (modulo LDU precision).``

    The per-stage forward equation for ``k > 0`` is

        CS_t1[k] = sigma_in[k](contribution[k]) + sigma_cs[k](CS_t0[k])

    where ``CS_t0[k] == CS_t1[k-1]`` (the prior stage's post-merge CS
    event). The matching reverse subtracts the EXACT same
    ``sigma_cs[k](CS_t0[k])`` term and applies ``sigma_in[k].reverse``
    to recover the contribution:

        contribution[k] = sigma_in[k].reverse(
            CS_t1[k] - sigma_cs[k](CS_t0[k]))

    The forward path stashes ``CS_t0[k]`` on
    ``cs._prev_cs_event_cache``; ``ConceptualSpace.reverse`` reads
    that cache to recompute the lift and subtract it.

    Tolerance: ``atol=1e-3`` per the gap description. The LDU
    butterfly + nonlinear (atanh/tanh) sigma roundtrip precision is
    sub-1e-6 in practice (per the diagnostic probes); 1e-3 absorbs
    cumulative drift across 3 stages plus any merge/normalize residue.
    """

    def test_per_stage_equation_closes_with_cache(self):
        """Stage k > 0: with ``cs._prev_cs_event_cache`` populated by the
        forward path, ``cs.reverse`` recovers the contribution to
        ``cs.forward`` to LDU precision.

        Drives a single CS stage directly (no merge, no SS contribution,
        no STM mean reduction) so the per-stage equation is the only
        thing tested. The full ``_forward_body`` path adds SS and merge
        layers that are independently verified elsewhere; this test
        isolates the Stage 10 PARALLEL residual subtraction in reverse.
        """
        from Spaces import SubSpace
        m = _make_parallel_model()
        # Stage 1 (k > 0) — the residual lift fires here.
        cs = m.conceptualSpaces[1]
        D = int(cs.concept_dim)
        N = int(cs.inputShape[0])  # 4 after stage-0 merge in MM_xor
        B = 1
        # Random contribution + prior CS event, both in [-0.5, 0.5] so
        # the atanh inside sigma_in stays well inside its domain.
        torch.manual_seed(0)
        contribution = torch.randn(B, N, D).clamp(-0.5, 0.5)
        prev_cs_event = torch.randn(B, N, D).clamp(-0.5, 0.5)
        # Forward equation: CS_t1 = sigma_in(contrib) + sigma_cs(prev)
        with torch.no_grad():
            fold = cs.sigma_in.forward(contribution)
            lift = cs.sigma_cs.forward(prev_cs_event)
            CS_t1 = fold + lift
            # Stash the prior event on the CS instance (mirrors what
            # ``BasicModel._forward_body`` does after the residual lift
            # fires).
            cs._prev_cs_event_cache = prev_cs_event.detach()
            # Reverse: cs.reverse subtracts sigma_cs(prev) then applies
            # sigma_in.reverse.
            sub = SubSpace(
                inputShape=(N, D), outputShape=(N, D),
                nInputDim=1, nOutputDim=1)
            sub.set_event(CS_t1)
            rec = cs.reverse(sub)
            rec_ev = rec.materialize()
        # The recovered tensor must equal the original contribution to
        # LDU precision. Without the residual subtraction the error
        # would be on the order of |sigma_cs(prev)| ~ 0.1+.
        err_max = (contribution - rec_ev).abs().max().item()
        self.assertLess(
            err_max, 1e-3,
            f"PARALLEL stage-k>0 reverse equation must close to LDU "
            f"precision (atol=1e-3); got max abs error {err_max:g}.")

    def test_per_stage_reverse_degenerates_to_sigma_in_only_without_cache(self):
        """When ``cs._prev_cs_event_cache is None`` (stage 0, or a
        degenerate fallback), reverse omits the subtraction and just
        applies ``sigma_in.reverse``.

        This is the pre-Stage-10-fix behaviour preserved for stage 0
        (which has no prior CS state) — verifies the cache wiring is
        gated on cache presence, not blindly invoked.
        """
        from Spaces import SubSpace
        m = _make_parallel_model()
        cs = m.conceptualSpaces[0]  # k == 0, no residual in forward
        D = int(cs.concept_dim)
        N = int(cs.inputShape[0])
        B = 1
        torch.manual_seed(1)
        contribution = torch.randn(B, N, D).clamp(-0.5, 0.5)
        with torch.no_grad():
            # Forward without residual lift (stage 0 path).
            CS_t1 = cs.sigma_in.forward(contribution)
            cs._prev_cs_event_cache = None
            sub = SubSpace(
                inputShape=(N, D), outputShape=(N, D),
                nInputDim=1, nOutputDim=1)
            sub.set_event(CS_t1)
            rec = cs.reverse(sub)
            rec_ev = rec.materialize()
        err_max = (contribution - rec_ev).abs().max().item()
        self.assertLess(
            err_max, 1e-3,
            f"Stage-0 (no cache) reverse must still close via "
            f"sigma_in.reverse alone; got max abs error {err_max:g}.")

    def test_forward_body_populates_caches_for_residual_stages(self):
        """End-to-end: after running ``_forward_body`` on a real input,
        ``cs._prev_cs_event_cache`` must be populated for stages k > 0
        (where the residual lift fires) and ``None`` for stage 0.

        The cache content at stage k must equal the prior stage's
        post-merge CS event (the ``prev_cs_for_residual`` snapshot in
        ``_forward_body``).
        """
        m = _make_parallel_model()
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub = m.inputSpace.forward(x)
                m._forward_body(in_sub)
        # Stage 0: no residual lift, cache should be None.
        cs0 = m.conceptualSpaces[0]
        self.assertIsNone(
            cs0._prev_cs_event_cache,
            "Stage 0 has no prior CS state — cache must stay None "
            "after forward.")
        # Stages 1..T-1: residual lift fires, cache must be populated.
        for k in range(1, len(m.conceptualSpaces)):
            cs = m.conceptualSpaces[k]
            self.assertIsNotNone(
                cs._prev_cs_event_cache,
                f"Stage {k} residual lift must have stashed the prior "
                f"CS event on cs._prev_cs_event_cache during forward.")
            # The cached event's last dim must match concept_dim.
            self.assertEqual(
                int(cs._prev_cs_event_cache.shape[-1]),
                int(cs.concept_dim),
                f"Stage {k} cache last-dim must equal concept_dim "
                f"({cs.concept_dim}); got "
                f"{cs._prev_cs_event_cache.shape[-1]}.")

    def test_forward_body_clears_stale_cache_at_start(self):
        """Between successive forward calls, ``_forward_body`` must
        clear stale ``_prev_cs_event_cache`` so the next call's reverse
        doesn't subtract a leak from the prior call.

        Pollute the cache with a recognisable tensor before forward;
        verify it's reset by the time the next stage-loop iteration
        runs by checking the *content* changed (the new cache won't
        equal the pollution).
        """
        m = _make_parallel_model()
        # Pollute every stage's cache with a sentinel tensor.
        for cs in m.conceptualSpaces:
            cs._prev_cs_event_cache = torch.full(
                (1, 1, int(cs.concept_dim)), -99.0)
        x = _one_input(m)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                in_sub = m.inputSpace.forward(x)
                m._forward_body(in_sub)
        # After forward, stage-0 cache must be None (cleared, no
        # residual at stage 0 to repopulate it).
        cs0 = m.conceptualSpaces[0]
        self.assertIsNone(
            cs0._prev_cs_event_cache,
            "Stage 0 cache must be cleared at start of forward "
            "(sentinel pollution must NOT leak through).")
        # Stages 1..T-1: cache repopulated, must NOT equal sentinel.
        for k in range(1, len(m.conceptualSpaces)):
            cs = m.conceptualSpaces[k]
            self.assertIsNotNone(
                cs._prev_cs_event_cache,
                f"Stage {k} cache must be repopulated by forward.")
            # The cached tensor must not be the -99 sentinel.
            self.assertFalse(
                torch.all(cs._prev_cs_event_cache == -99.0).item(),
                f"Stage {k} cache must be the fresh prior-stage event, "
                f"not the leaked -99 sentinel pollution.")


# ---------------------------------------------------------------------------
# 8. Identity init: conceptualOrder=1 (Gap 3)
# ---------------------------------------------------------------------------


_CONFIG_IDENTITY = os.path.join(_DATA_DIR, "idempotent.xml")


def _make_identity_model():
    """Build the idempotent model (single-stage, ``<conceptualOrder>1</...>``)."""
    init_config(path=_CONFIG_IDENTITY, defaults_path=_DEFAULTS)
    Language.TheGrammar._configured = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        m, _ = Models.BasicModel.from_config(_CONFIG_IDENTITY)
    m.eval()
    return m


class TestIdentityInitSingleStage(unittest.TestCase):
    """Stage 10 spec required test: ``Identity init: at
    <conceptualOrder>1</...>, only stage 0 is active; pipeline reduces
    to the single-stage substrate behavior.``"""

    def test_conceptual_order_one_gives_single_stage(self):
        """``<conceptualOrder>1</...>`` constructs exactly one CS stage."""
        m = _make_identity_model()
        self.assertEqual(
            len(m.conceptualSpaces), 1,
            "At <conceptualOrder>1</...> the per-stage ModuleList must "
            "have length 1 (only stage 0 active).")

    def test_single_stage_has_sigma_in_and_sigma_cs(self):
        """The lone stage still owns its sigma_in / sigma_cs pair, but
        sigma_cs is dormant in PARALLEL (no prior stage to lift from)."""
        m = _make_identity_model()
        cs = m.conceptualSpaces[0]
        self.assertTrue(hasattr(cs, "sigma_in"),
                        "Stage 0 must still own sigma_in.")
        self.assertTrue(hasattr(cs, "sigma_cs"),
                        "Stage 0 must still own sigma_cs (constructed "
                        "unconditionally per Stage 10).")

    def test_single_stage_forward_reverse_runs(self):
        """End-to-end forward + reverse on the single-stage model
        completes without error. The pipeline reduces to the
        single-stage substrate path (no residual lift, no merge cascade
        between stages)."""
        m = _make_identity_model()
        try:
            loader = m.inputSpace.data.data_loader(
                split="train", num_streams=1)
            inp_items, _ = next(iter(loader))
            x_input = m.inputSpace.prepInput(inp_items)
        except Exception as e:
            self.skipTest(f"idempotent data loader unavailable: {e}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                y = m.forward(x_input)
        self.assertIsNotNone(
            y,
            "Single-stage (conceptualOrder=1) forward must return a "
            "non-None output.")

    def test_single_stage_no_residual_lift_cache(self):
        """With only one stage there is no prior CS state to lift; the
        forward path must NOT populate ``cs._prev_cs_event_cache``
        (stage 0 is the only stage and its k == 0)."""
        m = _make_identity_model()
        try:
            loader = m.inputSpace.data.data_loader(
                split="train", num_streams=1)
            inp_items, _ = next(iter(loader))
            x_input = m.inputSpace.prepInput(inp_items)
        except Exception as e:
            self.skipTest(f"idempotent data loader unavailable: {e}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            with torch.no_grad():
                m.forward(x_input)
        cs0 = m.conceptualSpaces[0]
        self.assertIsNone(
            cs0._prev_cs_event_cache,
            "Single-stage pipeline: stage 0 has no prior CS state, so "
            "the residual-lift cache must stay None.")


# ---------------------------------------------------------------------------
# 9. SERIAL deferral marker (Gap 4)
# ---------------------------------------------------------------------------


class TestSerialPerStageComposeDeferral(unittest.TestCase):
    """Marker test for the deferred SERIAL-mode work.

    The Stage 10 plan specified:
        ``SERIAL forward per word walks stages 0..T-1;
        SyntacticLayer.compose is called once per stage
        (replaces sigma_cs[k]).``

    The implementer's DONE_WITH_CONCERNS deferred the per-stage
    ``SyntacticLayer.compose`` dispatch to a follow-up; SERIAL
    ``sigma_cs`` remains dormant on the current pass. This skip-marked
    test makes the gap visible in test reports.
    """

    @unittest.skip(
        "SERIAL per-stage SyntacticLayer.compose dispatch deferred to "
        "follow-up.")
    def test_serial_per_stage_compose_dispatch(self):
        self.fail(
            "SERIAL per-stage SyntacticLayer.compose dispatch not yet "
            "implemented; tracked as follow-up to Stage 10.")


if __name__ == "__main__":
    unittest.main()
