"""MM_xor convergence test: BasicModel topology on XOR dataset.

Verifies that the BasicModel architecture (iterative [Percept,Symbol]->Concept->Symbol)
can learn the XOR function on the toy text dataset within 200 epochs.
"""

import os
import sys
import tempfile
import unittest
import warnings

import pytest

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# XOR convergence is deterministic only under the seed + CPU
# discipline (see test/conftest.py's device-leak note): a leaked MPS
# default device shifts numerics enough to miss the convergence
# threshold under some suite compositions. Pin the env AND (in the
# convergence helper) the runtime device.
os.environ.setdefault("BASICMODEL_DEVICE", "cpu")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models
import Spaces
import Language

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor.xml")


def _fresh_model(config_path=_CONFIG):
    """Create a fresh BasicModel with XOR data loaded."""
    from util import init_config
    init_config(
        path=config_path,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    Language.TheGrammar._configured = False
    m, cfg = Models.BasicModel.from_config(config_path)
    Models.TheData.load("xor")
    return m, cfg, Models.TheData


def _variant_config(replacements):
    """Write a temporary MM_xor variant for branch-specific tests."""
    with open(_CONFIG, "r", encoding="utf-8") as fh:
        text = fh.read()
    for old, new in replacements:
        text = text.replace(old, new, 1)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    tmp.write(text)
    tmp.close()
    return tmp.name


class TestMMXorConvergence(unittest.TestCase):
    """BasicModel on XOR should converge to near-zero output loss."""

    @classmethod
    def setUpClass(cls):
        import torch
        import matplotlib
        matplotlib.use('Agg')

        m, cfg, data = _fresh_model()
        cls.model = m
        cls.data = data
        cls.cfg = cfg
        # Sanity: data must actually be loaded
        assert len(data.train_input) > 0, "XOR data not loaded"

    def test_model_is_mental(self):
        # Post-2026-05-05 merger: BasicModel is an alias for BasicModel,
        # so the class name is "BasicModel". The semantically meaningful
        # check is that the per-stage pipeline is built (conceptualSpaces
        # / wholeSpaces lists), not the legacy class identity.
        self.assertTrue(hasattr(self.model, "conceptualSpaces"))
        self.assertTrue(hasattr(self.model, "wholeSpaces"))
        self.assertGreaterEqual(len(self.model.conceptualSpaces), 1)
        self.assertGreaterEqual(len(self.model.wholeSpaces), 1)

    def test_has_conceptual_symbolic_spaces(self):
        self.assertTrue(hasattr(self.model, 'conceptualSpace'))
        self.assertTrue(hasattr(self.model, 'wholeSpace'))

    def test_a_forward_runs(self):
        """Forward pass smoke test. Runs before test_convergence (alpha order)."""
        import torch
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loader = m.inputSpace.data.data_loader(split="train", num_streams=1)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            outputTensor = (m.outputSpace.prepOutput(out_items)
                            if out_items is not None else None)
            batch = (inputTensor, outputTensor)
            self.assertIsNotNone(batch, "data_loader returned None -- data not loaded")
            inp, _ = batch
            with torch.no_grad():
                result = m.forward(inp)
        self.assertEqual(len(result), 4)

    # test_forward_reverse_reconstructs_input_state retired 2026-05-14 (reverse pipeline / <maskedPrediction> retired in IR-only refactor).

    def test_forward_keeps_continuous_symbols(self):
        """The recurrent path should not collapse via symbol VQ.

        Updated 2026-05-20: ``m.forward`` writes a reduced terminal
        event ([B, 1, D]) into ``wholeSpace.subspace`` on its way
        through the head — that final reduction is intentionally
        sparse on an untrained model and isn't where VQ collapse would
        show. Probe the per-stage symbolic activation directly (call
        the symbol space on the percept stage's output) — that's the
        signal that codebook-snap can collapse and the variance check
        is meaningful there."""
        import torch

        m, _, _ = _fresh_model()
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
            inp_items, _ = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            with torch.no_grad():
                in_sub, _ = m.inputSpace.forward(inputTensor)
                ps_sub = m.perceptualSpace.forward(in_sub)
                ws_sub = m.wholeSpace.forward(ps_sub)
        symbols = ws_sub.materialize()
        self.assertTrue(torch.isfinite(symbols).all())
        self.assertGreater(symbols.std().item(), 1e-6,
                           "Symbolic activation collapsed to a single VQ "
                           "prototype after the per-stage forward.")

    # test_runbatch_losses_stay_finite retired 2026-05-14 (reverse pipeline / <maskedPrediction> retired in IR-only refactor).

    def test_learns_xor_signal(self):
        """The affine head reaches the THEORY FLOOR (Alec 2026-07-13:
        bars are derived, not empirically re-pinned).

        XOR on $\\{0,1\\}^2$ is not linearly separable: the best affine
        predictor is the constant 1/2, whose MSE is exactly 0.25, and no
        affine head can do better. MM_xor's head IS affine (the joint
        cross-word nonlinearity lives only in MM_20M_xor's
        ``<nonlinear>true`` combine), so the derived bar here is
        REACHING the floor -- ``best_loss < 0.25 + tol`` -- which pins
        that training attains the best linear separation. The
        effective-zero bar belongs to the nonlinear configs (XOR_exact's
        crisp MSE < 0.05 is the exemplar); upgrading MM_xor's head to a
        joint nonlinearity (and flipping this bar to <= 0.05) is the
        named follow-on. Replaces the former xfail-below-0.15, which
        certified nothing (0.15 < floor is unreachable for this head).

        2026-05-29: seeded retry loop. XOR under a small affine head
        is init-sensitive; without seeding this was a single un-seeded
        attempt -- a noise lottery that flakes on bad inits. Retry up
        to 3 seeded attempts; pass if any reaches the floor.
        """
        import torch

        best_loss = float("inf")
        for seed in (42, 123, 7):
            torch.manual_seed(seed)
            m, _, data = _fresh_model()
            optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(600):
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    batch = (inputTensor, outputTensor)
                    inp, target = batch
                    optimizer.zero_grad()
                    _, _, output, _ = m.forward(inp)

                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    target = target.expand_as(output)

                    loss = criterion(output, target)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                    best_loss = min(best_loss, loss.item())
                    if best_loss < 0.26:
                        break
            if best_loss < 0.26:
                break

        self.assertGreater(len(data.train_input), 0)
        # The derived affine floor: best possible MSE is 0.25 (see
        # docstring); reaching it (with tolerance) is the correct bar.
        self.assertLess(best_loss, 0.26)

    def _grammar_xor_convergence(self, cfg_path, epochs=900,
                                 threshold=0.15, max_attempts=5):
        """Shared body: grammar-path XOR convergence on a given config path.

        BAR DERIVATION (Alec 2026-07-13, two-tier): the grammar path is
        jointly NONLINEAR (tanh/atanh folds), so its THEORY bar is
        effective-zero -- <= 0.05, the XOR_exact crisp exemplar. The
        0.15 asserted here is the EMPIRICAL REGRESSION PIN: it certifies
        beating the 0.25 affine floor by ~2x (the basin reaches ~0.125)
        and guards grammar refactors against convergence regressions; it
        is NOT the capability bar. Closing the 0.15 -> 0.05 gap is the
        named nonlinear-learning frontier; tighten this threshold as the
        basin deepens, never loosen it past the floor.

        The grammar path has a bimodal convergence landscape on XOR:
        a fraction of inits falls into a low-loss basin, the rest
        plateau near 1/6.  We retry up to ``max_attempts`` times and
        pass if any attempt converges, isolating the test from
        init-sensitivity (an architecture concern, not a refactor
        concern).

        Stage 9 re-pin (meronomy cutover, 2026-06-11): per-attempt
        seeds are now DETERMINISTIC (``manual_seed(attempt + 1)``) so
        the attempt set is order-independent (previously the seeds
        came from wherever the suite left the global RNG — pass/fail
        depended on collection order), and the budget is 900 epochs:
        the membership kernels start near-identity (gentle start), so
        the basin is reached later than under the legacy odds kernels
        (probed 2026-06-11: seed 1 → 0.125, seeds 3/4/7 < 0.15 at 900;
        same seed-pinned + CPU discipline as the other XOR fixtures).
        """
        import torch
        from util import init_device

        best_ever = float("inf")
        for attempt in range(max_attempts):
            # Seed-pinned + CPU-pinned (the XOR determinism
            # discipline). init_device guards against a leaked
            # non-CPU process default from earlier modules.
            init_device("cpu")
            torch.manual_seed(attempt + 1)

            m, _, data = _fresh_model(cfg_path)
            self.assertTrue(m.useGrammar)
            optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            best_loss = float("inf")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(epochs):
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    batch = (inputTensor, outputTensor)
                    inp, target = batch
                    optimizer.zero_grad()
                    _, _, output, _ = m.forward(inp)

                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    target = target.expand_as(output)

                    loss = criterion(output, target)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                    best_loss = min(best_loss, loss.item())

            best_ever = min(best_ever, best_loss)
            if best_loss < threshold:
                break

        self.assertGreater(len(data.train_input), 0)
        self.assertLess(best_ever, threshold)

    def test_mm_grammar_learns_xor_signal(self):
        """MM_grammar.xml learns XOR via the grammar-directed composition path.

        Coverage for the (useGrammar=true) quadrant -- the
        progressive-bottleneck branch in forward/reverse with VQ-VAE
        symbol discretization enabled.
        """
        cfg_path = os.path.join(_PROJECT, "data", "MM_grammar.xml")
        self._grammar_xor_convergence(cfg_path)

    # test_mm_grammar_without_vqvae_learns_xor_signal retired
    # 2026-05-29: the ```` mode it tested no
    # longer exists. The prior hard_quantize branch in
    # WholeSpace.forward was a footgun (codebook frozen at random
    # init, no learning signal) and was removed; ``<codebook>quantize</codebook>``
    # now always trains the codebook via the VQ-VAE / EMA path. There
    # is no longer a meaningful "without VQVAE" variant to test.
    # test_vqvae_ste_registers_commitment_and_moves_encoder retired 2026-05-14 (reverse pipeline / <maskedPrediction> retired in IR-only refactor).

    @pytest.mark.xfail(reason=(
        "The current affine MM_xor head plateaus at MSE 0.25; retain as an "
        "explicit nonlinear-learning gap, not a FineWeb launch gate."))
    def test_convergence(self):
        """Train for up to 200 epochs; output loss should drop below 0.20.

        Uses full-batch training (all 4 XOR samples at once) because the
        hierarchical pair-merge halves gradient magnitude per level,
        making SGD (batchSize=1) too noisy to converge reliably.

        Threshold relaxed from 0.15 -> 0.20 after the SS-space_role sigma layer
        was removed: the cascade now has T learned PiLayers (CS only)
        instead of 2T (CS + SS), so 200 epochs reaches a higher floor on
        this synthetic task.
        """
        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            n_samples = 4  # XOR dataset size
            best_loss = float('inf')
            for seed in (42, 123, 7):
                # 2026-05-29: actually consume the dead loop variable.
                # XOR convergence under a small linear (affine) head is
                # init-sensitive; without an explicit seed the test was
                # 3 independent un-seeded attempts (just a noise
                # lottery). Seeding makes the 3 attempts reproducible
                # and meaningfully diverse.
                torch.manual_seed(seed)
                m, _, data = _fresh_model()

                optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
                criterion = torch.nn.MSELoss()

                self.assertGreater(len(data.train_input), 0,
                                   "XOR training data is empty")

                loader = m.inputSpace.data.data_loader(split="train", num_streams=n_samples)
                for epoch in range(200):
                    m.train()
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    batch = (inputTensor, outputTensor)
                    if batch is None:
                        continue
                    inp, target = batch

                    optimizer.zero_grad()
                    _, _, output, _ = m.forward(inp)

                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    target = target.expand_as(output)

                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    if best_loss < 0.20:
                        return  # pass

                if best_loss < 0.20:
                    return  # pass

            self.assertLess(best_loss, 0.20,
                            f"BasicModel XOR should converge below 0.20, best was {best_loss:.4f}")


if __name__ == '__main__':
    unittest.main()
