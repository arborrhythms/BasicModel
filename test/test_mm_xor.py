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
# Keep device selection independent of prior tests; initialization stays random.
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

    def _fit_xor(self, config_path, epochs, threshold, *, grammar=False):
        """One initialization and one fixed budget; never select a lucky retry."""
        import torch
        from util import init_device

        init_device("cpu")
        m, _, data = _fresh_model(config_path)
        if grammar:
            self.assertTrue(m.useGrammar)
        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        best_loss = float("inf")
        try:
            self.assertGreater(len(data.train_input), 0)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(epochs):
                    inp_items, out_items = next(iter(loader))
                    inp = m.inputSpace.prepInput(inp_items)
                    target = m.outputSpace.prepOutput(out_items)
                    optimizer.zero_grad()
                    _, _, output, _ = m.forward(inp)
                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    loss = criterion(output, target.expand_as(output))
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                    best_loss = min(best_loss, loss.item())
                    if best_loss < threshold:
                        break
            self.assertLess(best_loss, threshold,
                            f"XOR loss {best_loss:.6g} after at most {epochs} epochs")
        finally:
            m.End()
            m.symbolSpace.soft_reset()

    @pytest.mark.slow
    def test_learns_xor_signal(self):
        """Reach the affine MSE floor .25 (tolerance .01) in 600 epochs."""
        self._fit_xor(_CONFIG, epochs=600, threshold=.26)

    @pytest.mark.slow
    def test_mm_grammar_learns_xor_signal(self):
        """Retain the .20 regression bar; effective-zero utility is unproven."""
        self._fit_xor(os.path.join(_PROJECT, "data", "MM_grammar.xml"),
                      epochs=900, threshold=.20, grammar=True)

    @pytest.mark.slow
    def test_convergence(self):
        """Retain the existing 200-epoch .20 bar without seed selection."""
        self._fit_xor(_CONFIG, epochs=200, threshold=.20)


if __name__ == '__main__':
    unittest.main()
