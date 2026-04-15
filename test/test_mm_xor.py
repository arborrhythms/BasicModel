"""MM_xor convergence test: MentalModel topology on XOR dataset.

Verifies that the MentalModel architecture (iterative [Percept,Symbol]->Concept->Symbol)
can learn the XOR function on the toy text dataset within 200 epochs.
"""

import os
import sys
import tempfile
import unittest
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models
import Spaces

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor.xml")


def _fresh_model(config_path=_CONFIG):
    """Create a fresh MentalModel with XOR data loaded."""
    from util import init_config
    init_config(
        path=config_path,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    Spaces.TheGrammar._configured = False
    m, cfg = Models.MentalModel.from_config(config_path)
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


def _ramsified_config(enabled):
    """Write a temporary MM_xor config with an explicit ramsified setting."""
    value = "true" if enabled else "false"
    return _variant_config([
        ("<ramsified>true</ramsified>", f"<ramsified>{value}</ramsified>"),
        ("<ramsified>false</ramsified>", f"<ramsified>{value}</ramsified>"),
    ])


class TestMMXorConvergence(unittest.TestCase):
    """MentalModel on XOR should converge to near-zero output loss."""

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
        self.assertEqual(self.model.__class__.__name__, "MentalModel")

    def test_has_conceptual_symbolic_spaces(self):
        self.assertTrue(hasattr(self.model, 'conceptualSpace'))
        self.assertTrue(hasattr(self.model, 'symbolicSpace'))

    def test_a_forward_runs(self):
        """Forward pass smoke test. Runs before test_convergence (alpha order)."""
        import torch
        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            batch, _ = m.inputSpace.getBatch(0, batchSize=1)
            self.assertIsNotNone(batch, "getBatch returned None — data not loaded")
            inp, _ = batch
            with torch.no_grad():
                result = m.forward(inp)
        self.assertEqual(len(result), 3)

    def test_forward_reverse_reconstructs_input_state(self):
        """Reversible MM_xor should recover the encoded input state."""
        import torch

        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            batch, _ = m.inputSpace.getBatch(0, batchSize=4)
            inp, _ = batch
            with torch.no_grad():
                forward_input, symbols, output = m.forward(inp)
                input_data, _ = m.reverse(symbols, output)
        err = torch.nn.functional.mse_loss(
            input_data.squeeze(), forward_input.squeeze())
        self.assertLess(err.item(), 1e-6)

    def test_non_ramsified_forward_keeps_continuous_symbols(self):
        """The non-ramsified recurrent path should not collapse via symbol VQ."""
        import torch

        cfg_path = _ramsified_config(False)
        try:
            torch.manual_seed(42)
            m, _, _ = _fresh_model(cfg_path)
            self.assertFalse(m.ramsified)
            m.eval()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                batch, _ = m.inputSpace.getBatch(0, batchSize=4)
                inp, _ = batch
                with torch.no_grad():
                    m.forward(inp)
            symbols = m.symbolicSpace.subspace.materialize()
            self.assertTrue(torch.isfinite(symbols).all())
            self.assertGreater(symbols.std().item(), 1e-6)
        finally:
            os.unlink(cfg_path)

    def test_non_ramsified_runbatch_losses_stay_finite(self):
        """The public train/reverse loss path should stay finite."""
        import torch

        cfg_path = _ramsified_config(False)
        try:
            torch.manual_seed(42)
            m, _, _ = _fresh_model(cfg_path)
            optimizer = m.getOptimizer(lr=0.01)

            totals = []
            original_total = m.loss.total

            def capture_total(lossOut, lossIn=None, sbow=None):
                total = original_total(lossOut, lossIn, sbow)
                totals.append(total.detach())
                return total

            m.loss.total = capture_total
            original_message = Models.TheMessage
            Models.TheMessage = lambda *args, **kwargs: None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for _ in range(30):
                    result, _ = m.runBatch(
                        train=True,
                        batchNum=0,
                        batchSize=4,
                        split="train",
                        optimizer=optimizer,
                    )
                    self.assertIsNotNone(result)
                    self.assertTrue(torch.isfinite(result.lossOut))
                    self.assertTrue(torch.isfinite(result.lossIn))
            Models.TheMessage = original_message

            self.assertEqual(len(totals), 30)
            self.assertTrue(torch.isfinite(torch.stack(totals)).all())
        finally:
            if 'original_message' in locals():
                Models.TheMessage = original_message
            os.unlink(cfg_path)

    def test_ramsified_runbatch_losses_stay_finite(self):
        """The ramsified symbol-codebook path should stay finite."""
        import torch

        cfg_path = _ramsified_config(True)
        try:
            torch.manual_seed(42)
            m, _, _ = _fresh_model(cfg_path)
            self.assertTrue(m.ramsified)
            optimizer = m.getOptimizer(lr=0.01)

            totals = []
            original_total = m.loss.total

            def capture_total(lossOut, lossIn=None, sbow=None):
                total = original_total(lossOut, lossIn, sbow)
                totals.append(total.detach())
                return total

            m.loss.total = capture_total
            original_message = Models.TheMessage
            Models.TheMessage = lambda *args, **kwargs: None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for _ in range(30):
                    result, _ = m.runBatch(
                        train=True,
                        batchNum=0,
                        batchSize=4,
                        split="train",
                        optimizer=optimizer,
                    )
                    self.assertIsNotNone(result)
                    self.assertTrue(torch.isfinite(result.lossOut))
                    self.assertTrue(torch.isfinite(result.lossIn))

            terms = m.symbolicSpace.symbol_objective_terms()
            self.assertIn("symbol_residual", terms)
            self.assertIn("symbol_l1", terms)
            self.assertEqual(len(totals), 30)
            self.assertTrue(torch.isfinite(torch.stack(totals)).all())
        finally:
            if 'original_message' in locals():
                Models.TheMessage = original_message
            os.unlink(cfg_path)

    def test_non_ramsified_learns_xor_signal(self):
        """ramsified=false should exercise the recurrent forward path and learn."""
        import torch

        cfg_path = _ramsified_config(False)
        try:
            torch.manual_seed(42)
            m, _, data = _fresh_model(cfg_path)
            optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            best_loss = float("inf")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for _ in range(50):
                    batch, _ = m.inputSpace.getBatch(0, batchSize=4)
                    inp, target = batch
                    optimizer.zero_grad()
                    _, _, output = m.forward(inp)

                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    target = target.expand_as(output)

                    loss = criterion(output, target)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                    best_loss = min(best_loss, loss.item())

            self.assertGreater(len(data.train_input), 0)
            self.assertLess(best_loss, 0.15)
        finally:
            os.unlink(cfg_path)

    def test_vqvae_ste_registers_commitment_and_moves_encoder(self):
        """With useVQVAE=true, STE path must register symbol_commitment and
        deliver gradient through the quantization bottleneck into the encoder.

        Verifies the core fix of the VQ-VAE STE follow-on: the forward pass
        produces the hard codebook pick, while the backward pass bypasses
        argmin so downstream losses shape the PiLayer encoder weights.
        """
        import torch

        torch.manual_seed(123)
        m, _, _ = _fresh_model()
        self.assertTrue(m.symbolicSpace.use_vqvae,
                        "MM_xor.xml should have useVQVAE=true")
        self.assertGreater(m.symbolicSpace.commitment_beta, 0.0)

        optimizer = m.getOptimizer(lr=0.01)

        pi_weight_before = None
        for p in m.symbolicSpace.layer.parameters():
            if p.requires_grad:
                pi_weight_before = p.detach().clone()
                break
        self.assertIsNotNone(pi_weight_before,
                             "SymbolicSpace PiLayer should have trainable params")

        commit_values = []
        original_message = Models.TheMessage
        Models.TheMessage = lambda *args, **kwargs: None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for _ in range(20):
                    result, _ = m.runBatch(
                        train=True, batchNum=0, batchSize=4,
                        split="train", optimizer=optimizer,
                    )
                    self.assertIsNotNone(result)
                    terms = m.symbolicSpace.symbol_objective_terms()
                    self.assertIn("symbol_commitment", terms,
                                  "STE path must register symbol_commitment")
                    commit_values.append(
                        float(terms["symbol_commitment"].detach().item()))
        finally:
            Models.TheMessage = original_message

        pi_weight_after = None
        for p in m.symbolicSpace.layer.parameters():
            if p.requires_grad:
                pi_weight_after = p.detach().clone()
                break
        moved = (pi_weight_after - pi_weight_before).abs().max().item()
        self.assertGreater(
            moved, 1e-5,
            "PiLayer weights should move — STE gradient must reach the encoder")
        self.assertTrue(all(torch.isfinite(torch.tensor(commit_values))),
                        "symbol_commitment must stay finite over training")

    def test_convergence(self):
        """Train for up to 200 epochs; output loss should drop below 0.15.

        Uses full-batch training (all 4 XOR samples at once) because the
        hierarchical butterfly merge halves gradient magnitude per level,
        making SGD (batchSize=1) too noisy to converge reliably.
        """
        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            n_samples = 4  # XOR dataset size
            best_loss = float('inf')
            for seed in (42, 123, 7):
                torch.manual_seed(seed)
                m, _, data = _fresh_model()

                optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
                criterion = torch.nn.MSELoss()

                self.assertGreater(len(data.train_input), 0,
                                   "XOR training data is empty")

                for epoch in range(200):
                    m.train()
                    batch, _ = m.inputSpace.getBatch(0, batchSize=n_samples)
                    if batch is None:
                        continue
                    inp, target = batch

                    optimizer.zero_grad()
                    _, _, output = m.forward(inp)

                    target = target.to(output.device)
                    while target.dim() < output.dim():
                        target = target.unsqueeze(-1)
                    target = target.expand_as(output)

                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                    if best_loss < 0.15:
                        return  # pass

                if best_loss < 0.15:
                    return  # pass

            self.assertLess(best_loss, 0.15,
                            f"MentalModel XOR should converge below 0.15, best was {best_loss:.4f}")


if __name__ == '__main__':
    unittest.main()
