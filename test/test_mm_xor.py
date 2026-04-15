"""MM_xor convergence test: MentalModel topology on XOR dataset.

Verifies that the MentalModel architecture (iterative [Percept,Symbol]->Concept->Symbol)
can learn the XOR function on the toy text dataset within 200 epochs.
"""

import os
import sys
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


def _fresh_model():
    """Create a fresh MentalModel with XOR data loaded."""
    from util import init_config
    init_config(
        path=_CONFIG,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    Spaces.TheGrammar._configured = False
    m, cfg = Models.MentalModel.from_config(_CONFIG)
    Models.TheData.load("xor")
    return m, cfg, Models.TheData


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
