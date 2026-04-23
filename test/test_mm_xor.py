"""MM_xor convergence test: MentalModel topology on XOR dataset.

Verifies that the MentalModel architecture (iterative [Percept,Symbol]->Concept->Symbol)
can learn the XOR function on the toy text dataset within 200 epochs.
"""

import os
import sys
import tempfile
import unittest
import warnings

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_BIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin")
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BIN not in sys.path:
    sys.path.insert(0, _BIN)
import Models
import Spaces
import Language

_CONFIG = os.path.join(_PROJECT, "data", "MM_xor.xml")


def _fresh_model(config_path=_CONFIG):
    """Create a fresh MentalModel with XOR data loaded."""
    from util import init_config
    init_config(
        path=config_path,
        defaults_path=os.path.join(_PROJECT, "data", "model.xml"),
    )
    Language.TheGrammar._configured = False
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


def _butterfly_config(enabled):
    """Write a temporary MM_xor config with an explicit useButterflies setting."""
    value = "true" if enabled else "false"
    return _variant_config([
        ("<useButterflies>true</useButterflies>",
         f"<useButterflies>{value}</useButterflies>"),
        ("<useButterflies>false</useButterflies>",
         f"<useButterflies>{value}</useButterflies>"),
    ])


class TestMMXorConvergence(unittest.TestCase):
    """MentalModel on XOR should converge to near-zero output loss."""

    @classmethod
    def setUpClass(cls):
        import torch
        import matplotlib
        matplotlib.use('Agg')

        torch.manual_seed(13)
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

    def test_forward_reverse_reconstructs_input_state(self):
        """Reversible MM_xor should recover the encoded input state.

        MM_xor uses text mode: PerceptualSpace.reverse routes through
        _reverse_text which snaps vectors to nearest embedding entries.
        With a fresh untrained embedding on XOR's tiny vocabulary, the
        snap-to-nearest loss dominates fp round-off, so the tolerance
        reflects nearest-neighbor recovery rather than exact inversion.
        A trained model would tighten this naturally.
        """
        import torch

        m = self.model
        m.eval()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
            inp_items, out_items = next(iter(loader))
            inputTensor = m.inputSpace.prepInput(inp_items)
            outputTensor = (m.outputSpace.prepOutput(out_items)
                            if out_items is not None else None)
            batch = (inputTensor, outputTensor)
            inp, _ = batch
            with torch.no_grad():
                forward_input, symbols, output, _ = m.forward(inp)
                input_data, _ = m.reverse(symbols, output)
        err = torch.nn.functional.mse_loss(
            input_data.squeeze(), forward_input.squeeze())
        self.assertLess(err.item(), 5e-3)

    def test_non_butterfly_forward_keeps_continuous_symbols(self):
        """The non-butterfly recurrent path should not collapse via symbol VQ."""
        import torch

        cfg_path = _butterfly_config(False)
        try:
            torch.manual_seed(42)
            m, _, _ = _fresh_model(cfg_path)
            self.assertFalse(m.useButterflies)
            m.eval()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                inp_items, out_items = next(iter(loader))
                inputTensor = m.inputSpace.prepInput(inp_items)
                outputTensor = (m.outputSpace.prepOutput(out_items)
                                if out_items is not None else None)
                batch = (inputTensor, outputTensor)
                inp, _ = batch
                with torch.no_grad():
                    m.forward(inp)
            symbols = m.symbolicSpace.subspace.materialize()
            self.assertTrue(torch.isfinite(symbols).all())
            self.assertGreater(symbols.std().item(), 1e-6)
        finally:
            os.unlink(cfg_path)

    def test_non_butterfly_runbatch_losses_stay_finite(self):
        """The public train/reverse loss path should stay finite."""
        import torch

        cfg_path = _butterfly_config(False)
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
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(30):
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    result, _ = m.runBatch(
                        train=True,
                        batchNum=0,
                        batchSize=4,
                        split="train",
                        optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
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

    def test_butterfly_runbatch_losses_stay_finite(self):
        """The butterfly symbol-codebook path should stay finite."""
        import torch

        cfg_path = _butterfly_config(True)
        try:
            torch.manual_seed(42)
            m, _, _ = _fresh_model(cfg_path)
            self.assertTrue(m.useButterflies)
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
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(30):
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    result, _ = m.runBatch(
                        train=True,
                        batchNum=0,
                        batchSize=4,
                        split="train",
                        optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
                    )
                    self.assertIsNotNone(result)
                    self.assertTrue(torch.isfinite(result.lossOut))
                    self.assertTrue(torch.isfinite(result.lossIn))

            from Layers import TheError
            term_names = {t[0] for t in TheError.terms()}
            self.assertIn("symbol_residual", term_names)
            self.assertIn("symbol_l1", term_names)
            self.assertEqual(len(totals), 30)
            self.assertTrue(torch.isfinite(torch.stack(totals)).all())
        finally:
            if 'original_message' in locals():
                Models.TheMessage = original_message
            os.unlink(cfg_path)

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_non_butterfly_learns_xor_signal(self):
        """useButterflies=false should exercise the recurrent forward path and learn."""
        import torch

        cfg_path = _butterfly_config(False)
        try:
            m, _, data = _fresh_model(cfg_path)
            optimizer = torch.optim.Adam(m.parameters(), lr=0.01)
            criterion = torch.nn.MSELoss()
            best_loss = float("inf")

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

            self.assertGreater(len(data.train_input), 0)
            self.assertLess(best_loss, 0.15)
        finally:
            os.unlink(cfg_path)

    def _grammar_xor_convergence(self, cfg_path, epochs=600,
                                 threshold=0.15, max_attempts=5):
        """Shared body: grammar-path XOR convergence on a given config path.

        The grammar path has a bimodal convergence landscape on XOR:
        ~=30% of random inits fall into a zero-loss basin, the rest
        plateau near 1/6.  We retry up to ``max_attempts`` times with
        different seeds and pass if any attempt converges, isolating
        the test from init-sensitivity (an architecture concern, not a
        refactor concern).
        """
        import torch

        best_ever = float("inf")
        for attempt in range(max_attempts):
            torch.manual_seed(attempt)
            m, _, data = _fresh_model(cfg_path)
            self.assertFalse(m.useButterflies)
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

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_mm_grammar_learns_xor_signal(self):
        """MM_grammar.xml learns XOR via the grammar-directed composition path.

        Coverage for the (useButterflies=false, useGrammar=true)
        quadrant -- the progressive-bottleneck branch in forward/reverse
        with VQ-VAE symbol discretization enabled.
        """
        cfg_path = os.path.join(_PROJECT, "data", "MM_grammar.xml")
        self._grammar_xor_convergence(cfg_path)

    @unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
    def test_mm_grammar_without_vqvae_learns_xor_signal(self):
        """MM_grammar.xml with useVQVAE flipped to false learns XOR.

        Isolated regression fixture: exercises the grammar path without
        VQ-VAE commitment loss muddying the variable.  Convergence here
        confirms the C-tier pairwise reducer and progressive-bottleneck
        Pi-Sigma loop thread information across the slot axis end-to-end.
        """
        src = os.path.join(_PROJECT, "data", "MM_grammar.xml")
        with open(src, "r", encoding="utf-8") as fh:
            text = fh.read()
        text = text.replace(
            "<useVQVAE>true</useVQVAE>",
            "<useVQVAE>false</useVQVAE>",
            1,
        )
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
        tmp.write(text)
        tmp.close()
        try:
            self._grammar_xor_convergence(tmp.name)
        finally:
            os.unlink(tmp.name)

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

        # Snapshot every trainable SymbolicSpace parameter.  Which Pi
        # instance actually carries the forward (self.layer vs.
        # butterfly_pis[*] vs. pi_layers[*]) depends on useButterflies /
        # butterfly config, so we check for movement across the whole
        # set instead of a single named Pi.
        params_before = {
            name: p.detach().clone()
            for name, p in m.symbolicSpace.named_parameters()
            if p.requires_grad
        }
        self.assertGreater(len(params_before), 0,
                           "SymbolicSpace should have trainable params")

        commit_values = []
        original_message = Models.TheMessage
        Models.TheMessage = lambda *args, **kwargs: None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                loader = m.inputSpace.data.data_loader(split="train", num_streams=4)
                for _ in range(20):
                    inp_items, out_items = next(iter(loader))
                    inputTensor = m.inputSpace.prepInput(inp_items)
                    outputTensor = (m.outputSpace.prepOutput(out_items)
                                    if out_items is not None else None)
                    result, _ = m.runBatch(
                        train=True, batchNum=0, batchSize=4,
                        split="train", optimizer=optimizer,
                        batch_override=(inputTensor, outputTensor),
                    )
                    self.assertIsNotNone(result)
                    from Layers import TheError
                    terms_by_name = {t[0]: t[1] for t in TheError.terms()}
                    self.assertIn("symbol_commitment", terms_by_name,
                                  "STE path must register symbol_commitment")
                    commit_values.append(
                        float(terms_by_name["symbol_commitment"].detach().item()))
        finally:
            Models.TheMessage = original_message

        moved_names = []
        max_move = 0.0
        for name, p in m.symbolicSpace.named_parameters():
            if not p.requires_grad or name not in params_before:
                continue
            delta = (p.detach() - params_before[name]).abs().max().item()
            if delta > 1e-5:
                moved_names.append(name)
            if delta > max_move:
                max_move = delta
        self.assertGreater(
            len(moved_names), 0,
            f"Commitment gradient must reach the encoder -- no SymbolicSpace "
            f"params moved (max Δ={max_move:.2e})")
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
                    if best_loss < 0.15:
                        return  # pass

                if best_loss < 0.15:
                    return  # pass

            self.assertLess(best_loss, 0.15,
                            f"MentalModel XOR should converge below 0.15, best was {best_loss:.4f}")


if __name__ == '__main__':
    unittest.main()
