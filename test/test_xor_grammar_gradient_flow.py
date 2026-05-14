"""Acceptance test for the 2026-05-01 syntactic-layer refactor (Step 11).

Spec: doc/specs/2026-05-01-syntactic-layer-refactor.md §7 acceptance
criterion 2: when the chart fires an `intersection` rule for a tier
backed by a host PiLayer, gradient flows from the loss into that
PiLayer's parameters via the chart-driven dispatch path -- not via a
parallel always-on Pi path.

Slow / opt-in: set ``RUN_SLOW=1`` to enable. Marked ``xfail`` because
the current implementation extracts hard rule IDs (Viterbi argmax) and
the chart's composed tensor is not yet woven into the loss path. The
test exists so a future change that wires `chart._chart_vec` (or the
soft-rule-mixture path) into the spaces' forward output can flip this
to passing.

To make this test pass, the chart's `composed` output (the soft
mixture-weighted root vector, with `_rule_bias` / `_rule_embed` /
`_marker_bias` / host-layer parameters all on the autograd tape) needs
to flow into a downstream tensor that contributes to the loss. The
ChartCompose pipeline step in bin/Pipeline.py currently passes the
input subspace through unchanged.
"""
import os
import sys
import unittest
from pathlib import Path

import pytest
import torch

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))
sys.path.insert(0, str(_project / "test"))

_RUN_SLOW = os.getenv("RUN_SLOW") == "1"


@unittest.skipIf(not _RUN_SLOW, "slow -- set RUN_SLOW=1")
@pytest.mark.xfail(
    reason="Chart composed-vector is not yet on the loss path; "
           "rule selection is extracted as hard int IDs after the chart "
           "compose. Acceptance criterion 2 requires further work.",
    strict=False,
)
class TestXORGrammarGradientFlow(unittest.TestCase):
    """End-to-end gradient-flow check for the chart-driven path."""

    def _build_model(self):
        """Construct a small AR-mode BasicModel with an intersection
        rule so the chart's grammar choice routes through ConceptualSpace's
        IntersectionLayer-wrapped PiLayer host layer.
        """
        import test_basicmodel as tb
        import Models, Language

        tb._populate_test_config(
            inputDim=4, perceptDim=4, conceptDim=4, symbolDim=1, wordDim=1,
            nInput=8, nPercepts=8, nConcepts=8, nSymbols=8, nWords=8,
            nOutput=8, perceptPassThrough=True, symbolPassThrough=True)
        Models.TheXMLConfig._data["architecture"]["syntax"] = True
        (Models.TheXMLConfig._data["architecture"]
            .setdefault("training", {})["maskedPrediction"]) = "AR"
        # Configure a grammar with intersection on C tier so the chart
        # picks up an entry under ('C', 'intersection') in the host_layer
        # registry.
        (Models.TheXMLConfig._data
            .setdefault("WordSpace", {})
            .setdefault("language", {}))["grammar"] = {
                "S": ["union(C, C)"],
                "C": ["intersection(C, C)"],
            }
        Language.TheGrammar._configured = False
        Language.TheGrammar._ensure_configured()

        m = Models.BasicModel()
        m.create(nInput=8, nPercepts=8, nConcepts=8, nSymbols=8,
                 nOutput=8)
        return m

    def test_chart_rule_bias_accumulates_gradient(self):
        """Backprop from a forward pass should populate `chart._rule_bias.grad`
        so the chart's grammar choice is on the loss path."""
        m = self._build_model()
        ws = m.wordSpace
        self.assertIsNotNone(ws)
        # Force the chart to lazy-build its rule-shaped Parameters by
        # running one compose call against a dummy tensor.
        D = ws.chart._pair_feature_dim
        ws.compose(torch.randn(1, 4, D))
        rule_bias = getattr(ws.chart, '_rule_bias', None)
        self.assertIsNotNone(
            rule_bias, "chart._rule_bias not built after compose")

        # Run a forward pass + tiny loss + backward.
        x = torch.randn(2, 8, 4).tanh()
        m.train()
        _, _, pred, _ = m.forward(x)
        if pred is None:
            self.skipTest("model.forward returned no pred under this config")
        loss = pred.pow(2).mean()
        loss.backward()

        # The chart's rule_bias must accumulate gradient from the loss
        # path. Per acceptance criterion 2, it should not be all zeros.
        self.assertIsNotNone(rule_bias.grad,
                             "chart._rule_bias.grad is None")
        self.assertTrue(
            rule_bias.grad.abs().sum().item() > 0,
            "chart._rule_bias.grad is all zeros -- chart's grammar "
            "choice is not on the loss path",
        )


if __name__ == '__main__':
    unittest.main()
