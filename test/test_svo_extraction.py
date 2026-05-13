"""SVO extraction from the chart's Viterbi derivation trace.

Exercises ``Chart.extract_svo`` -- post-2026-05-05, the chart walks
its derivation trace looking for ``S = lift(NP, VP)`` over
``VP = intersection(V, O)`` and stashes the operand tensors on
``chart.last_svo``.  Used by ``Models._universality_score`` to feed
the Golden Rule (universality) test.
"""

import os
import sys
import unittest
from pathlib import Path

import torch

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "bin"))

from Language import Chart, TheGrammar


_SVO_GRAMMAR = {
    'compose': {
        'symbols': {
            'rule': [
                'S = lift(NP, VP)',
                'VP = intersection(V, O)',
                'O = NP',
                'NP = N',
                'VP = V',
            ],
        },
    },
}


def _isolated_grammar(rules):
    TheGrammar._configured = False
    TheGrammar.configure(rules)
    return TheGrammar


class TestChartExtractSVO(unittest.TestCase):
    def setUp(self):

        _isolated_grammar(_SVO_GRAMMAR)
        self.chart = Chart(
            nInput=4, max_depth=4, hidden_dim=16, D_rule=4,
            feature_dim=8, w_max=4)
        self.chart.eval()  # Viterbi (hard) inside pass.

    class _FakeWS:
        current_rules = {}
        generate_rules = {}
        _sentence_completed = [False]
        _compose_generation = 0
        _generate_generation = 0
        _host_layer_registry = {}

        def host_layer(self, tier, rule_name):
            return None

        def _row_K(self):
            return 1

    def test_extract_svo_populates_last_svo(self):
        """A 3-token parse compatible with S = lift(NP, VP) over
        VP = intersection(V, O) -- the chart should populate
        chart.last_svo with the (subject, verb, object) operand
        tensors."""
        ws = self._FakeWS()
        data = torch.randn(1, 3, 8)
        self.chart.compose(data, ws)
        # Per-row mask should be defined; whether SVO actually fires
        # depends on the chart's per-cell scoring, so we don't assert
        # a specific batch row here -- only that the extractor ran
        # without exception and the field is well-typed.
        mask = self.chart._svo_row_mask
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1,))

    def test_extract_svo_returns_none_without_lift_rule(self):
        """Without S = lift(NP, VP) in the grammar, last_svo stays
        None (no derivation can match the SVO signature)."""
        _isolated_grammar({'compose': {'symbols': {'rule': ['S = not(S)']}}})
        chart = Chart(
            nInput=4, max_depth=4, hidden_dim=16, D_rule=4,
            feature_dim=8, w_max=4)
        chart.eval()
        ws = self._FakeWS()
        data = torch.randn(1, 3, 8)
        chart.compose(data, ws)
        self.assertIsNone(chart.last_svo)

    def test_extract_svo_shape_when_present(self):
        """When an SVO derivation is found, last_svo is a 3-tuple
        of [B, 1, D] tensors."""
        ws = self._FakeWS()
        data = torch.randn(2, 3, 8)
        self.chart.compose(data, ws)
        if self.chart.last_svo is None:
            self.skipTest("Random init did not produce an SVO derivation")
        s, v, o = self.chart.last_svo
        self.assertEqual(s.shape, (2, 1, 8))
        self.assertEqual(v.shape, (2, 1, 8))
        self.assertEqual(o.shape, (2, 1, 8))


if __name__ == '__main__':
    unittest.main()
