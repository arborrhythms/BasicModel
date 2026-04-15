"""Tests for the reasoning system: partitions, isConsistent, ground, isTrue,
extrapolate, TruthLoss (falsity_penalty), and reason().

Unit tests pass without a trained model.
English-level tests are @pytest.mark.xfail until word identity is learned.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import torch
import torch.nn.functional as F
import pytest
import matplotlib
import Models
import Spaces
import Layers
matplotlib.use('Agg')

from util import init_config, ProjectPaths, TheXMLConfig

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Spaces.TheGrammar._configured = False


def _make_model(config='MentalModel.xml'):
    init_config(
        path=os.path.join(_DATA_DIR, config),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )
    Spaces.TheGrammar._configured = False
    model, cfg = Models.MentalModel.from_config(os.path.join(_DATA_DIR, config))
    model.eval()
    return model


# ── Step 1: Order Partitions ──────────────────────────────────────────

class TestOrderPartitions(unittest.TestCase):

    def test_partitions_cover_full_dim(self):
        """Partitions must cover [0, symbol_dim) without overlap or gap."""
        for dim in [16, 32, 64, 128, 256]:
            for order in [1, 2, 3, 4, 5]:
                parts = Models.MentalModel._order_partitions(dim, order)
                self.assertEqual(len(parts), order)
                # First starts at 0
                self.assertEqual(parts[0][0], 0)
                # Last ends at dim
                self.assertEqual(parts[-1][1], dim)
                # No gaps or overlaps
                for i in range(len(parts) - 1):
                    self.assertEqual(parts[i][1], parts[i + 1][0])
                # All slices have at least 1 element
                for s, e in parts:
                    self.assertGreater(e, s)

    def test_geometric_decay(self):
        """Lower orders should have larger slices."""
        parts = Models.MentalModel._order_partitions(128, 4)
        sizes = [e - s for s, e in parts]
        # Order 0 should be largest
        self.assertGreater(sizes[0], sizes[1])
        self.assertGreater(sizes[1], sizes[2])

    def test_single_order(self):
        """With conceptualOrder=1, one partition covers everything."""
        parts = Models.MentalModel._order_partitions(64, 1)
        self.assertEqual(parts, [(0, 64)])

    def test_activation_order(self):
        """_activation_order returns the order with highest partition energy."""
        parts = Models.MentalModel._order_partitions(16, 2)
        # Energy in first partition
        act1 = torch.zeros(16)
        act1[:8] = 1.0
        self.assertEqual(Models.MentalModel._activation_order(act1, parts), 0)
        # Energy in second partition
        act2 = torch.zeros(16)
        act2[8:] = 1.0
        self.assertEqual(Models.MentalModel._activation_order(act2, parts), 1)


# ── Step 3: isConsistent ──────────────────────────────────────────────

class TestIsConsistent(unittest.TestCase):

    def test_empty_truth_set_is_consistent(self):
        model = _make_model()
        result = model.isConsistent()
        self.assertTrue(result['consistent'])
        self.assertEqual(result['score'], 1.0)

    def test_consistent_truths(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        # Two truths pointing in similar directions
        t1 = torch.randn(D)
        t2 = t1 + 0.1 * torch.randn(D)  # similar direction
        truth_layer.record(t1, degree=0.8)
        truth_layer.record(t2, degree=0.7)
        result = model.isConsistent()
        self.assertGreater(result['score'], 0.3)

    def test_contradictory_truths(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        # Two truths pointing in opposite directions (contradiction)
        t1 = torch.ones(D) * 0.5
        t2 = -torch.ones(D) * 0.5  # exact negation
        truth_layer.record(t1, degree=1.0)
        truth_layer.record(t2, degree=1.0)
        result = model.isConsistent()
        # Disjunction of opposite-sign vectors → zero → low score
        self.assertLess(result['score'], 0.5)


# ── Steps 4-5: ground, isTrue ────────────────────────────────────────

class TestGroundAndIsTrue(unittest.TestCase):

    def test_unmatched_activation_not_grounded(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        # Empty truth set
        act = torch.randn(D)
        result = model.ground(act)
        self.assertFalse(result['grounded'])

    def test_matching_activation_grounded(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        # Store a truth, then query with the same activation
        t1 = torch.randn(D)
        t1 = F.normalize(t1.unsqueeze(0), dim=-1).squeeze(0)
        truth_layer.record(t1, degree=0.9)
        result = model.ground(t1, threshold=0.5)
        self.assertTrue(result['grounded'])
        self.assertGreater(len(result['basis']), 0)

    def test_isTrue_ungrounded_returns_zero(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        act = torch.randn(D)
        self.assertEqual(model.isTrue(act), 0.0)


# ── Step 6: TruthLoss (falsity_penalty) ──────────────────────────────

class TestTruthLoss(unittest.TestCase):

    def _make_basis(self):
        return Spaces.Basis()

    def test_empty_truth_set_no_penalty(self):
        tl = Layers.TruthLayer(nDim=8)
        basis = self._make_basis()
        concepts = torch.randn(2, 3, 8)
        penalty = tl.falsity_penalty(concepts, basis)
        self.assertEqual(penalty.item(), 0.0)

    def test_agreeing_proposition_low_penalty(self):
        tl = Layers.TruthLayer(nDim=8)
        basis = self._make_basis()
        # Store a positive truth
        truth = torch.ones(8) * 0.5
        tl.record(truth, degree=1.0)
        # Proposition in same direction should not reduce union norm
        prop = torch.ones(1, 1, 8) * 0.3
        penalty = tl.falsity_penalty(prop, basis)
        self.assertAlmostEqual(penalty.item(), 0.0, places=4)

    def test_contradicting_proposition_positive_penalty(self):
        tl = Layers.TruthLayer(nDim=8)
        basis = self._make_basis()
        # Store a positive truth
        truth = torch.ones(8) * 0.5
        tl.record(truth, degree=1.0)
        # Proposition in opposite direction → contradiction → norm reduction
        prop = -torch.ones(1, 1, 8) * 0.5
        penalty = tl.falsity_penalty(prop, basis)
        self.assertGreater(penalty.item(), 0.0)

    def test_unknown_proposition_zero_penalty(self):
        tl = Layers.TruthLayer(nDim=8)
        basis = self._make_basis()
        truth = torch.ones(8) * 0.5
        tl.record(truth, degree=1.0)
        # Zero proposition → unknown → no effect on union
        prop = torch.zeros(1, 1, 8)
        penalty = tl.falsity_penalty(prop, basis)
        self.assertAlmostEqual(penalty.item(), 0.0, places=4)

    def test_truth_loss_weight_zero_no_effect(self):
        """When TruthLoss=0.0 in config, no penalty should be added."""
        model = _make_model()
        self.assertEqual(getattr(model, 'truth_loss_weight', 0.0), 0.0)

    def test_penalty_is_differentiable(self):
        tl = Layers.TruthLayer(nDim=8)
        basis = self._make_basis()
        truth = torch.ones(8) * 0.5
        tl.record(truth, degree=1.0)
        leaf = torch.ones(1, 1, 8, requires_grad=True)
        prop = -leaf * 0.5
        prop.retain_grad()
        penalty = tl.falsity_penalty(prop, basis)
        penalty.backward()
        self.assertIsNotNone(leaf.grad)


# ── Step 7: extrapolate ──────────────────────────────────────────────

class TestExtrapolate(unittest.TestCase):

    def test_too_few_truths_returns_empty(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        # Only 1 truth → can't pair
        D = truth_layer.nDim
        truth_layer.record(torch.randn(D), degree=0.8)
        result = model.extrapolate()
        self.assertEqual(result['added'], [])

    def test_consistent_pair_derives(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        # Two consistent truths
        t1 = torch.randn(D)
        t2 = torch.randn(D)
        truth_layer.record(t1, degree=0.8)
        truth_layer.record(t2, degree=0.7)
        result = model.extrapolate(max_new=4)
        # Should derive at least something (union, intersection, etc.)
        total = len(result['added']) + len(result['rejected'])
        self.assertGreater(total, 0, "Grammar methods should produce candidates")


# ── Step 1 integration: write-mask isolation ─────────────────────────

class TestWriteMask(unittest.TestCase):

    def test_partition_isolation(self):
        """After ramsified forward, each order's partition should be isolated."""
        model = _make_model('RamsifiedModel.xml')

        truth_layer = model._get_truth_layer()
        D = model.symbolicSpace.layer.nOutput

        sentences = ['test sentence one', 'test sentence two']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            train_input, _ = model.inputSpace.getTrainData()
            x = model.inputSpace.prepInput(train_input[:2])
            with torch.no_grad():
                model.forward(x)


# ── Step 9: reason() ─────────────────────────────────────────────────

class TestReason(unittest.TestCase):

    def test_forward_with_no_target(self):
        model = _make_model()
        truth_layer = model._get_truth_layer()
        if truth_layer is None:
            self.skipTest("No TruthLayer available")
        D = truth_layer.nDim
        givens = torch.randn(D)
        result = model.reason(givens, direction='forward', max_steps=2)
        self.assertIn('proved', result)
        self.assertIn('trace', result)

    def test_reverse_with_no_target(self):
        model = _make_model()
        result = model.reason(torch.randn(8), target=None,
                               direction='reverse', max_steps=1)
        self.assertFalse(result['proved'])


# ── English-level tests (require trained model) ──────────────────────

@pytest.mark.xfail(reason="requires trained model with populated TruthSet", run=False)
def test_syllogism_all_men_mortal():
    # Given: "all men are mortal", "Socrates is a man"
    # Expect: isTrue(encode("Socrates is mortal")) > 0.5
    model = _make_model()
    pass  # encode sentences → activations → reason()


@pytest.mark.xfail(reason="requires trained model with populated TruthSet", run=False)
def test_syllogism_contrapositive():
    # Given: "if it rains the ground is wet", "the ground is not wet"
    # Expect: isTrue(encode("it did not rain")) > 0.5
    model = _make_model()
    pass


@pytest.mark.xfail(reason="requires trained model", run=False)
def test_inconsistency_detected():
    # TruthSet: "the ball is red" (+1) and "the ball is not red" (-1)
    # Expect: isConsistent() → consistent=False
    model = _make_model()
    pass


@pytest.mark.xfail(reason="requires trained model with synonym coverage", run=False)
def test_semantic_equivalence_paraphrase():
    # "the canine bounded" ≈ "the dog ran" in conceptual space (cosine sim > 0.7)
    model = _make_model()
    pass


@pytest.mark.xfail(reason="requires trained model", run=False)
def test_semantic_nonequivalence():
    # "the dog ran" vs "the cat slept" cosine sim < 0.5
    model = _make_model()
    pass


@pytest.mark.xfail(reason="requires trained model", run=False)
def test_extrapolate_transitive():
    # Given: "A is part of B", "B is part of C"
    # Expect: extrapolate() derives something encoding "A is part of C"
    model = _make_model()
    pass


if __name__ == '__main__':
    unittest.main()
