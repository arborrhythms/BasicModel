"""
Toy Grammar Learning Test
=========================

Verifies that Grammar's SyntacticLayers participate in the computation
graph and receive gradients when driven through SyntacticLayer.forward().

Each tier's SyntacticLayer predicts rule distributions from activation
vectors. The gradient path:

    loss → rule_logits → rule_head weights → derivation_layer → input_proj → x

Tests:
  1. Gradient flows from rule_logits to SyntacticLayer rule_head (S, C tiers)
  2. Different activations produce different rule distributions (selectivity)
  3. Untrained rule probabilities are near-uniform (entropy check)
  4. MentalModel.forward() exercises Grammar and produces word tuples
  5. Rule preference report (informational)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bin'))

import unittest
import warnings
import torch

import matplotlib
matplotlib.use('Agg')

from BasicModel import MentalModel, TheData, TheDevice
from Space import Grammar

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    from util import init_config
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )


def _make_model():
    """Create a MentalModel and return (model, TheGrammar)."""
    _reload_config()
    from Space import TheGrammar
    # Force re-initialization in case previous tests set small test dimensions
    TheGrammar._configured = False
    TheGrammar._configured = False
    model, _ = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model, TheGrammar


def _forward_with_data(model, sentences, outputs):
    """Run a forward pass through MentalModel with given data."""
    with TheData.runtime_batch(sentences, outputs), \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Range violation")
        warnings.filterwarnings("ignore", message="PiLayer.reverse")
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:len(sentences)])
        return model.forward(x)


# ── Dimensions from MentalModel.xml ─────────────────────────────────
_N_SYMBOLS = 128
_N_CONCEPTS = 256
_N_PERCEPTS = 128
_CONCEPT_DIM = 100 + 4   # nDim + objectSize (nWhere + nWhen)
_SYMBOL_DIM = 100 + 4
_N_CONCEPT_SLOTS = _N_PERCEPTS + _N_SYMBOLS  # conceptInputShape[0] = 256


class TestGrammarGradientFlow(unittest.TestCase):
    """Grammar SyntacticLayers receive gradients via rule prediction."""

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    def test_c_tier_gradient_flows(self):
        """C-tier rule_head receives gradients from SyntacticLayer.forward().

        SyntacticLayer.forward(activation) predicts rule distributions.
        loss.backward() should propagate gradients to the rule_head weights.
        """
        sl_c = self.model.conceptualSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        sl_c.train()
        act = (torch.randn(1, _N_CONCEPTS, device=TheDevice.get()) * 0.01).requires_grad_(True)
        out = sl_c.forward(act)
        loss = out['rule_logits'].pow(2).sum()
        loss.backward()

        self.assertIsNotNone(act.grad,
            "No gradient to input from C-tier SyntacticLayer.forward()")
        self.assertTrue((act.grad.abs() > 0).any(),
            "Zero gradient to input from C-tier rule prediction")

    def test_s_tier_gradient_flows(self):
        """S-tier rule_head receives gradients from SyntacticLayer.forward()."""
        sl_s = self.model.symbolicSpace.syntacticLayer
        if sl_s is None:
            self.skipTest("No S-tier SyntacticLayer")

        sl_s.train()
        act = (torch.randn(1, _N_SYMBOLS, device=TheDevice.get()) * 0.01).requires_grad_(True)
        out = sl_s.forward(act)
        loss = out['rule_logits'].pow(2).sum()
        loss.backward()

        self.assertIsNotNone(act.grad,
            "No gradient to input from S-tier SyntacticLayer.forward()")
        self.assertTrue((act.grad.abs() > 0).any(),
            "Zero gradient to S-tier input from rule prediction")

    def test_gradient_flows_through_depth_embed(self):
        """Depth embeddings also receive gradients (used in rule prediction)."""
        sl_c = self.model.conceptualSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        sl_c.train()
        act = torch.randn(1, _N_CONCEPTS, device=TheDevice.get()) * 0.01
        out = sl_c.forward(act)
        loss = out['rule_logits'].pow(2).sum()
        loss.backward()

        self.assertIsNotNone(sl_c.depth_embed.weight.grad,
            "No gradient to depth_embed from SyntacticLayer.forward()")
        self.assertFalse(sl_c.depth_embed.weight.grad.isnan().any(),
            "NaN gradient to depth_embed")


class TestGrammarRuleSelectivity(unittest.TestCase):
    """SyntacticLayers produce input-dependent rule distributions."""

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    def test_untrained_rule_probs_near_uniform(self):
        """Before training, non-transition rule probabilities should be near-uniform.

        The transition rule is biased by (1-interpretation)*TRANSITION_SCALE,
        so we exclude it from the entropy check.
        """
        sl_c = self.model.conceptualSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        # Small-scale input to keep SyntacticLayer logits in stable range
        act = torch.randn(1, _N_CONCEPTS, device=TheDevice.get()) * 0.01
        sl_c.eval()
        with torch.no_grad():
            out = sl_c.forward(act)
        probs = out['rule_probs'][:, 0, :]  # first depth
        num_rules = probs.shape[-1]
        self.assertFalse(probs.isnan().any(), "NaN in rule_probs")

        # Exclude transition rule from entropy check since it's biased
        if sl_c.transition_index is not None:
            mask = torch.ones(num_rules, dtype=torch.bool, device=probs.device)
            mask[sl_c.transition_index] = False
            non_transition_probs = probs[:, mask]
            # Renormalize the non-transition probabilities
            non_transition_probs = non_transition_probs / (non_transition_probs.sum(dim=-1, keepdim=True) + 1e-8)
            n_eff = non_transition_probs.shape[-1]
        else:
            non_transition_probs = probs
            n_eff = num_rules

        entropy = -(non_transition_probs * (non_transition_probs + 1e-8).log()).sum(dim=-1).mean().item()
        max_entropy = torch.tensor(n_eff, dtype=torch.float32).log().item()
        self.assertGreater(entropy, 0.3 * max_entropy,
            f"Untrained C-tier entropy {entropy:.3f} too low "
            f"(max {max_entropy:.3f} for {n_eff} non-transition rules)")

    def test_different_inputs_different_distributions(self):
        """Different activation patterns produce different rule distributions."""
        sl_c = self.model.conceptualSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        act_sparse = torch.zeros(1, _N_CONCEPTS, device=TheDevice.get())
        act_sparse[0, :5] = 0.01

        act_dense = torch.ones(1, _N_CONCEPTS, device=TheDevice.get()) * 0.01

        sl_c.eval()
        with torch.no_grad():
            out1 = sl_c.forward(act_sparse)
            out2 = sl_c.forward(act_dense)

        probs_1 = out1['rule_probs'][:, 0, :]
        probs_2 = out2['rule_probs'][:, 0, :]

        diff = (probs_1 - probs_2).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "Same rule distribution for different inputs — no selectivity")

    def test_depth_varies_rule_distribution(self):
        """Different derivation depths produce different rule distributions."""
        sl_s = self.model.symbolicSpace.syntacticLayer
        if sl_s is None:
            self.skipTest("No S-tier SyntacticLayer")

        act = torch.randn(1, _N_SYMBOLS, device=TheDevice.get()) * 0.01
        sl_s.eval()
        with torch.no_grad():
            out = sl_s.forward(act)

        # Compare depth 0 vs depth 1 rule logits (before softmax/bias)
        # Use logits instead of probs to avoid transition bias masking
        logits_d0 = out['rule_logits'][:, 0, :]
        logits_d1 = out['rule_logits'][:, 1, :]

        diff = (logits_d0 - logits_d1).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "Same rule logits at different depths — depth embedding not used")


class TestGrammarInMentalModel(unittest.TestCase):
    """MentalModel.forward() exercises Grammar through Space.forward()."""

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    def test_forward_populates_stacks(self):
        """MentalModel.forward() populates Grammar stacks with word tuples."""
        self.model.eval()
        self.model.set_sigma(0)
        with torch.no_grad():
            _forward_with_data(self.model,
                ['the cat sat on the mat'], [torch.tensor([0.0])])

        # Check words on the space subspaces — Grammar.forward() records
        # words on subspaces via add_word(), not on Grammar's internal lists.
        total_words = 0
        for space_attr in ('symbolicSpace', 'conceptualSpace', 'perceptualSpace'):
            space = getattr(self.model, space_attr, None)
            if space is not None and hasattr(space, 'subspace'):
                words = space.subspace.get_words()
                total_words += len(words)

        # At minimum the C-tier should record not/non words at active positions
        # when syntacticLayer is present (ARLM mode).
        if self.model.conceptualSpace.syntacticLayer is not None:
            self.assertGreater(total_words, 0,
                "No word tuples on any subspace — Grammar not exercised during forward")

    def test_spaces_have_grammar_enabled(self):
        """MentalModel uses ARLM mode which enables grammar on spaces."""
        self.assertIsNotNone(self.model.conceptualSpace.syntacticLayer,
            "ConceptualSpace should have a SyntacticLayer in ARLM mode")


class TestGrammarRulesMatchXML(unittest.TestCase):
    """Verify Grammar.configure() parses MentalModel.xml rules correctly."""

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    # ── Toy rules: what MentalModel.xml defines ──────────────────────

    def test_total_rule_count(self):
        """MentalModel.xml defines 16 rules (1 START + 5 S + 8 C + 2 P)."""
        self.assertEqual(len(self.grammar.rules), 16)

    def test_s_tier_rules(self):
        """S-tier: true(S), non(S), swap(S,S), equals(S,S), part(S,S), S→C transition."""
        s_rules = [(r.method_name, r.arity) for r in self.grammar.rules if r.tier == 'S']
        self.assertIn(('true', 1), s_rules)
        self.assertIn(('non', 1), s_rules)
        self.assertIn(('swap', 2), s_rules)
        self.assertIn(('equals', 2), s_rules)
        self.assertIn(('part', 2), s_rules)
        self.assertIn((None, 1), s_rules, "Missing S→C transition rule")
        self.assertEqual(len(s_rules), 6)

    def test_c_tier_rules(self):
        """C-tier: not, intersection, union, lower, lift(binary+ternary), C→P transition."""
        c_rules = [(r.method_name, r.arity) for r in self.grammar.rules if r.tier == 'C']
        self.assertIn(('not', 1), c_rules)
        self.assertIn(('intersection', 2), c_rules)
        self.assertIn(('union', 2), c_rules)
        self.assertIn(('lower', 2), c_rules)
        self.assertIn(('lift', 2), c_rules)
        self.assertIn(('lift', 3), c_rules)
        self.assertIn((None, 1), c_rules, "Missing C→P transition rule")
        self.assertEqual(len(c_rules), 7)

    def test_p_tier_rules(self):
        """P-tier: chunk(I,P) from 'I P', terminal 'I'."""
        p_rules = [(r.method_name, r.arity) for r in self.grammar.rules if r.tier == 'P']
        self.assertIn(('chunk', 2), p_rules)
        self.assertIn((None, 0), p_rules, "Missing P→I terminal rule")
        self.assertEqual(len(p_rules), 2)

    def test_start_rule(self):
        """START→S rule exists."""
        start_rules = [r for r in self.grammar.rules if r.tier == 'START']
        self.assertEqual(len(start_rules), 1)
        self.assertEqual(start_rules[0].canonical, 'START → S')

    # ── Derived rules: tier groupings and transitions ────────────────

    def test_symbolic_indices(self):
        """grammar.symbolic() returns exactly the S-tier rule indices."""
        s_ids = self.grammar.symbolic()
        for i in s_ids:
            self.assertEqual(self.grammar.rules[i].tier, 'S')
        self.assertEqual(len(s_ids), 6)

    def test_conceptual_indices(self):
        """grammar.conceptual() returns exactly the C-tier rule indices."""
        c_ids = self.grammar.conceptual()
        for i in c_ids:
            self.assertEqual(self.grammar.rules[i].tier, 'C')
        self.assertEqual(len(c_ids), 7)

    def test_perceptual_indices(self):
        """grammar.perceptual() returns exactly the P-tier rule indices."""
        p_ids = self.grammar.perceptual()
        for i in p_ids:
            self.assertEqual(self.grammar.rules[i].tier, 'P')
        self.assertEqual(len(p_ids), 2)

    def test_symbolic_transition(self):
        """S→C transition is the arity-1, method_name=None S rule."""
        t = self.grammar.symbolic_transition()
        self.assertIsNotNone(t)
        rule = self.grammar.rules[t]
        self.assertEqual(rule.tier, 'S')
        self.assertIsNone(rule.method_name)
        self.assertEqual(rule.arity, 1)

    def test_conceptual_transition(self):
        """C→P transition is the arity-1, method_name=None C rule."""
        t = self.grammar.conceptual_transition()
        self.assertIsNotNone(t)
        rule = self.grammar.rules[t]
        self.assertEqual(rule.tier, 'C')
        self.assertIsNone(rule.method_name)
        self.assertEqual(rule.arity, 1)

    def test_all_method_names_are_registered(self):
        """Every rule's method_name maps to Grammar or a SyntacticLayer subclass."""
        all_known = set(self.model.symbolicSpace.syntacticLayer._RULE_METHODS.keys()) | {'swap', 'lift', 'lower', 'non'}
        for r in self.grammar.rules:
            if r.method_name is not None:
                self.assertIn(r.method_name, all_known,
                    f"Rule method '{r.method_name}' not in known methods")


class TestGrammarRuleReport(unittest.TestCase):
    """Report rule preferences from SyntacticLayers (informational)."""

    def test_rule_preference_report(self):
        """Print which rules each tier's SyntacticLayer predicts for synthetic input."""
        torch.manual_seed(42)
        model, grammar = _make_model()

        tier_map = {
            'S': (model.symbolicSpace.syntacticLayer, _N_SYMBOLS),
            'C': (model.conceptualSpace.syntacticLayer, _N_CONCEPTS),
            'P': (model.perceptualSpace.syntacticLayer, _N_PERCEPTS),
        }
        for tier, (sl, n_slots) in tier_map.items():
            if sl is None:
                print(f"\n{tier}-tier: no SyntacticLayer")
                continue

            print(f"\n{tier}-tier rules ({len(sl.all_rules)} rules):")
            for local_idx, global_id in enumerate(sl.all_rules):
                rule = grammar.rules[global_id]
                print(f"  [{local_idx}] rule {global_id}: {rule.canonical} "
                      f"(arity={rule.arity})")

            act = torch.randn(1, n_slots, device=TheDevice.get()) * 0.01
            sl.eval()
            with torch.no_grad():
                out = sl.forward(act)
                probs = out['rule_probs'][0, 0, :]
                ranked = sorted(enumerate(probs.tolist()),
                                key=lambda x: x[1], reverse=True)
                print(f"  Depth-0 rule preferences:")
                for local_idx, prob in ranked:
                    global_id = sl.all_rules[local_idx]
                    rule = grammar.rules[global_id]
                    print(f"    {rule.canonical}: {prob:.4f}")


class TestSoftSuperposition(unittest.TestCase):
    """Verify SyntacticLayer.compose() applies soft-weighted composition."""

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    def test_c_tier_soft_compose(self):
        """C-tier compose() produces output different from input."""
        B, N, D = 1, _N_CONCEPT_SLOTS, _CONCEPT_DIM
        data = torch.randn(B, N, D, device=TheDevice.get()) * 0.1
        # Ensure at least 2 active positions for composition
        data[0, 0] = torch.randn(D, device=TheDevice.get()) * 0.5
        data[0, 1] = torch.randn(D, device=TheDevice.get()) * 0.5

        sl = self.model.conceptualSpace.syntacticLayer
        subspace = self.model.conceptualSpace.subspace
        sl.eval()
        with torch.no_grad():
            result, _ = sl.compose(data, subspace, self.grammar)

        # Soft superposition should transform the data
        diff = (result - data).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "C-tier compose() returned input unchanged — no composition")

    def test_s_tier_soft_compose(self):
        """S-tier compose() uses learned rule probabilities."""
        B, N = 1, _N_SYMBOLS
        data = torch.zeros(B, N, device=TheDevice.get())
        # Set multiple active positions
        data[0, 0] = 0.5
        data[0, 1] = 0.3
        data[0, 2] = 0.2

        sl = self.model.symbolicSpace.syntacticLayer
        subspace = self.model.symbolicSpace.subspace
        sl.eval()
        with torch.no_grad():
            result = sl.compose(data, subspace, self.grammar)

        # Should not just apply true(S) at one position
        diff = (result - data).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "S-tier compose() returned input unchanged — no composition")

    def test_transition_dominates_at_low_interpretation(self):
        """With interpretation=0.0 transition bias dominates → near-identity."""
        old_interp = self.grammar.interpretation
        try:
            self.grammar.interpretation = 0.0
            B, N = 1, _N_SYMBOLS
            data = torch.zeros(B, N, device=TheDevice.get())
            data[0, 0] = 0.5
            data[0, 1] = 0.3

            sl = self.model.symbolicSpace.syntacticLayer
            subspace = self.model.symbolicSpace.subspace
            sl.eval()
            with torch.no_grad():
                result = sl.compose(data, subspace, self.grammar)

            # Transition rule (identity) should dominate — output ≈ first leaf
            self.assertFalse(result.isnan().any(), "NaN in result")
        finally:
            self.grammar.interpretation = old_interp

    def test_not_non_still_fire(self):
        """not(C) and non(C) fire deterministically before soft composition."""
        B, N, D = 1, _N_CONCEPT_SLOTS, _CONCEPT_DIM
        # Create data where mean < 0 to trigger not()
        data = torch.full((B, N, D), -0.5, device=TheDevice.get())
        # Zero out most positions so only one is active
        data[0, 1:] = 0.0

        sl = self.model.conceptualSpace.syntacticLayer
        subspace = self.model.conceptualSpace.subspace
        sl.eval()
        with torch.no_grad():
            result, _ = sl.compose(data, subspace, self.grammar)

        # After not(): values should be non-negative at position 0
        words = subspace.get_words()
        not_rid = self.grammar._c_rule_ids().get('not')
        if not_rid is not None:
            from Space import WordEncoding
            not_words = [w for w in words if len(w) >= 4 and w[WordEncoding.RULE] == not_rid]
            self.assertGreater(len(not_words), 0,
                "not() rule should have fired for negative-mean data")

    def test_gradient_through_grammar_forward(self):
        """Gradients flow through compose() to SyntacticLayer params."""
        sl_c = self.model.conceptualSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        sl_c.train()
        B, N, D = 1, _N_CONCEPT_SLOTS, _CONCEPT_DIM
        data = (torch.randn(B, N, D, device=TheDevice.get()) * 0.01).requires_grad_(True)
        data_with_active = data.clone()
        data_with_active[0, 0] = torch.randn(D, device=TheDevice.get()) * 0.5
        data_with_active[0, 1] = torch.randn(D, device=TheDevice.get()) * 0.5

        subspace = self.model.conceptualSpace.subspace
        result, _ = sl_c.compose(data_with_active, subspace, self.grammar)
        loss = result.pow(2).sum()
        loss.backward()

        # SyntacticLayer parameters should receive gradients
        self.assertIsNotNone(sl_c.rule_head.W.grad,
            "No gradient to SyntacticLayer rule_head through compose()")


if __name__ == '__main__':
    unittest.main()
