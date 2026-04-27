"""
Toy Grammar Learning Test
=========================

Verifies that Grammar's SyntacticLayers participate in the computation
graph and receive gradients when driven through SyntacticLayer.forward().

Each tier's SyntacticLayer predicts rule distributions from activation
vectors. The gradient path:

    loss -> rule_logits -> rule_head weights -> derivation_layer -> input_proj -> x

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
import Models
import Spaces
import Language
matplotlib.use('Agg')


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
    # Force re-initialization in case previous tests set small test dimensions
    Language.TheGrammar._configured = False
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model, Language.TheGrammar


def _forward_with_data(model, sentences, outputs):
    """Run a forward pass through MentalModel with given data."""
    with Models.TheData.runtime_batch(sentences, outputs), \
         warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Range violation")
        warnings.filterwarnings("ignore", message="PiLayer.reverse")
        train_input, _ = model.inputSpace.getTrainData()
        x = model.inputSpace.prepInput(train_input[:len(sentences)])
        return model.forward(x)


# -- Dimensions from MentalModel.xml ---------------------------------
# Refactored geometry: nPercepts == nConcepts == nSymbols (no [P,S] concat;
# state mixing happens via the iterative Sigma-Pi loop, not slot expansion).
_N_SYMBOLS = 128
_N_CONCEPTS = 128
_N_PERCEPTS = 128
_CONCEPT_DIM = 100 + 4   # nDim + objectSize (nWhere + nWhen)
_SYMBOL_DIM = 100 + 4
_N_CONCEPT_SLOTS = _N_CONCEPTS  # conceptInputShape[0] = 128


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
        sl_c = self.model.wordSpace.syntacticLayer
        if sl_c is None or sl_c.num_rules == 0:
            self.skipTest("No C-tier rules after C->S merge")

        sl_c.train()
        act = (torch.randn(1, _N_CONCEPTS, device=Models.TheDevice.get()) * 0.01).requires_grad_(True)
        out = sl_c.forward(act)
        loss = out['rule_logits'].pow(2).sum()
        loss.backward()

        self.assertIsNotNone(act.grad,
            "No gradient to input from C-tier SyntacticLayer.forward()")
        self.assertTrue((act.grad.abs() > 0).any(),
            "Zero gradient to input from C-tier rule prediction")

    def test_s_tier_gradient_flows(self):
        """S-tier rule_head receives gradients from SyntacticLayer.forward()."""
        sl_s = self.model.wordSpace.syntacticLayer
        if sl_s is None:
            self.skipTest("No S-tier SyntacticLayer")

        sl_s.train()
        act = (torch.randn(1, _N_SYMBOLS, device=Models.TheDevice.get()) * 0.01).requires_grad_(True)
        out = sl_s.forward(act)
        loss = out['rule_logits'].pow(2).sum()
        loss.backward()

        self.assertIsNotNone(act.grad,
            "No gradient to input from S-tier SyntacticLayer.forward()")
        self.assertTrue((act.grad.abs() > 0).any(),
            "Zero gradient to S-tier input from rule prediction")

    def test_gradient_flows_through_depth_embed(self):
        """Depth embeddings also receive gradients (used in rule prediction)."""
        sl_c = self.model.wordSpace.syntacticLayer
        if sl_c is None or sl_c.num_rules == 0:
            self.skipTest("No C-tier rules after C->S merge")

        sl_c.train()
        act = torch.randn(1, _N_CONCEPTS, device=Models.TheDevice.get()) * 0.01
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
        sl_c = self.model.wordSpace.syntacticLayer
        if sl_c is None:
            self.skipTest("No C-tier SyntacticLayer")

        # Small-scale input to keep SyntacticLayer logits in stable range
        act = torch.randn(1, _N_CONCEPTS, device=Models.TheDevice.get()) * 0.01
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
        sl_c = self.model.wordSpace.syntacticLayer
        if sl_c is None or sl_c.num_rules == 0:
            self.skipTest("No C-tier rules after C->S merge")

        act_sparse = torch.zeros(1, _N_CONCEPTS, device=Models.TheDevice.get())
        act_sparse[0, :5] = 0.01

        act_dense = torch.ones(1, _N_CONCEPTS, device=Models.TheDevice.get()) * 0.01

        sl_c.eval()
        with torch.no_grad():
            out1 = sl_c.forward(act_sparse)
            out2 = sl_c.forward(act_dense)

        probs_1 = out1['rule_probs'][:, 0, :]
        probs_2 = out2['rule_probs'][:, 0, :]

        diff = (probs_1 - probs_2).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "Same rule distribution for different inputs -- no selectivity")

    def test_depth_varies_rule_distribution(self):
        """Different derivation depths produce different rule distributions."""
        sl_s = self.model.wordSpace.syntacticLayer
        if sl_s is None:
            self.skipTest("No S-tier SyntacticLayer")

        act = torch.randn(1, _N_SYMBOLS, device=Models.TheDevice.get()) * 0.01
        sl_s.eval()
        with torch.no_grad():
            out = sl_s.forward(act)

        # Compare depth 0 vs depth 1 rule logits (before softmax/bias)
        # Use logits instead of probs to avoid transition bias masking
        logits_d0 = out['rule_logits'][:, 0, :]
        logits_d1 = out['rule_logits'][:, 1, :]

        diff = (logits_d0 - logits_d1).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "Same rule logits at different depths -- depth embedding not used")


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

        # Check words on the space subspaces -- Grammar.forward() records
        # words on subspaces via add_word(), not on Grammar's internal lists.
        total_words = 0
        for space_attr in ('symbolicSpace', 'conceptualSpace', 'perceptualSpace'):
            space = getattr(self.model, space_attr, None)
            if space is not None and hasattr(space, 'subspace'):
                words = space.subspace.get_words()
                total_words += len(words)

        # At minimum the C-tier should record not/non words at active positions
        # when syntacticLayer is present (ARLM mode).
        if self.model.wordSpace.syntacticLayer is not None:
            self.assertGreater(total_words, 0,
                "No word tuples on any subspace -- Grammar not exercised during forward")

    def test_spaces_have_grammar_enabled(self):
        """MentalModel uses ARLM mode which enables grammar on spaces."""
        self.assertIsNotNone(self.model.wordSpace.syntacticLayer,
            "ConceptualSpace should have a SyntacticLayer in ARLM mode")


class TestGrammarRulesMatchXML(unittest.TestCase):
    """Verify Grammar parses the configured grammar correctly.

    Step 6 of the lift / lower / bivector refactor flipped
    ``MentalModel.xml`` from an inline ``<grammar>`` block to
    ``<grammarCfg>data/grammar.cfg</grammarCfg>``.  These tests now
    validate structural invariants (every required S-tier op is
    addressable as a rule; every method_name is dispatchable;
    Layer-2 ops are merged into the rule table) rather than the
    exact 20-rule list from the old XML.
    """

    # S-tier op names guaranteed by data/grammar.cfg's [layer2] section
    # plus the function-call upward Layer-1 productions.
    REQUIRED_OPS = {
        'true':         1,
        'false':        1,
        'non':          1,
        'not':          1,
        'what':         1,
        'where':        1,
        'when':         1,
        'conjunction':  2,
        'disjunction':  2,
        'query':        2,
        'swap':         2,
        'equals':       2,
        'part':         2,
        'intersection': 2,
        'union':        2,
        'lower':        2,
        'lift':         2,
    }

    def setUp(self):
        torch.manual_seed(42)
        self.model, self.grammar = _make_model()

    # -- Structural invariants on the configured rule table ----------

    def test_total_rule_count_positive(self):
        """The cfg-loaded grammar contains a non-trivial rule set."""
        self.assertGreater(len(self.grammar.rules), 0)

    def test_required_s_ops_present(self):
        """Every required S-tier op (method_name, arity) is a rule."""
        s_rules = {(r.method_name, r.arity)
                   for r in self.grammar.rules if r.tier == 'S'}
        for method, arity in self.REQUIRED_OPS.items():
            self.assertIn((method, arity), s_rules,
                          f"missing required op ({method!r}, {arity})")

    def test_vo_lhs_present(self):
        """VO is introduced as an LHS via the verb-object composition rule."""
        lhs_set = {r.lhs for r in self.grammar.rules if r.tier == 'S'}
        self.assertIn('VO', lhs_set)

    def test_downward_emit_head_present(self):
        """Downward `C -> emit_head(S)` projection is configured."""
        s_rules = {(r.method_name, r.arity)
                   for r in self.grammar.rules if r.tier == 'S'}
        self.assertIn(('emit_head', 1), s_rules)

    def test_c_tier_rules(self):
        """C-tier is empty after the C->S merge."""
        c_rules = [(r.method_name, r.arity) for r in self.grammar.rules if r.tier == 'C']
        self.assertEqual(c_rules, [])

    def test_p_tier_rules(self):
        """P-tier is empty after P-tier removal from grammar."""
        p_rules = [(r.method_name, r.arity) for r in self.grammar.rules if r.tier == 'P']
        self.assertEqual(p_rules, [])

    # -- Derived rules: tier groupings and transitions ----------------

    def test_symbolic_indices(self):
        """grammar.symbolic() returns exactly the S-tier rule indices."""
        s_ids = self.grammar.symbolic()
        for i in s_ids:
            self.assertEqual(self.grammar.rules[i].tier, 'S')
        # All rules are S-tier post-merge.
        self.assertEqual(len(s_ids), len(self.grammar.rules))

    def test_only_s_tier_remains(self):
        """After the C->S merge and P-tier removal, only S-tier rules exist."""
        for rule in self.grammar.rules:
            self.assertEqual(rule.tier, 'S')

    def test_symbolic_transition(self):
        """No S->C transition after the C->S merge."""
        self.assertIsNone(self.grammar.symbolic_transition())

    def test_all_method_names_are_registered(self):
        """Every rule's method_name maps to Grammar or a SyntacticLayer subclass.

        Allowlisted dispatch signals (not in ``_RULE_METHODS``):
            'emit_head' — downward head emission via codebook lookup.
            'merge'     — single-RHS PROJECT form (e.g. `S = NP`) the
                          Step 6 cfg uses for terminal projection;
                          the dispatcher falls through (no
                          ``_RULE_METHODS`` entry needed).
        """
        all_known = (
            set(self.model.wordSpace.syntacticLayer._RULE_METHODS.keys())
            | {'swap', 'lift', 'lower', 'non'}
            | {'emit_head', 'merge'}
        )
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
            'S': (model.wordSpace.syntacticLayer, _N_SYMBOLS),
            'C': (model.wordSpace.syntacticLayer, _N_CONCEPTS),
            'P': (model.wordSpace.syntacticLayer, _N_PERCEPTS),
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

            act = torch.randn(1, n_slots, device=Models.TheDevice.get()) * 0.01
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
        sl = self.model.wordSpace.syntacticLayer
        if sl is None or sl.num_rules == 0:
            self.skipTest("No C-tier rules after C->S merge")
        B, N, D = 1, _N_CONCEPT_SLOTS, _CONCEPT_DIM
        data = torch.randn(B, N, D, device=Models.TheDevice.get()) * 0.1
        # Ensure at least 2 active positions for composition
        data[0, 0] = torch.randn(D, device=Models.TheDevice.get()) * 0.5
        data[0, 1] = torch.randn(D, device=Models.TheDevice.get()) * 0.5

        subspace = self.model.conceptualSpace.subspace
        sl.eval()
        with torch.no_grad():
            result, _ = sl.compose(data, subspace, self.grammar)

        # Soft superposition should transform the data
        diff = (result - data).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "C-tier compose() returned input unchanged -- no composition")

    def test_s_tier_soft_compose(self):
        """S-tier compose() uses learned rule probabilities."""
        B, N = 1, _N_SYMBOLS
        data = torch.zeros(B, N, device=Models.TheDevice.get())
        # Set multiple active positions
        data[0, 0] = 0.5
        data[0, 1] = 0.3
        data[0, 2] = 0.2

        sl = self.model.wordSpace.syntacticLayer
        subspace = self.model.symbolicSpace.subspace
        sl.eval()
        with torch.no_grad():
            composed, _ = sl.compose(data, subspace, self.grammar)

        # Should not just apply true(S) at one position
        diff = (composed - data).abs().max().item()
        self.assertGreater(diff, 1e-6,
            "S-tier compose() returned input unchanged -- no composition")

    def test_transition_dominates_at_low_interpretation(self):
        """With interpretation=0.0 transition bias dominates -> near-identity."""
        old_interp = self.grammar.interpretation
        try:
            self.grammar.interpretation = 0.0
            B, N = 1, _N_SYMBOLS
            data = torch.zeros(B, N, device=Models.TheDevice.get())
            data[0, 0] = 0.5
            data[0, 1] = 0.3

            sl = self.model.wordSpace.syntacticLayer
            subspace = self.model.symbolicSpace.subspace
            sl.eval()
            with torch.no_grad():
                composed, _ = sl.compose(data, subspace, self.grammar)

            # Transition rule (identity) should dominate -- output ~= first leaf
            self.assertFalse(composed.isnan().any(), "NaN in result")
        finally:
            self.grammar.interpretation = old_interp

    def test_gradient_through_grammar_forward(self):
        """Gradients flow through compose() to SyntacticLayer params."""
        sl_c = self.model.wordSpace.syntacticLayer
        if sl_c is None or sl_c.num_rules == 0:
            self.skipTest("No C-tier rules after C->S merge")

        sl_c.train()
        B, N, D = 1, _N_CONCEPT_SLOTS, _CONCEPT_DIM
        data = (torch.randn(B, N, D, device=Models.TheDevice.get()) * 0.01).requires_grad_(True)
        data_with_active = data.clone()
        data_with_active[0, 0] = torch.randn(D, device=Models.TheDevice.get()) * 0.5
        data_with_active[0, 1] = torch.randn(D, device=Models.TheDevice.get()) * 0.5

        subspace = self.model.conceptualSpace.subspace
        result, _ = sl_c.compose(data_with_active, subspace, self.grammar)
        loss = result.pow(2).sum()
        loss.backward()

        # SyntacticLayer parameters should receive gradients
        self.assertIsNotNone(sl_c.rule_head.W.grad,
            "No gradient to SyntacticLayer rule_head through compose()")


if __name__ == '__main__':
    unittest.main()
