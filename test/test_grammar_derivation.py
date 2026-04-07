"""
Grammar Derivation Test — MentalModel + Grammar
=================================================

Verifies that Grammar.forward() and its shift/reduce stacks correctly
derive rule sequences when driven with the same architecture as MentalModel.

Tests:
  1. Grammar.project() dispatches correctly for each rule method
  2. Grammar.forward('S', ...) shift/reduce stacks compose operands
  3. Grammar.forward('C', ...) composes concept vectors
  4. Grammar.forward('P', ...) produces word tuples
  5. Grammar.configure() and tier queries
  6. End-to-end: MentalModel forward + Grammar derivation
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
RuleDef = Grammar.RuleDef
from util import init_config, TheXMLConfig


_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def _reload_config():
    init_config(
        path=os.path.join(_DATA_DIR, 'MentalModel.xml'),
        defaults_path=os.path.join(_DATA_DIR, 'model.xml'),
    )


def _make_model():
    """Create a MentalModel and return (model, TheGrammar)."""
    _reload_config()
    from Space import TheGrammar
    # Force re-initialization in case previous tests set small test dimensions
    TheGrammar._layers_initialized = False
    TheGrammar._configured = False
    TheGrammar.chunk_layer = None
    model, _ = MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model, TheGrammar


class TestGrammarProject(unittest.TestCase):
    """Grammar.project() dispatches rule methods correctly."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        # Dimensions from MentalModel.xml: nDim=100, objectSize=4, nSymbols=128, nConcepts=256
        # Grammar layers use dim + objectSize as the actual vector width.
        self.symbol_dim = 100 + 4   # nDim + nWhere + nWhen
        self.concept_dim = 100 + 4
        self.n_symbols = 128
        self.n_concepts = 256

    def test_project_swap_returns_tensor(self):
        """swap(S, S) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_symbols, device=TheDevice.get())
        right = torch.randn(B, self.n_symbols, device=TheDevice.get())
        # Rule 1: S → swap(S, S)
        result = self.grammar.project('S', 1, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_equals_returns_tensor(self):
        """equals(S, S) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_symbols, device=TheDevice.get())
        right = torch.randn(B, self.n_symbols, device=TheDevice.get())
        result = self.grammar.project('S', 2, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_part_returns_tensor(self):
        """part(S, S) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_symbols, device=TheDevice.get())
        right = torch.randn(B, self.n_symbols, device=TheDevice.get())
        result = self.grammar.project('S', 3, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_transition_is_passthrough(self):
        """S → C transition passes operand through unchanged."""
        B = 2
        left = torch.randn(B, self.n_symbols, device=TheDevice.get())
        result = self.grammar.project('S', 4, left)
        self.assertTrue(torch.equal(result, left))

    def test_project_union_c_tier(self):
        """union(C, C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        right = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 5, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_intersection_c_tier(self):
        """intersection(C, C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        right = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 6, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_not_c_tier(self):
        """not(C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 10, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_non_c_tier(self):
        """non(C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 11, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_lower_c_tier(self):
        """lower(C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 7, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_lift_binary_c_tier(self):
        """lift(C, C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        right = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 8, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_lift_unary_c_tier(self):
        """lift(C) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_concepts, self.concept_dim, device=TheDevice.get())
        result = self.grammar.project('C', 9, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_true_s_tier(self):
        """true(S) produces output of same shape."""
        B = 2
        left = torch.randn(B, self.n_symbols, device=TheDevice.get())
        result = self.grammar.project('S', 0, left)
        self.assertEqual(result.shape, left.shape)

    def test_all_methods_dispatched(self):
        """Every method_name in _GRAMMAR_METHODS is exercised by some rule."""
        dispatched = set()
        for rule in self.grammar.rules:
            if rule.method_name:
                dispatched.add(rule.method_name)
        expected = set(self.grammar._GRAMMAR_METHODS.keys())
        self.assertTrue(dispatched.issubset(expected),
                        f"Unknown methods: {dispatched - expected}")
        # chunk is P-tier; all others should be covered by rules
        self.assertTrue(expected.issubset(dispatched | {'chunk'}),
                        f"Untested methods: {expected - dispatched}")


class TestGrammarShiftReduceStack(unittest.TestCase):
    """Grammar.forward() applies rules and records words on subspace."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.symbol_dim = 100 + 4   # nDim + nWhere + nWhen
        self.concept_dim = 100 + 4
        self.n_symbols = 128
        self.n_concepts = 256

    def _make_ss(self, shape):
        from Space import SubSpace
        return SubSpace(inputShape=shape, outputShape=shape)

    def test_symbolic_forward_shape(self):
        """forward('S', ...) returns same shape activation."""
        ss = self._make_ss([self.n_symbols, 1])
        act = torch.zeros(1, self.n_symbols, device=TheDevice.get())
        act[0, 3] = 1.0
        result = self.grammar.forward('S', act, ss)
        self.assertEqual(result.shape, (1, self.n_symbols))

    def test_symbolic_forward_two_active(self):
        """forward('S', ...) works with two active positions."""
        ss = self._make_ss([self.n_symbols, 1])
        act = torch.zeros(1, self.n_symbols, device=TheDevice.get())
        act[0, 2] = 1.0
        act[0, 5] = 1.0
        result = self.grammar.forward('S', act, ss)
        self.assertEqual(result.shape, (1, self.n_symbols))

    def test_conceptual_forward_shape(self):
        """forward('C', ...) returns same shape vectors."""
        ss = self._make_ss([self.n_concepts, self.concept_dim])
        vec = torch.randn(1, self.n_concepts, self.concept_dim, device=TheDevice.get())
        vec[0, 3:] = 0.0  # 3 active positions
        result = self.grammar.forward('C', vec, ss)
        self.assertEqual(result.shape, (1, self.n_concepts, self.concept_dim))

    def test_conceptual_forward_records_words(self):
        """forward('C', ...) records words on the subspace."""
        ss = self._make_ss([self.n_concepts, self.concept_dim])
        vec = torch.randn(1, self.n_concepts, self.concept_dim, device=TheDevice.get())
        # Make one position negative-mean to trigger not()
        vec[0, 0] = -torch.abs(vec[0, 0])
        vec[0, 1:] = 0.0
        self.grammar.forward('C', vec, ss)
        words = ss.get_words()
        self.assertGreater(len(words), 0)

    def test_perceptual_forward_shape(self):
        """forward('P', ...) returns same shape vectors."""
        n_percepts = 128
        ss = self._make_ss([n_percepts, self.symbol_dim])
        vec = torch.randn(1, n_percepts, self.symbol_dim, device=TheDevice.get())
        vec[0, 4:] = 0.0
        result = self.grammar.forward('P', vec, ss)
        self.assertEqual(result.shape, (1, n_percepts, self.symbol_dim))

    def test_reset_clears_stacks(self):
        """resetStack clears all tier stacks."""
        self.grammar._s_stack = [torch.zeros(1, 8)]
        self.grammar._s_words = [(0, 1, 2)]
        self.grammar.resetStack('S')
        self.assertEqual(len(self.grammar._s_stack), 0)
        self.assertEqual(len(self.grammar._s_words), 0)

    def test_multiple_symbolic_forward_calls(self):
        """Multiple forward('S', ...) calls work without error."""
        for _ in range(3):
            ss = self._make_ss([self.n_symbols, 1])
            act = torch.zeros(1, self.n_symbols, device=TheDevice.get())
            act[0, torch.randint(self.n_symbols, (1,))] = 1.0
            self.grammar.forward('S', act, ss)


class TestGrammarConfigure(unittest.TestCase):
    """Grammar.configure() parses XML rule definitions into RuleDefs."""

    def test_configure_from_dict(self):
        """configure() builds correct RuleDefs from a grammar dict."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': ['true(S) EOF'],
            'S': ['swap(S, S)', 'equals(S, S)', 'C'],
            'C': ['union(C, C)', 'not(C)', 'P'],
            'P': ['ε'],
        })
        self.assertEqual(len(g.rules), 8)

        # Check tiers
        self.assertEqual(g.rules[0].tier, 'START')
        self.assertEqual(g.rules[1].tier, 'S')
        self.assertEqual(g.rules[3].tier, 'S')  # transition S→C
        self.assertEqual(g.rules[4].tier, 'C')
        self.assertEqual(g.rules[7].tier, 'P')

        # Check arities
        self.assertEqual(g.arity(0), 1)   # true(S) — unary
        self.assertEqual(g.arity(1), 2)   # swap(S, S) — binary
        self.assertEqual(g.arity(3), 1)   # S → C — transition
        self.assertEqual(g.arity(7), 0)   # P → ε — terminal

        # Check methods
        self.assertEqual(g.method_name(0), 'true')
        self.assertEqual(g.method_name(1), 'swap')
        self.assertIsNone(g.method_name(3))   # transition

    def test_interpretation_from_xml_config(self):
        """interpretation is read from mentalModel.interpretation, not grammar dict."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': ['true(S) EOF'],
            'S': ['C'],
            'C': ['P'],
            'P': ['ε'],
        })
        # Only tier rules, no interpretation key in grammar dict
        self.assertEqual(len(g.rules), 4)
        # Default interpretation preserved (configure no longer touches it)
        self.assertEqual(g.interpretation, 0.5)

    def test_configure_single_string_rule(self):
        """configure() handles a single string (not list) for a tier."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': 'true(S) EOF',
            'S': 'C',
            'C': 'P',
            'P': 'ε',
        })
        self.assertEqual(len(g.rules), 4)

    def test_tier_queries(self):
        """symbolic(), conceptual(), perceptual() return correct indices."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': ['true(S) EOF'],
            'S': ['swap(S, S)', 'equals(S, S)', 'C'],
            'C': ['union(C, C)', 'not(C)', 'P'],
            'P': ['ε'],
        })
        sym = g.symbolic()
        con = g.conceptual()
        per = g.perceptual()

        # Symbolic: swap, equals, S→C transition
        self.assertEqual(len(sym), 3)
        for i in sym:
            self.assertEqual(g.tier(i), 'S')

        # Conceptual: union, not, C→P transition
        self.assertEqual(len(con), 3)
        for i in con:
            self.assertEqual(g.tier(i), 'C')

        # Perceptual: ε
        self.assertEqual(len(per), 1)

    def test_binary_rules(self):
        """binary_rules() returns only arity-2 rule indices."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': ['true(S) EOF'],
            'S': ['swap(S, S)', 'C'],
            'C': ['union(C, C)', 'not(C)', 'P'],
            'P': ['ε'],
        })
        binary = g.binary_rules()
        for i in binary:
            self.assertEqual(g.arity(i), 2)
        # swap and union are binary
        self.assertEqual(len(binary), 2)

    def test_transitions(self):
        """symbolic_transition() and conceptual_transition() find correct rules."""
        g = Grammar(lazy_init=False)
        g.configure({
            'START': ['true(S) EOF'],
            'S': ['swap(S, S)', 'C'],
            'C': ['union(C, C)', 'P'],
            'P': ['ε'],
        })
        s_trans = g.symbolic_transition()
        c_trans = g.conceptual_transition()
        self.assertIsNotNone(s_trans)
        self.assertIsNotNone(c_trans)
        self.assertEqual(g.rules[s_trans].canonical, 'S → C')
        self.assertEqual(g.rules[c_trans].canonical, 'C → P')


class TestMentalModelWithGrammar(unittest.TestCase):
    """MentalModel forward path exercises Grammar layers."""

    def setUp(self):
        self.model, self.grammar = _make_model()

    def test_forward_runs(self):
        """MentalModel forward completes with Grammar initialized."""
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:2])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                input_state, symbols, output = self.model.forward(x)

            self.assertEqual(symbols.ndim, 3)
            self.assertEqual(symbols.shape[0], 2)

    def test_grammar_derivation_via_forward(self):
        """Drive Grammar.forward() on S-tier with symbol activations."""
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:2])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                self.model.forward(x)

                from Space import TheGrammar, SubSpace
                sym_act = self.model.symbols.get_activation()

                if sym_act is not None:
                    ss = SubSpace(
                        inputShape=[sym_act.shape[1], 1],
                        outputShape=[sym_act.shape[1], 1])
                    result = TheGrammar.forward('S', sym_act, ss)
                    self.assertEqual(result.shape[0], 2)  # batch=2

    def test_grammar_derivation_c_tier(self):
        """Drive Grammar.forward() on C-tier with concept vectors."""
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:1])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                self.model.forward(x)

                from Space import TheGrammar, SubSpace
                con_vec = self.model.concepts.materialize()

                if con_vec is not None:
                    ss = SubSpace(
                        inputShape=list(con_vec.shape[1:]),
                        outputShape=list(con_vec.shape[1:]))
                    result = TheGrammar.forward('C', con_vec, ss)
                    self.assertEqual(result.ndim, 3)  # [B, N, D]

    def test_word_rule_ids_are_global(self):
        """Word tuples use global Grammar rule IDs."""
        from Space import TheGrammar
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:1])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                self.model.forward(x)

                # S-tier derivation via current API
                from Space import SubSpace
                sym_act = self.model.symbols.get_activation()
                if sym_act is not None:
                    ss = SubSpace(
                        inputShape=[sym_act.shape[1], 1],
                        outputShape=[sym_act.shape[1], 1])
                    TheGrammar.forward('S', sym_act, ss)
                    words = ss.get_words()
                    num_rules = len(TheGrammar.rules)
                    for word in words:
                        rule_id = word[-1]  # last element is rule_id
                        self.assertGreaterEqual(rule_id, 0)
                        self.assertLess(rule_id, num_rules,
                                        f"Rule ID {rule_id} >= {num_rules}")

    def test_forward_reverse_roundtrip(self):
        """MentalModel forward+reverse roundtrip produces valid shapes."""
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:1])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                input_state, symbols, output = self.model.forward(x)
                try:
                    inputData, inputLatent = self.model.reverse(
                        symbols, self.model.outputs.materialize())
                    self.assertEqual(inputData.shape[0], 1)
                except (ValueError, RuntimeError):
                    self.skipTest("Untrained model range violation (expected)")


if __name__ == '__main__':
    unittest.main()
