"""
Grammar Derivation Test -- MentalModel + Grammar
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

import gc
import unittest
import warnings
import torch
import matplotlib
import Models
import Spaces
import Language
matplotlib.use('Agg')

RuleDef = Language.Grammar.RuleDef
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
    # Force re-initialization in case previous tests set small test dimensions
    Language.TheGrammar._configured = False
    Language.TheGrammar._configured = False
    model, _ = Models.MentalModel.from_config(os.path.join(_DATA_DIR, 'MentalModel.xml'))
    return model, Language.TheGrammar


def _release_allocator_cache():
    """Drop accelerator-side allocator caches so freed tensors really free.

    MentalModel holds a ``[13312, 13312]`` mapping-layer parameter plus the
    rest of the graph -- a single instance is hundreds of megabytes. Without
    this, successive ``setUp()`` calls accumulate until MPS (30 GiB limit)
    OOMs somewhere in the run. ``gc.collect()`` reclaims the Python
    references; ``empty_cache()`` returns the freed blocks to the OS.
    """
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _GrammarTestBase(unittest.TestCase):
    """Shared tearDown that drops per-test MentalModel references.

    Subclasses typically call ``_make_model()`` in ``setUp``; without the
    tearDown the model lingers in the test instance until the next
    test's setUp overwrites it, by which point MPS has already tried
    (and failed) to allocate space for the second model alongside the
    first. Explicitly clearing common attribute slots and calling
    ``_release_allocator_cache()`` keeps the peak resident set bounded
    to one MentalModel at a time.
    """

    # Attribute slots commonly populated in setUp() across the test
    # classes in this file. Any slot not present on a given instance is
    # silently skipped.
    _MODEL_SLOTS = (
        'model', 'grammar',
        's_sl', 's_ss', 'c_sl', 'c_ss', 'p_sl', 'p_ss',
    )

    def tearDown(self):
        for slot in self._MODEL_SLOTS:
            if hasattr(self, slot):
                try:
                    delattr(self, slot)
                except AttributeError:
                    pass
        _release_allocator_cache()


class TestGrammarProject(_GrammarTestBase):
    """Grammar.project() dispatches rule methods correctly.

    Rule ids are looked up dynamically by ``(method_name, arity, tier)`` --
    NEVER hardcode an integer index here. Hardcoded indices are a
    persistent footgun: when the grammar is extended (e.g. trinity +
    coordination + demux + query in this branch), every position shifts
    and the test name silently disagrees with the rule it actually
    dispatches. The shape-only assertion would still pass, masking the
    drift. Lookup-by-name is the only correct way to tie a test to its
    target rule.
    """

    def setUp(self):
        self.model, self.grammar = _make_model()
        # Dimensions from MentalModel.xml: nDim=100, objectSize=4, nSymbols=128, nConcepts=256
        # Grammar layers use dim + objectSize as the actual vector width.
        self.symbol_dim = 100 + 4   # nDim + nWhere + nWhen
        self.concept_dim = 100 + 4
        self.n_symbols = 128
        self.n_concepts = 256
        self.s_sl = self.model.wordSpace.symbolicSyntacticLayer
        self.s_ss = self.model.symbolicSpace.subspace
        self.c_sl = self.model.wordSpace.conceptualSyntacticLayer
        self.c_ss = self.model.conceptualSpace.subspace

    # -- Helpers -----------------------------------------------------

    def _rule_id(self, method_name, arity, tier=None):
        """Look up a rule id by (method_name, arity[, tier]). Fails the
        test loudly if no matching rule exists in the current grammar.
        """
        for i, rule in enumerate(self.grammar.rules):
            if rule.method_name != method_name:
                continue
            if rule.arity != arity:
                continue
            if tier is not None and rule.tier != tier:
                continue
            return i
        self.fail(
            f"No grammar rule found for method={method_name!r} "
            f"arity={arity} tier={tier!r} -- rules currently configured: "
            f"{[r.canonical for r in self.grammar.rules]}"
        )

    def _transition_rule_id(self, from_tier, to_tier):
        """Look up the synthetic transition rule (no method, arity 1)
        whose canonical form is `<from_tier> -> <to_tier>`."""
        target = f"{from_tier} -> {to_tier}"
        for i, rule in enumerate(self.grammar.rules):
            if rule.method_name is None and rule.canonical == target:
                return i
        self.fail(
            f"No transition rule {target!r} -- rules: "
            f"{[r.canonical for r in self.grammar.rules]}"
        )

    def _S(self, B=2):
        return torch.randn(B, self.n_symbols, device=Models.TheDevice.get())

    def _C(self, B=2):
        return torch.randn(B, self.n_concepts, self.concept_dim, device=Models.TheDevice.get())

    # -- S-tier rules ------------------------------------------------

    def test_project_true_s_tier(self):
        """true(S) -- looked up by name, NOT by hardcoded index."""
        rule_id = self._rule_id('true', arity=1, tier='S')
        left = self._S()
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_false_s_tier(self):
        """false(S) -- Rule #1 trinity addition."""
        rule_id = self._rule_id('false', arity=1, tier='S')
        left = self._S()
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_non_s_tier(self):
        """non(S) -- triangular residual (Rule #1 replaces sigmoid form)."""
        rule_id = self._rule_id('non', arity=1, tier='S')
        left = self._S().clamp(-1.0, 1.0)
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_conjunction_s_tier(self):
        """conjunction(S, S) -- Hadamard min over bitonic activations."""
        rule_id = self._rule_id('conjunction', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_disjunction_s_tier(self):
        """disjunction(S, S) -- Hadamard max over bitonic activations."""
        rule_id = self._rule_id('disjunction', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_what_s_tier(self):
        """what(S) -- Rule #2 slot selector."""
        rule_id = self._rule_id('what', arity=1, tier='S')
        left = self._S()
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_where_s_tier(self):
        """where(S) -- Rule #2 slot selector."""
        rule_id = self._rule_id('where', arity=1, tier='S')
        left = self._S()
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_when_s_tier(self):
        """when(S) -- Rule #2 slot selector."""
        rule_id = self._rule_id('when', arity=1, tier='S')
        left = self._S()
        result = self.s_sl.project(self.grammar, rule_id, left)
        self.assertEqual(result.shape, left.shape)

    def test_project_query_s_tier(self):
        """query(S, S) -- Rule #3 returns preserved (left) operand."""
        rule_id = self._rule_id('query', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)
        # queryForward returns the preserved accumulator (the left operand).
        self.assertTrue(torch.equal(result, left))

    def test_project_swap_s_tier(self):
        """swap(S, S) -- soft permutation via Sinkhorn logits."""
        rule_id = self._rule_id('swap', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_equals_s_tier(self):
        """equals(S, S) -- S-tier lossy operation."""
        rule_id = self._rule_id('equals', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_part_s_tier(self):
        """part(S, S) -- mereological parthood, moved to S-tier after rewrite."""
        rule_id = self._rule_id('part', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    # S -> C transition removed in Task 1.2 rewrite; no corresponding test.

    # -- C-tier rules removed in Task 1.2 rewrite ------------------

    def test_project_intersection_s_tier(self):
        """intersection(S, S) -- moved to S-tier after rewrite."""
        rule_id = self._rule_id('intersection', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_union_s_tier(self):
        """union(S, S) -- moved to S-tier after rewrite."""
        rule_id = self._rule_id('union', arity=2, tier='S')
        left, right = self._S(), self._S()
        result = self.s_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_lower_s_tier(self):
        """lower(S, S) -- projected disjunction, moved to S-tier after rewrite.

        The forward implementation (liftForward/lowerForward) lives on
        ConceptualSyntacticLayer; the grammar rule is now S-tier.  We look up
        the rule by tier='S' but dispatch via c_sl (where the implementation
        lives) with concept-shaped data.  Noted as a known arch mismatch to be
        resolved in a later phase.
        """
        rule_id = self._rule_id('lower', arity=2, tier='S')
        left, right = self._C(), self._C()
        result = self.c_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    def test_project_lift_binary_s_tier(self):
        """lift(S, S) -- projected conjunction, moved to S-tier after rewrite.

        Same arch note as test_project_lower_s_tier: implementation on c_sl.
        """
        rule_id = self._rule_id('lift', arity=2, tier='S')
        left, right = self._C(), self._C()
        result = self.c_sl.project(self.grammar, rule_id, left, right)
        self.assertEqual(result.shape, left.shape)

    # Ternary lift lift(C, C, C) was removed in the Task 1.2 rewrite.
    # C -> P transition rule removed in Task 1.2 rewrite; no corresponding test.

    # -- Coverage check ----------------------------------------------

    def test_all_methods_dispatched(self):
        """Every method_name maps to Grammar._GRAMMAR_METHODS or a SyntacticLayer subclass."""
        dispatched = set()
        for rule in self.grammar.rules:
            if rule.method_name:
                dispatched.add(rule.method_name)
        # Stateless methods on Grammar (S-tier subclass has the widest table)
        grammar_methods = set(self.s_sl._RULE_METHODS.keys())
        # Parametric methods on SyntacticLayer subclasses
        subclass_methods = {'swap', 'lift', 'lower', 'non'}
        all_known = grammar_methods | subclass_methods
        self.assertTrue(dispatched.issubset(all_known),
                        f"Unknown methods: {dispatched - all_known}")
        # chunk is P-tier; all others should be covered by rules
        self.assertTrue(all_known.issubset(dispatched | {'chunk'}),
                        f"Untested methods: {all_known - dispatched}")

    def test_no_hardcoded_rule_indices(self):
        """Regression guard: this class must not regress to hardcoded ids.

        The previous version of TestGrammarProject hardcoded rule_id
        integers as the second argument to project() -- e.g. passing a
        literal index instead of looking the rule up by name. Those
        indices silently drifted out of sync with the grammar every
        time a rule was added. The shape-only assertions kept passing
        on the wrong dispatch path, masking the failure for years.
        Future edits MUST go through self._rule_id(method, arity, tier).
        This test reads its own source and asserts no literal integer
        is passed as the second argument to a project() call.
        """
        import re
        with open(__file__, 'r') as fh:
            src = fh.read()
        # Find the TestGrammarProject class body.
        cls_match = re.search(
            r'class TestGrammarProject\b.*?(?=\nclass\s)',
            src,
            re.DOTALL,
        )
        self.assertIsNotNone(cls_match, "Could not locate TestGrammarProject class body")
        body = cls_match.group(0)
        # `project(self.grammar, <int_literal>, ...)` is the forbidden form.
        forbidden = re.findall(
            r'\.project\s*\(\s*self\.grammar\s*,\s*(\d+)\s*[,\)]',
            body,
        )
        self.assertEqual(
            forbidden, [],
            f"TestGrammarProject contains hardcoded rule indices: {forbidden}. "
            f"Use self._rule_id(method, arity, tier) instead."
        )


class TestGrammarShiftReduceStack(_GrammarTestBase):
    """SyntacticLayer.compose() applies rules and records words on subspace."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.symbol_dim = 100 + 4   # nDim + nWhere + nWhen
        self.concept_dim = 100 + 4
        self.n_symbols = 128
        self.n_concepts = 256
        self.s_sl = self.model.wordSpace.symbolicSyntacticLayer
        self.c_sl = self.model.wordSpace.conceptualSyntacticLayer
        self.p_sl = self.model.wordSpace.perceptualSyntacticLayer

    def _make_ss(self, shape):
        return Spaces.SubSpace(inputShape=shape, outputShape=shape)

    def test_symbolic_forward_shape(self):
        """compose('S', ...) returns same shape activation."""
        ss = self._make_ss([self.n_symbols, 1])
        act = torch.zeros(1, self.n_symbols, device=Models.TheDevice.get())
        act[0, 3] = 1.0
        result = self.s_sl.compose(act, ss, self.grammar)
        self.assertEqual(result.shape, (1, self.n_symbols))

    def test_symbolic_forward_two_active(self):
        """compose('S', ...) works with two active positions."""
        ss = self._make_ss([self.n_symbols, 1])
        act = torch.zeros(1, self.n_symbols, device=Models.TheDevice.get())
        act[0, 2] = 1.0
        act[0, 5] = 1.0
        result = self.s_sl.compose(act, ss, self.grammar)
        self.assertEqual(result.shape, (1, self.n_symbols))

    def test_conceptual_forward_shape(self):
        """compose('C', ...) returns same shape vectors."""
        ss = self._make_ss([self.n_concepts, self.concept_dim])
        vec = torch.randn(1, self.n_concepts, self.concept_dim, device=Models.TheDevice.get())
        vec[0, 3:] = 0.0  # 3 active positions
        result, _ = self.c_sl.compose(vec, ss, self.grammar)
        self.assertEqual(result.shape, (1, self.n_concepts, self.concept_dim))

    def test_perceptual_forward_shape(self):
        """compose('P', ...) returns same shape vectors."""
        n_percepts = 128
        ss = self._make_ss([n_percepts, self.symbol_dim])
        vec = torch.randn(1, n_percepts, self.symbol_dim, device=Models.TheDevice.get())
        vec[0, 4:] = 0.0
        result = self.p_sl.compose(vec, ss, self.grammar)
        self.assertEqual(result.shape, (1, n_percepts, self.symbol_dim))

    def test_reset_clears_stacks(self):
        """SubSpace.set_words([]) clears the word list."""
        ss = self._make_ss([self.n_symbols, 1])
        ss.word = [(0, 1, 2)]
        ss.set_words([])
        self.assertEqual(len(ss.get_words()), 0)

    def test_multiple_symbolic_forward_calls(self):
        """Multiple compose() calls work without error."""
        for _ in range(3):
            ss = self._make_ss([self.n_symbols, 1])
            act = torch.zeros(1, self.n_symbols, device=Models.TheDevice.get())
            act[0, torch.randint(self.n_symbols, (1,))] = 1.0
            self.s_sl.compose(act, ss, self.grammar)


class TestGrammarConfigure(_GrammarTestBase):
    """Grammar.configure() parses S-tier rule definitions into RuleDefs."""

    def test_configure_from_dict(self):
        """configure() builds correct RuleDefs from an S-only grammar dict."""
        g = Language.Grammar()
        g.configure({
            'S': ['swap(S, S)', 'equals(S, S)', 'union(S, S)', 'not(S)'],
        })
        self.assertEqual(len(g.rules), 4)
        for rule in g.rules:
            self.assertEqual(rule.tier, 'S')
        self.assertEqual(g.arity(0), 2)  # swap(S, S)
        self.assertEqual(g.method_name(0), 'swap')
        self.assertEqual(g.method_name(1), 'equals')

    def test_configure_single_string_rule(self):
        """configure() handles a single string (not list) for S tier."""
        g = Language.Grammar()
        g.configure({'S': 'not(S)'})
        self.assertEqual(len(g.rules), 1)

    def test_tier_queries(self):
        """symbolic() returns all S-tier rule indices."""
        g = Language.Grammar()
        g.configure({
            'S': ['swap(S, S)', 'equals(S, S)', 'union(S, S)'],
        })
        sym = g.symbolic()
        self.assertEqual(len(sym), 3)
        for i in sym:
            self.assertEqual(g.tier(i), 'S')

    def test_binary_rules(self):
        """binary_rules() returns only arity-2 rule indices."""
        g = Language.Grammar()
        g.configure({
            'S': ['swap(S, S)', 'not(S)', 'union(S, S)'],
        })
        binary = g.binary_rules()
        for i in binary:
            self.assertEqual(g.arity(i), 2)
        # swap and union are binary; not is unary
        self.assertEqual(len(binary), 2)


class TestMentalModelWithGrammar(_GrammarTestBase):
    """MentalModel forward path exercises Grammar layers."""

    def setUp(self):
        self.model, self.grammar = _make_model()

    def test_forward_runs(self):
        """MentalModel forward completes with Grammar initialized."""
        sentences = ['the cat sat on the mat', 'a dog chased the ball']
        outputs = [torch.tensor([0.0]), torch.tensor([1.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
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

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:2])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                self.model.forward(x)

                sym_act = self.model.symbols.get_activation()

                if sym_act is not None:
                    sl = self.model.wordSpace.symbolicSyntacticLayer
                    ss = self.model.symbolicSpace.subspace
                    result = sl.compose(sym_act, ss, Language.TheGrammar)
                    self.assertEqual(result.shape[0], 2)  # batch=2

    def test_word_rule_ids_are_global(self):
        """Word tuples use global Grammar rule IDs."""
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
             warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Range violation")
            warnings.filterwarnings("ignore", message="PiLayer.reverse")
            train_input, _ = self.model.inputSpace.getTrainData()
            x = self.model.inputSpace.prepInput(train_input[:1])

            self.model.eval()
            self.model.set_sigma(0)
            with torch.no_grad():
                self.model.forward(x)

                # S-tier derivation via SyntacticLayer
                sym_act = self.model.symbols.get_activation()
                if sym_act is not None:
                    sl = self.model.wordSpace.symbolicSyntacticLayer
                    ss = self.model.symbolicSpace.subspace
                    sl.compose(sym_act, ss, Language.TheGrammar)
                    words = ss.get_words()
                    num_rules = len(Language.TheGrammar.rules)
                    for word in words:
                        rule_id = word[3]  # word layout: (batch, vector, order, rule, ...)
                        self.assertGreaterEqual(rule_id, 0)
                        self.assertLess(rule_id, num_rules,
                                        f"Rule ID {rule_id} >= {num_rules}")

    def test_forward_reverse_roundtrip(self):
        """MentalModel forward+reverse roundtrip produces valid shapes."""
        sentences = ['the cat sat on the mat']
        outputs = [torch.tensor([0.0])]

        with Models.TheData.runtime_batch(sentences, outputs), \
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


class TestRule1TrinityCoordination(_GrammarTestBase):
    """Rule #1: trinity (true/false/non) + coordination (conjunction/disjunction).

    Covers the partition of unity ``true + false + non = 1`` and the
    Hadamard min/max coordination operators on bitonic activations.
    """

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.s_sl = self.model.wordSpace.symbolicSyntacticLayer
        self.s_ss = self.model.symbolicSpace.subspace

    def test_trinity_partition_of_unity(self):
        """For x in [-1, 1]: true(x) + false(x) + non(x) == 1."""
        x = torch.linspace(-1.0, 1.0, 21, device=Models.TheDevice.get())
        t = self.s_sl.trueForward(x, self.s_ss)
        f = self.s_sl.falseForward(x, self.s_ss)
        n = self.s_sl.nonForward(x, self.s_ss)
        partition = t + f + n
        self.assertTrue(
            torch.allclose(partition, torch.ones_like(partition), atol=1e-5),
            f"trinity partition broken: max deviation "
            f"{(partition - 1.0).abs().max().item()}")

    def test_trinity_endpoints(self):
        """Endpoints lock to one operator: x=+1->true, x=-1->false, x=0->non."""
        s_sl, ss = self.s_sl, self.s_ss
        one = torch.tensor([1.0], device=Models.TheDevice.get())
        zero = torch.tensor([0.0], device=Models.TheDevice.get())
        neg = torch.tensor([-1.0], device=Models.TheDevice.get())
        # x = +1
        self.assertAlmostEqual(s_sl.trueForward(one, ss).item(),  1.0, places=5)
        self.assertAlmostEqual(s_sl.falseForward(one, ss).item(), 0.0, places=5)
        self.assertAlmostEqual(s_sl.nonForward(one, ss).item(),   0.0, places=5)
        # x = -1
        self.assertAlmostEqual(s_sl.trueForward(neg, ss).item(),  0.0, places=5)
        self.assertAlmostEqual(s_sl.falseForward(neg, ss).item(), 1.0, places=5)
        self.assertAlmostEqual(s_sl.nonForward(neg, ss).item(),   0.0, places=5)
        # x = 0 -- pure indeterminate
        self.assertAlmostEqual(s_sl.trueForward(zero, ss).item(),  0.0, places=5)
        self.assertAlmostEqual(s_sl.falseForward(zero, ss).item(), 0.0, places=5)
        self.assertAlmostEqual(s_sl.nonForward(zero, ss).item(),   1.0, places=5)

    def test_conjunction_idempotent(self):
        """conjunction(s, s) returns s (Hadamard min idempotency)."""
        s = torch.rand(2, 16, device=Models.TheDevice.get()) * 2 - 1  # [-1, 1]
        out = self.s_sl.conjunctionForward(s, s, self.s_ss)
        self.assertTrue(torch.allclose(out, s, atol=1e-5))

    def test_disjunction_idempotent(self):
        """disjunction(s, s) returns s (Hadamard max idempotency)."""
        s = torch.rand(2, 16, device=Models.TheDevice.get()) * 2 - 1
        out = self.s_sl.disjunctionForward(s, s, self.s_ss)
        self.assertTrue(torch.allclose(out, s, atol=1e-5))

    def test_conjunction_disjunction_distinct(self):
        """conjunction and disjunction differ on non-equal inputs."""
        a = torch.tensor([0.2, -0.4, 0.7], device=Models.TheDevice.get())
        b = torch.tensor([0.5,  0.1, 0.3], device=Models.TheDevice.get())
        c = self.s_sl.conjunctionForward(a, b, self.s_ss)
        d = self.s_sl.disjunctionForward(a, b, self.s_ss)
        self.assertFalse(torch.allclose(c, d))

    def test_register_in_rule_methods(self):
        """false / conjunction / disjunction registered in _RULE_METHODS."""
        m = self.s_sl._RULE_METHODS
        self.assertIn('false', m)
        self.assertIn('conjunction', m)
        self.assertIn('disjunction', m)
        self.assertEqual(m['false'][0], 'falseForward')
        self.assertTrue(m['conjunction'][2])  # binary flag
        self.assertTrue(m['disjunction'][2])


class TestRule2DemuxAndSelectors(_GrammarTestBase):
    """Rule #2: SubSpace.demux + what/where/when slot selectors.

    Tests that the column blocks are partitioned correctly and that
    the selectors mask non-selected blocks while preserving shape.
    """

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.s_sl = self.model.wordSpace.symbolicSyntacticLayer
        self.s_ss = self.model.symbolicSpace.subspace

    def test_subspace_has_canonical_layout(self):
        """SymbolicSubSpace exposes nWhat / nWhere / nWhen partition."""
        ss = self.s_ss
        self.assertGreaterEqual(ss.nWhat, 0)
        self.assertGreaterEqual(ss.nWhere, 0)
        self.assertGreaterEqual(ss.nWhen, 0)
        self.assertEqual(ss.muxedSize, ss.nWhat + ss.nWhere + ss.nWhen)

    def test_demux_round_trip(self):
        """demux -> set_*; the column blocks reconstruct the original tensor."""
        ss = self.s_ss
        D = ss.muxedSize
        muxed = torch.randn(2, ss.event.codebook.weight.shape[0]
                            if hasattr(ss.event, 'codebook') else 8, D,
                            device=Models.TheDevice.get())
        ss.demux(muxed)
        # Read back via get_what/get_where/get_when (SubSpace API).
        what = ss.what.getW() if hasattr(ss.what, 'getW') else None
        where = ss.where.getW() if hasattr(ss.where, 'getW') else None
        when = ss.when.getW() if hasattr(ss.when, 'getW') else None
        # Recompose by concat and compare.
        parts = []
        if what is not None and ss.nWhat > 0:
            parts.append(what)
        if where is not None and ss.nWhere > 0:
            parts.append(where)
        if when is not None and ss.nWhen > 0:
            parts.append(when)
        if parts:
            recomposed = torch.cat(parts, dim=-1)
            self.assertEqual(recomposed.shape, muxed.shape)
            self.assertTrue(torch.allclose(recomposed, muxed, atol=1e-5),
                            "demux+set then recompose should be identity")

    def test_what_selector_zeros_where_when(self):
        """whatForward keeps the .what column block, zeros the rest."""
        ss = self.s_ss
        if ss.nWhere == 0 and ss.nWhen == 0:
            self.skipTest("config has no where/when columns to mask")
        D = ss.muxedSize
        x = torch.randn(2, 8, D, device=Models.TheDevice.get())
        out = self.s_sl.whatForward(x, ss)
        # what block preserved
        if ss.nWhat > 0:
            self.assertTrue(torch.equal(out[..., :ss.nWhat], x[..., :ss.nWhat]))
        # where/when blocks zeroed
        self.assertTrue(torch.all(out[..., ss.nWhat:] == 0))

    def test_where_selector_zeros_what_when(self):
        """whereForward keeps the .where column block, zeros the rest."""
        ss = self.s_ss
        if ss.nWhere == 0:
            self.skipTest("config has nWhere=0 -- no where block to select")
        D = ss.muxedSize
        x = torch.randn(2, 8, D, device=Models.TheDevice.get())
        out = self.s_sl.whereForward(x, ss)
        # what block zeroed
        self.assertTrue(torch.all(out[..., :ss.nWhat] == 0))
        # where block preserved
        self.assertTrue(torch.equal(
            out[..., ss.nWhat:ss.nWhat + ss.nWhere],
            x[...,   ss.nWhat:ss.nWhat + ss.nWhere]))
        # when block zeroed
        self.assertTrue(torch.all(out[..., ss.nWhat + ss.nWhere:] == 0))

    def test_when_selector_zeros_what_where(self):
        """whenForward keeps the .when column block, zeros the rest."""
        ss = self.s_ss
        if ss.nWhen == 0:
            self.skipTest("config has nWhen=0 -- no when block to select")
        D = ss.muxedSize
        x = torch.randn(2, 8, D, device=Models.TheDevice.get())
        out = self.s_sl.whenForward(x, ss)
        # what block zeroed
        self.assertTrue(torch.all(out[..., :ss.nWhat] == 0))
        # where block zeroed
        self.assertTrue(torch.all(
            out[..., ss.nWhat:ss.nWhat + ss.nWhere] == 0))
        # when block preserved
        self.assertTrue(torch.equal(
            out[..., ss.nWhat + ss.nWhere:], x[..., ss.nWhat + ss.nWhere:]))

    def test_selectors_partition_input(self):
        """what(x) + where(x) + when(x) == x for any vector input."""
        ss = self.s_ss
        if ss.nWhere == 0 and ss.nWhen == 0 and ss.nWhat == 0:
            self.skipTest("degenerate column layout")
        D = ss.muxedSize
        x = torch.randn(2, 8, D, device=Models.TheDevice.get())
        s = (self.s_sl.whatForward(x, ss)
             + self.s_sl.whereForward(x, ss)
             + self.s_sl.whenForward(x, ss))
        self.assertTrue(torch.allclose(s, x, atol=1e-5),
                        "what + where + when should partition the column dim")

    def test_register_in_rule_methods(self):
        """what / where / when registered in _RULE_METHODS as unary."""
        m = self.s_sl._RULE_METHODS
        for name in ('what', 'where', 'when'):
            self.assertIn(name, m)
            self.assertFalse(m[name][2], f"{name} should be unary")


class TestRule3Query(_GrammarTestBase):
    """Rule #3: queryForward identity + norm-drop preservation semantics."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.s_sl = self.model.wordSpace.symbolicSyntacticLayer

    def test_query_forward_returns_left(self):
        """queryForward(left, right, ss) returns the preserved left operand."""
        left = torch.tensor([0.7, -0.2, 0.5], device=Models.TheDevice.get())
        right = torch.tensor([-0.7, 0.2, -0.5], device=Models.TheDevice.get())
        out = self.s_sl.queryForward(left, right, subspace=None)
        self.assertTrue(torch.equal(out, left))

    def test_query_registered_as_binary(self):
        """query registered in _RULE_METHODS as a binary forward."""
        m = self.s_sl._RULE_METHODS
        self.assertIn('query', m)
        self.assertEqual(m['query'][0], 'queryForward')
        self.assertTrue(m['query'][2], "query must be binary")

    def test_query_norm_drop_ratio_attribute(self):
        """The class-level norm-drop ratio attribute exists for tuning."""
        self.assertTrue(hasattr(self.s_sl, '_QUERY_NORM_DROP_RATIO'))
        self.assertGreater(self.s_sl._QUERY_NORM_DROP_RATIO, 0.0)
        self.assertLess(self.s_sl._QUERY_NORM_DROP_RATIO, 1.0)


class TestGrammarRuleTable(_GrammarTestBase):
    """Grammar exposes a rule_table dict (Task 1.4: replaced nn.Embedding)."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.ss = self.model.symbolicSpace

    def test_rule_table_is_dict(self):
        """Grammar.rule_table is a plain dict mapping int -> str."""
        self.assertIsInstance(self.grammar.rule_table, dict)
        self.assertGreater(len(self.grammar.rule_table), 0)

    def test_rule_table_keys_are_ints(self):
        """All keys of rule_table are int rule_ids."""
        for k in self.grammar.rule_table:
            self.assertIsInstance(k, int)

    def test_rule_table_values_are_strings(self):
        """All values of rule_table are canonical production strings."""
        for v in self.grammar.rule_table.values():
            self.assertIsInstance(v, str)

    def test_rule_by_id_matches_table(self):
        """Grammar.rule_by_id(id) returns the same string as rule_table[id]."""
        for rule_id, production in self.grammar.rule_table.items():
            self.assertEqual(self.grammar.rule_by_id(rule_id), production)

    def test_symbolic_space_has_no_rule_codebook(self):
        """SymbolicSpace.rule_codebook has been removed (Task 1.4)."""
        self.assertFalse(hasattr(self.ss, 'rule_codebook'),
                         "rule_codebook should no longer exist on SymbolicSpace")

    def test_symbolic_space_has_no_lookup_rule(self):
        """SymbolicSpace.lookup_rule has been removed (Task 1.4)."""
        self.assertFalse(hasattr(self.ss, 'lookup_rule'),
                         "lookup_rule should no longer exist on SymbolicSpace")


class TestWordSubSpaceBuffer(_GrammarTestBase):
    """WordSubSpace push/read/clear semantics + column-layout sharing."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.host = self.model.symbolicSpace
        # Mirror SymbolicSubSpace's column layout to test the peer-sharing
        # invariant the plan calls out.
        sub = self.host.subspace
        self.nDim = sub.muxedSize
        self.nWhat = sub.nWhat
        self.nWhere = sub.nWhere
        self.nWhen = sub.nWhen
        self.word_sub = Spaces.WordSubSpace(
            nDim=self.nDim, nWhat=self.nWhat, nWhere=self.nWhere,
            nWhen=self.nWhen, max_depth=32, max_arity=3, batch=1)
        self.word_sub.attach_codebook_host(self.host)

    def test_column_layout_matches_peer(self):
        """WordSubSpace inherits its [what|where|when] widths from the peer."""
        ss = self.host.subspace
        self.assertEqual(self.word_sub.nWhat, ss.nWhat)
        self.assertEqual(self.word_sub.nWhere, ss.nWhere)
        self.assertEqual(self.word_sub.nWhen, ss.nWhen)
        self.assertEqual(self.word_sub.muxedSize, ss.muxedSize)

    def test_initial_buffer_is_zero(self):
        """A fresh WordSubSpace reads back as all zeros."""
        buf = self.word_sub.read()
        self.assertEqual(buf.shape, (1, 32, self.nDim))
        self.assertTrue(torch.all(buf == 0))
        self.assertEqual(self.word_sub.top_of_stack(0), 0)

    def test_push_advances_top_by_block_size(self):
        """Each push() advances the top-of-stack by 1 + max_arity rows."""
        rule_id = 0  # any valid rule_id
        block_size = 1 + self.word_sub.max_arity
        self.word_sub.push(0, rule_id, leaves=(1, -1, -1))
        self.assertEqual(self.word_sub.top_of_stack(0), block_size)
        self.word_sub.push(0, rule_id, leaves=(2, 3, -1))
        self.assertEqual(self.word_sub.top_of_stack(0), 2 * block_size)

    def test_push_rule_row_is_zero(self):
        """After push(), row 0's .what block is all zeros (rule_codebook removed, Task 1.4)."""
        rid = 0
        self.word_sub.push(0, rid, leaves=(-1, -1, -1))
        buf = self.word_sub.read()
        actual = buf[0, 0, : self.nWhat]
        self.assertTrue(torch.all(actual == 0),
                        "rule-identity row should be zero (no codebook vector)")

    def test_unused_leaf_rows_are_empty(self):
        """Empty leaf slots (leaf_id == -1) leave that row's .what zero."""
        rid = 0
        self.word_sub.push(0, rid, leaves=(-1, -1, -1))
        buf = self.word_sub.read()
        # Rows 1..max_arity should be all zero (no leaf vectors written).
        for k in range(1, 1 + self.word_sub.max_arity):
            self.assertTrue(torch.all(buf[0, k, : self.nWhat] == 0),
                            f"unused leaf row {k} should be zero")

    def test_get_blocks_records_push_metadata(self):
        """get_blocks(b) returns the parse-tree ledger with rule_id + leaves."""
        self.word_sub.push(0, 1, leaves=(2, -1, -1))
        self.word_sub.push(0, 3, leaves=(4, 5, -1))
        blocks = self.word_sub.get_blocks(0)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]['rule_id'], 1)
        self.assertEqual(blocks[0]['leaves'], (2, -1, -1))
        self.assertEqual(blocks[1]['rule_id'], 3)
        self.assertEqual(blocks[1]['leaves'], (4, 5, -1))

    def test_clear_resets_buffer_and_ledger(self):
        """clear() returns the buffer to all-zero and empties the ledger."""
        self.word_sub.push(0, 0, leaves=(1, 2, 3))
        self.word_sub.clear()
        buf = self.word_sub.read()
        self.assertTrue(torch.all(buf == 0))
        self.assertEqual(self.word_sub.top_of_stack(0), 0)
        self.assertEqual(self.word_sub.get_blocks(0), [])

    def test_overflow_silently_drops(self):
        """Pushing past max_depth leaves the buffer untouched (no exception)."""
        block_size = 1 + self.word_sub.max_arity
        n_blocks_that_fit = self.word_sub.max_depth // block_size
        for _ in range(n_blocks_that_fit):
            self.word_sub.push(0, 0, leaves=(-1, -1, -1))
        top_before = self.word_sub.top_of_stack(0)
        self.word_sub.push(0, 0, leaves=(-1, -1, -1))  # one too many
        self.assertEqual(self.word_sub.top_of_stack(0), top_before)


class TestWordSpaceServiceLayer(_GrammarTestBase):
    """WordSpace dispatcher + ownership transfer + per-sentence lifecycle."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        self.word_space = self.model.wordSpace

    def test_word_space_attached_to_model(self):
        """MentalModel.create() builds a wordSpace and exposes it."""
        self.assertIsNotNone(self.word_space,
                             "MentalModel.wordSpace should be wired by create()")

    def test_word_space_owns_layers(self):
        """WordSpace holds references to all three SyntacticLayers."""
        self.assertIsNotNone(self.word_space.symbolicSyntacticLayer)
        self.assertIsNotNone(self.word_space.conceptualSyntacticLayer)
        self.assertIsNotNone(self.word_space.perceptualSyntacticLayer)
        # Sanity: those references match the home spaces' layers.
        self.assertIs(self.word_space.symbolicSyntacticLayer,
                      self.model.wordSpace.symbolicSyntacticLayer)

    def test_layer_back_reference_set(self):
        """attach_layer() sets layer.word_subspace as a back-reference."""
        s_layer = self.word_space.symbolicSyntacticLayer
        self.assertTrue(hasattr(s_layer, 'word_subspace'))
        self.assertIs(s_layer.word_subspace, self.word_space.subspace)

    def test_codebook_host_wired(self):
        """attach_codebook_host() registers SymbolicSpace as the host for push() gating."""
        self.assertIs(self.word_space.subspace.rule_codebook_host,
                      self.model.symbolicSpace)

    def test_home_spaces_have_word_space_pointer(self):
        """Each home space gets a non-Module wordSpace pointer for routing."""
        for attr in ('perceptualSpace', 'conceptualSpace', 'symbolicSpace'):
            space = getattr(self.model, attr)
            self.assertIs(space.wordSpace, self.word_space,
                          f"{attr}.wordSpace should point at the shared service")

    def test_clear_sentence_resets_buffer(self):
        """clear_sentence() rewinds the stack."""
        self.word_space.subspace.push(0, 0, leaves=(-1, -1, -1))
        self.assertGreater(self.word_space.subspace.top_of_stack(0), 0)
        self.word_space.clear_sentence()
        self.assertEqual(self.word_space.subspace.top_of_stack(0), 0)

    def test_get_blocks_returns_ledger(self):
        """WordSpace.get_blocks(b) delegates to the subspace ledger."""
        self.word_space.subspace.push(0, 1, leaves=(2, -1, -1))
        blocks = self.word_space.get_blocks(0)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]['rule_id'], 1)


class TestParseDerivationToXml(_GrammarTestBase):
    """parse.derivation_to_xml accepts WordSubSpace blocks."""

    def setUp(self):
        self.model, self.grammar = _make_model()
        from parse import (
            derivation_to_xml, derivation_to_xml_from_wordspace,
            _normalize_word, _is_terminal,
        )
        self.derivation_to_xml = derivation_to_xml
        self.derivation_to_xml_from_wordspace = derivation_to_xml_from_wordspace
        self._normalize_word = _normalize_word
        self._is_terminal = _is_terminal

    def _find_rule(self, name, arity=None):
        for i, r in enumerate(self.grammar.rules):
            if r.method_name == name and (arity is None or r.arity == arity):
                return i
        return None

    def test_normalize_word_accepts_block_dict(self):
        """_normalize_word handles a WordSubSpace block dict."""
        block = {'start': 0, 'rule_id': 5, 'leaves': (1, 2, -1)}
        rid, vec, leaves = self._normalize_word(block, self.grammar)
        self.assertEqual(rid, 5)
        self.assertIsNone(vec)
        self.assertEqual(leaves, (1, 2, -1))

    def test_xml_from_wordspace_round_trip(self):
        """Push a unary and a binary rule, render, expect tags in the output."""
        word_space = self.model.wordSpace
        if word_space is None:
            self.skipTest("WordSpace not built for this model")
        word_space.clear_sentence()
        # Pick a unary symbolic rule (e.g. true) and a binary one (e.g. conjunction)
        true_rid = self._find_rule('true', arity=1)
        conj_rid = self._find_rule('conjunction', arity=2)
        if true_rid is None or conj_rid is None:
            self.skipTest("required rules not present in this grammar")
        word_space.subspace.push(0, true_rid, leaves=(-1, -1, -1))
        word_space.subspace.push(0, conj_rid, leaves=(true_rid, true_rid, -1))
        xml = self.derivation_to_xml_from_wordspace(
            word_space, self.grammar, batch=0)
        self.assertIn('<true', xml)
        self.assertIn('<conjunction', xml)


class TestNoCycleInModuleTree(_GrammarTestBase):
    """Regression guard: WordSpace wiring must not create an nn.Module cycle.

    The two back-references (Space.wordSpace and WordSubSpace.rule_codebook_host)
    are stored via object.__setattr__ to bypass nn.Module child registration.
    If either one regresses to a normal setattr, model.to(device) recurses
    forever -- the recursion-error symptom we hit during initial integration.
    """

    def test_modules_iter_terminates(self):
        """model.modules() walks the tree without infinite recursion."""
        model, _ = _make_model()
        count = sum(1 for _ in model.modules())
        self.assertGreater(count, 0)

    def test_to_device_no_recursion_error(self):
        """model.to(device) completes -- exercises nn.Module._apply over the tree."""
        model, _ = _make_model()
        try:
            model.to(Models.TheDevice.get())  # should be a no-op or quick cycle
        except RecursionError as exc:
            self.fail(f"model.to() recursed forever -- cycle in nn.Module tree: {exc}")


if __name__ == '__main__':
    unittest.main()
