"""Pin the target data contract for the SubSpace.what STM refactor.

Source spec: doc/plans/2026-05-20-subspace-what-stm-signalrouter-refactor.md

The refactor reuses existing SubSpace modalities for the live STM stack
instead of introducing new fields. These tests are guardrails that fail
loudly if a future patch:

  * adds a parallel stack buffer (stack_c, stack_s, stack_depth, ...)
  * breaks the .what / .where / .activation roundtrip in stack-mode
  * stops gating dead stack slots through activation
  * collides the terminal-symbol and grammar-rule .where namespaces

Tests that pin contracts from later phases (Grammar registry, rule
codebook) are marked ``xfail(strict=True)`` so they will flip to pass
the moment that phase lands -- the strict flag means an unexpected
pass is a test failure, forcing this file to be updated when the
contract is fulfilled.
"""

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent           # basicmodel/
_wo_root = _project.parent                                  # WikiOracle/
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import pytest
import torch

from Spaces import SubSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stack_subspace(B=2, K=4, D=8):
    """Build a bare SubSpace shaped like a stack-mode STM.

    K is the fixed maximum STM capacity (slots per batch).
    D is the payload width (what content).
    """
    return SubSpace([K, D], [K, D], nInputDim=D, nOutputDim=D)


# ---------------------------------------------------------------------------
# Contract A: No new stack fields are introduced
# ---------------------------------------------------------------------------

# The spec is explicit: "Do not add new stack fields. The live STM stack
# is the forwarded SubSpace.what tensor." This test pins that constraint
# so any future refactor that smuggles a parallel buffer fails loudly.
_FORBIDDEN_STACK_FIELDS = (
    "stack_c", "stack_s", "stack_depth", "stack_valid",
    "stm_buffer", "stm_depth", "stm_stack",
)


def test_subspace_has_no_parallel_stack_fields():
    sub = _make_stack_subspace()
    present = [name for name in _FORBIDDEN_STACK_FIELDS if hasattr(sub, name)]
    assert present == [], (
        f"SubSpace must not grow parallel stack fields; found: {present}. "
        f"The live STM stack is .what / .where / .activation."
    )


def test_subspace_exposes_required_stack_modalities():
    sub = _make_stack_subspace()
    # The three modalities the refactor uses for stack-mode.
    for name in ("what", "where", "activation"):
        assert hasattr(sub, name), f"SubSpace missing required modality: {name}"


def test_subspace_setters_exist():
    sub = _make_stack_subspace()
    for setter in ("set_what", "set_where", "set_activation"):
        assert callable(getattr(sub, setter, None)), (
            f"SubSpace missing setter: {setter} -- stack-mode rewrites rely on it"
        )


# ---------------------------------------------------------------------------
# Contract B: .what / .where / .activation roundtrip
# ---------------------------------------------------------------------------

def test_what_roundtrip_through_setter():
    B, K, D = 2, 4, 8
    sub = _make_stack_subspace(B=B, K=K, D=D)
    payload = torch.randn(B, K, D)
    sub.set_what(payload)
    got = sub.materialize(mode="what")
    assert got is not None and got.shape == (B, K, D)
    assert torch.equal(got, payload)


def test_where_roundtrip_through_setter():
    B, K = 2, 4
    # Use a SubSpace with a real where width so .where has somewhere to land.
    from Spaces import WhereEncoding
    D_what, W = 8, 2
    sub = SubSpace(
        [K, D_what + W], [K, D_what + W],
        nInputDim=D_what + W, nOutputDim=D_what + W,
        whereEncoding=WhereEncoding(1, W),
    )
    locs = torch.randn(B, K, W)
    sub.set_where(locs)
    got = sub.materialize(mode="where")
    assert got is not None and got.shape == (B, K, W)
    assert torch.equal(got, locs)


def test_activation_roundtrip_through_setter():
    B, K = 2, 4
    sub = _make_stack_subspace(B=B, K=K, D=8)
    occ = torch.tensor([[1.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0]])
    sub.set_activation(occ)
    got = sub.materialize(mode="activation")
    assert got is not None
    # `mode="activation"` returns presence (|signed DoT|), which equals
    # the input here because the inputs are non-negative.
    assert torch.equal(got, occ.abs())


# ---------------------------------------------------------------------------
# Contract C: .activation gates dead stack slots in the muxed view
# ---------------------------------------------------------------------------

def test_dead_stack_slots_zero_in_materialized_event():
    """`materialize()` (default) returns event * activation_presence.

    The spec says dead stack slots zero out under existing
    materialization behavior. Pin that for stack-mode payloads.
    """
    B, K, D = 1, 4, 6
    sub = _make_stack_subspace(B=B, K=K, D=D)
    payload = torch.ones(B, K, D)
    sub.set_what(payload)
    # Two slots live, two empty.
    occ = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    sub.set_activation(occ)
    muxed = sub.materialize()        # default mode applies activation gate
    assert muxed is not None
    # Live slots keep their content magnitude; dead slots are zeroed.
    assert torch.all(muxed[0, :2] != 0.0)
    assert torch.all(muxed[0, 2:] == 0.0)


def test_zero_activation_does_not_corrupt_raw_what():
    """`mode="what"` returns the raw payload regardless of activation.

    Reductions need to *write* into a slot and then *promote* it via
    activation; we must not lose the payload when activation is 0
    before promotion.
    """
    B, K, D = 1, 4, 6
    sub = _make_stack_subspace(B=B, K=K, D=D)
    payload = torch.full((B, K, D), 3.0)
    sub.set_what(payload)
    sub.set_activation(torch.zeros(B, K))
    raw = sub.materialize(mode="what")
    assert raw is not None
    assert torch.equal(raw, payload)


# ---------------------------------------------------------------------------
# Contract D: .where namespace for terminal symbols vs grammar rules
# ---------------------------------------------------------------------------

# Plan §"Where Is Only Codebook Location" pins:
#     0                           empty
#     1..V_sym                    terminal symbol locations
#     V_sym+1..V_sym+R_rule       grammar rule locations
#
# Until Phase 1's GrammarRegistry exposes where_id_for_rule / where_id_for_symbol,
# these tests can't be checked against live accessors. Mark xfail(strict=True)
# so they flip to pass the moment Phase 1 lands.

def test_grammar_registry_surface_exists():
    """Phase 1: registry accessors are present on Grammar."""
    from Language import Grammar
    g = Grammar()
    for name in ("num_rules", "rule", "rules_for_space_role",
                 "where_id_for_rule", "where_id_for_symbol"):
        assert callable(getattr(g, name, None)), (
            f"Grammar missing registry accessor: {name}"
        )


def test_grammar_registry_where_id_namespaces_do_not_collide():
    """The .where namespace: 0=empty, 1..V_sym=symbols, V_sym+1..=rules."""
    from Language import Grammar
    g = Grammar()
    # Pretend a small symbol vocab is wired in (Phase 3 does this for real).
    g.symbol_vocab_size = 5

    # Symbol namespace: 1..V_sym.
    sym_ids = [g.where_id_for_symbol(i) for i in range(5)]
    assert sym_ids == [1, 2, 3, 4, 5], sym_ids

    # Rule namespace: starts at V_sym+1.
    rule_ids = [g.where_id_for_rule(i) for i in range(3)]
    assert rule_ids == [6, 7, 8], rule_ids

    # Empty / invalid -> 0.
    assert g.where_id_for_symbol(-1) == 0
    assert g.where_id_for_rule(-1) == 0
    assert g.where_id_for_symbol(None) == 0
    assert g.where_id_for_rule(None) == 0

    # Namespaces do not overlap.
    assert set(sym_ids).isdisjoint(set(rule_ids))


def test_grammar_registry_accessors_on_configured_grammar():
    """num_rules / rule / rules_for_space_role work on a real configured Grammar."""
    from Language import Grammar
    g = Grammar()
    g.configure({
        'compose': {
            'symbols': {'rule': ['S = not(S)', 'S = min(S, S)']},
            'concepts': {'rule': ['C = lift(C)']},
        },
    })
    n = g.num_rules()
    assert n == 3, f"expected 3 rules, got {n}"

    # rule(rule_id) returns a RuleDef with the expected fields.
    r0 = g.rule(0)
    assert r0.space_role in ('SS', 'CS')
    assert r0.arity in (1, 2)
    assert isinstance(r0.method_name, str)

    s_ids = g.rules_for_space_role('SS')
    c_ids = g.rules_for_space_role('CS')
    assert len(s_ids) == 2 and len(c_ids) == 1
    assert set(s_ids).isdisjoint(set(c_ids))

    # Arity filter narrows by arity.
    s_binary = g.rules_for_space_role('SS', arity=2)
    s_unary = g.rules_for_space_role('SS', arity=1)
    assert len(s_binary) == 1 and len(s_unary) == 1
    assert s_binary[0] != s_unary[0]


# ---------------------------------------------------------------------------
# Contract E: SyntacticLayer executor API (Phase 2)
# ---------------------------------------------------------------------------

def _make_syntactic_layer_with(host_layers, space_role='SS'):
    """Build a minimal SyntacticLayer for executor tests.

    Bypasses build_space_syntactic_layer (which depends on TheGrammar
    being configured for the host_space). We just need the dispatcher
    plus its _by_name table; the SymbolSpace it's wired to is a stub
    that satisfies register_host_layer.
    """
    from Language import SyntacticLayer

    class _StubSymbolSpace:
        def __init__(self):
            self.calls = []
        def register_host_layer(self, space_role, rule_name, layer):
            self.calls.append((space_role, rule_name))

    return SyntacticLayer(space_role=space_role, word_space=_StubSymbolSpace(),
                          host_layers=host_layers)


def test_execute_arity1_dispatches_to_layer_forward():
    """execute(rule_id, left) calls the arity-1 layer's compose -> forward."""
    from Language import TheGrammar, Grammar
    from Layers import NotLayer

    # Configure TheGrammar with the rule we care about so method_name() works.
    # NotLayer's method_name is 'not'.
    g_backup = Grammar()
    # Avoid mutating the global; instead, re-configure TheGrammar but restore.
    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols': {'rule': ['S = not(S)']}}})

        # not_layer's method_name must match TheGrammar's rule 0.
        rule0 = TheGrammar.rule(0)
        assert rule0.method_name == 'not'

        layer = _make_syntactic_layer_with({'not': NotLayer()})
        # A grammar slot holds an opaque concept code.
        x = torch.tensor([[[0.7, 0.2, 0.0, 0.0]]])  # [B=1, V=1, D=4]
        y = layer.execute(rule_id=0, left=x)
        assert y.shape == x.shape
        assert torch.allclose(y, x)  # negation exchanges poles, never the opaque code
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


def test_execute_arity2_requires_right():
    """execute on an arity-2 rule without `right` raises a clear error."""
    from Language import TheGrammar
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        layer = _make_syntactic_layer_with({'min': MinLayer()})
        left = torch.tensor([[0.4, 0.7]])
        right = torch.tensor([[0.6, 0.3]])

        # Arity-2 with right: should compose to min.
        y = layer.execute(rule_id=0, left=left, right=right)
        assert torch.allclose(y, torch.minimum(left, right))

        # Arity-2 without right: clear error.
        with pytest.raises(ValueError, match="requires `right`"):
            layer.execute(rule_id=0, left=left)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


def test_execute_superposed_independent_then_weighted_sum():
    """Each rule sees its own (left, right) and outputs combine by weighted sum.

    Pin the independent-contribution semantics: mutating one rule's
    output must not affect another's, because the combine is one
    stacked weighted sum (the plan's pseudo-code).
    """
    from Language import TheGrammar
    from Language import MinLayer, MaxLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols': {'rule': [
            'S = min(S, S)',
            'S = max(S, S)',
        ]}}})

        layer = _make_syntactic_layer_with({
            'min': MinLayer(),
            'max': MaxLayer(),
        })
        # Pre-compute hard outputs for each rule.
        left = torch.tensor([[0.4, 0.7]])
        right = torch.tensor([[0.6, 0.3]])
        and_out = torch.minimum(left, right)
        or_out  = torch.maximum(left, right)

        # 70/30 weight on min vs max.
        w = torch.tensor([[0.7, 0.3]])  # [B=1, R=2]
        got = layer.execute_superposed(
            rule_weights=w, left=left, right=right, rule_ids=[0, 1])
        expected = 0.7 * and_out + 0.3 * or_out
        assert torch.allclose(got, expected, atol=1e-6)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


# ---------------------------------------------------------------------------
# Contract F: grammar identity/location codebook
# ---------------------------------------------------------------------------

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def test_rule_codebook_class_basics():
    """RuleCodebook is a pure identity/location store; no embedding by default."""
    from Language import RuleCodebook
    rc = RuleCodebook(num_rules=3)
    assert rc.num_rules == 3
    # No embedding requested -> the parameter is registered as None.
    assert rc.embedding is None
    # Bare fallback location (no Grammar attached): rule_id + 1.
    assert rc.location(0) == 1
    assert rc.location(2) == 3
    # Invalid -> 0 sentinel.
    assert rc.location(-1) == 0
    assert rc.location(None) == 0


def test_rule_codebook_attached_grammar_routes_through_namespace():
    """When a Grammar is attached, .location respects V_sym offsetting."""
    from Language import Grammar, RuleCodebook
    g = Grammar()
    g.symbol_vocab_size = 5
    rc = RuleCodebook(num_rules=3, grammar=g)
    # Rule namespace starts at V_sym + 1 = 6.
    assert rc.location(0) == 6
    assert rc.location(1) == 7
    assert rc.location(2) == 8


def test_rule_codebook_with_embedding_initializes_xavier():
    """When embedding_dim>0, a learnable [R, D] parameter is created."""
    from Language import RuleCodebook
    rc = RuleCodebook(num_rules=4, embedding_dim=8)
    assert rc.embedding is not None
    assert tuple(rc.embedding.shape) == (4, 8)
    # Xavier-normal init produces non-zero values.
    assert torch.any(rc.embedding != 0)


@pytest.fixture(scope="module")
def _xor_model():
    """A real BasicModel with a property WholeSpace and symbolic grammar."""
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return m






    # Embedding is optional and is for SCORING, not parent vectors.
    # When present it's [R, D_embed] but the test fixture turns it off.
    # The contract: the router computes parent.what via SyntacticLayer
    # .execute(rule_id, left, right); the codebook only stamps .where.


# ---------------------------------------------------------------------------
# Contract G: LanguageLayer stack-rewrite path (Phase 4)
# ---------------------------------------------------------------------------

def _make_stack_subspace_with_where(B=2, K=4, D=8, W=2):
    """Build a stack-mode SubSpace with a real where dim for Phase 4 tests."""
    from Spaces import SubSpace, WhereEncoding
    # WhereEncoding(maxP, nWhere, nWhen) -- nWhere=W gives a W-wide
    # where buffer. We don't exercise sin/cos decoding here; the
    # encoder stamps an integer into element [0] of the W-wide row.
    we = WhereEncoding(maxP=10_000, nWhere=W, nWhen=0)
    sub = SubSpace(
        [K, D + W], [K, D + W],
        nInputDim=D + W, nOutputDim=D + W,
        whereEncoding=we,
    )
    # Seed empty stack state.
    sub.set_what(torch.zeros(B, K, D))
    sub.set_where(torch.zeros(B, K, W))
    sub.set_activation(torch.zeros(B, K))
    return sub


def _make_minimal_signal_router(D=8):
    """Build a LanguageLayer shell for stack-rewrite tests.

    The constructor needs widths even though shift/reduce don't read
    them. We provide modest values; no ops are attached because the
    stack path uses an externally-supplied SyntacticLayer.
    """
    from Language import LanguageLayer
    return LanguageLayer(n_input=4, n_output=4, hidden_dim=8,
                        feature_dim=D, max_depth=8)


def _make_syntactic_layer_for_stack(host_layers, space_role='SS'):
    """SyntacticLayer wrapper that doesn't require a real SymbolSpace."""
    from Language import SyntacticLayer

    class _StubSymbolSpace:
        def register_host_layer(self, *args, **kw):
            pass

    return SyntacticLayer(space_role=space_role, word_space=_StubSymbolSpace(),
                          host_layers=host_layers)


def test_shift_writes_into_first_empty_slot():
    """Hard SHIFT writes payload+where into the leftmost empty slot, sets occ=1."""
    router = _make_minimal_signal_router(D=8)
    sub = _make_stack_subspace_with_where(B=2, K=4, D=8, W=2)
    payload = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                            [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]])
    router.shift(sub, terminal_what=payload, where_id=3)

    what = sub.materialize(mode="what")
    where = sub.materialize(mode="where")
    occ = sub.materialize(mode="activation")
    # Slot 0 holds the payload; remaining slots empty.
    assert torch.equal(what[:, 0, :], payload)
    assert torch.all(what[:, 1:, :] == 0)
    # Where stamped (integer in slot 0 of W-wide row).
    assert where[0, 0, 0].item() == 3.0
    assert where[1, 0, 0].item() == 3.0
    assert torch.all(where[:, 1:, :] == 0)
    # Occupancy: first slot live, rest empty.
    assert torch.equal(occ, torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                          [1.0, 0.0, 0.0, 0.0]]))


def test_shift_appends_after_existing_live_slots():
    """A second SHIFT goes to slot 1, not slot 0."""
    router = _make_minimal_signal_router(D=4)
    sub = _make_stack_subspace_with_where(B=1, K=4, D=4, W=2)
    a = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    b = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
    router.shift(sub, a, where_id=1)
    router.shift(sub, b, where_id=2)
    what = sub.materialize(mode="what")
    where = sub.materialize(mode="where")
    occ = sub.materialize(mode="activation")
    assert torch.equal(what[0, 0, :], a[0])
    assert torch.equal(what[0, 1, :], b[0])
    assert torch.all(what[0, 2:, :] == 0)
    assert where[0, 0, 0].item() == 1.0
    assert where[0, 1, 0].item() == 2.0
    assert torch.equal(occ, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_shift_raises_on_full_stack():
    """Shifting into a fully-occupied row must raise (no silent overflow)."""
    router = _make_minimal_signal_router(D=2)
    sub = _make_stack_subspace_with_where(B=1, K=2, D=2, W=2)
    router.shift(sub, torch.tensor([[1.0, 1.0]]), where_id=1)
    router.shift(sub, torch.tensor([[2.0, 2.0]]), where_id=2)
    with pytest.raises(RuntimeError, match="stack full"):
        router.shift(sub, torch.tensor([[3.0, 3.0]]), where_id=3)


def test_reduce_writes_parent_in_left_zeros_right():
    """Hard REDUCE: parent at i=n_live-2, zero at j=n_live-1, occ updates."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        # Push two non-negative scalars (MinLayer = monotonic min).
        left_payload = torch.tensor([[0.3, 0.8]])
        right_payload = torch.tensor([[0.5, 0.2]])
        router.shift(sub, left_payload, where_id=1)
        router.shift(sub, right_payload, where_id=2)

        # Reduce -> slot 0 gets the elementwise min, slot 1 is zeroed.
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)

        what = sub.materialize(mode="what")
        where = sub.materialize(mode="where")
        occ = sub.materialize(mode="activation")
        # Parent = min(left, right).
        expected_parent = torch.minimum(left_payload, right_payload)[0]
        assert torch.allclose(what[0, 0, :], expected_parent)
        # Consumed slot zeroed.
        assert torch.all(what[0, 1, :] == 0)
        # Where: surviving slot stamped with rule location (V_sym + 1 + 0 = 6);
        # consumed slot zeroed.
        assert where[0, 0, 0].item() == 6.0
        assert torch.all(where[0, 1, :] == 0)
        # Occupancy: only the surviving slot is live.
        assert torch.equal(occ, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_reduce_raises_on_stack_underflow():
    """Reducing with fewer than 2 live slots must raise."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        # Only one slot live -> reduce should raise.
        router.shift(sub, torch.tensor([[0.5, 0.5]]), where_id=1)
        with pytest.raises(RuntimeError, match="underflow"):
            router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


def test_reduce_gradient_flows_to_child_payloads_and_op_params():
    """Plan acceptance: gradients reach op parameters AND child payloads.

    We dispatch REDUCE through a tiny parametric binary op (a learnable
    scale on left*right) registered as a 'min' rule. This is
    explicitly different from the parameter-free MinLayer/
    NotLayer cases above -- the point of this test is to pin the
    gradient path *into* the op's parameters.
    """
    import torch.nn as nn
    from Language import TheGrammar, RuleCodebook
    from Layers import GrammarLayer

    class _ParametricBinaryOp(GrammarLayer):
        """Tiny test op: parent = scale * (left * right). Arity 2."""
        rule_name = 'min'
        arity = 2
        space_role = 'SS'

        def __init__(self):
            super().__init__(0, 0)
            self.scale = nn.Parameter(torch.tensor(1.5))

        def forward(self, left, right):
            return self.scale * (left * right)

        def compose(self, left, right):
            return self.forward(left, right)

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        op = _ParametricBinaryOp()
        D = 2
        router = _make_minimal_signal_router(D=D)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=D, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': op})

        # Seed child payloads that require grad.
        left_payload = torch.tensor([[0.4, 0.6]], requires_grad=True)
        right_payload = torch.tensor([[0.3, 0.7]], requires_grad=True)
        # Stage them in the stack directly so the autograd graph from
        # the leaf payloads survives the set_what call (shift would
        # produce equivalent semantics but goes through more clones;
        # this isolates the REDUCE gradient path).
        sub.set_what(torch.cat([left_payload.unsqueeze(1),
                                right_payload.unsqueeze(1),
                                torch.zeros(1, 2, D)], dim=1))
        sub.set_activation(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))

        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
        parent = sub.materialize(mode="what")[0, 0, :]   # [D]
        loss = parent.sum()
        loss.backward()

        # Both child payloads receive non-zero gradient.
        assert left_payload.grad is not None and torch.any(left_payload.grad != 0)
        assert right_payload.grad is not None and torch.any(right_payload.grad != 0)
        # The op's scale parameter receives gradient too.
        assert op.scale.grad is not None and float(op.scale.grad) != 0.0, (
            "op parameter `scale` must receive gradient through REDUCE"
        )
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured



















# ---------------------------------------------------------------------------
# Contract K: Phase 7 -- decode .where + unreduce via layer.reverse
# ---------------------------------------------------------------------------

def test_grammar_decode_where_roundtrips_through_namespace():
    """decode_where is the strict inverse of where_id_for_symbol /
    where_id_for_rule across the full 0..V_sym+R_rule namespace."""
    from Language import Grammar
    g = Grammar()
    g.symbol_vocab_size = 5

    # Empty sentinel.
    assert g.decode_where(0) == ('empty', None)
    assert g.decode_where(-1) == ('empty', None)
    assert g.decode_where(None) == ('empty', None)

    # Terminal namespace 1..V_sym.
    for sym in range(5):
        wid = g.where_id_for_symbol(sym)
        assert g.decode_where(wid) == ('terminal', sym), (
            f"symbol {sym} -> where_id {wid} did not round-trip"
        )

    # Rule namespace V_sym+1..
    for rid in range(4):
        wid = g.where_id_for_rule(rid)
        assert g.decode_where(wid) == ('rule', rid), (
            f"rule {rid} -> where_id {wid} did not round-trip"
        )


def test_grammar_decode_where_tolerates_float_carrier():
    """The live router stores ints in a float tensor; decode_where
    must round to the right bucket even with mild fp noise."""
    from Language import Grammar
    g = Grammar()
    g.symbol_vocab_size = 5
    # Tiny float noise on a rule slot encoding.
    wid = g.where_id_for_rule(2)          # int 8 with V_sym=5
    noisy = float(wid) + 1e-7
    assert g.decode_where(noisy) == ('rule', 2)
    # 0-D tensor with the integer value.
    t = torch.tensor(float(wid))
    assert g.decode_where(t) == ('rule', 2)


def test_unreduce_calls_layer_reverse_and_writes_children():
    """Plan acceptance: reverse uses .where to decode rule, then applies
    the layer's reverse method to split parent into children.

    ADAPTED (2026-07-04 serial plan Task 1): the identity stub is
    REVOKED. The stack's ``.what`` is not a 2-D codebook, so the
    recommender cannot run and unreduce must FAIL LOUD with the
    Gate-S1 inventory row instead of writing (parent, parent)."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        left_in = torch.tensor([[0.3, 0.8]])
        right_in = torch.tensor([[0.5, 0.2]])
        router.shift(sub, left_in, where_id=1)
        router.shift(sub, right_in, where_id=2)
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
        # After reduce: slot 0 holds elementwise min; slot 1 zeroed.
        parent_after_reduce = sub.materialize(mode="what")[0, 0, :].clone()
        # .where[0] should now decode to a rule slot.
        where_after_reduce = sub.materialize(mode="where")
        kind, rid = TheGrammar.decode_where(where_after_reduce[0, 0, 0])
        assert kind == 'rule' and rid == 0, (
            f"After reduce, top slot .where should decode to rule 0; got {(kind, rid)}"
        )

        # Now unreduce -- the stub sanction is revoked: fail loud.
        import pytest
        with pytest.raises(NotImplementedError, match="min"):
            router.unreduce(sub, layer, rule_codebook=rc)
        assert parent_after_reduce is not None
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_unreduce_is_noop_on_terminal_slot():
    """A top slot stamped as a terminal (1..V_sym) must not be split.

    Terminals are leaves on this path -- their "reverse" is the
    codebook unsnap, which is Phase 8+ work. unreduce must return
    the subspace unchanged in this case.
    """
    from Language import TheGrammar, RuleCodebook

    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.symbol_vocab_size = 5

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=0, grammar=TheGrammar)
        # No real layers needed -- unreduce must short-circuit before
        # consulting the syntactic_layer when the top is a terminal.
        layer = _make_syntactic_layer_for_stack({})

        terminal_payload = torch.tensor([[0.5, 0.5]])
        # where_id=3 lies in the terminal namespace (1..V_sym=5).
        router.shift(sub, terminal_payload, where_id=3)
        what_before = sub.materialize(mode="what").clone()
        where_before = sub.materialize(mode="where").clone()
        occ_before = sub.materialize(mode="activation").clone()

        router.unreduce(sub, layer, rule_codebook=rc)
        what_after = sub.materialize(mode="what")
        where_after = sub.materialize(mode="where")
        occ_after = sub.materialize(mode="activation")

        assert torch.equal(what_after, what_before)
        assert torch.equal(where_after, where_before)
        assert torch.equal(occ_after, occ_before)
    finally:
        TheGrammar.symbol_vocab_size = saved_vsym


def test_unreduce_raises_on_empty_stack():
    """No live slots -> no top to unreduce; must raise loudly."""
    from Language import TheGrammar, RuleCodebook
    router = _make_minimal_signal_router(D=2)
    sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
    rc = RuleCodebook(num_rules=0, grammar=TheGrammar)
    layer = _make_syntactic_layer_for_stack({})
    with pytest.raises(RuntimeError, match="underflow"):
        router.unreduce(sub, layer, rule_codebook=rc)


def test_unreduce_raises_on_full_stack():
    """Full stack -> no room for the new right child slot; must raise."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 2
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        # K=2 stack: shift one terminal, then directly stamp the top
        # slot's .where to point at the rule so unreduce will try to
        # split. Fill BOTH slots first so the stack is full.
        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=2, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        router.shift(sub, torch.tensor([[0.5, 0.5]]), where_id=1)
        router.shift(sub, torch.tensor([[0.6, 0.6]]), where_id=2)
        # Manually stamp top slot as a rule slot to coerce unreduce
        # to attempt a split even though no real reduce happened.
        where = sub.materialize(mode="where").clone()
        where[0, 1, 0] = float(TheGrammar.where_id_for_rule(0))
        sub.set_where(where)

        with pytest.raises(RuntimeError, match="overflow"):
            router.unreduce(sub, layer, rule_codebook=rc)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_reverse_stack_unwinds_outermost_rule_only_under_identity_stub():
    """reverse_stack unwinds the outermost rule, then stops.

    Limitation worth pinning: under the identity-stub contract, an
    unreduce clears the children's .where to 0 (we have no provenance
    for the children -- the lossy parent doesn't tell us what rule
    each child came from). reverse_stack reads the top slot's .where
    each loop, so the very next iteration sees an 'empty' kind and
    halts.

    Forward: shift A, shift B, shift C, reduce (top-2), reduce (top-2).
    Stack ends with 1 rule-stamped root.

    ADAPTED (2026-07-04 serial plan Task 1): min has no faithful
    inverse on this path (the stack .what is not a codebook), so the
    reverse now FAILS LOUD; the unwinding contract is observable only
    once a real inverse exists (the Gate-S1 inventory consequence).
    """
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=6, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        # Forward: 3 shifts, 2 reduces -> 1 root.
        router.shift(sub, torch.tensor([[0.2, 0.9]]), where_id=1)
        router.shift(sub, torch.tensor([[0.4, 0.7]]), where_id=2)
        router.shift(sub, torch.tensor([[0.6, 0.5]]), where_id=3)
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)

        # Sanity: one live slot (the root) stamped with the rule .where.
        occ = sub.materialize(mode="activation")
        assert torch.equal(occ, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
        where = sub.materialize(mode="where")
        kind, _ = TheGrammar.decode_where(where[0, 0, 0])
        assert kind == 'rule'
        root = sub.materialize(mode="what")[0, 0, :].clone()

        # Reverse: the stub sanction is revoked -- min cannot
        # run a faithful inverse here, so the unwind FAILS LOUD.
        import pytest
        with pytest.raises(NotImplementedError, match="min"):
            router.reverse_stack(sub, layer, rule_codebook=rc)
        assert root is not None
        # The raise fires BEFORE any child write-back: the root slot
        # keeps its rule stamp (no partial mutation on failure).
        where = sub.materialize(mode="where")
        kind, _ = TheGrammar.decode_where(where[0, 0, 0])
        assert kind == 'rule', (
            f"failed unreduce must not partially mutate the stack; "
            f"got kind={kind}"
        )
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_unreduce_uses_identity_stub_when_layer_reverse_is_unsuitable():
    """ADAPTED (2026-07-04 serial plan Task 1): the identity-stub
    sanction is REVOKED. The base ``Layer.reverse`` single-tensor
    identity is the wrong shape for an arity-2 parent; unreduce must
    now FAIL LOUD (the Gate-S1 inventory row) instead of fabricating
    (parent, parent) children.
    """
    from Language import TheGrammar, RuleCodebook
    from Layers import GrammarLayer

    class _BinaryInheritingBaseReverse(GrammarLayer):
        """Arity-2 op that does NOT override reverse.

        Inherits ``Layer.reverse`` (a single-tensor identity), which is
        the wrong shape for an arity-2 unreduce -- the unsuitable-shape
        fallback path is what this test pins.
        """
        rule_name = 'min'
        arity = 2
        space_role = 'SS'

        def __init__(self):
            super().__init__(0, 0)
            # Layer.reverse asserts y.shape matches self.nOutput; set
            # it large enough that the inherited reverse won't blow up
            # before our fallback kicks in. (Even if it did blow up,
            # unreduce catches and falls back -- this is belt-and-
            # suspenders.)
            self.nOutput = 2

        def forward(self, left, right):
            return left * right

        def compose(self, left, right):
            return self.forward(left, right)

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        op = _BinaryInheritingBaseReverse()

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': op})

        router.shift(sub, torch.tensor([[0.5, 0.5]]), where_id=1)
        router.shift(sub, torch.tensor([[0.4, 0.6]]), where_id=2)
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
        parent = sub.materialize(mode="what")[0, 0, :].clone()

        # The stub sanction is revoked: the wrong-shape reverse raises
        # the inventory error (write a real reverse or remove the rule).
        import pytest
        with pytest.raises(NotImplementedError):
            router.unreduce(sub, layer, rule_codebook=rc)
        assert parent is not None
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


# ---------------------------------------------------------------------------
# Contract L: LanguageLayer is a Layer with canonical .forward / .reverse
# ---------------------------------------------------------------------------

def test_signal_router_is_a_layer_subclass():
    """LanguageLayer inherits from Layer so peer code can treat it
    uniformly with other Layer subclasses (forward/reverse contracts,
    nInput/nOutput attributes, ergodic dispatch).
    """
    from Language import LanguageLayer
    from Layers import Layer
    assert issubclass(LanguageLayer, Layer), (
        "LanguageLayer must inherit from Layer so its forward/reverse "
        "are recognized by Layer-aware call sites"
    )


def test_signal_router_init_sets_layer_attributes():
    """The Layer base contract requires nInput / nOutput to be set."""
    router = _make_minimal_signal_router(D=8)
    assert hasattr(router, 'nInput') and router.nInput == 4
    assert hasattr(router, 'nOutput') and router.nOutput == 4
    # The plain-list ``self.layers`` from Layer.__init__ is present
    # (ergodic interface), and is intentionally empty -- the trainable
    # scoring layers live in the ModuleDicts.
    assert isinstance(router.layers, list)


def test_forward_dispatches_to_forward_stack_with_actions():
    """languageLayer.forward(actions=...) is a thin wrapper around forward_stack."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        actions = [
            ('shift', torch.tensor([[0.6, 0.9]]), 1),
            ('shift', torch.tensor([[0.4, 0.5]]), 2),
            ('reduce', 0),
        ]
        # Spy on forward_stack to confirm the wrapper delegates to it.
        calls = {'n': 0}
        orig_fs = router.forward_stack

        def spy_fs(*a, **kw):
            calls['n'] += 1
            return orig_fs(*a, **kw)

        router.forward_stack = spy_fs
        try:
            out = router.forward(sub, layer, actions=actions, rule_codebook=rc)
        finally:
            router.forward_stack = orig_fs

        assert calls['n'] == 1
        assert out is sub
        # End state matches the manual shift+reduce path.
        what = sub.materialize(mode="what")
        assert torch.allclose(what[0, 0, :], torch.tensor([0.4, 0.5]))
        assert torch.all(what[0, 1, :] == 0)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_forward_without_actions_raises_with_pointer():
    """No learned policy yet -> explicit failure that points the caller
    at the lower-level primitives."""
    router = _make_minimal_signal_router(D=2)
    sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
    layer = _make_syntactic_layer_for_stack({})
    with pytest.raises(NotImplementedError, match="actions"):
        router.forward(sub, layer)


def test_reverse_dispatches_to_reverse_stack():
    """languageLayer.reverse(...) is a thin wrapper around reverse_stack."""
    from Language import TheGrammar, RuleCodebook
    router = _make_minimal_signal_router(D=2)
    sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
    rc = RuleCodebook(num_rules=0, grammar=TheGrammar)
    layer = _make_syntactic_layer_for_stack({})

    calls = {'n': 0, 'last_kwargs': None}
    orig_rs = router.reverse_stack

    def spy_rs(*a, **kw):
        calls['n'] += 1
        calls['last_kwargs'] = kw
        return orig_rs(*a, **kw)

    router.reverse_stack = spy_rs
    try:
        out = router.reverse(sub, layer, rule_codebook=rc, max_steps=3)
    finally:
        router.reverse_stack = orig_rs

    assert calls['n'] == 1
    assert out is sub
    # Wrapper passes kwargs through (max_steps, rule_codebook).
    assert calls['last_kwargs'].get('max_steps') == 3
    assert calls['last_kwargs'].get('rule_codebook') is rc




# ---------------------------------------------------------------------------
# Retired WholeSpace dispatch is covered by the property/grammar separation below.
# ---------------------------------------------------------------------------







def test_forward_stack_orchestrates_shift_then_reduce():
    """forward_stack runs a list of (shift/reduce) actions end-to-end."""
    from Language import TheGrammar, RuleCodebook
    from Language import MinLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    saved_vsym = TheGrammar.symbol_vocab_size
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.symbol_vocab_size = 5
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = min(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'min': MinLayer()})

        actions = [
            ('shift', torch.tensor([[0.6, 0.9]]), 1),
            ('shift', torch.tensor([[0.4, 0.5]]), 2),
            ('reduce', 0),
        ]
        router.forward_stack(sub, layer, actions=actions, rule_codebook=rc)

        what = sub.materialize(mode="what")
        occ = sub.materialize(mode="activation")
        assert torch.allclose(what[0, 0, :], torch.tensor([0.4, 0.5]))
        assert torch.all(what[0, 1, :] == 0)
        assert torch.equal(occ, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


# Item 7, section 11.4: grammar belongs to SymbolSpace; WholeSpace is properties.

def _shared_compose(model):
    """One real two-slot decision on a caller-owned STM state."""
    width = model.conceptualSpace.stm.concept_dim
    buffer = torch.arange(2 * width, dtype=torch.float32).reshape(1, 2, width) / (2 * width)
    state = (buffer, torch.tensor([2]), torch.ones(1, 2, dtype=torch.long),
             torch.zeros(1, 2, dtype=torch.long), torch.full((1, 2), -1, dtype=torch.long),
             torch.ones(1, 2))
    choice = model.languageSpace.choose_operation(state, torch.tensor([True]), slots=1, sample=False)
    return state, choice


def _shared_generate(model):
    """A tied inverse from the declared generate catalog, without a cursor."""
    language = model.languageSpace
    names = list(language._generate_binary_names)
    assert names
    index = 0
    width = model.conceptualSpace.stm.concept_dim
    parent = torch.full((1, width), .1)
    return language.reverse_binary_step(parent, torch.tensor([index]), torch.tensor([True]),
        reference=torch.full_like(parent, .2), ops=language._generate_binary_ops)


def _property_carrier(model):
    cs = model.conceptualSpace
    count, width = int(cs.subspace.inputShape[0]), int(cs.subspace.muxedSize)
    sub = SubSpace([count, width], [count, width], nInputDim=width, nOutputDim=width)
    sub.set_event(torch.ones(1, count, width))
    return sub


def test_symbolic_space_instance_owns_rule_catalog(_xor_model):
    from Language import OperationSelectionLayer
    language = _xor_model.languageSpace
    operation = _xor_model.symbolSpace.languageLayer.operation_layer
    assert isinstance(operation, OperationSelectionLayer)
    assert operation is language._tree_layer(2)
    assert len(language._compose_binary_rules) == len(operation.ops)


def test_word_admission_preserves_the_property_inventory(_xor_model):
    ws = _xor_model.wholeSpace
    before = ws.subspace.what.getW().detach().clone()
    _xor_model._concept_owner().new_concept()
    torch.testing.assert_close(ws.subspace.what.getW(), before, rtol=0, atol=0)
    assert not hasattr(ws, 'rule_codebook')




def test_symbolic_space_owns_signal_router(_xor_model):
    from Language import LanguageLayer
    ss = _xor_model.symbolSpace
    assert isinstance(ss.languageLayer, LanguageLayer)
    assert ss.languageLayer is _xor_model.languageSpace.language_layer
    assert not hasattr(_xor_model.wholeSpace, 'languageLayer')




def test_shared_operation_writes_the_caller_owned_stack(_xor_model):
    state, _ = _shared_compose(_xor_model)
    rounds = 2 * state[0].shape[1]
    for step in range(rounds):
        choice = _xor_model.languageSpace.choose_operation(
            state, torch.tensor([True]), slots=1, sample=False,
            rounds_left=rounds - step)
        state = _xor_model.conceptualSpace.apply_language_choice(state, choice)
    out = state
    assert out[0] is not None
    assert torch.isfinite(out[0]).all()
    assert out[1].tolist() == [1]
    assert bool((out[0][:, 0].abs().sum(-1) > 0).all())


def test_stack_router_does_not_touch_word_space_current_rules(_xor_model):
    ss = _xor_model.symbolSpace.subspace
    sentinel = {'SS': [['SENTINEL_NOT_TOUCHED']]}
    current, generate = ss.current_rules, ss.generate_rules
    ss.current_rules, ss.generate_rules = dict(sentinel), dict(sentinel)
    pre_compose_gen, pre_generate_gen = ss._compose_generation, ss._generate_generation
    try:
        _shared_compose(_xor_model)
        assert ss.current_rules == sentinel
        assert ss.generate_rules == sentinel
        assert ss._compose_generation == pre_compose_gen
        assert ss._generate_generation == pre_generate_gen
    finally:
        ss.current_rules, ss.generate_rules = current, generate


def test_stack_router_does_not_touch_conceptual_stm(_xor_model):
    stm = _xor_model.conceptualSpace.stm
    pre_buffer, pre_depth = stm._buffer.detach().clone(), stm._depth.detach().clone()
    _shared_compose(_xor_model)
    assert torch.equal(stm._buffer, pre_buffer)
    assert torch.equal(stm._depth, pre_depth)


def test_shared_compose_does_not_dispatch_a_cursor(_xor_model, monkeypatch):
    sl = _xor_model.symbolSpace.subspace.syntacticLayer
    calls = {'cursor': 0}
    original = sl._next_rule_name
    def cursor(*args, **kwargs):
        calls['cursor'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(sl, '_next_rule_name', cursor)
    _shared_compose(_xor_model)
    assert calls['cursor'] == 0


def test_property_forward_does_not_compose(_xor_model, monkeypatch):
    operation = _xor_model.symbolSpace.languageLayer.operation_layer
    calls = {'forward': 0}
    original = operation.forward
    def forward(*args, **kwargs):
        calls['forward'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(operation, 'forward', forward)
    _xor_model.wholeSpace.forward(_property_carrier(_xor_model))
    assert calls['forward'] == 0


def test_symbolic_space_stack_route_uses_canonical_forward(_xor_model, monkeypatch):
    operation = _xor_model.symbolSpace.languageLayer.operation_layer
    calls = {'forward': 0}
    original = operation.forward
    def forward(*args, **kwargs):
        calls['forward'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(operation, 'forward', forward)
    _shared_compose(_xor_model)
    assert calls['forward'] == 1


def test_symbolic_space_reverse_dispatches_to_declared_operator(_xor_model, monkeypatch):
    language = _xor_model.languageSpace
    calls = {'reverse': 0}
    original = language.reverse_binary_step
    def reverse(*args, **kwargs):
        calls['reverse'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(language, 'reverse_binary_step', reverse)
    _shared_generate(_xor_model)
    assert calls['reverse'] == 1


def test_property_reverse_does_not_call_grammar(_xor_model, monkeypatch):
    language = _xor_model.languageSpace
    calls = {'reverse': 0}
    original = language.reverse_binary_step
    def reverse(*args, **kwargs):
        calls['reverse'] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(language, 'reverse_binary_step', reverse)
    _xor_model.wholeSpace.reverse(_property_carrier(_xor_model))
    assert calls['reverse'] == 0


def test_symbolic_space_reverse_does_not_touch_generate_rules(_xor_model):
    ss = _xor_model.symbolSpace.subspace
    sentinel = {'SS': [['SENTINEL_NOT_TOUCHED']]}
    generate = ss.generate_rules
    ss.generate_rules = dict(sentinel)
    pre_generate_gen = ss._generate_generation
    try:
        _shared_generate(_xor_model)
        assert ss.generate_rules == sentinel
        assert ss._generate_generation == pre_generate_gen
    finally:
        ss.generate_rules = generate
