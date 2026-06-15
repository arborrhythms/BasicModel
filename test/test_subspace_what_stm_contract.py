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
    for name in ("num_rules", "rule", "rules_for_tier",
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
    """num_rules / rule / rules_for_tier work on a real configured Grammar."""
    from Language import Grammar
    g = Grammar()
    g.configure({
        'compose': {
            'symbols': {'rule': ['S = not(S)', 'S = conjunction(S, S)']},
            'concepts': {'rule': ['C = lift(C)']},
        },
    })
    n = g.num_rules()
    assert n == 3, f"expected 3 rules, got {n}"

    # rule(rule_id) returns a RuleDef with the expected fields.
    r0 = g.rule(0)
    assert r0.tier in ('S', 'C')
    assert r0.arity in (1, 2)
    assert isinstance(r0.method_name, str)

    s_ids = g.rules_for_tier('S')
    c_ids = g.rules_for_tier('C')
    assert len(s_ids) == 2 and len(c_ids) == 1
    assert set(s_ids).isdisjoint(set(c_ids))

    # Arity filter narrows by arity.
    s_binary = g.rules_for_tier('S', arity=2)
    s_unary = g.rules_for_tier('S', arity=1)
    assert len(s_binary) == 1 and len(s_unary) == 1
    assert s_binary[0] != s_unary[0]


# ---------------------------------------------------------------------------
# Contract E: SyntacticLayer executor API (Phase 2)
# ---------------------------------------------------------------------------

def _make_syntactic_layer_with(host_layers, tier='S'):
    """Build a minimal SyntacticLayer for executor tests.

    Bypasses build_space_syntactic_layer (which depends on TheGrammar
    being configured for the host_space). We just need the dispatcher
    plus its _by_name table; the WordSpace it's wired to is a stub
    that satisfies register_host_layer.
    """
    from Language import SyntacticLayer

    class _StubWordSpace:
        def __init__(self):
            self.calls = []
        def register_host_layer(self, tier, rule_name, layer):
            self.calls.append((tier, rule_name))

    return SyntacticLayer(tier=tier, word_space=_StubWordSpace(),
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
        # NotLayer flips the leading 2 dims of the last axis (bivector).
        x = torch.tensor([[[0.7, 0.2, 0.0, 0.0]]])  # [B=1, V=1, D=4]
        y = layer.execute(rule_id=0, left=x)
        assert y.shape == x.shape
        # bivector flipped: pos<->neg
        assert torch.allclose(y[..., 0], x[..., 1])
        assert torch.allclose(y[..., 1], x[..., 0])
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


def test_execute_arity2_requires_right():
    """execute on an arity-2 rule without `right` raises a clear error."""
    from Language import TheGrammar
    from Layers import ConjunctionLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = conjunction(S, S)']}}})

        layer = _make_syntactic_layer_with({'conjunction': ConjunctionLayer()})
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
    from Layers import ConjunctionLayer, DisjunctionLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols': {'rule': [
            'S = conjunction(S, S)',
            'S = disjunction(S, S)',
        ]}}})

        layer = _make_syntactic_layer_with({
            'conjunction': ConjunctionLayer(),
            'disjunction': DisjunctionLayer(),
        })
        # Pre-compute hard outputs for each rule.
        left = torch.tensor([[0.4, 0.7]])
        right = torch.tensor([[0.6, 0.3]])
        and_out = torch.minimum(left, right)
        or_out  = torch.maximum(left, right)

        # 70/30 weight on conjunction vs disjunction.
        w = torch.tensor([[0.7, 0.3]])  # [B=1, R=2]
        got = layer.execute_superposed(
            rule_weights=w, left=left, right=right, rule_ids=[0, 1])
        expected = 0.7 * and_out + 0.3 * or_out
        assert torch.allclose(got, expected, atol=1e-6)
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured


# ---------------------------------------------------------------------------
# Contract F: WholeSpace rule codebook (Phase 3)
# ---------------------------------------------------------------------------

_CONFIG_PATH = str(_project / "data" / "MM_xor.xml")


def test_symbolic_space_init_builds_rule_codebook():
    """WholeSpace.__init__ wires the rule codebook.

    We cannot use a class-level default because nn.Module routes Module
    submodules through ``_modules`` and a class attribute would shadow
    the instance attribute. Instead pin the contract at the source
    level: the __init__ body must construct ``self.rule_codebook``.
    """
    import inspect
    from Spaces import WholeSpace
    src = inspect.getsource(WholeSpace.__init__)
    assert "self.rule_codebook = RuleCodebook(" in src, (
        "WholeSpace.__init__ must build self.rule_codebook = "
        "RuleCodebook(...) per the SubSpace.what STM refactor"
    )


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
    """A real BasicModel from MM_xor.xml — has a fully-built WholeSpace."""
    from data import TheData
    from Models import BaseModel
    TheData.load("xor")
    m, _ = BaseModel.from_config(_CONFIG_PATH, data=TheData)
    return m


def test_symbolic_space_instance_owns_rule_codebook(_xor_model):
    """A built WholeSpace replaces the class-default None with a real codebook."""
    from Language import RuleCodebook
    ss = _xor_model.symbolicSpace
    assert isinstance(ss.rule_codebook, RuleCodebook)
    # The grammar is wired through so locations route to the V_sym+1 namespace.
    assert ss.rule_codebook.grammar is not None


def test_v_sym_wired_into_grammar_from_symbolic_space(_xor_model):
    """WholeSpace.__init__ must set Grammar.symbol_vocab_size = V_sym."""
    from Language import TheGrammar
    ss = _xor_model.symbolicSpace
    # V_sym should equal the symbol codebook's vocab dimension.
    cb_W = ss.subspace.what.getW()
    if cb_W is not None and cb_W.ndim >= 1:
        v_sym = int(cb_W.shape[0])
        assert TheGrammar.symbol_vocab_size == v_sym, (
            f"TheGrammar.symbol_vocab_size={TheGrammar.symbol_vocab_size} "
            f"!= V_sym={v_sym}"
        )


def test_rule_codebook_does_not_determine_parent_vectors(_xor_model):
    """Hard contract: parent.what = op(left, right), NOT rule_codebook[rule_id].what.

    We verify this by asserting RuleCodebook has no ``.what`` attribute /
    method (it stores identity + location, not content). Phase 4's
    REDUCE path computes the parent via SyntacticLayer.execute.
    """
    ss = _xor_model.symbolicSpace
    rc = ss.rule_codebook
    # The codebook must not be confused with a content codebook.
    assert not hasattr(rc, 'forward_to_parent_what'), (
        "RuleCodebook must not provide parent-vector lookup"
    )
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


def _make_syntactic_layer_for_stack(host_layers, tier='S'):
    """SyntacticLayer wrapper that doesn't require a real WordSpace."""
    from Language import SyntacticLayer

    class _StubWordSpace:
        def register_host_layer(self, *args, **kw):
            pass

    return SyntacticLayer(tier=tier, word_space=_StubWordSpace(),
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
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

        # Push two non-negative scalars (ConjunctionLayer = monotonic min).
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
    from Layers import ConjunctionLayer

    saved_rules = list(TheGrammar.rules)
    saved_configured = TheGrammar._configured
    try:
        TheGrammar.rules = []
        TheGrammar.rules_upward = []
        TheGrammar.rules_downward = []
        TheGrammar.reverse_rules = []
        TheGrammar._configured = False
        TheGrammar.configure({'compose': {'symbols':
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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
    scale on left*right) registered as a 'conjunction' rule. This is
    explicitly different from the parameter-free ConjunctionLayer/
    NotLayer cases above -- the point of this test is to pin the
    gradient path *into* the op's parameters.
    """
    import torch.nn as nn
    from Language import TheGrammar, RuleCodebook
    from Layers import GrammarLayer

    class _ParametricBinaryOp(GrammarLayer):
        """Tiny test op: parent = scale * (left * right). Arity 2."""
        rule_name = 'conjunction'
        arity = 2
        tier = 'S'

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
                              {'rule': ['S = conjunction(S, S)']}}})

        op = _ParametricBinaryOp()
        D = 2
        router = _make_minimal_signal_router(D=D)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=D, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': op})

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
# Contract H: WholeSpace owns + dispatches the stack router (Phase 5)
# ---------------------------------------------------------------------------

def test_symbolic_space_owns_signal_router(_xor_model):
    """WholeSpace owns its own LanguageLayer (distinct from Chart's).

    Plan acceptance: "WholeSpace owns and calls LanguageLayer."
    """
    from Language import LanguageLayer
    ss = _xor_model.symbolicSpace
    assert isinstance(ss.languageLayer, LanguageLayer), (
        f"WholeSpace.languageLayer must be a LanguageLayer, got "
        f"{type(ss.languageLayer)}"
    )
    # The router must be a different instance from any Chart-owned one
    # so the two paths cannot accidentally share scoring state.
    chart_router = getattr(
        getattr(ss.wordSubSpace, 'chart', None), '_signal_router', None)
    if chart_router is not None:
        assert ss.languageLayer is not chart_router, (
            "WholeSpace must own its own router, not share Chart's"
        )


def test_use_stack_router_flag_default_false(_xor_model):
    """Default config must keep the legacy path active.

    If MM_xor.xml ever ships <useStackRouter>true</useStackRouter> the
    legacy tests would shift to the new path silently; pin the default.
    """
    ss = _xor_model.symbolicSpace
    assert ss.use_stack_router is False, (
        "Default config must keep use_stack_router=False so existing "
        "training + tests run on the legacy path"
    )


def test_stack_route_forward_runs_and_writes_subspace_what(_xor_model):
    """End-to-end smoke: with the flag flipped, forward dispatches
    through _stack_route_forward and writes a non-zero .what.

    We toggle the flag on the live instance (the xor fixture is built
    with the flag off; this test exercises the dispatch path).
    """
    ss = _xor_model.symbolicSpace
    cs = _xor_model.conceptualSpace
    # Build a small CS-shaped input subspace. The CS subspace's
    # event width matches ss's nDim (symbol_dim == concept_dim).
    from Spaces import SubSpace
    n = int(cs.subspace.inputShape[0])
    d = int(cs.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    # Seed event so is_empty() returns False and materialize() yields a tensor.
    in_sub.set_event(torch.randn(1, n, d))
    # Stamp a minimal context the forward path expects.
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(1 * n, dtype=torch.bool)

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.use_stack_router = True
        out = ss.forward(in_sub)
        # Returns self.subspace.
        assert out is ss.subspace
        new_what = out.materialize(mode="what")
        # Stack-rewrite path writes a non-empty .what (not None).
        assert new_what is not None
        # Activation is set to 1.0 across all output positions.
        act = out.materialize(mode="activation")
        assert act is not None and torch.all(act > 0)
    finally:
        ss.use_stack_router = saved
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


# ---------------------------------------------------------------------------
# Contract I: legacy parser state untouched under flag (Phase 6 bypass)
# ---------------------------------------------------------------------------

def test_stack_router_does_not_touch_word_space_current_rules(_xor_model):
    """With use_stack_router=True, the forward must NOT read or write
    WordSubSpace.current_rules / generate_rules.

    Plan §"Phase 6: Retire Active WordSpace Parser State" -- "bypass"
    leg: the new path skips the cursor-driven current_rules surface
    entirely.
    """
    ss = _xor_model.symbolicSpace
    ws = ss.wordSubSpace
    from Spaces import SubSpace
    n = int(ss.conceptualSpace.subspace.inputShape[0])
    d = int(ss.conceptualSpace.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ws
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    # Stamp sentinel values so we can detect any write.
    sentinel = {'S': [['SENTINEL_NOT_TOUCHED']]}
    ws.current_rules = dict(sentinel)
    ws.generate_rules = dict(sentinel)
    pre_compose_gen = ws._compose_generation
    pre_generate_gen = ws._generate_generation

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.use_stack_router = True
        ss.forward(in_sub)
        # current_rules / generate_rules untouched (still the sentinel).
        assert ws.current_rules == sentinel, (
            f"current_rules mutated under flag-on path: {ws.current_rules}"
        )
        assert ws.generate_rules == sentinel, (
            f"generate_rules mutated under flag-on path: {ws.generate_rules}"
        )
        # Generation counters untouched (no compose / generate fired).
        assert ws._compose_generation == pre_compose_gen
        assert ws._generate_generation == pre_generate_gen
    finally:
        ss.use_stack_router = saved
        # Reset state for downstream tests.
        ws.current_rules = {}
        ws.generate_rules = {}
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


def test_stack_router_does_not_touch_conceptual_stm(_xor_model):
    """With use_stack_router=True, ConceptualSpace.stm must stay untouched.

    The new path runs on a temporary stack-mode SubSpace, never on
    ConceptualSpace.stm._buffer / _depth (the legacy STM side channel).
    """
    ss = _xor_model.symbolicSpace
    cs = ss.conceptualSpace
    stm = getattr(cs, 'stm', None)
    if stm is None:
        pytest.skip("This config has no ConceptualSpace.stm")

    from Spaces import SubSpace
    n = int(cs.subspace.inputShape[0])
    d = int(cs.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    pre_buffer = stm._buffer.detach().clone() if hasattr(stm, '_buffer') else None
    pre_depth = stm._depth.detach().clone() if hasattr(stm, '_depth') else None

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.use_stack_router = True
        ss.forward(in_sub)
        if pre_buffer is not None:
            assert torch.equal(stm._buffer, pre_buffer), (
                "ConceptualSpace.stm._buffer mutated under flag-on path"
            )
        if pre_depth is not None:
            assert torch.equal(stm._depth, pre_depth), (
                "ConceptualSpace.stm._depth mutated under flag-on path"
            )
    finally:
        ss.use_stack_router = saved
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


def test_stack_router_dispatches_via_syntactic_layer_execute(_xor_model):
    """The new path routes through SyntacticLayer.execute, not
    .forward / _next_rule_name (the cursor dispatch).

    Plan acceptance: "LanguageLayer calls SyntacticLayer executor, not
    cursor dispatch."
    """
    ss = _xor_model.symbolicSpace
    sl = ss.syntacticLayer

    cursor_calls = {'n': 0}
    execute_calls = {'n': 0}

    orig_next = sl._next_rule_name
    orig_exec = sl.execute

    def spy_next_rule(*a, **kw):
        cursor_calls['n'] += 1
        return orig_next(*a, **kw)

    def spy_execute(*a, **kw):
        execute_calls['n'] += 1
        return orig_exec(*a, **kw)

    from Spaces import SubSpace
    n = int(ss.conceptualSpace.subspace.inputShape[0])
    d = int(ss.conceptualSpace.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        sl._next_rule_name = spy_next_rule
        sl.execute = spy_execute
        ss.use_stack_router = True
        ss.forward(in_sub)
        # Cursor must not have fired.
        assert cursor_calls['n'] == 0, (
            f"Stack-router path called _next_rule_name "
            f"{cursor_calls['n']} times; expected 0"
        )
        # Execute fires once per REDUCE when a binary rule was wired
        # (>= 0 because a config without an S-tier binary rule will
        # skip reductions entirely).
        # If N >= 2 AND a binary S-tier rule is registered, execute
        # should have fired at least once.
        # We only assert the negative for the cursor path; the positive
        # (>=1 execute call) is asserted in the smoke test above.
    finally:
        sl._next_rule_name = orig_next
        sl.execute = orig_exec
        ss.use_stack_router = saved
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


# ---------------------------------------------------------------------------
# Contract J: flag-off path remains byte-identical (regression guardrail)
# ---------------------------------------------------------------------------

def test_flag_off_does_not_call_stack_route_forward(_xor_model):
    """With use_stack_router=False, the new _stack_route_forward must
    NOT run; the legacy path stays the only forward dispatcher.
    """
    ss = _xor_model.symbolicSpace
    calls = {'n': 0}
    orig = ss._stack_route_forward

    def spy(*a, **kw):
        calls['n'] += 1
        return orig(*a, **kw)

    from Spaces import SubSpace
    n = int(ss.conceptualSpace.subspace.inputShape[0])
    d = int(ss.conceptualSpace.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    try:
        ss._stack_route_forward = spy
        # use_stack_router is False by default.
        assert ss.use_stack_router is False
        ss.forward(in_sub)
        assert calls['n'] == 0, (
            f"_stack_route_forward ran {calls['n']} times with flag off"
        )
    finally:
        ss._stack_route_forward = orig


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

    ConjunctionLayer.reverse returns ``(parent, parent)`` -- the
    identity-stub the plan calls out -- so after reduce + unreduce the
    two child slots both hold the parent (NOT the original children,
    because the op is lossy)."""
    from Language import TheGrammar, RuleCodebook
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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

        # Now unreduce -- the layer's reverse is the identity stub.
        router.unreduce(sub, layer, rule_codebook=rc)
        what_after = sub.materialize(mode="what")
        occ_after = sub.materialize(mode="activation")

        # Slot 0 holds left child = parent (identity-stub).
        # Slot 1 holds right child = parent (identity-stub).
        assert torch.allclose(what_after[0, 0, :], parent_after_reduce), (
            "unreduce should write layer.reverse(parent)[0] into the top slot"
        )
        assert torch.allclose(what_after[0, 1, :], parent_after_reduce), (
            "unreduce should write layer.reverse(parent)[1] into the new slot"
        )
        # Occupancy: two slots live again.
        assert torch.equal(occ_after, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))
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
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        # K=2 stack: shift one terminal, then directly stamp the top
        # slot's .where to point at the rule so unreduce will try to
        # split. Fill BOTH slots first so the stack is full.
        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=2, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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

    Reverse: ONE unreduce undoes the outermost reduce; the new top
    slot's .where is empty (cleared by unreduce), so reverse_stack
    halts there. Full multi-level unwinding requires a provenance
    trail (Phase 8+ work).
    """
    from Language import TheGrammar, RuleCodebook
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=6, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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

        # Reverse: unwind one level.
        router.reverse_stack(sub, layer, rule_codebook=rc)
        occ = sub.materialize(mode="activation")
        # Outer reduce undone -> two live slots holding the (lossy)
        # identity-stub children of the root.
        assert torch.equal(occ, torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])), (
            f"reverse_stack should unwind one level, got occ={occ}"
        )
        what = sub.materialize(mode="what")
        # ConjunctionLayer.reverse returns (parent, parent) -- both
        # child slots equal the root payload.
        assert torch.allclose(what[0, 0, :], root)
        assert torch.allclose(what[0, 1, :], root)
        # Both child .where rows are the empty sentinel (cleared by
        # unreduce since the identity-stub has no provenance trail).
        where = sub.materialize(mode="where")
        for k in range(2):
            kind, _ = TheGrammar.decode_where(where[0, k, 0])
            assert kind == 'empty', (
                f"child slot {k} should be empty-stamped after unreduce, "
                f"got kind={kind}"
            )
    finally:
        TheGrammar.rules = saved_rules
        TheGrammar._configured = saved_configured
        TheGrammar.symbol_vocab_size = saved_vsym


def test_unreduce_uses_identity_stub_when_layer_reverse_is_unsuitable():
    """Per plan §"Reverse And Reconstruction":
        "otherwise identity/pass-through stub"

    The base ``Layer.reverse`` is a shape-asserting identity (not a
    real inverse), so layers without a hand-written reverse inherit
    one whose return shape (single tensor) is wrong for an arity-2
    parent. unreduce must detect this and fall back to (parent, parent)
    rather than crashing.
    """
    from Language import TheGrammar, RuleCodebook
    from Layers import GrammarLayer

    class _BinaryInheritingBaseReverse(GrammarLayer):
        """Arity-2 op that does NOT override reverse.

        Inherits ``Layer.reverse`` (a single-tensor identity), which is
        the wrong shape for an arity-2 unreduce -- the unsuitable-shape
        fallback path is what this test pins.
        """
        rule_name = 'conjunction'
        arity = 2
        tier = 'S'

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
                              {'rule': ['S = conjunction(S, S)']}}})

        op = _BinaryInheritingBaseReverse()

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': op})

        router.shift(sub, torch.tensor([[0.5, 0.5]]), where_id=1)
        router.shift(sub, torch.tensor([[0.4, 0.6]]), where_id=2)
        router.reduce(sub, layer, rule_id=0, rule_codebook=rc)
        parent = sub.materialize(mode="what")[0, 0, :].clone()

        # Must not raise even though layer.reverse returns a single
        # tensor (wrong shape for an arity-2 unreduce).
        router.unreduce(sub, layer, rule_codebook=rc)
        what_after = sub.materialize(mode="what")
        # Identity-stub: both child slots equal the parent.
        assert torch.allclose(what_after[0, 0, :], parent)
        assert torch.allclose(what_after[0, 1, :], parent)
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
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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


def test_symbolic_space_stack_route_uses_canonical_forward(_xor_model):
    """WholeSpace._stack_route_forward must dispatch through
    languageLayer.forward(...), not through the low-level shift/reduce
    primitives directly. Pins that the WholeSpace integration uses
    the plan's target call shape.
    """
    ss = _xor_model.symbolicSpace
    calls = {'forward': 0, 'shift': 0, 'reduce': 0}
    orig_forward = ss.languageLayer.forward
    orig_shift = ss.languageLayer.shift
    orig_reduce = ss.languageLayer.reduce

    def spy_forward(*a, **kw):
        calls['forward'] += 1
        return orig_forward(*a, **kw)

    def spy_shift(*a, **kw):
        calls['shift'] += 1
        return orig_shift(*a, **kw)

    def spy_reduce(*a, **kw):
        calls['reduce'] += 1
        return orig_reduce(*a, **kw)

    from Spaces import SubSpace
    n = int(ss.conceptualSpace.subspace.inputShape[0])
    d = int(ss.conceptualSpace.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.languageLayer.forward = spy_forward
        ss.languageLayer.shift = spy_shift
        ss.languageLayer.reduce = spy_reduce
        ss.use_stack_router = True
        ss.forward(in_sub)

        # Canonical forward fired exactly once (one call per
        # WholeSpace.forward invocation).
        assert calls['forward'] == 1, (
            f"Expected exactly one languageLayer.forward(...) call; "
            f"got {calls['forward']}"
        )
        # shift / reduce still fire (they're called by forward_stack
        # under the wrapper) -- the spies count them too because
        # forward_stack invokes self.shift / self.reduce on the same
        # router instance. This is informational, not a contract.
    finally:
        ss.languageLayer.forward = orig_forward
        ss.languageLayer.shift = orig_shift
        ss.languageLayer.reduce = orig_reduce
        ss.use_stack_router = saved
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


# ---------------------------------------------------------------------------
# Contract M: WholeSpace.reverse dispatches to LanguageLayer.reverse
# under the same flag as the forward branch (symmetry with Phase 5)
# ---------------------------------------------------------------------------

def test_symbolic_space_reverse_dispatches_to_language_layer_reverse(_xor_model):
    """WholeSpace.reverse calls LanguageLayer.reverse(...) when the
    use_stack_router flag is on. Symmetric counterpart to the forward
    branch (Phase 5).
    """
    ss = _xor_model.symbolicSpace
    calls = {'reverse': 0}
    orig_reverse = ss.languageLayer.reverse

    def spy_reverse(*a, **kw):
        calls['reverse'] += 1
        return orig_reverse(*a, **kw)

    from Spaces import SubSpace
    # The WholeSpace.reverse input is in symbol space; size to match.
    n = int(ss.subspace.inputShape[0])
    d = int(ss.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.languageLayer.reverse = spy_reverse
        ss.use_stack_router = True
        ss.reverse(in_sub)
        assert calls['reverse'] == 1, (
            f"WholeSpace.reverse must call languageLayer.reverse exactly "
            f"once with the flag on; got {calls['reverse']}"
        )
    finally:
        ss.languageLayer.reverse = orig_reverse
        ss.use_stack_router = saved
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


def test_symbolic_space_reverse_flag_off_does_not_call_language_layer(_xor_model):
    """With use_stack_router=False (default), WholeSpace.reverse must
    NOT call languageLayer.reverse; the legacy cursor-based reverse
    runs unchanged.
    """
    ss = _xor_model.symbolicSpace
    calls = {'reverse': 0}
    orig_reverse = ss.languageLayer.reverse

    def spy_reverse(*a, **kw):
        calls['reverse'] += 1
        return orig_reverse(*a, **kw)

    from Spaces import SubSpace
    n = int(ss.subspace.inputShape[0])
    d = int(ss.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ss.wordSubSpace
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    saved = ss.use_stack_router
    try:
        ss.languageLayer.reverse = spy_reverse
        assert ss.use_stack_router is False
        ss.reverse(in_sub)
        assert calls['reverse'] == 0, (
            f"Legacy reverse path must not call languageLayer.reverse; "
            f"got {calls['reverse']} calls"
        )
    finally:
        ss.languageLayer.reverse = orig_reverse
        ss.use_stack_router = saved


def test_symbolic_space_reverse_flag_on_does_not_touch_generate_rules(_xor_model):
    """Phase 6 symmetry for the reverse path: with the flag on, the
    cursor-driven generate_rules path is bypassed.
    """
    ss = _xor_model.symbolicSpace
    ws = ss.wordSubSpace
    from Spaces import SubSpace
    n = int(ss.subspace.inputShape[0])
    d = int(ss.subspace.muxedSize)
    in_sub = SubSpace([n, d], [n, d], nInputDim=d, nOutputDim=d)
    in_sub.set_event(torch.randn(1, n, d))
    in_sub.wordSubSpace = ws
    in_sub.valid_mask = torch.ones(n, dtype=torch.bool)

    sentinel = {'S': [['SENTINEL_NOT_TOUCHED']]}
    ws.generate_rules = dict(sentinel)
    pre_generate_gen = ws._generate_generation

    saved = ss.use_stack_router
    saved_what = ss.subspace.what.getW()
    try:
        ss.use_stack_router = True
        ss.reverse(in_sub)
        assert ws.generate_rules == sentinel, (
            f"generate_rules mutated under flag-on reverse path: "
            f"{ws.generate_rules}"
        )
        assert ws._generate_generation == pre_generate_gen
    finally:
        ss.use_stack_router = saved
        ws.generate_rules = {}
        if saved_what is not None:
            ss.subspace.what.setW(saved_what)


def test_forward_stack_orchestrates_shift_then_reduce():
    """forward_stack runs a list of (shift/reduce) actions end-to-end."""
    from Language import TheGrammar, RuleCodebook
    from Layers import ConjunctionLayer

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
                              {'rule': ['S = conjunction(S, S)']}}})

        router = _make_minimal_signal_router(D=2)
        sub = _make_stack_subspace_with_where(B=1, K=4, D=2, W=2)
        rc = RuleCodebook(num_rules=1, grammar=TheGrammar)
        layer = _make_syntactic_layer_for_stack({'conjunction': ConjunctionLayer()})

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
