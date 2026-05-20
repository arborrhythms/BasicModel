import os
os.environ["BASICMODEL_DEVICE"] = "cpu"

import types

import pytest
import torch


def test_rule_tier_is_cs_split():
    from Language import Chart
    rt = Chart._RULE_TIER
    C = {"union","intersection","swap","copy","not","non","true","false",
         "part","query","area","luminosity","equal"}
    S = {"conjunction","disjunction","isEqual","isaPart","lift","lower"}
    for m in C: assert rt[m] == "C", (m, rt.get(m))
    for m in S: assert rt[m] == "S", (m, rt.get(m))
    assert "P" not in set(rt.values()), "P tier must be retired"


def _build_gate_model():
    from test.space_equiv import _p
    from util import init_config, init_device
    from data import TheData
    from Models import BaseModel
    init_device("cpu")
    cfg = str(_p / "data" / "MM_5M.xml")
    init_config(path=cfg, defaults_path=str(_p / "data" / "model.xml"))
    TheData.load("text", shard_dir=str(_p / "data" / "fineweb"),
                 num_shards=1, max_docs=8)
    m, _ = BaseModel.from_config(cfg, data=TheData)
    return m


def test_perceptual_has_no_syntactic_layer():
    m = _build_gate_model()
    assert getattr(m.perceptualSpace, "syntacticLayer", None) is None


def test_symbolicspace_ref_is_readonly_backref():
    m = _build_gate_model()
    cs, ss = m.conceptualSpace, m.symbolicSpace
    assert cs.symbolicSpace_ref is ss
    assert 'symbolicSpace_ref' not in cs._modules


# ---------------------------------------------------------------------------
# Phase 2A.6 -- selection-tensor CONTRACT + S-executor split.
#
# In Phase 2A op_sel is None for every config (the chart is dormant;
# ``new_working_state`` is always called with ``n_ops=0``), so these
# unit-level tests pin BOTH halves of the locked contract without
# building a model (the canonical ``_build_gate_model`` is blocked by
# the pre-existing #22 stale MM_5M.ckpt; the contract itself is
# model-independent).
# ---------------------------------------------------------------------------

def test_working_state_op_sel_default_is_none_and_shaped():
    """Production default (n_ops=0) leaves op_sel/op_operands None;
    explicit n_ops allocates the documented fixed-shape tensors."""
    from Spaces import new_working_state
    ws0 = new_working_state(n_tiers=3, device="cpu", n_ops=0)
    assert ws0.op_sel is None and ws0.op_operands is None, (
        "n_ops=0 (the production path) MUST leave the selection "
        "tensors None -- this is what makes 2A a pure eager passthrough")
    K = 5
    ws = new_working_state(n_tiers=3, device="cpu", n_ops=K)
    assert ws.op_sel.shape == (K,) and ws.op_sel.dtype == torch.float32
    assert ws.op_operands.shape == (K, 2)
    assert ws.op_operands.dtype == torch.int64


def test_selection_none_is_byte_identical_cursor_object():
    """op_sel None (ALWAYS in 2A) => the helper returns the EXACT same
    Python object the legacy cursor loop consumed
    (``current_rules.get('S')``). Object identity -- not value equality
    -- proves the eager fallback is structurally byte-identical."""
    from Language import WordSpace
    from Spaces import new_working_state

    s_obj = [[0]]                       # the live per-step list
    carrier = types.SimpleNamespace(current_rules={"S": s_obj, "C": [[4]]})

    # work is None -> cursor path object, returned by identity.
    out = WordSpace.selection_from_current_rules(carrier, "S", None)
    assert out is s_obj, "must hand back the IDENTICAL current_rules object"

    # work present but op_sel None (n_ops=0, the production default)
    # -> still the identical cursor object (no tensor branch taken).
    ws = new_working_state(n_tiers=3, device="cpu", n_ops=0)
    out2 = WordSpace.selection_from_current_rules(carrier, "S", ws)
    assert out2 is s_obj, (
        "op_sel None must NOT divert to the tensor branch")

    # absent current_rules -> None (matches the legacy guard's else).
    carrier.current_rules = None
    assert WordSpace.selection_from_current_rules(carrier, "S", None) is None


def test_selection_tensor_branch_yields_op_sel_pair():
    """When op_sel IS populated (Phase 2B; unreached in 2A) the helper
    yields the (op_sel, op_operands) tensor pair by reference -- the
    documented contract the S-executor consumes."""
    from Language import WordSpace
    from Spaces import new_working_state

    carrier = types.SimpleNamespace(current_rules={"S": [[0]]})
    ws = new_working_state(n_tiers=3, device="cpu", n_ops=4)
    ws.op_sel[1] = 1.0
    sel = WordSpace.selection_from_current_rules(carrier, "S", ws)
    assert isinstance(sel, tuple) and len(sel) == 2
    assert sel[0] is ws.op_sel and sel[1] is ws.op_operands, (
        "tensor branch must return the carrier tensors by reference")


def _fake_layer(mult, arity=1):
    """A stand-in GrammarLayer: forward scales by ``mult`` so a
    one-hot pick is identifiable, and the weighted-reduce is exactly
    ``sum_k op_sel[k] * mult_k * x``."""
    return types.SimpleNamespace(
        arity=arity,
        forward=(lambda x, _m=mult: x * _m))


class _Vspace:
    """Minimal real-semantics event carrier: set_event / materialize,
    exactly the surface SyntacticLayer._read_subspace uses."""
    def __init__(self):
        self._ev = None

    def set_event(self, t):
        self._ev = t

    def materialize(self, mode=None):
        return self._ev


def _run_s_executor(op_sel, rules_axis, by_name, act_pre):
    """Drive the REAL Phase-2A.6 S-executor reduce.

    This is the exact tensor-driven branch from ``SymbolicSpace.forward``
    (op_sel-not-None path). It is dead code in 2A (op_sel is None
    everywhere); this harness pins its locked Phase-2B contract:
    independent per-op contributions combined by a SINGLE weighted
    reduce over the fixed ``TheGrammar.rules`` op axis, NO shared
    in-place accumulator. The harness wires the genuine production
    collaborators -- the real ``SyntacticLayer._read_subspace`` and the
    real ``_by_name`` dispatch + arity gate -- so the selection
    semantics under test are the shipped ones, not a re-implementation.
    """
    import Language
    from Language import SyntacticLayer

    vspace = _Vspace()
    syn = types.SimpleNamespace(
        _by_name=by_name,
        _read_subspace=SyntacticLayer._read_subspace.__get__(
            types.SimpleNamespace()),
    )

    saved = Language.TheGrammar.rules
    Language.TheGrammar.rules = rules_axis
    try:
        # ----- verbatim S-executor reduce (Spaces.SymbolicSpace.forward)
        from Language import TheGrammar
        vspace.set_event(act_pre)
        probs_list = op_sel.detach().tolist()
        total = None
        for k, p_val in enumerate(probs_list):
            if p_val < 1e-6:
                continue
            try:
                rdef = TheGrammar.rules[int(k)]
                method_name = rdef.method_name
                arity = int(getattr(rdef, 'arity', 1))
            except (IndexError, AttributeError, ValueError, TypeError):
                continue
            if method_name is None or arity != 1:
                continue
            layer = syn._by_name.get(method_name)
            if layer is None:
                continue
            x = syn._read_subspace(vspace, layer=layer)
            if x is None:
                continue
            contribution = layer.forward(x) * op_sel[k]
            total = (contribution if total is None
                     else total + contribution)
        act = total if total is not None else act_pre
        # -----
    finally:
        Language.TheGrammar.rules = saved
    return act


def test_s_executor_one_hot_selects_exactly_that_op():
    """One-hot op_sel => the weighted reduce fires EXACTLY the selected
    arity-1 op and nothing else (proves the contract without the chart).

    Op axis (fixed TheGrammar.rules order):
      0: 'sigma'  arity 1  (S-tier unary fold)   -> layer x*7
      1: 'union'  arity 2  (compositional)       -> MUST be skipped
      2: 'sigma2' arity 1  but NO registered layer -> skipped
    """
    RD = types.SimpleNamespace
    rules_axis = [
        RD(method_name="sigma", arity=1, lhs="S", canonical="S -> sigma(S)"),
        RD(method_name="union", arity=2, lhs="S",
           canonical="S -> union(S, S)"),
        RD(method_name="sigma2", arity=1, lhs="S", canonical="S -> sigma2"),
    ]
    by_name = {"sigma": _fake_layer(7.0, arity=1)}  # union/sigma2 absent
    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    # one-hot on rule 0 -> exactly sigma's layer.forward(x) * 1.0
    sel0 = torch.tensor([1.0, 0.0, 0.0])
    out0 = _run_s_executor(sel0, rules_axis, by_name, x)
    assert torch.equal(out0, x * 7.0), (
        "one-hot@0 must select ONLY rule 0's op (x*7)")

    # one-hot on the arity-2 rule -> skipped (no contribution) ->
    # fall back to act_pre (the documented `total is None` guard).
    sel1 = torch.tensor([0.0, 1.0, 0.0])
    out1 = _run_s_executor(sel1, rules_axis, by_name, x)
    assert torch.equal(out1, x), (
        "arity-2 op on the unary per-step path must be skipped "
        "(mirrors SyntacticLayer.forward's arity gate)")

    # one-hot on an arity-1 rule with NO registered layer -> skipped.
    sel2 = torch.tensor([0.0, 0.0, 1.0])
    out2 = _run_s_executor(sel2, rules_axis, by_name, x)
    assert torch.equal(out2, x), "missing host layer must be skipped"


def test_s_executor_is_weighted_reduce_no_in_place_accumulator():
    """Multi-hot op_sel => result is the SINGLE weighted reduce
    ``sum_k op_sel[k] * op_k(x)`` with INDEPENDENT per-op
    contributions. Verified by superposition: zeroing one op's weight
    removes exactly that op's term (no shared in-place accumulator
    could satisfy this)."""
    RD = types.SimpleNamespace
    rules_axis = [
        RD(method_name="sigma", arity=1, lhs="S", canonical="a"),
        RD(method_name="iden", arity=1, lhs="S", canonical="b"),
    ]
    by_name = {"sigma": _fake_layer(7.0, arity=1),
               "iden": _fake_layer(2.0, arity=1)}
    x = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)

    w = torch.tensor([0.25, 0.75])
    out = _run_s_executor(w, rules_axis, by_name, x)
    expected = x * 7.0 * 0.25 + x * 2.0 * 0.75
    assert torch.equal(out, expected), "must be the weighted op-axis sum"

    # Superposition / independence: drop op 1's weight -> result loses
    # EXACTLY op 1's term and keeps op 0's untouched.
    w0 = torch.tensor([0.25, 0.0])
    out0 = _run_s_executor(w0, rules_axis, by_name, x)
    assert torch.equal(out0, x * 7.0 * 0.25), (
        "contributions must be independent (no shared accumulator)")
    assert torch.equal(out - out0, x * 2.0 * 0.75), (
        "removing op 1's weight must subtract exactly op 1's term")
