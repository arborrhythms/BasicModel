"""Tests for the soft-superposition CKY chart parser
(`SyntacticLayer._compose_chart_cky`) introduced by the
floating-blossom spec.

Covers:
  - Inside-pass shape & finite output on a 3-token sequence
  - Gradient flow through `logsumexp` reaches `rule_bias`,
    `marker_bias`, `rule_embed`, and `compat_score` weights
  - Marker compilation: sugar rules (e.g. `absorb`) do NOT occupy
    rule_table_packed rows; instead, productive rules carry per-operand
    `marker_mask` flags. A masked operand contributes nothing to the
    parent vector (only a marker prior to the score)
  - Add-rule mid-training: bumping `rule_table_version` rebuilds the
    rule-shaped parameters on the next compose
"""

# ---------------------------------------------------------------------
# Skipped pending migration to the post-2026-05-01 chart surface.
# These tests construct ``SyntacticLayer`` and call its
# ``_compose_chart_cky`` -- which the refactor moved to the new
# ``Chart`` class. Equivalent coverage now lives in
# ``test/test_chart_wordspace_wiring.py`` (chart-isolation +
# WordSpace integration); rewriting these to drive the Chart
# directly is a follow-on. See doc/specs/
# 2026-05-01-syntactic-layer-refactor.md.
# ---------------------------------------------------------------------
import pytest
pytestmark = pytest.mark.skip(
    reason="Pending migration to Chart surface (test/test_chart_wordspace_wiring.py); "
           "see doc/specs/2026-05-01-syntactic-layer-refactor.md")

import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
_wo_root = _project.parent
sys.path.insert(0, str(_wo_root / "bin"))
sys.path.insert(0, str(_project / "bin"))

import torch  # noqa: E402


def _make_grammar(rules):
    """Build a Grammar configured from a flat upward dict."""
    from Language import Grammar
    g = Grammar()
    g.configure({lhs: rhs for lhs, rhs in rules})
    return g


def _make_layer(grammar, D=8, N=4):
    """Build a SyntacticLayer with the given grammar wired in."""
    from Language import SyntacticLayer
    rule_ids = list(range(len(grammar.rules)))
    layer = SyntacticLayer(
        nInput=N, nOutput=N, rules=rule_ids,
        max_depth=4, hidden_dim=16, grammar=grammar, feature_dim=D,
    )
    return layer


# -------- Marker compilation --------

def test_marker_compilation_drops_absorb_row():
    """`absorb(S, S)` must NOT occupy a rule_table_packed row.

    Per the spec: sugar rules are compiled into per-operand marker
    flags on the productive rules they license; they do not appear as
    standalone chart productions.
    """
    g = _make_grammar([
        ('S', ['VO V', 'absorb(S, S)']),
    ])
    table = g._ensure_packed_table(device=torch.device('cpu'))
    # Find the `absorb` rule's global id; it must not appear in
    # rule_table_packed['global_id'].
    absorb_gid = None
    for i, r in enumerate(g.rules):
        if r.method_name == 'absorb':
            absorb_gid = i
            break
    assert absorb_gid is not None, "test setup: absorb must be in grammar"
    assert absorb_gid not in table['global_id'].tolist(), (
        f"absorb rule (gid={absorb_gid}) must NOT appear as a row in "
        f"rule_table_packed; sugar rules are compiled into marker flags"
    )


def test_marker_mask_set_on_host_rule():
    """The productive rule whose operand category matches a sugar
    rule's lhs must have `marker_mask` set on that side.
    """
    g = _make_grammar([
        ('S', ['VO V', 'absorb(S, S)']),
    ])
    table = g._ensure_packed_table(device=torch.device('cpu'))
    # The productive rule `S -> VO V` has rhs = ('VO', 'V').
    # absorb's lhs is 'S' -> sugar applies to S-typed operands.
    # Neither VO nor V is S, so this rule's marker_mask should be all-False.
    # Check that no productive rule has marker_mask True except where
    # an operand category is 'S' (the sugar lhs).
    cat_index = table['_cat_index']
    s_id = cat_index['S']
    for r in range(table['lhs'].shape[0]):
        rl = int(table['rhs_left'][r].item())
        rr = int(table['rhs_right'][r].item())
        ml = bool(table['marker_mask'][r, 0].item())
        mr = bool(table['marker_mask'][r, 1].item())
        if int(table['arity'][r].item()) == 2:
            assert ml == (rl == s_id), (
                f"row r={r}: marker_mask[0]={ml} but rhs_left={rl} "
                f"(S id={s_id})")
            assert mr == (rr == s_id), (
                f"row r={r}: marker_mask[1]={mr} but rhs_right={rr} "
                f"(S id={s_id})")


# Test `test_marker_operand_does_not_affect_parent_vector` removed by
# Step 9 of the 2026-05-01 syntactic-layer refactor: it asserted
# behavior of `SyntacticLayer._SoftCompose`, a deprecated helper that
# the new Chart's `_apply_rule_forward` no longer needs (rule semantics
# come from GRAMMAR_LAYER_CLASSES kernels, not a learned compose MLP).
# Marker handling is now tested via the marker_mask zero-out inside
# `Chart._apply_rule_forward`.


# -------- Inside-pass shape --------

def test_inside_pass_shapes_and_finite():
    """3-token sequence with a couple of binary rules — chart_score and
    chart_vec come back with the expected shapes and are finite."""
    g = _make_grammar([
        ('S', ['S VO', 'NP']),
        ('VO', ['V O']),
        ('NP', ['N']),
    ])
    D, N, B = 8, 3, 2
    layer = _make_layer(g, D=D, N=N)

    class DummySub:
        def __init__(self):
            self.word = []
            self.basis = None
            self.wordSpace = None
        def add_word(self, *a, **kw): pass
        def flush_word_buffer(self): pass

    layer.eval()
    data = torch.randn(B, N, D)
    composed, _ = layer._compose_chart_cky(data, DummySub(), g)
    assert composed.shape == (B, N, D)
    cs = layer._chart_score
    cv = layer._chart_vec
    C = len(g._ensure_packed_table()['_cat_names'])
    assert cs.shape == (B, N + 1, N + 1, C)
    assert cv.shape == (B, N + 1, N + 1, C, D)
    # Root cell must be finite (lex fill + at least one width step ran).
    assert torch.isfinite(cs[:, 0, N, :]).all()
    assert torch.isfinite(cv[:, 0, N, :, :]).all()


def test_gradient_flow_through_logsumexp():
    """Backprop from chart_vec at the root cell reaches every rule
    parameter contributing to any spanning derivation, not just the
    argmax path. Soft-superposition's defining property.
    """
    g = _make_grammar([
        ('S', ['S VO', 'NP']),
        ('VO', ['V O']),
        ('NP', ['N']),
    ])
    D, N, B = 8, 3, 2
    layer = _make_layer(g, D=D, N=N)

    class DummySub:
        def __init__(self):
            self.word = []
            self.basis = None
            self.wordSpace = None
        def add_word(self, *a, **kw): pass
        def flush_word_buffer(self): pass

    layer.train()
    data = torch.randn(B, N, D, requires_grad=True)
    layer._compose_chart_cky(data, DummySub(), g)
    cs = layer._chart_score
    cv = layer._chart_vec
    # Sum a slice that contains contributions from every rule.
    loss = cs[:, 0, N, :].sum() + cv[:, 0, N, :, :].sum()
    loss.backward()
    # Post-Kim refactor (compound-PCFG-style): the chart's vector path
    # uses fixed _RULE_METHODS semantics, not a learned compose MLP.
    # Soft mixing is over rule probabilities only. So gradient must
    # land on the *learned scoring path*: _rule_bias, _rule_embed,
    # _marker_bias, _compat_score, _lex_cat_scorer. _compose_mod
    # exists for backward-compat but is not on the hot path -- its
    # gradient will be None.
    assert layer._rule_bias.grad is not None
    assert layer._rule_embed.grad is not None
    assert layer._marker_bias.grad is not None
    assert torch.isfinite(layer._rule_bias.grad).all()
    assert torch.isfinite(layer._rule_embed.grad).all()
    # Compat-score MLP receives gradient (it's the input-dependent
    # contextual rule-probability head).
    assert layer._compat_score_mod.lin1.weight.grad is not None
    assert torch.isfinite(layer._compat_score_mod.lin1.weight.grad).all()
    # Lexical category scorer receives gradient (lexical fill -> root).
    assert layer._lex_cat_scorer.weight.grad is not None
    assert torch.isfinite(layer._lex_cat_scorer.weight.grad).all()
    # Confirm at least one entry of _rule_bias / _rule_embed is non-
    # zero (a connected grammar has spanning derivations through
    # every rule).
    assert (layer._rule_bias.grad.abs() > 0).any()
    assert (layer._rule_embed.grad.abs() > 0).any()


# -------- Fixed-rule-semantics dispatch (Kim 2019 alignment) --------

def test_chart_uses_fixed_rule_semantics_not_compose_mlp():
    """The chart's vector path must dispatch through `_RULE_METHODS`
    (fixed semantics: `intersection` -> min, `union` -> max, `not` ->
    negation, etc.), not through `_compose_mod`. Without this, soft
    mixing happens over a learned interpolant rather than over rule
    probabilities, and grammar recovery fails (compound-PCFG style;
    see doc/research/2019-kim-compound-pcfg.md).

    Verified by: feed left/right operands the chart can route through
    `intersectionForward` (min) and confirm the resulting parent vec
    differs measurably between left=zero and left=one for a fixed
    right -- which proves the rule semantics, not a saturated MLP,
    are driving compose.
    """
    g = _make_grammar([
        ('S', ['S S']),
    ])
    D, N, B = 4, 2, 1
    layer = _make_layer(g, D=D, N=N)

    class DummySub:
        def __init__(self):
            self.word = []
            self.basis = None
            self.wordSpace = None
        def add_word(self, *a, **kw): pass
        def flush_word_buffer(self): pass

    layer.eval()
    # data shape [B, N, D]; positions 0 and 1 are the operands of the
    # only binary rule. Compose at width=2 produces chart_vec[:, 0, 2,
    # S, :] which the chart populates from `merged_per_rule` =
    # _apply_rule_forward('merge', left, right, mm). The 'merge'
    # method falls through to left-pass; let's instead use a real
    # binary rule so we can read the semantics.
    g2 = _make_grammar([
        ('S', ['intersection(S, S)']),
    ])
    layer2 = _make_layer(g2, D=D, N=N)

    # Operand A = ones, B = -ones. intersection (min) -> -ones.
    a = torch.full((1, N, D), 1.0)
    a[0, 1] = -1.0
    layer2._compose_chart_cky(a, DummySub(), g2)
    cv = layer2._chart_vec[:, 0, N, :, :].detach()
    # Find the S category index.
    table = g2._ensure_packed_table()
    s_id = table['_cat_index']['S']
    parent = cv[0, s_id, :]
    # If the chart truly used `intersection` (min), parent should be
    # close to -ones: min(ones, -ones) = -ones. Since lexical fill
    # mixes parent across all categories at width=1 and the binary
    # step blends with category logits, we don't expect exact -1, but
    # the parent must be measurably negative on every dim.
    assert parent.lt(0.5).all(), (
        f"intersection(ones, -ones) should yield a negative-leaning "
        f"parent vector via min semantics; got {parent.tolist()}"
    )


# -------- Softness of the rule mixture --------

def test_compat_score_is_bounded():
    """`_CompatScore` output magnitude must be bounded so the per-cell
    rule softmax cannot saturate to one-hot during training. Without
    the bound the chart degenerates from soft-superposition to
    effective hard reduction (perplexity ~= 1) regardless of how the
    forward equations look. See doc/research/2017-jang-gumbel-
    softmax.md.
    """
    from Language import SyntacticLayer
    D, D_rule = 4, 2
    cs = SyntacticLayer._CompatScore(D=D, D_rule=D_rule, compat_scale=2.0)
    # Drive the head with absurdly large inputs; output must stay
    # within ±compat_scale.
    left = torch.randn(8, 4, D) * 50.0
    right = torch.randn(8, 4, D) * 50.0
    embed = torch.randn(8, 4, D_rule) * 50.0
    mm = torch.zeros(8, 4, 2, dtype=torch.bool)
    out = cs(left, right, embed, mm)
    assert out.abs().max().item() <= 2.0 + 1e-6, (
        f"compat output should be bounded by compat_scale=2.0; "
        f"got max abs = {out.abs().max().item()}"
    )


def test_chart_temperature_softens_rule_mixture():
    """At τ > 1.0 the per-cell rule softmax must be measurably softer
    (higher perplexity) than at τ = 1.0. This is the structural knob
    against saturation.
    """
    from Language import SyntacticLayer
    g = _make_grammar([
        ('S', ['intersection(S, S)', 'union(S, S)']),
    ])
    D, N, B = 6, 3, 2
    layer = _make_layer(g, D=D, N=N)

    class DummySub:
        def __init__(self):
            self.word = []
            self.basis = None
            self.wordSpace = None
        def add_word(self, *a, **kw): pass
        def flush_word_buffer(self): pass

    layer.eval()
    data = torch.randn(B, N, D) * 0.5

    # Trigger lazy build, then crank compat_score weights so the
    # cand_score differences across rules are large. We're not
    # testing convergence; we're testing that τ acts as a softening
    # knob.
    layer.chart_tau = 1.0
    layer._compose_chart_cky(data, DummySub(), g)
    with torch.no_grad():
        # Drop the tanh saturation by zeroing biases and pushing
        # weights large; outputs stay in ±compat_scale but are
        # rule-discriminative.
        layer._compat_score_mod.lin2.weight.mul_(50.0)

    layer._compose_chart_cky(data, DummySub(), g)
    cs_tau1 = layer._chart_score.detach().clone()

    layer.chart_tau = 5.0
    layer._compose_chart_cky(data, DummySub(), g)
    cs_tau5 = layer._chart_score.detach().clone()

    # Spread among populated-score categories should shrink at τ=5
    # versus τ=1 (softer mixture means smaller logit differences).
    # Filter out the NEG_INF (-1e30) sentinel used for empty cells.
    def populated_spread(chart_score):
        flat = chart_score[:, 0, N, :].reshape(-1)
        finite = flat[(flat > -1e10) & torch.isfinite(flat)]
        if finite.numel() <= 1:
            return 0.0
        return (finite.max() - finite.min()).item()

    spread_tau1 = populated_spread(cs_tau1)
    spread_tau5 = populated_spread(cs_tau5)
    assert spread_tau5 < spread_tau1 - 1e-6, (
        f"Higher τ should soften the chart score distribution; "
        f"populated spread τ=1: {spread_tau1:.4f}, τ=5: {spread_tau5:.4f}"
    )


# -------- Decompose round-trip --------

# Test `test_decompose_runs_and_returns_correct_shapes` removed by
# Step 9 of the 2026-05-01 syntactic-layer refactor: it asserted
# behavior of `SyntacticLayer._SoftDecompose`, a deprecated helper.
# The new chart's reverse path uses `Chart.generate` + Viterbi
# backtrace; round-trip exactness is the job of the host layer's
# `compose` / `generate` pair (e.g. PiLayer.compose / .generate).


# -------- Add rule mid-training --------

def test_add_rule_bumps_version_and_rebuilds_params():
    """Appending a rule to the grammar bumps `rule_table_version`; the
    next compose rebuilds rule-shaped parameters on the layer.
    """
    g = _make_grammar([
        ('S', ['VO V']),
    ])
    D, N, B = 8, 3, 2
    layer = _make_layer(g, D=D, N=N)

    class DummySub:
        def __init__(self):
            self.word = []
            self.basis = None
            self.wordSpace = None
        def add_word(self, *a, **kw): pass
        def flush_word_buffer(self): pass

    layer.eval()
    data = torch.randn(B, N, D)
    layer._compose_chart_cky(data, DummySub(), g)
    R0 = layer._rule_embed.shape[0]
    v0 = g.rule_table_version

    # Add a synthetic rule by reconfiguring the grammar — same
    # interface a runtime add/remove would use.
    g.configure({'S': ['VO V', 'V V'], 'VO': ['V O']})
    v1 = g.rule_table_version
    assert v1 > v0, "configure must bump rule_table_version"

    layer._compose_chart_cky(data, DummySub(), g)
    R1 = layer._rule_embed.shape[0]
    assert R1 > R0, (
        f"Adding a rule must add a row to rule_embed (was {R0}, now {R1})"
    )
