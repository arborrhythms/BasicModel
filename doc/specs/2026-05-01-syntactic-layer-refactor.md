# Refactor Spec — Per-Space SyntacticLayer + Chart-on-WordSpace

**Date:** 2026-05-01
**Author:** handed off from the floating-blossom / soft-chart sequence
**Status:** ready to implement
**Related docs:** [doc/research/three-surfaces.md](../research/three-surfaces.md),
[doc/research/2019-kim-compound-pcfg.md](../research/2019-kim-compound-pcfg.md),
[doc/research/2017-jang-gumbel-softmax.md](../research/2017-jang-gumbel-softmax.md)

---

## 1. Why

Today's `SyntacticLayer` violates the Layer contract. It coordinates across
spaces, owns the entire CKY chart, holds chart-wide parameters, and supplies
20+ rule-execution methods that bypass the host space's parametrized
PiLayer/SigmaLayer/NotLayer instances. Result: the chart's grammar choice
doesn't drive the host's parametrized folds (the "passenger / driver"
failure mode reproduced on XOR_grammar repeatedly).

Refactor moves chart and inter-space coordination to **WordSpace**; each
**Space** (PerceptualSpace, ConceptualSpace, SymbolicSpace) owns a
**per-space SyntacticLayer** that holds the parametrized layers for that
tier's rules and dispatches based on a rule choice the chart wrote into the
WordSpace state.

WordSpace is already passed forward as a subspace through the spaces'
invocation chain, so the chart's per-tier rule selection is naturally
available to each space's syntactic layer via the subspace it receives.

---

## 2. Component responsibilities (post-refactor)

### `Chart` (new helper class, owned by `WordSpace`)

A self-contained `nn.Module`. Owns everything chart-specific:

- Parameters: `_rule_bias` `[R]`, `_rule_embed` `[R, D_rule]`,
  `_marker_bias` `[R, 2]`, `_compat_score_mod`, `_unary_compat_mod`,
  `_lex_cat_scorer`, the rule-prediction MLP (`input_proj`,
  `derivation_layer`, `rule_head`, `depth_embed`).
- Per-call state: `chart_score [B, N+1, N+1, C]`, `chart_vec`,
  `outside_score`, `outside_vec`, `derivation_trace`.
- Category machinery: `_category_names`, `_category_index`,
  `_ensure_category_table`.
- Soft-chart configuration: `chart_tau`, `w_max`, `unary_max_depth`,
  `D_rule`.
- Methods:
  - `compose(input, grammar) -> per-(tier, step) rule selections`.
    Runs the inside pass, then writes rule choices for each
    `(tier, step)` into `wordSpace.current_rules`.
  - `generate(target, grammar) -> per-(tier, step) generate rules`.
    Runs the outside pass / Viterbi backtrace, writes choices into
    `wordSpace.generate_rules`.
  - `_compose_chart_cky`, `_compose_chart_outside`, `_viterbi_extract`,
    `_apply_rule_forward`, `_signal_sentence_completed_chart`,
    `_ensure_soft_chart_built` — all moved over verbatim from
    `SyntacticLayer`, with `self.grammar` reads becoming `grammar`
    parameter reads (or a stored ref to the grammar passed in
    `compose`).

The Chart's per-cell rule dispatch consults `wordSpace.host_layer(tier,
rule_name)` to find the parametrized layer to fire. That is the unified
forward path: chart's grammar choice fires the host's owned layer.

### `WordSpace` (cross-space coordinator)

Already passed forward as a subspace through every space's invocation
chain. Now also owns the chart and the inter-space rule selection state.

- Owns: `chart: Chart`, `grammar: Grammar` reference,
  `current_rules: dict[tier, list[rule_id]]`,
  `generate_rules: dict[tier, list[rule_id]]`,
  `_host_layer_registry: dict[(tier, rule_name), GrammarLayer]`.
- Methods:
  - `compose(input)`: thin wrapper that calls `self.chart.compose(input,
    self.grammar)`. Idempotent within a forward pass.
  - `generate(target)`: thin wrapper that calls `self.chart.generate(...)`.
  - `register_host_layer(tier, rule_name, layer)`: each space's
    syntacticLayer calls this at construction to register its
    parametrized layers. Chart dispatch reads the registry.
  - `host_layer(tier, rule_name) -> GrammarLayer | None`: lookup.
    Returns None when no parametrized layer is registered for that
    (tier, rule_name) — chart treats this as a fallback to `Ops.*`
    (for non-host-space rules: lift, lower, swap, equals, part, etc.,
    see Q2 / §6 below).

### Per-space `SyntacticLayer` (per-space dispatcher)

Lives on PerceptualSpace, ConceptualSpace, and SymbolicSpace only. Holds
the parametrized layers for that tier's rules.

- Construction: `SyntacticLayer(tier, host_layers={rule_name: layer, ...})`.
  Iterates `host_layers` and calls `wordSpace.register_host_layer` for each.
  Stores layers in an `nn.ModuleList`.
  - `PerceptualSpace.syntacticLayer`: holds `SigmaLayer` for `union`.
  - `ConceptualSpace.syntacticLayer`: holds `PiLayer` for `intersection`.
  - `SymbolicSpace.syntacticLayer`: holds `SigmaLayer` for `union`,
    `NotLayer` for `not`, `ContiguousLayer` for `Contiguous`, plus any
    other per-space layers the configured grammar requires.
- Per-tier rule cursor: a counter advanced each time `forward()` fires.
  Reset at the start of each `WordSpace.compose()`. (Q4 / Q7 below.)
- API:
  - `forward(x)`: reads next rule choice from `wordSpace.current_rules[tier]`,
    advances cursor, dispatches to the matching parametrized layer's
    `compose(left, right)` (binary) or `forward(x)` (unary).
  - `reverse(y)`: same pattern with `wordSpace.generate_rules[tier]`,
    dispatches `.generate(parent)` (renamed from `decompose`).
  - When `softChartCompose=false`: `wordSpace.current_rules` is empty;
    syntacticLayer dispatches to a default layer (a per-space
    "default rule" attribute, e.g. `default='intersection'` for
    ConceptualSpace) — no behavior change for legacy configs (Q6 below).

### Each Space's `forward` / `reverse`

Becomes uniform:

```python
class ConceptualSpace(Space):
    def forward(self, subspace):  # subspace carries wordSpace forward
        # ... existing pre-fold work (codebook, masking, etc.) ...
        out = self.syntacticLayer.forward(act)
        # ... existing post-fold work ...
        return out

    def reverse(self, subspace):
        # mirror with self.syntacticLayer.reverse(...)
```

No more direct `self.forwardPi(x)` / `self.forwardSigma(x)` /
`self.propositional_negation(x)` / `self._contiguous_layer(x)` calls. The
parametrized layers are owned by `self.syntacticLayer.layers`.

WordSpace is touched **only from the per-space syntacticLayer**, which gets
its handle via the subspace it received. This satisfies Q1.

### `BasicModel.forward` / `reverse`

```python
def forward(self, x):
    inp_subspace  = self.inputSpace.forward(x)         # passes wordSpace forward on subspace
    self.wordSpace.compose(inp_subspace.materialize()) # NEW: chart parse, populates current_rules
    perc_subspace = self.perceptualSpace.forward(inp_subspace)
    conc_subspace = self.conceptualSpace.forward(perc_subspace)
    sym_subspace  = self.symbolicSpace.forward(conc_subspace)
    return self.outputSpace.forward(sym_subspace)
```

`reverse` is the mirror, with `wordSpace.generate(...)` running before
the spaces' reverse calls.

---

## 3. Migration inventory

### Move from `SyntacticLayer` → `Chart` (owned by `WordSpace`):

| Item | Notes |
|---|---|
| `_rule_bias`, `_rule_embed`, `_marker_bias` | Registered on Chart |
| `_compat_score_mod`, `_unary_compat_mod`, `_lex_cat_scorer` | Sub-modules of Chart |
| `_compose_chart_cky`, `_compose_chart_outside`, `_viterbi_extract` | Methods of Chart |
| `_apply_rule_forward`, `_signal_sentence_completed_chart` | Methods of Chart |
| `_ensure_soft_chart_built` | Method of Chart |
| `_chart_score`, `_chart_vec`, `_outside_score`, `_outside_vec` | Per-call state on Chart |
| `_derivation_trace` | Per-call state on Chart |
| `_category_names`, `_category_index`, `_ensure_category_table` | On Chart |
| `chart_tau`, `w_max`, `unary_max_depth`, `D_rule` | Read from XML in Chart's `__init__` |
| Rule prediction MLP (`input_proj`, `derivation_layer`, `rule_head`, `depth_embed`, `pair_scorer`) | On Chart |
| `forward()` (rule prediction) | Renamed `Chart.predict_rules()` |

### Stay on per-space `SyntacticLayer`:

| Item | Per-space layer ownership |
|---|---|
| Parametrized layers (PiLayer / SigmaLayer / NotLayer / ContiguousLayer / NonLayer / IntersectionLayer / UnionLayer / and any GrammarLayer subclass for an op the grammar uses) | Each space holds the layers for the rules its tier hosts (see Q2). |

### Remove entirely:

- `SyntacticLayer.intersectionForward / Reverse`,
  `unionForward / Reverse`, `notForward / Reverse`,
  `nonForward`, `ContiguousForward / Reverse`,
  `liftForward / Reverse`, `lowerForward / Reverse`,
  `equalsForward`, `partForward`, `trueForward`, `falseForward`,
  `swapForward`, `queryForward`, `whatForward`, `whereForward`,
  `whenForward`, `absorbForward`, `conjunctionForward`,
  `disjunctionForward` — all replaced by the per-space `SyntacticLayer`
  dispatching to GrammarLayer subclasses (see Q2 / §6).
- `_RULE_METHODS` dispatch table.
- `_GrammarOpFacade._registry` and the `_*OpFacade` classes (these were
  workaround facades for the SyntacticLayer-string-dispatch path; with
  unified host-space dispatch they're obsolete).
- `_SoftCompose`, `_SoftDecompose` (already deprecated).
- `Basis.conjunction`, `Basis.disjunction`, `Basis.negation`,
  `Basis.non` — only the *Reverse variants stay (codebook factoring on
  the reverse direction still uses the host's codebook, see §6).
- The legacy non-chart compose path (`_compose_vector_chart`,
  `_compose_activation`, `_compose_to_target`, the Phase-2 cascade in
  `_compose_vector`). Removing the legacy path was confirmed in Q6.

---

## 4. New / modified APIs

### `Chart` (new class, in `bin/Language.py`):

```python
class Chart(nn.Module):
    """Soft-superposition CKY chart parser. Owned by WordSpace.

    Holds chart parameters, runs inside / outside passes, dispatches
    per-cell rule applications through wordSpace.host_layer(tier, rule)
    to fire the host space's parametrized folds directly.
    """
    def __init__(self, grammar, *, chart_tau=1.0, w_max=8,
                 unary_max_depth=2, D_rule=32, ...):
        ...
        # registers _rule_bias, _rule_embed, _marker_bias,
        # _compat_score_mod, _unary_compat_mod, _lex_cat_scorer,
        # the rule-prediction MLP, etc.

    def compose(self, input_vectors, word_space) -> dict[str, list[int]]:
        """Run the inside pass. Returns per-(tier, step) rule
        selections. Side effects: populates word_space.current_rules
        and self._chart_score / self._chart_vec.
        """

    def generate(self, target_vectors, word_space) -> dict[str, list[int]]:
        """Run the outside pass + Viterbi extract. Returns
        per-(tier, step) generate rules. Side effects: populates
        word_space.generate_rules and self._outside_score /
        self._outside_vec.
        """
```

### `WordSpace` (additions):

```python
class WordSpace(SubSpace):
    chart: Chart                                                       # NEW
    current_rules: dict[str, list[int]]                                # NEW
    generate_rules: dict[str, list[int]]                               # NEW
    _host_layer_registry: dict[tuple[str, str], GrammarLayer]          # NEW

    def compose(self, input_vectors) -> None:
        self.current_rules = self.chart.compose(input_vectors, self)

    def generate(self, target_vectors) -> None:
        self.generate_rules = self.chart.generate(target_vectors, self)

    def register_host_layer(self, tier: str, rule_name: str,
                            layer: GrammarLayer) -> None:
        self._host_layer_registry[(tier, rule_name)] = layer

    def host_layer(self, tier: str, rule_name: str) -> GrammarLayer | None:
        return self._host_layer_registry.get((tier, rule_name))
```

### Per-space `SyntacticLayer`:

```python
class SyntacticLayer(Layer):
    """Per-space dispatcher.

    Construction:
        SyntacticLayer(tier='C', word_space=word_space,
                       host_layers={'intersection': pi_layer})

    Each entry in host_layers is registered with word_space at
    construction. The space's forward() and reverse() delegate here;
    forward() reads word_space.current_rules[tier], advances a per-
    tier cursor, and dispatches to the appropriate layer's compose /
    forward.
    """
    tier: str
    layers: nn.ModuleList
    default_rule: str | None        # for legacy / no-chart fallback

    def __init__(self, tier, word_space, host_layers, default_rule=None):
        super().__init__(0, 0)
        self.tier = tier
        self.layers = nn.ModuleList(list(host_layers.values()))
        self._by_name = dict(host_layers)
        self.default_rule = default_rule
        for rule_name, layer in host_layers.items():
            word_space.register_host_layer(tier, rule_name, layer)
        self._cursor = 0
        self._word_space = word_space  # weak ref-style; subspace passes it forward

    def forward(self, x):
        rule_name = self._next_rule(direction='compose')
        layer = self._by_name.get(rule_name)
        if layer is None:
            return x  # passthrough on missing rule (shouldn't happen)
        if layer.arity == 2:
            # Pair adjacent positions (or use chart-driven splits via
            # word_space.current_rules; see Q4 / §6).
            return layer.compose(x[..., 0::2, :], x[..., 1::2, :])
        return layer.forward(x)

    def reverse(self, y):
        # mirror, using word_space.generate_rules and layer.generate(parent)

    def _next_rule(self, *, direction):
        rules = (self._word_space.current_rules if direction == 'compose'
                 else self._word_space.generate_rules)
        per_tier = rules.get(self.tier, [])
        if self._cursor < len(per_tier):
            choice = per_tier[self._cursor]
            self._cursor += 1
            return choice
        return self.default_rule  # fallback (legacy or no-chart)
```

### `Space.forward / reverse` (updated for each of P/C/S):

```python
def forward(self, subspace):
    # ... pre-fold work (codebook quantize, masking, etc.) stays ...
    out = self.syntacticLayer.forward(act)
    # ... post-fold work stays ...
    return out
```

### `BasicModel.forward / reverse`:

```python
def forward(self, x):
    inp_sub  = self.inputSpace.forward(x)
    self.wordSpace.compose(inp_sub.materialize())  # NEW
    perc_sub = self.perceptualSpace.forward(inp_sub)
    conc_sub = self.conceptualSpace.forward(perc_sub)
    sym_sub  = self.symbolicSpace.forward(conc_sub)
    return self.outputSpace.forward(sym_sub)
```

---

## 5. Implementation order

1. **Create the `Chart` class** in [bin/Language.py](../../bin/Language.py).
   Move chart-specific parameters, methods, and per-call state from the
   old `SyntacticLayer` into it. Verify it works in isolation against
   the existing tests for the chart (`test_soft_chart.py`).

2. **Add chart-related state and methods to `WordSpace`**:
   `chart` instance, `current_rules`, `generate_rules`,
   `_host_layer_registry`, `compose / generate / register_host_layer /
   host_layer`. WordSpace is already passed forward via subspace, so no
   plumbing change needed there.

3. **Define the per-space `SyntacticLayer` class** (replacing the old
   global one). Per-tier cursor, host-layer registration at construction,
   forward / reverse dispatch through `wordSpace.current_rules` /
   `generate_rules`, default-rule fallback.

4. **Update each Space's `__init__`** (PerceptualSpace, ConceptualSpace,
   SymbolicSpace): instantiate the per-space SyntacticLayer with the
   appropriate `host_layers` dict, register it as `self.syntacticLayer`.
   The existing PiLayer/SigmaLayer/NotLayer/ContiguousLayer instances
   are owned by the syntacticLayer's `layers` ModuleList.

5. **Update each Space's `forward / reverse`** to delegate to
   `self.syntacticLayer.forward(act)` / `.reverse(y)`. Remove the
   manual `p_uni`, `p_neg`, `p_con` gating on SymbolicSpace.forward
   (the chart's rule selection subsumes it). Remove the
   `forwardPi`, `forwardSigma` aliases.

6. **Update `BasicModel.forward / reverse`** to call
   `self.wordSpace.compose(...)` / `.generate(...)` at the appropriate
   points in the data flow.

7. **Wire the chart's `_apply_rule_forward` to host-space dispatch.**
   Inside `Chart`, when a binary rule fires, call
   `wordSpace.host_layer(tier, rule_name).compose(left, right)`. When a
   unary rule fires, call `.forward(x)`. For the outside pass and
   Viterbi extract, call `.generate(parent)`. For non-host-space rules
   (lift, lower, swap, equals, part, etc.) — see §6.

8. **Make every grammar op have a GrammarLayer subclass** (Q2). The 14
   `_GrammarOpFacade` subclasses I added earlier (LiftLayer, LowerLayer,
   ConjunctionLayer, etc.) become real GrammarLayer subclasses with
   actual parametrized layers (or stateless for ops like NotLayer that
   are parameter-free). The `_GrammarOpFacade._registry` mechanism is
   removed; SymbolicSpace's syntacticLayer registers all the
   symbol-tier ops it owns.

9. **Remove obsolete code**: `SyntacticLayer.*Forward / *Reverse` methods,
   `_RULE_METHODS`, `_GrammarOpFacade`, `_SoftCompose`, `_SoftDecompose`,
   `Basis.conjunction / disjunction / negation / non` (forward direction
   only — *Reverse stays for codebook factoring), and the entire legacy
   non-chart compose path (Q6).

10. **Update tests.** Per Q8: more XML-driven model tests, fewer per-unit
    tests on synthetic SyntacticLayer instances.
    - Tests that constructed `SyntacticLayer(nInput, nOutput, rules,
      grammar=g)` directly: rewrite to construct via a small fixture
      XML or by building a per-space SyntacticLayer with explicit
      `host_layers`. The existing test grammars in `test_grammar_split`
      etc. become XML-driven.
    - Tests that read `_chart_score` / `_chart_vec` from the
      SyntacticLayer: read from `wordSpace.chart` instead.
    - Tests that asserted gradient flow into `_rule_bias`: same, read
      from `wordSpace.chart._rule_bias`.

11. **Re-run XOR_grammar.** Acceptance test: chart's grammar choice
    drives the host's parametrized fold. Confirm via gradient
    inspection that `wordSpace.conceptualSpace.syntacticLayer.layers[
    pi_index].W.grad` accumulates only from chart-fired
    intersection-rule firings, with no parallel always-on Pi path.

---

## 6. Open-question resolutions (from Q&A)

**Q1 (WordSpace touched only via subspace):** Confirmed. Per-space
syntacticLayer holds a reference to the wordSpace it was constructed
with; the subspace also carries it forward as the standard SubSpace
contract. No global wordSpace lookup needed.

**Q2 (every rule has a GrammarLayer):** Confirmed. The 14 op-facades I
added in the prior session become real GrammarLayer subclasses owned by
the appropriate per-space syntacticLayer. Each tier's syntacticLayer
holds the layers for the rules its tier hosts. Common forward/reverse
work lives on the GrammarLayer base class.

**Q3 (SyntacticLayer rename):** Confirmed easy choice. Keep the name
`SyntacticLayer`; the old global instance is replaced wholesale during
the refactor. No transition coexistence needed.

**Q4 (multi-step within tier — how does the space know which step?):**
Confirmed: the wordSpace traversed through the invocation chain knows.
Specifically, the per-space syntacticLayer holds a per-tier cursor that
advances on each `forward()` call within a single `wordSpace.compose()`
context, and resets at the start of each new compose. The wordSpace
holds the per-tier rule list; the syntacticLayer pops one rule per step
from that list.

**Q5 (multiple rule firings per space, but only one of each):**
Confirmed. A space may fire multiple rules sequentially within a
forward pass (one per layer-step), but each rule fires at most once per
forward. Enforced at the chart level: `wordSpace.current_rules[tier]`
is a list, and the chart guarantees no duplicates within one list.
This is your "one application of a given rule per layer" idea
operationalized.

**Q6 (legacy path removal):** Confirmed. The non-chart `_compose_vector`
Phase-2 cascade and the legacy `chartCompose=true` (greedy) path can
both be deleted. `softChartCompose=true` becomes the only active path.
The `chartCompose` and the cascade XML configs in legacy XMLs become
no-ops or hard errors (decision: hard error — fail-fast, force config
update).

**Q7 (symbol-tier rule ordering — chart-derived layer depth):**
Confirmed reasonable per quick lit check. The chart's binary
inside-pass naturally produces layer depth 0 at width=2, depth 1 at
width=3, ..., up to max-depth at width=N. The order of
`current_rules['S']` is the bottom-up sequence of rules fired across
those depths. This is consistent with bottom-up CKY conventions (Hopcroft
& Ullman; Eisner's chart-parser literature). Recommend implementing
this ordering and revisiting against literature if XOR_grammar reveals
a depth-shape mismatch.

**Q8 (test back-compat):** Per-unit `SyntacticLayer` tests are reduced
or rewritten as XML-driven model tests. Fixture XMLs for the test
grammars live in [test/fixtures/](../../test/fixtures/) (new directory).
Tests load via a small helper that constructs a stripped-down model
with just the spaces under test.

**Q9 (manual `p_*` gates removed from SymbolicSpace.forward):** Confirmed.

**Q10 (Chart helper class):** Confirmed. The `Chart` class is a real
`nn.Module` owned by WordSpace. Keeps WordSpace's surface compact
(WordSpace already owns TruthLayer, DiscourseSpace, etc.).

---

## 7. Acceptance criteria

1. **`make test` green** with the refactored architecture.
2. **XOR_grammar with the new dispatch**:
   `wordSpace.conceptualSpace.syntacticLayer.layers[<pi index>].W.grad`
   accumulates from chart-fired `intersection` rule firings, **not from
   a parallel always-on Pi path**. Verified by instrumented gradient
   inspection. The chart's grammar choice IS the host's parametrized
   fold.
3. **MM_grammar** continues to learn XOR (same task accuracy or
   better) with the chart's grammar choice now structurally on the loss
   path.
4. **The string `intersectionForward` does not appear anywhere in
   `bin/`** (and similar for the other removed methods).
5. **The legacy non-chart compose path is gone.** No `chartCompose=false`
   handling. `softChartCompose=true` is the only active path; tests for
   the legacy path are removed or migrated.

---

## 8. Estimated diff size

- New `Chart` class: ~600 lines (cut-and-paste from `SyntacticLayer`).
- New per-space `SyntacticLayer` class: ~200 lines.
- WordSpace additions: ~80 lines.
- Each Space's `forward / reverse` rewire: ~20 lines × 3 spaces = 60 lines.
- Removal of obsolete methods on `SyntacticLayer`: ~−800 lines.
- Test rewrites: ~300 lines.
- Net: roughly +500 lines of new code, ~−800 lines removed, +300 in
  tests. The codebase shrinks net.

Implementation should fit in a focused 2-3 day session for someone
familiar with the existing chart code.

---

## 9. References

- [doc/research/three-surfaces.md](../research/three-surfaces.md): the
  diagnostic that motivated this refactor (Surface 1 + 2 + 3
  unification).
- [doc/research/2019-kim-compound-pcfg.md](../research/2019-kim-compound-pcfg.md):
  fixed-rule-semantics architectural pattern.
- [doc/research/2017-jang-gumbel-softmax.md](../research/2017-jang-gumbel-softmax.md):
  STE / Gumbel-Softmax for hard-forward / soft-backward.
- [doc/research/2022-yang-rank-space-pcfg.md](../research/2022-yang-rank-space-pcfg.md):
  scaling structured inference for larger grammars (post-refactor).

---

## 10. Implementation flags + resolutions (post-spec Q&A)

After the spec was drafted, five secondary questions were raised during
review. Resolutions captured here so the implementer doesn't re-discover
them mid-work.

### Q10.1 — Per-tier cursor reset semantics during `conceptualOrder > 1`

**Question raised:** When `conceptualOrder > 1`, the conceptual and
symbolic spaces are traversed multiple times within a single forward
pass (one traversal per conceptual order level). When does the
per-tier cursor on `SyntacticLayer` reset?

**Resolution:** The cursor advances naturally through the multi-traversal.
Each traversal consumes one rule from the per-tier list; the next
traversal consumes the next rule. The cursor resets only at the start
of each new `wordSpace.compose()` call (one per BasicModel.forward). So
for `conceptualOrder=2` with a list of 2 rules in
`current_rules['C']`, traversal 0 consumes rule[0], traversal 1
consumes rule[1]. With `conceptualOrder=3`, the list should have 3
entries.

The chart populates `current_rules[tier]` with one rule per
`(tier, conceptual_order_step)` it expects to fire. The number of
entries equals the depth of the per-tier composition path. Implementer
needs to ensure the chart's rule-emission count matches the host
space's traversal count for the configured `conceptualOrder`.

### Q10.2 — Where does `wordSpace.compose()` read its input?

**Question raised:** Should `wordSpace.compose()` parse the InputSpace
output (raw embedding) or the PerceptualSpace output (post-perceptual
fold)?

**Resolution:** Neither — WordSpace doesn't take a separate input
parameter. WordSpace lives on a subspace that contains the data it
will operate on (and the data it will write rule choices into). The
chart reads its input from `subspace.materialize()` (or whatever the
contract for "the data carried by this subspace at this point in the
pipeline" is). The same subspace traversal that delivers wordSpace
forward through the spaces also delivers the data the chart parses.

Implementation: `wordSpace.compose()` is called BEFORE the spaces' folds
fire, with the subspace just emitted from InputSpace. The chart reads
from that subspace's data tensor. After the chart writes rule choices,
each space's syntacticLayer reads them as the spaces' folds fire on the
forward pass.

### Q10.3 — `Chart` class's grammar reference

**Question raised:** Does `Chart.compose()` take grammar as a per-call
parameter, or does Chart hold a reference at construction?

**Resolution:** Use `TheGrammar` (the global module-level singleton)
everywhere. It is stateless: rule catalog, packed-table machinery,
category index — all derived from the loaded XML. State that varied
per-call (which I had been considering pushing onto Grammar) lives
exclusively on WordSpace. The Chart reads from `TheGrammar` directly.

This means:
- No `grammar` parameter on `Chart.compose()` / `Chart.generate()`.
- No grammar reference stored on Chart.
- Tests can swap TheGrammar's configuration, run the chart, and the
  chart immediately picks up the new configuration via the singleton.

### Q10.4 — Where do non-host-space rules (lift, lower, swap, etc.) live?

**Question raised:** Every rule must have a GrammarLayer (Q2). But
S-tier symbolic ops like lift, lower, swap, equals, part, true, false,
conjunction, disjunction, query, what, where, when, absorb are rare
enough that registering all of them on every SymbolicSpace would be
wasteful.

**Resolution:** Lazy registration. SymbolicSpace's syntacticLayer reads
the configured grammar at construction and registers ONLY the layers
for rules the grammar actually contains. Each rule's GrammarLayer
class is looked up by `rule_name` against the project's GrammarLayer
class registry (the existing `_GrammarOpFacade._registry` retired but
its key→class mapping kept as a small lookup table during construction
of each space's syntacticLayer).

Implementation:
```python
# At SymbolicSpace.__init__:
host_layers = {}
for rule in TheGrammar.rules:
    if rule.tier != 'S' or rule.method_name is None:
        continue
    cls = _GRAMMAR_LAYER_CLASSES.get(rule.method_name)  # rule_name -> class
    if cls is None:
        continue  # rule has no parametrized layer; chart will fall back
    if rule.method_name not in host_layers:
        host_layers[rule.method_name] = cls(...)        # construct on demand
self.syntacticLayer = SyntacticLayer(
    tier='S', word_space=word_space, host_layers=host_layers,
    default_rule='intersection')
```

Same pattern for ConceptualSpace and PerceptualSpace, each filtering on
its own tier.

### Q10.5 — Viterbi vs soft superposition: when does each fire?

**Question raised:** Should Viterbi-hardened rule selection happen ONLY
after the soft superposition (as a separate hardening step), or CAN it
happen IN PLACE OF the soft superposition during the inside / outside
passes (e.g., at testing time)?

**Resolution:** Soft for compose / generate during training; Viterbi
in place of soft superposition during testing.

Concretely:
- **Training (`self.training == True`)**: soft inside / outside passes
  on the way up and way down. Logsumexp / softmax-weighted summing.
  Gradient flows broadly across alternatives. STE / Gumbel optional
  for sharpening.
- **Testing (`self.training == False`)**: Viterbi argmax IN PLACE OF
  the soft summing. The inside pass picks argmax(rule, k) per cell at
  every width step (one-hot, no mixing); the outside pass picks
  argmax(parent rule, k) per child cell. Both directions traverse the
  single committed tree.

This means the chart has TWO inside-pass implementations and TWO
outside-pass implementations:
- `compose_chart_soft` / `compose_chart_outside_soft` (training)
- `compose_chart_viterbi` / `compose_chart_outside_viterbi` (testing)

Or one implementation with a `mode` flag that switches between
softmax-mixing and argmax-selecting at the per-cell scatter step.

The Viterbi pass at test time is structurally cleaner than the current
"soft + post-hoc Viterbi extraction" pattern: at test time, every cell
holds the argmax-rule's vector (no softmax blending), the chart's
output is a single concrete derivation, and reverse traversal is a
genuine inverse of forward.

This resolves a lingering ambiguity: the previous design ran soft
superposition at both train and eval, with Viterbi only as a side-
effect for diagnostic purposes. The new design runs Viterbi as the
actual eval-mode forward, which is what the user has been asking for
(per-sentence committed grammar with hard rule choices at test).

Implementation: `Chart.compose()` and `Chart.generate()` branch on
`self.training` (or a separate `mode` parameter) at the per-cell scatter
step. Soft mode uses the existing logsumexp/softmax-sum machinery; hard
mode uses argmax+gather. Backward through the hard mode uses STE
(forward = argmax, backward = softmax) so gradient still flows during
mixed-mode evaluation if needed.

This is a meaningful expansion of the chart's implementation
responsibilities — implementer should plan ~+150 lines for the hard-mode
inside/outside passes, separate or interleaved with the soft-mode logic.
