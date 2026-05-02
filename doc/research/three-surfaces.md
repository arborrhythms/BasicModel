# Three parallel surfaces of grammar operations

A signal passing through `PerceptualSpace -> ConceptualSpace -> SymbolicSpace`
encounters **three distinct surfaces** that each implement (some of) the
grammar's operators. Right now they coexist; tasks can ride any of the three,
which is why the soft chart can be a passenger while the model still solves
its task.

This document maps which surface is invoked where, what direction (forward /
reverse) is used, and how `compose` / `decompose` are called.

## Surface 1: Naked operators (the always-on pipeline)

The fixed pipeline that runs on **every** forward pass regardless of grammar
flags. Owns its own parameters; trains end-to-end via the task loss.

| Space | Owns | forward call | reverse call | Other ops |
|---|---|---|---|---|
| `PerceptualSpace` | `SigmaLayer` (P -> sub-percept, [Spaces.py:5418-5422](../../bin/Spaces.py:5418)) | `forwardSigma(x)` | `reverseSigma(y)` (when `invertible`) | VQ-VAE codebook quantize via `self.quantize`; `l1_proximal`; `valid_mask` zeroing |
| `ConceptualSpace` | `PiLayer` (concept fold, [Spaces.py:6558-6591](../../bin/Spaces.py:6558)) | `forwardPi(x)` ([Spaces.py:6630](../../bin/Spaces.py:6630)) | `reversePi(y)` | Optional attention head; `_active` masking |
| `SymbolicSpace` | `SigmaLayer` (C -> S fold, [Spaces.py:6792-6809](../../bin/Spaces.py:6792)) | `forwardSigma(x)` ([Spaces.py:7503](../../bin/Spaces.py:7503)) | `reverseSigma(y)` (when `invertible`) | `l1_proximal`; codebook quantize via VQ-VAE; valid_mask zeroing |

**forward / reverse:** Both used. Forward is the always-on data-flow direction;
reverse is invoked by the reconstruction loss path (`runBatch.reconstruction`)
and by the AR rolling-cursor path.

**compose / decompose:** *Not* called at this surface. PiLayer and SigmaLayer
*do* expose `compose(left, right)` / `decompose(parent)` (binary tensor ops at
[Layers.py:1573-1610](../../bin/Layers.py:1573)) but the spaces' forward pipelines
use only the unary `forward` / `reverse`. The binary `compose` is reserved for
chart drivers; only Surface 2 wraps and re-exposes them.

## Surface 2: GrammarLayer wrappers

Each grammar operator wrapped as an `nn.Module` subclass of `GrammarLayer`
([Layers.py:1669-1979](../../bin/Layers.py:1669)). Two roles:
- Provide a uniform `compose / decompose` contract for binary operators (the
  CKY chart driver was supposed to use this).
- Provide a uniform `forward / reverse` for unary operators (some are used
  directly inline in space-forward paths).

| Class | rule_name | arity | invertible | lossy | Wraps | Where instantiated | Where called |
|---|---|---|---|---|---|---|---|
| `NotLayer` | `not` | 1 | yes | no | (parameter-free bivector swap) | `SymbolicSpace.__init__` as `self.propositional_negation` ([Spaces.py:6820](../../bin/Spaces.py:6820)) | `SymbolicSpace.forward` mixes `act = p_neg * not(act) + (1-p_neg) * act` ([Spaces.py:7530-7533](../../bin/Spaces.py:7530)) — gated by `rule_probability("not(S)")` |
| `NonLayer` | `non` | 1 | no | yes | (parameter-free `1 - |clamp(x,-1,1)|`) | Not currently constructed in any space | not invoked |
| `IntersectionLayer` | `intersection` | 2 | yes | no | A `PiLayer` instance | not constructed by any space | dormant; `compose` / `decompose` defined but no external caller |
| `UnionLayer` | `union` | 2 | yes | no | A `SigmaLayer` instance | not constructed by any space | dormant; `compose` / `decompose` defined but no external caller |
| `ContiguousLayer` | `Contiguous` | 1 | no | yes | (parameter-free per-axis `amax`) | `SymbolicSpace.__init__` as `self._contiguous_layer` ([Spaces.py:6926](../../bin/Spaces.py:6926)) | `SymbolicSpace.forward` mixes `act = p_con * contiguous(act) + (1-p_con) * act` ([Spaces.py:7517-7522](../../bin/Spaces.py:7517)) — gated by `rule_probability("Contiguous(S)")` |

**forward / reverse:** Used by NotLayer and ContiguousLayer in `SymbolicSpace.forward`
(unary, gated by rule probability). The `IntersectionLayer` / `UnionLayer`
`forward` exists but isn't invoked anywhere.

**compose / decompose:**
- Defined on every `GrammarLayer` subclass per the contract at
  [Layers.py:1722-1751](../../bin/Layers.py:1722).
- For arity-2 wrappers (Intersection/Union): delegate to inner `pi.compose` /
  `pi.decompose` and `sigma.compose` / `sigma.decompose`.
- **No external code calls these.** Chart and space paths bypass them.
- Only call sites in the entire codebase:
  - `IntersectionLayer.compose` -> `self.pi.compose` ([Layers.py:1884](../../bin/Layers.py:1884))
  - `IntersectionLayer.decompose` -> `self.pi.decompose` ([Layers.py:1888](../../bin/Layers.py:1888))
  - `UnionLayer.compose` -> `self.sigma.compose` ([Layers.py:1922](../../bin/Layers.py:1922))
  - `UnionLayer.decompose` -> `self.sigma.decompose` ([Layers.py:1926](../../bin/Layers.py:1926))

These are wired but dead code paths from the chart's perspective. They were
clearly built in anticipation of the chart driving them, but the chart never
ended up calling them.

## Surface 3: SyntacticLayer (the chart and `_RULE_METHODS` dispatch)

A single nn.Module owned by `WordSpace.syntacticLayer`. Holds rule-prediction
MLP, depth embeddings, soft-chart parameters, and the per-rule **method
table** mapping `rule_name` -> `*Forward` / `*Reverse` method on the layer
itself.

`_RULE_METHODS` ([Language.py:758-786](../../bin/Language.py:758)) maps each rule
name to a tuple `(forwardName, reverseName | None, is_binary)`. Methods like
`notForward`, `intersectionForward`, `unionForward`, `liftForward`,
`lowerForward`, etc. are defined on `SyntacticLayer` and reach for either
`Basis.conjunction` / `Basis.disjunction` / `Basis.negation` (when subspace
provides a Basis) or fall back to plain tensor primitives (`torch.min`,
`torch.max`, `-x`, `Ops.lift`/`Ops.lower`).

| Method | rule_name | arity | invertible | Implementation | Called where |
|---|---|---|---|---|---|
| `notForward` / `notReverse` | `not` | 1 | yes (self-inverse) | `Basis.negation` or `-x` | `SyntacticLayer.project` / chart unary path |
| `intersectionForward` / `intersectionReverse` | `intersection` | 2 | yes | `Basis.conjunction` or `torch.min(l, r)` | chart binary path |
| `unionForward` / `unionReverse` | `union` | 2 | yes | `Basis.disjunction` or `torch.max(l, r)` | chart binary path |
| `liftForward` / `liftReverse` | `lift` | 2 | yes | `Ops.lower(l, r, mode='AND', kind='smooth')` | chart binary path |
| `lowerForward` / `lowerReverse` | `lower` | 2 | yes | `Ops.lift(l, r, mode='OR', kind='smooth')` | chart binary path |
| `equalsForward` | `equals` | 2 | no (lossy) | `Basis.equal` (mutual parthood) | chart binary path; `_op_for_rule` |
| `partForward`, `trueForward`, `falseForward`, `nonForward`, `conjunctionForward`, `disjunctionForward`, `swapForward`, `whatForward`, `whereForward`, `whenForward`, `queryForward`, `absorbForward` | various | various | various | `Ops.*` or local kernel | chart and shift/reduce dispatch |
| `ContiguousForward` / `ContiguousReverse` | `Contiguous` | 1 | no (lossy) | per-axis amax envelope (matches `ContiguousLayer.forward`) | chart unary path |

**forward / reverse:**
- `SyntacticLayer.project(grammar, rule_id, left, right=None, ...)`
  ([Language.py:860-880](../../bin/Language.py:860)) is the rule-execution entry
  point: looks up the rule's method_name, finds the `*Forward` method, calls it.
- `SyntacticLayer.reverse_project(grammar, rule_id, result, ...)` is the
  matching inverse, calling `*Reverse` when present (unbound when the rule is
  lossy).
- Used by:
  - `SymbolicSpace._op_for_rule` ([Spaces.py:7345](../../bin/Spaces.py:7345)) for
    shift/reduce-style rule firing on stack entries.
  - The legacy `_compose_vector` Phase-2 cascade and `_compose_vector_chart`
    (greedy chart) paths via `_apply_rules_to_pairs` ([Language.py:1754-1768](../../bin/Language.py:1754)).
  - The new soft chart `_compose_chart_cky` via `_apply_rule_forward`
    ([Language.py:1881-1909](../../bin/Language.py:1881)) — this is a
    **separate dispatch** (also through `_RULE_METHODS`) that does the marker-
    masking gate before calling the rule's `*Forward`.

**compose / decompose:**
- `SyntacticLayer.compose(data, subspace, grammar, target_count=None)`
  ([Language.py:1402](../../bin/Language.py:1402)) is the **top-level entry
  point** for the entire grammar pipeline. It dispatches to:
  - 2D `data.shape == [B, N]` activation -> `_compose_activation`
  - 3D `data.shape == [B, N, D]` vector mode -> `_compose_vector`, which then
    dispatches to one of:
    1. `_compose_chart_cky` (when `softChartCompose=true`) — the new soft
       chart, fixed-rule-semantics dispatch via `_apply_rule_forward`.
    2. `_compose_vector_chart` (when `chartCompose=true`) — the older greedy
       chart.
    3. `_compose_to_target` (when `target_count` is set) — pairwise reduction.
    4. Phase-2 cascading soft-weighted composition (default, the legacy path).
- `SyntacticLayer.decompose` is the matching inverse, called from
  `WordSpace.decompose` ([Language.py:3939](../../bin/Language.py:3939)) on the
  reverse pass.
- The entire `WordSpace.composeSyntax` ([Language.py:3917-3929](../../bin/Language.py:3917))
  wraps this — that is the only caller into Surface 3's chart path.

## Where the three surfaces overlap

For the operators present in MM_grammar / XOR_grammar (`not`, `intersection`,
`union`, `lift`, `lower`):

| Operator | Surface 1 (naked) | Surface 2 (GrammarLayer) | Surface 3 (SyntacticLayer) |
|---|---|---|---|
| `not` | `Basis.negation` (used by Surface 3 fallback) | `NotLayer.forward` invoked in `SymbolicSpace.forward` (gated mix) | `notForward` calls `Basis.negation` |
| `intersection` | `PiLayer.forward` (the **always-on** C->S fold) | `IntersectionLayer.compose` -> `pi.compose` (dormant) | `intersectionForward` calls `Basis.conjunction` (different parametrization than the always-on PiLayer) |
| `union` | `SigmaLayer.forward` (the **always-on** P->C and C->S folds) | `UnionLayer.compose` -> `sigma.compose` (dormant) | `unionForward` calls `Basis.disjunction` (different parametrization than the always-on SigmaLayer) |
| `lift`, `lower` | not present (no dedicated layers) | not present | `liftForward`/`lowerForward` call `Ops.lower`/`Ops.lift` |
| `Contiguous` | not present | `ContiguousLayer.forward` invoked in `SymbolicSpace.forward` (gated mix) | `ContiguousForward` re-implements the same kernel |

The structural problem is concentrated on `intersection` and `union`:
**three different parameterizations of the same operator**. The always-on
`PiLayer` (Surface 1) trains on every batch through the task loss. The
chart's `intersectionForward` (Surface 3) trains only when the chart fires,
and uses a different math kernel (`Basis.conjunction` rather than the PiLayer
butterfly fold). They don't share parameters. The `IntersectionLayer` wrapper
(Surface 2) would have unified them but is never invoked.

This is why the chart can be diffuse or saturated without affecting the task
output: the model has a parallel always-on Pi path (Surface 1) that already
solves the task, and the chart's "intersection" rule (Surface 3) is computing
a *different* function with *different parameters*.

## What "fixing" this would look like

A unified architecture would be:
1. Each space owns a single `IntersectionLayer` / `UnionLayer` / etc.
   (Surface 2). The wrappers retain their inner Pi/Sigma instances as the
   parameterization.
2. The space's forward pipeline calls `intersection_layer.forward(x)` (or
   `compose(left, right)` for binary) instead of `forwardPi(x)`. Same
   parameters, same math, just exposed through the GrammarLayer surface.
3. The chart (Surface 3) dispatches its `intersectionForward` to the host
   space's `IntersectionLayer.compose(left, right)` rather than to
   `Basis.conjunction`. Parameters are shared end-to-end.

That collapses Surface 1 + Surface 2 into one parameter set and ties Surface
3 to it. Then the chart's rule choice actually steers which of the host's
parametrized folds executes — and the task loss flows through both paths
into the same weights.

That's a `Spaces.py` rewire, not a `Language.py` change.

---

## Naming alternatives for `<upward>` / `<downward>`

Current names overload the directional metaphor and read as orientation
rather than function. Candidates by clarity, in rough order of preference:

1. **`<parse>` / `<generate>`** — the most direct pair from the parser
   literature. Bottom-up = parsing (recognize structure from leaves);
   top-down = generation (emit leaves from structure). Single-syllable,
   active voice, unambiguous in the context of a grammar.
2. **`<analyze>` / `<synthesize>`** — domain-neutral, semantically precise.
   Useful if `<parse>` feels too NLP-specific given the model's other roles.
3. **`<recognize>` / `<produce>`** — accurate, slightly verbose. Reads well
   in prose ("the recognize rules…").
4. **`<inside>` / `<outside>`** — borrowed from the inside-outside algorithm
   tradition for PCFGs (Baker 1979). Compact and accurate but only legible
   to readers who know that literature.
5. **`<reduce>` / `<expand>`** — clean and short, but `reduce` collides with
   shift-reduce parser terminology (where reduce is a specific action, not
   a direction).

`<parse>` / `<generate>` is my recommendation. It's the cleanest pair, and
the symmetry with the rest of the project's vocabulary (`compose` /
`decompose`, `forward` / `reverse`) is intact: pairs of opposite-direction
ops at one level of abstraction, named for what they do.

If you want a rename, the migration is mechanical: `_fill_rule_list`
([Language.py:189-220](../../bin/Language.py:189)) reads from a `rules_dict`
keyed by `'upward'` / `'downward'`; rename in `_apply_cfg_sections` and the
section names in the cfg parser, plus the keys in `model.xml` /
`MM_grammar.xml`. Maybe 15 lines of changes plus the XML files.
