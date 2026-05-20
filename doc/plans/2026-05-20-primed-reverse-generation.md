# Primed reverse generation with weighted activation masks

## Context

Reverse grammatical operations today reach the codebook through
`Ops._binary_op_recommend` ([Layers.py:7857](../../bin/Layers.py)),
`Ops.conjunctionReverse` / `disjunctionReverse` ([Layers.py:7779](../../bin/Layers.py)),
and `Ops.liftReverseAll` / `lowerReverseAll` ([Layers.py:8095](../../bin/Layers.py),
[Layers.py:8195](../../bin/Layers.py)), invoked from
`Basis.lift(..., inverse=True)` / `Basis.lower(..., inverse=True)`
([Spaces.py:1055](../../bin/Spaces.py)). They select pairs `(x1, x2)`
over the augmented codebook `[⊥, W, ⊤]` with **no typed-admissibility
filter**: every codebook row is in the running, so a grammatically
inadmissible row can win on geometry alone.

In parallel, the order-typed grammar is in place
(`RuleOrderSignature` at [Language.py:151](../../bin/Language.py),
`_parse_category` recognizing `NP0` / `VP1` / `NP*`,
`refs_by_category` / `refs_by_order` materialized by
[bin/embed.py:428](../../bin/embed.py)). What's missing is the
**runtime use** of those typed indexes inside the reverse path, plus
a **psychologically realistic priming signal** that biases selection
within the admissible set. The companion plan
[2026-05-20-knowledge-artifact-order-typed-stm.md](2026-05-20-knowledge-artifact-order-typed-stm.md)
Phase 3 introduces the typed-admissibility piece; this spec layers a
soft priming multiplier on top of it.

Why now: lift/lower's `(parent, parent)` pseudo-inverse is being
replaced; the replacement should pick from the **right region** of
the codebook (typed) and **weighted by working-memory context**
(primed), in one composition.

## The invariant

```text
final_activation = .activation * priming_mask
                   (where hard admissibility has already zeroed
                    inadmissible candidates upstream in the
                    recommender)

hard_mask(r) == 0  =>  candidate r is never selected
priming_mask(r) defaults to 1.0 (multiplicative identity)
priming_mask(r) ∈ [1, boost_max], with boost_max ≈ 2 by default
```

The hard mask is binary admissibility, enforced before snap. The
priming mask is a **soft boost above unity** that biases ranking
inside the admissible set. With everything unprimed, priming = 1
everywhere and reverse generation is byte-for-byte equal to the
typed-only behavior of the sibling plan.

## Dependency on the order-typed-STM plan

This spec **depends on** the sibling plan landing first:

- **Phase 1** (already shipped per exploration): `RuleOrderSignature`,
  order-annotated grammar parsing.
- **Phase 2**: `WordSpace.attach_knowledge(view)` wires typed indexes
  into runtime; STM carries `_category`, `_order`, `_ref_id` parallel
  tensors; SHIFT pushes order-0 only; REDUCE builds typed admissibility.
- **Phase 3**: `Ops._binary_op_recommend` gains `left_rows` /
  `right_rows` kwargs; `Basis.lift/lower(..., inverse=True)` compute
  the category∩order row mask.

Phase 3's `left_rows` / `right_rows` are the **hard admissibility**
inputs. This spec adds the **priming multiplier** on top: it lives on
the Taxonomy, multiplies into `.activation`, and is also surfaced to
the recommender so the snap reflects priming.

## Hard admissibility mask (recap)

A candidate ref row is admissible (hard mask = 1) iff:

- **grammar category** — in `refs_by_category[rule.rhs_cat]`;
- **conceptual order** — in `refs_by_order[resolved_order]` where
  resolved order comes from the active rule's `RuleOrderSignature`
  plus operand-side order binding (STM's `_order` field);
- **word/category membership** — for lexical realization slots,
  passes word-table-derived constraints (SHIFT-time scoring);
- **operation compatibility** — the active op is allowed at this
  state by the order-typed grammar;
- **lattice feasibility** — Ops' `≤ y` / `≥ y` filter against the
  augmented codebook, with `⊥` / `⊤` always feasible.

Resolution at the reverse call site:

```text
sig = rule.order_signature
left_rows  = intersect(refs_by_category[sig.rhs_categories[0]],
                       refs_by_order[resolve_order(sig.rhs_order_exprs[0])])
right_rows = intersect(refs_by_category[sig.rhs_categories[1]],
                       refs_by_order[resolve_order(sig.rhs_order_exprs[1])])
```

Both flow into `Ops._binary_op_recommend` via the sibling plan's
`left_rows` / `right_rows` kwargs and are applied **before**
argmin/argmax. Inadmissible rows have weight 0; priming cannot
rescue them.

## Part/whole priming mask

### Storage: on the Taxonomy

The priming mask is **conjoined with `Taxonomy`**
([Language.py:5541](../../bin/Language.py)). The Taxonomy already
owns parent/children adjacency (the propagation graph) and is stable
across word activations; the priming state lives in the same object
so propagation can read parent/child ref_ids in place without
indirection.

Extend `Taxonomy` with a per-batch priming buffer:

```python
class Taxonomy:
    _parent   : dict[ref_id, ref_id]        # existing
    _children : dict[ref_id, list[ref_id]]  # existing

    # New:
    _priming  : torch.FloatTensor           # [B, V_ref_capacity]
                                            # default 1.0, primed ≥ 1.0
```

The buffer is **default 1.0 everywhere** (multiplicative identity —
no priming yet means no change to activations). It is allocated to
`V_ref_capacity` with the artifact's capacity-slack pattern; only
the first `V_ref_live` columns are read/written.

Lifecycle: per-batch, sentence-scoped. Reset to 1.0 at sentence
boundaries (same scope as STM). Decays between reverse calls within
a sentence (toward 1.0, not toward 0.0).

### Adjacency source: taxonomy parent / children

Propagation walks `Taxonomy._parent` and `Taxonomy._children` directly.
No separate CSR adjacency is materialized; the existing dict-backed
structure is sufficient for sentence-scale propagation depth
(`priming_depth ≤ 2` in practice).

If profiling later shows the dict walk dominates, swap to CSR-backed
parent/children tensors (already specified in the sibling plan's
artifact schema as `taxonomy.children_values` / `children_offsets`).
The user-facing semantics are unchanged.

`PartLayer`'s codebook-geometry parthood remains untouched — it is
the **forward-pass** mereological measure; the taxonomy is the
**reverse-pass priming graph**. They serve different roles and do
not need to be unified for this spec.

### Boost-above-unity semantics

The priming mask uses **1.0 as the default multiplicative identity**
and **boost values ≥ 1** to bias selection. Concretely:

```text
priming_value(r) = 1 + boost(r)
  boost(r) = 0       for unprimed r          → priming = 1.0
  boost(r) = 1       for freshly active r    → priming = 2.0
  boost(r) = α       for one-hop neighbor    → priming = 1 + α, α ≈ 0.5
  boost(r) → 0       between calls           → priming → 1.0
```

The boost dissipates multiplicatively toward the identity (1.0), not
toward zero. With everything unprimed the mask is identically 1 and
all downstream operations are unchanged.

### Propagation step

At each STM SHIFT / REDUCE that exposes refs to priming:

```text
1. Mark recently active refs:
       priming_mask[b, ref_id] = max(priming_mask[b, ref_id], 2.0)
   for every ref that just entered/changed in STM.

2. Propagate one hop upward to immediate wholes (parents):
       for parent in Taxonomy._parent[ref_id]:
           boost_at_parent = (priming_mask[b, ref_id] - 1) * hop_decay
           priming_mask[b, parent] = max(priming_mask[b, parent],
                                         1 + boost_at_parent)

3. Propagate one hop downward to immediate parts (children):
       for child in Taxonomy._children[ref_id]:
           boost_at_child = (priming_mask[b, ref_id] - 1) * hop_decay
           priming_mask[b, child] = max(priming_mask[b, child],
                                        1 + boost_at_child)

4. Repeat 2-3 up to `priming_depth` times (default 2).
```

Between reverse calls within a sentence:

```text
priming_mask = 1 + (priming_mask - 1) * temporal_decay
```

At sentence boundary: `priming_mask.fill_(1.0)`.

Siblings are **not** directly primed in step 1. They may pick up
boost only as a second-order effect: `A → whole(A) → sibling_part`.
No explicit sibling edge in the taxonomy is added.

### Configuration

Under `<architecture><priming>` (or equivalent):

```text
priming_depth     : int   (default 2)
hop_decay         : float (default 0.5)   # boost decay per hop
temporal_decay    : float (default 0.9)   # boost decay between calls
boost_initial     : float (default 1.0)   # added to 1.0 for active refs → 2.0
priming_enabled   : bool  (default true)  # false ⇒ mask identically 1.0
```

## Application: multiplied onto `.activation`

The priming mask multiplies into `.activation` on `Basis`
([Spaces.py:769](../../bin/Spaces.py)):

```text
.activation' = .activation * priming_mask[order]
```

Two coupled application points, both required:

### A. At selection time (inside the recommender)

Before `Ops._binary_op_recommend` ([Layers.py:7857](../../bin/Layers.py))
takes argmin/argmax over the augmented codebook, multiply the candidate
score by `priming_mask[ref_id]`:

```text
score'[r] = score[r] * priming_mask[r]
```

Inadmissible rows (zeroed by `left_rows` / `right_rows`) stay zero.
Admissible rows get an unchanged score (× 1) by default, or a boost
(× 2) when primed. The `⊥` / `⊤` sentinels are pinned to priming = 1.0
(no priming).

The recommender's contract is widened so `left_rows` / `right_rows`
may also accept float weights, OR a separate `left_priming` /
`right_priming` parameter is passed alongside the binary mask
(implementation choice — both have the same observable behavior).
Recommended: a separate priming parameter, since semantics are
distinct (binary admissibility vs. soft boost).

```python
Ops._binary_op_recommend(
    result, W, op_name, monotonic=True,
    left_rows:     LongTensor | None = None,    # sibling plan, binary admissibility
    right_rows:    LongTensor | None = None,
    left_priming:  FloatTensor | None = None,   # new: soft boost, default 1.0
    right_priming: FloatTensor | None = None,
)
```

### B. After snap (on `.activation` itself)

After `Codebook.forward` writes `.activation` ([Spaces.py:2055](../../bin/Spaces.py)),
multiply by the priming mask before downstream consumption:

```text
basis.activation = basis.activation * priming_mask[order][b, snapped_ref_id]
```

This makes the boost visible to `Basis.activeDense()` and any other
consumer that reads `.activation` directly. With priming = 1.0
(default), this multiplication is a no-op and current behavior is
preserved.

### Why both

- (A) without (B): the recommender picks a primed row, but downstream
  consumers see only the post-snap `.activation` and lose the priming
  signal.
- (B) without (A): the selected row is unprimed-optimal; multiplying
  priming onto `.activation` post-snap only re-weights a settled
  outcome, never changes the choice.
- Both: priming biases the choice **and** the downstream weighting,
  so cognitive state propagates end-to-end.

## Reverse operation flow

For each reverse grammatical operation:

```text
1. Determine target slot(s): left child, right child, or lexical.
2. Build hard row-mask(s) from the rule's RuleOrderSignature +
   WordSpace.knowledge typed indexes:
     hard_rows = intersect(refs_by_category[c], refs_by_order[o])
3. Read priming_mask[order] from Taxonomy._priming for the batch.
4. Call Ops._binary_op_recommend(
        result, W, op_name,
        left_rows=hard_left,   right_rows=hard_right,
        left_priming=priming_mask[order_left],
        right_priming=priming_mask[order_right],
   )
5. The recommender masks by hard_rows (zero ⇒ skip) then multiplies
   admissible scores by priming before argmin/argmax. Sentinels pinned
   to priming = 1.0.
6. Codebook.forward writes .activation as today (Spaces.py:2055).
   Then: basis.activation *= priming_mask[order][b, snapped_ref_id].
7. Mark the freshly snapped ref_ids as active in Taxonomy._priming;
   propagate one round (steps 1-4 of the propagation algorithm).
8. Return the selected candidate(s) to the caller.
```

The invariant holds: `hard_rows[r] == 0 ⇒ r never selected`; priming
only reranks within the admissible set.

## Optional learned predictor (deferred)

A later iteration may add an operation-conditioned MLP reranker over
the constrained candidate slice. Features:

```text
parent activation, operation id, target slot,
candidate prototype, candidate category, candidate order,
priming score, mereological features, STM context summary
```

For binary reverse, factorized top-k pair scoring rather than full
`O(V²)`:

```text
score left candidates  -> top_k_left
score right candidates -> top_k_right
pair-rerank top_k_left × top_k_right
```

**Out of scope** for the initial implementation; listed only so the
data layout (priming on Taxonomy, surfaced through `.activation`)
stays compatible.

## Staging

### Phase A — Taxonomy priming buffer

- Extend `Taxonomy` ([Language.py:5541](../../bin/Language.py)) with
  a `_priming: FloatTensor[B, V_ref_capacity]` buffer initialized to
  1.0, capacity-slack aligned with the reference codebook.
- Add `Taxonomy.prime(ref_ids, batch=b, boost=1.0)` →
  sets/maxes those entries to `1 + boost`.
- Add `Taxonomy.propagate(ref_ids, batch=b, depth=2, hop_decay=0.5)`
  → walks parent/children, applies boost-above-unity dissipation.
- Add `Taxonomy.decay(temporal_decay=0.9)` and
  `Taxonomy.reset()` (sentence-boundary hook).

### Phase B — Recommender priming kwargs

- `Ops._binary_op_recommend` ([Layers.py:7857](../../bin/Layers.py))
  gains `left_priming` / `right_priming` FloatTensor kwargs alongside
  the sibling plan's `left_rows` / `right_rows` (the sibling plan
  introduces those first).
- Apply `score *= priming` after binary masking, before argmin/argmax.
  Sentinel rows pinned to priming = 1.0.
- Forward through `conjunctionReverse` / `disjunctionReverse` /
  `liftReverseAll` / `lowerReverseAll`.

### Phase C — `.activation` post-snap multiplication + reverse-call wiring

- In `Basis.lift(..., inverse=True)` / `Basis.lower(..., inverse=True)`
  ([Spaces.py:1055](../../bin/Spaces.py)):
  - Compute hard rows (category ∩ order) per slot.
  - Fetch `priming_mask = WordSpace.taxonomy._priming[batch, :V_ref_live]`.
  - Restrict to the per-slot order and pass as `left_priming` /
    `right_priming` to the recommender.
- After `Codebook.forward` ([Spaces.py:2055](../../bin/Spaces.py))
  writes `.activation`, multiply by the priming mask at the snapped
  ref_id(s).
- On successful snap: call `Taxonomy.prime(snapped_ref_ids)` then
  `Taxonomy.propagate(...)`.

### Phase D — Lifecycle + configuration + telemetry

- STM sentence-boundary hook calls `Taxonomy.reset()`.
- Between reverse calls within a sentence, call `Taxonomy.decay()`.
- Expose config under `<architecture><priming>`:
  `priming_depth`, `hop_decay`, `temporal_decay`, `boost_initial`,
  `priming_enabled`. `priming_enabled = false` short-circuits the
  mask to identically 1.0 (preserves byte-for-byte typed-only
  behavior).
- Debug counter: "candidate selected with priming boost > 1" vs
  "candidate selected at priming = 1" so the contribution of priming
  to loss is attributable.

## Tests

### Taxonomy priming buffer

`test/test_taxonomy_priming.py`:

- `Taxonomy._priming` initialized to all-1.0; shape
  `[B, V_ref_capacity]`.
- `Taxonomy.prime([r], batch=0)` → `_priming[0, r] == 2.0`,
  all other entries still 1.0.
- `Taxonomy.propagate([r], depth=1, hop_decay=0.5)` → immediate
  parent / children of `r` get value `1.5`; non-neighbors stay 1.0.
- `Taxonomy.propagate([r], depth=2, hop_decay=0.5)` → siblings (via
  shared parent) get `1.25`; unrelated refs stay 1.0.
- `Taxonomy.decay(0.9)` → primed entries move from 2.0 to 1.9
  (and 1.5 to 1.45, etc.); identity entries stay 1.0.
- `Taxonomy.reset()` → all entries 1.0 again.
- Capacity-slack: appending refs via `extend_artifact` preserves
  `_priming` for existing rows; new rows initialize to 1.0.

### Hard mask × priming composition

`test/test_primed_reverse_hard_mask.py`:

- Inadmissible **category** candidates are never selected, even when
  `priming_mask[that_row] == 2.0`. The recommender returns an
  admissible-category row.
- Inadmissible **order** candidates are never selected, even when
  highly primed. The recommender returns a row at the admissible
  order.
- LIFT exposes only refs at `parent_order = operand_order + 1`.
- LOWER exposes only refs at `parent_order = operand_order - 1`.
  LOWER from order 0 is inadmissible.
- With `priming_enabled = false` (or `_priming` identically 1.0),
  reverse generation matches the sibling plan's Phase 3 typed-only
  behavior byte-for-byte.

### `.activation` post-snap multiplication

`test/test_activation_priming_multiply.py`:

- Snap a ref `r` whose `priming_mask[r] == 2.0`; assert
  `Basis.activation == raw_activation * 2.0`.
- Snap an unprimed ref; assert `.activation` unchanged from current
  behavior (identity multiplier).
- With `priming_enabled = false`, `.activation` post-snap equals the
  pre-priming value byte-for-byte.

### Combined / integration

`test/test_primed_reverse_combined.py`:

- Among admissible rows, the recommender prefers primed rows
  (priming = 2.0) over equally feasible un-primed rows (priming = 1.0).
- The recommender never selects a row outside `hard_rows` regardless
  of priming value.
- Sentinel feasibility: `⊥` / `⊤` selection unchanged by priming.

`test/test_primed_reverse_integration.py`:

- End-to-end: small grammar + lexicon, a sentence that activates a
  set of refs in STM, then a LIFT-reverse that should prefer a ref
  primed via shared whole.
- Without priming (`priming_enabled = false`), the same call selects
  a different (also-admissible) ref.
- Across consecutive reverse calls within a sentence, priming decays
  by `temporal_decay`; at sentence boundary, priming is gone.

### Regression

[test/test_inverse_recommend.py](../../test/test_inverse_recommend.py)
13 tests must pass with no priming params and `left_rows=None,
right_rows=None`. `test/test_lift_lower_restricted_inverse.py` from
the sibling plan must pass with `priming_enabled = false`.

## Verification

```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
./.venv/bin/python -m pytest \
    test/test_taxonomy_priming.py \
    test/test_primed_reverse_hard_mask.py \
    test/test_activation_priming_multiply.py \
    test/test_primed_reverse_combined.py \
    test/test_primed_reverse_integration.py \
    test/test_inverse_recommend.py \
    test/test_lift_lower_restricted_inverse.py \
    test/test_ops_lift_lower.py \
    test/test_grammar_binary_ops.py -q
```

## Critical files

- [bin/Language.py](../../bin/Language.py) — `Taxonomy` gains
  `_priming` buffer + `prime` / `propagate` / `decay` / `reset`.
- [bin/Layers.py](../../bin/Layers.py) — `Ops._binary_op_recommend`
  (~7857) gains `left_priming` / `right_priming` FloatTensor kwargs.
  `conjunctionReverse` / `disjunctionReverse` / `liftReverseAll` /
  `lowerReverseAll`: forward the priming kwargs.
- [bin/Spaces.py](../../bin/Spaces.py) — `Basis.lift(..., inverse=True)`
  / `Basis.lower(..., inverse=True)` (~1055): compute hard rows,
  fetch priming slice from `Taxonomy._priming`, pass both to
  recommender. Post-snap: `basis.activation *= priming_mask[order][b, ref]`.
- [bin/Models.py](../../bin/Models.py) — wire STM sentence-boundary
  hook to `Taxonomy.reset()`; between-call hook to `Taxonomy.decay()`.
- [bin/embed.py](../../bin/embed.py) — `extend_artifact`: when
  appending refs, extend `Taxonomy._priming`'s column count (init new
  columns to 1.0).
- [bin/Mereology.py](../../bin/Mereology.py) — unchanged.
- [bin/Layers.py:2840 PartLayer](../../bin/Layers.py) — unchanged.
  Geometric parthood remains the forward-pass measure; taxonomy is
  the reverse-pass priming graph.

## Integration with existing architecture

- `Taxonomy` ([Language.py:5541](../../bin/Language.py)) provides the
  parent/children graph; this spec adds the per-batch priming buffer
  to that same class.
- `ShortTermMemory` ([Spaces.py:8861](../../bin/Spaces.py)) provides
  the sentence-scoped lifecycle that drives `Taxonomy.reset()` and
  `Taxonomy.decay()` invocations.
- `RuleOrderSignature` + `refs_by_category` / `refs_by_order`
  ([Language.py:151](../../bin/Language.py),
  [embed.py:428](../../bin/embed.py)) supply the hard-mask inputs.
- `.activation` on `Basis` is **multiplied** by the priming mask
  post-snap; the existing `Codebook.forward` snap path is unchanged.
- `PartLayer` ([Layers.py:2840](../../bin/Layers.py)) is unchanged.
  Forward-pass geometric mereology and reverse-pass taxonomic priming
  coexist without coupling.

## Assumptions

- Priming is a **boost above unity**, never below; default value is
  1.0 (multiplicative identity).
- The Taxonomy's parent/children graph is the adjacency for priming
  propagation. Taxonomy depth is **not** conceptual order; the order
  axis for fetching `priming_mask[order]` comes from refs' intrinsic
  order via `refs_by_order`.
- Priming is transient working-memory state with sentence-scoped
  lifecycle.
- Hard admissibility (category ∩ order) is enforced upstream in the
  recommender via the sibling plan's `left_rows` / `right_rows`;
  priming is a downstream soft boost on the surviving candidates.
- The sibling plan's Phase 2 + Phase 3 land before this spec is
  useful; otherwise there is no typed admissibility to layer priming
  on top of.
- Learned MLP reranking is optional and out of scope here.

## Open questions deferred to implementation

- **Priming on `⊥` / `⊤` sentinels**: pinned to 1.0 (no priming) in
  this spec; whether to expose them as "always primed" or
  "explicitly excluded" is a tuning knob.
- **Per-order vs. flat priming buffer**: the spec stores
  `_priming: [B, V_ref]` flat (ref_id-indexed). Slicing by order
  happens at fetch time via `refs_by_order[k]`. If the order-slicing
  becomes hot, a `[B, num_orders, V_ref_at_order]` layout is a
  drop-in swap.
- **Boost combination rule**: the spec uses `max(...)` so multiple
  primings of the same ref don't compound past 2.0. An additive
  alternative is `priming = 1 + Σ_i boost_i`, capped. Pick during
  implementation based on observed behavior.
- **Whether `Taxonomy._priming` should grow on `extend_artifact`
  via the capacity-slack Parameter pattern** (cheap, but the buffer
  isn't a Parameter — no optimizer state to preserve). Plain
  `torch.cat` on the column dim is sufficient.
