# Symbolic heat retrieval: content proximity + taxonomy-guided priming

Date: 2026-06-06

## Goal

Replace QKV-style attention over the symbolic codebook with a simpler
symbolic retrieval mechanism:

```text
best-fit symbol = content proximity + transient heat
```

where:

* **content proximity** compares a CS idea/query against symbol codebook rows;
* **heat** is transient priming over symbol/ref rows;
* **taxonomy** guides heat propagation and admissibility masks;
* **grammar type / POS / order** restricts the candidate set before scoring.

This is primarily a reverse-path mechanism: grammatical generation and
inverse-recommendation need to find words/codebook rows that best form a basis
for an idea produced by the forward path and later prediction.

The forward path should maintain the heat state. The reverse path should
consume it.

## Existing hooks

The current codebase already has most of the scaffolding:

* `Language.Taxonomy` owns `_priming: [B, V_ref_capacity]`, initialized to
  `1.0` and decayed by:

  ```text
  priming <- 1 + temporal_decay * (priming - 1)
  ```

* `Taxonomy.prime(...)` boosts selected ref rows.
* `Taxonomy.propagate(...)` spreads boost along `KnowledgeView`
  parent/children adjacency.
* `WordSubSpace.priming_kwargs_for_slots(...)` intersects
  `refs_by_category` and `refs_by_order`, then emits:

  ```text
  left_rows, right_rows, left_priming, right_priming
  ```

* `Ops.disjunctionReverse` / `Ops.conjunctionReverse` already accept row
  restrictions and priming weights.
* `Basis.lift(..., inverse=True)` and `Basis.lower(..., inverse=True)` already
  forward those kwargs.

So this plan should extend the current priming/reverse-generation path, not
replace it.

## Handoff addendum: ownership and attention config

This mechanism should be implemented as a `Layer`-shaped retrieval/scoring
component owned by `SymbolicSpace` or `ConceptualSpace`, not as transformer
QKV attention over PS/CS event tensors.

Recommended ownership split:

```text
Taxonomy / WordSubSpace:
  persistent row heat, taxonomy propagation, ref-indexed buffers

SS or CS retrieval Layer:
  content-proximity scoring, heat fusion, top-k candidate union,
  optional first/second-order carrier math

Ops / Basis reverse helpers:
  existing grammar inverse recommendation over the restricted candidates
```

The persistent heat remains on `Taxonomy` because it is indexed by stable
symbol/ref ids and depends on taxonomy adjacency. The new Layer should consume
that state, not own it.

Repurpose the per-space `<attention>` parameter as the symbolic retrieval mode
for that space:

```xml
<ConceptualSpace>
  <attention>off|primer|second-order|low-rank</attention>
</ConceptualSpace>

<SymbolicSpace>
  <attention>off|primer|second-order|low-rank</attention>
</SymbolicSpace>
```

Mode semantics:

```text
off:
  no heat-biased retrieval; current reverse behavior

primer:
  row heat only; use Taxonomy._priming / priming_mask as the soft boost

second-order:
  derive and use a dense feature-space carrier C = A_S^T diag(r_S) A_S
  when D is small enough

low-rank:
  derive active factors U = diag(sqrt(r_S)) A_S and compute Cq = U^T(Uq)
  without materializing C
```

`<hasAttention>` should be retired or treated only as a deprecated legacy
alias. It should no longer request QKV/self-attention inside `PerceptualSpace`
or `ConceptualSpace`.

The old `AttentionLayer` enlistment in PS/CS should be removed during
implementation:

* PS should not construct or call `AttentionLayer` as a pre-`PiLayer` QKV pass.
* CS should not construct or call `AttentionLayer` as a tensor self-attention
  pass.
* Existing validation errors that only protected the old reshape-vs-QKV path
  should be removed or rewritten for the new retrieval modes.

If a future model needs QKV over tokens, it should use a separate explicit
knob/name. In this plan, `attention` means symbolic retrieval mode.

## Core representation

Keep three separate state variables:

```text
p: [B, V]      row priming / heat, neutral at 1.0
r: [B, V]      active heat above neutral, r = max(p - 1, 0)
z: [B, D]      first-order semantic heat, z = A_S^T r_S
C: [B, D, D]   optional second-order carrier, C = A_S^T diag(r_S) A_S
```

where:

```text
A: [V, D]      symbolic codebook rows
S              sparse active/top-k heat row set
```

The current `_priming` buffer is `p`. It should not be overloaded into `C`.
Instead, derive `z` and optionally `C` from `p` when needed.

## Retrieval score

Use a two-factor score after hard masks:

```text
score_i = alpha * proximity(q, a_i) + beta * heat_i
```

with optional first/second-order terms:

```text
score_i =
    alpha * proximity(q, a_i)
  + beta  * heat_i
  + gamma * dot(a_i, z)
  + delta * dot(a_i, C q)
```

Default should start with the simple form:

```text
score_i = alpha * proximity(q, a_i) + beta * log1p(max(heat_i - 1, 0))
```

Use `log1p` so a hot taxonomic ancestor does not swamp content proximity.

## Candidate generation

Never score the whole million-row codebook densely on the reverse path.

Use a candidate union:

```text
C_content = top_k_by_proximity(q, A)
C_heat    = top_k_by_heat(p)
C_typed   = refs_by_category(category) ∩ refs_by_order(order)
C         = (C_content ∪ C_heat) ∩ C_typed
```

Then rank only `C`.

If `C` is empty, fall back in this order:

1. typed rows only;
2. content top-k only;
3. existing sentinel fallback in the recommender.

## Taxonomy role

Taxonomy should mostly be a heat-transition and mask structure, not a separate
dense attention mechanism.

Forward/reverse update:

```text
heat <- decay(heat)
heat <- direct prime(selected_refs)
heat <- taxonomy_spread(heat)
```

The taxonomy contributes through:

* parent/child heat spreading;
* category/order admissibility masks;
* optional sibling propagation via parent in two hops;
* optional part/whole or meronomy edges later.

Conceptual-space geometry contributes through:

* content proximity `proximity(q, a_i)`;
* optional local geometric heat spread:

  ```text
  heat_i += eta * sum_j heat_j * kernel(a_i, a_j)
  ```

Keep explicit taxonomy and geometric proximity separate.

## Forward-path responsibilities

The forward path should update heat whenever it commits, snaps, predicts, or
observes symbol/ref rows.

### Word/percept commit

When PS/SS resolves an input word to a symbol/ref row:

```text
taxonomy.prime(ref_ids, batch=b)
taxonomy.propagate(ref_ids, batch=b, depth=priming_depth)
```

This is current behavior conceptually; verify all forward commit sites call it.

### Grammar reduction

When C-tier grammar reduction selects or creates a parent idea whose source
rows are known, prime:

```text
selected child refs
selected rule ref, if grammar rules are represented as refs
parent ref, if resolved
```

If the parent idea is not yet snapped to a symbol row, update only the
first-order semantic carrier `z` from the idea vector and defer row priming.

### Prediction

When the intra-sentence or discourse predictor produces a next-idea prior:

* do not hard-prime a row unless the predictor has snapped/resolved it;
* do update a query-side semantic seed, e.g. `taxonomy._heat_query[b] = q`.

The reverse path can combine that semantic seed with the row heat.

## Reverse-path responsibilities

The reverse path should use heat-biased retrieval whenever it needs a codebook
row basis for a CS idea.

Primary call sites:

* `IntersectionLayer.reverse(..., basis=...)`
* `UnionLayer.reverse(..., basis=...)`
* `Basis.lift(..., inverse=True, ...)`
* `Basis.lower(..., inverse=True, ...)`
* `Ops.disjunctionReverse(...)`
* `Ops.conjunctionReverse(...)`
* any word/codebook reverse decode that currently does nearest-row lookup only

The reverse flow should be:

```text
1. Derive q from the CS idea / parent tensor.
2. Fetch typed admissible rows from KnowledgeView.
3. Fetch heat from Taxonomy.priming_mask(batch=b).
4. Build candidate union: top-k content ∪ top-k heat ∩ typed rows.
5. Rank by proximity + heat.
6. Pass restricted row ids and row weights into the existing recommender.
7. After a row is selected, call taxonomy.note_selection(...)
8. Prime/propagate selected rows for subsequent reverse steps.
9. Decay between reverse calls within a sentence.
```

## API additions

Add methods on `Taxonomy`:

```python
def heat_mask(self, batch=0) -> Tensor | None:
    """Return max(_priming - 1, 0) over live rows."""

def topk_heat(self, k: int, batch=0, rows: Tensor | None = None) -> Tensor:
    """Return top-k hot live ref rows, optionally restricted."""

def build_semantic_heat(
    self,
    codebook_rows: Tensor,
    batch=0,
    rows: Tensor | None = None,
    topk: int | None = None,
) -> Tensor:
    """Return z = A_S^T r_S."""

def build_outer_heat(
    self,
    codebook_rows: Tensor,
    batch=0,
    rows: Tensor | None = None,
    topk: int | None = None,
    low_rank: bool = True,
) -> Tensor:
    """Return U = diag(sqrt(r_S)) A_S or C = U^T U."""
```

Add method on `WordSubSpace`:

```python
def retrieval_candidates_for_slot(
    self,
    query,
    basis,
    category,
    order,
    batch=0,
    topk_content=64,
    topk_heat=64,
) -> dict:
    """Return rows, priming weights, and optional score diagnostics."""
```

This should reuse `priming_kwargs_for_slots` where possible, but add
content-top-k and heat-top-k candidate union.

## Recommender changes

Extend `Ops._binary_op_recommend` with optional row-score inputs:

```python
left_scores=None
right_scores=None
```

or, more conservative:

* keep `left_priming` / `right_priming` as the only soft weights;
* have `WordSubSpace.retrieval_candidates_for_slot` write proximity+heat into
  the priming vector for candidate rows.

Preferred first implementation: do not modify `Ops` signature. Produce a
single boosted row-weight vector:

```text
weight_i = exp(beta_heat * h_i + alpha_sim * sim_i)
```

and pass it as `left_priming` / `right_priming`.

This keeps existing recommender tests meaningful and avoids another score
surface in `Ops`.

## Sparse million-row behavior

For `V ~= 1M`, never materialize:

```text
A A^T: [V, V]
```

Allowed representations:

```text
p: [B, V] sparse/dense scalar heat
z: [B, D]
U: [B, k, D] low-rank active outer-product factors
C: [B, D, D] only when D is small enough
B_S: [B, k, k] active symbol-symbol subgraph only
```

For `D=1024`, a dense `C` is about 4 MB per fp32 batch row and is plausible for
small batches. For larger `D`, use low-rank `U` and compute:

```text
Cq = U^T (U q)
```

without materializing `C`.

## Configuration

Use the per-space `<attention>` mode to select the retrieval mechanism:

```xml
<ConceptualSpace>
  <attention>off</attention>
</ConceptualSpace>

<SymbolicSpace>
  <attention>primer</attention>
</SymbolicSpace>
```

Additional scalar knobs may live under `<architecture><priming>` or equivalent
flat knobs if the XML parser does not support nested priming config yet:

```xml
<retrievalTopKContent>64</retrievalTopKContent>
<retrievalTopKHeat>64</retrievalTopKHeat>
<retrievalAlpha>1.0</retrievalAlpha>
<retrievalBeta>0.5</retrievalBeta>
<retrievalUseOuterProduct>false</retrievalUseOuterProduct>
<retrievalOuterTopK>32</retrievalOuterTopK>
```

Defaults should preserve current behavior:

```text
attention=off
```

or, if enabled by default, set:

```text
attention=primer
alpha=0, beta=1
```

to reproduce priming-only behavior.

## Tests

Add focused tests before integration:

1. `Taxonomy.heat_mask` returns `_priming - 1` clipped at zero.
2. `Taxonomy.topk_heat` returns only live rows and respects row restriction.
3. `build_semantic_heat` equals `A_S.T @ r_S`.
4. Low-rank outer product computes `Cq` equal to dense `C @ q`.
5. Candidate union includes content-nearest rows even when cold.
6. Candidate union includes hot rows even when content similarity is weak.
7. Typed masks exclude hot but inadmissible rows.
8. Unity heat leaves old recommender output byte-identical.
9. Heat+content weights prefer the expected candidate in
   `Ops.disjunctionReverse`.
10. Reverse generation primes selected rows for the next reverse step.
11. Sentence reset clears row heat and derived semantic/outer state.
12. Million-row smoke test uses sparse top-k and does not allocate `[V,V]`.

## Migration phases

### Phase 1: Derived heat helpers

Add `Taxonomy.heat_mask`, `topk_heat`, and semantic/outer builder helpers.
No behavior change.

### Phase 2: Candidate union helper

Add `WordSubSpace.retrieval_candidates_for_slot`. Keep old
`priming_kwargs_for_slots` as the compatibility path.

### Phase 3: Reverse recommender integration

Change reverse call sites that already pass `left_rows` / `right_rows` /
`left_priming` / `right_priming` to use the new candidate helper when
the owning space's `attention` mode is `primer`, `second-order`, or `low-rank`.

### Phase 4: Forward heat updates

Audit and wire all row-commit sites to prime/propagate selected refs.
Keep prediction-only ideas as semantic query seeds unless they snap.

### Phase 5: Optional outer-product carrier

Enable low-rank `U` first. Only materialize dense `C` for small `D` and tests.

## Non-goals

* Do not add QKV attention over the full symbolic codebook.
* Do not reuse `<attention>` to mean transformer/QKV tensor self-attention.
* Do not create a dense `V x V` association matrix.
* Do not let blended retrieval vectors become stable symbol identities.
* Do not remove existing hard row masks or sentinel fallbacks.

## Acceptance criteria

* Existing primed reverse-generation tests still pass with retrieval disabled.
* With retrieval enabled, reverse lookup uses:

  ```text
  typed mask + content proximity + heat
  ```

  and does not score all rows densely.
* Generated inverse operands remain valid under grammar category and order
  restrictions.
* A hot but type-invalid row is never selected.
* A content-near but cold row can beat a hot but distant row when `alpha` is
  high.
* A hot row can beat a nearby row when `beta` is high.
* No `[V,V]` tensor is allocated in million-row smoke tests.
