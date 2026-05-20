# Order-typed knowledge artifact + STM shift/reduce syntax

## Summary

A single co-versioned knowledge artifact holds the word table, the reference
codebook (order-indexed), the taxonomy (structural metadata), and the typed
membership indexes. The grammar gains explicit order typing. The runtime
adds an STM-backed shift/reduce parser as an opt-in alternative to the
existing chart, with typed admissibility masking inadmissible rules before
the reducer softmax.

**Cutover is incremental, not destructive.** Per the design call
(2026-05-20 owner session):

  * `parser_backend` switches between `chart` (default — existing CKY,
    unchanged), `stm` (the new typed shift/reduce driver), and
    `parallel` (both, chart authoritative). CKY remains as oracle,
    fallback, diagnostic parser, and generation backup.
  * `WordSpace.category_codebook` **stays** for now. Future retirement
    will introduce a clean ``category_embedding: nn.Embedding[N_categories,
    pos_dim]`` for grammar labels; `refs_by_category` from the artifact
    is for candidate masks, not for rule-predictor input features.
  * Symbol learning is a disabled-by-default subsystem. Statistics are
    collected detached; ``extend_artifact`` only fires at explicit flush
    boundaries, never inside autograd forward.
  * "Done" means no regression on chart-backed defaults, primitives
    shipped behind switches, and full STM-only cutover documented as a
    follow-up.

Written words enter as order-0 ramsified word symbols. Higher-order
meanings are reached only by ramsified derivations, and the grammar is the
source of order typing.

The key invariant:

```text
Only LIFT and LOWER change conceptual order.
All other syntactic operations are order-preserving.
```

This plan supersedes `2026-05-20-knowledge-artifact.md`.

## Context

Three problems converged onto one design:

1. **Lift/lower's `(parent, parent)` pseudo-inverse is structurally lossy**
   beyond what is mathematically necessary. The verified union/intersection
   recommender (`doc/plans/2026-05-17-inverse-recommend-union-intersection.md`,
   implemented at `bin/Layers.py:7857`) already produces a real `(x1, x2)`
   pair from the augmented codebook + sentinels for lattice ops, but
   lift/lower still return `(parent, parent)`
   ([Layers.py:2422](../../bin/Layers.py),
   [Layers.py:2529](../../bin/Layers.py)). They should run through the same
   recommender, **with the candidate set restricted by category AND order**.

2. **Three overlapping categorical-metadata structures, none linked**:
   `WordSpace.category_codebook` (Language.py:5734), `WordSpace.taxonomy`
   (Language.py:5750, never populated), and `SymbolicSpace.subspace.what.W`
   (Spaces.py:9795). And `PerceptualSpace.wv` (Spaces.py:2410) owns surface
   forms separately at the perceptual layer. Order has been variously
   inferred from word loop, taxonomy depth, or orthographic rows — none of
   which is the right authority.

3. **MPHF indexing brittleness**: implicit perceptual-row → symbolic-row
   mappings aren't stored; codebook changes silently invalidate them.

The unifying move: a single co-versioned artifact, **order-typed at every
layer**, where the grammar is the order-typing authority. The taxonomy
exists as structural metadata only — taxonomy depth ≠ conceptual order.

## Corrected Architecture

### Word ≠ object (Frege framing)

The codebase has conflated sign and reference. With this design the
distinction is structural:

- A **word-symbol** is the symbol of the written form. Ragged byte
  sequence of the surface. Lives in the word table. Order 0 by definition.
- A **reference-symbol** is the symbol of what the word denotes / what an
  operation composes. Learned scalar prototype in conceptual space at its
  intrinsic conceptual order. Lives in the reference codebook (=
  `SymbolicSpace.subspace.what`).

Direct written-word lookup hits only order-0 refs. Higher-order refs are
reached only by ramsified derivation (LIFT / LOWER and the rest of the
grammar over order-typed stack items). Frege's sign / reference.

### Orthography, words, references, and order

Four relations that were previously conflated:

```text
orthographic row
  perceptual surface form
  no independent conceptual order beyond lexical entry

order-0 word symbol
  ramsified lexical sign reached by written-word recognition
  direct target of word-table lookup

reference
  ramsified conceptual / theoretical role
  has intrinsic conceptual order, supplied by grammar

derivation
  ramsified syntactic operation over typed stack items
  may preserve or change conceptual order according to the grammar
```

Direct written-word lookup must only produce order-0 symbols:

```text
word row -> order-0 ref_id
```

Higher-order references are not direct orthographic lookup targets.

### The artifact (sidecar `.pt` alongside the existing `.kv`)

`BasicModel.knowledge.pt` is a torch-saved dict:

```python
{
    'word_table': {
        'keys_values':  uint8 tensor,                # CSR ragged surface bytes
        'keys_offsets': long tensor [N_lex + 1],
        'ref_ids':      long tensor [N_lex],         # every direct target has order == 0
    },
    'reference_codebook': {
        'references': float tensor [V_ref_capacity], # nn.Parameter (live rows = V_ref_live)
        'v_ref_live': int,
        'order':      long tensor [V_ref_capacity],  # intrinsic conceptual order per ref
    },
    'typed_indexes': {
        'refs_by_order':    CSR long,                # order k -> ref_ids at order k
        'refs_by_category': CSR long,                # category 'NP' -> ref_ids
    },
    'taxonomy': {
        'parent':           long tensor [V_ref_capacity],   # -1 = root
        'children_values':  long tensor,                    # CSR
        'children_offsets': long tensor [V_ref_capacity + 1],
    },
    'grammar': {
        'rule_order_signatures': list[RuleOrderSignature],  # see Order-Typed Grammar
    },
}
```

`refs_by_order` is consumed by symbol snapping and inverse recommendation.
`refs_by_category` is consumed by typed grammatical admissibility. Both
materialized at load; updated when symbol-learning appends a new ref.

The taxonomy is **structural metadata only** — parent / children indices
that record the partition relations we used to compute lazily. **Taxonomy
depth is not conceptual order** and must not be used as the source of
order typing. The grammar's `RuleOrderSignature` is the authoritative
source.

`ref_id` is the universal cross-reference. Sort order is stable, append-only.
Re-ordering would invalidate every external reference.

### Reference codebook + bivector retirement (narrow)

**Only `SymbolicSpace.subspace.what.W` changes shape.** Prototype storage
goes from `[V_ref, 2]` catuskoti bivectors to `[V_ref]` learned scalars —
references in the codebook are settled prototypes, fully determined scalar
centroids in DoT-space, no per-prototype pos/neg polarity needed.

**Everything downstream stays bivector.** Activations remain bivector;
the catuskoti polarity / partial-knowledge / contradiction machinery is a
property of the **live activation state**, not of the **settled prototype**.
TruthLayer is untouched. Mereology (`Contiguous`, `Continuous`, `Area`,
`Luminosity`, `isIsomorphic`) continues to consume `leaf[..., :2]`
bivector slices from `hoc_shape`'s back-projected activations.
`Ops.hyperrectangle_volume` / `corner_overlap` / `epsilon_delta`
unchanged. Lift/lower internals (`_gated_sigma_internal` at Layers.py:2378,
`_gated_pi_internal` at 2502) operate on activations and are unchanged.

What actually changes:

- `SymbolicSpace.subspace.what.W: nn.Parameter[V_ref, 2] → nn.Parameter[V_ref]`.
  Storage only; `subspace.nWhat = 2` stays at 2 (activation channeling
  unchanged).
- Codebook snap comparison switches to direct scalar-vs-scalar match
  (DoT reduction step already produces a scalar; the prototype side now
  matches without needing pole extraction).
- `set_activation` lift from scalar back to bivector (Spaces.py:9815) is
  retained — it lifts the snapped-scalar into a bivector activation
  for downstream consumers, exactly as today.

## Order-Typed Grammar

### Grammar syntax

Extend grammar categories with order annotations:

```text
NP0        category NP at order 0
VP1        category VP at order 1
NP*        category NP at rule-local order variable *
DET        bare category — syntactic sugar for DET0
```

**Bare categories are syntactic sugar for order 0.** This handles the
common case where the writer doesn't think about order — e.g.,
`NP = LOWER(DET, NP*)` says DET is order-0 (its known effect on a count
noun is to specify it via LOWER, which makes it contiguous). Bare-category
back-compat is preserved: legacy rules continue to parse, with all bare
categories binding to order 0.

The grammar parser splits each category token into:

```python
ParsedCategory(
    name: str,
    order_expr: OrderExpr,
)
```

where `OrderExpr` supports:

```text
constant integer:    0, 1, 2, ...        # NP0, VP1
variable:            *                    # NP*
variable plus delta: *+1                  # NP*+1
variable minus delta: *-1                 # NP*-1
(bare with no annotation = constant 0)
```

### Operation-derived order typing

The grammar and operation together determine the parent order.

For ordinary operations, order is preserved:

```text
NP* = conjunction(DET0, N*)      -> NP*    (binds * to N's order)
S*  = disjunction(NP*, VP*)      -> S*
```

For lift/lower, order changes:

```text
S = LIFT(NP*, VP1)
    => order(S) = * + 1
```

```text
NP = LOWER(DET, NP*)
    => order(NP) = * - 1
```

`LIFT` increments conceptual order. `LOWER` decrements conceptual order.
No other operation is allowed to change order unless explicitly added to
the operation registry as an order-changing operation.

### Rule metadata

Extend `TheGrammar.RuleDef` or the packed rule metadata:

```python
RuleOrderSignature(
    lhs_category:     str,
    lhs_order_expr:   OrderExpr,
    rhs_categories:   tuple[str, ...],
    rhs_order_exprs:  tuple[OrderExpr, ...],
    op_name:          str | None,
    order_delta:      int,  # +1 for LIFT, -1 for LOWER, 0 otherwise
)
```

**No static validation at grammar load.** Words are mapped to the
category codebook by *soft assignment* that participates in the parser's
superposition state — whether a given word fills an `NP3` vs `NP4` slot
is a runtime / superposition question, not a fact the grammar loader can
pre-empt. The rule's order signature is extracted statically; matching
against actual operand orders happens dynamically in STM REDUCE
(Phase 2), where the soft category distributions of operands are
compared against the rule's signature.

Bare categories (no annotation) bind to constant 0; orders must be
explicit constants (no Kleene variables — that machinery has been
removed).

## STM Shift/Reduce Runtime

STM is available behind ``parser_backend='stm'`` (or `'parallel'` with
the chart authoritative); the chart remains the default and the
fallback / oracle / diagnostic parser. The typed admissibility + STM
stack handle online syntax under the STM backend; the chart inside
pass continues to handle it under the chart backend.

### STM stack item metadata

The STM stack carries typed metadata alongside the vector payload. Each
live stack slot needs:

```python
StackItem:
    payload: tensor[D]
    category_id: int | category distribution
    order: int
    ref_id: int | -1
```

Stored as parallel tensors on the STM object:

```python
stm._buffer    # [B, cap, D]
stm._category  # [B, cap] or [B, cap, C]  (id or distribution)
stm._order     # [B, cap]
stm._ref_id    # [B, cap], -1 when unsnapped/unknown
stm._depth     # [B]
```

### SHIFT

`SHIFT` processes the next written word:

```text
written bytes -> order-0 word ref -> order-0 lexical meaning -> push STM
```

The pushed item must have:

```text
order = 0
ref_id = snapped order-0 symbol/reference id
category = lexical category or distribution
```

If a word has multiple order-0 lexical candidates, the parser may carry a
category/reference distribution or choose the best per the lexical scorer.
It must not jump directly to higher-order refs.

### REDUCE

`REDUCE` operates on the top stack items (normally top two):

```text
left  = STM[d - 2]
right = STM[d - 1]
```

Before scoring or applying a rule, build an admissibility mask:

```text
rule is admissible iff:
  rule.arity == operand count
  left.category  is compatible with rule.rhs_left
  right.category is compatible with rule.rhs_right
  left.order, right.order satisfy rule.rhs_order_exprs
  computed parent order is valid (>= 0)
```

Then score only admissible rules. **Inadmissible rules must be masked
before the reducer softmax**, not penalized after the fact.

After a rule is selected/applied:

```text
parent.payload  = op(left.payload, right.payload)
parent.category = rule.lhs_category
parent.order    = computed parent order
parent.ref_id   = snapped parent ref at parent.order, or -1 if unsnapped
```

The surviving STM slot receives parent metadata; the consumed slot is
cleared.

### LIFT and LOWER transitions

The only order-changing reduce actions.

```text
top stack: NP0, VP1
rule:      S = LIFT(NP*, VP1)
=> * binds to 0 from NP*, so S receives order 1.
   parent: S1
```

```text
top stack: DET0, NP2
rule:      NP = LOWER(DET, NP*)
=> * binds to 2 from NP*, so NP receives order 1.
   parent: NP1
```

If `LOWER` would produce a negative order, the rule is inadmissible for
that stack state.

## Lift/Lower Restricted-Candidate Inverse

`Ops._binary_op_recommend` gains per-operand candidate masks:

```python
Ops._binary_op_recommend(
    result, W, op_name, monotonic=True,
    left_rows:  long_tensor | None = None,   # ref_ids x1 may come from
    right_rows: long_tensor | None = None,   # ref_ids x2 may come from
)
```

When set, mask the augmented-codebook `≤ y` / `≥ y` filters with
`is_in(row, left_rows) | is_sentinel(row)` before `argmin` / `argmax`.
⊥ / ⊤ sentinels remain feasible regardless of restriction. When both are
`None`, behavior unchanged (`test/test_inverse_recommend.py` 13 tests
continue to pass byte-for-byte).

**Single mask per operand**: the recommender is generic lattice search;
typed-admissibility intersection is a grammar concern and belongs at the
call site. `Basis.lift(..., inverse=True)` / `Basis.lower(..., inverse=True)`
compute the intersection of category and order masks against the
artifact's indexes:

```python
sig = rule.order_signature
left_cat,  left_order  = sig.rhs_categories[0], resolve_order(sig.rhs_order_exprs[0], binding)
right_cat, right_order = sig.rhs_categories[1], resolve_order(sig.rhs_order_exprs[1], binding)
left_rows  = intersect(refs_by_category[left_cat],  refs_by_order[left_order])
right_rows = intersect(refs_by_category[right_cat], refs_by_order[right_order])
return Ops.disjunctionReverse(X1, X2, W,
                              left_rows=left_rows, right_rows=right_rows,
                              monotonic=monotonic, unit_ball=self.unit_ball)
```

`intersect(A, B)` is a small helper — `LongTensor` intersection by row index.
Both index slices are CSR-stored and small per call, so the intersection is
cheap.

## Automatic Symbol Building (in scope)

When the model operates in subsymbolic mode, CS (ConceptualSpace)
activations flow through pi/sigma without snapping to existing refs every
step. **Beneficial activations** — those that recur with consistent geometry
and would have reduced loss had they been representable as a symbol — are
promoted to new ref_ids in the codebook. This adds zero-order refs (new
primitives discovered in activation space) and higher-order refs
(compositions of existing refs that recur as a unit).

The "beneficial" criterion is **MDL-flavored loss reduction**: a candidate
is promoted iff its expected loss reduction exceeds its parameter cost.
Two layers, same principle:

- **Zero-order (new primitive)** — driven by **quantization error**.
  Maintain an EMA of `‖z_CS − z_q(z_CS)‖²` per activation region (online
  K-means / leader clustering over un-snapped CS activations). When a
  region's running QE exceeds threshold AND the cluster centroid is stable
  for N consecutive observations, promote to a new ref with
  `order = 0`. The QE drop is the loss reduction; the new ref's scalar
  parameter is the cost.

- **Higher-order (composition)** — driven by **PMI × frequency**.
  Maintain an EMA of adjacency counts for `(ref_a, ref_b)` pairs in STM.
  When `log[P(a,b) / (P(a)·P(b))] × count` exceeds threshold AND the
  composed activation has high QE against existing refs, promote with
  `order = max(operand_orders) + 1` *if and only if the firing rule is
  LIFT*, with `order = max(operand_orders) - 1` *if LOWER*, or
  `order = max(operand_orders)` for order-preserving rules. The promoted
  parent's category is the rule's LHS category.

Both layers funnel into the same `extend_artifact` entry point.

### Promotion path

- Append one row to `references` (Parameter grows via capacity-slack
  pattern); set its scalar to the promoted centroid.
- Append the order to `order` at the corresponding slot.
- Append a taxonomy node with `parent` = appropriate class node:
  - zero-order: parent = the POS class node whose existing children's
    centroids are nearest to the new centroid (deferred to symbol-learning
    logic to pick).
  - higher-order: parent = the rule's LHS class node.
- Append `(ref_id, new_ref)` to `refs_by_category[parent_category]` and
  to `refs_by_order[new_order]` CSR indexes; invalidate / re-materialize.
- No new `word_rows` for higher-order — composed refs have no single-word
  surface. Zero-order *may* add a word_row if the discovered cluster
  aligns with a surface byte sequence — symbol-learning's decision.

### Background literature (informational)

- **MDL / Minimum Description Length** (Rissanen; Grünwald): symbol
  added iff `L_data + L_model` drops. The framing this design follows.
- **Quantization-error reduction** (VQ-VAE / VQ-GAN dictionary-growth
  literature): drives the zero-order trigger.
- **PMI × frequency** (BPE / SentencePiece / NLP multi-word-expression
  literature): drives the higher-order trigger.
- **Predictive-coding residual** (Friston, Rao-Ballard): equivalent in
  spirit for residual-driven unit addition; not used here.
- **AIC / BIC**: classical model-selection penalties; MDL is the
  asymptotic equivalent and is what we instantiate.

Thresholds (QE threshold, stability `N`, PMI×frequency threshold) are
knobs of the symbol-learning module under
`<architecture><symbolLearning>` (or equivalent), not of the artifact
format.

### Parameter growth pattern

`references: nn.Parameter[V_ref_capacity]` cannot resize in-place without
breaking optimizer state. Match the existing `wv._vectors` growth pattern
at Spaces.py:2629-2648:

1. Reserve **capacity slack** at load:
   `V_ref_capacity = max(V_ref_initial * 2, 256)`. Parameter is allocated
   to capacity; only the first `V_ref_initial` rows are live.
2. Appends fill the next free slot in the slack region.
3. On capacity-exhaustion: reallocate to `capacity * 2`, copy old rows
   in, re-register on the owning Space's `nn.Module`, rebuild any
   optimizer state references via the same path `wv._vectors` uses.

`order`, `parent`, `children`, the CSR typed indexes are not Parameters;
they grow with normal `torch.cat` / list operations.

### Persistence

After a training run that exercised symbol-learning, the artifact is
re-written with the expanded contents. Format unchanged across writes;
only sizes grow.

## Retirements (deferred — incremental cutover)

Per the owner's 2026-05-20 design call, neither CKY nor
`category_codebook` is retired in this phase. The plan's earlier draft
that described their deletion is preserved below as the **future
direction**, but the Phase 2 closeout intentionally leaves both
operational.

### CKY — future direction (deferred)

Eventually `SyntacticLayer._compose_chart_cky` and supporting
chart-only machinery can be removed once the STM SHIFT/REDUCE loop
reaches inference parity and a defined acceptance gate passes. Until
then, chart remains the default parser, the oracle / fallback /
diagnostic parser, and the generation backup. The packed-rule-table
machinery (`TheGrammar._ensure_packed_table`, etc.) is shared infra
that both backends can use; no deletion required.

Diagnostic / debugging features that consume chart cells stay
chart-backed.

### `WordSpace.category_codebook` — future direction (deferred)

Future replacement is a clean `category_embedding: nn.Embedding[N_categories,
pos_dim]` for grammar labels. The owner's design call explicitly
distinguishes this from `refs_by_category` (the artifact's reference
membership index, which is appropriate for *candidate masks* but *not*
for rule-predictor input features — using reference-membership mean
vectors as syntactic-category representations would conflate semantics).

Migration order when this is undertaken:
1. Introduce `category_embedding` alongside the existing
   `category_codebook`.
2. Port consumers one at a time, validating no regression.
3. Once all four shared consumers (`category_lookup`, `pos_lookup`,
   `predict_rule`, `category_stack`) read from `category_embedding`,
   delete `category_codebook`.

Stays operational until that work lands.

## Status (as of session ending 2026-05-20)

**323 tests passing across all touched suites; zero regressions.**

- **Phase 1 — DONE.** Order-typed grammar parser + artifact writer +
  symbol-learning append (`extend_artifact`) all shipped with TDD
  coverage. ~46 tests across `test_grammar_order_typing.py` (with the
  static-validation tests retired per soft-assignment) and
  `test_knowledge_artifact_writer.py`.
- **Phase 3 (typed restricted-candidate inverse) — DONE.**
  - `Ops._binary_op_recommend` gains `left_rows` / `right_rows`
    kwargs ([bin/Layers.py:7857](../../bin/Layers.py)).
    ⊥ / ⊤ sentinels remain feasible regardless of restriction.
    Default `None` matches unrestricted byte-for-byte (the 13
    union/intersection recommender tests at
    `test/test_inverse_recommend.py` continue to pass unchanged).
  - Kwarg forwarding through `Ops.conjunctionReverse` /
    `Ops.disjunctionReverse` / `Ops._binary_op_inverse_impl`
    ([bin/Layers.py](../../bin/Layers.py)) and
    `Basis.lift(inverse=True)` / `Basis.lower(inverse=True)`
    ([bin/Spaces.py](../../bin/Spaces.py)) — full chain to the
    typed-mask call site.
  - End-to-end test demonstrating the typed-mask flow: artifact →
    `load_knowledge_view` → `refs_by_category ∩ refs_by_order` →
    `disjunctionReverse(..., left_rows=...)`. Verifies the mask
    actually restricts x1 to the requested subset, and that empty
    intersections fall back to sentinels.
  - **11 new tests** in `test/test_restricted_candidate_inverse.py`.
- **Phase 3 — remaining**: automatic symbol-building hooks (MDL / QE /
  PMI triggers that call ``extend_artifact``). This requires the
  parser to actively observe activations during training, which
  depends on Phase 2-integration (STM-as-parser activation).
  Deferred.

- **Phase 2 (primitives + STM data structure) — DONE.**
  - `KnowledgeView` facade + `load_knowledge_view` bridge.
  - Per-Space attach: `Space.attach_knowledge` base + `SymbolicSpace`
    override (trainable scalar `references: nn.Parameter` with
    capacity slack + `order` long buffer) + `PerceptualSpace`
    override (`wv.ref_ids`) + mirrored `WordSpace.attach_knowledge`.
  - Typed-admissibility primitives: `is_rule_admissible` predicate,
    `admissibility_mask` (BoolTensor over a rule list),
    `mask_logits` (sets inadmissible logits to -inf for pre-softmax).
  - **`TypedStack`** (`bin/typed_stack.py`) — STM stack data
    structure with `_buffer`, `_category`, `_order`, `_ref_id`,
    `_depth` parallel state plus `push` / `pop` / `top` /
    `reduce_admissibility` methods.
  - End-to-end integration test (`test_phase2_end_to_end.py`):
    artifact → save → load → 3-Space attach → admissibility mask
    over real loaded rule signatures → mask_logits → expected
    post-softmax distribution. Plus an `extend_artifact` →
    re-attach round-trip showing capacity-slack growth preserves
    `nn.Parameter` identity.
  - **~84 new Phase-2 tests across** `test_knowledge_view.py`,
    `test_word_space_attach_knowledge.py`, `test_phase2_end_to_end.py`,
    `test_typed_stack.py`. **276 tests passing across all touched
    suites; zero regressions.**

### Phase 2 — closeout (incremental cutover shipped 2026-05-20)

Per the owner's incremental-cutover guidance (no chart deletion, STM
behind a switch, category_codebook stays, symbol learning disabled by
default, acceptance = no regression + primitives), Phase 2 closeout
shipped these six steps:

1. **`parserBackend` config knob + dispatch shell.**
   `WordSpace.__init__` reads `<parserBackend>` from XML (default
   `chart`); validates against `{chart, stm, parallel}`. `compose()`
   and `generate()` dispatch on `self.parser_backend`. Unknown values
   raise `ValueError`. **Tests:** `test_parser_backend.py` — 6.
2. **`ConceptualSpace.stm_typed`** — a `TypedStack` allocated
   alongside the legacy `self.stm` (`ShortTermMemory`). Sized to the
   same capacity + dim. `_init_typed_stm(batch, max_depth, dim)`
   helper for re-allocation. **Tests:** `test_conceptual_stm_typed.py`
   — 3.
3. **`STMDriver` + `RuleScorer`** at `bin/stm_driver.py`. New small
   module, intentionally NOT a port of the chart's `_rule_embed`.
   Driver coordinates a `TypedStack` + rule signatures + scorer.
   `shift()` pushes a frame; `reduce_step()` builds admissibility mask
   via `TypedStack.reduce_admissibility`, applies `mask_logits` to the
   scorer's output, picks the highest-scoring admissible rule.
   Inadmissible rules with higher raw logits do not win.
   **Tests:** `test_stm_driver.py` — 8.
4. **WordSpace ↔ STM driver wiring.** `_init_stm_driver` lazily
   constructs the driver from the attached `KnowledgeView`'s rule
   signatures + `conceptualSpace.stm_typed`. `_compose_stm` /
   `_generate_stm` ship as wiring that lazily builds the driver and
   returns an empty rules dict. Driving SHIFT/REDUCE over actual
   `input_vectors` is follow-up (see "deferred for next phase").
   **Tests:** `test_stm_compose_wiring.py` — 7.
5. **`parallel` backend.** Constructs the STM driver before falling
   through to the chart. Chart is authoritative (returns
   `current_rules` / `generate_rules`); STM driver presence enables
   side-channel inspection and future training-stats collection.
   **Tests:** `test_parallel_backend.py` — 3.
6. **Symbol-learning statistics scaffold** at `bin/symbol_learning.py`.
   `SymbolLearningStats` — disabled-by-default detached-stats
   accumulator with two hook points: zero-order/QE via
   `observe_qe(activation, snapped)` and higher-order/PMI via
   `observe_reduce(left_ref, right_ref, parent_ref)`. `flush()`
   returns a snapshot and resets. Never mutates the artifact during
   forward — codebook mutations are caller's call at an explicit
   boundary. XML reader honors `<architecture><symbolLearning
   enabled="true"/>`. **Tests:** `test_symbol_learning_stats.py` —
   9.

**Acceptance checklist (all met, per the owner's decision):**

- ✓ Current tests still pass with default `parserBackend=chart`
  (323 across all touched suites, zero regressions).
- ✓ STM stack carries category/order/ref metadata (`TypedStack`).
- ✓ STM reduce masks actions by typed admissibility
  (`STMDriver.reduce_step` via `reduce_admissibility` +
  `mask_logits`).
- ✓ Grammar parses order annotations (`Grammar._parse_category` /
  `_rule_order_signature`).
- ✓ Reverse candidate masks can use category/order
  (`Ops._binary_op_recommend(left_rows, right_rows)`, forwarded
  through `conjunctionReverse` / `disjunctionReverse` /
  `Basis.lift` / `Basis.lower`).
- ✓ CKY still available (chart untouched; `parserBackend=chart`
  default; `parallel` mode runs it authoritatively).

### Phase 2 — deferred for next phase

Per the owner's design call, the following are explicitly out of scope
for this Phase 2 closeout and tracked here as named follow-up work:

1. **STM SHIFT/REDUCE loop over real input_vectors.** Today
   `_compose_stm` lazily constructs the driver and returns an empty
   rules dict. A real STM-driven parse would: tokenize input_vectors
   to (payload, category, ref_id) tuples (where order = 0 per the
   order-0 lexical entry contract); shift each token; run REDUCE
   until a single S-rooted frame remains; emit per-tier
   SyntacticLayer rule selections compatible with the existing
   `current_rules` consumer.
2. **`WordSpace.category_codebook` retirement.** Will become
   `category_embedding: nn.Embedding[N_categories, pos_dim]` — a
   learned grammar-label embedding (semantically distinct from
   `refs_by_category`, which carries reference membership and stays
   the candidate-mask source). 19 references to migrate; live
   consumers (rule predictor, `category_stack`) need careful
   rewiring. Estimated 4-6 hours.
3. **CKY retirement.** Stays as oracle / fallback / diagnostic /
   generation backup. Retirement waits until the STM SHIFT/REDUCE
   loop reaches parity on inference and there's a defined acceptance
   gate. ~700-1700 lines.
4. **Symbol-learning policy + `extend_artifact` integration.** Today
   `SymbolLearningStats` is statistics-only. A future symbol-learning
   policy will read flushed snapshots, apply MDL / QE-threshold /
   PMI × frequency criteria, and call `extend_artifact` at the
   explicit flush boundary. Hook points are in place; trigger logic
   is not.
5. **`bivector retirement`** (the narrow scope: only
   `SymbolicSpace.subspace.what.W: [V_ref, 2] → [V_ref]`). On hold
   while the rest of the additive-references path stabilizes; the
   trainable `SymbolicSpace.references` Parameter exists alongside
   the existing bivector codebook and downstream consumers can
   migrate at their own pace.

## Staging (as shipped)

Three phases. The actual cutover replaced the originally-planned
"Phase 2 = deletion + STM-as-only-parser" with an incremental,
backward-compatible plan per the owner's 2026-05-20 design call.

### Phase 1 — DONE — Artifact writer + order-typed grammar parsing

- `bin/embed.py`: writer emits the knowledge section
  (`word_table`, `reference_codebook`, `typed_indexes`, `taxonomy`,
  `grammar.rule_order_signatures`) as a section of the existing unified
  `.kv` artifact format (chosen over a separate sidecar `.pt` to match
  the existing `save_artifact` / `load_artifact` shape).
- `bin/Language.py:Grammar`: `_parse_category` + `_rule_order_signature`
  for order-typed category parsing. Kleene removed per the
  explicit-constants-only design call; no static validation per the
  soft-assignment design call.
- `extend_artifact(path, new_refs)` runtime append.

### Phase 2 — DONE — Loaders + STM primitives + opt-in STM backend

- `bin/Spaces.py`: loaders attach a `KnowledgeView` to each Space.
  `SymbolicSpace.attach_knowledge` creates a trainable
  `references: nn.Parameter[V_ref_capacity]` plus an `order` long
  buffer. `PerceptualSpace.attach_knowledge` stamps `wv.ref_ids`.
  `Space.attach_knowledge` provides the base.
- `bin/typed_stack.py`: `TypedStack` with parallel `_buffer`,
  `_category`, `_order`, `_ref_id`, `_depth` tensors;
  `reduce_admissibility(b, rule_signatures)` builds the
  typed-admissibility mask. `ConceptualSpace.stm_typed` allocates one
  alongside the existing `self.stm` (`ShortTermMemory`).
- `bin/stm_driver.py`: `STMDriver` + `RuleScorer` — new small module
  that masks reduce actions via `mask_logits(scorer(...), mask)`.
- `bin/Language.py:WordSpace`: `parserBackend` config knob
  (`chart`/`stm`/`parallel`, default `chart`). `compose()` /
  `generate()` dispatch internally; `_compose_stm` / `_generate_stm`
  lazily construct the driver. `parallel` mode runs both with chart
  authoritative. `attach_knowledge` mirrored on WordSpace.
- `bin/symbol_learning.py`: disabled-by-default `SymbolLearningStats`
  scaffold with detached QE / PMI hooks; flush boundary explicit;
  XML reader for `<architecture><symbolLearning enabled="..."/>`.
- **Not done (deferred):** STM SHIFT/REDUCE over real
  `input_vectors`, `category_codebook` retirement, CKY retirement,
  bivector retirement, symbol-learning policy + `extend_artifact`
  promotion. See "Phase 2 — deferred for next phase" above.

### Phase 3 — DONE — Typed restricted-candidate inverse + symbol-learning hooks (scaffold)

- `bin/Layers.py:Ops._binary_op_recommend`: `left_rows` / `right_rows`
  kwargs added. ⊥ / ⊤ sentinels remain feasible regardless of
  restriction. Default behavior unchanged byte-for-byte.
- Kwarg forwarding through `Ops.conjunctionReverse` /
  `Ops.disjunctionReverse` / `Ops._binary_op_inverse_impl`.
- `bin/Spaces.py:Basis.lift(inverse=True)` /
  `Basis.lower(inverse=True)`: forward `left_rows` / `right_rows`
  through to the recommender. Call-site computes
  `refs_by_category[cat] ∩ refs_by_order[k]` from attached
  `KnowledgeView`.
- Symbol-learning **hooks** in place (Phase 2 step 6); symbol-learning
  **policy** + `extend_artifact` integration deferred.

## Tests

### Phase 1

`test/test_knowledge_artifact_writer.py`:
- Build artifact from a fixed grammar + curated POS lexicon + 10-word
  corpus. Verify CSR ragged keys, ref_ids, reference_codebook (order
  intrinsic per row), typed_indexes (`refs_by_order`,
  `refs_by_category`), taxonomy parent/children.
- Round-trip: write, load, compare bit-for-bit (mod nn.Parameter wrap).
- `extend_artifact` appends rows and re-writes index slices; verify
  consistency after multiple appends.
- Capacity-slack: allocate at `V_ref_initial`, append to exhaustion,
  verify reallocation preserves Parameter identity / gradient flow.

`test/test_grammar_order_typing.py`:
- Parse `S4 = lift(NP3, VP1)` → assert RHS categories `(NP, VP)`,
  RHS orders `(3, 1)`, `order_delta == +1`.
- Parse `NP3 = lower(DET, NP4)` → assert RHS categories `(DET, NP)`,
  RHS orders `(0, 4)` (DET bare → 0), `order_delta == -1`.
- Parse ordinary rules → assert `order_delta == 0`.
- Kleene tokens (`NP*` / `NP*+1` / `NP*-1`) raise `ValueError`.
- No static validation: signatures are extracted but never rejected at
  load. Admissibility is a runtime concern (Phase 2 STM REDUCE).

### Phase 2 (as shipped)

`test/test_knowledge_view.py` (21):
- `KnowledgeView` accessors (refs / orders live-slice;
  refs_by_category; refs_by_order; rule_order_signatures;
  taxonomy_names; ref_id_for; category_of_ref; order_of_ref).
- `load_knowledge_view` bridge round-trips.
- Hard-category typed admissibility predicate covers binary /
  unary / mismatch / arity-mismatch cases.
- `admissibility_mask` + `mask_logits` produce expected
  post-softmax probability distributions.

`test/test_word_space_attach_knowledge.py` (14):
- `WordSpace.attach_knowledge` + `.knowledge` property.
- Inheritance: PerceptualSpace / SymbolicSpace inherit
  `attach_knowledge` from `Space`.
- `SymbolicSpace.attach_knowledge` creates `references: nn.Parameter`
  (registered in named_parameters) + `order` long buffer (registered
  in named_buffers); re-attach updates in place when capacity
  unchanged.
- `PerceptualSpace.attach_knowledge` stamps `wv.ref_ids` from the
  artifact.

`test/test_typed_stack.py` (11):
- TypedStack construct / push / pop / top / per-row isolation /
  overflow / underflow.
- `reduce_admissibility` returns the expected mask for binary and
  unary stack-top states.
- Full chain test: `mask_logits(scorer_output, mask) → softmax`
  concentrates probability on admissible rules.

`test/test_conceptual_stm_typed.py` (3):
- `ConceptualSpace._init_typed_stm` allocates a TypedStack.
- `stm_typed` property defaults to None before init.
- Re-init replaces.

`test/test_parser_backend.py` (6):
- Default `parser_backend == 'chart'`.
- Set to `'stm'` / `'parallel'` is accepted; with no knowledge,
  raises RuntimeError (clear message).
- Unknown backends raise ValueError.

`test/test_stm_driver.py` (8):
- `RuleScorer` construct + forward (binary and unary).
- `STMDriver` construct + shift + reduce_step picks admissible rule
  for binary / unary stacks.
- `reduce_step` raises with clear error when nothing is admissible.
- `mask_logits` ensures admissible-only pick survives an
  inadmissible-but-higher-raw-logit competitor.

`test/test_stm_compose_wiring.py` (7):
- `parser_backend='stm'` + no knowledge → RuntimeError.
- + no `conceptualSpace.stm_typed` → RuntimeError.
- With both attached → `stm_driver` constructed; rule_signatures and
  scorer dim match attached view / stm_typed.
- Subsequent compose() calls reuse the same driver.
- generate() under stm backend follows the same path.

`test/test_parallel_backend.py` (3):
- `parser_backend='parallel'` constructs stm_driver BEFORE the chart
  runs (chart errors on bare instances; we still see the driver).
- generate() under parallel mode also constructs the driver.

`test/test_symbol_learning_stats.py` (9):
- Default `enabled=False`; observe_qe / observe_reduce no-op.
- `enabled=True` accumulates squared QE; stats are detached (Python
  floats, no autograd tape).
- Pair counts accumulate for PMI.
- `flush()` returns snapshot + resets state.
- XML config reader honors `<symbolLearning enabled="true"/>`.

`test/test_phase2_end_to_end.py` (5):
- Full pipeline: write artifact → load → attach to 3 Spaces → query
  via attached KnowledgeView → admissibility mask → mask_logits →
  softmax → extend_artifact → re-load → re-attach with capacity-slack
  growth preserving Parameter identity.

### Phase 3 (as shipped)

`test/test_restricted_candidate_inverse.py` (11):
- Default kwargs match unrestricted baseline byte-for-byte.
- `left_rows` / `right_rows` restrict the recommender's candidate
  set for x1 / x2 respectively.
- Empty rows + degenerate-y → sentinel fallback.
- Intersection branch respects restrictions.
- Public wrappers (`conjunctionReverse` / `disjunctionReverse`)
  forward kwargs.
- End-to-end: build artifact → load view → compute
  `refs_by_category ∩ refs_by_order` → call `disjunctionReverse` →
  verify restriction respected; empty intersection falls back to
  sentinels.

### Deferred tests (when their target work lands)

The originally-planned `test_artifact_load_into_spaces.py`,
`test_bivector_retirement.py`, `test_cky_retirement.py`,
`test_category_codebook_retirement.py`,
`test_lift_lower_restricted_inverse.py`, and
`test_symbol_learning_promotion.py` were scoped against deletion /
retirement work that's now explicitly deferred. They become relevant
when their target work is undertaken.

## Verification

```bash
cd "/Users/arogers/Library/Mobile Documents/com~apple~CloudDocs/bits/projects/WikiOracle/basicmodel"
./.venv/bin/python -m pytest \
    test/test_knowledge_artifact_writer.py \
    test/test_grammar_order_typing.py \
    test/test_knowledge_view.py \
    test/test_word_space_attach_knowledge.py \
    test/test_typed_stack.py \
    test/test_conceptual_stm_typed.py \
    test/test_parser_backend.py \
    test/test_stm_driver.py \
    test/test_stm_compose_wiring.py \
    test/test_parallel_backend.py \
    test/test_symbol_learning_stats.py \
    test/test_phase2_end_to_end.py \
    test/test_restricted_candidate_inverse.py \
    test/test_inverse_recommend.py \
    test/test_ops_lift_lower.py \
    test/test_grammar_binary_ops.py \
    test/test_usegrammar_tristate.py \
    test/test_invertibility.py \
    test/test_grammar_start_rule.py \
    test/test_embed.py -q
```

**As-shipped result: 323 passing, zero regressions.**

The 13 union/intersection recommender tests at
`test/test_inverse_recommend.py` continue to pass with the new
`left_rows` / `right_rows` kwargs defaulting to `None` (byte-for-byte
equivalent default behavior). The chart's CKY path is untouched, so
all chart-dependent existing tests continue to pass under the default
`parser_backend='chart'`.

## Critical files

- [bin/embed.py](../../bin/embed.py) — artifact writer; knowledge
  section builders; `extend_artifact` runtime append; `KnowledgeView`
  facade; `is_rule_admissible` / `admissibility_mask` / `mask_logits`.
- [bin/typed_stack.py](../../bin/typed_stack.py) — STM stack data
  structure with parallel metadata; `reduce_admissibility`.
- [bin/stm_driver.py](../../bin/stm_driver.py) — STM shift/reduce
  driver + small RuleScorer.
- [bin/symbol_learning.py](../../bin/symbol_learning.py) —
  disabled-by-default detached-stats scaffold (QE + PMI hooks).
- [bin/Spaces.py](../../bin/Spaces.py) — `Space.attach_knowledge`
  base + per-class overrides (PerceptualSpace `wv.ref_ids`,
  SymbolicSpace `references` Parameter + `order` buffer);
  `ConceptualSpace._init_typed_stm` allocates `stm_typed`;
  `Basis.lift` / `Basis.lower` inverse paths with category∩order
  masks.
- [bin/Language.py](../../bin/Language.py) — `Grammar` order syntax
  + `RuleOrderSignature`; `WordSpace.attach_knowledge`,
  `parser_backend` dispatch, `_compose_stm` / `_generate_stm`,
  STM driver wiring.
- [bin/Layers.py](../../bin/Layers.py) — `Ops._binary_op_recommend`
  `left_rows` / `right_rows` kwargs + kwarg forwarding.
- [bin/Mereology.py](../../bin/Mereology.py) — unchanged.

## Integration With Existing Architecture

The current codebase already has useful pieces:

- `conceptualOrder` controls the number of conceptual stages.
- `conceptualSpaces` / `symbolicSpaces` are per-order `ModuleList`s.
- `work.recur_pass` selects the active conceptual pass.
- `WordEncoding` already has an `order` field.
- Current STM reducer performs bounded top-2 reductions.

So Phase 2 is mainly **semantic + type-tracking** changes plus an
opt-in STM driver behind a config switch. CKY and `category_codebook`
are not deleted; they remain operational, with retirement scoped as
named follow-up work.

## Assumptions

- References are ramsified and have intrinsic conceptual order.
- Words are ramsified but enter via direct written-word recognition at
  order 0.
- The grammar is the source of order typing.
- LIFT increments conceptual order by one; LOWER decrements by one.
- STM shift/reduce is available as an opt-in parser backend
  (`parser_backend='stm'` / `'parallel'`); chart stays as the default,
  the oracle / fallback / diagnostic parser, and the generation backup.
- `category_codebook` stays operational for now; future retirement
  introduces a `category_embedding: nn.Embedding[N_categories, pos_dim]`
  for grammar labels (distinct from `refs_by_category`, which is for
  reference candidate masks).
- Symbol learning is disabled by default and never mutates the
  artifact inside autograd forward; mutations happen at explicit flush
  boundaries.
- Bare grammar categories are syntactic sugar for order 0.

## Open questions deferred to implementation

- **Polysemy disambiguation**: schema supports multiple `word_rows` with
  identical `keys` bytes pointing at different `ref_ids` under different
  POS. The lookup path needs a disambiguator (POS context) at SHIFT.
  Deferred to STM SHIFT implementation.
- **Zero-order promotion's parent-class choice**: when a new zero-order
  cluster's centroid lands between two POS class centroids, pick by
  nearest distance? Or by additional features (the surface byte
  sequences that triggered the cluster)? Deferred to symbol-learning
  logic.
- **CKY-using diagnostic / debug outputs**: anything in the codebase
  that printed chart cells for debugging needs STM-equivalent output or
  removal. Enumerated during Phase 2.
