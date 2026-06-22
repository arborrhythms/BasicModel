# Language

This document is synchronized to the code as of 2026-06-02. The source
of truth is `bin/Language.py`, `bin/embed.py`, and (for the perceptual
analyzer) `bin/perceptual_analyzer.py`. (The earlier
`bin/typed_stack.py`, `bin/stm_driver.py`, and `bin/parse_state.py`
modules were retired in the 2026-05-21 / 2026-05-29 refactors; their
functionality folded into `bin/Layers.py` and `bin/Language.py`.)

> **2026-06-02 deltas (subsymbolic analyzer + terminal emitter).** Plan:
> [doc/old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md](old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md).
>
> - **PS/SS grammar sections.** A `.grammar` file may nest its
>   `<compose>`/`<generate>` under `<PartSpace>` and
>   `<WholeSpace>`. `Grammar.configure` parses them into separate rule
>   tables: `ps_rules` (tier `P`, read by the PS analyzer) and the
>   canonical symbolic `rules` (`ws_rules`). A bare `<compose>`/`<generate>`
>   file loads as `<WholeSpace>` (backward-compat). The legacy `.cfg`
>   loader is gone.
> - **Grammar rewrite.** `*_MARK` categories and the copy/swap MARKER
>   helper rules are deleted; surface markers are learned and owned by the
>   operator. Each operator-argument position is its own category
>   (`CONJ_L45`/`CONJ_R45`, ...). copy/swap are retired from the symbolic
>   grammar (kept only as the T5 elision surface policies).
> - **SurfaceSchema + absorb/emit** (`bin/Layers.py`). Five universal
>   templates T1-T5 declare each operator's marker slot + order; T4
>   `BINARY_JUXTAPOSE` is the default. `GrammarLayer.absorb` binds a
>   co-occurring marker to the operator (many-to-one); `emit` replays it
>   from recorded route metadata, never the lossy `generate()`.
> - **Operators in the SS codebook, not the STM idea space** (amends plan
>   decision #2). `WholeSpace.insert_operations(grammar)` registers each
>   operation in a dedicated operator codebook on WholeSpace
>   (`_operation_vectors` / `_operation_positions`), separate from the
>   `subspace.what` whole-percept codebook so the percept / idea / `.where` position
>   namespace is untouched; it is wired into `SymbolicSubSpace.__init__` so every
>   built model's operator-prefixed parse-tree nodes are codebook-resolvable. The STM idea space holds only combined meanings --
>   the operator says *how* meanings combine, contributing none of its own.
>   The rule-id stays in `.where` (its presence marks a slot as a *computed*
>   idea, which is by definition not a codebook vector).
> - **ObjectSubSpace** (`bin/Language.py`) -- the PS-meronymic carrier
>   analogue of `SymbolicSubSpace`: span buffers + parent/child links + route
>   ids/scores + the marker-route replay fields
>   (`_marker_ps_id`/`_marker_span`/`_order_bit`/`_marker_position`).
> - **Perceptual analyzer** (`bin/perceptual_analyzer.py`). `EndpointSumWhere`
>   is the invertible span key $where = phase(start) + phase(end)$: the
>   angle decodes the span center, the magnitude the span length.
>   `MeronymicAnalyzer` analyzes a surface into terminals (compatibility
>   mode reuses the word/byte tokenizer as the `boundary` op, so it matches
>   the current lexer), writes durable spans to an `ObjectSubSpace`, exposes
>   a fixed-capacity terminal-stream view, and reverse-synthesizes surface
>   (`synthesize` exact replay; `synthesize_tree` from an operator-prefixed
>   tree with `emit`). `soft_operator_compose` + `WholeSpace
>   .operator_superposition` apply a soft operator distribution over the SS
>   operation codebook (one-hot $\to$ the typed grammar; spread $\to$ the
>   superposition that discriminates `A AND B` from `A OR B`).
> - **PS-to-SS binding.** `WholeSpace.resolve_ps_terminal(ps_id)` emits
>   `null_sem()` before a binding exists, counts exposures, and promotes a
>   repeated terminal into a fresh SS row.

> **2026-05-29 deltas:**
>
> - `unreduce()` passes the tier-local Basis (Codebook) to binary
>   GrammarLayer reverses as `basis=tier_basis` (replacing the prior
>   raw-`W` form). `UnionLayer.reverse` / `IntersectionLayer.reverse`
>   **and now `ConjunctionLayer.reverse` / `DisjunctionLayer.reverse`**
>   extract `W = basis.getW()` internally and dispatch to
>   `Ops.disjunctionReverse` / `Ops.conjunctionReverse` — the codebook
>   recommender recovers the operand pair exactly on a discrete
>   vocabulary (the serial XOR reconstruction path); without a basis they
>   keep the lossy `(parent, parent)` fallback. The genuinely
>   non-invertible predicate folds (`isEqual` / `isPart` / `exist`)
>   declare `invertible = False` and are not invertible by design (a
>   truth/predicate value does not retain its operands). Layers that
>   don't accept the `basis` kwarg yet are handled by a `TypeError`
>   fallback. No back-ref is stored on the layer.
> - `MetaLayer` was renamed to `SymbolizeLayer` (no semantic change).
> - Word-mode parse appends a `\x00` null sentinel after the words
>   slab for explicit end-of-sequence on the forward path.
> - See [doc/old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](old/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

## Current Parser Surface

`SymbolSpace.compose()` and `SymbolSpace.generate()` are the public parser
entry points. `SymbolSpace.parser_backend` selects the implementation:

| backend | status | notes |
|---|---|---|
| `chart` | default | Compatibility path. Runs the existing chart / Viterbi machinery. |
| `stm` | active | Shift/reduce over `ConceptualSpace.stm_typed`; requires an attached `KnowledgeView`. |
| `parallel` | bridge | Builds the STM driver, then runs the chart path authoritatively. |

`SymbolSpace.routerKind` is separate from `parserBackend` and only affects
the chart backend:

| routerKind | status |
|---|---|
| `chart` | Default chart scorer / inside pass. |
| `signal` | Signal-router grammar path used by configs such as `XOR_grammar.xml`. |

The chart parser has not been deleted. It remains the default because
legacy configs can instantiate without a knowledge artifact, while STM
cannot.

## Retired XML Knobs

These XML fields are no longer read by the runtime and should not appear
in `data/*.xml`:

- `SymbolSpace.useGrammar`
- `SymbolSpace.chartCompose`
- `SymbolSpace.softChartCompose`
- `bivectorOutput`

Grammar mode is derived from the loaded grammar: default-only unary
`pi` / `sigma` rules derive `useGrammar == "none"` internally; any
non-default operator rule derives `useGrammar == "all"`.

> **SS-analysis vs CS-execution.** `SymbolicSubSpace.compose` is the
> SS-side *analysis* stage (it selects the per-tier hard rule dict
> `current_rules`); the CS-side *execution* (applying lift / lower /
> union / intersection / swap / quantize / not to the concept tensors)
> runs in `ConceptualSpace.forward` and the per-tier `SyntacticLayer`
> cursors. This split is a clean code boundary **only on the
> default-only path**: on the full-router path
> `LanguageLayer.compose` does both selection and tensor reduction, and
> the per-tier cursors are deliberately bypassed
> (`not _grammar_is_default_only`). See
> [STM.md Section 5](STM.md#5-routing-parser-ws-analysis-vs-cs-execution)
> for the accurate, audited account.

## Grammar

`TheGrammar` is the singleton `Grammar` instance. Rules are loaded from
XML `<SymbolSpace><language><grammar>` blocks or from a configured grammar
CFG. A `RuleDef` stores:

```text
(tier, canonical, arity, method_name, lhs, rhs_symbols)
```

The grammar accepts explicit conceptual-order suffixes:

```text
NP3
NP4
VP1
MP1
S4
S5
```

The intended current style is explicit ordered rules, for example:

```text
S4 = lift(NP3, VP1)
S5 = lift(NP4, MP1)
```

Here all NPs share base category `NP`, but participate at different
conceptual orders. The parser carries stack frames as `(category, order)`,
so `NP3` and `NP4` both have `category == "NP"` but orders `3` and `4`.

Kleene order forms such as `NP*` are still parsed for compatibility, but
they are not required by the current grammar style.

## Knowledge Artifacts

`embed.build_knowledge_section(grammar)` creates the parser knowledge
section:

- `taxonomy`: base category refs plus explicit ordered refs.
- `reference_codebook`: scalar prototypes and `order` per ref.
- `typed_indexes`: `refs_by_category` and `refs_by_order`.
- `grammar.rule_order_signatures`: serialized rule typing.

For explicit ordered grammar, the taxonomy stores ordered refs as
children of their base category:

```text
NP
|-- NP3
`-- NP4
```

`KnowledgeView.category_of_ref(ref_id)` returns the base category
(`NP`), while `KnowledgeView.order_of_ref(ref_id)` returns the
conceptual order (`3`, `4`, etc.).

`SymbolSpace.category_codebook` has been retired. The live category
embedding is:

```python
SymbolSpace.category_embedding: nn.Embedding
```

## STM Shift/Reduce

The STM backend uses:

- `ConceptualSpace.stm_typed`: a `TypedStack` carrying payload,
  category, order, and ref id.
- `STMDriver`: shift/reduce coordinator.
- `RuleScorer`: MLP over top stack payloads.
- `embed.admissibility_mask`: hard category/order mask over grammar
  rule signatures.

SHIFT snaps each input vector to the nearest live reference, then pushes:

```text
payload, category, order, ref_id
```

REDUCE computes a typed admissibility mask, applies it to rule logits,
softmaxes over admissible rules, chooses the argmax for the current
greedy path, and records the chosen score/probability.

Typed admissibility is exact for explicit rules:

```text
S4 = lift(NP3, VP1)
```

fires only when the stack top matches:

```text
left:  category NP, order 3
right: category VP, order 1
```

## Syntax And SVO

`BasicModel.write_syntax_tree()` (in `bin/Models.py`) emits an XML
syntax-tree dump per batch row when configured with a syntax output
path. It reads the chart's per-row derivation trace and the symbolic
codebook's per-atom category tags; see the function docstring for the
exact format. SVO extraction is performed by the STM path on the
canonical subject lift over a transitive verb-phrase derivation; the
object role is the NP operand inside that verb-phrase derivation, not a
separate semantic category.

## Role-Collapsed Grammar and the Operator Codebook

The role-collapsed grammar (`data/role_collapsed.grammar`) replaces
part-of-speech categories with *operator roles*. Instead of a fixed
`NP` / `VP` / `S` taxonomy, each operator contributes its own argument
and result roles, named `<op>_I1`, `<op>_I2` (inputs) and `<op>_O1`
(output). The rule `isEqual` therefore exposes `isEqual_I1`,
`isEqual_I2`, `isEqual_O1`, and the grammatical "category" of a span is
just the set of operator roles it can fill. Dimensionality is recovered
from participation (`bin/participation.py`) rather than declared up
front: symbols that fill the same roles cluster into the same category.

Starts are scoped per space (`_configure_starts`):

- `PartSpace.start` is the universal whole-input role `U` --- the
  analyzer begins from the entire surface and decomposes it.
- `WholeSpace.start` is the set of operator output roles
  (`isEqual_O1`, `isPart_O1`, `exist_O1`, ...) --- parsing begins from
  what an operator can *produce*.

The **operator codebook** is a second codebook on `WholeSpace`,
separate from the whole-percept codebook:

- `_operation_vectors`: operator name $\to$ identity vector.
- `_operation_positions`: operator name $\to$ codebook position.

It stores one prototype per operator (`isEqual`, `isPart`, `exist`,
`conjunction`, `disjunction`, ...) and is kept CPU-explicit so the
host-side identity lookup never mixes with an ambient MPS / CUDA default
device.

### Soft Operator Superposition

`operator_superposition(query_vec)` is a softmax over the cosine
similarity between a query and every operator vector.
`soft_operator_compose(dist, left, right)` is the distribution-weighted
sum of each operator's `compose`, evaluated per operator *arity* ---
unary operators such as `exist` are called as `compose(a)`, binary
operators as `compose(a, b)`. A one-hot distribution reduces exactly to
the typed grammar; a spread distribution superposes operators, and that
superposition is the mechanism that discriminates `A AND B` from
`A OR B` while a single layer is chosen.

`shape_operators(examples, op_names, steps, lr, seed)` trains the
operator vectors by backpropagating an MSE between the superposed
prediction and the supervised result, writing the shaped vectors back
into `_operation_vectors`. The op set may span the whole grammar, so the
shaper filters to layers of arity 1 or 2 and dispatches `compose` by
arity.

**Status (D1 gate --- met).** Role-collapse does not swap declared POS for
another single-label POS system; it replaces declared shared categories
with operator-local participation categories that a word may fill several
of (overlaps are expected). The D1 gate is therefore not a single-label POS
recovery test --- it asks whether those participation patterns are
*structured enough to drive a learned collapse* into the smaller
mutually-exclusive category set the live parser needs. They are:
`participation.learned_collapse` proposes merges by participation similarity
and accepts only those that keep every grammar rule distinguishable
(`collapse_conflicts == 0`); on the transitional `complete.grammar` it
compacts 43 context-unique symbols into 14 mutually-exclusive categories
with zero parser conflicts (`test/test_d1_pos_recovery_gate.py`). The exact
substitutability congruence is trivial there (every symbol is
context-unique), so "recovers the grammar" means the parser's rule
decisions survive the collapse, not exact rule regeneration. With the gate
met, `role_collapsed.grammar` is now the **default** mental-model grammar
(`MentalModel.xml`, 2026-06-03); `complete.grammar` is retained as the
compatibility baseline (and is what the D1 collapse is measured on). The
part operator is unified there: the grammar declares the single relative op
`isPart` and the query/assert split is recovered by dispatch (`isPart`
$\to$ `queryPart` in a query context). The operator codebook, soft
superposition, and participation clustering are live and tested. See the
status blocks in
[doc/old/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md](old/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md).

### Participation Categories as the Chooser's Syntactic-Category Context

*(Design direction. The category SOURCE — operator codebook, soft
superposition, participation clustering — is live and tested, and the
**MetaSymbol** the category attaches to already exists in the live forward
(below). Learning a category codebook over it from perception, and threading
it into the placement chooser, is PLANNED. See status at the end.)*

> **Terminology note** (per `doc/old/2026-06-21-terminology-percepts-concepts-symbols.md`):
> the CS part↔whole relation table is the **Concept codebook** (concepts), not a
> "symbol table"; WholeSpace holds **whole-percepts**; only `SymbolSpace` / `MetaSymbol`
> things are **symbols**. Code identifiers (e.g. `subspace.what`, `_sym_*`) are
> unchanged here — that is a separate code pass.

A word's grammatical **category is its frequency of participation across the
operator roles** above (`<op>_I<n>` inputs / `<op>_O1` output), **learned from
perception** (analysis of input), not from generation. No part-of-speech label
is declared or needed: if *cat* fills `ADV`'s operand role and `LIFT`'s
argument role with roughly equal frequency, *collapsing* that frequency profile
is what makes it a **noun** — the model knows the category by its role
distribution, never by a name. This is exactly what `participation.learned_collapse`
formalizes (merge by participation similarity, keeping every rule
distinguishable).

This role-participation profile is the **primary determinant of syntax** — what
a constituent *is* (its category) governs what it can combine with more than its
surface content does. So it is precisely the context the placement chooser
(`MLPTransformChooser`, the soft route's scorer — see
[Soft-superposition route](#soft-superposition-route-the-learning-two-pass))
needs when scoring "should this pair reduce, and with which operator?": the
chooser must see the **category of the value already sitting in each slot**, not
only the candidate operator's own output. The design realizes this as a small
**Category codebook keyed by the MetaSymbol**, learned online by E/M from
perception:

- **The MetaSymbol unifies word and object (it already exists, live).** The
  Concept codebook stores `concept → code`; a **MetaSymbol** is the exception — one
  symbol that holds *two* codes, the **word code and the object code**, a
  deliberate equivalence class asserting *this word ≡ this object*. This is the
  live **META node** in the WholeSpace taxonomy: `WholeSpace.insert_meta`
  allocates one SS-codebook row tagged `"meta"` whose two taxonomy children are
  the PS object position and the SS word position, minted during **perception**
  by the autobind hook (`ConceptualSpace._maybe_autobind_meta` at the sentence
  boundary). Because the category attaches to the MetaSymbol, the syntactic
  signal learned from the *word's* role participation directly shapes the
  *object's* category, and vice versa.
- **A small Category codebook, not a per-word count table.** Each MetaSymbol
  already carries a learned ND vector (its `subspace.what` row, EMA-updated on
  revisit); that vector VQ-assigns to the nearest of `K ≈ n_roles (~30)`
  **category centroids**, and each centroid carries the uncollapsed `~30`-D role
  vector (`<op>_I<n>` inputs + `<op>_O1` output). Reuses the live
  `VectorQuantize` machinery; the role vector is a sidecar mirroring the
  `category_logits` EMA lifecycle.
- **E/M learned from perception, with emergent collapse.** *E-step*: assign each
  MetaSymbol's vector to its nearest centroid. *M-step*: EMA the centroid's role
  vector toward the roles the object filled **during analysis** (read off the
  parse route — `op_I<n>` for the operands a reduction consumes, `op_O1` for its
  result), and EMA-recentroid (the VQ's own update). Starting from `K ≈ 30` and
  letting unused centroids decay, **effective K shrinks as words pull centroids
  together** — the online realization of `participation.learned_collapse`, where
  "noun" emerges without a label.
- **Feeds the per-slot category to the chooser.** The chooser conditions each
  slot on the **role vector of the centroid its object maps to** — gathered, at
  the terminal layer, via `percept id → taxonomy parent (MetaSymbol) → centroid
  → role vector` and concatenated into `feat` (`MLPTransformChooser`). The
  default anchor-dot chooser ignores it, so the addition is opt-in and
  basin-preserving.

This splits into two phases: **(1)** learn the category codebook from perception
in the autobind hook (no change to the layer forwards — reads the stashed
analysis route); **(2)** thread the per-slot category through `compose` /
`score_binary` / `score_unary` into the chooser `feat` (the larger change — the
layer forwards carry only `[B,N,D]` today, with no per-slot symbol identity).

**Status (Phase 1 IMPLEMENTED, gated dark by `<categoryCodebook>`; Phase 2
PENDING).** Phase 1 is wired and byte-identical when off:
`WholeSpace.enable_category_codebook` builds the VQ (`codebook_retire=False`)
+ a `_category_role[K, n_roles]` sidecar, enumerated from `compute_role_vocabulary`;
it is requested at build and **lazily enabled on the first perception forward**
(the grammar's role rules are not configured at build, so a build-time enable
sees 0 roles). `LanguageLayer._collect_round0_role_obs` stashes the first binary
tier's round-0 reduces (`op_I1`/`op_I2` per operand) on the SS, and the autobind
hook (`_maybe_autobind_meta`) runs the E/M: assign each MetaSymbol to a centroid
(`assign_category`, + the VQ's free recentroid) and EMA the centroid's role
vector (`update_category_role`). The codebook mechanics (assign / role-EMA /
gather / decay-collapse) are unit-tested; the off path is verified byte-identical
(full suite green). NOT yet done: a live end-to-end E/M smoke (needs a small
`role_collapsed.grammar` + tiny-dataset fixture — XOR has 0 operator roles,
MentalModel needs the fineweb corpus), and **Phase 2** (thread the per-slot
centroid role vector into `MLPTransformChooser.feat`; the layer forwards carry no
per-slot symbol identity today). The current round-0/first-tier observation is
parallel-mode-correct; serial (`symbolicOrder>=1`) attribution is approximate.
The old `WholeSpace.category_codebook` was retired 2026-05-20 and is gone; the
dormant declared-POS tables (`category_embedding`, `category_logits`/
`category_ids`, the order-taxonomy admissibility gate) are superseded and slated
for follow-up retirement, not reuse.

## Shared Weighted-Deduction Framework

PartSpace analysis and WholeSpace parsing are two readings of
the *same* item graph under semiring-weighted dynamic programming
(weighted deduction). The graph is scored once; the semiring chosen
selects the quantity:

- **sum-product** gives soft inside / forward marginals (every viable
  rule keeps positive mass, and gradient flows through $\log Z$);
- **max-plus** gives the single Viterbi route used for the hard forward
  value.

Both spaces share the same direction-agnostic primitives in
`bin/Language.py`:

- `binary_tiling_viterbi` --- max-plus best route.
- `binary_tiling_soft_dp` --- sum-product marginals
  (`reduce_marginal_op`, `logZ`).

The WholeSpace reducer (`BinaryStructuredReductionLayer`) scores
copy / reduce items over rule columns; the PartSpace analyzer
(`MeronymicRouter`, `bin/perceptual_analyzer.py`) scores merge evidence
over perceptual atoms. Because the soft marginals are retained even when
the Viterbi route hardens to one rule, two tied reduce rules both keep
positive mass and both receive gradient --- the property pinned by
`test/test_signal_router_layer.py::test_layer_keeps_soft_superposition_over_reduce_rules`.

### Soft-superposition route (the `<learning>` two-pass)

The straight-through forward above is the **default** (and the byte-identical
basin when `<architecture><learning>` is off): the forward value commits to
the Viterbi route while the soft marginals carry the gradient. Under the
two-pass `<learning>` mode the structured layers instead run a **pure
sum-product superposition at a temperature**, and the chooser sits in the
gradient path *directly* --- no argmax, no `.detach()`, no straight-through.

A single scalar `superposition_temperature` $t \in [0, 1]$ drives it
(`BinaryStructuredReductionLayer` / `UnaryStructuredLayer`):

- the route scores are scaled by `superposition_scale(t)` $= 1 - t$ before
  the soft DP / softmax, so $t = 0$ is the chooser's own (sharp) softmax and
  $t = 1$ collapses the scores to uniform (flat, maximally exploratory);
- the forward value **is** `binary_tiling_soft_dp`'s temperature-scaled
  marginals (binary) or the temperature-scaled action softmax (unary). The
  op blend is the pure `op_soft` posterior, not the hardened one-hot.

`binary_tiling_viterbi` is still computed, but only to read off the routing
masks for the tree (below) --- that read-off is outside the gradient path in
both modes. When `superposition_temperature` is unset (`None`) every branch
falls back to the legacy straight-through, object-for-object, so the
default basin is unchanged.

**Two passes as two trials.** With `<learning>` on, `runEpoch` runs each
sentence through `runBatch` *twice*, as two independent forward/loss/backward
trials (no `loss_A + loss_B`, no shared graph):

1. **pass A** at $t = 0$ (sharp / deterministic) --- recorded into the
   batch error like any ordinary step;
2. **pass B** at $t =$ `<exploreTemperature>` (default `0.5`, flatter) ---
   an exploration trial whose value is trimmed from the per-batch error and
   does **not** increment the batch count.

Pass B is temperature sampling, not a separate objective: a flatter route
lets the chooser escape a local commitment that pass A's sharp argmax would
otherwise lock in, and because the chooser is differentiable in both passes
the exploration gradient updates the same anchors. `BasicModel`
`_set_superposition_temperature(t)` walks the router's `_unary_layers` /
`_binary_layers` (and the STM reducer) to stamp $t$; `runBatch` sets it
before the forward and resets to `None` in a `finally`.

**Reading a tree.** A hard derivation is always recoverable on demand: pin
$t = 0$, run a temp-0 analysis, and read the argmax routing trace
(`action_kind` / `action_op` / the copy / reduce masks) that
`BasicModel.write_syntax_tree()` already walks. Running several sentences
through a temp-0 pass yields one tree each.

This is the standard semiring-parsing pattern (Goodman 1999, *Semiring
Parsing*; the SCFG inside-outside line of Lari and Young 1990; weighted
logic programming / parsing transformations, Eisner and Blatz 2006; and
neural grammar induction, Kim, Dyer, and Rush 2019). See the plan's
section 5.6 for the full mapping.
