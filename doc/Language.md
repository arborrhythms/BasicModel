# Language

This document is synchronized to the code as of 2026-06-02. The source
of truth is `bin/Language.py`, `bin/embed.py`, and (for the perceptual
analyzer) `bin/perceptual_analyzer.py`. (The earlier
`bin/typed_stack.py`, `bin/stm_driver.py`, and `bin/parse_state.py`
modules were retired in the 2026-05-21 / 2026-05-29 refactors; their
functionality folded into `bin/Layers.py` and `bin/Language.py`.)

> **2026-06-02 deltas (subsymbolic analyzer + terminal emitter).** Plan:
> [doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md](plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md).
>
> - **PS/SS grammar sections.** A `.grammar` file may nest its
>   `<compose>`/`<generate>` under `<PerceptualSpace>` and
>   `<SymbolicSpace>`. `Grammar.configure` parses them into separate rule
>   tables: `ps_rules` (tier `P`, read by the PS analyzer) and the
>   canonical symbolic `rules` (`ss_rules`). A bare `<compose>`/`<generate>`
>   file loads as `<SymbolicSpace>` (backward-compat). The legacy `.cfg`
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
>   decision #2). `SymbolicSpace.insert_operations(grammar)` registers each
>   operation in a dedicated operator codebook on SymbolicSpace
>   (`_operation_vectors` / `_operation_positions`), separate from the
>   `subspace.what` symbol codebook so the symbol / idea / `.where` position
>   namespace is untouched; it is wired into `WordSubSpace.__init__` so every
>   built model's operator-prefixed parse-tree nodes are codebook-resolvable. The STM idea space holds only combined meanings --
>   the operator says *how* meanings combine, contributing none of its own.
>   The rule-id stays in `.where` (its presence marks a slot as a *computed*
>   idea, which is by definition not a codebook vector).
> - **ObjectSubSpace** (`bin/Language.py`) -- the PS-meronymic carrier
>   analogue of `WordSubSpace`: span buffers + parent/child links + route
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
>   tree with `emit`). `soft_operator_compose` + `SymbolicSpace
>   .operator_superposition` apply a soft operator distribution over the SS
>   operation codebook (one-hot $\to$ the typed grammar; spread $\to$ the
>   superposition that discriminates `A AND B` from `A OR B`).
> - **PS-to-SS binding.** `SymbolicSpace.resolve_ps_terminal(ps_id)` emits
>   `null_sem()` before a binding exists, counts exposures, and promotes a
>   repeated terminal into a fresh SS row.

> **2026-05-29 deltas:**
>
> - `unreduce()` passes the tier-local Basis (Codebook) to binary
>   GrammarLayer reverses as `basis=tier_basis` (replacing the prior
>   raw-`W` form). `UnionLayer.reverse` / `IntersectionLayer.reverse`
>   extract `W = basis.getW()` internally and dispatch to
>   `Ops.disjunctionReverse` / `Ops.conjunctionReverse`. Layers that
>   don't accept the `basis` kwarg yet are handled by a `TypeError`
>   fallback. No back-ref is stored on the layer.
> - `MetaLayer` was renamed to `SymbolizeLayer` (no semantic change).
> - Word-mode parse appends a `\x00` null sentinel after the words
>   slab for explicit end-of-sequence on the forward path.
> - See [doc/plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md](plans/2026-05-29-clean-stack-stm-basis-arg-radixlayer.md).

## Current Parser Surface

`WordSpace.compose()` and `WordSpace.generate()` are the public parser
entry points. `WordSpace.parser_backend` selects the implementation:

| backend | status | notes |
|---|---|---|
| `chart` | default | Compatibility path. Runs the existing chart / Viterbi machinery. |
| `stm` | active | Shift/reduce over `ConceptualSpace.stm_typed`; requires an attached `KnowledgeView`. |
| `parallel` | bridge | Builds the STM driver, then runs the chart path authoritatively. |

`WordSpace.routerKind` is separate from `parserBackend` and only affects
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

- `WordSpace.useGrammar`
- `WordSpace.chartCompose`
- `WordSpace.softChartCompose`
- `bivectorOutput`

Grammar mode is derived from the loaded grammar: default-only unary
`pi` / `sigma` rules derive `useGrammar == "none"` internally; any
non-default operator rule derives `useGrammar == "all"`.

> **SS-analysis vs CS-execution.** `WordSubSpace.compose` is the
> SS-side *analysis* stage (it selects the per-tier hard rule dict
> `current_rules`); the CS-side *execution* (applying lift / lower /
> union / intersection / swap / quantize / not to the concept tensors)
> runs in `ConceptualSpace.forward` and the per-tier `SyntacticLayer`
> cursors. This split is a clean code boundary **only on the
> default-only path**: on the full-router path
> `LanguageLayer.compose` does both selection and tensor reduction, and
> the per-tier cursors are deliberately bypassed
> (`not _grammar_is_default_only`). See
> [STM.md Section 5](STM.md#5-routing-parser-ss-analysis-vs-cs-execution)
> for the accurate, audited account.

## Grammar

`TheGrammar` is the singleton `Grammar` instance. Rules are loaded from
XML `<WordSpace><language><grammar>` blocks or from a configured grammar
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

`WordSpace.category_codebook` has been retired. The live category
embedding is:

```python
WordSpace.category_embedding: nn.Embedding
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

- `PerceptualSpace.start` is the universal whole-input role `U` --- the
  analyzer begins from the entire surface and decomposes it.
- `SymbolicSpace.start` is the set of operator output roles
  (`isEqual_O1`, `isPart_O1`, `exist_O1`, ...) --- parsing begins from
  what an operator can *produce*.

The **operator codebook** is a second codebook on `SymbolicSpace`,
separate from the symbol codebook:

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
[doc/plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md](plans/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md).

## Shared Weighted-Deduction Framework

PerceptualSpace analysis and SymbolicSpace parsing are two readings of
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

The SymbolicSpace reducer (`BinaryStructuredReductionLayer`) scores
copy / reduce items over rule columns; the PerceptualSpace analyzer
(`MeronymicRouter`, `bin/perceptual_analyzer.py`) scores merge evidence
over perceptual atoms. Because the soft marginals are retained even when
the Viterbi route hardens to one rule, two tied reduce rules both keep
positive mass and both receive gradient --- the property pinned by
`test/test_signal_router_layer.py::test_layer_keeps_soft_superposition_over_reduce_rules`.

This is the standard semiring-parsing pattern (Goodman 1999, *Semiring
Parsing*; the SCFG inside-outside line of Lari and Young 1990; weighted
logic programming / parsing transformations, Eisner and Blatz 2006; and
neural grammar induction, Kim, Dyer, and Rush 2019). See the plan's
section 5.6 for the full mapping.
