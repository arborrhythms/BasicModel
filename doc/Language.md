# Language

This document is synchronized to the code as of 2026-06-02. The source
of truth is `bin/Language.py`, `bin/embed.py`, and (for the perceptual
analyzer) `bin/perceptual_analyzer.py`. (The earlier
`bin/typed_stack.py`, `bin/stm_driver.py`, and `bin/parse_state.py`
modules were retired in the 2026-05-21 / 2026-05-29 refactors; their
functionality folded into `bin/Layers.py` and `bin/Language.py`.)

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

The language layer is the architecture's DisCoCat-facing surface. Like
Categorical Compositional Distributional semantics, it treats grammar as a typed
composition discipline over vector meanings: reductions such as lift, lower,
union, intersection, `part`, and `equal` decide how meanings combine. Unlike a
typical transformer LLM, these reductions are explicit operators rather than
latent behaviors distributed across heads. Their operands are tied back to the
Formal Concept Analysis side of the model through concept order, role
participation, and part/whole support in the codebooks.

> **2026-06-02 deltas (subsymbolic analyzer + terminal emitter).** Plan:
> [doc/old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md](old/2026-05-30-subsymbolic-analyzer-terminal-emitter.md).
>
> - **PS/SS grammar sections.** A `.grammar` file may nest its
>   `<compose>`/`<generate>` under `<PartSpace>` and
>   `<WholeSpace>`. `Grammar.configure` parses them into separate rule
>   tables: `ps_rules` (space-role `P`, read by the PS analyzer) and the
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
>   namespace is untouched; it is wired into `SymbolSubSpace.__init__` so every
>   built model's operator-prefixed parse-tree nodes are codebook-resolvable. The STM idea space holds only combined meanings --
>   the operator says *how* meanings combine, contributing none of its own.
>   The rule-id stays in `.where` (its presence marks a slot as a *computed*
>   idea, which is by definition not a codebook vector).
> - **IdeaSubSpace** (`bin/Language.py`) -- the PS-meronymic carrier
>   analogue of `SymbolSubSpace`: span buffers + parent/child links + route
>   ids/scores + the marker-route replay fields
>   (`_marker_ps_id`/`_marker_span`/`_order_bit`/`_marker_position`).
> - **Perceptual analyzer** (`bin/perceptual_analyzer.py`). `EndpointSumWhere`
>   is the invertible span key $where = phase(start) + phase(end)$: the
>   angle decodes the span center, the magnitude the span length.
>   `MeronymicAnalyzer` analyzes a surface into terminals (compatibility
>   mode reuses the word/byte tokenizer as the `boundary` op, so it matches
>   the current lexer), writes durable spans to an `IdeaSubSpace`, exposes
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
> - `unreduce()` passes the space-role-local Basis (Codebook) to binary
>   GrammarLayer reverses as `basis=space_role_basis` (replacing the prior
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
entry points. There is no longer a backend selector: the signal router
(`LanguageLayer`) is the single canonical parser.

The `<parserBackend>` knob is RETIRED (Stage 3, 2026-05-27). The CKY chart
and the STM shift/reduce parsers it used to select have been deleted in
favour of the signal router. The retired `parserBackend` values
(`chart` / `stm` / `parallel`) no longer exist.

`<routerKind>` is also RETIRED alongside the chart; the `signal` behaviour
it once selected is now unconditional. The retired
`<parserBackend>`, `<routerKind>`, `<chartTau>`, `<chartTopK>`, and
`<chartNoiseEps>` elements raise a loud `ValueError` at config load if a
`<SymbolSpace>` config still sets them
(`Language._assert_retired_chart_knobs_absent`); see `data/model.xsd`.

## Retired XML Knobs

`SymbolSpace.chartCompose`, `SymbolSpace.softChartCompose`, and
`bivectorOutput` are no longer read by the runtime and should not appear
in `data/*.xml`.

`SymbolSpace.useGrammar` is a different case: it is **still read**, not
silently ignored. Its mere presence in a config trips a loud
`DeprecationWarning` and forces a fallback load of `default.grammar`,
discarding whatever `<grammar>` file the config actually named
(`Language.py:1607-1622`). A surviving `<useGrammar>` tag therefore
silently swaps in the wrong grammar rather than doing nothing — remove
the tag, don't rely on the fallback.

Grammar mode is no longer a `useGrammar` string at all; the retired
`"none"` / `"all"` vocabulary collapsed into one boolean,
`SymbolSubSpace._grammar_is_default_only`. It is `True` when every
compose rule is either an implicit passthrough or the default unary
`pi` / `sigma` substrate fold (no operator grammar loaded), and flips to
`False` the moment any non-default operator rule (`part`, `equal`,
`conjunction`, ...) is present.

> **SS-analysis vs CS-execution.** `SymbolSubSpace.compose` is the
> SS-side *analysis* stage (it selects the per-space hard rule dict
> `current_rules`); the CS-side *execution* (applying lift / lower /
> union / intersection / swap / quantize / not to the concept tensors)
> runs in `ConceptualSpace.forward` and the per-space `SyntacticLayer`
> cursors. This split is a clean code boundary **only on the
> default-only path**: on the full-router path
> `LanguageLayer.compose` does both selection and tensor reduction, and
> the per-space cursors are deliberately bypassed
> (`not _grammar_is_default_only`). See
> [STM.md Section 5](STM.md#routing-parser)
> for the accurate, audited account.

## Grammar

`TheGrammar` is the singleton `Grammar` instance. Rules are loaded from
XML `<SymbolSpace><language><grammar>` blocks or from a configured grammar
CFG. A `RuleDef` stores:

```text
(space_role, canonical, arity, method_name, lhs, rhs_symbols,
 width_min, width_max, query)
```

The grammar file carries BOTH directions: the `<compose>` section holds
the forward rules (`op_O1 = op.forward(op_I1, op_I2)`) and the
`<generate>` section the reverse rules (`op_I1, op_I2 = op.reverse(op_O1)`)
— parsed into `rules_upward` / `rules_downward` and concatenated into the
one flat `TheGrammar.rules` table. Both directions carry the BARE
`method_name` (the `.forward`/`.reverse` suffix is stripped and survives
only in `canonical`), so a generate rule resolves the SAME host layer the
compose rule uses. Note the arity asymmetry: a generate rule's
`RuleDef.arity` counts its RHS *call* arguments (1 for
`op.reverse(op_O1)`); its two-output nature is implicit in the LHS
string. Enumerating "the binary reverse ops" therefore filters on
`.reverse in canonical` and the HOST's `arity == 2`
(`BasicModel._grammar_reverse_ops`), not on `RuleDef.arity`.

The live grammar style is **operator-role categories**, not a declared
part-of-speech taxonomy: every operator contributes its own
`<op>_I1`, `<op>_I2` (inputs) and `<op>_O1` (output) categories. For
example, from `data/complete.grammar`:

```text
part_O1 = part.forward(part_I1, part_I2)
lift_O1 = lift.forward(lift_I1, lift_I2)
conjunction_O1 = conjunction.forward(conjunction_I1, conjunction_I2)
```

All four shipped `.grammar` files (`default`, `complete`, `xor`,
`shamatha`) are exclusively in this role-only form. See
[Role-Collapsed Grammar and the Operator Codebook](#role-collapsed-grammar-and-the-operator-codebook)
below for the full account, including how a word's category is
recovered from role participation rather than declared.

`RuleDef.lhs` / `.rhs_symbols` parsing (`Grammar._parse_category`) still
accepts an explicit conceptual-order suffix — `NP3`, `S4 = lift(NP3,
VP1)`, `NP*` Kleene forms — for backward compatibility, but that style is
now **explicitly legacy**: `NP3`, `NP4`, `VP1`, `MP1`, `S4`, and `S5` are
members of `_FORBIDDEN_STATE_TOKENS` in
`test/test_role_collapsed_grammar.py`, and no shipped grammar file
declares any of them.

### GrammarLayer forward / reverse inventory

The per-operator contracts, as implemented (2026-07-14; every op's
`forward` is what `<compose>` fires and its `reverse` is what
`<generate>` fires — see the section pairing above). "Recommender"
means the basis-threaded codebook walk (`Ops.conjunctionReverse` /
`Ops.disjunctionReverse` $\to$ `Ops._binary_op_recommend`); "snap"
means the op-respecting dot-metric word snap
(`snap=True` $\to$ `Ops.word_pair_snap` — see
`doc/plans/2026-07-14-signed-space-snap-design.md`). Ops with no
faithful inverse raise (`raise_no_inverse`, the fail-loud contract) —
fabricating a split would corrupt the reconstruction.

| op (`rule_name`) | arity | role | forward | reverse |
|---|---|---|---|---|
| `not` | 1 | CS | pole swap | self-inverse (`forward(y)`) |
| `non` | 1 | CS | non-affirming complement | self-inverse |
| `intersection` | 2 | CS | `Ops.intersection` (RadMin / lattice min; ADJ mask, meet) | recommender w/ basis; `snap=True` $\to$ MEET-aware snap (priming-led — the meet is lossy); no basis $\to$ raise |
| `union` | 2 | CS | `Ops.union` (RadMax / lattice max, OR-region, join) | recommender w/ basis; `snap=True` $\to$ JOIN snap (fit-determined); no basis $\to$ raise |
| `chunk` | 2 | CS | additive `left + right` (PS-style chunking) | PEEL w/ basis: best-cosine row `x1`, exact residual `(x1, parent − x1)`; empty-set decomposition `(parent, 0)` without |
| `sum` | 2 | CS | element-wise `left + right` | empty-set decomposition `(parent, 0)` — recomposes exactly |
| `product` | 2 | CS | element-wise `left * right` | **raise** (zeros annihilate; many-to-one) |
| `lift` | 2 | CS | order-raising fold (internal SigmaLayer; optional gate) | `Ops.liftReverseAll` w/ basis ($\to$ disjunctionReverse); balanced `_sigma.generate` split without |
| `verb` | 2 | CS | sparse verb-conditioned spectral operator | requires `verb_what` (`reverse_required_kwargs`); returns `(unapply_verb(parent, verb_what), verb_what)` |
| `adverb` | 2 | CS | VP eigenmodifier (`apply_adverb`) | **not dispatchable** (`reverse_dispatchable = False`; lossy) |
| `lower` | 2 | CS | order-lowering (internal PiLayer; DET) | `Ops.lowerReverseAll` w/ basis ($\to$ conjunctionReverse); `_pi.generate` without |
| `preposition` | 2 | CS | `.where`-relation refinement of NP/VP | `(x, x)` with the `.where` rotation undone (content-exact, marker-lossy) |
| `bind` | 2 | CS | contextual missing/controlled-NP resolution | **raise** (context not preserved in the parent) |
| `tense` | 1 | CS | phase rotation of the `.when` band (`shift_time(+delta)`) | exact inverse rotation (`shift_time(-delta)`) |
| `aspect` | 1 | CS | identity (rewrite() planned; not a live rule) | identity |
| `morphology` | 1 | CS | surface inflection $\to$ `.when` (tense/aspect feature ops) | analyzes features, undoes aspect ops in reverse order, then tense |
| `symbolize` | 2 | CS | bind PS percept row + WS symbol row into an idempotent META node (fused average; `insert_meta`) | recover META children by nearest WS row; balanced `(parent/2, parent/2)` split without wired stores |
| `conjunction` | 2 | SS | `Ops.intersection` monotonic (scalar activation min; RadMin under `<radialStmReduce>`) | recommender (`monotonic=True`, radial-aware); `snap=True` $\to$ MEET-aware snap; no basis $\to$ raise |
| `disjunction` | 2 | SS | `Ops.union` monotonic (scalar activation max; RadMax under `<radialStmReduce>`) | recommender; `snap=True` $\to$ JOIN snap; no basis $\to$ raise |
| `exist` | 1 | SS | identity (EXISTS roots the minimal event) | identity |
| `isEqual` | 2 | SS | identity-assertion truth bivector | **raise** (max-fold not bijective) |
| `isPart` | 2 | SS | parthood-assertion truth bivector | **raise** (A's identity not preserved) |
| `part` | 2 | CS | returns the encompassing parent (parthood learned by codebook geometry) | **raise** (A's identity not preserved) |
| `whole` | 2 | CS | converse of `part` (PartLayer subclass) | **raise** (same) |
| `equal` | 2 | CS | geometric mutual-parthood on concept bivectors (Layers.EqualLayer) | lossy `(parent, parent)` pseudo-inverse |
| `query` | 2 | CS | geometric parthood query $\to$ truth bivector | **raise** (two operands collapse to a truth value) |

Notes. (1) The binary lattice reverses (union/intersection,
conjunction/disjunction) accept `left_rows` / `right_rows` (typed
candidate restriction), `left_priming` / `right_priming` (soft boosts),
`radial` (signed-magnitude order), and `snap` — recovery is owned by the
layer and DIFFERS by op: the join is fit-determined; the lossy meet leans
on priming (which words are present). (2) The trace-free free-derivation
decode (`_reverse_reduce_unfold` $\to$ `_reverse_choose_op`) CHOOSES
among the `<generate>` binary ops per un-fold step by round-trip fit —
`op.compose(op.reverse(parent)) \approx parent` — with no forward record.
(3) `VerbLayer`/`AdverbLayer` subclass `LiftLayer`; `WholeLayer` subclasses
`PartLayer`; `TenseLayer`/`AspectLayer` share `_WhenOpMixin`; `EqualLayer`
lives in `bin/Layers.py`, all others in `bin/Language.py`.

## Knowledge Artifacts

`embed.build_knowledge_section(grammar)` creates the parser knowledge
section, five sub-sections in all:

- `word_table`: bootstrap CSR word table (`build_word_table_initial(wv)`,
  `embed.py:857`) — UTF-8 surface-form bytes keyed by
  `keys_values`/`keys_offsets`, plus a `ref_ids` column initialized to
  `-1` (unassigned) until a curated POS lexicon / tagger populates it.
- `taxonomy`: base category refs plus explicit ordered refs.
- `reference_codebook`: scalar prototypes and `order` per ref.
- `typed_indexes`: `refs_by_category` and `refs_by_order`.
- `grammar.rule_order_signatures`: serialized rule typing.

The `taxonomy` builder (`build_taxonomy_from_grammar`) still supports
ordered-ref children of a base category — the machinery is generic,
parsing whatever `Grammar._parse_category` finds — but that shape is now
**legacy**: it only appears for a grammar declaring explicit-order
categories (`NP3`, `NP4`, ...; see the [Grammar](#grammar) section
above), and no shipped `.grammar` file declares any. On the live
role-only grammars every category is order-flat (`part_O1`, `lift_I1`,
...), so `taxonomy` has no ordered children in practice. Historically the
illustration was:

```text
NP
|-- NP3
`-- NP4
```

`KnowledgeView.category_of_ref(ref_id)` returns the base category
(`NP`), while `KnowledgeView.order_of_ref(ref_id)` returns the
conceptual order (`3`, `4`, etc.) — accurate for a grammar that still
uses this legacy style.

`SymbolSpace.category_codebook` has been retired. The live category
embedding is:

```python
SymbolSpace.category_embedding: nn.Embedding
```

## STM Shift/Reduce

`ShortTermMemory.shift` / `.reduce_step` / `.reduce_step_soft` and the
`_RuleScorer` MLP (`bin/Layers.py`) are a typed admissibility-masked
shift/reduce driver — SHIFT snaps an input vector to the nearest live
reference and pushes `(payload, category, order, ref_id)`; REDUCE masks
rule logits by typed admissibility, softmaxes over what's admissible,
and records the argmax. It has **zero production callers**: every call
site is `test/_stm_test_fixtures.py`, which keeps the surface alive as a
compat shim for tests written before the 2026-05-21 SymbolSubSpace
refactor. The live parsing mechanism is `LanguageLayer.compose`'s
soft-DP / Viterbi weighted deduction (`binary_tiling_soft_dp` /
`binary_tiling_viterbi`); see [STM.md Section 5](STM.md#routing-parser)
for the accurate, audited account of SS-analysis vs CS-execution.

`ConceptualSpace.stm` (a `ShortTermMemory` instance) is, in the live
model, a plain payload/idea stack, not a typed parser stack: its
per-batch slab and depth pointer (`_buffer` / `_depth`) are ordinary
tensors written by `push_step_masked(ideas, gate)` (the gated
newest-at-slot-0 push) and read back by `snapshot()`. It carries no
category / order / ref-id typing — that typed buffer is what the dead
shift/reduce driver above expects, and nothing in the live model
populates it.

## Syntax And SVO

`BasicModel.write_syntax_tree()` (`bin/Models.py`) is currently unwired
infrastructure, not a live path. The function hardcodes `chart = None`
and `traces = None` — the CKY chart is retired and the signal router
does not yet populate per-leaf derivation traces to replace it — so
every call emits a bare `<noTrace/>` element regardless of input. The
docstring's `<node>`/`<leaf>` format is what the function would produce
once a trace source is wired, not what it produces today.

SVO extraction has the same status. `SymbolSpace.set_last_svo` /
`get_last_svo` / `clear_last_svo` (`bin/Language.py`) exist, and
`clear_last_svo` does fire from production code (every CS-forward cycle
and at Reset / soft-reset boundaries), but `set_last_svo` itself has no
production caller anywhere in `bin/*.py` — it is exercised only by
`test/test_per_batch_state_isolation.py` and
`test/test_subspace_context.py`. Until something calls it, `_last_svo`
never leaves its post-clear zero state in the live model.

## Role-Collapsed Grammar and the Operator Codebook

The live role-only grammars replace part-of-speech categories with
*operator roles*. Instead of a fixed
`NP` / `VP` / `S` taxonomy, each operator contributes its own argument
and result roles, named `<op>_I1`, `<op>_I2` (inputs) and `<op>_O1`
(output). The rule `equal` therefore exposes `equal_I1`,
`equal_I2`, `equal_O1`, and the grammatical "category" of a span is
just the set of operator roles it can fill. Dimensionality is recovered
from participation (`bin/participation.py`) rather than declared up
front: symbols that fill the same roles cluster into the same category.

Starts are scoped per space (`_configure_starts`):

- `PartSpace.start` is the universal whole-input role `U` --- the
  analyzer begins from the entire surface and decomposes it.
- `WholeSpace.start` is the set of operator output roles
  (`equal_O1`, `part_O1`, `exist_O1`, ...) --- parsing begins from
  what an operator can *produce*.

The **operator codebook** is a second codebook on `WholeSpace`,
separate from the whole-percept codebook:

- `_operation_vectors`: operator name $\to$ identity vector.
- `_operation_positions`: operator name $\to$ codebook position.

It stores one prototype per operator — the live codebook is `equal`,
`part`, `whole`, `exist`, `conjunction`, `disjunction`, `adverb`, `bind`,
`intersection`, `lift`, `lower`, `morphology`, `non`, `not`,
`preposition`, `product`, `sum`, `tense`, `union`, `verb`
(`insert_operations`, `bin/Spaces.py:19582`) — and is kept CPU-explicit
so the host-side identity lookup never mixes with an ambient MPS / CUDA
default device.

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
met, the former standalone role-collapse file has been absorbed into
`complete.grammar`, which is the broad live role-only grammar used by
`MentalModel.xml`. The part relation is unified there: the grammar
declares the single compositional op `part` (plus its converse `whole`);
`isPart` / `queryPart` survive only as `<Queries>` predicates, outside
`<compose>`. The query/assert split is *not* recovered by
`_dispatch_method_name_for_rule` (`bin/Language.py:4187`) — that helper
only fires for `query="true"` rules, and `complete.grammar` declares
none. The live unification is `reasoning.py`'s `_SURFACE_TO_KIND` table
(`bin/reasoning.py:27`), which maps every surface form — `part`,
`isPart`, and `queryPart` alike — to the same `KIND_IS_PART` reduction
kind (and symmetrically `equal` / `isEqual` / `queryEqual` $\to$
`KIND_IS_EQUAL`). The operator codebook, soft
superposition, and participation clustering are live and tested. See the
status blocks in
[doc/old/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md](old/2026-06-02-unified-subsymbolic-analyzer-and-role-collapsed-grammar.md).

### Participation Categories as the Chooser's Syntactic-Category Context

*(Live path. The category source is learned from role participation during
perception, attached to the **MetaSymbol**, and threaded into the placement
chooser as grammatical context.)*

> **Terminology note** (per `doc/old/2026-06-21-terminology-percepts-concepts-symbols.md`):
> the CS part$\leftrightarrow$whole relation table is the **Concept codebook** (concepts), not a
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
  Concept codebook stores `concept -> code`; a **MetaSymbol** is the exception — one
  symbol that holds *two* codes, the **word code and the object code**, a
  deliberate equivalence class asserting *this word $\equiv$ this object*. This is the
  live **META node** in the WholeSpace taxonomy: `WholeSpace.insert_meta`
  allocates one SS-codebook row tagged `"meta"` whose two taxonomy children are
  the PS object position and the SS word position, minted during **perception**
  by the autobind hook (`ConceptualSpace._maybe_autobind_meta` at the sentence
  boundary). Because the category attaches to the MetaSymbol, the syntactic
  signal learned from the *word's* role participation directly shapes the
  *object's* category, and vice versa.
- **A small Category codebook, not a permanent per-word count table.** The VQ
  lives directly in role-participation space: $K \approx$ `n_roles` (55 on the
  live `complete.grammar`, `compute_role_vocabulary`) initial
  centroids, one seeded from each labelled role (`<op>_I<n>` inputs +
  `<op>_O1` outputs). Unlearned MetaSymbols have only a bounded temporary row in
  `MetaSymbolCategoryLearner`; learned MetaSymbols keep just
  `MetaSymbol -> category_id`.
- **E/M learned from perception, with emergent collapse.** Each analysis route
  contributes a sparse role vector to the MetaSymbol that occupied the terminal
  position. The pending row accumulates that evidence until mass, confidence,
  margin, and short-term stability thresholds are met. Then the MetaSymbol
  commits to one VQ centroid and the pending row is discarded. Starting from one
  centroid per role and letting unused centroids decay, **effective K shrinks as
  role-use profiles pull centroids together** — the online realization of
  `participation.learned_collapse`, where "noun" emerges without a label.
- **Feeds the per-slot category to the chooser.** The chooser conditions each
  slot on the **role vector of the committed centroid**; while a word is still
  unsettled, the pending row supplies a temporary role context. The gather path
  is `percept id $\to$ taxonomy parent (MetaSymbol) $\to$ committed category or pending
  evidence $\to$ role vector`. `MLPTransformChooser` receives the vector as a
  feature block; anchor-dot/default routing uses the same vector as a
  labelled-role score prior.

This splits into two phases: **(1)** learn the category codebook from perception
in the autobind hook (no change to the layer forwards — reads the stashed
analysis route); **(2)** thread the per-slot category through `compose` /
`score_binary` / `score_unary` into the chooser `feat` (the larger change — the
layer forwards carry only `[B,N,D]` today, with no per-slot symbol identity).

**Status (implemented behind `<categoryCodebook>`, default true).**
`WholeSpace.enable_category_codebook` builds the role-space VQ
(`codebook_retire=False`) + `_category_role[K, n_roles]`, enumerated from
`compute_role_vocabulary`; it is requested at build and **lazily enabled on the
first perception forward**. `LanguageLayer._collect_round0_role_obs` stashes the
first binary space-role's round-0 reduces (`op_I1`/`op_I2` per operand), and the
autobind hook (`_maybe_autobind_meta`) feeds those observations to
`MetaSymbolCategoryLearner`. The learner owns the pending per-MetaSymbol role
rows, commits stable symbols into `WholeSpace._category_assign`, and drops the
pending row. Structured grammar layers use the resulting role context for all
transform choosers: MLP as an input feature, anchor-dot/default as a
labelled-role score prior. The current round-0/first-space-role observation is
parallel-mode-correct; serial (`<serial>true</serial>`) attribution is approximate.
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

## Future work: nouns from PartSpace, adjectives from WholeSpace

The signed-space snap
(`doc/plans/2026-07-14-signed-space-snap-design.md`) models a concrete
noun as an adjective pre-applied to the top domain --- `black(cat($\ldots$))`
treats "cat" as the same *kind* of function as "black", the later one
already mapped over $\mathbb{1}$. That modeling choice is probably already
latent in the mereology poles. The lattice poles are vectors of the
presence domain (see `doc/Architecture.md`): **NOTHING** $= [0,0,\ldots]$
is the part/bottom pole, **EVERYTHING** $= [1,1,\ldots]$ is the whole/top
pole. So:

- a **whole (property)** is pre-applied to EVERYTHING --- narrowing the
  wide-open $\mathbb{1}$ object downward (the `assert_concept_relation`
  first-concrete-whole-replaces-EVERYTHING move); this is the *adjective*
  shape, a modifier that carves the domain;
- a **part (particle)** is pre-applied to NOTHING --- building presence up
  from $\mathbf{0}$ (the first-concrete-part-replaces-NOTHING move); this
  is the *concrete-noun* shape, an object accreted from constituents.

The conjecture for a future pass: source **concrete nouns from PartSpace**
and **adjectives from WholeSpace**, so the grammatical part-of-speech
distinction falls out of which pole a symbol is pre-applied to, rather than
being a learned category. This would give the snap's ADJ(N) intersection a
principled home --- the noun's broad PS-side support and the adjective's
narrowing WS-side mask are then *typed by origin*, and the
support-restricted metric (score over the parent's support, unaffected by
the modifier's attenuation) becomes the natural recovery law rather than a
tuning choice. Open questions: how order-raising (`maybe_raise_order`)
interacts with a PS/WS part-of-speech split; whether abstract nouns want
the WholeSpace (property-like) or PartSpace (object-like) origin; and how
the `<Anchors>` closed-class relation surfaces sit relative to this axis.
