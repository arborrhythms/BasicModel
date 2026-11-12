# Language

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are built (accepted as implemented, Alec, 2026-11-01).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - Amended 2026-10-23 (recorded 2026-10-30): `bind` is live both as a compose operation (`BindLayer`, `bind.forward` in `<compose>`) and as a thought operation (`bind.thought`), once per site; see [Architecture](Architecture.md)'s read-first box.
> - Operators are minimal: tense, aspect, the preposition and the modal are one `position` operation (the modal with a world as object; no `MP` rule); the verb is a group element (`verbSpectrum` diagonal, log-ratio teacher); `surface` is absorb and transpose only; no language is named or special-cased in code, configuration, fixtures or tests. Defaults: NP = everything, VP = *occurs* (read for "exists"), MP = all possible worlds.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23; recorded 2026-10-30):** an LTM1 row is two tagged references (each to an LTM0 occurrence, a symbol or another LTM1 row, its tag saying which) with its kind read from the referents, not three. The unused `LTM0` columns are removed and the row has this layout (Alec, 2026-11-01).
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).


The live grammar is owned by `LanguageSpace` in `bin/Language.py`.
A sentence stores its end state and discards its temporary reading record
after reconstruction and the comparison of its paths.
WholeSpace supplies primitive properties; ConceptualSpace owns word and
object concepts, their index and learned taxonomy.

## Relation to LLMs, Formal Concept Analysis, and DisCoCat

### Input reconstruction

Reconstruction is free read-back from the concluded root by the generate
walk: no recorded rule sequence, operand position, retained operand or
witness offset chooses its actions. Both binary operands are found
by bounded search over the sentence's primed concept bank, including when
the operator has an analytic balanced split. Candidate word scoring is
signed activation × cosine × priming, shared with the reconstruction gate.
Hard retrieval addresses are detached; candidate codes and the soft search
gradient remain live. `reconstructionBasisLimit` bounds candidates per side;
no candidate contributes no reconstruction term and is counted.

Input realization does not enter the answer's free generate chart.
`Understanding` owns the result; the answer consumes its detached record.
[GradientFlow](GradientFlow.md) lists its fields, ownership and trained cost.

### Product, probabilistic sum, and the operator catalogue

The searched operators use the same numerical binding in all three faces.
Reverse and generate search a supplied concept basis through the compose
kernel; absent a legal pair they fail loudly. Conjunction of the same native reference is
that reference, whereas coincident codes at different addresses still bind.
The code parameters have no norm constraint. These kernels apply to operands
without a meaning block; on a meaning block a full-width concept composes its
form and meaning separately, as the
[inventory](#grammarlayer-forward--reverse-inventory) below describes.

| Name | Compose / forward | Generate | Reverse |
|---|---|---|---|
| `conjunction` | `norm(x) * norm(y) * unit(x*y)`; repeated reference → `x` | Search a pair through product binding | Same search |
| `disjunction` | `(norm(x)+norm(y)-norm(x)*norm(y))*unit(x+y-x*y)` | Search a pair through probabilistic sum | Same search |
| `sum` | Arithmetic mean `(x+y)/2` | Direct balanced split; free decoder searches the primed bank | Direct `(parent,parent)`; known operand gives `2*parent-witness` |
| `min` | Coordinate minimum | Search a pair through minimum | Same search |
| `max` | Coordinate maximum | Search a pair through maximum | Same search |

`complete.grammar`, XOR_grammar and MM_grammar (through `default.grammar`)
select `conjunction` and `disjunction`.
No current grammar file selects `min` or `max`; both are live catalogue entries
with tests, available to later grammars. `sum` keeps the mean as the additive
control. As decided in item 6.8 (Alec, 2026-10-04), the XOR table
measures the composition mechanism: nonlinear composition,
decoding, affine reading and single-objective ownership. The four-sentence
fixture does not measure which operator a grammatical construction means.
The unchanged bars measure convergence; nonlinearity does not guarantee it.

**Why these operators.** An adjective restricts its noun's extension, and applying it again removes nothing: a red red bird is no more red than a red bird. A restriction is idempotent; repetition on the surface is a different operator, as reduplication in Hawaiian intensifies or pluralizes (nui to nunui, wiki to wikiwiki; Elbert and Pukui, *Hawaiian Grammar*), a degree or number operator, not a second AND. Among t-norms only min is idempotent, and a product assumes independence, while the same predicate twice is maximally dependent ($P(A \wedge A) = P(A)$). Hence conjunction of the same reference is that reference, and numerically the conjunction of distinct references is their product: it binds two identities into a vector unlike either, invertible by division or by search, as binding is in vector symbolic architectures (Plate 1995, 2003; Gayler 2003; Kanerva 2009), where clean-up memory is search over known items. The product of two unit codes has norm about $1/\sqrt{D}$, so the direction is renormalized and the activations carry the magnitude. A mean is affinely dependent on XOR's four sentences and reads it only at the floor of one quarter, which is why `sum` serves as the additive control.

### Concept composition

The language layer is the architecture's DisCoCat-facing surface. Like
Categorical Compositional Distributional semantics, it treats grammar as a typed
composition discipline over vector meanings: reductions such as lift, lower,
union, intersection, `part`, and `equal` decide how meanings combine. Unlike a
typical transformer LLM, these reductions are explicit operators rather than
latent behaviors distributed across heads. Their operands are tied back to the
Formal Concept Analysis side of the model through concept order, role
participation, and part/whole support in the codebooks.

The grammar's historical `<WholeSpace>` section name denotes symbolic
rules (`ws_rules`); it does not give the property tower a word dictionary.
PartSpace rules describe perceptual analysis. LanguageSpace shares numerical
operators across compose, generation and the declared thought faces.
ConceptualSpace's index supplies native concept references; WholeSpace has
no operator codebook, terminal emitter, word rows or META taxonomy.

> **2026-05-29 deltas:**
>
> - `unreduce()` passes the space-role-local Basis to binary reverses.
>   The lattice `UnionLayer` / `IntersectionLayer` use `Ops.unionReverse`
>   and `Ops.intersectionReverse`. Product/mean and min/max use bounded
>   search through their own kernels. No-basis lossy inverses fail loudly.
> - `MetaLayer` was renamed to `SymbolizeLayer` (no semantic change).
> - Word-mode parse appends a `\x00` null sentinel after the words
>   slab for explicit end-of-sequence on the forward path.

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

## Sentence expectation and interaction memory

`SymbolSubSpace` always owns one `WhatInteractionMemory`; expectation uses
its own `SymbolExpectation`, one activation predictor over the symbol bank
for words, sentences and rows. Construction is in
[`SymbolSubSpace.__init__`](../bin/Language.py), and model thinking
reads the owner through [`_what_memory`](../bin/Models.py).
The retired `whatThinkingMemory` switch and discourse delegates are removed.

`sentenceExpectation` defaults to true. Its three-role envelope is only the
memory interface: it holds one native representative, never a predicted
syntactic structure.
Composition has no expectation capability. Subtraction belongs to the closing;
the chooser receives detached conceived roles. A declared `not.thought` may
conclude a serial inference, but absence itself executes nothing. The positive
production prior seeds `<generate>` only. See
[ExpectationRetention](ExpectationRetention.md).
[`set_sentence_expectation`](../bin/Models.py) can switch it at runtime;
[`ensure_sentence_expectation`](../bin/Language.py) creates its parameters
once and registers them for optimization when first enabled. Re-enabling starts
a fresh observation stream. Soft packed-brick resets preserve an enabled
stream; hard resets and document changes make the affected row cold.
See `SymbolExpectation.Reset` in [Layers](../bin/Layers.py).

## Open reading and completed fields

The selected compose operations are recorded while reading for the tied
inverse objective and the explore paths' departures. The numerical
journal captures operands and results at execution time; closing assigns
grammatical metadata without executing those operators again.

After the row is written, generation begins from its actual one-slot or
three-slot field and the current generate policy. Answer materialization and
recall use `SentenceEndState`, with no saved input program. The standalone
[`forward_binary_step`](../bin/Language.py) dispatch remains available for
explicit operation calls and its
[value and gradient checks](../test/test_recorded_compose_dispatch.py#L44).

## Retired XML Knobs

`SymbolSpace.chartCompose`, `SymbolSpace.softChartCompose`, and
`bivectorOutput` are no longer read by the runtime and should not appear
in `data/*.xml`.

`SymbolSpace.useGrammar` is a different case: it is **still read**, not
silently ignored. Its mere presence in a config trips a loud
`DeprecationWarning` and forces a fallback load of `default.grammar`,
discarding whatever `<grammar>` file the config actually named
(`Grammar` loading in `bin/Language.py`). A surviving `<useGrammar>` tag therefore
silently swaps in the wrong grammar rather than doing nothing — remove
the tag, don't rely on the fallback.

Grammar mode is no longer a `useGrammar` string at all; the retired
`"none"` / `"all"` vocabulary collapsed into one boolean,
`SymbolSubSpace._grammar_is_default_only`. It is `True` when every
compose rule is either an implicit passthrough or the default unary
`pi` / `sigma` substrate fold (no operator grammar loaded), and flips to
`False` the moment any non-default operator rule (`part`, `equal`,
`conjunction`, ...) is present.

**What the grammar's `sigma` and `pi` bind to (item 11c, 2026-09-24).** The
perceptual towers no longer own fold layers, so a unary `P = sigma(P)` or
`C = pi(C)` rule has no host layer and the dispatcher passes the operand
through unchanged (`_by_name` miss → `return subspace`); `sigma` and `pi`
are not in `GRAMMAR_LAYER_CLASSES` and no host layer is shared across
roles. The grammar's own folds are the `GrammarLayer` operators it declares
and owns: `lift` (numerically a `SigmaLayer` inside `LiftLayer`), `lower`,
`union`, `intersection` and the rest of the registry. They act within
conceptual space and never raise order: a constituent's order is read from
the concept it denotes, and definitions raise it (below).

> **Orders are learned (Alec, 2026-10-21; built in item 3).** No ordered
> grammar is written; orders live in the concepts
> ([Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)).
> Definitions raise: several "is" rows that share a side, with distinct
> other sides, make an equivalence class one order up. One composition
> lowers one order, stopping at 0: `lift` as NP + VP (amended 2026-10-22
> and 2026-10-23: the determiner selects, the modal withholds the
> lowering). So `lift` will lower rather than keep the subject's order.
> Parts of speech come only from positions and occupations, never from
> definitions.
> **Amended 2026-11-08:** formal arguments are explicit higher-order concepts. Their members come from kept grammar participation and their geometry learns through the general bidirectional hierarchy; see [Learned argument concepts](#learned-argument-concepts).

> **SS-analysis vs CS-execution.** `SymbolSubSpace.compose` is the
> SS-side *analysis* stage (it selects the per-space hard rule dict
> `current_rules`); the CS-side *execution* (applying lift / lower /
> union / intersection / not to the concept tensors)
> runs in `ConceptualSpace.forward` and the per-space `SyntacticLayer`
> cursors. This split is a clean code boundary **only on the
> default-only path**: on the full-router path
> `LanguageLayer.compose` does both selection and tensor reduction, and
> the per-space cursors are deliberately bypassed
> (`not _grammar_is_default_only`). See
> [STM.md Section 5](STM.md#routing-parser)
> for the accurate, audited account.

## Grammar

Selected grammar roles resolve lexical references at the requested concept
order. The form-to-identity index, order assignments and missing/ambiguous
reference rule are defined in [Lexicon](Lexicon.md#word-forms-and-concept-orders).
The selected program owns those semantic addresses separately from its
input reconstruction leaves.

`TheGrammar` is the singleton `Grammar` instance. Rules are loaded from
XML `<SymbolSpace><language><grammar>` blocks or from a configured grammar
CFG. A `RuleDef` stores the rule's spelling and shape (`space_role`,
`canonical`, `arity`, `method_name`, `surface_name`, `lhs`, `rhs_symbols`,
`width_min`, `width_max`), its grammar attributes (`thought_family`,
`thought_permutation`, `reference_roles`, `clause_form`, `head_role`,
`determiner_mode`, `operand_kinds`) and the declarations it copies from its
implementation (`Grammar._declared_rule`: effects, footprints, relation kind,
polarity effect, reference modes and the rest).

`query` is a retained false compatibility slot in the in-memory tuple; XML
`query` attributes are rejected. `thought_family` and
`thought_permutation` are grammar metadata for a canonical structural family
(for example, `whole` as the `(argin2, argin1)` converse spelling of `part`), never a
second executable catalogue.

The grammar file has three peer sections in this order: `<compose>`,
`<thought>`, `<generate>`. Compose holds forward rules
(`op_argout1 = op.forward(op_argin1, op_argin2)`), thought holds the per-model boundary
allow-list (`op_argout1 = op.thought(op_argin1, op_argin2)`), and generate holds reverse
rules (`op_argin1, op_argin2 = op.reverse(op_argout1)`). Only `<thought>` creates
`Grammar.thought_operations`; compose/generate membership does not imply
thought permission. Matching faces share one identity and role contract but
receive their phase-specific context. The capitalized `<Queries>` section is
rejected, not treated as a parallel catalogue.

The field candidates `divide`, `descend` and `gloss` are declared in
`<compose>` with field operands and no predefined word surfaces. Their
`<generate>` face is absent: the closing owns their bracket witness. `and`,
`or` and `not` also act within a field; order-dependent operators require
symbol operands between brackets. Parser validation enforces this distinction.
The pinned word stop masks gloss above words and descent below known words.
The same chooser scores priming, bracket actions and typed space reads.

The line between the two is superposition against transformation. An encoding is symbolic when it can be written in symbol values alone, positive and negative: a field, or any signed superposition of rows, whether or not a row exists for the result. It is conceptual when it cannot: an adjective applied to a noun moves the noun to a point that need not be a symbol. Both kinds are indexed by order. The operations closed on a symbolic encoding are the coordinatewise ones on the signed pairs (`and`, `or`, `not`, `non`), which are also the order-independent ones, so they act within a field; `lift`, `lower`, `verb` and `preposition` displace a point off the span of the rows and need an identified concept between brackets. An operator's signature therefore has two axes: the encoding it takes and the cardinality it needs, a pooled plural reading or one identified concept. `conjunction` and `disjunction` are the symbolic pair, over activation patterns; `intersection` and `union` are the conceptual pair, over vectors in conceptual space.

All executable operators declare their reads, writes, operand kinds, order
change and head roles: one capability declaration over the accessible
subsystems (`AccessibleMind.OperatorEffects`) and one footprint over form,
meaning and poles (`OperatorFootprint`), checked when the grammar loads. An
operator's effects are the same in every face that holds it (compose,
generate, thought): a face's permissions can only remove writes from the
declared set, never add one, and an operation that needs a different effect
in another face is a different operator. Aliases carry their
implementation's declarations, not their spelling. One round permits one
writer per affected subsystem (`EffectRound`). `sum` averages, `synthesize`
adds, `non` withholds without a faithful inverse, and `not` exchanges the
evidence poles. Retired names raise at load with their replacement: `chunk`
(`synthesize`, inverse `analyze`), `what` (`ask` or `query`), `lookup`
(`query`), `true` (`isTrue`), `quantize`, `arma` and `symbolize`; `exist`
keeps only its thought face. `generic` has its own identity. Compound
operators select eligible primed cases before applying their ordinary
algebra.

Compose/generate rules are parsed into `rules_upward` / `rules_downward` and
concatenated into the one flat `TheGrammar.rules` table. Both structural
directions carry the BARE `method_name` (the `.forward`/`.reverse` suffix is
stripped and survives only in `canonical`), so a generate rule resolves the
SAME host layer compose uses. Note the arity asymmetry: a generate rule's
`RuleDef.arity` counts its RHS *call* arguments (1 for
`op.reverse(op_argout1)`); its two-output nature is implicit in the LHS
string. Enumerating "the binary reverse ops" therefore filters on
`.reverse in canonical` and the HOST's `arity == 2`
(`BasicModel._grammar_reverse_ops`), not on `RuleDef.arity`.

Closed-class `<Anchors>` map each case-folded surface spelling to a grammar
form. The eager forward resolves that form while staging the PartSpace word
and records it by its retained WORD row. Completed-answer capture reads this
decision by row, without consulting text or the anchor table again; raw
spelling, dictionary row, and native address are not semantic features. The
row map belongs to the current staging and clears with the word rows on Start,
hard reset, or the next forward. Each captured program keeps its own form
strings across later staging and detached recall copies. That lets a
lexical `whole` (including a declared anchor paraphrase) recover the shared
canonical `part` VP with `whole`'s declared `(argin2, argin1)` permutation. A legacy
record lacking this metadata declines a shared-VP ambiguity rather than
choosing a declaration-order form.

The live grammar style is **operator-role categories**, not a declared
part-of-speech taxonomy: every operator contributes its own
`<op>_argin1`, `<op>_argin2` (inputs) and `<op>_argout1` (output) categories. For
example, from `data/complete.grammar`:

```text
part_argout1 = part.forward(part_argin1, part_argin2)
lift_argout1 = lift.forward(lift_argin1, lift_argin2)
conjunction_argout1 = conjunction.forward(conjunction_argin1, conjunction_argin2)
```

The shipped `.grammar` files declare their symbolic rules in this role-only
form. See
[Role-Collapsed Grammar and the Operator Codebook](#role-collapsed-grammar-and-the-operator-codebook)
below for the full account, including how a word's category is
recovered from role participation rather than declared.

`RuleDef.lhs` / `.rhs_symbols` parsing (`Grammar._parse_category`) still
accepts an explicit conceptual-order suffix — `NP3`, `S4 = lift(NP3,
VP1)`, `NP*` Kleene forms — for backward compatibility, but that style is
now **explicitly legacy**: `NP3`, `NP4`, `VP1`, `MP1`, `S4`, and `S5` are
members of `_FORBIDDEN_STATE_TOKENS` in
`test/test_role_collapsed_grammar.py`, and no shipped grammar file
declares any of them. None is planned: orders are learned, not declared
([Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)).

### GrammarLayer forward / reverse inventory

![Operators: their arguments and outputs, read from data/complete.grammar](diagrams/operator_signatures.svg)

*The figure is generated from `data/complete.grammar` (2026-11-01) and fails to generate when an operator is missing from it. The [operator sheet](diagrams/grammar_operators.svg) adds what each face may read and write.*

Every operation's `forward` is what `<compose>` fires and its `reverse` is what `<generate>` fires. An operation with no faithful inverse raises (`raise_no_inverse`) rather than fabricate a split, which would corrupt the reconstruction.

**Form and meaning.** On a configuration whose conceptual width exceeds its form width (a meaning block), every grammatical and logical operation composes the two blocks separately (`GrammarLayer.compose_blocks`). The form block is the centroid of the operands' forms, each weighted by the number of leaves it covers; it has no parameters. The meaning block receives the operation: the declared exact rule for `not`, `non`, `conjunction`, `disjunction`, `intersection`, `union` and `sum` (`MeaningCodes.compose`); the diagonal action of `lift` and `adverb` (below); the case selection of `compound`; and one lane-preserving kernel (`MeaningKernels.LaneKernel`) for every other face, while `interpret`, `bind`, `transpose` and the field operations pass the head's meaning through. A kernel starts at the identity and keeps its derivative there, so a fresh model reproduces the head passthrough exactly; since centroids have no parameters, two faces with the same head compute the same value at initialization, and operator choice is learned by the chooser and its credit, not from forward differences. No kernel mixes the two evidence lanes. Leaf counts travel with the stack and the operation journal. `reverse_blocks` solves the centroid equation exactly given one operand and the counts, and inverts the meaning operation where it can; a binary inverse without a known operand needs a context witness or the candidate search, never a balanced pseudo-split. Configurations without a meaning block keep the numerical kernels the table names.

| op | arity | forward | reverse |
|---|---|---|---|
| `not` | 1 | exchanges the two evidence lanes, or an explicit pole pair; the form is unchanged | itself |
| `non` | 1 | withholds the expressed evidence without asserting its opposite: the for lane goes to zero, an explicit pole pair to $(0, 0)$; the operand stays in the journal as a withheld witness | none: raises |
| `intersection` | 2 | meet: for lanes by min, against lanes by max; numerically the coordinate min, a zero coordinate silent | candidate search |
| `union` | 2 | join: for lanes by max, against lanes by min; numerically the coordinate max | candidate search |
| `conjunction` | 2 | the meet on a meaning block; numerically product binding (table above); a repeated reference returns itself | candidate search through compose |
| `disjunction` | 2 | the join on a meaning block; numerically the probabilistic sum | candidate search through compose |
| `min`, `max` | 2 | numerically the coordinate min or max | candidate search |
| `sum` | 2 | arithmetic mean | `2*parent - witness` given one operand; free decoding searches the bank |
| `product` | 2 | numerically the Hadamard product | given one operand on a meaning block; numerically none (zeros annihilate) |
| `synthesize` | 2 | numerically adds two constituents, in the cube chart for native identity values | `analyze`: the other operand exactly, given one; with a basis the best row and its exact remainder; otherwise `(parent, 0)` |
| `lift` | 2 | NP + VP: the verb phrase's diagonal action on the subject; numerically the sigma fold `tanh(W(atanh a + atanh b) + b)` | the action's inverse given the verb phrase |
| `verb` | 2 | the verb (`argin1`, its head) sub-typed by its object (`argin2`); numerically the object's sparse log-spectrum applied to the verb (`apply_verb`) | numerically `unapply_verb` given the object; on a meaning block the object is searched |
| `adverb` | 2 | the modifier's diagonal action on the verb phrase, in one shared basis; numerically `apply_adverb` | exact given the adverb |
| `compound` | 2 | case selection: the modifier's learned test selects among the head's native cases (`CaseSelection`) | case search over the primed case bank |
| `lower`, `definite`, `universal` | 2 | the determiner: its marker contributes no content; numerically the description passes through | given the marker |
| `preposition` | 2 | a marker-headed phrase; numerically the phrase passes through | given the marker |
| `surface` | 2 | a marker attached to a complete content constituent (below) | given the content; numerically given either operand |
| `transpose` | 2 to 2 | exchanges the two newest constituents | itself |
| `lower_unary`, `definite_unary`, `preposition_unary`, `surface_unary`, `compound_unary` | 1 | the marker-headed operation without its marker (below) | from the operand alone; emits no leaf |
| `bind` | 2 | reads the chosen off-stack field entry, or keeps the mention's own content | no generate face |
| `interpret`, `sign` | 1 | word-concept to object-concept, or the sign's pointed-at token ([below](#interpret-word-concept-to-object-concept-item-9b-2026-09-25)) | lexicalization |
| `part`, `whole` | 2 | numerically the relation's whole (`part` returns its second operand, `whole` its first); the closing writes the relation | none: raises |
| `equal` | 2 | numerically mutual parthood `part(A, B) * part(B, A)` | lossy `(parent, parent)` numerically; otherwise candidate search |
| `generic`, `ask` | 1 | carriers that record the operation; the value passes through (on a meaning block, through the lane kernel) | the same, inverted |

`VerbLayer` and `AdverbLayer` subclass `LiftLayer`, `WholeLayer` subclasses `PartLayer`, and the unary faces subclass `MarkerUnaryLayer`. `EqualLayer` lives in `bin/Layers.py`, `InterpretLayer` and `SignLayer` in `bin/Interpret.py`, and all others in `bin/Language.py`. The legacy assertive relations `isEqual` and `isPart` remain only for explicit historical grammars.

**Two negations.** `not` exchanges a pair's poles, $(c^+, c^-) \to (c^-, c^+)$: an isometry of the resultant that keeps the dissonance and is its own inverse, the affirming negation; the clause it heads has its polarity flipped. `non` withholds the operand's evidence without asserting its opposite. Nothing is affirmed in its place: the clause it heads carries the evidence pair $(0, 0)$, and the operand survives only as a withheld witness (`ClauseJournal`), so reconstruction and expectation can read what was withheld. It has no inverse; it is the non-affirming negation of attention.

**The verb's action.** `lift` and `adverb` act on meaning through `MeaningKernels.MeaningAction`: the positive-definite map $Q\,\mathrm{diag}(e^{w})\,Q^{\top}$, applied with the same $w$ and $Q$ to each evidence lane in the cube chart and restricted to each lane's defined coordinates, so an absent coordinate stays absent and the lanes never mix. $w$ is a linear readout of the verb phrase's meaning, zero-initialized and soft-thresholded (threshold .1, straight-through gradient, bounded to $\pm 4$), so a fresh verb is the identity and a coordinate enters its support only when its push recurs; at $w = 0$ the action is an exact passthrough that keeps its derivative. For a transitive verb phrase, $Q$ is built from `verbBasisReflections` Householder reflections read from the object's meaning and confined to the object's support, the identity elsewhere. For an intransitive one, $Q$ is its verb group's basis (`VerbGroups`): a verb whose fit residual exceeds `verbGroupFit` on a third distinct occurrence joins the best-fitting group or opens one, up to `verbGroups`, so the shared bases are a joint diagonalization learned by gradient. The modifier uses one shared basis. The inverse solves the same restricted system. A numeral object is a count, not a basis: "plus 2" is the successor applied twice, by thought.

**Centered forms.** When an order-0 result is composed with whole witnesses, the active wholes of its span (`FormCentroids`), its form is the parts' leaf-weighted centroid $\bar p$ moved toward the wholes' average $\bar w$: $(1 - \rho/2)\,\bar p + \rho\,\bar w/2$, with $\rho$ the wholes' presence capped at 1, so $\tfrac12(\bar p + \bar w)$ at full presence. The retained inverse receives the same witness and removes the whole center before splitting the parts. The free inverse has no witness: it reads whole support from the candidates' own forms, compares the plain and the centered equation, and keeps the centered hypothesis only when its recovered form is finite and inside $[0, 1]$; no value is clipped. A candidate with no known spelling supplies no whole witness, and the inverse stays approximate.

**Synthesis.** `synthesize` adds two constituents and `analyze`, its reverse spelling, returns the other operand exactly when one is given. A phrase beyond the word, an idiom, is a synthesized concept admitted by recurrence. WholeSpace's analysis of the field, its cut by wholes, is a separate process and not the reverse of a synthesis.

**Surface form and markers.** `surface` is the operator of surface form. One lexeme with its inflected and reordered forms is one word with its transformations, and a language's freedoms of word order are transformations of one structure, not separate syntaxes. `surface` attaches a marker to a complete content constituent: the meaning is the content's, with its polarity, scope and structure, while both operands stay in the numerical carrier for the next choice; the chooser, never a word table, decides whether a word is a marker. `transpose` rewrites the two newest constituents in place, a 2-to-2 step that is exact and its own inverse, so a question read in its surface order can compose to the same meaning as its declarative. Splitting a word and inserting a silent element are not grammar operations: sub-word wholes nest inside the word whole ([Lexicon](Lexicon.md#word-forms-and-concept-orders)), and the word stays the unit of attention. A marker belongs to its operator. Each marker-headed operation (`lower`, `definite`, `preposition`, `surface`, `compound`) has a unary face that shares its binary implementation and is the operation without its leaf, so the zero plural, the omitted relativizer and the unstated subject are operators without their leaves. The absent marker is the operation's left identity where it has one (zero for `lower` and `preposition`), a learned marker for `surface`, whose map has no fixed identity, and for `compound_unary` the head's unfiltered field as its implicit object; on a meaning block the unary face is its binary kernel without context. Irregular forms are stored whole and block the rule. There is no morphology analyzer and no tense, aspect or morphology operation: tense and aspect are case selection over a verb phrase's phases ([below](#words-narrow-the-domain-of-discourse-2026-09-28)).

#### Concept events are opaque to the grammar ops (2026-09-13)

Percepts have explicit `.what`, `.where` and `.when` coordinates.
Concepts do not: a word that has been resolved to its object concept is a
code from a codebook lookup that has generalised over the where and when
modalities as well, so the conceptual event is a full-width thing that
cannot be cleanly divided (Alec, 2026-09-13). The CS grammar ops
(`LiftLayer`, `VerbLayer`, `AdverbLayer`, `LowerLayer`,
`PrepositionLayer`) are therefore sized to the muxed concept width and
compose and reverse the whole event; the split of a muxed event into
content, `.where` and `.when`, the tense shift of `.when` on lift and
lower, and the preposition's `.where` rotation that lived inside those
ops are gone. Splitting happens only where a percept or a symbol is
formed: the symbolic layer muxes and demuxes around `execute`, and the
perceptual layer reads coordinates.

#### `interpret`: word-concept to object-concept (item 9b, 2026-09-25)

Decided (Alec, 2026-09-25; amended for DEF rows on September 29). The
resolution the paragraph above describes — a word resolved to its object
concept — is a declared `<compose>` operator:

```text
interpret_argout1 = interpret.forward(interpret_argin1)
interpret_argin1 = interpret.reverse(interpret_argout1)     # generate face: lexicalization
```

Unary, like `not`; its second argument is implicit, the current attentive
field. `interpret_argin1` is the **word-concept**: the code the word arrives
as, which PartSpace has already looked up as the recurring unit the fold
ladder admitted — `interpret` is never the byte-to-word step.
`interpret_argout1` is the **object-concept** already associated with the word,
at whatever order that object has. For example, *cat* returns a known cat
kind if that is its association. Grammar selects among ambiguous existing
associations. A new object replaces the word in its inventory row, with
that row's order; no singleton fold raises it merely to bind a word.
A default call never adds a particular beside a known kind.
Reference orders follow
[Lexicon](Lexicon.md#word-forms-and-concept-orders); the operator never
reads the surface form, and no word is anchored to it.

It is not a mode and not chooser-routed: **every arriving word is
interpreted under every binding**. Its parts fuse before lookup, and its
concept retains its parts and wholes. Presence requires the part inside
the word's bracket; a shared property never evidences an absent word.
Re-reading the same canonical parts mints and writes nothing, regardless
of changes to the learned property reading. Different parts remain an
alternative witnessed definition.

A new word is admitted pending, with capacity reserved for itself, its
object and their definition row; its object comes later, when a kept
defining clause or the recurrence of its mentions promotes it (below). The
word continues to exist by identity in `word DEF object`. DEF operands are
symbols named by identity, with null operand vector slots. `interpret`
alone writes these rows, and the closing alone writes assertions. Where
the field discovers objects, the DEF row is completed when that field
admits its case. Word admission never takes the case's place. Refusal
leaves no partial word or definition.

One derived index provides form/unit → word, word → objects, and
object → words. It owns generation's lexical inverse as well as forward
resolution; learned-code changes do not change its answers. The META
triple, its fold, `ReferenceTable` and the allocator's duplicate binding
records are retired. Old META checkpoints migrate to DEF rows.

The eager word transaction runs before a graph; the serial tensor face
publishes the resolved object's activation and atom. Sentence boundaries
retain recognized-word and category-codebook work. A DEF row has its own
fixed `.when`; re-reading refreshes its recency timestamp. It can be
forgotten, after which the word is interpreted afresh. It has ordinary
row provenance and does not inherit the sentence's source trust.

**Amended (Alec, 2026-11-01): `interpret` is optional and initially forced;
its interpretation is a sign or a symbol.** "Every arriving word is
interpreted under every binding" above stands while the forcing holds, and
in reading mode it holds: Alec, "interpretation can be initially forced
since words are so often meaningful beyond themselves." What the
architecture says is that `interpret` is optional, a leaf at which it is
not applied being the no-op, Peirce's icon, where the percept stands for
itself and the leaf value is `interpret_argin1`, the word-concept code; and
that an interpretation, when it happens, is one of two readings chosen by
the compose chooser with ordinary credit: `sign`, Peirce's index, the
operation first declared as `witness`, which opens the pointed-at
occurrence's MTM token with the null-concept as its kind; or the symbol,
the object above, the interpreted leaf left standing as that object, with
no operation of its own. The eager word transaction remains the admission
of the form and the lookup of an existing association, mints nothing, and
is not the reading; the derived index and the DEF rows are unchanged. No
word is anchored to any reading, and `sign` is available at every
interpreted leaf. The forcing is lifted only with a field, or inside a
certificate ([Orders §11](Orders.md), "The three readings forced").

**Grounding is textual (Alec, 2026-10-19).** The model perceives word forms (letter parts and wholes) and forms word-concepts from them. A word-concept references an object-concept of any order: "Bill" an individual, "water" a mass, "chair" a count kind. Since the model experiences only text, a word gets an object-concept of its own, minted when a kept defining clause or the recurrence of its mentions promotes the word's pending identity (`FamiliarTier.define_pending`, `promote_candidates`), and the model learns about that object from what the text states: a definition ("a table has a top and four legs") adds parts to the conceptual mereology, and a kind statement ("tables are furniture") adds to the conceptual hierarchy. There is no perceptual grounding stage. A newly minted object-concept is nearly empty, resemblance between words arrives through what the text says of them rather than through percepts, and lessons demand behaviors, never non-empty meanings.

#### Tensor reverses for the compiled loops (2026-09-13)

`LanguageSpace.reverse_binary_step(parent, op_local, valid, reference,
inverses)` and `reverse_unary_step(x, op_local, valid)` are the fixed-shape
forms of the inventory above for a `torch.while_loop` body: every op's
reverse is evaluated on the parent and the recorded op selects
(`local_op_from_rule_ids` inverts the rule map).
`generate_policy` is the conceptual decoder for both reconstruction and output,
including numeric-answer configurations. It infers a binary undo, unary undo
or STOP from the current root/top. Binary undo searches the primed echoic
shortlist, including the sentence's own symbols and weighting candidates by
activation × cosine × priming. Unary undo invokes the operator's **generate**
face. The journal, stamps, rule sequence and operand positions never choose
decoder actions. Compose-only declarations expose those same operators'
generate faces implicitly; explicit generate declarations keep their catalogue.

Eligibility at the top of the walk (`decoder_eligibility`) compares the pair search's squared reconstruction residual with the best one-code least-squares explanation of the top; there is no tunable purity cutoff. A supported pair takes precedence over a single-symbol projection, and at equal fit two readable, progressing children are preferred, so a known synthesized concept is undone into its parts. Emitting a symbol is legal only at a singular top: a tie between different one-code explanations is not singular, and an inverse that merely repeats its parent is not a decomposition. STOP against undo is therefore eligibility, not a learned margin; the policy learns which operation to undo.

Reconstruction owns the generate chooser, parameterized faces and codes.
Its hard stack choices have the straight-through numerical transition described
in [GradientFlow](GradientFlow.md). Output owns the question conditioner and
reads a detached conceptual state and shortlist. The byte loss, gate read-back
and output spelling use this same walk and shortlist. Clause closing still
reads its own journal to preserve the semantics of the operation that ran;
that journal is not a decoding recipe.

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
role-only grammars every category is order-flat (`part_argout1`, `lift_argin1`,
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

*Moved to [DevelopmentHistory](DevelopmentHistory.md#stm-shiftreduce) (2026-10-30): "STM Shift/Reduce".*
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
and result roles, named `<op>_argin1`, `<op>_argin2` (inputs) and `<op>_argout1`
(output). The rule `equal` therefore exposes `equal_argin1`,
`equal_argin2`, `equal_argout1`, and the grammatical "category" of a span is
just the set of operator roles it can fill. Dimensionality is recovered
from participation (`bin/participation.py`) rather than declared up
front: symbols that fill the same roles cluster into the same category.

The grammar declares its perceptual start and its symbolic start separately.
The property tower has no grammar dictionary. The symbolic sentence finishes
as one `S` slot or three `S REL S` slots. A relation's middle slot holds the
native identity of `part`, `implies` or `operator`, supplied by the declared
thought registry. Numerical operators live in LanguageSpace and their
conceptual identities live in ConceptualSpace.

### Chooser architecture capacity

The grammar placement policy and the thought-step policy are separate modules.
`MLPTransformChooser` scores a grammar operation at a slot/pair from its current
state, candidate result, role context, tool embedding and position. With
`transformChooser=mlp`, `<architecture><transformChooserHidden>` selects its
hidden width (`0` means `max(8, d_model)`) and `transformChooserDepth` selects
the number of hidden Linear/GELU blocks (default `1`). Every block has the same
width; a scalar Linear head follows. The 29-dimensional What context still
enters through a separate linear operation bias, not through this MLP.

Binary heads retain operand and role **order**: means enter the established
feature layout, while signed left/right differences enter the same first hidden
layer through `operand_order`. This is an input projection of the existing MLP,
not another language policy. The projection starts at zero without changing
the RNG stream. Old checkpoints extend with zeros, preserving predictions and
existing optimizer moments; current checkpoints save its learned parameters.
Unary heads retain their existing layout. Raw WORD rows and native IDs never
enter these numeric features.

This grammar path owns natural word → operator learning. Numerical learning
inside a declared operator is allowed; a fallback interpreter or pre-generate
language decoder is not. Technical anchor spellings such as `partOf` and
`isEqual` are explicit syntax. Ordinary words, including `equals`, are not
bootstrap aliases. Structural preference and language-quality studies remain
the design goals in [SelectedMeaning](SelectedMeaning.md).

Thought is chosen by the compose selection layer: `OperationSelectionLayer.thought_logits`
scores the operations offered at a boundary and `conclude` from the masked root,
active and candidate roles and their context, and not thinking is the chooser's
free first choice; credit is the whole-path score on $R + A$. The older
`SelectedThoughtChooser` class remains in `bin/Language.py` for
[its tests](../test/test_chooser_architecture.py), but the model does not build it,
and `whatThinkingHidden` and `whatThinkingDepth` are parsed but unread.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#soft-operator-superposition) (2026-10-30): "Soft Operator Superposition".*
### Learned argument concepts

`LanguageSpace.parts_of_speech` maps every declared formal input and output to a prebuilt native concept, including distinct semantic faces and both outputs of transpose. Word concepts have native order 0, these argument concepts order 1, and classes over arguments order 2. Declaration gives each argument a seat and a zero vector, radius and view count; it supplies no learned evidence.

Each carried lexical occurrence in a kept application supplies a member view through `ConceptHierarchy.observe_member`. Members teach the same centroid and view-and-agreement confidence as other explicit classes. Shared words can belong to several arguments and bring their directions together. The general concept hierarchy also conditions members from their higher-order context, so word meaning and class meaning influence one another. Parts of speech have special construction and grammar use only; their learning and hierarchical feedback are ordinary concept mechanisms. Operator names are not inserted into lexical context. Annotation operands and unkept trials supply no member views.

The chooser's existing argument carrier reads candidate coverage of the radius-bearing class vector by a word's unit meaning direction. Scores add once per carried occurrence, without normalization or exclusive labels. Zero classes score zero. Output arguments receive member views; compose reads input occupation and generate reads both input and output occupation. Operation authority is the minimum input definiteness, an argument's separability times its view evidence; separability is the fit-weighted fraction of its members that fit it strictly better than every populated argument outside the 15-degree contiguity angle (`CLASS_CONTIGUITY_COSINE`), so contiguous classes do not compete. The fixed and identity-mechanics exemptions remain one. Empty measurable arguments supply zero readiness. A form-only chart supplies neutral operation authority and undefined clause readiness. Kept-use counts remain telemetry. Centroids refresh once at the kept boundary; only observed members and their ancestors receive contextual conditioning.

Word flow (`PartsOfSpeech.flow_mask`): symmetric `intersection`, `union`, `conjunction`, `disjunction`, `product` and `sum` retain every operand's words; `not` and `non` retain their operand's words. Headed operations retain the declared head operand's words, including lower/definite and their unary faces and verb. Adverb declares argin1 as head. Clause-forming operations retain none, including unary ask and generic. Thus both black and cat in `lower(a, intersection(black, cat))` occupy lower's second argument and travel with the description. Transpose swaps both carriers; repeated occurrences remain repeated contributions. Generate reads parent argout and child argin occupation as policy features, as described below. The symmetric rule follows the coordination of likes: a symmetric result keeps its inputs' type, as a coordinated result has its conjuncts' type, and unlike conjuncts that satisfy the same selector ("a Republican and proud of it") fill the same formal argument, which is what argument-level classes provide. A selection such as tense applies once to a coordinated verb phrase, so conjuncts stay one part of speech; where a language uses different conjunction words by type, the difference is in the marker word, learned by exposure, not in the operator's result type.

Argument vectors, memberships, orders and confidence use ordinary native concept checkpoint storage. Argument schema 4 saves the grammar-to-concept mapping and scalar uses; old occupation-count checkpoints require producer replay. Geometric cosines remain reported for inspection. The retired category VQ and probability/argmax table have no replacement side bank. The corpus controls the teaching sequence: word concepts, object concepts and early simple clauses/questions, then further operations; nouns and verbs precede the other parts of speech. The NP sequence below is the retained lesson-specific record, governed by the current curriculum specification.

The NP curriculum scopes the existing operation validity masks to STOP,
`lower`, `definite`, `intersection`, `product` and `bind`. Reference-expanded
candidates remain unchanged. The bracket lesson labels the student's own
visited states and teaches operation choice; it never supplies member views.
An adjective-noun annotation names a `modification` bracket, not an operator:
its fold target sums the admissible `intersection` and `product` actions
(`GrammarLessons.MODIFICATION_OPERATORS`), a modification bracket requires a
student-state lesson, and which of the two the student keeps is reported,
never required. The curriculum introduces one unknown at a time, against
material already established by a previous lesson. The first NP lesson
teaches one determiner construction over stage-3 words; the other determiner
and adjective composition wait for its result. Bare adjective-noun
descriptions teach modification before a later lesson nests it inside a
determiner NP. This changes the examples, not UG: there is no noun-aware mask
or head rule. The corpus and budget are fixed before training.

## One operation per round (item 7.5)

`OperationSelectionLayer` scores every binary operator at every adjacent live
pair and every unary operator at every live position. One model softmax
covers all of those candidates and, once the sequence fits its LTM row, STOP.
Exactly one candidate fires per active round. Binary operations remove one
operand; unary operations preserve length. Above the row's slot limit there
is no COPY, wait, or other no-operation candidate. STOP is eligible at depth
one for an absolute row and at depth three or below for a relative row.
Every menu has a fixed layout: the number of variants per operation and the
candidate each slot denotes depend only on the row's own state and on global
sizes, never on other rows of the batch, and an unavailable candidate is
masked, never removed, so a flat action index names the same candidate in
every path.

The tensor slab and round budget stay fixed. An active mask freezes a row after
STOP; it does not change the compiled shape. The parallel budget is at least
twice the slab width. Serial STM uses this same owned layer on its newest two
slots, with three rounds after each word and a closing budget of twice STM capacity
(bounded by a positive `syntacticOrder`). Occupancy and the STOP test use the
whole STM's depth, not just the two-slot window. Online rounds allow `K - 1`
occupied slots, reserving admission for the next word. Endings allow one absolute
slot or three relative slots; two slots are a transient state, not an allowance.
An absolute sentence ends in one slot because its NP and VP fuse to one point:
the primitives of the model's reality are spacetime events, and a sentence
names one (`sentence_row_slots`). A relative sentence, S REL S, keeps its two
sentences and the relation in three slots.

With depth `d`, allowance `a`, and `r` rounds left including the current round,
the required reductions are `n = max(0, d - a)`. When `r - n <= 0`, only binary
candidates remain legal. Before that deadline, unary operations remain legal;
STOP also requires both the row limit and the phase allowance to fit. Every
binary logit receives `reducePressure * (d/a + n/r)`, with zero pressure on an
empty stack. `architecture.reducePressure` defaults to `1.0`; the fixed prior
adds no learned parameters and enters the model's gradient credit. See
[Params](Params.md) for the numerical guards and the declared measurement value.
Three online rounds and `2K` closing rounds suffice to admit every word and finish
each row. A deliberately infeasible initial budget may still yield an incomplete
forest that trains without publishing a row. Arrival at a full stack is an
assertion failure; compose has no overflow-dropping or overflow-incomplete path.

In training, compose is one walk of a whole path
([Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11)).
One XML parameter, `architecture.composeTemperature` (finite, nonnegative,
default `0`), controls the greedy path's hard choices: at zero the greatest
logit wins, exact logit ties keeping item 8's structural preference, and a
positive temperature samples `softmax(logits / composeTemperature)`. Each of
`explorePaths` alternatives reuses the greedy prefix and departs once: a walk
is drawn uniformly among the walks that still offer an alternative
(attention, compose, candidate, bind, reconstruction, thought, output), a
round uniformly within it, distinct from earlier departures, and at that
round an action other than the greedy one is drawn from the proposal
$q = f\,u + (1-f)\,\mathrm{softmax}(\ell / T)$ over the eligible alternatives
that change the result, with $u$ uniform over them, $f$ =
`explorationUniformFraction` and $T$ = `explorationTemperature`
(`SentenceCredit.departure`, `Expectation.alternative_probabilities`).

Perception runs once and all paths share the input. Each stream keeps its
strictly cheapest complete path on R + A, ties keeping the greedy path, and
only then commits its program, STM, LTM and thought effects before the next
sentence is perceived; keeping the cheaper neighbour is a hill climb in
derivation space, one departure at a time. There is one optimizer step per
sentence, and the batch clock and public training counter advance once.
Evaluation runs the greedy path alone, with deterministic logit argmax, no
exploration and no optimizer.

For selected candidate `c` with probability `p`, the straight-through value is
`detach(c) + (p*c - detach(p*c))`: forward execution is hard while both the
operator and chooser receive the probability-weighted derivative. `p` is the
untempered model softmax including reduction pressure and deadline masks, before a departure excludes the greedy action;
neither temperature nor that exclusion changes the credit distribution. Reconstruction
and the existing task costs train compose directly through this derivative; a
departure also receives its whole path's change in R + A, corrected by the
drawn action's proposal probability
([GradientFlow](GradientFlow.md#whole-path-credit-and-concept-episodes-2026-11-11)).
Compose has no DP-prior objective. There is no tiling forward/backward,
Viterbi pass, expected tile compaction, length DP, separate unary layer, or
retained switch to run them.

Alec's rationale: training by softmax lets one gradient do all the optimization;
DP is the pre-backprop symbolic solution and should not be mixed into a working
MLP. The sampled alternative is a tractable approximation to the full
superposition over operators and locations, not an exact marginalization.

## Words narrow the domain of discourse (2026-09-28)

**Item 6.5 mechanism, accepted by Alec (October 7).** An operation that
requests an individual now has variants for each bounded situation column
and, where permitted, mint. All variants use the grammar operator's existing
scoring parameters in the one global operation softmax. The numerical journal
keeps the original operator index and the chosen reference. The old local
reference argmax does not participate in this production path; ambiguous
interpretation returns unresolved instead of raising. Kind/extension
operations choose no individual. Mint is a choice during reading and a
recurrence-gated effect after the kept closing. No word spelling anchors any
of these choices. *Superseded in part (2026-10-19, 2026-10-23): `bind` is the
only binder and the one compose operation that chooses an individual;
determination supplies an object, never a reference alternative, and an
operand reads its existing typed concept at every order but one
([Orders §10](Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3)).*

Expectation and population sparse-coding costs supply the existing sentence
choice credit. The columns' meanings exclude reference geometry; situated
operands retain it as binding evidence. Tests of this mechanism do not prove
that natural determiners or pronouns have learned the intended routing.

Decided in direction by Alec on 2026-09-28: words are a formula for
narrowing attention, each word a projection operator that narrows the
domain of discourse, the register the situation seeds and the determiner
individuates from. The consequences for the architecture are in
[Architecture](Architecture.md#words-are-a-formula-for-narrowing-attention-2026-09-28);
this section gives the language mechanics. The binding mechanism above is
implemented; the full narrowing account and its natural-language behaviour
remain unmeasured.

**The parse.** "A fake gun" is `lower(fake(gun(thing)))`. `thing` is the
domain of discourse. `gun()` and `fake()` are the same kind of operation,
each a projection that narrows it, and they commute, so
`gun(fake(thing))` means the same. The determiner's `lower` individuates
from what is left. The operator that narrows nothing is the symbol
`everything`, the top pole.

**Parts of speech are roles.** A codebook row is an operator. Read against
the whole domain it is a noun; applied to a narrowed domain it restricts.
English shows both directions: "a fake" is a noun, and in "gun oil" `gun`
restricts. No word is anchored to an operator, and a row is routed per
frame by learned credit.

| word class | what it does | index |
|---|---|---|
| noun, adjective | narrows which thing | |
| determiner | individuates: "a" mints, "the" binds, "every" does not lower *(2026-10-19: a cue input to `bind`, the only binder)* | thing |
| verb, adverb | narrows which stretch of time, and what changes | |
| tense, aspect | individuates the time *(position over the verb phrase's phases, below)* | time |
| conditional | narrows which alternatives are in play | |
| modal | individuates: "might" some, "must" all *(position with a world as object, the second difference; a modal `adverb` withholds the lowering; default all possible worlds)* | alternative |
| deictic: pronoun, indexical *(a reading, not a built class; Alec, 2026-10-31; [Orders §11](Orders.md), "No forcing; the class is the raise's outcome" and "The identity that empties is the identity across time")* | points: a word minted at its first definition and raised at `raiseCount` distinct relata, whose identity across time, what persists across the things it was, empties as they diversify, so it could be a symbol for anything; its mention then constrains nothing, and location, recency and expectation decide the bind; an indexical's candidates come from the field once there is one (future work) | thing |

**Which operators commute.** One that acts the same on any domain commutes
with the others (*fake*, *wooden*, *noble*). One that reads its scale from
the domain it is given does not, and must follow the noun (*big*,
*skillful*). Compounds are the plainest case: "gun oil is a kind of oil; oil
gun is a kind of gun", so order of application matters (Alec, 2026-09-28).
The Ground the modifier is read against is the head's own shape, its cases
at every order, which "already exists in virtue of the folds at every
order, so no new parameterization should be necessary". A head must
therefore have a shape: "'gun Felix' does not work precisely because Felix
is a proper noun, not a kind". Modification keeps the head's order, "oil
"Order only drops when the determiner is applied, adjectives do not perform
that role." (*Amended, Alec, 2026-09-29:* "the order drops at the
determiner and nowhere else" is "probably wrong"; what stands is that
adjectives do not drop it.) Order and part of speech are different things and "only
sometimes correspond": "Felix is a cat" and "cats are animals" are both
taxonomic relations and both can take part in the sigma fold, the first
between a proper noun and a count noun and the second between two count
nouns. A set and a part are different relations: "cats and dogs, two
discrete concepts, create animals, which is then necessarily of a different
order", while "blue cats is a part of the extension of cats" and keeps its
order. The determiner moves from a set to a member; the adjective moves
from a whole to a part. A compound is neither: "oil gun and gun oil may be
interesting cases of sub-typing without creating a new term, which is
masquerading as an adjective" (decided, Alec, 2026-09-29: "Compounds are
subtyping, which explains their order effects"). It sub-types
its head, so the order of its two nouns matters, and adjectives are left
free to commute. It does not lower the order: "I don't think compounds do
the work of determiners, but they select from the determined set. An oil is
a kind of oil, a gun oil is a specific kind of oil."  The order drops at the
determiner; the words "and nowhere else" are withdrawn (Alec, 2026-09-29).
The test is whether the word can stand as a noun over the
whole domain. In both cases the domain is kept as the Ground against which
the narrowed Figure has its meaning; it is not discarded as perceptual
attention discards what it excludes.

**Modification (Alec, 2026-10-18).** Intersection cannot serve every adjective: a relative adjective reads its scale from its noun, so what is small for a big noun is larger than what is small for a small noun, and a function relative to its argument, such as `product`, fits it better. The grammar offers both `intersection` and `product`, and the chooser picks; a lesson's adjective-noun bracket names the constituent, never the operator ([Learned argument concepts](#learned-argument-concepts)). A noun is an adjective pre-applied to the universe, the all-ones vector that is the product's identity. Every operator has an identity element, and an operator applied without its written operand is applied to that identity, so a unary face carries its operation's identity where it has one (NP = everything, VP = occurs, MP = all possible worlds).

**Position.** Tense, aspect, the preposition and the modal are one operation: unfold the head into its cases, select among them by the object, fold what is left. The head decides what is selected: a noun phrase's cases are things laid out in space, a verb phrase's are its phases along the change (the state before, the change, the state after). The object decides the test: a place compares location ("sat on the mat"), a time or an event compares order ("the meeting after lunch"), a world compares possibility. A preposition makes an adjective or an adverb by this selection, as a compound selects among its head's cases, and tense and aspect are a verb phrase's prepositions whose object is now or a reference event R and whose preposition is not written. Past keeps the phase in which now falls after the change, future the phase before it; the progressive asserts only part of the change, so "was crossing" does not entail "crossed". A selection is idempotent, not a gain: "on the mat, on the mat" says no more than "on the mat". So an adverb of degree is a gain, while an adverb made from a preposition is a further restriction of the event. Two selections (past, then perfect; a path with two landmarks) are the operation called again, and a selection written as nothing is the call without a leaf. Not calling it keeps every case, which the answer reads as unknown, never as now. The claim is universality: any way a language writes a selection of time or place reaches these operations by exposure under one configuration, and a language that needed another operation would be a finding against them, never a case added.

As built, the selection is `compound` and its absent-marker face `compound_unary`, which share one kernel (`CaseSelection`). The head supplies its cases (a noun phrase's native cases; a verb phrase's before, change and after from an accessible actual event's snapshots), and the object supplies content, including a referred-to event for the reference time R: the reference time is content, never a `.when` column. `CaseSelection.CaseView` keeps the head's cases, their two-lane memberships and an accumulated content test; a selection intersects the test, so repeating the same selection is idempotent, and the fold refolds each evidence lane without subtracting one from the other. `CaseFlow` carries the view beside the conceptual stack; the case bank is staged once per reading, before parsing, and no clock or lexical class enters it. The absent-marker face uses the staged reference, or the head's unfiltered field, as its implicit object. A phase question is certified only when each lane covers the whole requested content: no selection, an empty selection or a proper fragment answers unknown, and an expired event history reports `phase_history_expired`. A phase selection keeps the verb phrase's order; only a modal `adverb` withholds the lowering. The preposition itself is a marker-headed phrase (`preposition`, with its unary face) whose content is its object.

**"Every".** "Every cat sleeps" does not lower: "it remains a high-order
relation" between the concepts at their own order, the subsumption
`sleeps(cat) = cat`, and it is lowered only when thought applies it to a
particular. The grammar rule `lower(DET, NP)` therefore covers "a" and
"the"; "every" marks the subject generic, which writes a
relation row. *Amended (Alec, 2026-09-30): "every lowers like all but has a
different plurality", and a bare plural lowers by an implicit plural
determiner, "all" or "some"; the determiner row of the table above is
superseded.*

**Verbs.** A verb adds dimensions to the description and removes freedom from the thing described. The noun is silent on when, and on what the thing does; the verb constrains those dimensions. A verb phrase is an element of a group acting on its identity's frame: a map from the state before to the state after, with an identity element (the verb that changes nothing; an undefined verb is that element, read as unknown), an inverse (the change undone) and composition. It acts as a diagonal in a basis, $Q\,\mathrm{diag}(e^{w})\,Q^{\top}$ in the cube chart with $w$ sparse ([the verb's action](#grammarlayer-forward--reverse-inventory)), so it is the identity wherever the verb is silent. The basis is shared by a verb group, the verbs that change the same thing (motion, change of state, transfer), or for a transitive verb is supplied by its object. The object is not a second operand of the sentence but a modifier of the verb phrase that sub-types the verb, as a compound's modifier sub-types its head: "chased the mouse" is a kind of chasing as "gun oil" is a kind of oil. The verb is the head, and the object is kept by reference, so subject and object are not confused. The action is taught by the identity's observed state after: its prediction from the state before is compared with that state in the chart, on the coordinates both define (`VerbLearning`), which for a diagonal teaches the log-ratio of after over before; where the before-state is zero nothing is learned. Selections and changes are two algebras: a selection is an idempotent meet with no inverse, while the verb proper is invertible and not idempotent (ran twice is more than ran once). States are relations, not group elements, and a verb with several participants is the verb-phrase rule applied again, each application its own diagonal on its own participant.

**What the code does today.** Adjective-noun modification composes by `intersection` or `product`, whichever the chooser keeps, and `lower` is the determiner. On a meaning block `intersection` is the meet of the evidence lanes, idempotent and commutative, as decided on 2026-09-29 ("a red red bird is no more red than a red bird"); `product` there is a lane kernel of one operand with the other as context, the identity at initialization, which can learn a modifier relative to its noun. `compound` selects among its head's cases, so "gun oil" and "oil gun" differ by their heads. Without a meaning block the kernels are numerical: `intersection` the coordinate min, `product` the Hadamard product, `lift` the sigma fold. The verb's soft threshold (.1) and bound ($\pm 4$ in the meaning action, $\pm 8$ in the numerical spectrum) are hard-coded.

*Moved to [DevelopmentHistory](DevelopmentHistory.md#future-work-nouns-from-partspace-adjectives-from-wholespace) (2026-10-30): "Future work: nouns from PartSpace, adjectives from WholeSpace".*
## Exist and thought operations at the completed boundary

`exist` has no compose face; the default verb phrase is *occurs*. At a
completed boundary, the grammar-owned `exist`
thought operator reads accepted LTM evidence for the complete description,
preserving occupied roles, scope, bindings, references, and both support
polarities. It cannot execute during composition.

`part` reads native conceptual-taxonomy references; `equal` receives full-width
concepts; every other thought operation receives only its declared capability
views. See [the common contract](QueryContracts.md).

## Common grammar-face signatures and shared VP identities

`Grammar.configure` rejects the legacy `<Queries>` spelling and `query`
attributes before changing existing rules. It derives structural contracts from
role-labelled `<compose>`/`<generate>` faces, then builds its immutable
catalogue only from the model's explicit `<thought>` declarations. Static
anchor strings remain whole when the loader expands order alternatives.
`complete.grammar` includes the structural `ask`, which marks a completed
question interrogative; the retired `what` raises at load.
[Declarations](../bin/Language.py).

The explicit shared-VP adapter forms `[NP1, VP, NP2]` with native references,
mode, polarity, scope and bindings. `whole` reverses surface operands into the
same canonical relation as `part`; open `argin1` returns parts and open `argin2` returns
wholes. Formation does not execute. A selected question dispatches from its
middle VP and occupancy through the grammar-selected descriptor. No keyword
matching is used in this API.

`LanguageSpace.program_meaning()` is the current direct linguistic derivation
adapter. It recovers a completed binary relation from its immutable local
compose-action snapshot and preserves the actual signed leaves that a lossy
fold may have discarded. It also accepts an unreduced lexical
`[NP1, VP, NP2]` program when—and only when—the middle reference names an
installed native thought VP. That reference selects grammar identity and
provenance; the arbitrary surface VP tensor is not reinterpreted as a semantic
value. The registered canonical VP supplies the middle role while the two live,
signed noun leaves supply the operands. A declared outer `not` or `non` changes
canonical polarity without changing the retained operands. The adapter
also recovers a direct-leaf unary thought whose descriptor takes exactly one
full-width `concept`: its registered
native VP supplies the grammar role and the selected signed leaf remains the
live operand. A unary `description`/reference operation (`exist`, for
example) still requires an actual owned LTM or live-thought occurrence; a word
concept ID is never converted into one. The adapter intentionally declines a
nested physical fold until it has a stable existing occurrence reference;
declining it is safer than flattening, rebinding, or inventing a semantic role.
So the direct relation and direct concept-unary routes are live, while broad
sense selection, paraphrase realization and general syntactic nested-clause
adaptation remain separate work.
[Formation](../bin/Queries.py),
[dispatch](../bin/Queries.py),
[full contract](QueryContracts.md). Per-row thought permission is now enforced
at the completed answer boundary; the normal linguistic derivation adapter,
sense selection and paraphrase realization remain separate work. See
[Query phases](QueryPhases.md).

### Generate reads learned argument concepts (item 2.0, 2026-11-09)

The argout faces taught by `PartsOfSpeech.observe_kept` now condition generation.
Each menu action supplies three occupation scores: the parent's fit to argout1,
and the proposed left and right children's fit to argin1 and argin2. The staged
argument directions and commitments use `PartsOfSpeech.argument_scores`'s
candidate-coverage measure on normalized meaning coordinates. STOP and an
undeclared fixed face supply zero. These are policy features, never eligibility
masks; their new weights start at zero. The two free generate walks use compose's
configured proposal mixture and paired-cost score-function credit. Explicit
generate grammar lessons remain; the compose-tree walk teacher is retired.
