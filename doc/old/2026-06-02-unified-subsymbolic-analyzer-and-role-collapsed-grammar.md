# Unified Subsymbolic Analyzer + Role-Collapsed Grammar

> **Status (2026-06-02): unified spec.** This document is the single
> self-contained end-state spec for the perceptual analyzer + the
> role-collapsed symbolic grammar. It folds in and supersedes:
>
> - `doc/plans/2026-05-30-subsymbolic-analyzer-terminal-emitter.md`
>   (the PS analyzer / terminal emitter; **largely implemented** -- see
>   *Implementation status*).
> - `doc/plans/2026-06-02-complete-grammar-role-collapse.md`
>   (the role-collapse grammar rewrite; implemented as
>   `data/role_collapsed.grammar` and **promoted to the default** mental-model
>   grammar on 2026-06-03 once the D1 gate was met).
>
> It is written to be handed to a fresh conversation: it states the full
> target, what is already built, and the remaining phased work. Where this
> document and a source spec disagree, this document wins (notably the I/O
> role-naming convention below, which corrects the source role-collapse
> spec).

> **Implementation update (2026-06-02): Phases R1--R5 landed.** All five
> phases are implemented and tested.
> - **R1** (loader/runtime): space-scoped starts, `isPart` + `query`
>   dispatch, relative-rule detection over `{isEqual, isPart}`,
>   SymbolicSpace-scoped identity rule (`bin/Language.py`; tests
>   `test_grammar_space_scoped_starts.py`, `test_ispart_query_dispatch.py`,
>   `test_relative_rule_detection_collapsed.py`).
> - **R2** (role-collapse grammar): `data/role_collapsed.grammar` (operator
>   roles only) is now the **default** mental-model grammar
>   (`MentalModel.xml`, 2026-06-03, D1 gate met); `data/complete.grammar` is
>   retained as the compatibility baseline (test
>   `test_role_collapsed_grammar.py`).
> - **R3** (learned PS router): `MeronymicRouter` reusing the shared
>   `binary_tiling_viterbi` / `binary_tiling_soft_dp` primitive
>   (signed-neighborhood evidence, one hard route + soft marginals, depth
>   penalty), wired into `MeronymicAnalyzer.analyze_routed`
>   (`bin/perceptual_analyzer.py`; tests `test_ps_meronymic_router.py`,
>   `test_ps_analyze_routed.py`).
> - **R3-live (2026-06-03)**: the analyzer is now the DEFAULT
>   PerceptualSpace front end (`<chunking>analyse</chunking>`, the default in
>   `data/model.xml` + code). The default available op is a space-lexer:
>   InputSpace passes an unanalyzed host surface for `analyse`, PS splits it
>   into word runs, and those runs resolve through the word codebook. This
>   preserves the default word-lexer model behavior while deliberately making
>   byte-lexer `analyse` differ from byte-level `lexicon`: PS owns the live
>   segmentation. Bottom-up word learning is `PerceptualSpace.learn_merges`
>   (BPE-style, whitespace-bounded) + `chunk_static` mode='analyse'; cold
>   standalone analysis is byte-level and learns words from characters by
>   merge, converging to word lexing.
>   `bin/Spaces.py` (`_embed_analyse`, `chunk_static`, `learn_merges`); tests
>   `test_chunk_static_analyse.py`, `test_analyse_word_learning.py`,
>   `test_analyse_chunking_forward.py`. **C done (2026-06-03):** InputSpace
>   hands PS the unanalyzed, UNTRUNCATED surface (`_analyse_surfaces`, NUL
>   padding stripped) for `analyse`, so the analyzer's space-lexer is not
>   limited by the legacy `nWhat` `.where` byte width -- byte-lexer `analyse`
>   yields full word runs (`hello`) where the legacy lexicon path clips to the
>   byte-buffer width (`hel`); the lexicon path is unchanged. Tests
>   `test_forward_analyse_is_not_limited_by_token_byte_width`,
>   `test_forward_analyse_space_lexer_owns_full_surface_lexing`.
> - **R4** (dimensionality-from-participation): `bin/participation.py`
>   recovers the role-collapsed op-roles from the transitional grammar's
>   role participation, and `participation.learned_collapse` drives the
>   parser-recovering collapse the corrected D1 gate requires (see the D1
>   block below). With D1 met, role-collapse is now the default -- tests
>   `test_participation_pos_recovery.py`, `test_d1_pos_recovery_gate.py`.
> - **R4-sem (2026-06-03)**: grammatical categories distinguished by the
>   SEMANTIC EFFECT of their operators (= the operator-codebook vector). The
>   live codebook is shaped from truth/consequence through the soft
>   superposition (`SymbolicSpace.shape_operators`, generalizing R5 to the
>   live `_operation_vectors`); a symbol's category is then recovered by
>   clustering on the aggregate of its operators' vectors
>   (`bin/semantic_categories.py`). The payoff over structural participation:
>   symbols whose DIFFERENT operators share a semantic effect unify. Tests
>   `test_operator_codebook_shaping.py`, `test_semantic_categories.py`,
>   `test_semantic_categories_pipeline.py`.
> - **R5** (connective supervision): `bin/connective_supervision.py` -- a
>   truth/consequence signal makes the slot-0 operator superposition
>   load-bearing, discriminating `A AND B` from `A OR B` from consequence
>   alone (surface-identical operands) -- test
>   `test_connective_supervision.py`.
> - **D1 gate (2026-06-03, criterion corrected): MET.** The first evaluation
>   used the wrong criterion -- single-label nearest-neighbor POS agreement
>   (~0.19) -- which penalizes exactly the multi-category participation
>   overlaps role-collapse is built on. Role-collapse does not recover a
>   single-label POS system; it replaces declared shared categories with
>   operator-local participation categories a word may fill several of. The
>   correct D1 question is whether those patterns are structured enough to
>   drive a LEARNED COLLAPSE into the smaller mutually-exclusive category set
>   the live parser needs. They are: `participation.learned_collapse` merges
>   symbols by participation similarity, accepting only merges that keep every
>   rule distinguishable (`collapse_conflicts == 0`). On `complete.grammar`
>   the exact substitutability congruence is trivial (all 43 symbols are
>   context-unique), yet the participation-guided conflict-free collapse still
>   compacts them into **14 mutually-exclusive categories with zero parser
>   conflicts** -- the parser recovers, so the gate is met. "Recovers"
>   therefore means the parser's rule decisions survive the collapse, not
>   exact rule regeneration. With the gate met, `role_collapsed.grammar` is
>   now the **default** mental-model grammar (`MentalModel.xml`, 2026-06-03);
>   `complete.grammar` is retained as the compatibility baseline. The part
>   operator is unified there (single `isPart`; `queryPart` via dispatch), so
>   the mental-model op contract (`test_mental_model.py` `REQUIRED_OPS`) and
>   the relative-rule test helpers were made grammar-agnostic (find relative
>   rules via `_relative_rule_id_set` / `_RELATIVE_OP_NAMES`, not a hardcoded
>   `REL_T` lhs). Tests: `test_d1_pos_recovery_gate.py`
>   (`test_participation_drives_recovering_collapse`).

---

## 1. Vision

Replace fixed word lexing with a PerceptualSpace (PS) *meronymic analyzer*
that reuses the existing signal-router machinery in the analysis direction,
and shrink the symbolic grammar to declare **only operator roles** -- not
parts of speech and not surface markers.

Two ideas, one system:

1. **Subsymbolic analyzer.** PS analyzes a surface whole into perceptual
   parts (terminals) by running router operations in reverse; SS executes
   the recognized structure; reverse runs the same operations forward to
   re-realize surface. Surface markers (spaces, affixes, "and", "of", "'s")
   are **learned and owned by the operator**, bound on analysis and
   replayed on synthesis -- never hand-authored tokenizer artifacts.

2. **Role-collapsed grammar.** The grammar file answers exactly one
   question: *which operator is applied, and which role does each operand
   occupy?* It does **not** answer "what part of speech is this operand?"
   POS / category membership (noun, verb, determiner, ...) is **learned**
   from stable participation in operator roles, not declared.

The grammar declares roles; the model learns categories; the analyzer
learns markers.

---

## 2. Architecture synthesis

```text
forward:  surface --PS meronymic analysis--> terminals (.what / .where / .activation)
          --PS-to-SS binding--> operator-prefixed structure  --SS executes--> CS idea
reverse:  CS idea --SS taxonomic analysis--> operator-prefixed structure
          --PS meronymic synthesis (emit)--> surface
```

- **PS forward** = meronymic analysis (parent surface $\to$ child surfaces),
  the mirror of SS taxonomic synthesis (child symbols $\to$ parent idea).
- **PS reverse** = meronymic synthesis (child surfaces $\to$ parent surface).
- The same operation exposes `compose` (synthesis/reduce) and `generate`
  (analysis/unreduce); SS-forward uses `compose`, SS-reverse uses
  `generate`, PS-forward uses `generate`, PS-reverse uses `compose`.

### Codebook vs STM (the load-bearing split)

- **Codebook** holds *identities*: the terminal symbols **and the
  operators**. Every node of the deterministic operator-prefixed parse
  tree -- operations plus terminal symbols -- is a codebook entry, so the
  soft superposition over the parse can resolve any node by codebook
  lookup.
- **STM** holds *computed ideas*: the combined meanings. STM is the tensor
  passed CS $\leftrightarrow$ SS in the `forward()` conceptual-order
  recursion. An operator defines *how* meanings combine and contributes no
  meaning of its own, so **operators are not written into the STM idea
  space** -- only combined meanings are.
- A grammar **rule-id** stamped in `.where` therefore marks a slot as a
  *computed* idea (a reduce/transform happened), which by definition is not
  a codebook vector. So the rule-id legitimately stays in `.where`; there
  is no need to move operator identity out of `.where`.

---

## 3. Locked decisions

1. **`.where` is a pure endpoint-sum span key for analyzer terminals**
   ($where = phase(start) + phase(end)$); the magnitude carries length, the
   angle carries center. The legacy scalar rule-id namespace in `.where`
   remains for the symbolic stack (see decision 2).
2. **Rule-id stays in `.where`.** Its presence marks a *computed* idea, not
   a codebook vector, so it never collides with a codebook resolution. The
   risky `.where` rewrite is dropped.
3. **Operators live in the codebook, not the STM idea space.** STM idea
   slots hold combined meanings only. `SymbolicSpace.insert_operations`
   registers each grammar operation in a **dedicated operator codebook on
   SymbolicSpace** (`_operation_vectors` / `_operation_positions`),
   separate from the `subspace.what` symbol codebook -- so it never
   perturbs the symbol / idea / `.where` position namespace
   (`allocate_position`, `symbol_vocab_size`, the relation taxonomy).
4. **Markers are learned, owned by the operator** (absorb on analysis, emit
   on synthesis), bound from co-occurrence. Binding is many-to-one
   (markers $\to$ operator) with a canonical operator $\to$ default-marker
   for emit. The grammar carries no `*_MARK` categories and no copy/swap
   MARKER helper rules.
5. **The grammar declares operator roles only** (Section 4). POS / category
   membership is learned from operator-role participation.
6. **One relation per family, dispatched by `query`.** `isEqual` and
   `isPart` are each a single grammar relation; `query="true"` selects
   answer-producing semantics, `query="false"` selects assertive semantics.
   This replaces `queryPart` / `assertPart`.
7. **Space-scoped starts.** `PerceptualSpace` carries the singular surface
   start `U` ("everything"); `SymbolicSpace` carries its own completion
   starts (the operator outputs). No grammar-wide top-level `<start>`.
8. **A AND B vs A OR B are surface-indiscriminable**; they are
   discriminated by the slot-0 operator superposition shaped by deep
   structure over a large corpus. Dependency: corpus-scale
   truth/consequence supervision (operator-superposition becomes
   load-bearing for connectives).

---

## 4. Role-collapsed grammar contract

### 4.1 Role naming convention

```text
op_I<n>   input role  -- an operand/child fed INTO the operator (position n)
op_O1     output role -- the result/parent produced BY the operator
```

Canonical binary contract:

```xml
<rule>op_O1 = op.forward(op_I1, op_I2)</rule>
<rule>op_I1, op_I2 = op.reverse(op_O1)</rule>
```

Canonical unary contract:

```xml
<rule>op_O1 = op.forward(op_I1)</rule>
<rule>op_I1 = op.reverse(op_O1)</rule>
```

This matches the existing forward/reverse layer contract exactly: the
forward rule's LHS is the result (the output `O1`) and the RHS are the
operands (the inputs `I<n>`); `compose(operands) -> parent`,
`generate(parent) -> operands`.

> **Correction vs the source role-collapse spec.** The source spec used
> `op_I1 = op.forward(op_O1, op_O2)` with `I = input/result/parent` and
> `O = output/operand/child`. That conflates "input" with "result". This
> document inverts it to the intuitive reading: **I = the inputs you feed
> in, O = the output you get out.** Forward maps inputs to output
> (`O1 = op.forward(I1, I2)`); reverse maps output to inputs
> (`I1, I2 = op.reverse(O1)`).

Examples:

```xml
<rule query="true">isEqual_O1 = isEqual.forward(isEqual_I1, isEqual_I2)</rule>
<rule query="true">isEqual_I1, isEqual_I2 = isEqual.reverse(isEqual_O1)</rule>

<rule query="false">isPart_O1 = isPart.forward(isPart_I1, isPart_I2)</rule>
<rule query="false">isPart_I1, isPart_I2 = isPart.reverse(isPart_O1)</rule>

<rule query="false">not_O1 = not.forward(not_I1)</rule>
<rule query="false">not_I1 = not.reverse(not_O1)</rule>

<rule query="false">lift_O1 = lift.forward(lift_I1, lift_I2)</rule>
<rule query="false">lift_I1, lift_I2 = lift.reverse(lift_O1)</rule>
```

### 4.2 What disappears

- **All POS / category names** as grammar state names: `NP3`, `AP`, `VP1`,
  `N3`, `DET`, `S345`, `MP1`, `PP`, ... gone.
- **All transitional per-operator-position role categories** from the
  interim grammar: `CONJ_L45`, `CONJ_R45`, `DISJ_L45`, `QLEFT_NP3`,
  `QRIGHT_AP`, `NP_EQ3`, `S_PART34`, `QLEFT_PART34`, ... gone.
- **All category-rename rules** (`QLEFT_NP3 = NP3`, `CONJ_L45 = S45`, ...)
  and their generate mirrors. POS membership is recovered from role
  participation, not declared by projection rules.
- **`queryPart` / `assertPart`** as separate operators (folded into
  `isPart` + `query`).

### 4.3 Operator coverage

At least: `exist`, `isEqual`, `isPart`, `not`, `non`, `conjunction`,
`disjunction`, `intersection`, `union`, `lift`, `lower`. Each operator's
role arity comes from its `GrammarLayer` subclass and the existing
forward/reverse contract. Lossy / non-invertible operators either omit the
generate rule with an explicit comment or use the operator's documented
pseudo-inverse -- never a stale category mirror.

### 4.4 Starts

```xml
<PerceptualSpace>
  <start name="everything">U</start>
  ...
</PerceptualSpace>

<SymbolicSpace>
  <start name="relative_truth">isEqual_O1</start>
  <start name="relative_truth">isPart_O1</start>
  <start name="absolute_truth">exist_O1</start>
  ...
</SymbolicSpace>
```

`U` is the whole perceptual surface ("everything currently visible to the
analyzer") -- it is the analyzer's root span. The SS start set is the
operator **outputs** that count as completed symbolic expressions (final
SS start membership is finalized during implementation).

---

## 5. PerceptualSpace analyzer

### 5.1 Endpoint-sum `.where`

For a span `[start, end)`: $where = phase(start) + phase(end)$ where
$phase(p) = [\sin(p \cdot div\_term), \cos(p \cdot div\_term)]$. By the
sum-to-product identity the angle decodes the center and the magnitude
decodes the length. Constraints: do not normalize the key; keep max span
length below half the period; keep centers inside one recoverable period;
snap decoded endpoints to the integer grid; share `div_term` across spaces.

### 5.2 Meronymic operations + ObjectSubSpace

A small, direction-neutral operation inventory declared in the grammar's
`<PerceptualSpace>` section: `stop` (accept a span as a terminal / realize
a terminal surface), `boundary` (split/combine at boundary evidence),
`uniform` (split/combine near the midpoint), plus the learned-chunk ops
`prefix` / `suffix` / `compound` / `coordination` / `quote_or_bracket`.
Byte/char fallback is total (every byte has a codebook row), so the
analyzer always has a valid cover.

Durable analysis state lives on `ObjectSubSpace` (the PS analogue of
`WordSubSpace`): per-span `_buffer` / `_part_id` / `_span_start` /
`_span_end` / `_span_where` / `_parent_id` / `_left_id` / `_right_id` /
`_route_id` (PS-only meronymic route) / `_route_score` / `_depth`, plus the
marker-route replay fields `_marker_ps_id` / `_marker_span` / `_order_bit`
/ `_marker_position`. Fixed physical capacity; live count is `_depth`.

### 5.3 SurfaceSchema + absorb / emit (UG = shared templates)

Five universal templates declare each operator's marker slot + operand
order:

| id | name | arity | marker | order |
|---|---|---|---|---|
| T1 | UNARY_AFFIX | 1 | 1 slot, learned position {PRE,SUF,CIRCUM} | trivial / marked inversion |
| T2 | BINARY_INFIX | 2 | 1 INFIX/CIRCUM slot; may select which op fires | free {id,swap} |
| T3 | BINARY_DIRECTIONAL | 2 | (position,marker) co-varies with order | marked {id,swap} recorded |
| T4 | BINARY_JUXTAPOSE | 2 | none | marked {id,swap} |
| T5 | BINARY_ELISION | 2 | none | survivor only; other absorbed |

T4 is the default base schema so any operator round-trips. `absorb(left,
right, marker_id=...)` binds a co-occurring marker to the operator
(many-to-one) and returns the content operand; `emit(marker_id=...)`
replays the recorded marker (route metadata) or the operator's canonical
default. **Emit MUST use recorded route metadata, never the lossy
`generate()`=`(parent,parent)` inverse.**

### 5.4 Terminal stream + PS-to-SS binding

The analyzer exposes a fixed-capacity terminal-stream view over
`ObjectSubSpace` leaves: `what [B,Kmax,D]`, `where [B,Kmax,2]`,
`ids [B,Kmax]`, `mask [B,Kmax]`, `len [B]`. PS-to-SS binding resolves a
terminal's `ps_id` to an SS row: bound $\to$ the SS vector; unbound but
identified $\to$ `NULL_SEM` + an exposure count, promoting to a fresh SS
row after repeated stable exposure; unidentified (byte fallback) $\to$
always `NULL_SEM`.

### 5.5 Operator-superposition

`operator_superposition(query_vec)` is a softmax over the cosine similarity
between a query and each operation's vector in the operator codebook;
`soft_operator_compose(dist, left, right)` is the weighted sum of each
operation's compose. A one-hot distribution reduces to the typed grammar; a
spread distribution superposes operators -- the mechanism that discriminates
`A AND B` from `A OR B`.

### 5.6 Shared weighted-deduction framework

The shared formal frame for PS analysis and SS parsing is semiring-weighted
dynamic programming / weighted deduction. The same item graph can be read in
different semirings: sum-product for soft inside/forward-backward marginals
and max-plus for the single Viterbi route. This is the standard parsing
pattern described by [Goodman 1999, "Semiring
Parsing"](https://aclanthology.org/J99-4004/), the SCFG
inside-outside line of [Lari and Young
1990](https://doi.org/10.1016/0885-2308(90)90022-X), and weighted logic
programming / parsing transformations such as [Eisner and Blatz
2006](https://www.cs.jhu.edu/~jason/papers/eisner%2Bblatz.fg06.pdf). The
neural grammar-induction analogue is to parameterize rule scores but still
marginalize latent trees with dynamic programming, as in [Kim, Dyer, and
Rush 2019](https://aclanthology.org/P19-1228/).

Implementation mapping:

- SS parser/reducer: `BinaryStructuredReductionLayer` scores copy/reduce
  items over rule columns, returns `reduce_marginal_op` from
  `binary_tiling_soft_dp`, and uses `binary_tiling_viterbi` for the hard
  route.
- PS analyzer: `MeronymicRouter` uses the same `binary_tiling_soft_dp` /
  `binary_tiling_viterbi` pair over perceptual atoms and signed-neighborhood
  merge evidence.
- Soft-over-rules guarantee: `test_signal_router_layer
  ::test_layer_keeps_soft_superposition_over_reduce_rules` asserts that two
  tied viable reduce rules both retain positive marginal mass and both
  receive gradient through `logZ`, even though the forward route is hardened.

---

## 6. Loader / runtime requirements

| requirement | status |
|---|---|
| Parse `<PerceptualSpace>` / `<SymbolicSpace>` sections into separate PS/SS rule tables; bare `<compose>`/`<generate>` loads as SymbolicSpace | DONE |
| Remove the legacy `.cfg` loader path | DONE |
| `SymbolicSpace.insert_operations`, wired into the model build | DONE |
| **Space-scoped starts** -- `PerceptualSpace.start` configures PS starts, `SymbolicSpace.start` configures SS starts (today `<start>` is global metadata) | DONE (R1.1) |
| **`isPart` grammar layer + `query` dispatch** -- register `GRAMMAR_LAYER_CLASSES["isPart"]`; `isPart + query=true` $\to$ query-answer semantics, `query=false` $\to$ assertive (mirrors `isEqual`/`queryEqual`) | DONE (R1.2) |
| **Relative-rule detection** -- replace the `{isEqual, queryPart, assertPart, part, REL_T}` set with `{isEqual, isPart, SS relative-start role states}` | DONE (R1.3) |
| **Identity-rule injection** -- scope it to SymbolicSpace once starts are space-scoped (currently injects for the global primary start) | DONE (R1.4) |

---

## 7. Implementation status

### 7.1 DONE (from the subsymbolic-analyzer spec; this codebase)

- **Loader** -- PS/SS sections + back-compat; `.cfg` removed
  (`bin/Language.py`; `test/test_grammar_ps_ss_sections.py`).
- **Transitional grammar rewrite** -- four `.grammar` files gained PS/SS
  sections; `*_MARK` + copy/swap MARKER rules deleted; per-operator-position
  categories (`CONJ_L45`/`CONJ_R45`, ...) with marker-free role transitions.
  **This is the interim grammar the role-collapse (Section 4) replaces.**
  (`test/test_grammar_rewrite.py`.)
- **SurfaceSchema + absorb/emit** -- T1-T5 + per-operator schema
  (`bin/Layers.py`; `test/test_surface_schema.py`).
- **ObjectSubSpace** -- PS carrier + marker-route fields
  (`bin/Language.py`; `test/test_object_subspace.py`).
- **Operations in a dedicated operator codebook** -- `insert_operations`,
  build-wired in `WordSubSpace.__init__`; operators registered in
  `SymbolicSpace._operation_vectors` / `_operation_positions`, separate
  from the symbol codebook so the symbol/idea/`.where` namespace is
  untouched (`bin/Spaces.py`; `test/test_ss_codebook_operations.py`).
- **PS analyzer (compatibility mode)** -- `EndpointSumWhere`,
  `MeronymicAnalyzer` (reuses the word/byte tokenizer as `boundary`),
  terminal-stream view, reverse synthesis, `soft_operator_compose`
  (`bin/perceptual_analyzer.py`; `test/test_ps_where.py`,
  `test/test_ps_analyzer.py`, `test/test_ps_reverse_e2e.py`).
- **PS-to-SS binding** -- `resolve_ps_terminal` / `null_sem`
  (`test/test_ps_ss_binding.py`).
- **Operator-superposition** -- `SymbolicSpace.operator_superposition`
  (`test/test_operator_superposition.py`).
- **Docs** -- `doc/Language.md`, `doc/STM.md`, `doc/Spaces.md`.

### 7.2 LANDED SINCE 7.1 / REMAINING GATES

- **Role-collapse grammar rewrite** -- landed as
  `data/role_collapsed.grammar` and **promoted to the default** mental-model
  grammar (`MentalModel.xml`, 2026-06-03, D1 gate met); `data/complete.grammar`
  is retained as the compatibility baseline.
- **Loader / runtime** -- space-scoped starts, `isPart`, query dispatch,
  relative-rule detection, and SymbolicSpace-scoped identity injection are
  implemented (Section 6).
- **Learned PS router** -- `MeronymicRouter` and `analyze_routed` use the
  shared Viterbi / soft-DP primitive; live `PerceptualSpace.forward` supports
  `<chunking>analyse</chunking>`.
- **Dimensionality-from-participation** -- implemented as
  `bin/participation.py`; `learned_collapse` drives the parser-recovering
  collapse that satisfied the (corrected) D1 gate, so role-collapse is now the
  default.
- **Corpus-scale connective supervision** -- implemented as
  `bin/connective_supervision.py`; truth/consequence supervision makes the
  slot-0 operator superposition load-bearing for connectives.
- **Gates -- both met (2026-06-03).** `role_collapsed.grammar` is promoted to
  default (D1 met), and the host-token reconstructed analyze surface is
  replaced by a raw unanalyzed `[B,1,N]` byte buffer (`<lexer>raw</lexer>`,
  which keeps the word-level codebook and only changes the analyze surface).
  The one remaining item is D2's *cross-language* empirical validation (the
  lift/lower collapse is implemented and the POS-recovery half is validated by
  D1; the cross-language generalization claim is not yet benchmarked).

---

## 8. Phased plan (implemented checkpoints)

Ordered so the grammar role-collapse lands *with/after* the learner that
makes it safe (decision D1), not before.

- **Phase R1 -- Loader/runtime prerequisites (DONE).** Space-scoped starts;
  `isPart` layer + `query` dispatch; relative-rule detection update;
  identity-rule scoping. Landed first so the rewrite can load. Tests:
  space-scoped-starts, `isPart` dispatch by `query`.
- **Phase R2 -- Role-collapsed grammar (DONE; now the DEFAULT).** Produced the
  operator-roles-only `data/role_collapsed.grammar` (Section 4). Landed first
  as a variant validated against the transitional grammar, then promoted to
  the default (`MentalModel.xml`, 2026-06-03) once R4's `learned_collapse`
  satisfied the D1 gate. Grammar tests inverted/extended
  (`test_role_collapsed_grammar.py`).
- **Phase R3 -- Learned PS router (DONE).** Factor the shared inverse routing
  primitive out of `unreduce`/`reverse_stack`; meronymic Viterbi/soft-DP
  with signed-neighborhood evidence; auto-wire into `PerceptualSpace
  .forward`. Tests: byte-fallback vs known-word, Viterbi-not-beam,
  depth penalty.
- **Phase R4 -- Dimensionality-from-participation (DONE).** Recover POS/order
  from operator-role participation -- the learner that justifies dropping POS
  categories. **D1 gate (met):** `learned_collapse` drives a parser-recovering
  collapse of the participation patterns, so role-collapse was promoted to the
  default (see §9 D1).
- **Phase R5 -- Corpus-scale connective supervision (DONE).** Truth/consequence
  signal so the operator superposition discriminates connectives; this is
  where operator-superposition becomes load-bearing.

---

## 9. Design risks / open questions

- **D1 (sequencing, decided -- and now satisfied).** Role-collapse moves POS
  from *declared* to *learned from role participation*. The learner that
  recovers POS is Phases R3/R4, so role-collapse was not made the default
  before R4. As of 2026-06-03 R4's `learned_collapse` demonstrates a
  parser-recovering collapse (the corrected D1 criterion -- not single-label
  POS recovery; see the D1 status block at the top), so role-collapse is now
  the default and the transitional grammar (Section 7.1) is the retained
  compatibility baseline.
- **D2 (lift/lower collapse, empirical -- collapse landed; cross-language
  validation open).** Collapsing every `lift`/`lower` construction
  (`lift(NP,VP)`, `lift(P,NP)`, `lower(VP,PP)`, `lower(ADJ,NP)`, ...) into a
  single `lift_O1 = lift.forward(lift_I1, lift_I2)` / `lower` pair removes the
  distinct-construction and operand-order constraints from the grammar, moving
  them entirely into learned role-participation. This is **more flexible** --
  if role-participation recovers the POS/construction structure, it is expected
  to generalize better across languages. **Status:** the full collapse is
  implemented in `data/role_collapsed.grammar` (the single `lift`/`lower`
  pair), and the *"does it recover POS?"* half is validated -- that is exactly
  the D1 gate (`learned_collapse` / `test_d1_pos_recovery_gate.py`). The
  *"is it better cross-language?"* half is the **one open empirical item**: no
  multi-language corpus / benchmark exists for it yet. It remains an empirical
  validation rather than an a-priori assertion.
- **Carry-forward concerns (addressed 2026-06-03; resolutions inline).**
  - The current analyzer is still compatibility-mode tokenization plus
    fallback. It proves the carrier, terminal stream, and surface replay
    path, but it is not yet learning segmentation or meronymic routes.
    *Resolved (R3): the learned `MeronymicRouter` + `analyze_routed` now do
    learn segmentation / meronymic routes via the shared Viterbi/soft-DP;
    compatibility mode remains the documented fallback.*
  - `merge` should probably remain a structural action completed directly
    by the modified trie / role matcher. It should not be inserted as a
    learned operator in the operator trie or operator-superposition table
    unless a concrete semantic layer is added for it. *Resolved:
    `SymbolicSpace.insert_operations` now registers only method names with a
    concrete `GrammarLayer`, so structural `merge` is excluded from the
    operator codebook / superposition (`test_merge_structural_not_operator`).*
  - Markers are surface text, but by the time SS learns them they should
    have PS identities. Replay should therefore resolve marker PS ids back
    to canonical surface bytes/text, not interpolate opaque marker ids as
    literal output. *Resolved: `synthesize_tree` takes a `marker_resolver`
    and never interpolates an opaque PS id; a surface-string marker is placed
    directly (`test_marker_replay_canonical`).*
  - Byte fallback must be UTF-8 exact: byte terminals should preserve the
    original byte offsets and reconstruct the original surface, including
    non-ASCII text, without mojibake or overlapping spans. *Resolved:
    `analyze_routed` carries each terminal's exact bytes and reconstructs
    from bytes, so a multi-byte glyph split across byte terminals round-trips
    without mojibake (`test_ps_byte_fallback_utf8`). (The word lexer still
    splits non-ASCII words for percept lookup -- a separate tokenizer item.)*
- **Open:** the exact SS start set (which operator outputs count as
  completed expressions); whether `U` (the PS root) needs sub-starts for
  partial surfaces; the canonical operator $\to$ default-marker policy per
  language.

---

## 10. Verification (unified test list)

Already green (Section 7.1): `test_grammar_loads_ps_and_ss_sections`,
`test_cfg_loader_removed`, `test_conj_disj_isequal_share_one_template`,
`test_default_schema_is_bare_juxtapose`, `test_marker_binds_from_cooccurrence`,
`test_emit_replays_marker`, `test_emit_uses_route_meta_not_lossy_generate`,
`test_endpoint_sum_where_decodes_span`, `test_boundary_matches_word_lexer`,
`test_ps_analyzer_byte_fallback`, `test_terminal_stream_*`,
`test_meronymic_reverse_replays_surface`, the analyze$\to$execute$\to$
reverse$\to$emit$\to$surface E2E (marker learned, not a grammar token),
`test_ps_to_ss_null_before_binding`, `test_ps_to_ss_binding_after_repetition`,
`test_operator_superposition_*`, `test_operations_inserted_into_ss_codebook`.

Added in R1-R5: `role_collapsed.grammar` has no `queryPart`/`assertPart`; no
bare POS/category rename rules; every relation rule carries explicit
`query`; `isEqual` uses `isEqual_O1`/`isEqual_I1`/`isEqual_I2`; `isPart`
uses `isPart_O1`/`isPart_I1`/`isPart_I2`; no top-level `<start>`;
`PerceptualSpace` has `<start name="everything">U</start>`; `SymbolicSpace`
owns its starts; `isPart + query` dispatch; relative-rule detection over
`{isEqual, isPart}`; the learned router selects one Viterbi route + soft
marginals; the participation-learner recovers the transitional grammar's
role assignments on a fixture, and `learned_collapse` drives a
parser-recovering collapse on the real grammar (D1 met), so
`role_collapsed.grammar` is the default.

---

## 11. Non-goals

Full English syntax; full generation quality; beam / K-best parsing in the
first patch; a large learned meronymic inventory in the first patch;
multimodal perceptual analysis. The role-collapse does not *delete*
linguistic structure -- it moves POS/construction structure from declared
grammar categories to learned role participation, and that move is gated on
the learner (D1) and validated empirically (D2).
