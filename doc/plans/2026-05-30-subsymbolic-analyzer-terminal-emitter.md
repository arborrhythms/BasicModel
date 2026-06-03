# Subsymbolic Analyzer and Terminal Emitter

> **2026-06-02 integration update.** The sections from "Integrated decisions"
> through "Execution manifest" below consolidate the absorb/emit/swap codification
> and the architectural decisions taken after the original draft. They **amend** the
> phased plan that follows; where they conflict, this update wins. The original phase
> detail is retained below as the execution reference.

## Integrated decisions (2026-06-02)

1. **`.where` is a pure endpoint-sum span key.** No operation/rule ids in `.where`.
2. **`.what` carries the operator.** A relative idea is `[op, arg1, arg2]` as
   `[B, 3, D]`; the operator occupies predicate slot 0. SS executes by resolving slot
   0 against the operator codebook. Absolute ideas stay `[B, 1, D]`. Operators are
   fixed-arity and sit in prefix position, so relaxing the STM length lets `.what`
   hold a full syntax tree as an unambiguous prefix (Polish) serialization
   `[B, Kmax, D]` -- no bracketing; `[B, 3, D]` is the minimal single-operator case.
3. **Operator dispatch = codebook resolution of `.what[:,0]`.** SS operator identity
   lives in `.what`; `ObjectSubSpace._route_id` is only the PS meronymic route used
   for surface replay.
4. **Markers are learned, owned by the operator** (absorb/emit), bound from the
   surface; the grammar `*_MARK` categories and `copy/swap` MARKER helper rules are
   deleted. The slot-0 operator vector (e.g. `AND`) is a distinct representation from
   the co-occurring surface marker percept (e.g. `"and"`): the operator is
   symbolic/executable, the marker is PS bytes. Binding is many-to-one
   (markers -> operator) with a canonical operator -> default-marker for emit.
5. **A AND B vs A OR B** are surface-indiscriminable; discriminated by the slot-0
   operator vector shaped by deep structure over a large corpus. Dependency:
   corpus-scale truth/consequence supervision (operator-superposition becomes
   load-bearing for connectives).
6. Grammar files gain `<PerceptualSpace>` and `<SymbolicSpace>` sections.
7. Grammar files are rewritten with unique per-operator-position categories; MARKER
   derivation rules removed.
8. Legacy `.cfg` grammar path is removed (files + loader code).
9. Full-English grammar operations are enumerated in `todo.md` as the penultimate
   task before training.
10. Symbol dimensionality is no longer carried by category names (drop `NP345`-style
    suffixes); recover a symbol's dim from its distribution of participation across
    operator positions (e.g. LHS/RHS of `lift`/`lower`). See `todo.md`.

## Architecture synthesis

```
forward:  surface --PS meronymic analysis--> terminals (.what/.where/.activation)
          --PS-to-SS binding--> [op, arg1, arg2] in .what  --SS executes--> CS idea
reverse:  CS idea --SS taxonomic analysis--> [op,...] --PS meronymic synthesis (emit)--> surface
```

- `.where` = `phase(start) + phase(end)` (magnitude = length, angle = center).
- `.what` = symbolic content; relative-idea slot 0 = operator vector, slots 1..2 =
  operands; SS reads slot 0 to dispatch. Fixed-arity + prefix => relaxed STM length
  yields a full-tree prefix serialization `[B, Kmax, D]`.
- `ObjectSubSpace._route_id` = PS meronymic route id (reconstruction only).
- Operator vs marker: the slot-0 operator vector (`AND`, executable) is distinct from
  the surface marker percept (`"and"`, PS bytes); marker-role binding maps
  markers -> operator (many-to-one) and operator -> default-marker (emit).

## Absorb / Emit / Swap codification (UG = shared templates)

Surface primitives (the existing `copy/swap` MARKER idiom in `complete.grammar`,
generalized and learned):

| primitive | analysis (PS forward) | synthesis (PS reverse) |
|---|---|---|
| mark | absorb a sub-span as a marker (not a content child) | emit the marker bytes at the schema position |
| swap | infer operand order from the matched arrangement | reorder operands per the recorded order bit |
| copy | pass an operand sub-span to the child worklist | realize an operand surface in place |

Universal templates (the only declared part):

| id | name | arity | marker | order |
|---|---|---|---|---|
| T1 | UNARY_AFFIX | 1 | 1 slot, position learned {PRE,SUF,CIRCUM} | trivial / marked inversion |
| T2 | BINARY_INFIX | 2 | 1 INFIX/CIRCUM slot; may select which op fires | free {id,swap} |
| T3 | BINARY_DIRECTIONAL | 2 | (position,marker) co-varies with order | marked {id,swap} recorded |
| T4 | BINARY_JUXTAPOSE | 2 | none | marked {id,swap} |
| T5 | BINARY_ELISION | 2 | none | survivor only; other absorbed |

T4 is the default base schema (bare concatenation) so any op round-trips.

Per-operator schema:

| operator | arity | template | learned marker | order | selects op? |
|---|---|---|---|---|---|
| not / non | 1 | T1 | negator | n/a | no |
| query | 1 | T1 | interrogative | marked swap (aux-inversion) | no |
| exist | 1 | T1 | existential | n/a | no |
| conjunction | 2 | T2 | and-class | free | yes -> min |
| disjunction | 2 | T2 | or | free | yes -> max |
| isEqual | 2 | T2 | copula | free | yes -> join |
| union | 2 | T2 | optional | free | yes |
| intersection | 2 | T2 | optional (verb-object, adj-stack) | free | yes |
| part / queryPart / assertPart | 2 | T3 | 's (id) / of (swap) | marked | partial |
| lift | 2 | T3/T4 | DET/MP prefix or bare | marked | no |
| lower | 2 | T3/T4 | DET / ADV / PP modifier or bare | marked | no |
| copy | 2 | T5 | none | id | no |
| swap | 2 | T5 | none | swap | no |

Learning story: (1) templates declared, marker slots NULL; (2) marker-free structural
bootstrap (`boundary`/`uniform`/`stop`) segments operands; (3) SS routes by
semantic/truth fit, the co-occurring marker binds to the operator's slot (a
Phase-7-sibling marker-role binding) and to its slot-0 operator vector; (4) once
bound, the marker is cheap `signed_neighborhood_evidence`; (5) EMIT replays the bound
marker (vector `generate()` is lossy `(parent,parent)`, so emit MUST use route
metadata); (6) order (`id`/`swap`) co-learned for T3.

New code: `SurfaceSchema` + `absorb`/`emit` on `GrammarLayer` (`bin/Layers.py`);
route-metadata on `ObjectSubSpace` (`_marker_ps_id, _marker_span, _order_bit,
_marker_position`); marker-role binding hook in PS-to-SS binding.

## Grammar file rewrite

Each operator-argument position becomes its own category in the operator's namespace
(role explicit, no MARKER helper needed); the operator is the slot-0 `.what` vector.
Worked example (`complete.grammar` conjunction):

```
BEFORE:
  S_CONJ45 = copy.forward(S45, CONJ_MARK)        # MARKER helper  -- DELETE
  S45      = conjunction.forward(S_CONJ45, S45)
  + CONJ_MARK category, + "and|..." word list

AFTER:
  <SymbolicSpace>
    <compose>  S45 = conjunction.forward(CONJ_L45, CONJ_R45) </compose>
    <generate> CONJ_L45, CONJ_R45 = conjunction.reverse(S45) </generate>
  </SymbolicSpace>
  # surface "and" is a learned marker on conjunction's T2 schema.
  # STM holds [conjunction_op, X, Y] in .what [B,3,D]; SS dispatches on slot 0.
```

All MARKER helper rules and `*_MARK` categories in `complete.grammar` (lines 21-42:
EOS/EQUALS/PART/DO/NOT/NON/CONJ/DISJ/PLURAL/QUERY_MARK) are deleted. Existing
modification rules (`lower(ADV,VP)`, `lower(VP1,PP)`, `intersection(V1,NP3)`, ...) are
kept, restated under `<SymbolicSpace>` with per-position categories.

Dimensionality recovery: dropping `NP345`/`S45`-style names removes the dim suffix;
recover a symbol's dim from its operator-participation distribution (the positions it
validly fills, e.g. `lift`/`lower` LHS/RHS, pin its dim). See `todo.md`.

Files to rewrite: `data/complete.grammar`, `data/default.grammar`,
`data/xor.grammar`, `data/shamatha.grammar`.

## PS/SS grammar sections + loader change

```xml
<grammar name="...">
  <start .../>
  <PerceptualSpace>
    <compose>  <!-- parts -> whole : PS reverse / surface synthesis --> ... </compose>
    <generate> <!-- whole -> parts : PS forward / surface analysis --> ... </generate>
  </PerceptualSpace>
  <SymbolicSpace>
    <compose> ... taxonomic synthesis (markers removed) ... </compose>
    <generate> ... taxonomic analysis ... </generate>
  </SymbolicSpace>
</grammar>
```

Extend `Grammar.load_from_grammar_file` (`bin/Language.py`) to parse the two sections
into separate PS/SS rule tables. Backward-compat: a file with bare
`<compose>`/`<generate>` loads as `<SymbolicSpace>`.

## Remove legacy `.cfg`

- Delete files (plain `rm`; user stages/commits): `data/grammar2.cfg`,
  `data/grammar_legacy.cfg`.
- Delete code in `bin/Language.py`: `load_from_cfg`, `_parse_cfg_lines`, the
  `grammarCfg` branch, the `cfg_path` scan, and `grammar.cfg` docstring references.
  No active `data/*.xml` references `grammarCfg`.

## Full-English operations

See `todo.md` (top): Group A (new operators: relativize/bind_gap, embed/complementize,
tense, aspect, compare, subordinate), Group B (compositions: ditransitive, passive,
VP/AP coordination, wh-questions, quantifier scope, appositives), and Infrastructure
(symbol dimensionality determination). Penultimate task before training.

## Integration map (amendments to the phased plan below)

- Phase 1 Terminal adapter: unchanged; introduces endpoint-sum `.where`.
- Phase 2 Router audit: SS operator identity moves to `.what` slot 0 (not
  `_route_id`); `.where` freed; `reverse_stack`/`unreduce` read the operator from
  `.what` slot 0; per-row (not row-0) loop control.
- Phase 3 Meronymic op registry: declared in grammar `<PerceptualSpace>`.
- Phase 4 ObjectSubSpace: `_route_id` is PS-only; add marker-route fields.
- Phase 5 Stack view: unchanged.
- Phase 6 Routing PS analyzer: unchanged.
- Phase 7 SS binding: extended with marker-role binding + slot-0 operator binding.
- Phase 8 Reverse synthesis: uses `emit`.
- NEW Phase 8b: grammar rewrite + PS/SS sections + loader change + `.cfg` removal.
- Phase 9 Operator-superposition: now load-bearing for connective discrimination.
- Phase 10 Docs/tests: + write `todo.md`; + the absorb/emit/swap codification.

## Execution manifest

1. (done in planning session) this integration written here; `todo.md` updated.
2. Rewrite the four `.grammar` files (scheme above).
3. Extend `load_from_grammar_file` for PS/SS sections.
4. Remove `.cfg` files + loader code.
5. Implement `SurfaceSchema`/`absorb`/`emit` and the rest per the phase order below.

Verification (targeted pytest, added with the phase that introduces each behavior):
`test_operator_lives_in_what_slot0`, `test_where_carries_no_operator`,
`test_conj_disj_isequal_share_one_template`, `test_marker_binds_from_cooccurrence`,
`test_emit_replays_marker`, `test_emit_uses_route_meta_not_lossy_generate`,
`test_grammar_loads_ps_and_ss_sections`, `test_grammar_has_no_marker_helper_rules`,
`test_cfg_loader_removed`, `test_pp_modifies_vp_via_lower`,
`test_default_schema_is_bare_juxtapose`. End-to-end: round-trip a sentence
(analyze -> [op,X,Y] -> execute -> reverse -> emit -> surface) with the marker word
lists deleted.

## Implementation status (2026-06-02)

Implemented and verified (targeted pytest, all green; the model-consumer
suite for `complete.grammar` still passes):

- **Loader** -- PS/SS sections + bare->SS back-compat
  (`bin/Language.py` `Grammar.configure` / `load_from_grammar_file`;
  `test_grammar_ps_ss_sections.py`).
- **Grammar rewrite** -- four `.grammar` files: PS/SS sections,
  per-operator-position categories, `*_MARK` + copy/swap deleted
  (`test_grammar_rewrite.py`).
- **SurfaceSchema + absorb/emit** -- T1-T5 templates, per-operator schema,
  marker bind/replay (`bin/Layers.py`; `test_surface_schema.py`).
- **ObjectSubSpace** -- PS carrier + marker-route fields
  (`bin/Language.py`; `test_object_subspace.py`).
- **Operations in a dedicated operator codebook** --
  `SymbolicSpace.insert_operations`, wired into `WordSubSpace.__init__`;
  operators registered in `_operation_vectors` / `_operation_positions`,
  separate from the symbol codebook so the symbol/idea/`.where` namespace
  is untouched (`bin/Spaces.py`; `test_ss_codebook_operations.py`).
- **PS analyzer** -- `EndpointSumWhere`, `MeronymicAnalyzer`
  (compatibility mode), reverse synthesis, `soft_operator_compose`
  (`bin/perceptual_analyzer.py`; `test_ps_where.py`, `test_ps_analyzer.py`,
  `test_ps_reverse_e2e.py`).
- **PS-to-SS binding** -- `resolve_ps_terminal` / `null_sem`
  (`test_ps_ss_binding.py`).
- **Operator-superposition** -- `SymbolicSpace.operator_superposition`
  (`test_operator_superposition.py`).

**Amendment to decision #2 (2026-06-02, user-locked).** The operator does
NOT live in the STM idea space. Operators are kept in the **codebook**
(`insert_operations` into the SS codebook) so the soft-superposition over
the deterministic operator-prefixed parse tree (held in
`WordSubSpace`/`ObjectSubSpace`) can resolve them; the STM idea space holds
only **combined meanings** -- an operator defines *how* meanings combine
and contributes no meaning of its own. Every node of the operator-prefixed
tree (operations + terminal symbols) exists in the codebook, while the
computed ideas (not in the codebook) live in STM -- the tensor passed CS
$\leftrightarrow$ SS in the `forward()` conceptual-order recursion.
Consequently the rule-id stays in `.where`: its presence marks the slot as
a *computed* idea, which is by definition not a codebook vector, so the
risky `.where`-rewrite (and `test_where_carries_no_operator`) is dropped.

**Compatibility vs learned.** The `MeronymicAnalyzer` ships in
*compatibility mode* (reuses the word/byte tokenizer as `boundary`, byte
fallback for unknown words), reproducing the current lexer's terminals.
The learned router path (Viterbi/soft-DP meronymic routing with signed-
neighborhood evidence over the shared `BinaryStructuredReductionLayer`),
corpus-scale connective supervision, and auto-wiring the analyzer into
`PerceptualSpace.forward` remain follow-ups.

---

## Context

This plan follows
[2026-05-29-stm-serial-parallel-modes.md](2026-05-29-stm-serial-parallel-modes.md).
The STM Serial/Parallel work should land first. It gives the model a
stable terminal-stream target: predict-then-perceive sequencing, router
context, STM end-state discipline, reverse-from-STM, and absolute vs
relative end-state handling.

The purpose of this follow-up is to replace fixed word lexing with a
subsymbolic perceptual analyzer without inventing a second parser. The
PerceptualSpace analyzer should reuse the same signal-router machinery
already used by the symbolic parser:

```text
binary_tiling_soft_dp
binary_tiling_viterbi
compact_hard
compact_soft
BinaryStructuredReductionLayer
LanguageLayer reduce / unreduce / reverse_stack
```

The difference is direction.

```text
SS forward path:
  symbolic parts -> taxonymic synthesis -> symbolic whole / idea

PS forward path:
  perceptual whole -> meronymic analysis -> perceptual parts
```

The PS side is therefore the mirror image of the SS side, not a new beam
parser.

## Core Claim

The model should not require a hand-authored inventory of syntactic
categories or parts of speech in the grammar file.

Instead:

1. PerceptualSpace analyzes a surface whole into known perceptual parts
   by running router operations in the analysis direction.
2. Recognized perceptual parts are emitted into STM as terminals.
3. SymbolicSpace maps repeated perceptual terminals to symbolic codebook
   entries.
4. SymbolicSpace synthesizes symbolic vectors with the existing
   rule-router in the composition direction.
5. ConceptualSpace integrates the perceptual and symbolic results and
   scores reconstruction, prediction, and truth.

ConceptualSpace is not purely meronymic or purely taxonymic. It is the
integration site for PS and SS. Destructive interference is expected when
surface structure and symbolic structure compete for the same
representation.

The directional symmetry is:

```text
forward input path:
  PS meronymic analysis:    parent surface -> left surface, right surface
  SS taxonymic synthesis:   left symbol, right symbol -> parent idea

reverse output path:
  SS taxonymic analysis:    parent idea -> left symbol, right symbol
  PS meronymic synthesis:   left surface, right surface -> parent surface
```

Parts of speech are learned from stable participation in SS operator
roles. They are not required tokenizer labels.

## Existing Parser Machinery To Reuse

The current parser is a one-route signal router with soft training
support:

```text
BinaryStructuredReductionLayer.forward
  builds candidate op outputs for adjacent pairs
  scores COPY and REDUCE actions
  runs binary_tiling_soft_dp for marginals
  runs binary_tiling_viterbi for the hard route
  compacts with compact_hard / compact_soft
```

Important constraints:

```text
no beams in the first implementation
one Viterbi route in the forward pass
soft DP marginals for gradient support
fixed physical tensor shape
logical lengths and masks for variable active structure
```

The PS analyzer should not introduce a separate weighted-forest parser.
If K-best routing is ever needed, it should be a later extension to the
shared router, not a PS-only design.

The current `LanguageLayer` is not completely stateless: it owns
trainable routing modules and transient caches such as the last route
trace. But it should not own durable sentence state. In the symbolic
path, durable parser state lives on `WordSubSpace`:

```text
WordSubSpace:
  current_rules / generate_rules
  parser cursors
  typed STM payload and metadata buffers
  category stack and rule predictor
  truth / discourse / reconstruction stores

LanguageLayer:
  routing layers
  op registry
  Viterbi / soft-DP execution
  short-lived route caches
```

The perceptual path should use the same ownership split. A
LanguageLayer-like meronymic router may own trainable routing modules and
transient caches, but durable perceptual analysis state should live on a
new `ObjectSubSpace`.

## Operator Interface

Both PS and SS operations should expose two directional methods.

For symbolic taxonymic operations:

```python
op.synthesize(left_symbol, right_symbol) -> parent_symbol
op.analyze(parent_symbol) -> left_symbol, right_symbol
```

For perceptual meronymic operations:

```python
op.analyze(parent_surface) -> left_surface, right_surface
op.synthesize(left_surface, right_surface) -> parent_surface
```

This can be adapted to the current grammar layer convention:

```python
compose(left, right)   # synthesis / reduce direction
generate(parent)       # analysis / unreduce direction
```

The same operation can therefore be used in both spaces:

```text
SS forward uses compose.
SS reverse uses generate.

PS forward uses generate.
PS reverse uses compose.
```

This is the main architectural point. The PS analyzer should be built by
running the existing parser machinery in the opposite direction, with PS
meronymic operations attached instead of SS taxonymic operations.

## Forward / Reverse Ownership

The `forward()` and `reverse()` methods on each space are the lifecycle
and tensor-contract boundaries. They should orchestrate the analyzer,
binding, router, and reconstruction components, but should not inline all
of their implementation details.

The intended ownership split is:

```text
PerceptualSpace.forward
  orchestrates surface input -> PS analyzer -> ObjectSubSpace leaves
  -> PS terminal stream

PerceptualSpace.reverse
  orchestrates SS/PS terminals -> exact or generative meronymic replay
  -> reconstructed surface bytes/chars

ObjectSubSpace
  owns durable meronymic analysis state: spans, parent/child links,
  route ids, route scores, depths, and replay metadata

RadixLayer / percept store
  owns percept identity: trie lookup, byte fallback, codebook rows,
  canonical bytes, and reversible lookup

Recursive trie lattice / object codebook
  owns guaranteed covers from EVERYTHING to bytes/chars, ordered
  part-signature paths, overlap policies, common-object caches, and
  exact recursive reconstruction

Signed-neighborhood encoder
  owns oriented contextual evidence for a part occurrence; this is PS
  .what evidence for route scoring, not the canonical surface identity

LanguageLayer-like router
  owns trainable route scoring, shared reduce/unreduce primitives,
  Viterbi/soft-DP execution, and transient route caches

ConceptualSpace / SymbolicSpace binding
  owns PS OBJECT -> SS WORD-symbol resolution, NULL_SEM handling,
  exposure counting, and promotion into stable symbolic rows

STM / stack-mode SubSpace
  owns the terminal stream mechanics consumed by shift/reduce/unreduce
```

This keeps `Spaces.py` from becoming a large control module. Space
methods should compose the components and preserve the public
forward/reverse contracts; specialist layers should own the actual
routing, storage, and binding policies.

## Carrier State

The mirror-image design is not a field-for-field copy. `WordSubSpace`
stores taxonomic state for symbolic parsing. `ObjectSubSpace` should
store meronymic state for perceptual analysis.

Current `WordSubSpace` durable state includes:

```python
_buffer:         [B, cap, concept_dim]  # typed STM payload
_category:       [B, cap]               # category id
_order:          [B, cap]               # order/dimension metadata
_ref_id:         [B, cap]               # symbolic/taxonomic ref id
_depth:          [B]                    # logical live depth
_category_names: list[B][cap]           # host-side category names
```

It also owns parser-side host state:

```text
current_rules
generate_rules
cursor
category_stack
rule_predictor
truth_layer
discourse
reconstruction_stack
```

The new `ObjectSubSpace` should be the PS analogue:

```python
_buffer:       [B, cap, percept_dim]  # PS span/part vectors
_part_id:      [B, cap]               # PS codebook row id, or -1
_span_start:   [B, cap]               # inclusive byte/atom start
_span_end:     [B, cap]               # exclusive byte/atom end
_span_where:   [B, cap, 2]            # endpoint-sum spatial key
_parent_id:    [B, cap]               # derivation parent slot, or -1
_left_id:      [B, cap]               # left child slot, or -1
_right_id:     [B, cap]               # right child slot, or -1
_route_id:     [B, cap]               # selected meronymic operation id
_route_score:  [B, cap]               # local route confidence / score
_depth:        [B]                    # logical live depth
```

It should also own PS-side route state:

```text
analysis_routes      # PS forward: whole -> parts
synthesis_routes     # PS reverse: parts -> whole
meronymic_cursor     # host cursor if needed for replay
surface_bytes_ref    # reference to the encoded source surface
```

All GPU tensors keep fixed physical capacity. Variable numbers of live
surface parts are represented by masks and logical lengths/depth, as in
`WordSubSpace`.

Hard inference may expose a boundary mask as a derived view:

```python
cut_after: [B, Lmax - 1]   # 1 means cut after this byte/atom
```

But the durable router state should remain operation-based, because
reverse() needs to know which operation produced the surface
realization.

The temporary parser stack remains separate. It is the existing
stack-mode `SubSpace` contract:

```python
subspace.what:        [B, K, D]
subspace.where:       [B, K, W]
subspace.activation:  [B, K]
```

`ObjectSubSpace` and `WordSubSpace` are durable carriers; stack-mode
`SubSpace` is the transient route executor view.

## Where Encoding And Spans

For the analyzer path, `subspace.where` should be a two-element
endpoint-sum span key. It is not a semantic codebook key and it is not a
raw `[start, end]` pair. It is an invertible spatial key over a bounded
input namespace.

The legacy single-position quadrature decoder is not a constraint for
this plan. Analyzer migration should replace or extend the current
`.where` encode/decode helpers so endpoint-sum spans are the target
contract.

Define the boundary phase for a scalar position `p`:

```text
phase(p) = [sin(p * div_term), cos(p * div_term)]
```

For a span `[start, end)`:

```text
where = phase(start) + phase(end)
```

Using the sum-to-product identities:

```text
where = 2 * cos((end - start) * div_term / 2)
        * [sin(center * div_term), cos(center * div_term)]

center = (start + end) / 2
length = end - start
```

Therefore:

```text
angle(where)     decodes the span center
magnitude(where) decodes the span length
```

The reverse decode is:

```text
center = atan2(where[..., 0], where[..., 1]) / div_term
radius = norm(where)
length = 2 * arccos(clamp(radius / 2, -1, 1)) / div_term
start  = center - length / 2
end    = center + length / 2
```

This gives `.where` an invertible spatial meaning as long as the
namespace obeys these constraints:

```text
do not normalize `.where`; magnitude carries length
keep maximum span length below half the sinusoidal period
keep centers inside a single recoverable namespace period
snap decoded start/end to the intended byte/atom boundary grid
use the same div_term when comparing spans across spaces
```

Under those constraints `.where` is a unique spatial key for the
occurrence span. It can distinguish spans with the same center but
different extent by magnitude. For example, a short span and a larger
span centered at the same point have the same angle but different
radius.

When an input-space span is transformed into PS codebook vectors, the
spatial origin changes: the codebook-local center may move because the
entry is indexed from the start of the codebook item rather than the
start of the input surface. The magnitude must not change, because the
span length is unchanged. This lets the model compare span extent across
input space and codebook-local space even when the center coordinate is
rebased.

`.where` recovers the occurrence span; it does not by itself resolve the
semantic or perceptual row. The byte-to-codebook path is:

```text
1. Decode `.where` to input-space start/end.
2. Slice surface_bytes[start:end].
3. Lookup those bytes in the PS codebook index.
4. Return the PS part id and learned `.what` vector.
```

The PS codebook should therefore keep both reconstructable bytes and a
fast byte lookup index:

```text
canonical_bytes_blob: uint8[M]
row_offsets:          int[V + 1]   # cumulative byte lengths
row_lengths:          int[V]
row_hashes:           uint64[V]
length_buckets:       int[max_len + 2]
optional radix/trie:  prefix and longest-match proposals
```

`row_offsets` is the cumulative-length field that maps a codebook row
back to bytes:

```text
row_i_bytes = canonical_bytes_blob[row_offsets[i]:row_offsets[i + 1]]
```

For lookup, cumulative length alone is not enough. Use decoded span
length to choose a length bucket, hash the surface bytes, binary-search
the sorted hashes for that length, and byte-verify against
`canonical_bytes_blob` to avoid collisions. The radix/trie can still be
used for proposal generation and longest-match fallback.

The exact decoded start/end should also be kept on `ObjectSubSpace` for
route replay and reconstruction bookkeeping:

```python
_span_start: [B, cap]
_span_end:   [B, cap]
```

## Perceptual Meronomy

PerceptualSpace starts with a guaranteed meronomy:

```text
S = whole surface
atoms = bytes / characters
```

The byte or character codebook is total, so the analyzer always has a
valid fallback cover.

As the PS codebook grows, larger entries become available:

```text
bytes -> characters -> morphemes -> words -> phrases -> constructions
```

A PS codebook entry is a perceptual part. It may or may not yet have a
stable symbolic meaning.

The PS meronomy stores ordered part structure:

```text
PS row id -> canonical byte/atom sequence
PS row id -> ordered child PS row ids
PS row id -> learned perceptual vector
```

Lookup can use exact hash/MPHF for complete spans and optional prefix
indexes for proposals. The meronomy itself is not the index; it is the
learned part-whole relation that lets a recognized surface construction
be re-expanded during reconstruction.

## Recursive Trie Lattice

PS should expose a guaranteed lattice of surface covers rather than a
fixed tokenization. The top and bottom are always present:

```text
top / EVERYTHING percept
  one object covering the whole input span
  maximal compression, minimal detail

bottom / byte-char percepts
  one object per byte/character successor
  maximal detail, always reconstructable
```

The learned codebook lives between those extremes:

```text
bytes/chars -> bridges -> morphemes -> words -> phrases -> constructions
```

The recursive trie is the exact index for this lattice. Bytes/chars are
self-indexing. Higher rows are paths over already-known child row ids and
overlap/rewrite labels:

```text
base alphabet:
  byte_or_char_id

higher alphabet:
  child_part_id + overlap_policy + rewrite_label

terminal node:
  PS OBJECT row id
```

Examples:

```text
"t"
  byte/char object

"the"
  path over byte/char objects

"the book"
  path over [the, "e b", book]

"book book"
  path over [book, "ok b", book]
  overlap policy = earliest compatible suffix/prefix match
```

Common words or common constructions may be cached as direct rows, but
the cache is an acceleration of the same recursive structure, not a
second identity system. Every cached object must still be recursively
re-expandable to bytes/chars.

The analyzer chooses a cover of the input span from this lattice:

```text
cover = ordered non-overlapping output objects
        whose recursive expansions reconstruct the input span
```

The two trivial covers are:

```text
[EVERYTHING(input)]
[byte_0, byte_1, ..., byte_N]
```

Useful covers are learned between them. Attention guides object size:

```text
tight / local attention
  favors smaller local objects

broad / sentence-level attention
  can favor phrases, constructions, or the EVERYTHING percept
```

The scoring objective should therefore choose the largest useful
compositional partition, not the largest possible object blindly:

```text
score(cover) =
  reconstruction confidence
  codebook familiarity
  route confidence
  attention compatibility
  STM / prediction usefulness
  lexical embedding coherence
  depth/detail penalty
```

If a coarse object reconstructs well and helps prediction, the depth
penalty rewards stopping there. If it loses needed detail, route and
reconstruction losses should push the analyzer to split further.

This makes "word" behavior emerge as one useful scale of PS object. The
same mechanism also supports non-word objects, punctuation, bridges,
phrases, and future modalities.

## Signed Neighborhood Evidence

PS lookup should separate surface identity from contextual evidence.
The trie/radix index answers "which known surface part could this span
be?" The signed-neighborhood encoder answers "what oriented context does
this occurrence carry?"

The existing Gaussian attention window is symmetric:

```text
d_i = i - k
g_i = exp(-(d_i * d_i) / (2 * sigma * sigma))
```

This preserves the center part and softly includes nearby parts. For PS
analysis, add an oriented companion channel:

```text
signed_g_i = sign(d_i) * g_i
```

The signed channel is zero at the center, so the center content must stay
explicit. A practical evidence vector is:

```text
center        = x_k
sym_context   = sum_i g_i * x_i
signed_context = sum_i signed_g_i * x_i
ps_evidence   = fuse(center, sym_context, signed_context)
```

An equivalent implementation may keep separate left and right context
channels instead of a single signed channel:

```text
left_context  = sum_{i < k} g_i * x_i
right_context = sum_{i > k} g_i * x_i
ps_evidence   = fuse(center, left_context, right_context)
```

This gives the analyzer non-symmetric evidence: `the book` and
`book the` no longer collapse to the same local context. The center part
keeps full-gain identity while proximal positions orient the occurrence
inside PS.

This evidence should feed route scoring, not replace structural
identity. Reconstruction still depends on:

```text
PS part id
canonical bytes
span_start / span_end
parent / child links
route id
surface rewrite metadata
```

The router should learn co-occurrence in rule context:

```text
score(route_id,
      parent_part_or_span,
      left_part_or_span,
      right_part_or_span,
      argument_position,
      return_position,
      signed_neighborhood_evidence)
```

This is where determiner-like, noun-like, suffix-like, punctuation-like,
or bracket-like behavior emerges. These are stable participation
patterns in route argument and return positions, not hand-authored POS
labels.

## Meronymic Operations

The first meronymic operation inventory should be small and
direction-neutral:

```text
stop
  Terminal case. In analysis, accept the current span as a PS codebook
  item. In synthesis, realize the terminal surface.

uniform
  Divide or combine near the midpoint. This is the fallback for unknown
  long spans and bootstrapping.

boundary
  Divide or combine at boundary evidence such as whitespace,
  punctuation, casing changes, byte-class transitions, or entropy
  spikes.

prefix
  Relate a known left perceptual chunk to the remaining span.

suffix
  Relate a known right perceptual chunk, suffix, clitic, or ending to
  the remaining span.

compound
  Relate a larger surface to two reusable known-ish chunks.

coordination
  Relate conjunction-like or disjunction-like material to its conjuncts.

quote_or_bracket
  Relate paired delimiters, quotes, parentheticals, and embedded spans.

```

`boundary.analyze` on spaces gives the current word lexer behavior:

```text
the book has a cover
-> [the] [book] [has] [a] [cover]
```

But it is only one meronymic operation. Other operations can learn that a
larger or smaller unit is better:

```text
[the book] [has] [a cover]
[the] [book] [has a cover]
[the book has a cover]
```

## Reusing LanguageLayer

The implementation should make PS analysis look like the reverse side of
the existing parser, with durable data stored on `ObjectSubSpace`.

Current symbolic parser:

```text
LanguageLayer.compose(data)
  repeatedly applies BinaryStructuredReductionLayer.forward
  adjacent symbols are reduced
  hard route comes from binary_tiling_viterbi
  soft route comes from binary_tiling_soft_dp
```

Desired PS analyzer:

```text
MeronymicLanguageLayer.analyze(object_subspace)
  repeatedly applies inverse/reverse routing
  active surface wholes are un-reduced into child surfaces
  hard route comes from the same Viterbi-style choice
  soft route comes from the same soft-DP/STE machinery
  writes durable meronymic route state to ObjectSubSpace
```

Where practical, this should be an adapter around `LanguageLayer` rather
than a new parser class. The adapter should:

```text
attach PS meronymic operations using the same op registry shape
read active PS spans from ObjectSubSpace
call generate/unreduce-style direction for PS forward analysis
call compose/reduce-style direction for PS reverse synthesis
write route ids, child links, and terminal hits back to ObjectSubSpace
```

If the current `generate()` path depends too much on cached compose
routes, the implementation should first factor out a shared inverse
routing primitive from `LanguageLayer.unreduce` / `reverse_stack` rather
than creating a separate PS-only router.

Do not assume that route replay is sufficient for PS analysis. The PS
forward path must be able to choose a decomposition for a newly observed
surface root even when no prior symbolic `compose()` trace exists. The
shared primitive should therefore expose:

```text
candidate route scoring
one hard Viterbi-style route
soft marginals for training
per-row route/provenance metadata
logical lengths and masks
```

The important ownership boundary is:

```text
LanguageLayer-like router:
  owns trainable routing modules and transient route caches

ObjectSubSpace:
  owns PS span buffers, part ids, parent/child links, route ids, and
  replayable analysis/synthesis traces
```

## Analyzer Contract

Input:

```python
surface_root: [B, 1, D] or encoded surface bytes/chars
```

Output:

```python
ps_terminal_what: [B, Kmax, D_ps]
ps_terminal_where: [B, Kmax, 2]
ps_terminal_ids:  [B, Kmax]
terminal_mask:    [B, Kmax]
terminal_len:     [B]
```

This output should be a terminal-stream adapter view over
`ObjectSubSpace` leaves whose
`_part_id` is known or whose fallback byte/atom terminal is guaranteed.
It is not a new durable term object. It is a source-agnostic view that
lets the current word/radix path, byte fallback, future PS analyzer, and
future multimodal analyzers all present the same contract.

It is not yet the symbolic parser's stack input. The PS-to-SS binding
layer converts these perceptual terminals into the existing stack-mode
`SubSpace` contract that `LanguageLayer` already consumes.

After PS-to-SS binding, STM should not care whether terminals came from:

```text
word lexer compatibility route
byte fallback
PS codebook span match
phrase construction match
future multimodal perceptual analyzer
```

## Forward PS Analysis

The first implementation should be Viterbi-style, matching the current
parser. The analysis target is a cover from the recursive trie lattice,
not a fixed token stream.

1. Start with one active root span covering the whole surface.
2. Build PS evidence for candidate spans, including center content,
   symmetric neighborhood context, and signed or left/right neighborhood
   context.
3. Score `stop` against the PS codebook for that span.
4. If `stop` wins above threshold, emit the span as a terminal.
5. Otherwise score meronymic `generate` candidates. Candidate scoring
   may use span identity, boundary features, part ids, signed
   neighborhood evidence, and role-specific projections for left child,
   right child, and parent/return positions.
6. Choose one hard route with the shared routing machinery.
7. Keep soft marginals for gradient support.
8. Repeat until every active leaf is a recognized terminal or byte/atom
   fallback.

The initial root span may stop at the EVERYTHING percept. It may also
split down to bytes/chars. Between those extremes, trie rows and ordered
part-signature rows offer candidate objects at word, bridge, phrase, or
construction scale. Attention and losses should bias which cover is
selected.

The word-lexer compatibility mode is:

```text
allowed routes = stop, boundary, uniform
boundary feature = whitespace
fallback = byte/character
route selection = one Viterbi route plus soft marginals
```

This preserves current word-level behavior while removing the assumption
that words are the permanent terminal unit.

## Reverse PS Synthesis

When reverse() runs from a CS-integrated idea:

```text
CS-integrated idea -> SS taxonymic analysis -> SS terminals
SS terminals -> PS meronymic synthesis -> reconstructed surface
```

PS synthesis should run the same meronymic operations in `compose`
direction:

```python
boundary.synthesize(left_surface, right_surface) -> parent_surface
suffix.synthesize(base_surface, suffix_surface) -> parent_surface
quote_or_bracket.synthesize(open_surface, body_surface) -> parent_surface
```

Spaces, affixes, punctuation, word order, and repair material are
surface realizations learned by PS meronymic operations. They should not
be hard-coded tokenizer artifacts.

The forward pass should preserve enough route metadata for reverse:

```text
operation id
parent span id
left/right child span ids
terminal PS row id
route score / entropy
surface rewrite markers when needed
```

Reverse can then choose between:

```text
exact replay of observed route metadata
generative replay from learned meronymic route scores
```

## PS to CS to SS Binding

Each emitted PS terminal is an OBJECT percept. It carries:

```python
ps_id:   [B, Kmax]
ps_vec:  [B, Kmax, D]
```

ConceptualSpace integrates the OBJECT percept and asks SymbolicSpace to
resolve it to a stable symbolic row:

```python
ss_id:   [B, Kmax]
ss_vec:  [B, Kmax, D]
```

If no stable symbolic entry exists:

```text
ss_id = -1
ss_vec = NULL_SEM
grounded = false
```

Repeated presentation creates or strengthens an SS entry. For text
objects, SymbolicSpace should also ensure a stable `WORD` parent symbol
exists and attach the object-derived symbol under it:

```text
PS OBJECT row "book"
  -> CS integrated object idea
  -> SS row BOOK_SYMBOL
  -> parent SS row WORD
```

This makes words a taxonomic family of symbols, not a separate tokenizer
ontology. Non-text modalities can bind their object rows to the same
symbol when experience supports that identity:

```text
PS text OBJECT "book"      -> SS BOOK_SYMBOL -> WORD
PS visual OBJECT book_img  -> SS BOOK_SYMBOL -> OBJECT / IMAGE_OBJECT
PS audio OBJECT spoken_book -> SS BOOK_SYMBOL -> WORD / AUDIO_OBJECT
```

The SS row may have multiple parents or typed role links. The important
point is that the symbol is grounded in PS object rows, while the
taxonomic organization lives in SS.

The taxonomy may later attach abstractions such as grammatical roles, but
the grammar file should not require those categories to be pre-authored.

## Lexical Object Embeddings

The recursive trie gives exact identity and reconstruction. The PS
`.what` vectors should still be tuned relative to one another so OBJECT
rows acquire lexical embeddings.

This tuning should not replace the trie or ordered part signatures. It
should train the continuous geometry used for route scoring,
neighborhood prediction, and PS-to-SS binding:

```text
exact identity:
  recursive trie path
  child part ids
  overlap policy
  canonical byte reconstruction

learned lexical geometry:
  PS OBJECT .what vector
  signed-neighborhood evidence
  co-occurrence / prediction context
  route participation
  PS-to-SS binding consistency
```

The PS codebook should be able to learn that objects are near, far,
part-of, or contextually substitutable without losing exact
reconstruction. Candidate training signals:

```text
signed-neighborhood prediction
  objects that appear in similar oriented contexts move closer

route co-occurrence
  objects that repeatedly participate in the same meronymic rule roles
  develop compatible vectors

monotone parthood
  child.what <= parent.what where the recursive trie records parthood

contrastive lexical separation
  objects with different bytes / incompatible routes remain distinct

PS-to-SS consistency
  PS objects bound to the same SS symbol are pulled toward a stable
  shared symbolic neighborhood without becoming the same structural row
```

For a monotone mereological embedding, enforce parthood softly:

```text
part_loss(child, parent) = relu(child.what - parent.what + margin).sum()
```

The vector is therefore a learned lexical/mereological summary. The
recursive trie remains the database of exact parts, order, overlap, and
bytes.

## STM Interface

The analyzer emits terminals into STM.

The required STM-facing contract is not a new `term_*` object. It is the
existing stack-mode `SubSpace` used by `LanguageLayer.shift`,
`LanguageLayer.reduce`, and `LanguageLayer.unreduce`:

```python
stack_subspace.what:        [B, Kmax, D_ss]
stack_subspace.where:       [B, Kmax, W]
stack_subspace.activation:  [B, Kmax]
```

For analyzer terminals, `.where` is the endpoint-sum spatial key of the
source occurrence span:

```text
stack_subspace.where = phase(span_start) + phase(span_end)
```

The symbolic terminal identity is carried by `.what` after PS-to-SS
binding. The perceptual terminal identity is carried by `ps_id` /
`_part_id`. Rule identity and selected operations are carried by
route/provenance metadata (`_route_id`, rule ids, analysis routes), not
by replacing `.where` with a grammar namespace integer.

The live length is derived, not separately stored:

```python
terminal_len = (stack_subspace.activation.abs() > 0).sum(dim=-1)
```

The streaming path can push one terminal at a time through the existing
API:

```python
LanguageLayer.shift(stack_subspace, terminal_what, span_where)
```

where `span_where` is the endpoint-sum key for the source span. The
current scalar `where_id` implementation must be updated as part of the
analyzer migration rather than preserved as the target contract.

A batch analyzer may also initialize the same buffers directly, provided
it preserves the exact stack-mode meanings:

```text
.what        = SS terminal vectors after PS-to-SS binding
.where       = endpoint-sum source-span keys
.activation  = occupancy mask, 1 for live terminal slots
```

The current word-derived terminal sequence remains a compatibility shim,
but its emitted `.where` should follow the endpoint-sum span contract:

```text
current:
  word lexer -> terminal stream -> STM

after this plan:
  PS meronymic analyzer -> PS-to-SS binding -> stack SubSpace -> STM
```

## Symbolic Composition

Once terminals are in STM, the existing symbolic router applies a soft
superposition of taxonymic operations rather than a hand-written
part-of-speech grammar.

The grammar file should shrink toward operator availability:

```text
lift
lower
union
intersection
not / non
conjunction / disjunction
isEqual
isPart
query
exist
copy / swap / mark as surface policies
```

Categories and parts of speech become learned taxonomic structure, not
required grammar syntax.

Stable surface partitions become PS meronymic entries. Stable symbolic
roles become SS taxonomic entries. A "noun", "determiner", "verb",
"modal", or "relation marker" is therefore a learned role family, not an
input requirement.

Composition target shapes:

```text
absolute idea:
  concepts: [B, 1, D]

relative idea:
  concepts: [B, 3, D]  # predicate, idea1, idea2
```

Physically both live in fixed `[B, Kmax, D]` buffers with `lengths` equal
to 1 or 3.

## Reconstruction Objective

Forward:

```text
surface -> PS meronymic analysis -> STM terminals
STM terminals -> SS taxonymic synthesis -> CS-integrated idea
```

Reverse:

```text
CS-integrated idea -> SS taxonymic analysis -> SS terminals
SS terminals -> PS meronymic synthesis -> reconstructed surface
```

Losses:

```text
surface reconstruction loss
PS-to-SS binding consistency loss
STM prediction loss
truth/semantic consistency loss
analysis-depth penalty
route entropy penalty or regularizer
lexical object embedding losses
```

If reconstruction fails from a coarse perceptual cover, the model should
learn to analyze more finely. If reconstruction succeeds from a coarse
cover, the depth penalty rewards stopping there.

## Implementation Phases

### Phase 0: Close STM Prerequisites

Finish the STM Serial/Parallel work before replacing the terminal source.
This means STM should already have a stable predict-then-perceive path,
serial boundary behavior, rule/context-aware prediction, trained
intermediate supervision where required, reverse-from-STM behavior, and
clear absolute vs relative end-state handling.

The analyzer should not compensate for incomplete STM sequencing. It
should only replace how terminals are produced.

### Phase 1: Terminal-Stream Adapter

Add a terminal-stream adapter around the existing word/radix path with no
change to emitted terminal order or terminal content. This adapter should
also introduce the endpoint-sum `.where` contract for spans.

The adapter should expose:

```python
terminal_what:  [B, Kmax, D]
terminal_where: [B, Kmax, 2]
terminal_ids:   [B, Kmax]
terminal_mask:  [B, Kmax]
terminal_len:   [B]
```

It should also be able to materialize the existing stack-mode `SubSpace`
after PS-to-SS binding:

```python
subspace.what
subspace.where
subspace.activation
```

Keep externally visible word-lexer APIs stable. Rename only comments,
docstrings, and internal helper names where doing so reduces ambiguity.

The adapter may carry optional PS evidence fields for analyzer-side
experiments, but the initial terminal sequence must remain equivalent to
the current word/radix terminal path.

### Phase 2: Shared Router Direction Audit

Document the existing reduce and unreduce paths:

```text
LanguageLayer.compose
LanguageLayer.generate
LanguageLayer.reduce
LanguageLayer.unreduce
LanguageLayer.reverse_stack
BinaryStructuredReductionLayer.forward
binary_tiling_viterbi
binary_tiling_soft_dp
compact_hard
compact_soft
```

Identify the smallest shared primitive needed so PS forward analysis can
call the same route machinery in the inverse direction.

This phase must decide whether the current `generate()` path is enough or
whether route scoring must be factored out of `compose()`,
`unreduce()`, and `reverse_stack()` first. The required result is an
independent route selector that can analyze a new perceptual surface
without relying on a cached symbolic compose trace.

Because `.where` becomes a spatial span key, `unreduce()` and
`reverse_stack()` must not rely on `.where` to decode grammar rule ids.
Reverse routing should read rule identity from route/provenance metadata
or an explicit rule carrier.

### Phase 3: Recursive Trie Lattice And OBJECT Codebook

Build the PS object codebook as a recursive trie lattice:

```text
EVERYTHING percept
byte/character percepts
ordered child-part paths
overlap/rewrite labels
terminal OBJECT rows
common-object cache rows
```

Every terminal OBJECT row must be recursively re-expandable to
bytes/chars. The cache for common words or constructions may skip search
work, but it must not bypass the recursive identity model.

The codebook should expose:

```text
lookup bytes/span -> candidate OBJECT rows
lookup ordered part signature -> compound OBJECT row
expand OBJECT row -> ordered children or bytes/chars
object .what vector -> learned lexical/mereological evidence
object parent/children -> explicit reconstruction structure
```

The lattice must always admit both trivial covers: EVERYTHING and
byte/char partition.

### Phase 4: Minimal Meronymic Operation Registry

Register PS meronymic operations with the same shape as grammar ops:

```text
compose(left_surface, right_surface) -> parent_surface
generate(parent_surface) -> left_surface, right_surface
```

Start with deterministic or simple learned versions of:

```text
stop
boundary on whitespace
uniform fallback
byte/character fallback
```

Use the existing radix/percept store for `stop`, known span lookup,
canonical bytes, and byte fallback. Do not duplicate percept identity
inside `ObjectSubSpace`.

Route scorers should be allowed to consume endpoint-sum span keys and
signed-neighborhood evidence, but deterministic compatibility routes
must not depend on signed-neighborhood evidence.

### Phase 5: ObjectSubSpace Carrier

Add an `ObjectSubSpace` carrier analogous to `WordSubSpace`, but for PS
meronymic state. It should be a durable state holder, not a parser
implementation.

Minimum buffers:

```python
_buffer:       [B, cap, percept_dim]
_part_id:      [B, cap]
_span_start:   [B, cap]
_span_end:     [B, cap]
_span_where:   [B, cap, 2]
_parent_id:    [B, cap]
_left_id:      [B, cap]
_right_id:     [B, cap]
_route_id:     [B, cap]
_route_score:  [B, cap]
_depth:        [B]
```

It should expose push/pop or insert/update helpers with the same
discipline as `WordSubSpace`: all parallel buffers stay in sync, and
logical depth/masks determine which slots are live.

### Phase 6: Temporary Stack View

Build a stack-mode `SubSpace` adapter view over active `ObjectSubSpace`
items when invoking the shared routing machinery:

```python
subspace.what:        [B, K, D_ps]
subspace.where:       [B, K, W]
subspace.activation:  [B, K]
```

This mirrors how the symbolic stack route uses a temporary stack-mode
`SubSpace` while durable symbolic state remains on `WordSubSpace`.
For PS analyzer terminals, `subspace.where` should be populated from
`ObjectSubSpace._span_where`.

### Phase 7: Routing PS Analyzer

Refactor the PS radix/lexer path into a routing analyzer:

```text
active span -> stop or meronymic generate route -> terminal or child spans
```

Before scoring a route, the analyzer should build or retrieve occurrence
evidence for each candidate span:

```text
center content
endpoint-sum span where and decoded length
symmetric Gaussian neighborhood
signed Gaussian neighborhood or explicit left/right contexts
span length and boundary features
part-id / trie-hit features
```

The first route selector should match the current parser:

```text
one Viterbi route
soft DP marginals for training
no beam
```

The analyzer writes selected part ids, route ids, span links, and route
scores back to `ObjectSubSpace`.

The first analyzer mode should be a compatibility mode: with only
`stop`, whitespace `boundary`, `uniform`, and byte/character fallback
enabled, it should match the current lexer/radix terminal sequence.

### Phase 8: PS-to-CS-to-SS Binding Hook

When a PS terminal enters SymbolicSpace:

1. Look up existing PS-to-SS binding.
2. If present, emit the SS vector.
3. If absent, emit `NULL_SEM` and record exposure.
4. Promote repeated stable exposures into SS codebook rows.
5. Ensure text OBJECT rows bind under an SS `WORD` parent symbol.
6. Allow non-text modality OBJECT rows to bind to the same SS symbol when
   evidence supports shared identity.

### Phase 9: Reverse PS Synthesis

Use the same meronymic operations in compose direction to reconstruct
surface forms from SS terminals.

Start with exact replay from `ObjectSubSpace` route metadata and radix
canonical bytes. Then allow generative replay from learned route scores.
This keeps reconstruction testable before the model learns good
surface-realization policies.

### Phase 10: Operator-Superposition Composition

Allow STM to apply soft operator distributions without requiring explicit
part-of-speech categories in the grammar file.

This should preserve compatibility with the existing typed grammar while
allowing an operator-only grammar variant.

This is a later symbolic-composition experiment, not a prerequisite for
the first PS analyzer. Existing grammar categories may remain in place
while terminal emission is migrated.

### Phase 11: Documentation And Test Closure

Before the analyzer migration is considered complete, update the docs and
tests that describe or exercise `.where`, terminal streams, recursive
trie lattice lookup, PS OBJECT rows, PS-to-CS-to-SS binding, WORD-parent
taxonomy, lexical object embedding, and reverse reconstruction.

Documentation updates should include:

```text
doc/STM.md
doc/Language.md
doc/Spaces.md or the nearest SubSpace/encoding doc
doc/Params.md / XML schema notes if nWhere or encoding config changes
this plan file's implementation status
```

Test updates should land with the phase that changes the behavior they
cover. At minimum, the final closure should verify endpoint-sum spatial
decode, recursive trie top/bottom covers, compound OBJECT re-expansion,
byte-to-codebook lookup, WORD-parent symbol binding, terminal stream
`.where`, lexical vector tuning, and reverse surface reconstruction.

## Tests

Proposed tests:

```text
test_ps_analyzer_byte_fallback.py
  Unknown surface is covered by byte/character terminals.

test_ps_analyzer_prefers_known_word.py
  Once "book" exists in PS codebook, analyzer emits one terminal instead
  of four byte terminals.

test_terminal_stream_fixed_capacity.py
  Analyzer output shape remains [B, Kmax, D] while terminal_len changes.

test_object_subspace_parallel_buffers.py
  ObjectSubSpace updates part ids, spans, route ids, and depth without
  desynchronizing parallel buffers.

test_object_subspace_terminal_view.py
  Recognized ObjectSubSpace leaves produce the PS terminal stream view.

test_ps_to_ss_null_before_binding.py
  New PS terminal emits NULL_SEM before SS binding exists.

test_ps_to_ss_binding_after_repetition.py
  Repeated PS terminal creates or resolves an SS symbol row.

test_stm_accepts_analyzer_terminals.py
  STM consumes analyzer output through the same terminal-stream contract
  used by the current word lexer.

test_analysis_depth_penalty.py
  Coarse and fine covers with equal reconstruction prefer the coarser
  cover.

test_boundary_matches_word_lexer.py
  With only boundary-on-space enabled, analyzer output matches the
  current word lexer terminal sequence.

test_meronymic_route_uses_viterbi_not_beam.py
  Analyzer selects one hard route and exposes soft marginals, matching
  BinaryStructuredReductionLayer behavior.

test_meronymic_reverse_replays_surface.py
  ObjectSubSpace route metadata reconstructs the observed surface.

test_meronymic_synthesis_can_emit_spaces.py
  boundary.synthesize can place spaces without relying on word-tokenizer
  state.

test_signed_neighborhood_preserves_center.py
  Signed context is zero at the center while the center content remains
  full-gain.

test_signed_neighborhood_is_directional.py
  Reversing neighbor order changes signed/left-right evidence even when
  the unordered neighbor set is the same.

test_route_scoring_consumes_signed_context_without_identity_loss.py
  Route logits can use signed-neighborhood evidence, but reverse
  reconstruction still uses part ids, spans, and route metadata.

test_endpoint_sum_where_decodes_span.py
  Endpoint-sum `.where` decodes center and length, then recovers
  start/end on the byte/atom boundary grid.

test_endpoint_sum_where_rejects_ambiguous_period.py
  Span lengths that approach or exceed the recoverable half-period are
  rejected or forced through a larger namespace period.

test_endpoint_sum_where_rebase_preserves_length.py
  Rebasing an input-space span to codebook-local coordinates changes the
  decoded center but preserves `.where` magnitude / decoded length.

test_codebook_lookup_from_recovered_span_bytes.py
  Decode `.where` to start/end, slice source bytes, length-bucket and
  hash lookup the PS row, then byte-verify against canonical bytes.

test_terminal_stream_where_is_endpoint_sum.py
  Analyzer terminals carry endpoint-sum source-span keys in
  stack_subspace.where.

test_recursive_trie_has_top_and_bottom_covers.py
  The PS lattice always admits the EVERYTHING percept and the byte/char
  partition for the same input span.

test_recursive_trie_compound_reexpands_to_bytes.py
  Cached words and compound OBJECT rows recursively reconstruct to the
  same canonical bytes as their child-part paths.

test_largest_useful_partition_respects_attention.py
  Broad attention can select phrase/construction objects; tight
  attention biases toward smaller local objects.

test_ordered_overlap_signature_builds_compound.py
  Ordered reusable parts plus overlap policy build an invertible
  compound such as [book, "ok b", book] -> "book book".

test_object_percept_creates_word_parent_symbol.py
  A stable text OBJECT percept creates or resolves an SS symbol under
  the WORD parent.

test_multimodal_object_binds_to_existing_word_symbol.py
  A non-text OBJECT can bind to an existing SS word symbol when evidence
  supports shared identity.

test_ps_object_vectors_receive_lexical_tuning.py
  Co-occurrence, signed-neighborhood, route-role, and parthood losses
  update PS OBJECT .what vectors without changing recursive identity.
```

## Non-Goals

This plan does not implement:

```text
full English syntax
full generation quality
learned grammatical categories
beam or K-best parsing in the first patch
large learned meronymic inventory in the first patch
multimodal perceptual analysis
```

It replaces fixed word lexing with a PS meronymic analyzer that reuses
the current signal-router parser machinery in the opposite direction.
