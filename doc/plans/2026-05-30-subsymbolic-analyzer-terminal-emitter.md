# Subsymbolic Analyzer and Terminal Emitter

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
ps_terminal_ids:  [B, Kmax]
terminal_mask:    [B, Kmax]
terminal_len:     [B]
```

This output should be a view over `ObjectSubSpace` leaves whose
`_part_id` is known or whose fallback byte/atom terminal is guaranteed.
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
parser.

1. Start with one active root span covering the whole surface.
2. Score `stop` against the PS codebook for that span.
3. If `stop` wins above threshold, emit the span as a terminal.
4. Otherwise score meronymic `generate` candidates.
5. Choose one hard route with the shared routing machinery.
6. Keep soft marginals for gradient support.
7. Repeat until every active leaf is a recognized terminal or byte/atom
   fallback.

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

## PS to SS Binding

Each emitted PS terminal carries:

```python
ps_id:   [B, Kmax]
ps_vec:  [B, Kmax, D]
```

SymbolicSpace attempts to resolve it to:

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

Repeated presentation creates or strengthens an SS entry. The shared
taxonomy binds the perceptual and symbolic rows:

```text
PS row "book" <-> taxonomy/meta node <-> SS row BOOK_SYMBOL
```

The taxonomy may later attach abstractions such as grammatical roles, but
the grammar file should not require those categories to be pre-authored.

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

Current first-patch `.where` encoding stores the integer location in
`where[..., 0]`. For a terminal symbol this is:

```python
where_id = grammar.where_id_for_symbol(ss_id)
```

The live length is derived, not separately stored:

```python
terminal_len = (stack_subspace.activation.abs() > 0).sum(dim=-1)
```

The streaming path can push one terminal at a time through the existing
API:

```python
LanguageLayer.shift(stack_subspace, terminal_what, where_id)
```

A batch analyzer may also initialize the same buffers directly, provided
it preserves the exact stack-mode meanings:

```text
.what        = SS terminal vectors after PS-to-SS binding
.where       = grammar namespace locations for terminal symbols
.activation  = occupancy mask, 1 for live terminal slots
```

Current per-word emission remains a compatibility shim:

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
```

If reconstruction fails from a coarse perceptual cover, the model should
learn to analyze more finely. If reconstruction succeeds from a coarse
cover, the depth penalty rewards stopping there.

## Implementation Phases

### Phase 1: Terminal-Stream Naming

After the STM Serial/Parallel plan, audit code paths that say "word" when
they mean "terminal". Rename comments, docstrings, and internal helper
names where safe.

Do not rename externally visible word-lexer APIs in this phase.

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

### Phase 3: ObjectSubSpace Carrier

Add an `ObjectSubSpace` carrier analogous to `WordSubSpace`, but for PS
meronymic state. It should be a durable state holder, not a parser
implementation.

Minimum buffers:

```python
_buffer:       [B, cap, percept_dim]
_part_id:      [B, cap]
_span_start:   [B, cap]
_span_end:     [B, cap]
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

### Phase 4: Temporary Stack View

Build a stack-mode `SubSpace` adapter view over active `ObjectSubSpace`
items when invoking the shared routing machinery:

```python
subspace.what:        [B, K, D_ps]
subspace.where:       [B, K, W]
subspace.activation:  [B, K]
```

This mirrors how the symbolic stack route uses a temporary stack-mode
`SubSpace` while durable symbolic state remains on `WordSubSpace`.

### Phase 5: Meronymic Operation Registry

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

### Phase 6: Routing PS Analyzer

Refactor the PS radix/lexer path into a routing analyzer:

```text
active span -> stop or meronymic generate route -> terminal or child spans
```

The first route selector should match the current parser:

```text
one Viterbi route
soft DP marginals for training
no beam
```

The analyzer writes selected part ids, route ids, span links, and route
scores back to `ObjectSubSpace`.

### Phase 7: SS Binding Hook

When a PS terminal enters SymbolicSpace:

1. Look up existing PS-to-SS binding.
2. If present, emit the SS vector.
3. If absent, emit `NULL_SEM` and record exposure.
4. Promote repeated stable exposures into SS codebook rows.

### Phase 8: Reverse PS Synthesis

Use the same meronymic operations in compose direction to reconstruct
surface forms from SS terminals.

Start with exact replay from `ObjectSubSpace` route metadata. Then allow
generative replay from learned route scores.

### Phase 9: Operator-Superposition Composition

Allow STM to apply soft operator distributions without requiring explicit
part-of-speech categories in the grammar file.

This should preserve compatibility with the existing typed grammar while
allowing an operator-only grammar variant.

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
