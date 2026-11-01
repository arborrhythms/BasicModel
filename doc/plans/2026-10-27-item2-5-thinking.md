# Item 2.5: thinking as derivation over chunks with trust, `intuit` as proposer (Claude, 2026-10-27; spec in code names, with decisions for Alec)

Written from Alec's eight answers and two exchanges of 2026-10-27
([notes §10](2026-10-26-thought-design-notes.md#10-alecs-answers-to-the-eight-decisions-2026-10-27-with-claudes-readings-and-the-open-points))
over the exploration of 2026-10-26 (notes §§1–9) and the `intuit` spec
([2026-10-21-intuit.md](2026-10-21-intuit.md) §5). Runs after item 2.7
lands (Alec, 2026-10-28: one more 2.7 round finishes that item, then 2.5).
State on 2026-10-28: D1–D4 decided (§7); round A's Codex text is §8, held
until 2.7 lands; §§9–11 record the 2026-10-28 exchanges on training,
memory and the pick and govern round B; §12 is the summary.

## 1. The record (Alec, 2026-10-27)

1. The eight STM chunks are the premise set, and a chunk stores its trust
   channel when a row is pulled from LTM. Directly observed text carries
   1; when words are turned into objects the trust goes to 0, since the
   statement's truth for other objects is unknown. 0 is unknown.
2. The serial slot is the episode's focus chunk, pinned; STM slots are
   referenced by `.where`, as the percept and concept banks are.
3. No `descend`/`return` discipline: `query()` pulls LTM into STM, and a
   chunk constitutes either a relation or a truth (a leaf) of the proof
   tree.
4. `intuit`'s answer is a number of relations over a number of ideas,
   held in one STM slot: sets of STM0 ideas combined into terms (AND),
   STM1 relations over the terms building a disjunction (OR), a
   disjunctive normal form; NOT applied to the ideas as the last
   operation before the relations.
5. One `infer`: "if A then B" is a relation; finding A (a type) or a (a
   token of it) adds B (or b) if not already present; negation on
   operands gives the contrapositive, so no `refute`; equality is a
   relation, so no `substitute`; reasoning runs from antecedent to
   consequent and from consequent back toward the antecedent.
6. `intuit` writes chunks.
7. Intuition builds the candidates; `infer` selects from them, creating
   a backprop chain over them.
8. Derived conclusions go to LTM0; reasoning chains are preserved and
   forgotten later if unused.
9. A question chunk is the empty reference; it claims no trust bound.

## 2. What exists, in code names

- *Chunks.* `STMChunks.empty(batch, capacity, width)` builds the thirteen
  `FIELDS`: `buffer [B, C, width]`, `depth`, `orders`, `grammar_orders`,
  `concept_rows`, `concept_activations`, `chunk_refs [B, C, 2, 2]`
  (tagged references, tags `SYMBOL`, `IDENTITY_REF`, `ACTION`,
  `RELATION`, `OCCURRENCE`), `chunk_identity`, `chunk_timeline`,
  `chunk_stack` (newest-first position, `-1` off stack), `chunk_read`
  (tick), `chunk_where [B, C, 4]`, `chunk_when [B, C, 4]`. Functional
  updates: `allocate_chunk` (a free off-stack chunk, else the oldest read),
  `deposit(..., where=, when=, stack=)`, `read`, `clear`, `keep_close`,
  `refresh`, `bind_menu`, `apply_choice`. No trust field.
- *Records.* `Thoughts.ThoughtRecord(id, episode, kind, level, meaning,
  operation, cost, budget, pressure, forced, support_true, support_false,
  evidence_kind, sources, reason, semantic_delta, result, address,
  opened_by_chooser)`; kinds `begin`, `thought`, `descend`, `return`,
  `cutoff`, `finish` (the nested-context transitions `ask` already uses
  stay); evidence kinds include `inference`, `question`, `retrieval`,
  `subgoal`, `implication`, `truth`, `reference`. `LevelledThoughtHistory`
  (`begin_thought_episode(meaning, b, work_budget)`, `commit_thought`,
  `finish_thought`, `cutoff_thought`, `replay_thoughts`, `thought_extras`).
- *Operations.* `Queries.THOUGHT_EXECUTORS`: `ThoughtExecutorDescriptor(
  semantic_id, domain, argument_kinds, write_target: Mind, read_scope,
  write_scope, evidence_kind, executor, evidence_pair, ...)`; today
  `query`, `ask`, `bind`, `not`, `part`, `whole`, `equal`, `isPart`,
  `isEqual`, `implies`, `isImplied`, `isTrue`, `exist`, `gain`; the
  `<thought>` grammar section lists them as `x_argout1 = x.thought(...)`.
  `Mind` is `AccessibleMind.Subsystem` (`SERIAL`, `KNOWING`, `SYMBOLIC`,
  `LTM`, `PRIMING`, `BUDGET`, ...); `Mind.SERIAL` is the write target of
  every serial operation (34 uses in `Queries.py`).
  `ThoughtFaces.is_implied` returns `(min(link+, premise+), min(link-,
  premise+))` with the witnesses; `Queries.ThoughtConceptualCapability.negate`
  is the lane exchange; `ConceptEvidence.union` is the idempotent max.
- *Chooser and credit.* `Models._choose_selected_thought_action(root,
  active, actions, row, level, pressure, ...)` puts every
  `ThoughtOperationCandidate(operation, request, open_roles)` into the
  grammar's softmax; credit through `SentenceCredit.score_function` and
  `ThoughtCredit`.
- *Writer.* `ThoughtStream.write(model, meaning, row)` admits conclusions
  and unresolved questions into LTM0 (`LTM0.KINDS` has `inference`,
  `question`, `fact`, ...) through the semantic-chain writer or the
  checked-binding writer; `_thought_witnesses`, `_bound_roles`,
  `_open_references`, `_producing_operation` are the bindings it reads.
- *`intuit`.* Specified (intuit spec §5), not built: one layer of
  max-min inference over LTM1 rows into the knowing, the knowing effect
  write; the priming surface (item 2.9) is its filter (notes §4).

## 3. The design, in code names

### 3.1 Chunks carry trust

`STMChunks.FIELDS` gains `chunk_truth [B, C, 2]` (float, the for and
against lanes), preallocated in `empty`, cleared with the chunk, written
by `deposit(..., truth=)` and never resized. Sources of the pair:

- a row pulled by `query` or `ask`: the LTM0 row's evidence pair or the
  LTM1 row's lanes (`store.meaning_of(location)`'s `evidence_pair`);
- the registered sentence's own ideas (observed text): `(1, 0)` for the
  idea about the token it was about;
- an idea rewritten over a type or another object with no derivation
  (a generalization): `(0, 0)`, unknown, until `infer` or evidence fills
  it;
- a derived idea: the derivation's pair (3.4);
- a question chunk (3.4): `(0, 0)`.

`ThoughtRecord.support_true/false` continue to carry the pair in the
history; the chunk's pair is the working copy the operators read.

### 3.2 Chunks addressed by `.where`; the focus chunk

A thought operand or result names a chunk by its `.where` band, as the
percept and concept banks name their rows: `STMChunks.locate(state,
where) -> index` by exact match on `chunk_where` (`-1` when absent), and
every executor that reads or writes a chunk resolves through it.
`ThoughtRecord.address` becomes the chunk's `.where` (four floats) rather
than a physical index; `replay_thoughts` resolves it the same way.
`Mind.SERIAL` stays the subsystem name in the descriptors; its write is
`_write_serial(context, value)`, which deposits into the episode's focus
chunk. `begin_thought_episode` deposits the root with `stack=True`, so
the focus is pinned on `chunk_stack` while the episode is open, and
records `focus_where`; the serial accumulation is that chunk. Eight
chunks bound live premises plus focus; an intermediate result not pinned
can be evicted by `allocate_chunk`'s oldest-read rule, which is
forgetting a lemma, recoverable by `query` or by deriving it again.
*Claude's call (§5 D3).*

### 3.3 `intuit`: the DNF in one slot, the candidates beside it

`intuit` (descriptor as the intuit spec §5 gives it, `write_target`
`Mind.SERIAL` in addition to `Mind.KNOWING`) runs one layer over the rule
rows the priming surface selects: the LTM1 rows whose referenced symbols
carry excess on `_priming_boosts` (through `_csw_row_of`), bounded by
`work_budget`. For each rule `r = (S1, S2)` with lanes `(t+, t-)`:

- NOT first: an operand idea read with the lane exchange when the rule's
  antecedent names its negation (`negate`), applied once, before any
  combination; relations are never negated;
- AND: the term is the meet of the knowing with the antecedent's ideas,
  the meet ratio `coverage` of intuit spec §5, giving `f+ = min(t+,
  coverage)`, `f- = min(t-, coverage)`;
- OR: the slot's `buffer` is the union (coordinatewise max per lane) of
  every fired consequent `c(S2)` clipped at its firing degree, the knowing
  effect's own representation; the focus chunk receives it with
  `chunk_truth` the maximum firing pair and evidence kind `intuition`.

The relations over ideas are not in the vector; they are the candidate
list the firing produced: one `ThoughtOperationCandidate(infer, request)`
per fired rule, `request` the instantiated consequent with the firing
pair, `open_roles` empty, sources `(premise chunk .where, rule row)`.
The list is the menu `_choose_selected_thought_action` draws from on the
next step, and one `ThoughtRecord(kind='thought', operation='intuit',
sources=(rule references...), evidence_kind='intuition')` records the
firing so the slot is replayable. The pair `(f+, f-)` is provisional:
strong forcing (3.6) does not accept it as an answer until `infer`
selects it.

Three later readings govern how this is built (2026-10-28): the terms
are evaluated through live reads of the referenced rows, so the answer's
loss reaches the referenced ideas' definition edges and the chooser and
nothing in LTM (§9, §10); the layer is order zero to order zero, higher
orders entering as their zeroth-order constituents (§11); the pick of
the most significant candidate is the chooser's affine readout over the
focus and the candidate, not a charted Sigma or Pi (§11).

### 3.4 `infer`: one operator, two directions, negation on operands

`ThoughtExecutorDescriptor('infer', 'symbolic-derivation', ('reference',
'reference'), Mind.SERIAL, (Mind.SERIAL, Mind.SYMBOLIC, Mind.LTM,
Mind.TAXONOMY, Mind.BUDGET), (Mind.SERIAL,), 'inference', _infer,
evidence_pair=True, relation_directions=((0, 2), (2, 0)),
method_grants=(('ltm', ('relation_evidence', 'premise_evidence')),
('taxonomy', ('evidence', 'neighbors'))))`, grammar rule `infer_argout1 =
infer.thought(infer_argin1, infer_argin2)` with the permutation family
for the backward direction. Operands are chunks (by `.where`): a premise
or goal, and a rule, an LTM1 row (kind implication, an IS row, a
kind-level statement, or an equality).

- *Match.* The premise matches `S1` by identity (`isEqual`), by
  membership (the premise's subject a member of `S1`'s kind along the Part
  edges `ConceptIndex` records, read through `Priming.part_edges`'s cache
  of item 2.9), or by instantiation (a token of the type); the match kind
  is recorded in `reason`.
- *Forward* (`(0, 2)`): writes `S2` with the member substituted into the
  focus chunk, pair `(min(premise+, rule+), min(premise+, rule-))` (the
  lane rule `is_implied` already uses), evidence kind `inference`,
  `sources = (premise .where, rule row)`. If a chunk with the same
  `role_refs` already exists, its pair is raised by the maximum (OR over
  alternatives) and no chunk is added: B is added only if not present
  before.
- *Backward* (`(2, 0)`): the goal matches `S2`; writes `S1`
  (instantiated) as a question chunk: `_open_references=(('referent',
  0),)`, pair `(0, 0)`, evidence kind `subgoal`. The chooser then fills
  it by `query` or `ask`; a later forward `infer` over the filled chunk
  produces the goal with its pair. Nothing is claimed by the question.
- *Negation.* `not` on an operand before `infer` is the contrapositive;
  no `refute`.
- *Equality.* An `isEqual` row as the rule substitutes one term for the
  other in the premise; no `substitute`.
- *Chaining.* Transitivity over IS rows is two forward steps; a
  relation-over-relation rule matches by kind and by its two references'
  identity or membership, no further unification (the v1 boundary).

### 3.5 The implicit route: pervasion in order-0 space (notes §9)

No new operator. A question about an object and a property narrows the
knowing to the object's mereology (its parts and wholes by the Part edges
and the percept `partOf` order of item 2.8) and conditions it with the
property's code; `intuit` over that narrowed knowing gives, per part, the
meet of the property with what is known of the part in both lanes; the
aggregate over the parts is the minimum per lane (pervasion is complete
only when every part carries it; the mixed case keeps both lanes, never
an average). The result is written into a chunk with evidence kind
`pervasion` and that pair. At the close the implicit and the explicit
results are judged together: agreement fills one lane, disagreement
fills both, and that corner prompts refinement (Orders §10's rule for
implicit and explicit definitions). *Claude's calls (§5 D2).*

### 3.6 Forcing: intuition proposes, `infer` selects, the close accepts a pair

Weak: the `<thought>` section at a question offers `ask`, `query`,
`intuit`, `infer`, `not`, `isTrue`, `bind`, `exist` and conclude, so every
path is a sequence of sound steps and the chooser decides which premises
and rules. Strong: the answer path (`ThoughtAnswer`, the closing) accepts
as the answer only focus content whose pair came from `infer`, from
recall (`query`, `isTrue`, a derivation of length zero with the row's own
trust) or from `pervasion`; an `intuition` pair is provisional and is
not accepted until `infer` has selected it. Credit: the chooser's choice
among `intuit`'s candidates goes through the existing softmax and score
function, so the selection is the backprop chain over the candidates
Alec named.

### 3.7 Trust computed, never learned

Along a path the pair is the minimum of the supports; across alternatives
for one conclusion the maximum; negation is the lane exchange; nothing is
netted, so a conclusion reached for and against keeps both lanes. The
pair is a function of the records and `replay_thoughts` recomputes it.
The chooser learns the path only. The leaves' lanes and pairs are what
their closes wrote: LTM is fixed (Alec, 2026-10-28, §10), no loss or
credit changes a row, and what an answer trains is conceptual space (the
definitions and the taxonomy's membership edges) and the chooser.

### 3.8 Derived conclusions to LTM0

At the close, a kept derived conclusion is written by `ThoughtStream.write`
with evidence kind `inference`, its pair, and `_thought_witnesses` naming
the chain (premise rows, rule rows, in order), so a later thought recalls
it instead of re-deriving; a question chunk left open is written as
`question` as today. The chain is preserved; `Forgetting` values the row
as any other and removes it when unused.

### 3.9 What retires, what stays

The `descend`/`return` discipline of notes §1 is withdrawn (never built);
the record kinds stay for `ask`'s nested contexts. `isImplied` retires
into `infer` forward with the rule fixed (D1). The intuit spec §5's scan
over the "store's leaf-code index" is replaced by the priming surface.
Everything else in the catalogue stays.

## 4. Gates (two rounds, one freeze each; sized to the change)

*Round A, the serial path:* (1) `chunk_truth` preallocated, written by
`deposit`, `query` and the sentence's own ideas with the pairs of 3.1,
cleared with the chunk, pointer constant; (2) `locate` exact on
`chunk_where`, the focus pinned for the episode and evictable after,
`ThoughtRecord.address` a `.where` that replays; (3) `infer` on a fixture
of three rows: forward by identity, membership and instantiation with the
lane rule; present-conclusion raised by max, not duplicated; backward
writes a question chunk at `(0, 0)` with an open referent; `not` then
`infer` equals the contrapositive; an equality rule substitutes; (4)
forcing: at a question the grammar offers the listed operations only; a
provisional `intuition` pair is refused as an answer and accepted after
`infer`; the score function's credit over a two-candidate menu is
nonzero on the chosen one; (5) the close writes the derived row to LTM0
with kind `inference` and the chain, and one unused derived row is
forgotten under the existing forgetting configuration; (6) the worked
line, the item's exit: a question about a token and a property, a
kind-level rule and a membership row in LTM: `query` brings the rule,
backward `infer` asks the membership, `query` fills it, forward `infer`
writes the conclusion with the minimum pair; the records replay the
derivation with its pair; (7) `isImplied` absent, with a certificate.

*Round B, the parallel face:* (8) `intuit` on a three-rule bank: the slot
equals the max over terms of the min of knowing and antecedent, per
lane; NOT before AND; one candidate per fired rule, each an `infer`
request with the firing pair; the knowing write unchanged; (9) the
implicit route on a fixture whose object has parts carrying the property
for and against: both lanes nonzero, the minimum per lane, kind
`pervasion`; agreement with the explicit chain fills one lane,
disagreement both; (10) a conflict `A -> B` and `A -> not B` keeps both
lanes (intuit spec gate 4); (11) an answer loss through `intuit` reaches
the definition edges of a referenced idea and the chooser's parameters
and no lane; the LTM state dict is byte-equal before and after the
backward and the optimizer step (§9, §10); (12) a wrong answer through
one rule moves the definition or membership edges of the concepts on
the path and no row (§10); (13) the candidate's firing pair reaches the
chooser's logit through the request's semantic metadata (§11); (14) the
slot's vector equals the max over terms of the min over each term's
referenced activations, order-0 content only, with an individual's
prepended value as its constituents (§11).

*Both rounds:* the scoped groups of the thought tests
(`test_item6_2_thinking`, `test_thought_*`, `test_bind_thought`,
`test_levelled_thought_history`, `test_normal_thought_controller`,
`test_stm_chunk*`) and the core unchanged sets on one hash, one native
lesson smoke; no seed, no resize, no `RUN_SLOW`; no language named in
code (the worked line's words are test data).

## 5. Decisions for Alec, with recommendations

- **D1. Retire `isImplied` into `infer`.** Recommended: yes; it is
  `infer` forward with the rule fixed, and a capability obtainable by
  calling an existing rule is not a rule.
- **D2. The implicit route.** Narrow the knowing by the Part edges and
  item 2.8's `partOf` order; aggregate by the minimum per lane; write the
  result with kind `pervasion`. Recommended as stated in 3.5.
- **D3. `ThoughtRecord.address` as the chunk's `.where`** rather than a
  physical index, resolved by `locate` on replay. Recommended: yes, it is
  what "reference STM slots with `.where`" means for the history.
- **D4. Two rounds** as §4 divides them, the serial path first.
  Recommended: yes; round A alone already replays the worked line.

## 6. Where

After item 2.7 lands (its final round, 2.7 record §31). Round A touches `STMChunks`, `Thoughts`,
`Queries` (one descriptor, `_write_serial`, `locate`), `ThoughtFaces`
(`_infer`), `ThoughtStream.write`, `ThoughtAnswer`/the closing's
acceptance, the `<thought>` sections of the shipped grammars. Round B
touches `intuit` (new executor as the intuit spec gives it), the knowing
write and the candidate menu. The notes file's §§1–9 are the exploration
and stay as they are.

## 7. Decided (Alec, 2026-10-27): D1–D4 approved as recommended

`isImplied` retires into `infer`; the implicit route as 3.5 states; the
thought record's address is the chunk's `.where`; two rounds, the serial
path first. The round A text below is held until Codex returns from
item 2.7.

## 8. Text for Codex: item 2.5 round A (the serial path)

~~~
Item 2.5 round A: thinking as derivation over chunks with trust, the
serial path. Precondition: item 2.7 is closed and committed at
472f5000850e (2.7 record §§47–48; its landing sweep is November's); start
from that commit with a clean tree. Round A also carries D8 from 2.7 §47:
the answer trial cost becomes the graded binding cost (the meet of the
kept answer idea's content with the supplied word's code) plus the text
cost, so a departure that binds better wins and the chooser is credited;
certificate: two trials with different bindings get different costs and
the better one wins. The spec is
doc/plans/2026-10-27-item2-5-thinking.md §§1–12 (Alec's answers §1,
design §3, gates §4 round A, decisions §5 approved in §7, the 2026-10-28
readings §§9–11, the summary §12); the
exploration is doc/plans/2026-10-26-thought-design-notes.md §§1–10,
which the spec supersedes where they differ. Round B (`intuit`, the
implicit route) is a later text; build nothing of §3.3 or §3.5 now
beyond the hooks named below.

Build, in code names:

1. STMChunks: `chunk_truth [B, C, 2]` float, appended to FIELDS and
   `empty`, cleared by `clear`, written by `deposit(..., truth=)`,
   never resized. Pairs: a row pulled by `query` or `ask` carries the
   row's evidence pair or lanes; the registered sentence's own ideas
   carry (1, 0); a derived idea carries its derivation's pair; a
   question chunk carries (0, 0). `STMChunks.locate(state, where)
   -> index` by exact match on `chunk_where`, -1 when absent.
2. Addressing: every thought executor that reads or writes a chunk
   resolves it through `locate`; `ThoughtRecord.address` becomes the
   chunk's `.where` (four floats) and `replay_thoughts` resolves it the
   same way; `Mind.SERIAL` stays the subsystem name, and its write is
   `_write_serial(context, value)` into the focus chunk.
   `begin_thought_episode` deposits the root with `stack=True` and
   records `focus_where`; the focus is pinned while the episode is open
   and evictable after.
3. `infer`: the descriptor of spec §3.4, the grammar rule
   `infer_argout1 = infer.thought(infer_argin1, infer_argin2)` with the
   permutation family for direction (2, 0), in every shipped grammar's
   `<thought>` section. Forward: match by identity (`isEqual`),
   membership (the Part edges through `Priming.part_edges`) or
   instantiation, recorded in `reason`; write S2 instantiated into the
   focus with pair (min(premise+, rule+), min(premise+, rule-)),
   evidence kind `inference`, sources (premise .where, rule row); if a
   chunk with the same role_refs exists, raise its pair by max and add
   nothing. Backward: write S1 instantiated as a question chunk
   (`_open_references=(('referent', 0),)`, pair (0, 0), evidence kind
   `subgoal`). `not` on an operand before `infer` is the contrapositive;
   an `isEqual` row as the rule substitutes. `isImplied` retires into
   `infer` with a certificate test asserting absence; its descriptor,
   grammar rules and `ThoughtFaces.is_implied` go; its tests port to
   `infer` with before/after text.
4. Forcing: the `<thought>` sections at a question offer `ask`,
   `query`, `intuit` (descriptor registered now, executor deferred to
   round B and refused with a clear error if selected), `infer`, `not`,
   `isTrue`, `bind`, `exist` and conclude; the answer path accepts as
   the answer only focus content whose pair came from `infer`, recall
   (`query`, `isTrue`) or kind `pervasion`; an `intuition` pair is
   refused until `infer` selects it. The chooser's choice stays in the
   existing softmax and score function.
5. The close: `ThoughtStream.write` writes a kept derived conclusion
   with evidence kind `inference`, its pair and `_thought_witnesses`
   naming the chain in order; forgetting treats the row as any other.
6. Trust is computed (spec §3.7): min along the path, max across
   alternatives, lane exchange for `not`, never netted; the pair is
   recomputable from the records.
7. Documents, by append or exact replacement: Architecture's thought
   section names `chunk_truth`, `.where` addressing and `infer`; the
   intuit spec §5 gets one appended line pointing to spec §3.3 for the
   candidate menu; Params.md if any knob is added (none is expected).

Gates, one frozen hash: spec §4 round A items 1–7, including the worked
line (6): a question about a token and a property, a kind-level rule and
a membership row in LTM; `query` brings the rule, backward `infer` asks
the membership, `query` fills it, forward `infer` writes the conclusion
with the minimum pair, and the records replay the derivation with its
pair. Then the thought groups (`test_item6_2_thinking`, `test_thought_*`,
`test_bind_thought`, `test_levelled_thought_history`,
`test_normal_thought_controller`, `test_stm_chunk*`) and the core
unchanged sets on the same hash, one short native lesson smoke before
the freeze. Receipt under doc/benchmarks/2026-10-2X-item2-5-round-a/ with
the exact commands, frozen-source.json, every port's before/after text,
the retirement certificate and the allocation read. No full sweep, no
RUN_SLOW, no seed pinning, no resize, nothing retired beyond `isImplied`,
no language named in code (the worked line's words are test data), fail
loud on construction errors. Return through Alec for Claude's review;
no commit or publication.
~~~

## 9. How the DNF in `intuit` is trained (Alec's question, 2026-10-28; Claude's reading for round B)

Alec: if `intuit` is called to answer a sentence and fills an STM slot
with ideas and relations, can those be learned within the slot, or
should we train a sparse subnetwork of a dense Sigma and Pi layer that
corresponds to the relations and ideas copied into the slot?

Reading: the second. A slot has no parameters: `STMChunks` buffers are
working memory, updated functionally and cleared at the close, and what
a slot holds is a reference (`chunk_refs`) with a cached working value.
Learning "within the slot" would train a copy that is gone at the next
sentence. What persists is what the references point at: the ideas'
concepts, whose definitions are the sparse Sigma and Pi edges of the
concept layer (`ConceptualAttentionLayer.features.values`, the
`conjunctive` values, the percept and property codebooks beneath them),
and the relations' rows in LTM1, whose two lanes are evidence. So the
DNF is evaluated through live reads of the referenced rows, never through
detached copies: a term is the meet of the referenced ideas' current
activations, the slot's vector is the union over terms, and the answer's
loss flows back through max and min along the active branches into
exactly the edges those references select, a sparse subnetwork of the
dense layer, and through the score function into the chooser's choice of
`intuit`, `infer` and the premises. This is the intuit spec §5's
"gradient" paragraph made concrete for the slot, and the familiar tier
already reads candidates this way (`read_candidates`, "a differentiable,
sparse read through the current member codes").

Two things the gradient does not train, by the two truths: the lanes on
the relations and the pairs on the ideas, which are evidence accumulated
at closes and computed along paths (§3.7); and the structure of the
DNF, which is the record of what fired, not a weight matrix over the
slot's ideas. A dense relation matrix over the sentence's ideas would be
the pairwise structure the priming design declined; the rules stay
memory rows and the firing degree stays the meet of content. What the
gradient trains is what the ideas mean; what evidence trains is how true
the relations are.

For round B this fixes the implementation: `intuit`'s executor builds
the slot's vector from `lookup_rows`/the MTM reads of its references
with gradient enabled in training, deposits the detached value and the
references into the chunk, and records the firing; a certificate asserts
that an answer loss reaches the definition edges of a referenced idea
and the chooser, and reaches no lane.

## 10. What makes `intuit` give good answers (Alec's question, 2026-10-28; Claude's reading)

`intuit` itself has no parameters and is not trained: it is the exact
max-min calculus over memory (intuit spec §5), deterministic, the same
forward in eager and compiled form. Its answers get better because four
things around it learn, each by its own owner, and none by a weight
inside the operation:

1. *Memory: which rules exist and how true they are.* Rules are LTM1
   rows written at kept closes (an observed succession `S1 -> S2`, a
   stated if-then, an IS row) with lanes from the closing's evidence,
   and derived conclusions written back at the close (§3.8), so a
   conclusion reached once is recalled next time. LTM is fixed (Alec,
   2026-10-28): no answer, loss or credit changes a row or its lanes;
   a row's lanes are what its closes gave it, and a contrary statement
   observed later adds its own witness through the ordinary close. A
   wrong rule is not corrected in memory; it stops mattering because the
   concepts and taxonomy it is stated over learn (2), the chooser learns
   not to take it (4), and forgetting removes it when it goes unused.
   Derived conclusions written at the close are memory writes, records
   of what was derived, not training.
2. *Meaning: how well an antecedent matches the knowing.* The firing
   degree is the meet of the knowing with the antecedent's content, and
   that content is read through the concepts' definitions, the sparse
   Sigma and Pi edges of §9. The answer's loss reaches those edges
   through the live read, so the ideas the rules are stated over come
   to mean what makes the right rules fire and the wrong ones not. This
   is the one gradient path, and it is the same path that trains
   definitions from supervised sentences today. The taxonomy is in the
   same layer: a rule stated over a kind fires for a member through the
   membership edge, and that edge's weight is a parameter of conceptual
   space (the sigma column `cover_members` writes), so a rule that
   misfires for a member it should not cover is corrected by the
   membership weight falling, while the LTM1 membership row that
   recorded the statement stays as it was. Alec, 2026-10-28: "we have to
   train the taxonomy and concepts from conceptual space, which has the
   weights as parameters already." That is the whole of what the answer
   trains besides the chooser.
3. *Place: which rules are considered.* The rules `intuit` scans are the
   LTM1 rows whose symbols carry excess on the priming surface, placed
   by item 2.9's location diffusion over item 3.0's placement; placement
   by context at closes moves the right rules into reach of the right
   knowing, within the work budget.
4. *Choice: when to call it and what to do with it.* The chooser decides
   whether to call `intuit`, which candidate `infer` selects from the
   menu it produces, which premises are pinned, and when to conclude;
   it learns by the score function's credit from the answer comparison,
   the backprop chain over the candidates Alec named (§3.6). What it
   learns is the path, never the truth of a step.

So the honest answer is that `intuit` is trained the way a lookup is
trained: by improving the table (1), the keys (2), the index (3) and
the caller (4). A weight inside the calculus, a learned reliability per
rule or a learned firing threshold, would duplicate the lanes or break
the exactness and determinism the intuit spec's gates require, and is
not proposed. Round B adds to its gates: a wrong answer through one rule
moves the definition or membership edges of the concepts on the path and
no LTM row or lane; a right answer moves no row or lane either; the
definition edges of a referenced idea receive gradient and the lanes
none; the LTM state dict is byte-equal before and after an answer's
backward and optimizer step.

*Superseded in part (2026-10-30, §22.3): `intuit` has its own learned
Sigma-over-Pi weights, trained only by its use in output; items 1-4 stand,
and round B's gates are read as §23 states them.*

## 11. Order zero to order zero, and the conscious read (Alec's question, 2026-10-28; confirmed reading)

Yes on both counts, with two precisions.

*The layer is order zero to order zero by construction.* The knowing
field is the order-0 cube (intuit spec §5: event types, overlap and
containment, no identity); a rule row's two occurrences have order-0
content, the firing degree is the coverage of the antecedent's content
by the knowing, and the consequent's content is unioned back into the
knowing. Higher orders enter only as their zeroth-order constituents in
the same cube: an individual by the parts and properties bound to it
(its prepended MTM value is in the field), a kind by the content its
members share, a rule stated over a kind by the kind's activation. That
is the parallel of Orders §8 and of the priming spec §8, and it is why
`intuit` "stops where individuation begins": it cannot tell two
individuals apart except by their order-0 differences. Telling them
apart is the serial grammar's work.

*The conscious read pulls the most significant for a particular
activation.* The slot's union is the intuition; the candidate menu
beside it (§3.3) is what the serial path reads. "Most significant for
some particular activation" is the focus chunk's content, the question's
subject or open slot, pinned in STM: `infer` selects from the menu by
the chooser's softmax over candidates whose request carries the firing
pair, `query`'s ranking against the focus reads the same knowing, and
the selected term is written into the focus with its pair, which is the
strong forcing of §3.6. Significance is not computed in the layer; it is
learned at the choice by credit (§10 item 4), so the breadth of the
parallel pass and the narrowness of the serial pick are two different
learners: the first exact and unlearned, the second the chooser.

**The pick is affine, not a chart (Alec's question, 2026-10-28).** The
tanh and cube charts belong to Sigma and Pi because a combination must
stay monotone and bounded inside the cube: it is logic. The pick of the
most significant term is not a combination but a choice, and a choice
needs a logit, unbounded, with the softmax turning logits into the
probabilities the score function credits. The code already has this
split: the grammar's operation chooser scores a candidate by the inner
product of its output with a per-rule learnable anchor
(`AnchorDotTransformChooser`), and the thought chooser
(`SelectedThoughtChooser.logits`) scores each candidate's complete
semantic context, the masked root and active roles, the candidate
meaning, the bounded semantic metadata and the visible memory, through a
small MLP (`Linear`, `GELU`, `Linear`, last layer zero at construction)
to one logit, then `log_softmax` and a hard index. So "most significant
for a particular activation" is the MLP over the focus and the candidate
together, affine in each with one nonlinearity between, and no chart.
Round B supplies the candidate's firing pair through the request's
semantic metadata so the readout can weigh degree against relevance; a
gate asserts the pair reaches the logit. A parameter-free alternative,
ranking by the meet ratio of candidate and focus, would be cube-valued
and need no affine map, but it could not learn what counts as
significant; the chooser is kept.

## 12. State of this spec (2026-10-28), for the record

- *Decided by Alec:* the eight answers of §1 (2026-10-27), D1–D4 of §5
  (2026-10-27, §7), and on 2026-10-28: the DNF is trained through the
  sparse subnetwork its references select, never within the slot (§9);
  LTM is fixed and the answer trains conceptual space and the chooser
  (§10); the layer is order zero to order zero and the conscious read
  pulls the most significant candidate for the focus (§11); the pick is
  an affine readout, not a chart (§11).
- *Sequence:* item 2.7's final round (2.7 record §31) lands first; then
  round A (§8's text, the serial path: chunk trust, `.where` addressing,
  `infer`, forcing, derived rows to LTM0, `isImplied` retired); then
  round B (`intuit` with the candidate menu and live reads, the implicit
  route, gates 8–14 of §4), whose Codex text is written after round A's
  review.
- *What round B must also carry from the 2.7 decisions:* the field
  configurations keep the parallel flag until `intuit` can reach S for
  a grammar with no serial options (2.7 record §22), after which the
  flag retires; candidates are MTM entries (2.7 record §25), so the
  knowing `intuit` reads includes their prepended values.
- *Decided later the same day (§15):* at round B the open-pass `intuit`
  over the learned manifold replaces location diffusion as the spreading
  mechanism and consolidates conceptual expectation and priming into one
  operator; 2.9's step runs as built until round B lands; round B
  measures the two on one hash.
- *Exit of the item:* the spec's gates green on one hash and one thought
  episode's records replaying a derivation with its pair (todo item
  2.5).
- *Updated 2026-10-30:* round A landed (§21, `3718d9d0dc`); §22's decisions
  govern round B and the one-backward round, one Codex text (§23).

## 13. `intuit` and the priming surface (Alec's question, 2026-10-28; Claude's reading)

Two surfaces over the same concepts, of different kinds, coupled at
three points.

*What each is.* The priming surface (`_priming_boosts`, item 2.9) is
energy: one value per concept, the excess over neutral, bumped by what
is seen and desired, spread by location in conceptual space (nearness of
meaning, `Priming.propagate`), conserved and decaying; it carries no
truth and no logic, and it is read as a bias: `relevance_weights` into
attention, the chooser and retrieval, and the filter that selects the
rules `intuit` scans. The knowing (`_concept_activations`, the order-0
field) is truth: two lanes per concept in the cube, held about the
sentence and the STM entries, composed by `intuit` along stated rules
with AND, OR and NOT into a new knowing, which answers compose from. One
is where attention is likely to go; the other is what is held true and
false. Item 2.9 §8 D1 drew the line on purpose: relations conduct no
energy (the taxonomy edges place rows, they do not prime), and location
conducts no truth (nearness in meaning asserts nothing). `intuit`
conducts truth along relations; priming conducts energy along location.

*Where they touch.*
1. *Selection.* The rules `intuit` scans are the LTM1 rows whose symbols
   carry excess on the priming surface (spec §3.3). Priming is the index
   of the rule table, not a term of the inference: a primed rule fires
   only to the degree the knowing covers its antecedent, and an unprimed
   rule does not fire at all within the budget. This is the "top-k": it
   is priming's, by energy, over rules; `intuit`'s own pick (§11) is the
   chooser's, by significance, over fired candidates.
2. *Return.* The rows a forward activates, including the consequents
   `intuit` fired, are bumped by `_prime_seen_step` after the forward
   (2.9 §8 D3, no propagation) and spread at the next sentence's open
   pass. So what intuition concluded primes what the next sentence
   attends to. The loop knowing to priming to rule selection to knowing
   is bounded by decay, the conserved budget and the work meter, and the
   placement guard holds: codes read only the seen ledger
   (`_priming_seen`), never energy that `intuit` produced.
3. *Focus.* Attention, biased by priming, decides which STM entries and
   which object's mereology form the knowing `intuit` reads (§3.5's
   narrowing); priming does not enter the knowing's values.

*Not overlapping.* Priming never ANDs, ORs or negates; it has one lane
and no evidence. `intuit` never spreads by nearness; a rule between two
concepts fires whether or not they are near, and two near concepts
without a rule exchange no truth. The retired taxonomy-edge priming of
step 2 was the one place the two had merged, and 2.9 separated them.

## 14. If `intuit` replaced the priming surface (Alec's question, 2026-10-28; Claude's analysis, with a recommendation)

The proposal: no priming surface; concepts are activated by the
percepts, then augmented by one `intuit` step, and that augmented
knowing is both the activation and the diffusion, read by attention,
the chooser and retrieval.

*What would change.*

1. *Channel.* Spread by stated relations instead of by nearness of
   meaning. Gained: spread that respects truth, two lanes, no energy
   derived from codes, so the placement guard is trivially true and the
   radius controller and the `geometric*` knobs go. Lost: spread to
   concepts that are near in meaning but have no rule with the active
   ones, which is how a synonym or an unstated association is reached
   today; a rule over a kind reaches its members only once the taxonomy
   has learned the membership, so generalization by similarity waits on
   the taxonomy where today it is immediate from placement.
2. *Persistence.* The priming surface decays at 0.9 per sentence and so
   carries roughly ten sentences of discourse context as a graded trace.
   The knowing is recomputed each sentence from what STM and the
   prepended MTM entries hold, so context across sentences would be
   whatever working memory retains, eight chunks and the MTM horizon,
   not a graded trace over all concepts. More principled, since context
   would be content held rather than energy left behind, but a loss of
   the cheap topical trace unless the knowing itself were given a decay,
   which would reintroduce a surface.
3. *Stability.* Priming is conservative; `intuit` is a union, bounded
   by the cube and idempotent, so repeated steps grow the knowing
   monotonically toward the deductive closure within the budget.
   Nothing blows up, but nothing sharpens either: a step cannot make a
   concept less active, and the sparsity priming keeps by conservation
   and decay would have to come from the budget and the lanes' minima.
4. *Cost and the index.* Priming's step is one `[B, V] x [V, P]` product
   with `V` fixed. `intuit`'s step scans rules, and the rule store grows
   with the corpus, since every kept close writes a succession row, so an
   unindexed scan is `O(R)` per sentence at a million rows. Today the
   priming surface is `intuit`'s index (§13). Without it, the index would
   be the recurrence bank: `recurrence_codes [recurrenceCapacity, width]`
   already holds each reading group's content, and the rules of the
   matched seats are the ones to fire, a fixed-shape bank product per
   sentence that compiles. That is feasible, but it is a content index,
   and it replaces one `[V, P]` product with one `[seats, width]` product
   at 262,144 seats against 1,048,576 concepts by 448 pairs; comparable,
   not free.
5. *Desire.* `prime_desire`'s signed valence is a goal bias, not a truth,
   and has no home in a knowing; it would stay as its own channel or
   become an open slot's expectation. The seen ledger `_priming_seen`
   also stays, because placement's objective reads the registered
   sentence from it; so the "priming surface" would reduce to the ledger
   plus the knowing, not vanish.

*Recommendation.* Keep both kinds and do what the parenthetical says in
the open pass: concepts are activated by the percepts and the location
step (energy, cheap, similarity, the discourse trace), and then
augmented by one unconditional `intuit` step in reading mode (truth
along relations), so the knowing that composition, attention and the
chooser read at the sentence's start already carries intuition's
consequents. That is "always intuit when reading", and it adds nothing
the design does not already have: §3.3's layer run once at the open pass
with the priming surface as its index, its consequents into the knowing
and their rows bumped for the next sentence (§13 item 2). Attention's
relevance then reads the union of priming's excess and the knowing's
salience (the maximum over lanes). Whether that augmentation helps is
measurable in round B with the supplied-answer lessons: attention and
answers with and without the open-pass `intuit`, on one hash. Replacing
priming outright is not recommended now, for items 1, 2 and 4; it can be
revisited when the taxonomy carries most of the generalization and the
recurrence bank is the index, which the million-row run will show.

## 15. Diffusion over the learned logical manifold (Alec, 2026-10-28; Claude's revised reading, with a decision)

Alec: replacing the priming surface with the `intuit` step "would replace
a diffusion to contextually related concepts with a diffusion to
contextually related concepts over a learned logical manifold", strictly
superior in that it learns; and if the resulting activation structures
placement, space continues to be structured by context, over that
manifold.

Revised reading. §14 weighed `intuit` as a scan over LTM1 rows; read as
a step over the learned manifold it is a different thing, and Alec's
view holds.

- *The manifold is the definition layer.* The sparse Sigma and Pi edges
  of conceptual space (parts and wholes, the taxonomy's memberships, the
  conjunctive literals) are a fixed-size parametric structure over the
  inventory, learned by backprop through the sparse subnetwork (§9) and
  initialized at every raise (2.7 record §26.3). One `intuit` step over
  it is `cs_forward_content`'s own layer applied once more with the
  knowing fed back: activations reach their wholes and kinds and their
  parts, two lanes, max over OR and min over AND, which is the implicit
  route's pervasion (§3.5) run over the whole field. That is contextual
  spread, because concepts that share wholes are concepts that occur in
  the same contexts, and it is learned, because the edges are trained by
  what answers reward. Location diffusion spreads by placement's
  geometry, which learns positions by a fixed local rule and never from
  an answer.
- *Priming becomes expectation.* A concept activated by the step is an
  expected concept: the step over successions and definitions is the
  concept-level predictor of 2.7 record §13 ("we will expect concepts
  also, in the same way"). The priming surface was a stand-in for that
  predictor before `intuit` existed; with the step at the open pass,
  what primes attention, the chooser and retrieval is the expected
  knowing, and the residual against the observed field is the learning
  signal §13 names.
- *Placement over the manifold.* If placement's context is the knowing
  after the step rather than the registered sentence alone, space is
  structured by context and its implications. The guard holds without a
  ledger: no placement code enters the step (the knowing comes from
  percepts through definitions, the rules are memory), so codes never
  read energy computed from codes, and `_priming_seen` is no longer
  needed as a separate ledger.
- *What location keeps.* `bind`'s content term, the derived class
  centroids (2.9 §8), MTM's candidate matching (2.7 §26.1), the
  reconstruction bank's selection and the tower projection read
  positions and stay; only the location *diffusion* step and its
  controller and knobs retire, with the `_priming_boosts` surface
  replaced by the expected knowing or retired with it.
- *What §14 still holds.* Item 2, the graded trace: context across
  sentences is then what STM, MTM and the successions carry, and the
  ten-sentence energy trace is gone; a measurable change, not a known
  loss. Item 5, desire: a goal bias stays its own channel. Item 4
  dissolves: a step over the manifold is a fixed-size sparse product
  per sentence, compiled as the definition layer already is; the scan
  over LTM1 rows remains the thought-time `intuit` of §3.3, chosen and
  budgeted, with the expected knowing as its index.

*Decided (Alec, 2026-10-28: "this sounds correct, and it seems to
consolidate conceptual expecting and priming into the `intuit()`
operator, which is a good thing on several levels").* At 2.5 round B the open-pass `intuit`
over the learned manifold becomes the spreading mechanism: priming is
the expected knowing; placement reads it as its context; location keeps
the roles above and the location diffusion step retires with its
controller, knobs and seen ledger. Until round B lands, item 2.9's step
runs as built, so item 2.7 is not touched. Round B's measurement:
attention, answers and placement with the manifold step against the
location step, on one hash, with the supplied-answer lessons.
Consolidation: conceptual expectation and priming are one operator,
`intuit`; percept-level expectation (the level-by-level predictor over
percepts, `Expectation.gain_field`, `ActivationPredictor`, the
recombination of 2.7 record §13) stays with item 2.7 and is not
duplicated at the concept level there.

**Round B also carries the residual split (2026-10-29; item 2.7 record §42).**
With the open-pass `intuit` as the conceptual layer's local expectation, the
completed understanding's readback into the order-0 concept field is trained
on `observed - stop_gradient(expected)` per lane, conditioned on the detached
expectation, using the frozen pre-arrival value; no top-down gradient reaches
`intuit`'s inputs through it.

*Superseded in part (2026-10-30, §22.3): the spreading mechanism at round B
is `ActivationPredictor`'s predicted next state, not an open-pass `intuit`;
the location step still retires after the measurement; the residual split is
not built in round B (D7 to future work, §22.2).*

## 16. One error signal, one backward (Alec, 2026-10-29; decided direction)

Alec's proposal, agreed: (1) stop-gradient for attention and expectation;
(2) jitter of reconstruction's strength against output; (3) the chooser's
log-probabilities summed into the same objective, so the chooser and the
signal it chooses share one gradient. One error signal and one backward
replace the per-owner backward of item 6.9 (`ObjectiveOwnership.backward_owned`,
one `torch.autograd.grad` per objective). The full gradient picture, with
the alternatives that led here, is the read-first section of
`doc/GradientFlow.md`.

*The objective.* One surrogate, differentiated once:

```python
loss = (s * reconstruction            # s: jittered strength, below
        + output                      # now reaches the understanding
        + expectation_local           # trains only expectation
        + attention_terms             # train only attention
        + sum(detach(cost - baseline) * log_prob(choice)
              for choice in the trial's choices))
loss.backward()                       # one call, one optimizer
```

1. *Stop-gradient bands for expectation and attention.* Wherever either
   reads another owner's value, the read is detached in the forward, so in
   the single backward their parameters receive exactly their own terms'
   gradients. Expectation keeps item 2.7 record §42's residual split with
   the value frozen at arrival; expectation doing all the work is welcome
   and gets no jitter. LTM is fixed and receives nothing (§10).
2. *Reconstruction against output, by jitter of strength.* Output's terms
   now reach the understanding's parameters (the grammar operators and
   inverses, the conceptual definitions), which therefore have two writers:
   reconstruction and output. Reconstruction's term is scaled per step by a
   random strength `s`; while it is weak, output moves the understanding
   unopposed, and on average this is a weaker reconstruction plus
   exploration. Maps with a certified exact inverse keep reconstruction as
   an audit (item 2.7 §38.3) and are unaffected. The target is never
   jittered, only the strength.
3. *One gradient for chooser and signal.* The paired trials stay: the
   baseline for each choice is the other trial's cost, as
   `_compose_score_function_loss` computes now. The change is that the
   chooser reads live features, not detached ones, so its credit reaches
   the meaning representation that made a choice easy or hard, and the
   meanings learn in part to make good parses likely. The score-function
   term is noisy and enters shared coordinates under Adam's per-coordinate
   scaling; it carries its own scale, and its variance is measured before
   the scale is raised.

*What stays.* The two trials per sentence, the work budgets, the choosers'
legal masks, the frozen-arrival expectation, the exact-map audits, and the
rule that a predictor never trains its own target.

*Certificates.* (a) One backward call per training step (counted), with
the perception pullback in the same graph or named as the only second
call. (b) Expectation's and attention's parameters receive gradients equal
to their per-owner computation on one sentence. (c) The understanding's
parameters receive reconstruction (scaled by the drawn `s`), output and
chooser terms, each separately attributable on a fixture. (d) The chooser
term's gradient reaches a semantic parameter on a fixture, and its
variance is reported per parameter group. (e) Step time against the
per-owner backward on the same sentences. (f) A lesson with the supplied
answers, against the same lesson under the per-owner backward.

*Details for Codex to propose, Alec to decide.* The jitter's distribution
and mean (a per-step uniform multiplier in `[1 - j, 1]` is the simplest
form); which choosers read live features first (the compose chooser, then
thought and generate, is the cautious order); the chooser term's initial
scale; and the round's place in item 2.5's sequence (after rounds A and B
is the conservative placement; earlier if the step-time gain is wanted
for the long run).

### 16.1 Re-check (Claude, 2026-10-29, same day): three corrections and a wording fix

1. *Stop the band at its output, not only at its input.* Item 1 above
   says the band's reads are detached. That is half of it. Under the
   per-owner backward it did not matter that attention's mask multiplies
   the content live, or that expectation's gain scaled it live, because
   reconstruction's gradient was never taken with respect to their
   parameters. Under one backward it does matter: a live mask carries the
   task gradient into attention, and a live gain carries it into
   expectation. So every consumer detaches the band's output at the point
   of use (`attention.detach() * signal`; `expected.detach()` in the
   recombination and in every reader of the forecast, as item 2.7 record
   §38.3 ports them), and each band learns only from its own term on its
   own detached inputs. "Removing a band from the signal and stopping its
   gradient" means exactly this. Certificate (b) catches a missed consumer.
2. *One backward, but not one optimizer step for everything.* Percept
   expectation updates inside the processing loop at each arrival, with
   its own optimizer state (item 2.7 record §39, §40 A2). That stays: it is
   a separate, more frequent step by design. "One optimizer" in §16 means
   one for the sentence-level objective; the in-loop update is excluded
   from it and from the single backward, which sees expectation's
   contribution only as detached values.
3. *The conflict item 6.9 removed returns in weighted form, so measure it.*
   With output and the chooser's credit reaching the understanding, the
   grammar operators and conceptual definitions have three writers summed
   before Adam's per-coordinate scaling, which is where 6.9 saw the XOR
   codes drift. The jitter re-weights that conflict; it does not remove
   it. Add to the certificates: the XOR standing gate and the packing
   parity gate pass under the new backward, and the per-operator gradient
   cosine between reconstruction and the other writers (the dissonance of
   2026-09-20) is reported per epoch of the lesson in (f).
4. *Wording.* The chooser term is the existing surrogate of
   `_compose_score_function_loss`, the departure's probability with its
   sampled-proposal correction times the detached paired advantage, not a
   bare `log_prob`; the objective line stands as its shape, the code as its
   form.

The direction is unchanged; these make the single backward give each
parameter what the per-owner backward gave it, except where the design
now adds a writer on purpose.

### 16.2 One optimizer step, compiled; the separate chooser as a model variable (Alec, 2026-10-29; Claude's structure)

*One step.* Alec wants a single optimizer step per sentence, so the loop
compiles. Structure: percept expectation's arrival-to-arrival adaptation
is carried by *state*, not parameters: at each arrival the moments and the
local context (`PerceptMoments`, the predictor's context carry) update
without gradient as loop carries, and the arrival's local loss, the
pre-arrival forecast against the detached arrival, is accumulated into a
carry. The sentence's one backward differentiates that accumulated loss
with the rest of the objective, and the one optimizer step applies it.
"The next arrival sees the update" then holds for the state; the
parameters move once per step. This replaces §39's per-arrival parameter
update and its private optimizer (item 2.7 record §39, §40 A2), which
were the eager form; the certificate becomes: the forecast at arrival `k`
reads the state updated at arrival `k - 1`, every arrival's loss is in the
step's backward exactly once, and the compiled brick equals the eager
loop. Expectation's parameters remain its own by the detached reads.

*The separate chooser as a model variable.* `<separateChooser>` in the
architecture section, default `true`: `true` is today's per-owner
backward with detached chooser features; `false` is the single backward
with live chooser features. `<reconstructionJitter>` (default 0) is a
separate knob, the half-width of the per-step uniform strength `s in
[1 - j, 1]`, so the leak can be measured with and without the chooser
change. The round's measurement runs the same lesson under both settings
of each and reports step time, the lesson's answers and objects, the XOR
and packing gates, and the per-operator dissonance; the comparison, not
the design, decides the default.


## 17. Codex round A implementation and review receipt (2026-10-29)

The implementation starts at committed BasicModel `0aedc06414`. The
predecessor's `472f5000850e` names its source snapshot, not a Git commit.
No commit or publication has been made. Round B and the one-backward
round remain outside this change.

The serial path now keeps a fixed `chunk_truth` owner, exact `.where`
lookup, and a pinned episode focus. Recalled premises are retained in
addressed STM slots. `infer` handles forward and backward directions,
equality, membership through the existing priming cache, substitution
into relation conclusions, and operand negation. An unmatched inference
leaves the open goal unchanged. Derived alternatives record their input
pairs and combine by maximum without allocating another conclusion.
The worked line replays with pair `(0.7, 0.3)` and ordered membership and
rule witnesses; tampered support is rejected. Kept derived conclusions
use the ordinary LTM0 writer and forgetting policy.

The answer boundary accepts the specified provenance at both ordinary
closing and answer resolution. Conceptual presence, equality, and
prediction still execute and retain their typed trace; they do not
supply an answer without accepted evidence. The corresponding adapter
and controller tests are ported, with exact before/after files in the
receipt. D8 uses detached content coverage plus text cost: the production
paired-walk certificate measures costs approximately `1.8` and `1.2`,
keeps the second binding, and produces chooser credit.

The receipt is
[`2026-10-29-item2-5-round-a`](../benchmarks/2026-10-29-item2-5-round-a/).
`protocol/checks.py` records every gate command, environment, and source
snapshot. `frozen-source.json` uses the compact canonical JSON hash of
its inline `source_snapshot`; `ports.json` contains every changed test
file before and after; `retirement.json` scopes the executable certificate
to Python implementation and shipped grammars, excluding the archived
original grammar fixture. `allocation-read.json` records construction,
slot ownership, trial restoration, and the history-only compatibility
boundary. Earlier development failures and the superseded candidate are
retained under `development/`.

The core gate also exposed an STM trial-restoration defect: restoring the
full construction changed the active stream count to its preallocated
maximum. A one-stream query consequently met a larger symbolic query in
attention. `ThoughtTrialState` now restores both the active stream count
and the stack-depth summary, while preserving the full fixed buffers.
A direct three-stream-construction/one-stream-use regression covers this
producer. The remaining old binding-cost assertions are ported to D8's
content coverage and additive text rule.

One policy question is outstanding for Alec. Sections 3.6 and 8 require
`intuit` in the ordinary menu, while its round A executor must raise a
clear error if selected. The implementation follows both requirements;
a development regression sampled that action and raised the prescribed
`NotImplementedError` (`development/thought-ports.log`). Codex asked
whether to mask ordinary selection until round B, keeping the descriptor,
grammar and direct-call error. No answer has yet been received, and no
mask has been introduced. A green finite run would not remove this
known reachable error.

Final frozen results follow.
No full sweep, `RUN_SLOW`, seed pinning, capacity change, or new model
knob is part of this round.


### 17.1 Frozen result and remaining menu decision (2026-10-29)

Source SHA-256: `2515d19ee75ea69f0b9fe98488ef0d5588058fe3445568342a3d18d58b33f89d`
(959 snapshot files). The compact canonical hash was recomputed after
all checks and the tree still matches. The native smoke and both gate
groups use this source. Exact commands and environments are in
`commands.json`; detailed counts, skips and the complete failure are in
`summary.json`. All eleven changed test files have before/after text.

- Thought group: 254 passed, 4 skipped, no failures; all 13 new round A
  certificates pass, including the worked-line replay, D8's kept
  departure, and restoration of the active stream count.
- Core unchanged selection: 1,644 passed, 36 skipped, 1 failed across
  all 1,681 selected cases. The first two-worker invocation stopped at
  its 8 GiB aggregate memory limit after 672 completed cases. Its
  successful results were retained. The continuation selected exactly
  the 1,009 unfinished cases, at 16 GiB aggregate on this 36 GiB machine;
  it did not rerun completed cases. Both parts' commands, source and
  resource receipts are retained. The result is a complete scoped
  measurement with one failed case, not a green core gate.
- Native smoke: four sentences, exit 0, approximately
  39.626 seconds, 11 native forms and zero rung-zero audit errors.
  Both supplied-answer checks remain incorrect and acquisition is false.
  This certifies execution only.

The sole frozen failure is
`test_math_chain.py::test_native_worked_document_trains_unforced_without_numeric_oracles`:
the ordinary chooser selected `intuit`, whose executor raised the clear
round-B error required by §8. The earlier attention-shape exception is
repaired; the native run progressed beyond it. No failing assertion or
exception was suppressed, and no seed was chosen to avoid the menu item.

Round A is implemented and frozen for review, with approval held on the
menu decision described in §17. Codex recommends masking ordinary
`intuit` selection until round B while retaining its registered grammar,
descriptor and direct-call error. That is a change to §8 and remains for
Alec to decide; it has not been implemented. No commit or publication.


## 18. Claude's review of round A (`2515d19ee75e`) and D9 (2026-10-29)

**Verified.** The hash recomputes from the inline snapshot and all 959
files equal the live tree (new: `ThoughtChunks`, `ThoughtInference`,
`test_thought_infer`). Every gate run (thought, core in two parts, the
smoke) is on that hash with `RUN_SLOW` and `BASIC_SEED` unset; the core's
second part selected exactly the 1,009 cases the first left at its memory
stop, so together they cover all 1,681. All eleven ports carry before text
equal to HEAD and after text equal to the tree. The retirement certificate
finds no hit in the 154 source paths or the shipped grammars (the archived
fixture grammar excluded, correctly). No seed, no resize, no knob:
`chunk_truth` is preallocated with a constant pointer, and the sidecar
`_thought_contents` is constructed per stream, saved and restored with the
trial, and read only after `locate` confirms the address. The
trial-restore defect Codex found (the active stream count) is repaired at
its producer with a regression.

**The exit holds.** The worked line runs `query`, backward `infer`,
`query`, forward `infer`; the conclusion's pair is
$(\min(.7,.9), \min(.7,.3)) = (.7, .3)$ with the membership and rule rows
as ordered witnesses; the records replay with that pair, the derived row
is in LTM0 with kind `inference`, and a tampered support is rejected. D8
grades: two wrong-text trials with different bindings cost about 1.8 and
1.2, the better is kept, and the chooser's surrogate carries credit.

**Results.** Thought 254 passed, 4 skipped; core 1,644 passed, 36
skipped, 1 failed; smoke four sentences, exit 0. The failure is D9.

**Repairs.**

1. *D9, decided (Alec, 2026-10-29): `intuit` leaves the ordinary menu
   until round B.* The legal mask excludes it while its executor is
   deferred; the descriptor, the grammar rule and the direct-call error
   stay; round B enables it with the executor. §8 asked for both "offered"
   and "refused if selected", which a sampling chooser cannot honor; the
   defect was the spec's.
2. *The match's support joins the path minimum (§3.7).* `derive` records a
   membership or equality proof's witnesses but not its support, so the
   pair is $(\min(p^+, r^+), \min(p^+, r^-))$ whatever the edge's
   strength. The proof is a step on the path: the pair becomes
   $(\min(p^+, m^+, r^+), \min(p^+, m^+, r^-))$, with $m^+ = 1$ for a
   direct identity, recorded in `derivation` and recomputed by
   `replay_pair`. Certificate: an edge weaker than premise and rule caps
   both lanes at the edge.
3. *A duplicate occupied `.where` fails loud.* `locate` returns the first
   match silently. In `ThoughtChunks`, `put` without `existing` raises on
   an occupied address and `_index` raises on more than one match;
   `STMChunks.locate` stays traceable.
4. *Gate 3's instantiation is asserted.* No test asserts
   `reason == 'instantiation'`; the worked line's forward step takes that
   route, so assert it there.

Approval follows the repairs green on one new hash; that check is the
hash, the four certificates and the formerly failing core case.

## 19. Text for Codex: round A repairs (2026-10-29)

~~~
Item 2.5 round A, repairs on the review freeze 2515d19ee75e (thinking
spec §18). Alec decided D9: intuit leaves the ordinary thought menu until
round B; the legal mask excludes it while its executor is deferred, and
the descriptor, grammar rule and direct-call NotImplementedError stay.

Build: (1) D9, with a certificate that a thought step at a question never
offers intuit and a direct call still raises; (2) the match's support in
the path minimum: a membership or equality proof's support_true m+ enters
the pair as (min(p+, m+, r+), min(p+, m+, r-)), m+ = 1 for a direct
identity, recorded in `derivation` and recomputed by replay_pair;
certificate: an edge weaker than premise and rule caps both lanes; (3) a
duplicate occupied .where fails loud in ThoughtChunks (put without
existing on an occupied address; _index on more than one match),
STMChunks.locate unchanged; (4) assert reason 'instantiation' on the
worked line's forward step.

Then one new hash: the thought group, the core unchanged set (a second
part only for cases a memory stop leaves, its selection recorded, as
before) and the four-sentence smoke; the native worked-document core case
must pass unseeded. Receipt under
doc/benchmarks/2026-10-29-item2-5-round-a/repairs/; append the result to
the thinking spec as §20. No full sweep, no RUN_SLOW, no seed, no resize,
fail loud. Return through Alec; no commit.
~~~

## 20. Round A repairs and frozen verification (2026-10-29)

The four §18 repairs are complete on source SHA-256
`3a829495289c3687de9263cdd681d0b03e5160eac149a4a4a9b314bda7ead503`
(959 snapshot files). The tree matched review freeze `2515d19ee75e`
before editing. Five source/test files changed from that freeze; their
before text hashes to the reviewed snapshot and their after text equals
the repaired tree. The complete receipt is under
[`repairs/`](../benchmarks/2026-10-29-item2-5-round-a/repairs/), including
the [incremental patch](../benchmarks/2026-10-29-item2-5-round-a/repairs/repair-changes.patch),
[before/after ports](../benchmarks/2026-10-29-item2-5-round-a/repairs/ports.json),
[commands and environments](../benchmarks/2026-10-29-item2-5-round-a/repairs/commands.json),
and [verified results](../benchmarks/2026-10-29-item2-5-round-a/repairs/summary.json).

- **D9:** the controller's legal menu excludes `intuit` until round B,
  including when it is the current request. The descriptor and shipped
  grammar rules remain, and direct execution still raises the round-B
  `NotImplementedError`. The question-step certificate observes the real
  chooser's offered actions without replacing its selection and asserts
  that no offered action is `intuit`; it also exercises the direct error.
- **Match support:** membership and equality matches retain their positive
  support as `derivation.match_support_true`. Forward inference and
  `replay_pair` both use
  $(\min(p^+, m^+, r^+), \min(p^+, m^+, r^-))$ before the existing
  negation and alternative-path operations. Direct identity uses $m^+=1$;
  a direct row identity does not spend a separate equality proof. Native
  membership and equality witnesses with support `.2`, weaker than both
  premise and rule lanes, each cap both resulting lanes at `(.2, .2)` in
  execution and replay.
- **Address uniqueness:** `ThoughtChunks.put(existing=False)` raises before
  mutation on an occupied `.where`; `_index` raises if more than one
  occupied chunk matches. The certificate checks rejection without a
  state change, a valid explicit update, and duplicate reads/updates.
  `STMChunks.locate` remains unchanged and traceable.
- **Instantiation:** the worked line now explicitly asserts
  `reason == 'instantiation'` on its forward inference record. Its ordered
  witnesses, `(.7, .3)` pair, stored inference row, successful replay and
  rejection of tampered support remain certified.

The focused inference file passed all 15 certificates. The requested
frozen checks then completed on the same hash:

- Thought group: **256 passed, 4 skipped, no failures**; all 260 selected
  cases completed.
- Core unchanged set: **1,645 passed, 36 skipped, no failures**; all 1,681
  selected cases completed. The selection and order equal the reviewed
  run. This invocation used two workers with a 16 GiB aggregate process
  limit and the existing 8 GiB per-worker limit. No memory stop or second
  part was needed. The formerly failing
  `test_native_worked_document_trains_unforced_without_numeric_oracles`
  **passed unseeded**, completing its native training batches in about
  60.65 seconds with the test's arithmetic-oracle and runtime-seed
  prohibitions intact.
- Four-sentence native smoke: **exit 0**, about **34.42 seconds**, **11
  native forms**, and **zero rung-zero audit errors**. Both supplied-answer
  checks remain incorrect and acquisition is false; this is an execution
  certificate only.

The receipt verifier confirms the live source still equals the frozen
snapshot, all gate/smoke snapshots match it, the core selection is
unchanged, every repair certificate passed, and the direct executor,
`STMChunks` and shipped grammars equal the reviewed versions. Skip reasons
and full native-case evidence are retained in `summary.json`.

Returned through Alec for review. No full sweep, `RUN_SLOW`, seed pinning,
model capacity resize, new model knob, commit or publication.


## 21. Round A approved (`3a829495289c`; Claude, 2026-10-29)

The hash recomputes and all 959 files equal the live tree; the thought
group (256 passed, 4 skipped), the core unchanged set (1,645 passed, 36
skipped, the same 1,681 cases in the same order, one part) and the smoke
ran on it with `RUN_SLOW` and `BASIC_SEED` unset. The native
worked-document case passes unseeded. The four repairs read as asked: the
menu skips `intuit` while the descriptor, the grammar rule and the direct
error stay (round B removes that skip with the executor); the match's
support enters the path minimum in execution and in `replay_pair`, an
edge at `.2` capping both lanes; a duplicate occupied `.where` raises,
before mutation on `put`; the worked line asserts `instantiation`. Round
A is approved as the serial path of item 2.5, with the worked line's
replay as its exit. Recommended: commit it before round B, generated
evidence excluded as before.

## 22. Decided (Alec, 2026-10-29): the rest of item 2.5 in one text

1. *Scope.* Round B, then the one-backward round, built in that order
   from one text, with a freeze after each stage and one review at the
   end; Codex stops early only on a gate it cannot repair at its producer
   or a decision the spec does not settle. No separate design draft: the
   spec is §§3.3–16.2.
2. *D7: to future work.* Alec: subtraction is meaningful because the
   expectation's gradient is stopped. Not confirmed; recorded in
   `doc/FutureWork.md` and item 2.7 record §49. The conceptual residual
   split (§15's last paragraph, 2.7 record §42) is not built in round B.
3. *Conceptual priming: bottom-up and top-down (Alec, 2026-10-30;
   supersedes the reading first written here).* Bottom-up: the spread of
   activation is an expectation about the next concept state, which
   covers expectation and the spreading of observation alike; it is a
   simple predictor as for percepts, the existing `ActivationPredictor`,
   whose predicted field replaces `_priming_boosts` and the location
   diffusion step as the spread, is read detached by every consumer, and
   trains only on its own error against the detached next state.
   Placement keeps the seen ledger as its context, since the predictor
   reads keys and codes must not read a value computed from codes.
   Top-down: restricted to `intuit()`, a Sigma over Pi with negation at
   the literals (§1 item 4, §3.3) and its own learned weights (Alec:
   "Because it is not understanding the input, it is biased toward doing
   something with it (problem solving)"), from which the relations and
   ideas that fired are pulled into an STM slot (§9's sparse subnetwork).
   It is trained only by its use in output, its weights owned by output,
   with zero gradient from reconstruction and expectation; its weights
   shape what it proposes and how strongly, while an accepted pair is
   still `infer`'s, computed from evidence (§3.6, §3.7). There is no
   open-pass `intuit`. This amends §10 (a learned layer inside `intuit`,
   there not proposed), §15 (expectation and priming consolidate into one
   predictor, not into `intuit`) and the intuit spec §5 ("no learned
   weights").
4. *The §16 details, the cautious package.* The strength
   $s \sim U[1 - j, 1]$ per step with $j$ = `<reconstructionJitter>`,
   default 0; `<separateChooser>` default `true`; under `false` only the
   compose chooser reads live features this round, thought and generate
   staying detached; the chooser term's scale 1.0, its variance reported
   per parameter group before any raise; the round follows round B in the
   same text.

### 22.1 The architecture in psychological terms (Alec and Claude, 2026-10-30)

Alec: "Intuition is beyond understanding, it is conditioned by desired
output, and thoughts are a distillation of intuition." Understanding
explains the input, intuition is understanding put to use, and thought is
intuition that has been checked. The reading, with where it fits and where
it departs psychologically, is in `doc/Reasoning.md`, "Intuition,
understanding and thought".

## 23. Text for Codex: item 2.5 round B, then the one-backward round (2026-10-30)

~~~
Item 2.5, the rest of the item in one text: round B, then the
one-backward round. Governing: thinking spec §§1, 3.3-3.7, 9-16.2, §21
(round A approved) and §22 (Alec's decisions of 2026-10-29/30: scope, D7
to future work, conceptual priming bottom-up and top-down with intuit's
own layer, the §16 package). Build from the spec; append any reading you
had to make.

This replaces the earlier §23 text you began from. Withdrawn from it: the
open-pass Sigma-over-Pi prediction over the definition edges (what you
started as ConceptExpectation), since the bottom-up spread is now
ActivationPredictor's; and intuit re-reading the definition edges, since
intuit now has its own learned weights. Keep what still fits (the
measurement switch, the reference plans' addressing), rework the rest, and
record what you discarded. Round A is committed at 3718d9d0dc.

Stage 1, round B:
1. Bottom-up spread (§22.3): ActivationPredictor's predicted next concept
   state replaces _priming_boosts and the location diffusion step as the
   spread; attention, the chooser, retrieval and intuit's rule selection
   read it detached; it trains only on its own error against the
   detached next observed state. Placement keeps the seen ledger as its
   context. Desire stays its own channel; location's other readers stay.
   The location diffusion step, its controller and knobs retire after the
   measurement in 5 has run on one hash with both available (a temporary
   switch, not a knob, removed before the freeze).
2. Top-down intuit (§1 item 4, §3.3, §9, §22.3): thought-time only; a
   Sigma over Pi with negation at the literals and its own learned
   weights in preallocated sparse banks (no inventory-square tensor; one
   capacity knob in Params.md, its default proposed and recorded); terms
   and edges seeded from the rule rows the predicted field selects,
   within work_budget; at its initial weights it equals §3.3's max-min
   DNF over the seeded rules, per lane, NOT before AND. The fired terms'
   concepts and relations are pulled into one STM slot as references,
   the value deposited detached; one infer candidate per fired rule, its
   firing pair in the request's semantic metadata reaching the chooser's
   logit; order zero to order zero, with MTM candidates' prepended values
   in the knowing (§12). Its weights are output's: trained only by its
   use in output, zero gradient from reconstruction and expectation; an
   accepted pair is still infer's, computed from evidence (§3.6, §3.7).
   Remove D9's menu skip. Once intuit reaches S on a grammar with no
   serial options, retire the parallel flag with that certificate.
3. The implicit route (§3.5): narrowing by the Part edges and partOf, the
   per-lane minimum over parts, kind pervasion, and the close's judgement
   of implicit against explicit (agreement one lane, disagreement both).
4. Not built: the conceptual residual split (D7 is future work).
5. Measurement: lesson-13 once with the predictor's spread and once with
   the location step on one hash: attention, answers, placement, objects.
Gates: §4's 8-10 and 13 as written, 8 and 14 at intuit's initial
weights; 11 and 12 read with §22.3: an answer loss through intuit
reaches intuit's weights on the fired subnetwork and the chooser, no lane
and no LTM byte (state dict byte-equal across the backward and step), and
a wrong answer through one rule moves that rule's weights and no row;
plus the predictor's target receives no gradient. Freeze stage 1 with
the thought groups, the core unchanged set and the smoke; then continue.

Stage 2, the one-backward round (§16, §16.1, §16.2, §22.4):
<separateChooser> (default true, today's per-owner backward) and
<reconstructionJitter> (default 0; s uniform in [1 - j, 1] per step) in
the architecture section and Params.md. Under separateChooser false: one
surrogate differentiated by one backward with one optimizer step per
sentence; output's terms reach the understanding; the compose chooser
reads live features while thought and generate stay detached; the
chooser term at scale 1.0. Every band is stopped at its output where
consumed. Percept expectation's arrival adaptation becomes loop state
with the accumulated arrival loss in the step's backward (§16.2),
replacing the per-arrival optimizer. Certificates: §16 (a)-(d), with (b)
covering ActivationPredictor as an expectation band and intuit's weights
as output's; §16.1's XOR and packing gates and the per-operator
dissonance; §16.2's arrival-state certificate and the compiled brick
equal to the eager loop. Measurement (e)-(f): lesson-13 under (true, 0),
(false, 0) and (false, 0.5): step time, answers and objects, XOR and
packing, dissonance per epoch, the chooser term's variance per group.
Freeze stage 2.

Both stages: receipts under doc/benchmarks/<date>-item2-5-round-b/ and
doc/benchmarks/<date>-item2-5-one-backward/ (the real date; rename the
3X folder) with
frozen-source.json, commands with environments, ports with before/after
text, retirement certificates and allocation reads; Architecture and
Params.md by append or exact replacement; results appended to the
thinking spec as §24 and §25. Stop and return early only on a gate you
cannot repair at its producer or a decision the spec does not settle. No
full sweep, no RUN_SLOW, no seed, no resize, preallocated shapes, no
language named in code, fail loud. Return through Alec for one review; no
commit beyond round A's.
~~~

## 26. Claude's review of the one-backward return (2026-10-30)

Reviewed: Codex's return receipt, `decision.json`, §§24, 25 and 25.1, the
normalization-review ports, the GradientFlow read-first replacement, and
the code at `Generative._relative_energy`, `relative_residual`,
`walk_residual`, `Models._sentence_path_cost` and `Layers.Error.error`.
Round B's frozen source is as §24 records; stage 2's gates are green on
`f9fbf450...` and its measurement is incomplete, so stage 2 is not accepted.
Both are the right states.

**The return is correct, and the diagnosis stands.** Withdrawing the
precision-versus-domain question was right: the overflow is a symptom of
the loss definition, not a float32 problem. Mechanism, from the code:
`_relative_energy` takes `scale` as the maximum magnitude of the live
reference and divides the error energy by the reference energy at that
scale. A step whose parent is numerically near zero, not exactly zero,
gives a tiny `scale`, so `(error/scale)^2` is astronomical: the recorded
forward loss of $2.5\times10^{29}$ is that number, and the live
denominator's derivative is its square root order larger, the
$1.6\times10^{43}$ cotangent. The general contract (`Error.error`,
objective-conflicts spec §10.2a) is built against exactly this: the
baseline is detached, and numerator and baseline are aggregated over every
active entry of the named term before one division, so no single near-zero
target can supply its own normalization. Generation takes neither property:
it divides per step and per lane with the reference live, sums the ratios,
and registers the finished scalar through `Error.add`. The forward value is
already meaningless before any gradient, and a cost of $10^{29}$ in $R$
also decides which trial is kept (`SentenceCredit.comparison` keeps by
$R + A$) in both chooser modes. The `(true, 0)` arm's clean run is
trajectory, not safety: its epoch records hold no value above $10^{20}$,
while the `(false, 0)` arm's gradient report shows chooser variance and
squared norms at $10^{34}$ in the idea epochs, before the non-finite entries.
Not adding a cap, floor, detachment or dropped term was right; each would
have hidden the wrong loss.

**On balancing output and reconstruction (Alec's question).** Under
`separateChooser` false the shared understanding sees $s\,R + A + C$. $R$
and $A$ are addable only because the registry puts each in the same units,
a ratio to its own detached uninformed baseline; that is unit
normalization, not balancing, and nothing adaptive belongs on top of it.
The only weight is $s$, drawn from $U[1 - j, 1]$, and the `(false, 0.5)`
arm is its measurement. The generation residual is the one imbalance in the
system today, and it is a unit violation rather than a weight. Balance
should be read, not tuned: I ask that the gradient report record, per
epoch, the gradient norm into the shared parameters from $R$ and from $A$
separately, as a reading beside the existing per-operator dissonance and
chooser variance.

**Direction, for Alec's decision (D-i to D-iii).**

- **D-i.** Generation's training loss follows the contract: per lane, the
  raw squared error and the detached squared parent, aggregated over the
  walk's steps and the sentence's streams, one division per named term;
  exactly zero parent energy stays an ordinary penalty; no floor. This
  removes an unintended per-step weighting; it adds no balance. If 2.7 §38's
  per-step weighting is wanted for training, it has to be argued for
  separately, because a per-step ratio has no protection against a
  near-zero parent short of a floor, which the contract forbids.
- **D-ii.** The per-step relative residual may stay as the menu's score for
  choosing a proposal, detached, since selection is a hard choice; it is not
  the registered loss.
- **D-iii.** Defaults stay `(true, 0)` until the three arms complete on one
  hash.

Then one short Codex round: the producer repair at `walk_residual` and
`_sentence_path_cost`; the reproducer turned into a passing certificate
(finite loss and gradients on the admissible terminal that overflowed, and
the kept trial unchanged on the standing fixtures, where no near-zero
parent occurs); the three lesson-13 arms and the required verification on
the resulting hash; the stage 2 freeze. No precision change.

**Two readings for Alec, not for this round.** The `(false, 0)` arm bound
2 of 32 answers in four idea epochs against 0 of 80 in every other arm of
rounds A, B and this one; it is the first nonzero binding anywhere, under
the single backward where output reaches the understanding. A measurement,
not a claim, until the arms complete. And the cued LTM0 read is a 32-row
window in address order, not a content search (item 1.5 plan §7.3): with
occupancy above 32 rows, a content read without a reference cannot reach a
later row. It may bear on the zero binding and is Alec's decision in item
1.5.

Codex's documentation ports (GradientFlow's read-first replacement,
Training, §25.1) match the code I read; I have no corrections.

### 26.1 Alec's decisions on retrieval, and the detail of D-i to D-iii (2026-10-30, later)

**Retrieval (decided).** The model must be able to reach all of LTM0. The
cued read keeps a window, but the window is a count of rows scored, not a
count of rows read: its default rises from 32 to 1,024, and the matches
return a top-$K$ by match, never the first $K$ in address order. In code
names, for the next Codex round: `EntryRead.cued_rows` takes the explicit
references first, then the window of the most recent occupied rows (1,024
by default, `max_candidates`), scores every candidate in one product of the
cue's roles against the candidates' stacked roles (cosine per role, masked
by the cue's occupied roles, averaged, as the per-row cosine does today),
applies the kind, scope, binding and role-mask filters as masks, and
returns the top-$K$ by match with the address as the tie-break. $K$ is
`max_records` under the shared meter, and the meter is charged one
`record` per returned record, not per scanned candidate, since scoring the
window is one read; `records_scanned` reports the window. Relation rows
enter the window as today. The future-work parameter: an LTM `.when`
bracket override, an interval in model time that replaces the count window
as the bound on candidates, default unset (FutureWork.md, "LTM retrieval
bounded by a `.when` bracket"). This is a behavior change, so it goes into
the next round's text with the D-i repair and is measured by the same three
arms; no with-and-without arm is needed, since Alec decided the direction.

**D-i, detail.** Today `Generative.relative_residual` returns, per step and
per lane (form, for, against), the ratio `_relative_energy(error, parent)`
with the parent's energy live in the denominator; `walk_residual` sums the
ratios over steps and lanes and adds the frontier ratio; `_sentence_path_cost`
stacks the per-trial scalars, and the close registers the finished scalar
through `errors.add('reconstruction.walk_residual', ...)`. The contract
(`Error.error`) wants raw numerators and detached baselines, summed over the
term's active entries, then divided once. The repair keeps the arithmetic of
each step and changes where the division happens:

1. `relative_residual` returns, per lane, the raw error energy
   $\sum (\text{recovered} - \text{parent})^2$ and the raw parent energy
   $\sum \text{parent}^2$ (the same quantities `_relative_energy` forms
   before dividing, without the detached common scale, which cancels), the
   parent energy detached.
2. `walk_residual` sums each over the walk's steps, adds the frontier's
   error energy and the initial energy to the same sums, and returns three
   numerator and three baseline tensors, one pair per lane, not a ratio.
3. The sentence close registers three named terms,
   `reconstruction.walk_residual.form`, `.for`, `.against`, through
   `errors.error(name, numerator, baseline, mask=active, category='reconstruction')`,
   so the registry sums across the streams and divides once per term; an
   exactly zero parent energy gives the ordinary squared penalty, as the
   contract says; no floor anywhere. Three terms rather than one keep the
   lanes in their own units, which is what 2.7 §38's per-lane normalization
   wanted; what goes is the per-step division. A near-zero parent at one
   step then adds a tiny numerator and a tiny baseline to the sentence's
   sums and cannot create its own scale.
4. The per-step ratio's live-denominator derivative disappears with the
   division, so the $10^{43}$ cotangent has no site; the reproducer's
   terminal gives a finite loss and finite gradients, and becomes a passing
   certificate. The equality gate does not apply to this step: $R$ changes
   value and therefore so can the kept trial; the gates are the certificate
   above, the standing XOR, packing and native worked-document cases, and
   the three lesson-13 arms.

**D-ii, detail.** `eligibility_menu` and the kept-step accumulation call
`relative_residual` per proposal to form the menu's residual features
(form and the two lanes) and the exactness test `f <= 1e-8`. These are
features for a detached chooser and a hard selection, so they stay
per-step, but a per-step ratio with a near-zero parent is as meaningless a
feature as it was a loss. Recommended: the menu's residual features become
the raw error energies of the step divided by the *sentence's* detached
parent energy accumulated so far, per lane (the same baselines D-i sums),
with the first step dividing by its own parent energy; the exactness test
reads the raw error energy against the same $10^{-8}$ of the parent energy.
Detached as before. Alternative, cheaper: keep the ratios and clamp the
feature at a fixed large value; rejected by the rules (a floor that changes
a baseline), kept here only as the contrast.

**D-iii, detail.** `separateChooser` stays `true` and `reconstructionJitter`
stays `0` in `data/model.xml` and `data/BasicModel.xml` until the three arms
$(\text{true}, 0)$, $(\text{false}, 0)$ and $(\text{false}, 0.5)$ complete on
one hash with D-i in place. The decision then reads, per arm: bound and
exact answers, objects and their orders, mean and median training step
time, per-operator dissonance, chooser variance, and the new reading of the
gradient norms into the shared parameters from $R$ and from $A$ per epoch.
The single-backward arm's 2 of 32 is the hypothesis the rerun confirms or
retires. Only Alec changes the default, after that reading.

### 26.2 D-i accepted; D-ii reopened: no ratios of error terms (Alec, 2026-10-30)

Alec: D-i "seems reasonable, make sure the mechanics are in the doc/*.md";
on D-ii, "I object to ratios of error terms. Can't we find a better option?
How exactly is this being used?" The mechanics are now in GradientFlow.md,
"The generation residual: how it is used, and its repair".

**How the residual is used** (code read 2026-10-30, the live tree): the
menu's per-proposal residuals are (1) three of the five detached features
the generate chooser scores (`InverseMenu.features`,
`generate_policy_logits`), (2) the exactness test ($\le 10^{-8}$, with the
pair-first rule), (3) an admissibility gate (non-finite removes the
proposal), and (4) the chosen step's term in `walk_residual`, recomputed
with gradient by `selected_residual` so it trains the operators.

**Why the ratio fails in all four.** All proposals at one step share the
parent, so the division never changed the ranking at a step; it rescaled
the features by a factor that explodes as the parent's lane energy goes to
zero, falls back to the raw error at exactly zero (a discontinuity), and
through the gate can remove a correct proposal. My D-ii recommendation of
dividing by the sentence's accumulated parent energy was still a ratio;
withdrawn. D-i as first written has the same flaw at sentence scale: the
against lane is usually zero or nearly so for a whole sentence.

**Recommended instead: absolute error in the code's own chart, in all four
uses.** Symbolic codes are in the unit cube, so the mean squared error per
lane is bounded in $[0, 1]$; the symbolic exact-pair test in
`decoder_eligibility` already uses it. Numeric walks (`cube_domain=False`)
use the squared difference of $\operatorname{asinh}$ values. Exactness reads
the same measure against the same tolerance; the finiteness gate becomes a
fail-loud assertion; the loss registers with the constant unit baseline the
registry already uses for `reconstruction.reference`,
`reconstruction.free_percept` and the expectation terms. D-i's other
mechanics stand: three per-lane terms through `Error.error`, summed over the
walk without dilution, no floor. Consequence to measure, not compensate:
$R$'s scale against $A$ changes in the kept-trial comparison and in the
single backward; the three arms read it. Pending Alec's yes.

### 26.3 D-ii decided: absolute error in all four uses; the text for Codex (Alec, 2026-10-30)

Alec, on §26.2's recommendation: "Yes, let's try that." The generation
residual is an absolute error in the code's own chart wherever it is used:
the chooser's features, exactness, admissibility and the walk loss, which
registers with the registry's constant unit baseline. GradientFlow.md's
section "The generation residual: how it is used, and its repair" now reads
as decided. With §26.1's retrieval decision, this is the next round:

~~~
Item 2.5, stage 2 repair round (2026-10-30). Governing: thinking spec
§§16-16.2, §22, §25, §25.1, §26 and §§26.1-26.3; GradientFlow.md, "The
generation residual: how it is used, and its repair"; the error registry
contract (Layers.Error.error; objective-conflicts spec §10.2a). Build on
your unaccepted stage 2 source f9fbf450 with its normalization-review
documentation; round B stays frozen at c54db0a8; round A is committed at
3718d9d0dc and the documentation commits since are on main. Defaults stay
separateChooser true, reconstructionJitter 0.

1. The generation residual, absolute in the code's own chart, in all four
   uses.
   a. Generative._relative_energy retires. relative_residual becomes
      absolute_residual(parent, recovered, form_width, *, numeric): per
      lane (form, for, against) the mean squared error of recovered against
      parent. Symbolic walks (cube codes) compare plain values; numeric
      walks (the menus built with cube_domain=False) compare the asinh of
      both values. No division, no floor, no detached scale.
   b. eligibility_menu, inverse_menu, decoder_eligibility and
      selected_residual use it in the walk's chart. InverseMenu.features
      keeps its five columns.
   c. Exactness: absolute mean squared error at most 1e-8, an RMS code
      error of 1e-4, the reader's existing code tolerance. One constant in
      eligibility_menu, inverse_menu and decoder_eligibility, whose
      parent-scaled, floored tolerance retires.
   d. Admissibility: the finite-residual gate becomes a torch._assert_async
      over admissible proposals. A non-finite residual on an admissible
      proposal is a producer bug and fails loud; it removes nothing.
   e. walk_residual sums the chosen steps' per-lane errors over the walk
      and adds the unresolved stack energy per lane, its mean square in the
      walk's chart, without dilution by the step count. It returns [B, 3].
   f. The sentence close registers reconstruction.walk_residual.form, .for
      and .against through errors.error(name, per_stream_sum, 1.,
      mask=active, category='reconstruction'), replacing errors.add of the
      finished scalar. The trial's R in the kept-trial comparison is the
      same three terms.
   Certificates: the native producer reproducer becomes a passing float32
   test (finite loss and finite parameter gradients on the admissible
   terminal that overflowed); menu features within [0, 1] per lane on cube
   codes; a parent lane of exactly zero and one of 1e-30 give residuals
   within 1e-12 of each other; the symbolic exact-pair choice unchanged on
   the standing fixtures; a retirement certificate for _relative_energy and
   relative_residual over bin/ and test/ (the documents name them as the
   record).

2. LTM0 retrieval reaches all of memory. EntryRead.cued_rows takes the
   explicit references first, as today, then the newest occupied LTM0 rows
   (highest address first; if address order is not time order after
   forgetting, use the .when stamp and say so in the readings), then the
   newest relation rows, up to a window of <architecture><ltmWindow>
   candidates, default 1024, with a Params.md row. Every candidate is
   scored in one batched product by the existing match: per-role cosine
   over the cue's occupied roles, when-coordinates zeroed, clamped to
   [0, 1], averaged. The existing kind, scope, binding and role-mask filters
   apply, as masks where tensor-valued and in score order where they read
   host metadata. The read returns the top-K by match with the address as
   the tie-break, as today. K is max_records; the shared meter is charged
   one 'record' per returned record, not per scanned candidate.
   records_scanned reports the window; incomplete names 'ltm_window' when
   the occupied rows exceed it and 'work_budget' when the meter cut K short.
   Certificates: on a store whose candidates fit within 32 and the meter,
   the same records in the same order as today; on a store of 2,000 rows, a
   matching row past address 32 inside the newest 1,024 is returned and one
   outside the window is not, with 'ltm_window' reported; the meter is
   charged per returned record. Leave recall_time, premise_evidence and
   relation_evidence unchanged, and list in your report every LTM reader
   that still scans in address order under the meter.

3. Gradient report, read-only: per epoch, the gradient norm R sends and the
   norm A sends into the understanding's parameters (codes, grammar
   operators and inverses, conceptual definitions), from the existing
   diagnostic collection, its time excluded from step time as now. Under
   separateChooser true, A's contribution there is zero by construction and
   the reading shows it.

4. Then, on one source hash: the required selections as in §25
   (certificates, thought, core, ports-and-finite), the native worked
   document unseeded, the four-sentence smoke, and lesson-13 under
   (true, 0), (false, 0) and (false, 0.5), sequentially. Each arm reports
   bound and exact answers, objects and their orders, mean and median
   training step time, per-operator dissonance, chooser variance and the
   R and A gradient reading per epoch. Freeze stage 2 on that hash.

Receipts under doc/benchmarks/<date>-item2-5-repair/ (the real date):
frozen-source.json, commands with environments, ports with before/after
text, the retirement certificate, allocation reads. GradientFlow.md,
Architecture.md, Training.md and Params.md by append or exact replacement;
results appended to the thinking spec as §27. Stop and return early only on
a gate you cannot repair at its producer or a decision the spec does not
settle. No full sweep, no RUN_SLOW, no seed, no resize, preallocated shapes,
no language named in code, fail loud. Do not read or merge the item 1.5
sandbox branch. Return through Alec for one review; no commit.
~~~

## 28. Claude's review of the stage 2 repair return (2026-10-30)

Reviewed: §27, `review.md`, `readings.md`, the order-boundary reproducer and
the code at `cb8a159a6792...`. The frozen source verifies against the live
tree (970 files, none differ). The early return is the right call: the
order case is a decision the spec does not settle, and Codex changed no
contract to get past it.

**The repair is built as §26.3 reads.** `absolute_residual` is the per-lane
mean squared error, plain for cube codes and in the $\operatorname{asinh}$
chart for numeric walks; `walk_residual` returns the undiluted per-lane sums
plus the unresolved stack energy; the close registers the three lanes
through `Error.error` with the unit baseline; one exactness constant,
`EXACT_RESIDUAL_TOLERANCE`, serves the menu, the inverse menu and the
decoder; admissible proposals assert finiteness. The overflow reproducer
now gives a finite loss of 0.25 with finite gradients. The cued read scores
one window (explicit references, then the newest LTM0 rows, then relation
rows by stamp) in one batched product, returns the top-$K$ by match and
charges one record per returned record; the 2,000-row certificate reaches
row 1777. The R and A reading shows 125.36 and 0 under the separate
chooser, as ownership requires.

**Decision for Alec (D-iv): which positive rows between same-order
individuals are structure.** Two individuals already witnessed at order 1,
and a sentence asserts `equal` or `part` between them. The structural
projection (`ConceptAllocator.refs(cid, 'part')`, read by `_part_order`)
counts every positive Part row as a member, so the whole derives order 2
against its recorded order 1 and the epoch boundary raises, correctly, by
the guard. Recommended: *order counts only lower-order members.* For a whole
with a recorded order $n$, a positive row whose member has order $n$ is
testimony: stored with its pair and read by queries, thought, `intuit` and
pervasion, but not a structural member, so it neither derives the whole's
order nor teaches its definition or class membership. A member below $n$ is
structure as today; a member above $n$ still trips the guard, as the
presentation-memory test requires. A whole with no recorded order (an
implicit definition or a class) counts every positive member, as today. No
new column: the projection reads the members' recorded orders. This agrees
with the orders spec: one equality writes its row without changing order,
and distinct witnesses form a separate raised class, whose members are of
lower order. Rejected: typing the row at write time (the grammar supplies no
such type); restamping the whole or dropping the row (Codex's reasons).

**Three small items for the same round.**

1. `LanguageSpace._bounded_binary_reconstruction` still tests progress
   against a threshold scaled by the parent's energy. It was outside the
   named sites, but it is the same ratio test; it takes
   `EXACT_RESIDUAL_TOLERANCE`.
2. The R and A reading keys its epoch vectors by parameter identity, so
   contributions to a sparse definition parameter replaced mid-epoch are
   dropped. Key by parameter name so the reading survives growth.
3. For item 1.5, not this round: the cued read calls `LTM0.row()` for every
   window candidate, about fifteen host reads each, before scoring. Roles
   for occurrence rows can be gathered from `slots`, and host metadata read
   only in rank order. This may explain part of the first epoch's mean step
   of 22.6 s against 12.8 s before (medians 8.4 and 9.2); one epoch is not a
   measurement.

On Alec's answer the next round implements D-iv and the two items, then runs
the three arms and freezes stage 2 on one hash.

### 28.1 D-iv decided: IS is not marked; rows between known concepts are relations; the text for Codex (Alec, 2026-10-31)

Alec answered D-iv through the demonstratives in the failing epoch; his
words are in Orders.md §11. In short:

- A demonstrative is a pointer. It draws attention to a region and can point
  at an event, an individual or a variable concept; the operation for it,
  maybe `point()`, is language-independent. Pointers are found by their
  confusion: the identity the learner gives the word is left to build up
  confusion and is disambiguated when it shifts too much. A very polymorphic
  word "can only be known as 'this' or 'suchness'" and is promoted with no
  consistent differentiating identity. Future work (FutureWork.md,
  "Polymorphic words, suchness and `point()`").
- Identity is marked by articles and shown by non-commutativity: "the cat is
  black, but black is not the cat."
- "We shouldn't mark 'is' as special." The testimony column proposed in
  conversation is withdrawn, and so is §28's "order counts only lower-order
  members". An equality between known concepts is a relation across them,
  not a single idea.

The reading this round builds (Orders §11): a positive LTM1 row whose two
referents are both symbols with recorded orders is a member of the whole
only when the member is exactly one order below, a class and its member;
any other such row is a relation across two known concepts, kept and read as
before but a member of neither. In the failing epoch, row 72 joins two
order-1 concepts, so it is a relation, concept 56 keeps order 1 and the
confusion is left to build, as Alec wants. The presentation-memory guard
case, an affirmed relation member, is unchanged.

~~~
Item 2.5, stage 2 completion round (2026-10-31). Governing: thinking spec
§§26.3, 27, 28 and 28.1; Orders.md §10 and §11; step 2 §71 (a row's kind is
read from its referents). Build on your unaccepted repair source
cb8a159a6792 with its receipts; round B stays frozen at c54db0a8; round A is
committed at 3718d9d0dc. Defaults stay separateChooser true,
reconstructionJitter 0.

1. A row between known concepts is a relation across them (Orders §11).
   a. In the structural projection, the reading that
      ConceptAllocator.refs(cid, 'part') and records give to _part_order,
      settle, taxonomy_children, class_members and every other reader of
      that projection: a positive LTM1 row whose two referents are both
      symbols with recorded orders (reference_orders) is a member of the
      whole only when the member's order is exactly one below the whole's.
      Any other such row is a relation across two known concepts and a
      member of neither.
   b. The row itself is unchanged: written with its pair and stamp, kept in
      LTM1, and read as before by queries, thought, intuit, pervasion,
      define_pending and the raise. PartClosing._observe_equality and
      ConceptIndex.index_part_row are unchanged.
   c. Occurrence and relation members, and every member of a whole with no
      recorded order, are read as today; settle still raises when they
      disagree with the recorded order.
   d. Nothing reads the operator, the clause or the row's origin; no
      column, flag or write-time type is added.
   Certificates: your order-boundary reproducer becomes a passing test for
   equal and for part: the row is written with its pair, both concepts keep
   order 1, the row is found by parts.find and is absent from the whole's
   structural records, and the epoch boundary passes. A known member one
   order below a known whole remains a structural member; a known member
   above it is a relation, not a member, and the boundary passes.
   test_unasserted_part_does_not_raise_a_retained_identity_order passes
   unchanged, and so do the existing raise tests, including the second
   distinct witness raising the shared side. If a standing test asserts
   that a symbol with a recorded order is a structural member of a known
   whole at any order other than one above it, stop and return the test
   named; do not edit it.

2. The two small items of §28.
   a. LanguageSpace._bounded_binary_reconstruction's progress test reads
      the absolute residual against EXACT_RESIDUAL_TOLERANCE; its
      parent-scaled threshold retires. Certificate: parent lanes of exactly
      zero and of 1e-30 give the same decision.
   b. The R and A reading keys its epoch vectors by parameter name, so a
      sparse definition parameter replaced mid-epoch keeps its
      contribution. Certificate: a replacement mid-epoch retains the
      earlier contribution under the same name.
   §28's third item (the cued read's per-candidate LTM0.row calls) belongs
   to item 1.5, not this round.

3. Then, on one source hash: the required selections as in §25
   (certificates, thought, core, ports-and-finite), the native worked
   document unseeded, the four-sentence smoke, and lesson-13 under
   (true, 0), (false, 0) and (false, 0.5), sequentially. Each arm reports
   per epoch: bound and exact answers, objects and their orders, mean and
   median training step time, per-operator dissonance, chooser variance and
   the R and A reading. Also per epoch, as observation for the future work
   on polymorphic words: for each idea sentence, the symbols its two sides
   resolved to, with their orders, and the count of relations across known
   concepts written. Freeze stage 2 on that hash.

Receipts under doc/benchmarks/<date>-item2-5-completion/ (the real date):
frozen-source.json, commands with environments, ports with before/after
text, allocation reads. Architecture.md by append or exact replacement for
the projection's reading; results appended to the thinking spec as §29.
Stop and return early only on a gate you cannot repair at its producer or a
decision the spec does not settle. No full sweep, no RUN_SLOW, no seed, no
resize, preallocated shapes, no language named in code, fail loud. Do not
read or merge the item 1.5 sandbox branch. Return through Alec for one
review; no commit.
~~~

### 28.2 Alec's answers of 2026-10-31, later: the fold and the cover, `raiseCount`, symmetric equality; the addendum for Codex

Alec, on Claude's question whether the fold and the cover follow §11:
"This should be done now; extension in LTM1 creates higher-order concepts
(classes), and a raiseCount of at least 3 seems necessary." On the
disambiguation of indexicals: "'this is Felix' and 'this is ciao' are
actually better read as symmetric, because that creates a transitivity that
breaks (which is what we want as evidence for a polymorphic reference)."
Recorded in Orders §11 (its last paragraph) and FutureWork ("Detection by
broken transitivity"). Correction to §28.1's item 1a: `class_members`
(FamiliarTier) is not a reader of the projection; it and
`DefinitionFold.examples` walk the Part store directly, which is why item 4
below exists. The symmetric raise itself is not in this round; the round
observes it (item 6). This text follows §28.1's, and Codex applies both on
one source hash:

~~~
Item 2.5, stage 2 completion round, addendum (2026-10-31). Governing as
§28.1, plus Orders §11's last paragraph (extension creates classes;
raiseCount; symmetric equality) and FutureWork, "Detection by broken
transitivity". Applies with §28.1's text on the same source hash.

4. The definition fold and the class cover read membership as the order
   projection does. DefinitionFold.examples and FamiliarTier.class_members
   exclude a row whose two referents are both symbols with recorded orders
   unless the member's order is exactly one below the whole's. Every other
   row they read today they still read, in both lanes: negative evidence
   on a whole with no recorded order, occurrence members, members of an
   unbound shared side. Certificates: a positive row between two order-1
   concepts yields no fold example and no cover weight for either side; a
   member one below still yields both; a negative row on an unbound shared
   side still yields its example; the standing fold tests pass unchanged.

5. The raise counts distinct members: <ConceptualSpace><raiseCount>,
   default 3, with a Params.md row; a value below 2 is rejected. The
   shared side acquires its class, one order up, when the number of
   pairwise distinct members witnessed above raiseEvidence reaches
   raiseCount, in PartClosing._observe_equality and on define_pending's
   minting path alike. Until then the witnesses are kept, the singleton
   bookkeeping extended to raiseCount - 1 members per shared side with
   distinctness judged as today, by token or by event; the rows are
   present as evidence; no concept exists one order above the shared side;
   a word on that side has no object. At the count, one parent is created
   with every kept member as a structural row and the word is defined to
   it, as today's second witness does; a later one-below member joins as
   today. conceptMintCount, already 3, is unchanged. Certificates: two distinct
   witnesses raise nothing, both rows are found by parts.find and neither
   concept changes order; the third raises one parent at the shared side's
   order plus one with three members, and the word's object is that
   parent; a fourth joins it. The standing raise tests gain a third
   distinct witness rather than a lowered count; a test that cannot be
   given one is returned by name, not edited.

6. Observation only, per epoch in each lesson arm, beside item 3's: for
   each concept with a recorded order, the number of distinct concepts of
   the same recorded order joined to it by positive LTM1 rows, read in
   either position; the number of concepts for which that count is 2 or
   more, the transitivity breaks of Orders §11; and for each such concept
   its word forms and the joined concepts with their orders. No raise,
   promotion or fold reads this.

Receipts, Architecture.md and Params.md by append or exact replacement,
results in §29, and every other condition as §28.1.
~~~

### 28.3 Alec's correction of 2026-10-31, later still: no symmetry handling, no break counter, no `point()`; the addendum for Codex, replacing §28.2's

Alec: "The 'symmetric raise' should just be a raise on raiseCount, we
don't need any symmetry handling (the architecture itself is symmetric).
And we don't need to count breaks, that can be seen by inspecting LTM1, so
we just need to do that periodically. This eliminates an explicit point(),
words that are pronouns or indexicals will not find an identity in
concepts, so they will find only temporary matches in MTM." Recorded in
Orders §11 and FutureWork, both in their closing paragraphs. §28.2's item 6
is withdrawn, and so is §28.1 item 3's count of relations across known
concepts, for the same reason: both are read off the LTM1 rows the epoch
receipts already keep. §28.2's item 5 is restated so that the count reads
no position. Codex applies this text with §28.1's, in place of §28.2's:

~~~
Item 2.5, stage 2 completion round, addendum (2026-10-31, replacing the
earlier addendum). Governing as §28.1, plus Orders §11's last two
paragraphs and FutureWork's closing paragraph. Applies with §28.1's text
on the same source hash. §28.1 item 3's count of relations across known
concepts is dropped; its per-sentence resolution report stays.

4. The definition fold and the class cover read membership as the order
   projection does. DefinitionFold.examples and FamiliarTier.class_members
   exclude a row whose two referents are both symbols with recorded orders
   unless the member's order is exactly one below the whole's. Every other
   row they read today they still read, in both lanes: negative evidence
   on a whole with no recorded order, occurrence members, members of an
   unbound shared side. Certificates: a positive row between two order-1
   concepts yields no fold example and no cover weight for either side; a
   member one below still yields both; a negative row on an unbound shared
   side still yields its example; the standing fold tests pass unchanged.

5. The raise counts distinct members: <ConceptualSpace><raiseCount>,
   default 3, with a Params.md row; a value below 2 is rejected. A concept
   joined by positive rows above raiseEvidence to raiseCount pairwise
   distinct concepts at its own order, distinctness judged as today by
   token or by event, acquires its class one order up with those concepts
   as structural members. A row's two positions are not read for this
   count. An unbound shared side takes its members' order as today. This
   holds in PartClosing._observe_equality and on define_pending's minting
   path alike. Until the count the witnesses are kept, the singleton
   bookkeeping extended to raiseCount - 1 per concept; the rows are present
   as evidence; no concept exists for it one order above; a word on that
   side has no object. At the count, one parent is created and the word is
   defined to it, as today's second witness does; a later one-below member
   joins as today. conceptMintCount, already 3, is unchanged. Certificates:
   two distinct witnesses raise nothing, both rows are found by parts.find
   and neither concept changes order; the third raises one parent at the
   concept's order plus one with three members, and the word's object is
   that parent; a fourth joins it; three rows that share their first
   referent raise it as three that share their second do. The standing
   raise tests gain a third distinct witness rather than a lowered count;
   a test that cannot be given one is returned by name, not edited.

6. Nothing counts breaks and no pointer operation is built. The epoch
   receipts keep their LTM1 rows as they do now; that is the inspection.

Receipts, Architecture.md and Params.md by append or exact replacement,
results in §29, and every other condition as §28.1.
~~~

## 30. Claude's review of the completion return (2026-10-31)

Reviewed §29 and §29.1 with the receipt under
`doc/benchmarks/2026-10-31-item2-5-completion/`. Verified here: the
candidate's source hash `30f6e1613555` is the working tree's (971 files,
recomputed), its base is `7723cfce2a`, and it is built from the repair
source `cb8a159a6792`. The twelve completion certificates pass; the
standing raise, fold, taxonomy, sweep and presentation-memory files give
109 passed and the one failure Codex reports from the baseline. Codex's
selection of 156 (155 passed, 1 failed) did not include
`test_sweep_repairs.py`, so that test's state on the candidate is from my
run: it fails at line 17 as it did on `cb8a159a6792`.

**As specified.** Item 2a: the absolute residual against
`EXACT_RESIDUAL_TOLERANCE`; parent lanes of 0 and 1e-30 decide alike. Item
2b: the R and A reading keyed by parameter name, with edge identities
across a replaced sparse parameter. Addendum item 4: `DefinitionFold.examples`
and `FamiliarTier.class_members` gate on `ConceptAllocator.class_member`;
the three certificate cases pass. Addendum item 5: `raiseCount`, default 3,
below 2 rejected at construction, with its Params.md row and schema floor;
`PartClosing.observe_members` admits each endpoint as a member of the
other, so no position is read; witnesses sit in `raiseCount - 1` seats on
the fixed slab with their rows in `singleton_rows`, checkpointed; the
minting path calls the same observation with the minted symbol and its
original row, which the seat check recognizes and does not count twice; the
standing raise tests gain a third witness and none is returned. Addendum
item 6: no break counter and no relation count; the epoch record now lists
the LTM1 rows with their references and evidence, which is the inspection.
The per-sentence operand observation of §28.1 item 3 is built; the
measurement is not run.

**Two notes, no action.** `_admit_member` skips a symbol member that has
no MTM token instead of seating it; the earlier code seated it and could
never judge it distinct, so the seat stayed blocked until forgotten. The
definition index skips, on rebuild, an object-to-word row whose object is
a member of the word's own raised class, keyed on the `part-base` index
entry; the certificate that rebuilds the index after a raise passes.

**One deviation, Alec's decision (D-v).** §28.1 item 1a and Orders §11 read
a positive row between two symbols with recorded orders as a member only
one order below and otherwise as a relation, a member of neither. The
candidate adds an exception at order 0: `ConceptAllocator.records` keeps a
positive row between two order-0 symbols as a structural part, visible to
`refs`, `taxonomy_children` and pervasion, while a second reading,
`records(cid, order_only=True)`, excludes it from `_part_order` and
`settle`, and `class_member` excludes it from the fold and the cover.
Codex names Orders §1's "overlap and containment" at order 0 and "the last
clarification in the hand-off" as the ground. Orders §1's containment is
among percepts in the field; an order-0 symbol is a word form or a placed
type, and Orders §11 has no order-0 clause. The exception does not rescue
the standing test that prompted §29, whose rows carry no evidence.
Claude's recommendation: one rule at every order, as recorded; remove the
exception and the second reading, return `_part_order` to `refs`, rewrite
the containment certificate to the one rule, and correct the first
paragraph of Architecture.md's new section. If Alec wants containment
between order-0 symbols, it is a decision for Orders §11, recorded there
first. If the clarification Codex cites was Alec's, it is recorded the
same way.

**Two tests, Alec's disposition.**

- `test_sweep_repairs.py::test_semantic_word_chain_does_not_redefine_a_word_form_order`
  (item 3.0). Failing on the starting source already: its chain row has
  evidence (0, 0), which the structural filter excludes; it was outside the
  §27 selections. Both halves encode the earlier reading: a same-order row
  as a structural part (line 17) and a same-order part forcing an order
  disagreement (lines 23 to 26). Under Orders §11 both are false. Claude's
  recommendation: keep the test's name and purpose, rewrite its body to
  the one rule: the row is found by `parts.find`, absent from both
  structural projections, both orders stay 0, and a third order-0 concept
  joined to the left word by a positive row also stays 0 with no error
  from `settle`.
- `test_item25_repair.py::test_reflexive_part_fact_is_not_an_intuition_definition_cycle`
  (Codex's own, stage 2 repair, unaccepted). Its last assertion expects
  reciprocal positive rows between two order-1 concepts to form a
  structural cycle; under Orders §11 they are relations and nothing
  cycles. Codex's proposed patch asserts both rows readable and neither a
  member, the intuition value unchanged, then writes an actual constituent
  cycle through the store's own writer and expects the guard's error.
  Claude's recommendation: accept the patch as proposed.

**Pending on the next hash.** Thought, core with the native worked
document, ports-and-finite, smoke, the three lesson-13 arms and the stage
2 freeze, all on one source hash after the edits above; §28.1's conditions
unchanged. The Codex text follows Alec's two answers.

### 30.1 Alec's decisions on the review (2026-10-31)

Alec accepted the recommendations in §30 and asked that one thing be recorded.

- **One rule at every order.** The order-zero containment exception is withdrawn. The rule of Orders §11 holds at order zero (Orders §11, 2026-10-31). §30.2 item 7 gives Codex the change.
- **The word-chain test.** `test/test_sweep_repairs.py::test_semantic_word_chain_does_not_redefine_a_word_form_order` is rewritten to the one rule, keeping its name and purpose (§30.2 item 8).
- **The reflexive-cycle test.** Codex's proposed patch, `doc/benchmarks/2026-10-31-item2-5-completion/proposed-cycle-fixture.patch`, is accepted as proposed. It applies cleanly to the current `test/test_item25_repair.py`.
- **Relations are built first.** Alec: "we build relations in stm1 or ltm1 first, and those will eventually become evidence for raising the order of concepts." Recorded in Orders §11 with the two strata named. The completion round changes neither where a relation is built nor which rows the raise reads.

### 30.2 Addendum for Codex, items 7 to 10 (Alec, 2026-10-31)

Governing as §28.1 and §28.3, with §30.1 as Alec's decisions, on one source hash. Results go in §31.

7. One rule at every order, order zero included. Remove the containment exception that the completion round added.
   - In `ConceptAllocator.records`, delete the branch that keeps a positive row between two order-zero symbols as a structural part, and delete the `order_only` reading. A positive row between two symbols with recorded orders is a structural part only when the member is exactly one order below the whole. Otherwise it is a relation: found by `parts.find`, absent from `refs` and `taxonomy_children`, and deriving no order.
   - `_part_order` reads `refs` again.
   - `class_member` keeps its reads for members and wholes with no recorded order, as §28.1 item 1 says.
   - In `test/test_item25_completion.py`, rewrite the order-boundary certificate to the one rule and rename it `test_order_boundary_membership_follows_one_rule`. A positive row between two order-zero symbols is a relation, and a member one order below still yields membership.
   - In the section you appended to `doc/Architecture.md`, replace the two sentences on containment at the same order in its first paragraph with the one rule, keep the rest, and retitle the heading to "Class membership and the counted raise (2026-10-31)".

8. Two standing tests, by Alec's decision.
   - In `test/test_sweep_repairs.py`, rewrite `test_semantic_word_chain_does_not_redefine_a_word_form_order` to the one rule, keeping its name and purpose. The row is found by `parts.find`. Neither word is listed in the other's `refs` or `taxonomy_children`. Both orders stay 0. A third order-zero concept joined to the left word by a positive row also leaves both orders at 0, and `settle` raises nothing. Drop the disagreement assertion.
   - Apply `doc/benchmarks/2026-10-31-item2-5-completion/proposed-cycle-fixture.patch` to `test/test_item25_repair.py` as proposed.
   - These are the only standing tests this addendum changes. If another standing test fails after item 7, return it by name and do not edit it.

9. Relations come first. Alec, 2026-10-31: "we build relations in stm1 or ltm1 first, and those will eventually become evidence for raising the order of concepts." Nothing in this round changes where a relation is built or which rows the raise reads.

10. Run and freeze. After items 7 to 9, on one source hash, run the standing selection, now including `test/test_sweep_repairs.py`; the twelve completion certificates; thought; core with the native worked document; ports-and-finite; smoke; and the three lesson-13 arms, (true,0), (false,0) and (false,0.5). Then take the stage 2 freeze. Receipts go under `doc/benchmarks/2026-10-31-item2-5-completion/`. The conditions of §28.1 hold: no RUN_SLOW, and targeted node IDs while iterating. Stop and return early only as §28.1 says.

### 30.3 Alec's answers to the open items (2026-10-31); the addendum for Codex, items 7 to 10, revised (replaces §30.2)

Alec answered the three questions left open after §30.2. Each answer is recorded in Orders §11, under its own paragraph.

- **Pronouns and indexicals.** One word class, not noun; its name is open, and nothing in this round builds it. Orders §11's reading that an indexical is minted first and raised to an amorphous class was not corrected, so it stands. Whether an indexical is minted at all stays open until Alec says otherwise.
- **Minting and raising.** Alec: "Minting and raising are different." `raiseCount` governs raising only, and `conceptMintCount` governs minting only. The minting path still feeds the raise as evidence, as §28.3 item 5 has it; that observation is not a mint. Items 7 to 10 change neither count.
- **The clarification.** The sentence the readings cite as "the last clarification in the hand-off" is "The conflict is whether a row joining two order-0 symbols makes one a structural part of the other." Alec's answer: "No, it's a part/whole relation if it's below symbols, and a set/superset if it's above symbols." The answer is no, so the ground the readings give for the order-zero exception does not hold. Item 7 records this and corrects the citation.

The addendum for Codex follows. It replaces §30.2's items 7 to 10 and applies on one source hash.

~~~
Item 2.5, stage 2 completion round, addendum 2, revised (Alec, 2026-10-31). Governing as §28.1 and §28.3, with §30.1 and §30.3 as Alec's decisions, on one source hash. Results go in §31.

7. One rule at every order, order zero included. Remove the containment exception that the completion round added. Alec's answer to the clarification that readings.md cites is no (Orders §11, 2026-10-31).
   - In `ConceptAllocator.records`, delete the branch that keeps a positive row between two order-zero symbols as a structural part, and delete the `order_only` reading. A positive row between two symbols with recorded orders is a structural part only when the member is exactly one order below the whole. Otherwise it is a relation: found by `parts.find`, absent from `refs` and `taxonomy_children`, and deriving no order.
   - `_part_order` reads `refs` again.
   - `class_member` keeps its reads for members and wholes with no recorded order, as §28.1 item 1 says.
   - In `test/test_item25_completion.py`, rewrite the order-boundary certificate to the one rule and rename it `test_order_boundary_membership_follows_one_rule`. A positive row between two order-zero symbols is a relation, and a member one order below still yields membership.
   - In the section you appended to `doc/Architecture.md`, replace the two sentences on containment at the same order in its first paragraph with the one rule, keep the rest, and retitle the heading to "Class membership and the counted raise (2026-10-31)".
   - In `doc/benchmarks/2026-10-31-item2-5-completion/readings.md`, correct line 3 to the one rule: the clarification it cites is answered no, and its same-order and concrete containment readings are withdrawn. Bring the rest of that file into line with items 7 and 8.

8. Two standing tests, by Alec's decision.
   - In `test/test_sweep_repairs.py`, rewrite `test_semantic_word_chain_does_not_redefine_a_word_form_order` to the one rule, keeping its name and purpose. The row is found by `parts.find`. Neither word is listed in the other's `refs` or `taxonomy_children`. Both orders stay 0. A third order-zero concept joined to the left word by a positive row also leaves both orders at 0, and `settle` raises nothing. Drop the disagreement assertion.
   - Apply `doc/benchmarks/2026-10-31-item2-5-completion/proposed-cycle-fixture.patch` to `test/test_item25_repair.py` as proposed.
   - These are the only standing tests this addendum changes. If another standing test fails after item 7, return it by name and do not edit it.

9. Relations come first. Alec, 2026-10-31: "we build relations in stm1 or ltm1 first, and those will eventually become evidence for raising the order of concepts." Nothing in this round changes where a relation is built or which rows the raise reads.

10. Run and freeze. After items 7 to 9, on one source hash, run the standing selection, now including `test/test_sweep_repairs.py`; the twelve completion certificates; thought; core with the native worked document; ports-and-finite; smoke; and the three lesson-13 arms, (true,0), (false,0) and (false,0.5). Then take the stage 2 freeze. Receipts go under `doc/benchmarks/2026-10-31-item2-5-completion/`. The conditions of §28.1 hold: no RUN_SLOW, and targeted node IDs while iterating. Stop and return early only as §28.1 says.
~~~

### 30.4 Alec's further answers (2026-10-31); the addendum for Codex, items 11 to 13 (addendum 3)

Alec answered on indexicals and on the word class, and asked for the Codex text.

- **Indexicals and pronouns.** Alec: "Indexicals are roughly word-concepts that reference percepts; so they are a bit parallel to the interpret() operation, in that they invite the model to witness(). ... Since the model is not currently getting visual input, it can remain an MTM reference, but add it to the future work for multimodal processing." Recorded in Orders §11 ("Indexicals witness, pronouns bind") and FutureWork ("Indexicals under multimodal input"). Nothing in this round builds the witness.
- **The class.** Alec: "I don't know what you mean, but am willing to accept whatever proposal. What is its function?" The class is named deictic, with its function stated in Orders §11 ("The deictic class") and a row in Language.md's word-class table. Items 11 to 13 build it. Alec: "Ok, then emit the code for codex, please."

The addendum follows. It joins §30.3's items 7 to 10 in the same round, on one source hash: items 11 to 13 are applied before item 10's run and freeze. Results go in §31.

~~~
Item 2.5, stage 2 completion round, addendum 3 (Alec, 2026-10-31). Governing with §30.3 on one source hash; items 11 to 13 precede item 10's run and freeze. Results go in §31.

11. The deictic word class: pronouns and indexicals, one class, not noun (Orders §11, "The deictic class").
   - Representation. The class is a word-kind held as the WORDS class is held (`ConceptualSpace._register_recognized_word`): one higher-order concept, `ConceptualSpace._deictic_concept_id`, admitted on first membership through `_automatic_concept_admitted(self, 1, reason="DEICTIC class")` and `new_concept`, whose parts are `('sym', word)` for each member word. Two methods on `ConceptualSpace`: `add_deictic(form)` registers a surface form, and the word admitted for that form (`InterpretLayer.lookup_word`, `word_for_form`) joins the class at admission, so tests and the harness can force membership before the first sentence; `is_deictic(word)` reads membership by word id. No parameter in `data/model.xml` and no word list in the grammar: parts of speech are learned, not written (Orders §10), and tests force this one as they force the others. Learning membership is not this round.
   - A deictic never acquires an object. `InterpretLayer.define(word, obj)` raises `ValueError` naming the word when `is_deictic(word)`; `no_object(word)` therefore stays true for a member, and `definitions.objects(word)` stays empty. `promote_identity` (`FamiliarTier.py`) removes deictic words from the set of names it defines to the minted identity, both the explicit `word` and the occurrences' readings from `pool.reading_of`. The raise is untouched: a deictic word has no rows of its own.
   - Each mention is its own request. `InterpretLayer.reading(word)` does not call `pool.named` for a deictic; it returns the word as its own placeholder with the word's row and order, as its frozen path does, and opens no MTM token, so two mentions of "this" share no token unless `bind` joins them. The mention reaches `bind` as an open referent exactly as an objectless word does today (`ClauseRow`'s `('referent', role)`), with the open marker and the mint option as its fallbacks when no antecedent is found: the mint is the fresh occurrence an indexical points at while there is no field. `bind` is not changed: no word class participates in its read (its docstring), and this item adds none.
   - A kept definition with a deictic subject defines nothing to the deictic. At the site that calls `define_pending` and records `_kept_definition_witnesses` (`PartClosing._write`), `word` is `None` when the subject's word is a deictic, and in `define_pending` (`FamiliarTier.py`) the subject's token is found by the left reference, never by `('name', word)`. Then: if the other side is a known identity at the order the token would take, the token's occurrences join that identity as its `('occ', ...)` parts, the row between the occurrence and the identity is written, and no new identity is minted; otherwise the token is promoted as today, with the row written through `observe_members`, and the only word defined to the new identity is an unbound label on the other side (`ConceptIndex.unbound_label`), which names what was pointed at. No row written at a sentence close has a deictic word symbol as an endpoint. So "This is Felix. That is Felix. This is Felix." yields one identity named Felix, at order 1, with three occurrences as its parts, while "this" and "that" have no object (Training.md, "Word classes in order"). The one rule of Orders §11 holds: each occurrence is exactly one order below Felix.
   - Fail loud (§28.1): a member word that reaches `define`, a deictic subject whose token cannot be found by its reference, and a sentence-close row with a deictic word symbol as an endpoint each raise, with the word and the reference in the message.

12. Certificates, in `test/test_item25_completion.py`, forcing membership through `add_deictic`:
   - `test_deictic_word_never_acquires_an_object`: "this" is a member; after a kept "This is Felix.", `interpret.no_object(this)` is true, `definitions.objects(this)` is empty, `interpret.define(this, cid)` raises, `is_deictic(this)` is true, and no row written at the close has `(SYMBOL, this)` as an endpoint.
   - `test_deictic_definition_collects_occurrences_under_the_name`: "this" and "that" are members; after kept "This is Felix. That is Felix. This is Felix.", exactly one identity is named Felix, its order is 1, its parts are the three occurrences, each occurrence-to-Felix row is found by `parts.find`, and no identity or class exists for "this" or "that". With `raiseCount` at its default, Felix does not raise: its three joined symbols are its members one order below, not peers.
   - `test_pronoun_binds_in_mtm_and_stays_objectless`: "it" is a member; after a kept "A cat sleeps." the second sentence "It is black." lists the cat's identity among the bind's candidates for its subject, and with that candidate selected, as `test_live_references.py` selects proposals, the subject resolves to it; "it" has no object, and MTM holds no token named by "it".
   - `test_name_still_defines_at_first_definition`: "Felix" is not a member; a kept "Felix is a cat." defines Felix to the minted identity as today. This pins that the class, not objectlessness, makes the difference.
   - The lesson-13 harness (`bin/corpus_lessons.py`; the "proper" lesson of `data/corpus-lessons.json` has "This is Felix." and "That is Ciao.") forces "this" and "that" into the class through `add_deictic` at model construction, as unit tests force parts of speech (Training.md), so item 10's three arms measure the class. Record the forced forms in the receipts.

13. Documents, by append or exact replacement as §28.1 has it. `doc/Architecture.md`: one paragraph after the section retitled in item 7, headed "The deictic class (2026-10-31)", stating the representation, the four invariants of item 11 and their sites. `doc/Params.md`: nothing, since no parameter is added; say so in §31. `doc/Language.md`: the row in the word-class table stands; replace "proposed 2026-10-31" in it with "decided 2026-10-31" once the certificates pass. Receipts under `doc/benchmarks/2026-10-31-item2-5-completion/`, with the forced forms of item 12.
~~~

### 30.5 Alec's correction (2026-10-31, latest): no forcing; §30.4's items 11 to 13 withdrawn; addendum 3 revised

Alec: "This is Felix / that is Ciao should not do any forcing. Even 'This is a / this is b' will create two LTM1 relations that both involve 'this', which will cause a raise, but eventually the ambiguity of 'this' will cause its identity to be empty (meaning that it could be a symbol for anything)." This confirms the mechanism on the hash and withdraws the class §30.4 proposed: nothing marks a word deictic, no membership is forced, `define_pending`, `promote_identity`, `InterpretLayer.define` and `InterpretLayer.reading` are unchanged, and the lesson-13 harness forces nothing. Recorded in Orders §11 ("No forcing; the class is the raise's outcome"), FutureWork, and Language.md's word-class row.

The revised addendum follows. It replaces §30.4's fenced block in full.

~~~
Item 2.5, stage 2 completion round, addendum 3, revised (Alec, 2026-10-31, latest). Governing with §30.3 on one source hash; items 11 and 12 precede item 10's run and freeze. Results go in §31. §30.4's items 11 to 13 are withdrawn: if any part of them has been applied, revert it on the same hash and say so in §31.

11. No deictic class is built, forced or read. The mechanism is the one on the hash: a word's first kept definition mints the identity its mention named and defines the word to it (`define_pending`, `promote_identity`); later definitions with distinct relata write rows; at `raiseCount` distinct relata the identity acquires its class one order up with the relata as its members, and the word is defined to that class (`PartClosing._admit_member`, `ConceptIndex.index_part_row`); the class's content is the evidence-weighted centroid of its members (`FamiliarTier.cover_members`, `MereologicalCodes.derive_class_placements`), which prefers none of them. Change none of these sites. The lesson-13 harness runs as it is; nothing is forced. `doc/Language.md`'s word-class row is Claude's and is already rewritten to this reading; do not edit it.

12. One certificate, in `test/test_item25_completion.py`: `test_ambiguous_pointer_identity_empties`. With `raiseCount` at its default, three order-one concepts a, b and c whose rows share no nonzero coordinate and have equal mass, and the kept definitions "This is a. This is b. This is c.": after the first, "this" is defined to the identity its mention minted, and that identity's row with a is found by `parts.find`; after the second, the row with b is found and no class exists one order above the identity; after the third, the identity has its class one order up, a, b and c are its members, each exactly one order below (the one rule), and "this" is defined to the class. The class's content is the centroid of a, b and c: the share of it that each member covers is equal within 1e-6, and the share that a fourth concept d covers, d sharing no nonzero coordinate with them, is zero. Then the kept definition "This is d." adds d as a fourth member and each member's share falls to one quarter. Nothing in this certificate forces a part of speech. If the mechanism on the hash fails a step, stop and return the step by name as §28.1 says; do not repair it in this round.
~~~

### 30.6 Alec on the identity that empties (2026-10-31, latest still); the certificate of §30.5 item 12 replaced

Alec: "The reading that you want to check; the symbols and percepts are rooted in their centroids. I am talking about the identities across time, in conceptual space. If 'he' refers to Alec, and he becomes tired, then we have an MTM identity that unites the two, and has the parts and whole belonging to tired(Alec). So that is rooted in conceptual space, with a word line that corresponds to the (nominal) identity." The centroid stays a symbol's or percept's content; the emptying is of the identity across time (Orders §11, "The identity that empties is the identity across time"). §30.5's item 11 stands as written. Its item 12 is replaced by the following, on the same terms: one source hash, before item 10's run and freeze, results in §31.

~~~
12 (revised, 2026-10-31, latest still). One certificate, in `test/test_item25_completion.py`: `test_ambiguous_pointer_identity_across_time_is_empty`. With `raiseCount` at its default, three order-one concepts a, b and c whose rows share no nonzero coordinate, and the kept definitions "This is a. This is b. This is c.": after the first, "this" is defined to the identity its mention minted, and that identity's row with a is found by `parts.find`; after the second, the row with b is found and no class exists one order above the identity; after the third, the identity has its class one order up, a, b and c are its members, each exactly one order below (the one rule), and "this" is defined to the class. The identity across time is empty: the coordinatewise meet of the members' rows, what persists across the things "this" was, is zero. The control, beside it in the same file: after kept "A cat sleeps. It is black.", with the bind's selection of the cat's antecedent made as `test_part_closing.py::test_pronoun_at_occurrence_closes_verb_over_antecedent_kind_without_seating` makes it, one MTM token unites the two mentions, and the second sentence's kept row binds `_identity` to the first mention's address with the value `max(point, predicate)`, as that standing test reads it: the identity carries black(cat), rooted in conceptual space. No class content, centroid or share is asserted; the centroid is a symbol's content and is not this certificate's subject. Nothing forces a part of speech. If the mechanism on the hash fails a step, stop and return the step by name as §28.1 says; do not repair it in this round.
~~~

### 30.7 Alec's direction for pointing words: witness() with the unknown (2026-10-31, latest of all); a draft item, not in the round

Alec: "'this' or 'that' are just words for objects that do not find a match in MTM. We can allow that as witness() as we talked about, an alternative to interpret(), that will find a temporary perceptual reference and not conceptualize it. As I mentioned, though, it can't work until we have an alternate way of cognizing reality like vision; we can use a concept with zero perceptual content and maximal conceptual uncertainty (an unknown)." And: "'Worldline' is not a row, it is a physics term denoting a spatiotemporal occurrent that is the basis for identity." Recorded in Orders §11 ("Worldline, not word line; witness() and the unknown"). §30.5's item 11 and §30.6's item 12 stand; the round is unchanged. The item below is a draft for Alec's yes. It does not govern until he says so, and Codex does not apply it.

~~~
Draft, not governing (Claude, 2026-10-31, for Alec's yes). Item 2.5 or a later round: witness().

13. `witness`: a unary grammar operation beside `interpret` (`GrammarLayer`, `rule_name='witness'`, `arity=1`, space role SS, not mandatory), chosen by the compose chooser with ordinary R + A credit. No word list and no word class: no word is anchored to it, and `interpret` stays available for the same word in the same frame.
   - Forward takes an admitted word to a temporary perceptual reference. Without a perceptual field it opens a pending MTM token for the mention (`MediumTermMemory.open`, `kind=None`, `pending=True`, a transient id) whose value is the unknown: zero form content, both lanes at zero (Architecture.md 11c, complete uncertainty). It defines nothing: the word keeps no object at that mention, and the token is never promoted to a concept (`promote_candidates` and `define_pending` leave tokens opened by `witness` alone). With a field, later, the token's value is the witnessed percept's content and its location the percept's `.where` and `.when`.
   - A kept definition whose subject is a witnessed token writes its row between the other side and the occurrence, an OCCURRENCE reference, which the one rule of Orders §11 makes a member one order below: the occurrence joins the other side's identity as a part, no identity is minted for the token, and an unbound label on the other side is defined as today, so what was pointed at gets its name. No row has the pointing word's symbol as an endpoint.
   - Reverse writes back the word it was applied to, as `bind`'s does.
   - Certificates: after kept "This is Felix. That is Felix." with the chooser's selection set to `witness` at the two demonstratives (a selection in the test, not a word class), Felix's identity has the two occurrences as parts, "this" and "that" have no object, and no row has their symbols as endpoints; the control selects `interpret` at the same sites and gets the path on the hash, mint and later raise. Nothing forces a part of speech.
~~~

### 30.8 Alec's design for witness(): the null-concept (2026-10-31, night); the draft item 13 of §30.7 revised, not yet governing

Alec: "So witnessing is an alternative to interpreting that gives a perceptual identity a null-concept, a concept that is not limiting because it is unknowing. It must be a frozen concept, and it must be exempt from 'normal' conceptual operations that rely on conceptual definition. But it does give something for MTM to temporarily identify; there is a perceptual referent when Jesus says 'what am I like?', or when the zen master asks of an orange, 'what is this?'." Recorded in Orders §11 ("Witnessing gives a perceptual identity a null-concept"). This replaces §30.7's draft block. It governs when Alec says so, on one source hash with §30.3, §30.5 and §30.6; until then Codex does not apply it. The §31 stop is independent of it, and Claude's recommendation there stands: the occurrence-keyed row is the witnessed identity's row.

**What the hash has, and what changes (Claude, 2026-10-31, to Alec's question "is that concept of suchness constructively in the current architecture, and what needs to change").** Constructively present: a perceptual identity with no concept, the pending MTM token (`MediumTermMemory.open` with `kind=None`, `pending=True`, a transient id), which the reference machinery opens for an anonymous mention; the row keyed to the occurrence, written at the close before any minting (§31); two lanes that can both be zero, with certainty as lane length; a reserved NULL row in the codebook as the precedent for a reserved symbol; the percept path that expectation never subtracts from; and a chooser that selects operations by credit. Not present: `witness` as an operation, so the eager word boundary names an objectless word's token by the word (`InterpretLayer.reading`, `pool.named`), one token per word rather than per mention; the null-concept as a frozen, exempt symbol with no order, since every writer (`_populate_concept_weights`, `cover_members`, the fold, priming) writes any row it touches; the definition path, which defines the pointing word and mints an identity beside a known one (`define_pending`, `promote_identity`) instead of naming the token by the other side's label or absorbing the occurrence into the known identity; and the skip in `promote_candidates`. Those four are the change, and nothing in the percept path, MTM's structure, the rows or the one rule moves. Answering "what is this?" with the token's content rather than a name is a fifth, small and separable.

~~~
Draft, not governing (Alec's design, 2026-10-31, night; for his word). witness() and the null-concept.

13. `witness`: a unary grammar operation beside `interpret` (`GrammarLayer`, `rule_name='witness'`, `arity=1`, space role SS, not mandatory), chosen by the compose chooser with ordinary R + A credit. No word list and no word class: no word is anchored to it, and `interpret` stays available for the same word in the same frame.
   - The null-concept. One reserved concept id, allocated at construction beside the NULL row and never trained: form zero, both lanes zero, certainty frozen at zero. No evidence writes its row (`_populate_concept_weights`, `cover_members`, `derive_class_placements` and the definition fold skip it; `activate_code` never scales it). It is exempt from the definition index (`interpret.define` to or from it raises), the class cover, membership and the raise (`index_part_row`, `_admit_member`), the taxonomy and pervasion, the order projection (it has no order), priming and expectation learning on its row, and forgetting (never released). No row written at a sentence close has it as an endpoint.
   - Forward takes an admitted word to a perceptual identity: it opens an MTM token for the mention (`MediumTermMemory.open`) with the null-concept as its kind, its value the witnessed percept's content (zero while there is no field), and the mention's `.where` and `.when`. The pointing word keeps no object. Later mentions bind to the token as to any token; events attach to its timelines; it expires as any token does. `promote_candidates` skips tokens whose kind is the null-concept.
   - A kept definition resolves the token as a pending identity is resolved today, and never defines the pointing word (`define_pending` receives `word=None` for it). An unbound label on the other side names the token: "This is Felix." with Felix new promotes the token under the name Felix (`promote_identity` with the label's word). A known identity at the token's order absorbs the occurrence as its `('occ', ...)` part and no identity is minted. A kind on the other side gives the token that kind, as "a" mints. The row is written between the occurrence and the other side, and the one rule of Orders §11 makes the occurrence a member one order below.
   - Reverse writes back the word it was applied to, as `bind`'s does.
   - Certificates, in `test/test_item25_completion.py`, with the chooser's selection set to `witness` at the pointing word (a selection in the test, not a word class): (i) "This is Felix." with Felix new: one token, kind the null-concept before the close; after the close one identity named Felix whose part is the occurrence, "this" with no object, no row with the null-concept or with `(SYMBOL, this)` as an endpoint, and the null-concept's row unchanged byte for byte. (ii) "This is Felix. That is Felix." with Felix known from (i): Felix has two occurrence parts, no second identity. (iii) "What is this?" after (i): the question's "this" binds to the latest token and the supplied answer reads the token's content, not a name. (iv) The control selects `interpret` at the same sites and gets the path on the hash, mint and later raise. Nothing forces a part of speech. A fail-loud check that the null-concept's row and lanes are unchanged runs at every stage 2 freeze.
~~~

**§30.8, one bullet added (Alec, 2026-10-31, night, closing).** Alec: "this, as an indexical, takes a temporary definition based on a perceptual centroid, has a null or uncertain concept, and cannot be reasoned about because it has no conceptual content. So it is a temporary finger pointing at the moon, a name that does not exist beyond its use." The draft block above gains this bullet under item 13, between the reverse and the certificates, and is otherwise unchanged:

~~~
   - Not reasoned about. No thought operation reads a witnessed token's concept: pervasion, the definition fold's judgment, `intuit`'s rule selection and the symbolic order loop treat it as having none. It can be bound again, carry events on its timelines, and be named or kinded by a definition; it is never inferred over, never consolidated as itself, and it expires from MTM as any token does. The temporary definition it carries is its value, the witnessed percept's centroid, zero while there is no field.
~~~

## 32. Review of Codex's §31 return; the stop's disposition; addendum 4: witness() and the null-concept join the round (Claude, 2026-10-31, night)

**The return, verified.** Codex's §31 is in the working tree, uncommitted. On its source `20a0fb503590e4cd9cc1ae04ad10075d33259dd31f63fe5799a347bdac4a96ec`: the hash reproduces from `protocol.snapshot()` on the tree as it stands, and the frozen snapshot's 971 files match the tree with no mismatch. The claim that the §30.4 work is withdrawn byte for byte holds: `FamiliarTier.py`, `Interpret.py`, `PartClosing.py`, `Spaces.py`, `corpus_lessons.py`, `Models.py`, `ReferenceClose.py`, `ConceptIndex.py`, `DefinitionFold.py` and `data/model.xml` carry the same hashes as the reviewed candidate `30f6e1613555`; no "deictic" remains in `bin/`, `test/`, Architecture.md, Params.md or the receipts; Language.md equals main. `Layers.py` differs from the candidate by item 7 alone: `ConceptAllocator.class_member` (a member exactly one order below its whole, true when either order is unrecorded), the `records` guard that reads it and requires positive evidence, the reflexive guard, `order_only` absent, and `_part_order` as on main. The order-boundary certificate is `test_order_boundary_membership_follows_one_rule` and asserts the one rule at order zero and membership one below. The word-chain test is rewritten as item 8 says, with the disagreement assertion dropped, and the cycle-fixture patch is applied to `test_item25_repair.py`. Architecture.md's section is retitled "Class membership and the counted raise (2026-10-31)" and opens with the one rule including order zero; readings.md line 3 is corrected. Params.md is unchanged by the addendum. The §30.6 certificate fails at its first step exactly as reported: kept row 11 has endpoints `((1, -6290487884758523310), (2, 2))`, the subject's first occurrence and a, and no row joins symbol 12 to 2. The control passes. On the same tree, `test/test_item25_completion.py` without the pointer certificate, the word-chain test and the cycle test give 15 passed. Approved; the stop was the right call under §28.1.

**The disposition (Alec, 2026-10-31: "Ok, accepted").** The occurrence-keyed row is the witnessed identity's row: what "This is a." equated with a is the subject's first occurrence, a point on a worldline, and the identity minted from it holds that occurrence as its part. Nothing is missing on the hash. The certificate's first step was written against the wrong endpoint and is revised below; the run continues on the same hash.

**Addendum 4 (Alec, 2026-10-31: "Everything discussed can go to the current 2.5 implementation").** §30.8's draft item 13 governs, with three decisions written in by Claude for Alec to overrule if he wishes: the choice point for `witness` is a reference mode on a bare noun-phrase operation, since `interpret` is the eager, mandatory word-boundary read and the compose chooser is the only credit-bearing choice; the null-concept has no order; and certificate (iii) is corrected to lesson 13's measurement, so the "fifth, separable change" of §30.8 is withdrawn. Results go in §33.

~~~
Item 2.5, stage 2 completion round, addendum 4 (Alec, 2026-10-31, night). Governing with §30.3, §30.5 and §30.6 on one source hash. Order: item 12 as revised here, item 13, then item 10. Results go in §33. Stop and return early only as §28.1 says.

12 (first step revised; the rest as §30.6). The occurrence-keyed row is right. After the kept "This is a.", look the row up by the subject's first occurrence reference, the mention's address, and a: it is found by `parts.find`, "this" is defined to the identity minted from that occurrence, and that identity's parts include the occurrence. After "This is b." and "This is c.", where "this" now denotes the identity, the rows are between the identity's symbol and b, and the identity's symbol and c, and at the third the identity acquires its class one order up with a, b and c as members, "this" is defined to the class, and the coordinatewise meet of the members' rows is zero. The control stands as it passed.

13. `witness` and the null-concept (Orders §11, "Witnessing gives a perceptual identity a null-concept" and "The finger and the moon").
   - The choice point. `witness` is a reference mode, `'witness'`, declared by a bare noun-phrase unary operation `witness_unary` beside `lower_unary` and `definite_unary` (`MarkerUnaryLayer`, `rule_name='witness_unary'`, `determiner_mode='witness'`), selected by the compose chooser with ordinary R + A credit. No word list and no word class: no word is anchored to it, and the operations that mint or bind stay available for the same word in the same frame. `interpret` stays the eager word-boundary read; for a word with no object it returns its open handle as today. `bind` is unchanged: no word class participates in its read, and a mention under `'witness'` takes no antecedent.
   - The null-concept. One reserved concept id, allocated at construction beside the NULL row and never trained: form zero, both lanes zero, certainty frozen at zero. No evidence writes its row (`_populate_concept_weights`, `cover_members`, `derive_class_placements` and the definition fold skip it; `activate_code` never scales it). It is exempt from the definition index (`interpret.define` to or from it raises), the class cover, membership and the raise (`index_part_row`, `_admit_member`), the taxonomy and pervasion, priming and expectation learning on its row, and forgetting (never released). It has no entry in `reference_orders`; `settle` and the order projection skip it, and a caller that asks its order fails loud. No row written at a sentence close has it as an endpoint.
   - The token. At the kept close, a mention under `'witness'` opens its own MTM token (`MediumTermMemory.open`) with the null-concept as its kind, its value the witnessed percept's content (zero while there is no field), and the mention's `.where` and `.when`. It is found thereafter by the mention's reference, never by `('name', word)`, so two mentions of "this" share no token unless a later bind joins them. The pointing word keeps no object. Later mentions bind to the token as to any token; events attach to its timelines; it expires as any token does. `promote_candidates` skips tokens whose kind is the null-concept.
   - A kept definition resolves the token as a pending identity is resolved today and never defines the pointing word (`define_pending` receives `word=None` for it, and the token is found by the left reference). An unbound label on the other side names the token: "This is Felix." with Felix new promotes the token under the name Felix (`promote_identity` with the label's word). A known identity at the token's order absorbs the occurrence as its `('occ', ...)` part and no identity is minted. A kind on the other side gives the token that kind, as "a" mints. The row is written between the occurrence and the other side, and the one rule makes the occurrence a member one order below.
   - Not reasoned about. No thought operation reads a witnessed token's concept: pervasion, the definition fold's judgment, `intuit`'s rule selection and the symbolic order loop treat it as having none. It can be bound again, carry events, and be named or kinded by a definition; it is never inferred over and never consolidated as itself.
   - Reverse writes back the word it was applied to, as `bind`'s does.
   - Certificates, in `test/test_item25_completion.py`, with the chooser's selection set to `witness_unary` at the pointing word (a selection in the test, not a word class): (i) "This is Felix." with Felix new: one token with the null-concept as kind before the close; after the close one identity named Felix whose part is the occurrence, "this" with no object, no row with the null-concept or with `(SYMBOL, this)` as an endpoint, and the null-concept's row unchanged byte for byte. (ii) "This is Felix. That is Felix." with Felix known from (i): Felix has two occurrence parts and no second identity exists. (iii) "What is this?" after (i): the question's "this" binds to the latest token, which the definition named Felix, so the supplied answer is Felix, as lesson 13 measures; after a witnessed mention that no definition named, the answer channel yields no concept. (iv) The control selects the minting operation at the same sites and gets the path on the hash, mint and later raise. Nothing forces a part of speech. A fail-loud check that the null-concept's row and lanes are unchanged runs at every stage 2 freeze.
   - Documents, by append or exact replacement: one paragraph in `doc/Architecture.md` after "Class membership and the counted raise (2026-10-31)", headed "Witnessing and the null-concept (2026-10-31)", with the sites above; `doc/Params.md`: nothing, since no parameter is added; say so in §33. Receipts under `doc/benchmarks/2026-10-31-item2-5-completion/`.

10. Then run and freeze as §30.3 item 10 says, with items 12 and 13 in place, so the three lesson-13 arms measure the round with `witness` available to the chooser.
~~~

### 32.1 The choice point, corrected (Alec, 2026-10-31, night): witness is interpret's sibling, not a reference mode

Alec: "Interpret is a chooser option, not all perceptual content is symbolic. And an alternative to that is Witness. We might also do neither, but for the case of written text (reading mode) we are forcing a few things. Witness is useful because it binds to an uncertain concept, which is otherwise unlikely (we do not want an object-concept that corresponds to the word-concept 'this', which should remain a referent to be filled by perception)." The first bullet of item 13 in §32 is replaced by the following; `witness_unary` and the `'witness'` reference mode are withdrawn. Everything else in item 13 stands, with "the chooser's selection set to `witness` at the leaf" read for the certificates.

~~~
13, first bullet, revised (Alec, 2026-10-31, night).
   - The choice point. `witness` is `interpret`'s sibling at the same site: a unary rule over a leaf, declared beside `interpret` in the grammar files (`data/default.grammar` and `data/complete.grammar`: `witness_argout1 = witness.forward(witness_argin1)` under compose and `witness_argin1 = witness.reverse(witness_argout1)` under generate), registered as `interpret` is (`GRAMMAR_LAYER_CLASSES['witness']`, the builtin layer on `ConceptualSpace`), as `WitnessLayer(GrammarLayer)` with `rule_name='witness'`, `arity=1`, `space_role='SS'`, `invertible=True` and `mandatory=False`. The three readings of a percept are interpret, witness and neither. In reading mode the forcing stands as built: `interpret` is mandatory at the word transaction, so every word is interpreted before composition and neither is excluded; `witness` is the chooser's alternative for the same leaf, applied in the derivation to a word whose interpretation yielded an open handle, taking that handle to the witnessed token with the null-concept as its kind. Outside reading mode, with a field, the chooser may interpret, witness or leave a percept alone. Credit is the ordinary R + A credit of any chosen operation; no word list and no word class; `bind` unchanged. The reason, Alec's: `witness` binds the mention to an uncertain concept, which the chooser is otherwise unlikely to do, and there must be no object-concept for the word-concept "this", which remains a referent to be filled by perception.
~~~

### 32.2 Alec's decisions on the two older opens (2026-11-01); addendum 5, items 14 to 21: resolve everything before accepting 2.5

Alec on open 1: "Please ignore the previous decision D-iii. You can implement everything and xfail what tests you need to, let's lock down what is working going forward." On the seven audit deviations of 2026-10-30 (handoff, "Doc-vs-code audit"): (1) "No, expectation should not be calling grammar operators, only selecting active percepts and concepts, the latter for the parallel pass only." (2) asked what the free-percept term means; Claude's explanation and proposal are in item 16. (3) "Fix it, then; remove the unused columns." (4) "Forgetting had been checking if rows were easily deductible; let's delete that requirement." (5) "Water is a count noun with an elided form/shape in some contexts, so it's not a good example of a mass noun. Water would be order-1 at lowest, since it is an abstraction of an event." (6) asked what "relevance" means; Claude's explanation and proposal are in item 20. (7) "OK". And: "All of these should be resolved before accepting 2.5." Items 18 and 19 are document corrections and Claude applied them (Reasoning.md, Orders §7); the rest are Codex's.

~~~
Item 2.5, stage 2 completion round, addendum 5 (Alec, 2026-11-01). Governing with §30.3, §30.5, §30.6, §32 and §32.1 on one source hash. Order: items 12 and 13 as §32 and §32.1 have them, then items 15 to 21, then item 14's run and freeze. Results go in §33.

14. D-iii is withdrawn; lock down what works. The one-backward path and round B's work are accepted as implemented. The three lesson-13 arms run as measurements, not as gates. The configured defaults stay `separateChooser=true` and `reconstructionJitter=0` until Alec changes them, no longer tied to the arms. §28.1's early return is relaxed: a standing test or certificate that fails on a deviation this addendum does not resolve is marked `pytest.mark.xfail(strict=True, reason=...)` with the reason naming the deviation, listed in §33, and the round proceeds; stop only for a source-hash mismatch or a numerical divergence, which stays fail-loud. Then run the standing selection, the completion certificates, thought, core with the native worked document, ports-and-finite, smoke and the three arms, and take the stage 2 freeze with the xfails recorded in the receipts.

15. Expectation calls no grammar operator (audit 1). Remove `expectation.reference_mentions` at both sites: the mention loss the bind resolver registers under expectation (`ReferenceClose.py`, the `errors.add('expectation.reference_mentions', ...)` in the resolver's mention accounting) and the supplied answer's reference loss registered under expectation in `Models.py` (`registry.add('expectation.reference_mentions', ...)`). Bind's choices receive the sentence or thought episode's R + A credit through the ordinary departure rule, as Architecture.md's identity amendment says; the supplied answer's loss belongs to output, which owns its readers (Training.md). Expectation's remit is selection: the predicted next state selects active percepts on both paths and active concepts on the parallel pass only; on the serial path expectation selects no concept. If a serial-path consumer reads the concept expectation today, remove that read and name it in §33. Architecture.md gets one paragraph, "Expectation selects; it calls no grammar operator (2026-11-01)", by append.

16. The free-percept term (audit 2; Claude's proposal, Alec may overrule before Codex reaches it). `reconstruction.free_percept` is the free decoder's percept-residual identity: the generate walk decodes the understanding back to the sentence's leaves, and this term compares that decode with the leaves' percept residual, what the percept expectation did not predict, at `reconstructionScale` with the trial root live. Its byte sibling `reconstruction.free_bytes` is reporting only and decides which decoder walk is kept. The design had the percept term as an audit; the build trains it, and GradientFlow.md was corrected on 2026-10-30 to say so. Proposal: keep it trained, since it is the only reconstruction pressure that reaches the percept residual through the free decoder, and treat the 2026-10-30 correction as the record. No code change.

17. LTM0's columns (audit 3). Bring `LTM0` to the amended row layout: a row holds its slots, `.when`, the two truth lanes, the access count and `surprise`, and nothing else. `c_plus` and `c_minus` become `truth_plus` and `truth_minus` as on LTM1; `witness_count` becomes `access_count`. `source_trust`, `order`, `row_ids`, `origin`, `record_kind` and `polarity` are removed from `LTM0`. A reader of a removed column is redirected: a row's order is read from the recorded orders of its referents, its kind and polarity from the referents as the amended LTM1 row design has it, and a reader with no such replacement is returned by name in §33 rather than kept on a column. Checkpoints in the old layout are not loaded; loading one fails loud with the column name. Every use of the removed names in `bin/` and `test/` goes; `rg` them and show zero in the receipts.

18. Deducibility (audit 4): deleted as a requirement. Reasoning.md's "Retention utility" now states that retention reads the recorded surprise, the same-scope dissonance flag and the optional coverage term, with no deducibility pass and no deletion certificate; the forgetting spec keeps its text as the record with a dated note. Applied by Claude on 2026-11-01; nothing for Codex.

19. "The water" (audit 5): a document correction, applied by Claude on 2026-11-01 in Orders §7. Water is a count noun with an elided form or shape in some contexts, not a mass-noun example, and is order 1 at lowest, an abstraction of an event; "the water" selects a shaped portion, an individual at order 1, as §7 already said. The build, which lowers only from order 2 and leaves an order-1 term at 1, agrees; no code change.

20. Relevance (audit 6; Claude's proposal, Alec may overrule before Codex reaches it). "Relevance" is Architecture.md section C's law, relevance equals origin times axis, realized as one priming surface per space: rows seen are primed by being perceived, rows desired or hated by signed intent, and the conceptual surface projects onto the pyramid's inventory rows as a ranking score, boost minus one, the priority read. `<architecture><relevance>` was meant to gate that integration. As built the seen priming writes are unconditional (Alec, 2026-07-12), the priority read runs unconditionally on the parallel path and never on the serial path, and the knob is parsed into `relevance_on` which nothing reads. Round B then made `ActivationPredictor`'s expectation what attention, the chooser and retrieval read. Proposal: retire the knob, removing `<relevance>` from `data/model.xml`, `model.xsd`, `relevance_on` and the Params.md row, and record the priority read as built, parallel path only, in Architecture.md by exact replacement of the audit note. No behaviour change.

21. `whereRungRatio` and `ltmConsolidation` (audit 7): the build stands. Params.md's rows for both say live, with their readers (`Models.py` for `whereRungRatio`; `ltmConsolidation` gates the serial word-major path), by exact replacement of the retired wording and the audit notes.
~~~
