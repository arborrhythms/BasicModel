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

### 32.3 The three readings forced: no-op, sign, interpret; `witness` renamed `sign` (Alec, 2026-11-01); addendum 6, item 22

*Refined the same day in §32.4 below, which revises item 22; where the two disagree, §32.4 governs.*

**The decision.** Claude's reading of 2026-11-01, in reply to the Dharmakīrti analysis: `witness` as built is called on a word's handle, `WitnessLayer.forward` returns the word, and the kept close in `ClauseJournal.py` writes zero content for a word and the leaf's value only for a native percept, so the operation never reaches the percept and is `interpret` with no object. The clear distinction is the attitude toward the percept, taken as itself or as standing for something else, and Peirce's icon, index and symbol sort the words: an onomatopoeia is an icon, "this" an index, an ordinary word a symbol. In reading mode the text is the field, so a percept taken as itself has content, the word's form, and "zero until there is a field" holds only for an index whose object is absent. Alec: "Yes, please record this, and let's force a three way decision of an object (no op, sign(), or interpret() are the icon, sign, and symbol), so that the confusing term witness is replaced with pierces SIGN." Recorded in [Orders §11](../Orders.md), "The three readings forced"; [Architecture](../Architecture.md), the bullet after "Pointing words"; [Language](../Language.md), the amendment under `interpret`; [FutureWork](../FutureWork.md), "Renamed `sign`"; and [Philosophy](../Philosophy.md), "Witnessing Suchness", closing paragraph. Peirce's name for the middle relation is the index and "sign" is his genus; Alec names the operation `sign`, and the documents and the code follow him.

**Addendum 6, item 22, for Codex.** Joins the completion round before item 14's run and freeze, so that the freeze locks `sign` and the no-op's content rather than `witness` and zero. Everything in item 13 stands except as stated here. This text specifies the behaviour and names the sites; Codex reports the mechanics in §33.

~~~
22. The three readings forced; `witness` renamed `sign` (Orders §11, "The three readings forced").
   - The rename. `witness` the operation becomes `sign` everywhere it is the operation and nowhere else: `WitnessLayer` to `SignLayer` with `rule_name='sign'` (`Interpret.py`); the four grammar rules in `data/default.grammar` and `data/complete.grammar` (`sign_argout1 = sign.forward(sign_argin1)` under compose, `sign_argin1 = sign.reverse(sign_argout1)` under generate); `GRAMMAR_LAYER_CLASSES['sign']`, `builtin_layers['sign']` and `ConceptualSpace.sign` (`Language.py`); `witness_leaf` to `sign_leaf` and `witnessable` to `signable` (`ClauseJournal.py`, `PartClosing.py`, `ReferenceContext.py`, `ReferenceClose.py`, `Language.py`) and every `method_name == 'witness'` test at those sites; the certificates and helper of item 13 in `test/test_item25_completion.py`; the receipts; and Codex's Architecture.md section, retitled "Sign and the null-concept (2026-10-31; renamed 2026-11-01)" by exact replacement of its own text. "witness" in every other sense, the evidence witnesses of rows, members, spans and `_witnessed_rows`, is untouched: a grep for `witness` in `bin/` after the round finds only those.
   - The forced decision. At every leaf, in every mode, the compose chooser's options are the no-op, `sign` and `interpret`, exactly one selected per leaf, ordinary R + A credit, no word list and no word class. `interpret` is no longer mandatory on its own: `InterpretLayer.mandatory`, the `rule.mandatory and rule_name != 'interpret'` exclusion in the chooser catalog (`Language.py`) and the reading-mode comment in `ReferenceContext.py` are Codex's to resolve; the no-op is available in reading mode; `sign` is available at every leaf and not only at a pending one (the `signable` mask, today the leaf's pending identity, no longer restricts it). The eager word transaction is unchanged: it admits the form, reads an existing object and mints nothing (`Interpret.py`: the form owns a row before it has an object; interpretation cannot invent it). It is not the decision. Under `interpret` a pending word is minted at its first kept definition as today.
   - The no-op. Its leaf value is `interpret_I1`, the word-concept code the word arrives as, not the object's activation: a no-op leaf publishes that code. At the kept close a no-op leaf opens its own occurrence's MTM token (`MediumTermMemory.open`) with that code as its content whether or not the leaf is a word (the branch that writes zeros for a word applies to `sign` alone), with the null-concept as its kind, at the mention's `.where` and `.when`. Later mentions bind to it, `define_pending` names or kinds it, `promote_candidates` skips it, and it expires as any token does, exactly as item 13's token. Its reverse is the form itself.
   - `sign`. Item 13's mechanics under the new name: the token is the pointed-at occurrence, its content zero in reading mode and the pointed-at percept with a field, its kind the null-concept.
   - Certificates, in `test/test_item25_completion.py`, with the chooser's selection set in the test, not by a word class. (i) to (iv) of item 13 under the new name. (v) "The dog went woof. That was a bark." with the no-op selected at "woof": before the first close, one token whose kind is the null-concept and whose content equals the published word-concept code of "woof", compared exactly and nonzero; after the second close, "that" bound to that token and the token kinded bark, as a kind on the other side kinds the token in item 13 (4); no object minted for "woof" and no DEF row joining "woof" to an object; the null-concept's row unchanged byte for byte; and the reverse of the no-op leaf equal to the form of "woof". Its control selects `interpret` at "woof" and reads or mints the word's object as today. (vi) In a kept reading-mode derivation every leaf carries exactly one of the three readings, and at a known word's leaf all three were available to the chooser; a fail-loud check of the same at every stage 2 freeze.
   - Documents, by append or exact replacement of Codex's own text only: the Architecture.md section above; Params.md nothing, since no parameter is added, say so in §33; receipts under `doc/benchmarks/2026-10-31-item2-5-completion/`. Results in §33, with item 14's three arms measured with the three readings available.
~~~

### 32.4 Refined the same day: `interpret` optional and initially forced, its interpretation a sign or a symbol (Alec, 2026-11-01); item 22 revised

**The refinement.** Alec, on §32.3: "I think the better way to implement this is for interpret() to be optional, and if it happens, then the interpretation may be a sign or a symbol. Again, interpretation can be initially forced since words are so often meaningful beyond themselves." So the three readings are two decisions, one above the other, not one flat choice. The first is whether to interpret the percept: `interpret` is optional in the architecture, and a leaf at which it is not applied is the no-op, the icon, standing for itself with its form as its value. The second is taken only when the first interprets: the interpretation is `sign`, the index, the pointed-at occurrence's token with the null-concept as its kind; or the symbol, the word's conventional object, the interpreted leaf left standing as that object, which needs no operation of its own. Both are the compose chooser's by ordinary R + A credit, anchored to no word; the first is initially forced in reading mode, as built: `interpret` mandatory at the word transaction, so the chooser's decision in reading mode is the second one. This is closer to the build than §32.3 was. In reading mode `interpret` is mandatory at admission and `witness` is the chooser's alternative on the interpreted leaf; in a field the chooser may already interpret, witness or leave the percept alone (`ReferenceContext.py`, the `interpret` options under `not bank.close_references`). The two-level structure is the build's; what changes is the name, the second decision's availability, and the no-op's close. Recorded by exact replacement of Claude's own paragraphs of §32.3's record in [Orders §11](../Orders.md), [Architecture](../Architecture.md), [Language](../Language.md), [FutureWork](../FutureWork.md) and [Philosophy](../Philosophy.md). Claude's write-ins, for Alec to overrule: the forcing is `InterpretLayer.mandatory` together with the reading-mode branch that gives `interpret` no chooser option, and no parameter is added this round, the certificate lifting it on the layer; `sign` is available at every interpreted leaf; the symbol is the abstain of the second decision, and no `symbol` layer or rule is declared.

**Item 22, revised.** §32.3's item 22 stands except as follows; where the two disagree, this text governs.

~~~
22, revised (Alec, 2026-11-01, later the same day).
   - The rename stands as §32.3 states it, with one site added: the `_reference_leaf` buffer in `Language.py` (`('interpret', 'witness')` becomes `('interpret', 'sign')`).
   - The decision is two decisions. First, at every leaf, whether `interpret` is applied; a leaf at which it is not is the no-op. Second, at every interpreted leaf, whether the interpretation is `sign` or the symbol, exactly one: `sign` is the unary rule applied to the interpreted leaf in the derivation, as `witness` is today, and the symbol is the interpreted leaf left standing as its object, with no rule and no layer. Both decisions are the compose chooser's by ordinary R + A credit, no word list and no word class.
   - The first decision is initially forced. §32.3's "`interpret` is no longer mandatory on its own" is withdrawn: `InterpretLayer.mandatory` stays `True`, the `rule.mandatory and rule_name != 'interpret'` catalog exclusion stays, and the reading-mode branch of `ReferenceContext.py` that gives `interpret` no valid option stays, so in reading mode every word is interpreted at admission as today and the chooser takes the second decision only. No parameter is added. The forcing is lifted inside certificate (v) alone, on the layer instance; the lifted configuration is reading mode with `interpret` a chooser option as in a field today: a leaf arrives as its form, the chooser's `interpret` takes it to its object, and declining is the no-op.
   - The second decision is available at every interpreted leaf: the `signable` mask (today `witnessable`, the leaf's pending identity in reading mode) no longer restricts `sign` to a leaf whose interpretation left an open handle; `sign` is valid at every interpreted leaf in both modes, and the chooser learns where it pays.
   - The no-op, as §32.3 states it, built behind the forcing: a leaf at which `interpret` is not applied publishes its form, `interpret_I1`, the word-concept code; at the kept close it opens its own occurrence's MTM token (`MediumTermMemory.open`) with that code as content, whether or not the leaf is a word, with the null-concept as its kind, at the mention's `.where` and `.when`; later mentions bind to it, `define_pending` names or kinds it, `promote_candidates` skips it, it expires as any token does, and its reverse is the form itself.
   - `sign`, as §32.3 states it: item 13's mechanics under the new name.
   - Certificates, in `test/test_item25_completion.py`. (i) to (iv) of item 13 under the new name, with the forcing on. (v) "The dog went woof. That was a bark." with the forcing lifted on the layer for the test and the chooser's selection set to the no-op at "woof": the checks of §32.3's (v) unchanged; its control, with the forcing on, interprets "woof" at admission and the chooser selects the symbol, reading or minting the word's object as today. (vi) replaces §32.3's (vi): in a kept reading-mode derivation with the forcing on, every leaf is interpreted and carries exactly one of the two readings, `sign` or the symbol, and at a known word's leaf both were available to the chooser; with the forcing lifted, the no-op was available at every leaf; a fail-loud check of the forced configuration at every stage 2 freeze.
   - Documents, as §32.3 states them. §33 reports the mechanics, including how the forcing is lifted for (v) and that no parameter was added.
~~~

## 34. Review of the §33 return (Claude, 2026-11-01): items 12, 13 and 15 to 22 built; the stop stands; stage 2 not frozen

**Scope and environment.** Reviewed on source `c17a9350596bcacc22bba1b4f570e7be428fdb251f3c975100123a1ac89236a4`, the working tree over `069df3fbff`, by reading the diff against §§32 to 32.4 and by targeted runs. The runs here are on a venv rebuilt during the review (torch 2.15.0, executorch 1.5.1; `make test` triggered the rebuild, Claude's doing, and it is recorded so it is not repeated); Codex's receipts are from the venv before it. The full suite on the rebuilt venv: `make test`: 7217 selectors, 82 distinct failures, 2 xfailed, 2 xpassed (`output/tests/20261101-111946-0923ae`). Codex's gates were selections (standing 110, certificates 90, thought 287, core 1674, ports and finite 116), not the suite; the suite is the criterion (Testing.md), and under the freeze rule every failure on an unresolved deviation needs a strict xfail naming it, so the freeze could not have been taken on this source even without the stop. Of the 82: 26 are dead relative links in the untracked benchmark receipts (`test_doc_links`), 24 of them in the receipt directories of the last week. Eight were the rebuilt venv's pytest 9.2, whose `monkeypatch` undo deletes an attribute the target's own dict never held: `test_sweep_compile_repair` patched `torch._dynamo.config.recompile_limit`, whose keys live behind `__getattr__`, so the undo deleted the key and the two later compile tests in that worker found it missing; `test_math_chain` patched `ensure_microbatch` on `SymbolSpace`, which forwards writes to its coordinator, so the undo raised and left `ModelAttention.stage_input` patched for the worker, failing that test, the three `test_math_chain_section14` cases and the two `test_medium_term_memory` cases after it. Both are fixed in `951c713a4c` (SymbolSpace gains the mirror `__delattr__`; the test uses torch's `config.patch`), and the eight pass together on torch 2.15.0, which is the environment from today by Alec's decision: the latest stable torch, nothing pinned, what the upgrade breaks fixed in the code. The remaining 48 were outside Codex's selections, among them six that read as tests not updated for this round's changes (`sign` as a unary rule in the catalog: `test_bounded_stm_fold`, `test_word_loop_buckets_and_orders`; the generate cost per lane: `test_output_walk`, `test_generate_repair`; `_observe_native_cases` removed: `test_primitive_properties`; conceptual width 368: `test_wholespace_property_migration`), three where idempotent.xml's learning rate of zero reaches `SparseAdam` because this round adds the intuition parameters to the sparse set (`Models.py`, `sparse_ptrs.update(... intuition.parameters())`), and the rest untriaged. The 48 non-link failures are Codex's to triage on the next return: a regression is fixed, a stale test is updated with the change named, an unresolved deviation gets its strict xfail; the dead links are repaired or the receipts pruned..

**Item 22, built as specified.** The points of §32.4 were checked one by one: the rename at every named site, including the `_reference_leaf` buffer (`Language.py` 6711); `SignLayer` with `rule_name='sign'` and `mandatory=False` (`Interpret.py` 77 to 95); the four grammar rules; the forcing kept (`InterpretLayer.mandatory=True` at `Interpret.py` 102, the catalog exclusion at `Language.py` 11361, the reading-mode branch at `ReferenceContext.py` 661); lifted only on the layer instance in certificate (v); `sign` valid at every interpreted leaf in both modes, with no pending-handle mask (`Language.py` 12798: `signable = (leaf & forced) | INTERPRETED_LEAF`); the no-op publishes the word's own row (`Interpret.py` 501) and at the kept close opens its occurrence's token with that form as content, the null-concept as kind and the mention's time and place (`ClauseJournal.py` 119, `ReferenceClose.py` 1225), while the zero-content branch applies to `sign` alone (`ClauseJournal.py` 280); the reverse is the form; no parameter (nothing sign-related in model.xml or model.xsd); `define_pending` names, joins or kinds a null token and `promote_candidates` skips it. `test/test_item25_completion.py`: 25 passed. Loose ends, none blocking: (a) the receipts still say witness in the operation sense (readings.md lines 17 and 19; the `section-32-witness-*` directories); (b) certificate (vi) is a unit and menu-level test plus the in-model fail-loud `leaf_readings` and the freeze script's assertion, with no end-to-end kept reading-mode sentence, and the freeze check has not run because the freeze was not taken; (c) "the eager and compiled chooser paths agree" is dynamo capture with the eager backend, not inductor; (d) the null-concept byte-zero check reads three dedicated zero buffers outside the inventory, so the protection is the exemption raises, which certificate (i) exercises; (e) Params.md byte-equal to its amendment baseline: verified, and the claim covers item 22 alone.

**Items 14 to 21.** Item 15 built: both `expectation.reference_mentions` registrations are gone, bind's departure credit is SentenceCredit's ordinary rule, and the serial-path concept-expectation read is removed, which §33 does not name. Item 16 built with no change needed: `reconstruction.free_percept` is registered as a reconstruction error (`Models.py` 20324), as at HEAD. Item 17 built for the row: six buffers removed, two renamed, `truth_schema` 14, the loader names the obsolete column, no remaining use in bin or test. Its handling, recorded here since §33 does not name it: `origin` is not removed but re-homed as a persisted per-document authority (`addressing.source_authorities`, `_document_sources`, `Layers.py` 8884 and 8939), which is what the new Architecture section says ("the input document owns supplied-fact authority") and leaves the row with nothing else; `kind_of` (`Layers.py` 9067) derives the fact, observation or inference classification from provenance and, for direct writes, from the kind encoded in the document-key digest; the sentence-level `trust` write at HEAD `ClauseRow.py` 531 is deleted with no replacement; and the readers of the removed columns are listed only in the archived checkpoint review (`through-section-32-2/review.md`), not in §33 as item 17 asked. Item 20 built: `relevance_on`, the xsd element, the Params row and the Architecture audit note are gone; model.xml never had the element. Item 21 built: the two Params.md rows (563 to 564) name the live readers (`Models.py` 6676, 10208, 25409); no test reads `whereRungRatio`, so "covered by the bounded selection" rests on the code site alone. Items 18 and 19 are Claude's document edits, at HEAD. Housekeeping: `ReferenceClose.mention_loss` and `answer_loss` are dead helpers. Item 14: the xfail rule is built, with two strict xfails that each name their deviation (`ClockDependentBinding`, `test/test_item55_mechanisms.py` 172; the runtime-conditional `UnsupportedReversePremiseBinding`, `test/test_math_chain_section14.py` 442); the measurement stopped at the numerical gate, the (false, 0.5) arm never started, and the freeze is not taken.

**The numerical stop.** The stop is the rule working: a finite surrogate, a non-finite gradient, a raise before the step. The capture names the parameter and the surrogate and not the producer; the optimizer preflight (`Optimizer.py` 236 to 251) raises at the first non-finite parameter in optimizer order, so other parameters may also have been non-finite, and the per-objective gradient report already in memory (`SentenceTraining.py` 147 to 171) was not written. Two findings narrow it. First, the documented mechanism is excluded: the run's `Generative.py` hash equals the working tree's, where `absolute_residual` (lines 36 to 49) has no denominator and `relative_residual` and `_relative_energy` no longer exist, so the live-denominator derivative of GradientFlow.md lines 53 to 62 cannot have produced this stop, and that text and §25.1 now describe a mechanism removed before the run. Second, the signature is structural: the weight is 392 by 29 (`WHAT_CONTEXT_DIM = 29`, `Language.py` 6408), and 928 = 32 x 29 is exactly one meaning lane of the conditioned root's cotangent, the form band finite. The parameter's only consumer is the comparison reader's `functional_call` of `_sentence_reader_error` (`AnswerComparison.py` 44 to 57), which runs `reverseOutput` and `_condition_answer_on_question` (`Models.py` 12655 and 9227) into the output walk; the paths into it are A (`_reverse_event_loss`, `Models.py` 20646), R (the comparison walk's own `walk_residual`, registered lane by lane, `Models.py` 9643 and 20498) and the first-menu edge through every rejected inverse image, `(options.sum() + others.sum()) * 0` (`Models.py` 13253; `Generative.py` 482). Under `separateChooser=false` the idea is not detached (`Models.py` 20623). The favoured producer is a zero times infinity in backward through a zero-weighted or `torch.where`-selected edge (`Models.py` 13253; `WalkTrials.py` 120, the losing trial's residual), not a float32 range overflow: a synthetic probe run from the scratchpad reproduces the exact signature (928 entries, rows 360 to 391, finite loss) identically in float32 and float64, while the October 30 overflow mechanism vanishes in float64. The reconstruction norm of 438735 beside an output norm of zero into the understanding points the same way. The arm was not re-run here: 875 s unloaded, the machine under the suite, a different torch, and the command's outputs inside the repository.

**Readings for Alec**, from epochs.md and leaf-decisions.json. The complete arm (true, 0) missed the lesson criterion over eight epochs, 0/80 bound and 0/80 exact. The chooser selected `sign` 0 of 704 times in that arm and 1 of 64 in the other, so the second decision is live but almost never taken. The output norm into the understanding was zero in all nine published epochs, including the one under `separateChooser=false`, where the shared-writer design of §16 expects A to reach the shared parameters: Codex should say whether that zero is by construction (the concluded understanding detached, A reaching only its own readers) or a gap in the path. The reconstruction gradient norm ran from 9.9 to 1555 in the true arm and reached 438735 in the false arm's one complete epoch, the precursor of the stop.

**Verdict.** Items 12, 13 and 15 to 22 are approved as built against their specifications, with the deviations above recorded, and the round stays open until `make test` is green or every failure carries its named strict xfail; the next return names the item-15 and item-17 readers in §33 by append and renames the receipts. Item 14 stays open and stage 2 unfrozen. Next for Codex, in order: (1) re-run the single arm alone, in scratch, with `trace-backward.py`'s `large` gate removed so every grad_fn is hooked and the first non-finite input cotangent's node and saved tensors are written, write the finiteness of `_epoch_objective_gradients` per objective at the failure, and repeat once in float64 as the discriminator between overflow and a structural zero times infinity; then repair at the producer under the general error contract: rejected inverse images and the losing trial's residual leave the backward graph (detached before the `*0` edge or the `torch.where`), singular inverse derivatives bounded as `_BoundedAtanh` is, never `nan_to_num`, clamping or gating; and correct GradientFlow.md's "Read first" and §25.1 to say the denominator was removed before this stop; (2) re-run the (false, 0) arm and start (false, 0.5); (3) the freeze, with certificate (vi)'s forced-configuration check recorded in leaf-decisions.json. Accepting 2.5 is Alec's call once item 14 closes.

### 34.1 Addendum 7, items 23 to 27, for Codex (Claude, 2026-11-01)

Read §34 first. Items 12, 13 and 15 to 22 are approved as built against their specifications; everything in §§32 to 32.4 stands. The round stays open on item 14 and on the suite. One environment note: by Alec's decision this machine runs the latest stable torch, today 2.15.0 with executorch 1.5.1 and pytest 9.2.0; nothing is pinned, and what an upgrade breaks is fixed in the code (the eight pytest-9.2 failures of §34 are fixed in `951c713a4c`). State the torch version in every receipt. Results go in §35.

~~~
23. The producer of the stop. Re-run the single (false, 0) arm alone, in scratch, with every output outside the repository, with `trace-backward.py`'s `large` gate removed so every grad_fn is hooked; write the first non-finite input cotangent's node and its saved tensors, and the finiteness of `_epoch_objective_gradients` per objective at the failure; repeat once in float64 as the discriminator between a range overflow and a structural zero times infinity. Name the producing operation in §35 with its path to `comparison_reader.weights.question_conditioners/392/weight`. Make the preflight capture complete for next time: every non-finite parameter, not the first in optimizer order (`Optimizer.py` 236 to 251), and the per-objective gradient report already in memory (`SentenceTraining.py` 147 to 171) written into the capture.
24. The repair, at the producer, under the general error contract. Rejected inverse images and the losing trial's residual leave the backward graph: detached before the `*0` edge (`Models.py` 13253; `Generative.py` 482) or the `torch.where` (`WalkTrials.py` 120); a singular inverse derivative is bounded as `_BoundedAtanh` is. No `nan_to_num`, clamping or gating anywhere. A certificate reproduces the probe's signature (928 non-finite entries, one meaning lane, finite loss) on the unrepaired code and is finite on the repaired code, in float32 and in float64. GradientFlow.md's "Read first", its "normalization contract and the generation exception" paragraph and spec §25.1 are corrected by exact replacement of your own text: the live denominator was removed before this stop; what remains is the producer item 23 names.
25. The suite. `make test` is the criterion (Testing.md); the gates were selections. Triage the 48 remaining non-link failures recorded at `output/tests/20261101-111946-0923ae` and listed in §34: a regression is fixed; a test stale against this round's changes is updated, with the change named in §35 (the six read that way: `sign` as a unary rule in the catalog, `test_bounded_stm_fold` and `test_word_loop_buckets_and_orders`; the generate cost per lane, `test_output_walk` and `test_generate_repair`; `_observe_native_cases` removed, `test_primitive_properties`; conceptual width 368, `test_wholespace_property_migration`); an unresolved deviation gets its strict xfail naming it; the three cases where idempotent.xml's zero learning rate reaches `SparseAdam` through the intuition parameters this round adds to the sparse set are resolved: a zero rate constructs no sparse optimizer, or those parameters are not sparse. The 26 dead relative links in the receipts are repaired or the receipts pruned. Return with `make test` green or every failure carrying its named strict xfail, and the run's result.json path in the receipts.
26. The measurement and the freeze. Re-run the (false, 0) arm to completion and start (false, 0.5); then the stage 2 freeze, with certificate (vi)'s forced-configuration check recorded in leaf-decisions.json as `forced_configuration_passed`. Answer in one paragraph of §35, with the code path: is the zero output norm into the understanding in all nine published epochs by construction (the concluded understanding detached, A reaching only its own readers), or a gap in the shared-writer path under `separateChooser=false`?
27. Housekeeping, by append to §33 or in §35: name the item-15 removal of the serial-path concept-expectation read and the item-17 readers of the removed columns, as the archived checkpoint review does; rename the remaining operation-sense "witness" in the receipts (readings.md lines 17 and 19; the `section-32-witness-*` directories); delete the dead helpers `ReferenceClose.mention_loss` and `answer_loss`; add one test that reads `whereRungRatio`, or drop the claim that the bounded selection covers it.
~~~

## 36. Review of the §35 return (Claude, 2026-11-01, night): items 23, 24 and 27 built; the suite green at the default workers; a regression under an expected failure; stage 2 not frozen

Reviewed against §34.1 on the working tree over `b41692f006` at torch 2.15.0, with three independent read-throughs and my own probes; Codex committed nothing. Verdict first. Items 23, 24 and 27 are approved as built. Item 25 is met on the suite and not on the expected failures: `make test` is green here at the default ten workers, but one of the two strict expected failures hides a regression. Item 26 completed the (false, 0) arm and stopped in the (false, 0.5) arm at a new producer; the freeze did not happen, correctly. The round stays open on §36.1.

**Item 23, the producer.** Verified. The native replay traces the first non-finite cotangent to `DivBackward0` at the boundary solution `ay / aa` of `MeaningCodes.recover_lanes` under `LanguageSpace._bounded_binary_reconstruction` for Sum: denominator `1.6155871338926322e-27`, 31 nodes to `comparison_reader.weights.question_conditioners/392/weight`, 928 entries in rows 360 to 391, loss 0.25. I reproduced it independently: with the new bound swapped back for plain division, the current tree gives 928 of 11,368 non-finite entries in float32 and none in float64; with the bound, none in either. So the native producer is a float32 range overflow of the mean inverse's derivative. The two producers §34 favoured, a rejected inverse image and the losing walk's residual through the discarded edges, give 928 in both precisions in Codex's controls, and both are repaired. Codex states that the first operation of the archived §33 run is unrecoverable from its capture; honest, accepted. The float32 and float64 probes (41,887,596 and 26,875,637 hooked nodes over 160 sentences) ran on the starting source `6b516b9081ee`, which §35 calls "the unrepaired tree" without naming it. The preflight is complete: `NonFiniteGradients` (`Optimizer.py` 100) carries every affected parameter with its counts, and the per-objective vectors with their finite counts, and is raised after the whole scan (`Optimizer.py` 262 to 274). The vectors are present only when the gradient report is on, and `producer-failure.json` is written by the benchmark capture script, not by `bin/`; acceptable, since every measured arm runs with the report.

**Item 24, the repair.** Verified, and it is the allowed pattern. `_BoundedInverseRatio` (`MeaningCodes.py` 45 to 62) returns the exact quotient and bounds each local partial before the cotangent is applied: the partial in the numerator is $1/\max(d, 0.19)$, the partial in the denominator is $-n/d_s^2$ with $d_s \ge \sqrt{0.19\,|n|}$, so both are at most $1/0.19 \approx 5.26316$, the `_BoundedAtanh` cap of `Layers.py` 46; the arriving cotangent passes through unchanged, so a non-finite cotangent still propagates. It replaces the four quotients of `recover_lanes`. The menu is built under `torch.no_grad()` (`Models.py` 13266: "The menu is an observation. Only the selected inverse is an operand."); the `(options.sum() + others.sum()) * 0` edge is gone, and `selected_inverse` (`Generative.py` 469 to 504) differentiates the chosen inverse per stream, substituting the neutral operand for the rows another stream selected before the operation rather than masking after it. `WalkTrials.kept_walk` (101) returns the winning walk whole when every stream agrees and otherwise replays the recorded integer actions once, asserting equality; `output_pair` (114) detaches the residuals and costs for the keep decision and keeps only the winner's residual live; the decoder mirrors it. The `torch.where` over the two live residuals is gone. No `nan_to_num`, hook or gradient mask was added anywhere in the diff, and a HEAD gate in `Generative.py` that filtered proposals by `isfinite` became an assertion, the right direction.

Three places are near the contract and are recorded for Alec. `InverseMenu.features` (`Generative.py` 23 to 30) zeroes, by a forward `torch.where` over the detached feature tensor, the possibly infinite residual features of inadmissible candidates, so that the scorer's weight gradient cannot form $0 \times \infty$, and asserts the admissible features finite. `SentenceCredit.score_function` masks each factor of an untrained entry before the product, after asserting the trained advantages finite. `CaseSelection` replaces a `clamp_min(tiny)` denominator by `where(mass > 0, mass, 1.)` where the numerator is zero on the empty lane. The second and third substitute neutral operands and are fine. The first masks non-finite values, but values of candidates that are by definition unavailable, with the admissible ones asserted; I accept it, and Alec may overrule.

The momentum retirement is a consequence of this item. With the zero edge gone, a rejected operator has no `.grad`, so SGD skips the parameter and its momentum buffer; before, every operator in the first menu received a zero gradient and a momentum-only step. The old behaviour was Codex's own specification (verb-and-modal §13, `doc/plans/2026-10-23-verb-and-modal.md` 775 to 779; 2.7 corpus §32, `doc/plans/2026-10-27-item2-7-corpus.md` 1719 to 1722), with no decision of Alec's on it in the record, and `ObjectiveOwnership` already treats discarded trials this way. Accepted; the two plan sentences and the fixture comment at `test/fixtures/item27-103c-output-walk.py` 46 to 48 need the retirement note (item 34). `test_inverse_work` and `test_meaning_kernels` were rewritten to the new contract and pass.

The certificates: `test/test_generation_selection_backward.py` (the losing walk's residual, the native Sum inverse, the rejected inverse and the unary chosen by another stream, each in float32 and float64) and the new preflight case in `test/test_finite_gradient_guard.py`; 27 passed in 8 s. Against a snapshot of HEAD the file fails as a certificate should: the rejected-inverse case in both precisions with the 928-entry, rows 360 to 391 signature, the native Sum case in float32 only; the other two fail at HEAD for interface reasons. The tests assert finiteness and the exact expected gradients, not the count; the count is recorded only in the receipts of the scratch probes. That meets the item. Two selections of the same shape as the removed residual line survive elsewhere: `WalkTrials.select_output` (86) over the live outputs of both trials, called at `Models.py` 10448 for `_costed_thought_outputs`, and `SentenceUnderstanding.py` 163, `torch.where(gate, right, left)` over two live trial fields. Neither produced either stop; the hazard is the same (item 31).

The docs: GradientFlow.md's "Read first" is replaced and dated 2026-11-01, and the new sections "Current gradient boundaries and loss normalization" and "The normalization contract and the generation exception" state that the denominator was gone before §33 and name the producer and the bound; §25.1 is corrected in the same terms; every link of §35 resolves. Two blemishes: the sentence "Everything below this box is the historical record" was dropped, so the 2026-10-30 section at GradientFlow.md 930 still opens "Today every residual is a ratio" with only its date as the marker, and §33's table carries the empty-set glyph (U+2205) that `make doc` cannot set (item 34).

**Item 25, the suite.** I ran `make test` with no arguments on this tree: `output/tests/20261101-215133-6e7ce2`, exit 0, 7,232 of 7,232 cases completed in 1,329 s at ten workers, the same two strict expected failures and one unexpected pass, peak 27.26 GiB against the 28 GiB ceiling. Green, with 0.74 GiB of margin where yesterday's run of the same selection had 1.3 GiB (item 32).

The receipt `output/tests/20261101-160817-638325` is as stated: 7,232 selected, 6,906 passed, 323 skipped, 2 strict expected failures, 1 unexpected pass, 2,108 s, peak 9.17 GiB, at `TEST_ARGS='--workers 3'`. The default is ten workers (processors minus four) under a 28 GiB aggregate ceiling (physical memory minus 8 GiB); `--workers` changes only the concurrency. Codex's ten-worker attempt `20261101-160050` was killed at 28.11 GiB, where yesterday's ten-worker run of the same selection peaked at 26.7 GiB and completed.

The triage: of the 82, 26 were the receipt links (395 rewrites inside the 26 gitignored files, 8 to GitHub permalinks, 2 de-linked as absent; `test_doc_links.py` 834 passed), 9 the pytest 9.2 undo fixed at HEAD, 2 the zero learning rate in `bin/` alone, 1 the zero rate with a tolerance rewrite, 1 the expected failure, and 43 closed by editing the failing test, with or without a change in `bin/`. Of the six I named stale, four changed exactly as the cause requires and are stricter or equivalent: `test_bounded_stm_fold` and `test_word_loop_buckets_and_orders` assert the unary catalog `['interpret', 'sign']`, `test_output_walk` reads the three lanes, `test_wholespace_property_migration` reads 368, 352 and 360. Two also loosened: `test_primitive_properties` accepts any unit's column where it accepted the one fused column, and `test_wholespace_property_migration` checks two named `_owned_bases.what` tensors where it checked every one. Elsewhere, `test_item55_mechanisms::test_surface_inverses` compares to one ULP of the total where it compared to $\pm 1.2 \times 10^{-6}$ absolute, looser once the total exceeds about 12.6; `test_operators_round2e` runs one batch where it ran an epoch; `test_free_reconstruction` filters out the new replay calls and flips `free_bytes` from trained to not trained; `test_generate_repair`'s frozen comparison requires each of the three cost lanes to equal the old scalar. Two of the 81 "passed" are renamed tests with different contracts: `test_wholespace_fixed_capacity` ("W is fixed and its property members are optimizer-visible" for "identity frozen and optimizer-visible") and `test_mixing_reconstruction_bank` (percept bytes in the target, for target and candidate bytes). Each of these needs its original assertion back or a sentence (item 30).

The expected failures. `ClockDependentBinding` (`test_item55_mechanisms.py` 172) is the §32.2 deviation, strict and unchanged. `MixingCapacityBeforeXORCriterion` (`test_mm_xor.py` 181, on `test_mm_grammar_learns_xor_signal`) is a regression, not a deviation: the test passed on 2026-10-19 (`output/tests/20261019-013504-e963dc`) with the same 20-row inventory, `data/MM_grammar.xml` being unchanged since `3561f39aa0`; it failed yesterday with "identity promotion exceeds the constructed inventory" and now exhausts the order-one seats; the two exception classes the marker accepts, `IdentityPromotionCapacityError` (`FamiliarTier.py` 454, raised from the new `_preflight_concept_row`) and `ClauseCapacityError` (`ClauseRow.py` 12), were both created this round. Item 25 said a regression is fixed, and nothing in §35 names what now consumes the seats (item 28). The unexpected pass is the non-strict historical top-k criterion, unchanged from HEAD; it stays.

The zero learning rate: `Models.py` 2861 to 2865 puts the sparse adaptive parameters in a zero-rate SGD group when the rate is zero and in `SparseAdam` otherwise, neither of the two resolutions I named and the same effect (no `SparseAdam` state; the parameters stay visible to `zero_grad` and the preflight); the three tests pass. Accepted. The held answer (`PreparedGeneration`, `Output.py` 133 to 140; `Models.py` 8949, 9698, 12657) and the compiled unary strides (`Generative.py`, the stacked planes under `torch.cond`) are production fixes with their tests. The packed reconstruction report (`Models.py` 21846 to 21856) zeroes, in the report only, the costs of streams the global lexicon decoded without a native target; no test names it, and it narrows what the identity audit sees (item 30).

**Item 26, the measurement.** The (false, 0) arm completed: 8 epochs, 160 sentences, 0 of 80 bound and exact, as §33's (true, 0) arm. The (false, 0.5) arm completed 4 epochs and 64 sentences at 0 of 32 and stopped at sentence 69, "A cat lives.", relations epoch 1, batch 4, with the loss finite at $3.9 \times 10^{12}$ and, before the step, every entry non-finite in four parameters: 512 in the verb meaning action's readout, 2,048 in its object vectors, 16 each in the log scale and bias of the sign kernel (`unary_ops.9`); the same 2,592 entries in the reconstruction, output and chooser accumulations; expectation and attention finite; no per-node hook. A different signature from §33: whole parameters, three objectives, and a loss ten orders above the arm's others. The 78 sign leaves of the published decisions are all in this arm (the unjittered arm chose 0 of 704), and two of the four parameters are the sign kernel's (item 29). `forced_configuration_passed` is null and stage 2 is unfrozen, correctly.

The answer on the output gradient into the understanding is accepted: under `separateChooser=true` it is zero by construction (`ObjectiveOwnership.backward_owned`); under `false` the shared writer does receive it, 2.7427 into `conceptualSpaces.0.layers.4.coefficients`, the percept-to-word readout, nonzero in all eight epochs, while the 51-tensor group of codes, grammar operators and inverses, and conceptual definitions received zero in all 12 epochs. The old diagnostic read catalogs the live layer did not expose; `SentenceTraining.operator_groups` now reads `_tree_layer(2)`. One sentence is still owed: with the kept walk now differentiated through its chosen inverse, zero output gradient into the inverses in all 12 epochs wants its reason (every kept walk exact at the root, or the inverses without parameters on these sentences).

**Item 27.** All verified: `ReferenceClose.mention_loss` and `answer_loss` gone; `readings.md` 17 and 19 and the `section-32-sign-*` directories renamed (the `witness-allocation` receipts keep the evidence sense, outside the item); `test_question_location_reads_the_configured_rung_ratio` (`test_where_bracket.py` 39 to 49) sets `whereRungRatio` 8 and asserts the period; the item-15 serial-path read is retired in `ConceptSpread.open_prediction`, with `Spaces.relevance_weights` and `project_priming_to_towers` gated on it, both `expectation.reference_mentions` registrations are gone, and the item-17 readers are named as §35.2 lists them.

**Outside the addendum.** The float64 probe found a lifetime bug in the medium-term memory: a newly kept admission had no access stamp until the close finished, so a full 64-seat pool evicted six of seven before their deferred null-kind definition read them. `MediumTermMemory.keep_access` (162) stamps a kept admission with the current close; `_room(retain)` (168) keeps a pending definition's operand seats and orders eviction by touched-this-close, last access and time, raising when nothing is evictable; `ReferenceClose.commit` threads the retained seats through its three opens. `test_item25_completion.py::test_two_kept_signs_survive_a_full_pool_until_their_definition_is_read` covers it with a two-seat pool. Accepted as the repair the probe required; the changed eviction order belongs in Params.md's `mtmHorizon` row (item 34).

### 36.1 Addendum 8, items 28 to 34, for Codex (Claude, 2026-11-01, night)

Read §36 first. Items 23, 24 and 27 are approved as built; items 25 and 26 stay open. The runtime is unchanged: torch 2.15.0, executorch 1.5.1, pytest 9.2.0, nothing pinned; state the torch version in every receipt. Results go in §37.

~~~
28. The XOR regression. `test_mm_grammar_learns_xor_signal` passed on 2026-10-19 (`output/tests/20261019-013504-e963dc`) with the same 20-row MM_grammar inventory and is now a strict expected failure on seat exhaustion. Withdraw the marker. Count what consumes the order-one seats in that fit on this source and on the newest commit where the test passes (identities, occurrences, sign tokens, clause rows), name the change that grew the count, and restore the pass. If the count shows the inventory is now insufficient by design, write both counts in §37 and stop there: whether `MM_grammar.xml` grows is Alec's decision, not a marker.
29. The producer of the (false, 0.5) stop. Item 23's method on that arm: hooks on every node, outputs outside the repository, float32 and float64, to sentence 69; the first non-finite input cotangent's node and saved tensors, its route to the four parameters, and the per-sentence loss trajectory, since a finite loss of 3.9e12 marks the divergence before the gradient does. Check the sign kernel (`unary_ops.9`) first: two of the four parameters are its, and this is the only arm with sign leaves. Repair at the producer under item 24's contract, with a certificate that fails on the unrepaired source and passes on the repaired one in both precisions.
30. Test-side closures. For each of `test_primitive_properties` (the accepted columns), `test_wholespace_property_migration` (the two named tensors), `test_item55_mechanisms::test_surface_inverses` (one ULP), `test_operators_round2e` (one batch), `test_free_reconstruction` (replay filtered, `free_bytes` not trained) and `test_generate_repair` (three lanes equal to the old scalar): restore the original assertion with the `bin/` fix it needs, or one sentence in §37 naming the design change that retires it. The two renamed tests keep their old contract beside the new one or record its retirement. The packed reconstruction report's zeroing gets a test that names it or is reverted.
31. The surviving live selections. `WalkTrials.select_output` at `Models.py` 10448 and `SentenceUnderstanding.py` 163 select between two live trial fields with `torch.where`, the shape item 24 removed from the residual. The loser leaves the graph as in item 24, or one sentence why its field is finite by construction, with a certificate either way.
32. The memory margin. The default ten-worker `make test` peaked at 27.26 GiB here, 26.7 GiB yesterday on the same selection, and 28.11 GiB in your attempt `20261101-160050`, which the 28 GiB ceiling killed. The criterion is the default command. From the worker receipts of the two completed runs, name the workers whose peaks grew and bring the peak back to yesterday's; a run that needs `--workers 3` is not the criterion.
33. The measurement, after items 28 and 29: the (false, 0.5) arm to completion on the repaired source, the (true, 0) arm on the same source, and the freeze with the three-arm forced-configuration check recorded as `forced_configuration_passed`, as item 26 specified.
34. Housekeeping, by append: the starting source `6b516b9081ee` named in §35.1; the empty-set glyph in §33's table replaced by `$\emptyset$`; the historical marker restored above GradientFlow.md's 2026-10-30 section; the retirement note at verb-and-modal §13, 2.7 corpus §32 and the fixture comment; `producer-failure.json` carrying its sentence, epoch and batch; the MTM eviction order and the exhaustion error in Params.md's `mtmHorizon` row; the sentence of §36 on the zero output gradient into the inverses.
~~~

### 36.2 Audit of the test-side closures and of the seat consumption (Claude, 2026-11-02, early morning)

Alec asked two questions after §36.1 went to Codex: whether the test edits that closed the §35 failures hide anything, and whether the MM_grammar XOR failure comes from the softmax chooser. Both were measured on the working tree as it stood between 00:06 and 00:37 on 2026-11-02, while Codex was already acting on addendum 8; the file times are given where they matter. The method for the tests was the same throughout: the HEAD copy of each edited test ran against the working tree's `bin/` from a directory outside `test/`, with the HEAD `conftest.py` beside it, so the verdict is about the assertion and not the name; where only an API had moved, the copy was patched to the new name with the original assertion kept.

**The seats.** The premise of item 28 was mine and wrong. The 2026-10-19 pass (`180288c32f`, worker 016 of `20261019-013504-e963dc`, 318 s) ran the eight-row `MM_grammar.xml`, before `FamiliarTier.promote_identity` existed: the MTM identity promotion arrived between 2026-10-21 and 2026-10-23, and the test's presentation call arrived with the 20-row inventory on 2026-10-24 (`3561f39aa0`). Codex's §37.1 has this history right. An instrumented replica of `_fit_xor` on the §35 source (started 00:06, with `FamiliarTier.py` as of 00:01) reproduced the suite failure: exhaustion at epoch 10 after 1,226 s, against the suite's 1,150 s and 1,255 s. Over those epochs `ReferenceClose.choose` was called 238 times and selected an existing candidate 238 times, never the zero-logit none option and never without candidates; the largest candidate logit was always about 1.0. The six promotions were of word ids 3, 2, 3, 2, 2 and 4, to concept ids 13, 25, 41, 44, 46 and 52: the same word was promoted to a fresh order-1 concept again and again. The tail seats went 1, 1, 2, 2, 3, 7, 8 of 8 across epochs 0 to 6, with the order-1 partition at 4 of 4 from epoch 0 (the covers) and four order-0 seats unused throughout, as §47 of the verb-and-modal plan says they must be. A second replica called production's epoch boundary (`Forgetting.model_boundary(epoch_boundary=True)`, as `runEpoch` does) after every step (started 00:23, `Models.py` as of 00:10): the pool was empty at every epoch end, one promotion happened per epoch (word ids 4, 2, 3 to concept ids 13, 20, 27), two tail rows went per epoch, and `MemoryConsolidation` exhausted the pool at epoch 5; 24 choices, 24 candidates. On the 10-19 source under the same loop the rows stayed at 4 of 8 for 39 epochs. So the chooser is not the consumer. The consumer is `promote_identity`, which mints a concept for every promoted token with no recognition that the token's word is already defined, so each presentation of the same four sentences costs the same words new order-1 seats; the softmax in the resolver only records probabilities and a diagnostic cost. Codex's in-progress §37.1 names the lost lexical ID in the mixing reader as the reason the promotion cannot recognise the word; its repair was still landing when these runs loaded their modules (`Models.py` 00:10, `MemoryConsolidation.py` and `PresentationMemory.py` 00:31) and is reviewed with §37. Two consequences are recorded for Alec. First, Codex's working tree replaces the test's `begin_presentation()` with the native epoch boundary; that is a change to the regression test's contract, defensible because production releases epoch experience and because without it LTM0 fills its 1,024 rows (Codex's run at `0aedc06414`, epoch 253), but it is his to accept. Second, the same defect is visible in `test_grammar_object_leaves[aligned]`: the HEAD assertion, which reads each word back through the lexicon after the reading, fails in two of three runs here and two of five in the audit, because after the kept close two, three or all four distinct words resolve to one freshly admitted object at row 20; the new assertion compares the resolver's output to the rows it was itself staged with and passes regardless. That closure hides the regression of item 28.

**The closures.** Four parallel audits covered the 43 test-side closures of the triage, the tests edited in the same round without a triage entry, and the two renamed tests.

| Verdict | Cases |
|---|---|
| Legitimate adaptation, assertion strength kept | `test_bounded_stm_fold`, `test_word_loop_buckets_and_orders` and `test_primitive_properties` (catalog and admission changes of items 22 and 2.7, recorded at §32.3 and §34); `test_constituent_lesson`, `test_dimensional_governance`, `test_forward_discard`, `test_wholespace_property_migration` (widths 360, 368 and 488 of items 2.7 and 2.8; the `byte_bits` buffer is exempted from one check and given an exact-shape check); `test_wholespace_fixed_capacity` (W became a buffer in `a8a09a6acb`, item 2.8); `test_decoder_exploration`, `test_generate_repair`, `test_output_walk`, `test_walk_trials` (three lanes and the single replay, stricter than before; the frozen 103c comparison still exact); `test_prepared_answer_boundary` (signature, plus a stronger no-lookup check); `test_inverse_work` and `test_meaning_kernels` (the momentum retirement accepted at §36); `test_priming_diffusion` (the eleven removed tests are round B's retirement of location diffusion, §22.3 and §23 item 1, not this round); `test_reference_mentions`, `test_reference_review10` (items 15 and 17); `test_iterated_symbolic_wave` (the reflexive guard approved at §32); `test_operators_round2c` and `test_operators_round2e` (the audit is cleared at the epoch boundary, as at HEAD, and both read the one batch they always had); `test_selected_nested_meaning` (renamed stores, ported with the original strength); `test_sentence_boundary` (two source-address writes, as before, on this tree); `test_sparse_compile_boundary` (body and head each checked for the same policy); `test_sparse_concept_e2e` (the intuition parameters acknowledged at §34; the concept predictor is not named anywhere, a minor gap); `test_packed_reconstruction_parity` (configuration only; the packed report's zeroing has no assertion here, so item 30 stands). |
| Stale before this round: the HEAD assertion fails against HEAD's own `bin/` | `test_free_reconstruction` (`free_bytes` has been reporting-only since GradientFlow.md's HEAD text; the registry's extra terms exist at HEAD), `test_generation_catalog` (`menu_score` exists at HEAD), `test_item55_mechanisms::test_surface_inverses` (the 2.14e-6 discrepancy at totals near 62 is identical at HEAD; one float32 ULP there is 3.8e-6). Item 30's sentence in §37 is still owed for each, but no `bin/` fix of this round is being hidden. |
| Hides a change | `test_grammar_object_leaves[aligned]` (above). `test_review14_contracts::test_compose_score_function_moves_toward_the_cheaper_departure`: the HEAD test passes against HEAD's `bin/` and fails now because this round books the compose, thought and walk-credit score-function terms to a `chooser` objective (`Models.py` 19939 and 21639, `ThoughtCredit.py` 25; absent at HEAD), the gradient values unchanged; no sentence records the re-ownership and GradientFlow.md line 117 still says the term is booked to reconstruction. `test_mixing_reconstruction_bank` (renamed): the non-ASCII word is no longer a reconstruction-bank candidate (lookup row $-1$), reversing the decision recorded in the 6.9 XOR-grammar plan at lines 725 to 730 ("make the candidate surfaces agree"); the renamed test asserts the new behaviour, no retirement is recorded, and the staging function is byte-identical to HEAD, so the producer is upstream of it. `test_reading_curriculum`: the content inverse now returns the native dictionary code exactly, while the children it was asked to reproduce were pushed as presence-scaled values (per-word factors 0.03 to 0.78), 550 of 2,880 entries differing by up to 0.97; the test switched its reference to the dictionary code; no decision records that the inverse should ignore presence (GradientFlow.md lines 409 to 411 describe form times presence). `test_relation_closing`: on a first encounter the mixing program carries word ids 1 to 4 while staging mints object ids 20 to 23 and writes two definition rows; the test now pre-admits every form before the forward and asks that definitions be unchanged rather than empty, so the disagreement between the two id sources is never exercised. This is item 28's family seen from the relation close. |
| Unclear | `test_grammar_sentence_scope`: the fixture now zeroes the eight occurrence coordinates (columns 352 to 359) before the exact comparison; the HEAD assertion passes on the tree as of 00:10 but failed at triage time at exactly column 352, and no `bin/` hunk or sentence explains either. `test_review14_addendum[MM_grammar-20]`: non-word concepts may now carry `('occ', ...)` parts, which accommodates the same growth as item 28 and is judged with it. `test_sentence_compose`: the committed relation now has two occupied roles where the old test expected one and the single-point context reader is unchanged; no decision either way. `test_unified_thought_controller`: `equal` is no longer among the child's actions after a query (the offered set is `not`, `exist`, `intuit`, `ask`, `infer`, `query`), and the test substitutes `not` instead of saying why `equal` left; no producer located. `test_sentence_comparison`: the third pullback per trial is the chooser objective (legitimate, §26.3), but the new assertion accepts any equal count of at least two where exactly three per trial would keep the strength. |

**Addendum 8, continued.** Items 35 to 43 join §36.1; results go in §37.

~~~
35. Item 28, corrected. Its premise was wrong: the 2026-10-19 pass ran the eight-row inventory before identity promotion existed, as your §37.1 says. The count is settled by §36.2: the resolver never chose the none option in 262 recorded choices, and `promote_identity` minted a new order-1 concept for the same word at every presentation. Verify your lost-lexical-ID diagnosis with a certificate that fails on the §35 source and passes on the repaired one, counting promotions per word over ten presentations: at most one. The test's switch from `begin_presentation()` to the native epoch boundary is a contract change: record it in §37 as Alec's decision to confirm, with the repaired fit's result under both loops beside each other.
36. `test_grammar_object_leaves[aligned]`. Restore the post-reading lexicon assertion (`interpret.forward(word_for_form(w))` for the four words) and make it pass deterministically. Explain why distinct words came to share row 20: the promotion defines every reading of the token as a name of one identity. This is item 28's defect seen from the lexicon; one certificate may serve both.
37. The chooser objective. Record in §37, in one sentence with the three sites, that the compose, thought and walk-credit score-function terms are booked to the chooser objective, and correct GradientFlow.md's sentence at line 117 that still books them to reconstruction. The record is what Alec decides on; the gradient values are unchanged, so the only question is the owner.
38. `test_grammar_sentence_scope`. Restore the full-width comparison, without zeroing columns 352 to 359, if it passes on the repaired source, and explain the triage-time failure at (0, 352); otherwise the sentence naming the design change that removes the occurrence band from recovered leaves.
39. `test_review14_addendum[MM_grammar-20]`. After item 35, re-judge the loosening: if the repaired fit places no `('occ', ...)` part on a non-word concept, restore the old assertion; if it does, the sentence saying why an occurrence part belongs there.
40. `test_mixing_reconstruction_bank`. The non-ASCII word stopped being a bank candidate (lookup row -1), against the decision at 6.9 XOR-grammar plan lines 725 to 730, and `_stage_mixing_reconstruction_bank` is byte-identical to HEAD, so name the upstream producer. Restore the candidate-bytes assertion with the fix, or the sentence naming the design change and where it was decided; the renamed test keeps the old contract beside the new one until then.
41. `test_reading_curriculum`. The content inverse returns the native dictionary code while the pushed children were presence-scaled (factors 0.03 to 0.78). Restore the comparison to the pushed children's values with the fix, or the sentence naming the design change that makes the inverse ignore presence, against GradientFlow.md lines 409 to 411.
42. `test_relation_closing`. Restore the first-encounter assertions: the program's concept ids equal the staged object ids, and no definition rows are written. On the §35 source the program carries word ids 1 to 4 while staging mints objects 20 to 23 and writes two definition rows; that is item 35's defect at the relation close, so one repair may serve both. Pre-admission stays only as a second case, not as the replacement.
43. Three sentences in §37, one each: why the committed relation in `test_sentence_compose` now has two occupied roles where it had one; which change removed `equal` from the child's post-query actions in `test_unified_thought_controller`, with the assertion restored if the removal is a defect; and `test_sentence_comparison` restored to exactly three pullbacks per trial in trial order, since the third is the chooser objective.
~~~

## 38. Review of the §37 return (Claude, 2026-11-02, morning): the XOR regression repaired and certified; the verb reflection repaired exactly; the suite green at 24 GiB; stage 2 not frozen on two decisions for Alec

The return was verified on the working tree at source `39da97f4f161`, which is the tree the final suite ran on: all 972 files in `output/tests/20261102-015613-794765/source-manifest.json` hash to the files on disk. The suite's result file says exit 0, 7,251 selected and completed, 873 s, peak 24.05 GiB, 96 worker processes under the new recycling, torch 2.15.0. Three parallel verifications read every hunk of `source-4/ports.patch` (the whole round's diff from the §35 source), ran every certificate on this tree and, where the return claims a failure on the §35 source, on a scratch tree rebuilt by reverse-applying the patch (the hashes of all fifteen `bin/` files matched `ports.json`'s before values; a plain `PYTHONPATH` does not work for this, since `test/conftest.py` inserts the repository's own `bin/` first). Verdicts by item follow; the decisions for Alec are at the end.

**Items 28 and 35, the XOR regression: built and certified.** The cause named in §37.1 is confirmed in the code: on the §35 source the mixing reader staged the grammar object without its word id, so the kept reading had no word ids, `commit_read_words` located nothing, and `promote_identity` minted unconditionally and then named the new concept with every alias reading of its occurrences, which is also why distinct words came to share one row in `test_grammar_object_leaves`. Now the reader records the word beside the object (`Models.py` 11324 to 11336), `commit_read_words` binds each word's retained occurrence to its named token, `promote_identity` reads the token's introducing word and reuses the single existing denotation at the witnessed order (`FamiliarTier.py` 477 to 489), and recurrence counts only mentions whose reading equals the name. The ten-presentation certificate passes here and fails on the §35 tree with a second allocation for both words at the second presentation, the numbers in `native-ten-presentations.json`. The unseeded MM_grammar fit passed here in 210 s. The §37.1 table's four receipts exist and their per-file hashes equal the named commits. The kept-word alias repair of §37.8 is certified in both directions and fails on the §35 tree. Two caveats are recorded as items 44 and 45: reuse happens only when exactly one denotation exists at the witnessed order, and with two or more the code falls through and mints a new seat with nothing recorded; and the ten-presentation receipts name a test file that does not exist and temporary directories instead of hashes, so their provenance rests on the reverse-patched reproduction, not on the receipts.

**Item 29, the producer: repaired exactly; identified by a controlled trace, not natively.** `MeaningKernels._reflection_directions` rescales each direction by exact powers of two before the inner product (`MeaningKernels.py` 76 to 84, used at 92 and 106); the reflection formula and its pre-existing zero-direction guard are byte-identical. A Householder reflection is invariant under scaling its direction, the scale is an integer exponent invisible to autograd, so forward and gradient are exact, not bounded. The certificate passes here and fails on the §35 tree in both precisions with all-NaN gradients. `verb-sign-producer.json` records `DivBackward0` at `householder_action`, subnormal divisors ($4.0 \times 10^{-42}$ in float32, $4 \times 10^{-312}$ in float64) and routes to all four parameters. The native float32 run on the §35 source hooked 10,123,072 nodes through sentence 69 with every cotangent finite and a loss of 14.56 there; both float64 trajectories stopped earlier on semantic errors (§37.10, §37.12). So the first bad node of the historical arm was not identified natively; the controlled trace identifies a producer of the same four-parameter signature, and the repair removes it. Accepted as the best available: the historical capture did not save its first operation, and the new runs are unseeded trajectories that do not revisit it. §37.7's phrase "jitter probe" is wrong, the command was unseeded with no jitter (item 46).

**Item 31, the live selections: built.** `WalkTrials.select_output` returns the already costed output for a single winner and, for mixed winners, replays the recorded actions once and checks the replay against every cached trial under no_grad; `SentenceUnderstanding.select` does the same for the sentence record. No `torch.where` with live operands survives in the selected value; the one added at `SentenceUnderstanding.py` 164 builds a graph-free comparison target, and those in `SentenceCompose.kept_reading_plan` select integer actions under no_grad. The thirteen certificates pass; the square-root-at-zero ones establish absence from the graph, since a selected-away loser would leave a NaN gradient where the tests require `None`.

**Item 33, the measurement: not done.** The three arms, the forced-configuration check and the freeze wait on the §37.12 stop.

**§37.10 and §37.11: built.** The finish record now attaches a child's proof only when its unresolved roles match the caller's, returning the proof on the child's record otherwise, a rejection by contract rather than a silent attachment; certified on both trees. A predicate with neither a point nor a relative child raises; a relative child carries its address in the occupied role as the operands already do; certified on both trees.

**§37.12: a raised error, correctly left open.** The receipt reproduces the stop on both trees as `ValueError('recorded concept order disagrees with its parts')` from `settle` at the epoch boundary, with the three controls as narrated. Orders §11 makes a positive row a member only one order below and says the order guard checks occurrence and relation members, so row 48's relation-handle endpoint is under the guard; Codex's two options are stated faithfully.

**Items 36 and 42: built, with two notes.** The post-reading lexicon assertion is HEAD's text again for all three parameters and passed three of three runs each. The MM mixing case is read under `force_absolute_reading`, which masks the binary menu to lift, union, intersection and sum and the unary menu to not and non, so the sign and Part rules that can define an unknown label after the read are excluded; the test says so in a comment and §37.6 says why. The relation-closing test has the first-encounter and pre-admitted cases side by side, both exact. The equality is now against `_ar_grammar_object_ids` redefined as the word concept at the physical row, the transient referent having moved to `_ar_word_object_ids`; that is the identity design, and the test's meaning changed with it.

**Item 39: the test is sound; the sentence is an inference.** Orders §11 supports an occurrence member exactly one order below, which is what the test checks; it does not say that an identity's introducing occurrence belongs in its parts. Recorded here so that the next reader does not look for a sentence that is not there.

**§37.3, the repeated consolidation and directions: built; one pre-existing counter noted.** The epoch boundary keeps learned recurrence prototypes and symbols while clearing their rows, so repeating the same observations returns the same symbol; the change detector recognises a residual within the angular threshold of an existing column as a witness on that column. The "optional admission drop" of the change detector is a capacity preflight caught and counted in `admission_drops` rather than raised; it predates this round and the receipts show zero drops with the check. It is a capacity event, not a numeric one, and stays as is.

**Items 30, 37, 38, 40, 41 and 43, the contracts: built, with four notes.** The restored assertions pass. `test_grammar_sentence_scope` is byte-identical to HEAD again and both layouts pass at the full width; the fix is a general rule in the walk body that emits the live parent on an exact STOP (`Models.py` 13328 to 13329), not a fixture case, and the receipt shows the .86079836 at column 352 emitted where 0 was. `test_sentence_comparison` requires exactly three pullbacks per trial in order. The unified-thought test is HEAD's text, with the whitelist at `ThoughtStream.py` 85 to 90 admitting a closed `equal`; HEAD's own code failed it, so the loss of `equal` dates from round A's whitelist, as §37.4 says. The reading-curriculum test restores HEAD's presence-scaled assertion for admitted objects and adds the pending-name case, which the STM plan's bind contract supports. The clock marker is gone and all nine clocks compare. The surface-inverse sentence is true (HEAD's test fails on HEAD's code by the same 2.14e-6) and the one-ULP bound is documented in the test. The packed report's zeroing has its test. The mixing-bank test keeps the new contract only, which item 40 allowed once the design change was named: the word-whole admission rule landed in `0aedc06414` and F1 of the 2.7 corpus plan (lines 2229 to 2230) says a terminal proposal is an actual admitted native code, from which the exclusion follows; §37.6 states F1 the other way round, but the conclusion is the same. The notes: GradientFlow.md names only one of the three chooser-objective sites (item 50); under no_grad the exact-STOP terminal is still the menu's terminal with zeros in the occurrence band while training emits the parent's band, `Models.py` 13319 to 13329 (item 51); `test_sentence_compose`'s new assertion restates `from_row`'s own one-point projection and is weaker than HEAD's check that the context equals the stored relation, and the projection is recorded nowhere but §37 (item 52); the clock certificate zeroes the four clock coordinates inside the reference sites' content, cue and predicate before comparing, an exemption that "complete canonical semantics" does not mention (item 53).

**Item 32, the memory: the criterion met by the runner.** The runner recycles a worker after its current test once it passes 80 percent of the smaller of its reservation and its share of the aggregate budget, 2.24 GiB here; the default count and both hard ceilings are unchanged; the final default run peaked at 24.05 GiB. The per-worker comparison in `memory-before.json` is by worker number over two different selections, so it says where the peaks sat, not what grew them. Accepted: the criterion is the default command's peak, and it is under yesterday's.

**Item 34, housekeeping: all seven done.** §35.1 names `6b516b9081ee` as the unrepaired source; §33's table has `$\emptyset$` and the glyph is gone; GradientFlow.md has its historical marker; the three retirement notes are in place; the positioned capture carries sentence, epoch and batch and the SHA-256 of the raw producer file; Params.md's `mtmHorizon` row states the eviction order and the error; §37.9 answers the zero-gradient sentence. Glyphs: §37 carries a U+2192 arrow at line 2709, an ellipsis at 2678 and a superscript one at 2760; GradientFlow.md line 997 gained a superscript two and Params.md line 412 an ellipsis, and both files are PDF chapters (item 54).

**Decisions for Alec.**
1. The XOR regression test's boundary (§37.9). The native epoch loop is production's contract and passed at epoch 20 with only the words and covers seated; the document-only loop still exhausted the twelve order-1 seats at epoch 10 on the repaired source, so repeated presentations as separate documents grow the inventory, which §47 of the verb-and-modal plan says they must not. Either the epoch loop becomes the regression contract, or the document loop stays and the inventory derivation changes.
2. The same-order relation-to-identity row (§37.12): keep the row as evidence without membership, extending Orders §11 to a relation endpoint, or reject the membership at clause close with a new admission rule.
3. The explore-exploit proposal, in answer to Alec's question of 2026-11-02. Since item 4 (`f7e74af04f`, 2026-10-19, B3) the departure is sampled from $\phi \cdot \mathrm{uniform} + (1-\phi)\cdot\mathrm{softmax}(\mathrm{policy}/\tau)$ over the eligible alternatives with $\phi = .5$ and $\tau = 1$, credited by its sampled probability; before item 4 it was sampled from a uniform proposal over the eligible alternatives and credited by their count. No receipt compares the two on the same source: the §33 and §35 arms all used the mixture, and item 33's arms are outstanding. Setting `explorationUniformFraction` to 1, which the schema allows, reproduces the previous proposal exactly, so a paired comparison on one initialization is possible; it is proposed as item 48 for Alec to keep or strike.

**Addendum 9.** Items 44 to 54 join the open item 33 and item 35's boundary decision; results go in §39.

~~~
Read §38 first. Items 28 to 32 and 34 to 43 are approved as built, with items 50 to 54 completing 34, 37, 38 and 43; item 33 and the boundary decision of item 35 stay open on Alec's decisions in §38. The runtime is unchanged: torch 2.15.0, executorch 1.5.1, pytest 9.2.0, nothing pinned.

44. Ambiguous denotation at promotion. `promote_identity` reuses an existing identity only when exactly one denotation exists at the witnessed order; with two or more it mints a new seat and records nothing (`FamiliarTier.py` 481 to 489). Either raise, naming the word and the denotations, or choose by a stated rule written in Orders.md; a certificate either way.
45. Receipt provenance. `native-ten-presentations.json` names `test/test_lexical_promotions.py`, which does not exist, and temporary directories where hashes belong; `introducing-word-certificate.json` names a probe under `probes/`. Name the committed test nodes and the source hashes, as the other receipts do.
46. §37.7's "exhaustive float32 jitter probe" was an unseeded run with no jitter (`exhaustive-native-traces.json`, command `trace-backward.py --eager --curriculum lesson-13 --gradient-report`); correct the sentence by append.
47. The §37.9 fits ran on source-3, which differs from the final source in `ClauseJournal.py`, `ThoughtStream.py` and their two tests; one sentence in §39 that those files are outside the identity and boundary path, or the two fits rerun on the final source.
48. The explore-exploit proposal, paired. On the first complete epoch of the (false, 0) arm, run `explorationUniformFraction` .5 (the current mixture) and 1 (the pre-item-4 uniform proposal) from the same initialization and the same random stream, and report per epoch the kept trials' R + A, the chooser-group variance, the acquisition reads and the departure acceptance rate. This item is Alec's to strike.
49. Item 33, after Alec's decision on §37.12: the three arms on the final source, the forced-configuration check recorded as `forced_configuration_passed`, and the freeze.
50. GradientFlow.md names only `Models._compose_score_function_loss` for the chooser objective; add `ThoughtCredit.register` and the generation walk credit in `Models._run_sentence_word_bricks` to the same sentence, by append.
51. Exact STOP under no_grad. The walk body emits the live parent on an exact STOP only inside the selected-graph branch (`Models.py` 13319 to 13329); under no_grad the terminal is still the menu's, with zeros in the occurrence band. One sentence why inference should emit differently from training at the same step, or the same rule in both branches with the sentence-scope test run under no_grad as well.
52. `test_sentence_compose`. The new assertion restates `ReferenceContext.from_row`'s own one-point projection and is weaker than HEAD's check that the next sentence's context equals the stored relation. Record the projection of a two-endpoint Part relation to one head point beside the two-tagged-reference row layout in GradientFlow.md (near line 267) or in Architecture.md, and say what the context loses; or restore HEAD's assertion with the fix.
53. The clock certificate zeroes the four clock coordinates inside the reference sites' content, cue and predicate before comparing (`test_item55_mechanisms.py` 53 to 67), which §37.7's "complete canonical semantics" does not mention. The sentence saying why those coordinates are clock-dependent by design, with the exemption named in the test, or the exemption removed.
54. Glyphs, by edit in place: the arrow at §37 line 2709 becomes `$\to$`, the superscript two at GradientFlow.md line 997 becomes `$^2$` or `$10^{2}$` as the context needs, the ellipses at Params.md line 412 and §37 line 2678 become three periods, and the superscript at §37 line 2760 becomes `$3.907495 \times 10^{12}$`. GradientFlow.md and Params.md are PDF chapters and the LaTeX build fails on glyphs the fonts lack.
~~~

### 38.1 Alec's decisions on the epoch boundary and the explore-exploit history (2026-11-02)

**The epoch contract (decision 1 of §38).** Alec: multiple epochs were previously making LTM grow. An epoch exists to give the weights more opportunity to learn from the same history, not to hold several representations of one history; BasicModel does not treat epochs as a transformer does, which has no memory of its own to duplicate. The memory of the history therefore returns to its state at the start of the epoch, and only what has changed the weights carries over. There is precedent in networks with explicit memory: Santoro et al. (2016, ICML, "Meta-learning with memory-augmented neural networks", §4.2.1) wipe the external memory from episode to episode and report that learning is slower and less robust when they do not, since what persists across episodes interferes; and the complementary-learning-systems account (McClelland, McNaughton and O'Reilly, 1995) keeps one fast episodic trace per experience while the slow weights learn from its repeated replay. Consequences: the MM_grammar XOR regression test uses the native epoch boundary, as Codex's working tree already does; the document-only loop is not the contract; §47 of the verb-and-modal plan, which sizes the inventory for one presentation, stands.

**LTM1 at the epoch boundary (decision 2 of §38).** Alec: LTM1 relations are memory in sync with LTM0, so they too return to their pre-epoch state at the boundary. Today `PresentationMemory.epoch` withdraws only the relations whose references reach a withdrawn occurrence, transitively; a relation between two symbols, or between a relation handle and a symbol, survives. That is how row 48 of §37.12 outlived the epoch and met the order guard. Under the decision, rows 47 and 48 are withdrawn with the LTM0 rows of the epoch that wrote them, so neither of the two options in §37.12 is needed: the row does not survive to be judged.

**Minted concepts stay.** Alec briefly extended the decision to every concept minted during the epoch, then withdrew the extension: a minted concept cannot be removed, because it has been wired into existing concepts and the weights have changed with it. Minted concepts therefore stay, word admissions included, with their names, recorded orders and their own wiring, as the 2026-10-28 contract (Architecture.md, "Presentation and epoch boundaries") and §37.3 have them, and a later epoch recognises them instead of minting again. What returns to its pre-epoch state is the memory of the history: LTM0's rows, LTM1's relations with their evidence, and MTM. This reading takes LTM1's relations to include definitions written during the epoch, which supersedes that contract's clause that definitions survive; the concept a definition named keeps its name. If Alec wants definitions kept, item 55 gains that one exception. Where a minted concept's wiring into existing concepts is itself stored as an LTM1 relation, the two parts of the decision meet; item 55 asks Codex to establish that before building.

**Item 48 runs.** Alec sees no reason to strike it; its last sentence is withdrawn.

**When the explore-exploit mixture was introduced (Alec's question).** The departure proposal $\phi \cdot \mathrm{uniform} + (1-\phi)\cdot\mathrm{softmax}(\mathrm{policy}/\tau)$ arrived in `f7e74af04f` (2026-10-19, item 4), together with the expectation gains, the retired slot predictor and the `non` change, so suite counts cannot isolate it.

| Run | Cases | Passed | Skipped | Failed | Expected failures |
|---|---:|---:|---:|---:|---:|
| Last full suite before, `20261019-013504-e963dc` on `180288c32f` | 5,805 | 5,473 | 334 | 0 | 1 |
| Item 4 landing sweep on its fixed source (2026-10-19 item 4 repair receipt) | 5,834 | 5,475 | 322 | 39 | 1 |

Of the 39 failures, 14 were the sampler's interface: nine forced-chooser fixtures that expected two return values where the selector now also returns the proposal, three tests of the old sampler's assumptions (among them review 16's test that the departure is uniform and independent of the logits), and two reservoir fixtures whose random draws the proposal now consumed. All fourteen were repaired as fixture updates and no behavioural defect was recorded against the proposal; the other 25 belonged to the retired predictor, the `non` change, strict XML loading and the percept statistic's construction. The learning measurement of the time (two seeds, graded against shuffled order, six paired batches, gains at .5) ran only with the mixture: graded order beat shuffled for seed 19 and lost for seed 23, and seed 23's chooser did not update because its three compose departures tied at equal corrections. It never compared the mixture with the uniform proposal, which is what item 48 now measures.

~~~
Read §38.1. Alec decided both questions of §38 and kept item 48.

55. LTM1 at the epoch boundary. At `model_boundary(epoch_boundary=True)` LTM1's relations and their evidence return to their state at the start of the epoch, in sync with LTM0's rows: a relation written during the epoch is withdrawn whatever its endpoints (occurrence, symbol or relation handle), not only when it reaches a withdrawn occurrence as `PresentationMemory.epoch` does now, and a pre-existing relation whose evidence or access count changed during the epoch reverts. Definitions written during the epoch are withdrawn with the rest. What stays: the learned parameters and the optimizer's state; every concept minted during the epoch (words, identities, consolidation prototypes, change sources) with its names, recorded order and its own wiring into existing concepts, since that wiring has changed the weights; and given knowledge. Before building, list the records that wire a minted concept into existing concepts and say whether any of them is an LTM1 relation; if one is, stop and report it in §39, since where that line falls is Alec's. Certificates: after a boundary every LTM1 buffer and index equals its snapshot from the epoch's start, apart from the minted concepts' own records; §37.12's native write survives the epoch without the order error, with its three controls; ten epochs of one fixed document end with the same LTM0 row count and LTM1 relation count after every boundary, with the number of concepts minted per epoch reported, zero after the first; the MM_grammar XOR test passes.
56. The epoch contract in Architecture.md, by append after the "Presentation and epoch boundaries" paragraph: Alec's statement of §38.1 that an epoch re-presents one history to the weights; that the memory of the history (LTM0's rows, LTM1's relations, MTM) returns to its pre-epoch state; and that minted concepts stay, because they are wired into existing concepts and so into the weights. Cite Santoro et al. 2016 §4.2.1 and McClelland, McNaughton and O'Reilly 1995, and mark the superseded definitions clause of the 2026-10-28 paragraph as historical. The XOR test's comment at `_fit_xor` names this contract.
57. After item 55: the float64 discriminator to completion on the final source, then item 49 (the three arms, `forced_configuration_passed` and the freeze), with item 48's paired comparison beside the (false, 0) arm.
~~~

### 38.2 Minted concepts are re-tied by the next epoch; the reset runs between epochs (2026-11-02)

Alec: a concept minted during an epoch was derived from that epoch's LTM, and as long as another epoch follows, it is tied into the new LTM when the same history is read again. The boundary may therefore withdraw a minted concept's grounding in the history (its occurrences, the LTM1 relations that reach it, the definitions that name it) with the rest of LTM0 and LTM1. What stays is the concept, its recorded order and its wiring into other concepts, and the next epoch grounds it again through the recognition paths repaired in §37: the introducing word's single denotation for an identity, the learned prototype for a consolidation, the angular match for a change source, the lexicon for a word. The clause of item 55 that asked Codex to stop if a minted concept's wiring is an LTM1 relation is withdrawn: an LTM1 relation that grounds a minted concept is part of the history and goes.

"As long as another epoch is happening" also fixes where the reset belongs. The lesson driver (`corpus_lessons.py` 241), the math driver (`MathChainTraining.present`, line 36) and the XOR test reset before each epoch's first document, so after the last epoch the memory that grounds the minted concepts is still there. `BaseModel.runEpoch` (`Models.py` 16170 to 16172), the production `train` path, resets after each completed training epoch, the last included, so a model trained through it ends holding minted concepts without the history they were derived from, and a checkpoint written at an epoch's end has the same gap. Read with Alec's sentence, the reset is an operation between epochs: it runs before an epoch begins and never after the last. This is recorded as the reading of his sentence; item 58 makes `runEpoch` agree with the other drivers.

~~~
Read §38.1 and §38.2. This block replaces the block for items 55 to 57 given with §38.1.

55. LTM1 at the epoch boundary. At `model_boundary(epoch_boundary=True)` LTM1's relations and their evidence return to their state at the start of the epoch, in sync with LTM0's rows: a relation written during the epoch is withdrawn whatever its endpoints (occurrence, symbol or relation handle), not only when it reaches a withdrawn occurrence as `PresentationMemory.epoch` does now, and a pre-existing relation whose evidence or access count changed during the epoch reverts. Definitions written during the epoch are withdrawn with the rest. What stays: the learned parameters and the optimizer's state; every concept minted during the epoch (words, identities, consolidation prototypes, change sources) with its names, its recorded order and its wiring into other concepts; and given knowledge. A minted concept's grounding in the epoch's history (its occurrences, the LTM1 relations that reach it, its definitions) goes with the rest, and the next epoch ties it into the new LTM when the same history is read and recognised (§38.2). Certificates: after a boundary every LTM1 buffer and index equals its snapshot from the epoch's start, apart from the minted concepts' own records; §37.12's native write survives the epoch without the order error, with its three controls; ten epochs of one fixed document end with the same LTM0 row count and LTM1 relation count after every boundary, with the number of concepts minted per epoch reported, zero after the first, and every concept minted in the first epoch grounded again by the occurrences of each later epoch; the MM_grammar XOR test passes.
56. The epoch contract in Architecture.md, by append after the "Presentation and epoch boundaries" paragraph: Alec's statements of §38.1 and §38.2 that an epoch re-presents one history to the weights; that the memory of the history (LTM0's rows, LTM1's relations, MTM) returns to its pre-epoch state; that minted concepts stay, because they are wired into other concepts and so into the weights, and are tied into the next epoch's LTM when it re-reads the history; and that the reset runs between epochs, never after the last. Cite Santoro et al. 2016 §4.2.1 and McClelland, McNaughton and O'Reilly 1995, and mark the superseded definitions clause of the 2026-10-28 paragraph as historical. The XOR test's comment at `_fit_xor` names this contract.
57. After items 55 and 58: the float64 discriminator to completion on the final source, then item 49 (the three arms, `forced_configuration_passed` and the freeze), with item 48's paired comparison beside the (false, 0) arm.
58. The reset runs between epochs. `BaseModel.runEpoch` resets after every completed training epoch, the last included (`Models.py` 16170 to 16172). Move the reset to the start of a training epoch, before its first batch, as `corpus_lessons.py`, `MathChainTraining.present` and the XOR test already place it, so that a trained model and an end-of-training checkpoint keep the last epoch's memory, in which its minted concepts are grounded. A resumed incomplete epoch does not reset, as the corpus-cursor rule of the 2026-10-28 contract already requires. Certificates: after `train` for two epochs on one document, the second epoch's LTM0 rows and LTM1 relations are present and every minted concept has at least one grounding record; a checkpoint written mid-epoch and resumed keeps its rows; the first epoch of a fresh model starts from the same state with or without the moved reset.
~~~

## 40. Review of the §39 return (Claude, 2026-11-03): the epoch contract built; the freeze not accepted, because the frozen source diverges

The return was checked on the working tree, which is the frozen source `6bafd5191962`: all 973 files of `output/tests/20261103-010851-e25879/source-manifest.json` hash to the files on disk, and that suite's result is exit 0, 7,262 cases, 882 s, peak 22.77 GiB, torch 2.15.0. The weekly slow-test record is 32 days old by the runner's own warning. Code claims were checked on this tree and, where a certificate should fail on the previous source, on `39da97f4f161` rebuilt by reverse-applying the round's patch (all six `bin/` hashes matched `ports.json`).

**The freeze is not accepted.** Two of the four measured runs on the frozen source diverged, and §39 reports both numbers without saying so.

| Run on `6bafd5191962` | Where | What the receipt shows |
|---|---|---|
| float64 discriminator, (false, .5) | sentence 33, the first of ideas epoch 3, through sentence 69 | Loss 14 on sentence 32, then $1.45 \times 10^{13}$, and between $2.7 \times 10^{10}$ and $6.9 \times 10^{11}$ on every one of the remaining 36 sentences. The epoch reports put the reconstruction norm into the understanding group at 146, $1.32 \times 10^{6}$ and $1.44 \times 10^{13}$ in ideas epochs 1 to 3, and the chooser-to-reconstruction ratio of the Equal layer at $2.8 \times 10^{12}$ in epoch 3 and $5.0 \times 10^{15}$ in epoch 4, with the Sign and generic unary layers above $10^{12}$. |
| float32 arm (true, 0), the production default | sentence 7 of ideas epoch 1, a greedy compose with no departure | Kept compose cost $1.93 \times 10^{10}$ in float32; the reconstruction norm into the understanding group for that epoch is $1.37 \times 10^{10}$ (§39.6's table). Later epochs recover to single digits. |

Every cotangent stayed finite and no preflight raised, so the stops of §33 and §35 did not recur. But a loss of $10^{10}$ to $10^{13}$ from objectives built on bounded charts is the same failure as §35's $3.9 \times 10^{12}$ loss, whose gradients did overflow, and the float64 run shows a trained state that does not come back. The reflection repair of §37.2 removed one producer of non-finite gradients; it did not remove whatever produces these magnitudes. The project's numerical rule asks that divergence be found and fixed at its producer, not survived. Item 59 asks for the producer at both places; the freeze waits on it. Neither run was seeded, so the trajectories are not reproducible as recorded; the item asks for a capture that is.

**The XOR regression test passes about two runs in three.** §39.3 reports three unseeded misses at the .25 floor before the final pass. Six further unseeded runs of the maintained test, in parallel on the frozen source, gave four passes (246 to 255 s each) and two failures that never left the floor (.248648 and .248067 after 900 epochs). Six runs of the same test on the last committed source `d3a083b50a` gave three passes and three failures, all three the seat exhaustion that §37 repaired. So the repair removed the exhaustion; what remains is a fit that, in about a third of initializations, does not learn at all. §39.3 observed that the passing run's only nonzero output gradients were the affine head and the answer-record reader. The test is unseeded by design (one initialization, no lucky retry), so the default `make test` now fails about one time in three on this case. Item 61 measures the rate and the cause; what the test should assert afterwards is Alec's decision.

**The paired proposal (item 48).** On one initialization and the first ideas epoch of the (false, 0) arm, with matching streams, the pre-item-4 uniform proposal kept lower costs than the mixture. The first three sentences agree to three figures; from sentence 5 the mixture's kept $R + A$ rises (23.7, 33.1, 36.5, 80.6, 92.6 among its later values) while the uniform's stays mostly below 5. Median kept $R + A$ is about 10 for the mixture and 2.3 for the uniform, and the chooser gradient variance in the understanding group is $6.4 \times 10^{7}$ against 27.7. Both acquired order-one objects for Felix and Ciao and scored 0 of 8. One pair is an observation, not a result; whether the mixture's rising costs are the proposal's or the same divergence as above is not separable from this pair. Item 60 repeats the pair on four further initializations, after item 59, so that Alec can decide whether the uniform fraction returns to 1.

**Item 46 was mine and wrong.** The historical float32 probe ran `single-0.5.xml`, separate chooser false and jitter .5 (`addendum-9/float32-setting.json`), so §37.7's "jitter probe" was accurate; only the absence of a seed was right in item 46.

**The epoch contract (items 55, 56 and 58): built, with four defects.** The first training boundary snapshots LTM0's occupied rows with their records and document sources, the whole LTM1 state and MTM; every later boundary restores them in place and rebuilds the indexes, so relations of every endpoint type and the epoch's definitions go, and evidence and access counts revert. Learned parameters, optimizer state, minted concepts with their names (`ConceptAllocator.named_identities`), recorded orders, constituent records and weight edges stay. `runEpoch` opens the epoch before its first batch and never resets after the last; a resumed cursor does not reset; the end-of-training checkpoint carries the last epoch's memory and the start state, and a loaded checkpoint trains another epoch from it without minting. The nine certificates of `test_epoch_history.py` pass here and all fail on `39da97f4f161`; the receipts were written by the frozen suite and regenerate byte for byte, apart from one address that varies between runs. The four defects, found by probes beside the certificates:
- **The start state freezes whatever preceded the first training epoch.** It is taken once, at the first training boundary. A validation `runEpoch` before training (or `runTrial`'s baseline pass under `BASIC_RUN_TEST`) leaves its rows in memory, and every later boundary restores them: five rows and one relation of validation history at every epoch, where the previous source had none after the first.
- **Given knowledge supplied between epochs keeps its LTM0 rows and loses its LTM1 relations.** `_begin_rows` keeps given rows through `clear_origin`, but LTM1 is restored wholesale, so a relation from a given row's occurrence to a symbol is withdrawn. §38.1 keeps given knowledge.
- **Restored snapshot rows carry stale recurrence readings.** `reset(keep_learned_readings=True)` clears the recurrence seats' use and identities, and only supplied rows have their readings set to $-1$ (`PresentationMemory.py` 68); a later regrouping put an orthogonal row (cosine 0) into a snapshot row's seat.
- **A reused relation handle can change meaning under a kept record.** Vacant automatic handles are reused (`LTM1.py` 52, `PresentationMemory.py` 171 to 183), and the guard checks only the handle's own records, not records that refer to it. A given row written in the epoch that names `('sym', h)` survives the boundary while relation $h$ is withdrawn; the next relation takes handle $h$ and the row now names it, without an error. The case was constructed by hand; no native producer was found.

**Items 44, 45, 47, 50 to 54: built.** Promotion raises on more than one denotation at the witnessed order, naming the words and competing ids, before it touches the token or a seat; the certificate fails on the previous source. `Interpret.reading` returns the unique retained denotation for a new epoch's first read and writes no definition; reverting just that branch fails its test. The candidate-order repair of §39.4 records the order a class's members derive, preflights the seat and recognises the class at that order, and the boundary reads an older class's order before withdrawing its grounds; the same round's item 55 had introduced the defect, which the certificate reproduces on the intermediate source. The promotion receipts name maintained nodes and hashes. The source-3 note of item 47 is given. GradientFlow names the three sites and the one-point projection; the walk emits the parent on an exact STOP under no_grad as well, with the sentence-scope test parametrised over both; the clock exemption is named and explained. The test edits of this round were read one by one: none accepts more outcomes than before. The 103c fixture's first step now compares exactly with the parent, as item 51 requires, and the remaining steps exactly with 103c. Three loose ends go in item 63.

**The lesson-13 arms (item 49).** Each arm trained 160 sentences in eight epochs; none met the acquisition criterion (bound 0, 0 and 1 of 80; exact 1, 0 and 0); the forced-configuration check passed and is recorded. These are the measurements item 26 asked for. They stand as measurements; the freeze does not, for the reason above.

~~~
Read §40. The epoch contract (items 55, 56, 58) and items 44, 45, 47 and 50 to 54 are approved as built, with the four defects of item 62 to repair; item 46 is withdrawn, §37.7 was right. The freeze of `6bafd5191962` is not accepted: two of its four measured runs diverged. The runtime is unchanged: torch 2.15.0, executorch 1.5.1, pytest 9.2.0, nothing pinned.

59. The producer of the magnitudes. On the frozen source: (a) the float64 (false, .5) run's divergence, which begins between sentence 32 (loss 14) and sentence 33 (loss $1.45 \times 10^{13}$, the first of ideas epoch 3) and never recovers, with the Equal layer's chooser-to-reconstruction ratio at $2.8 \times 10^{12}$ and $5.0 \times 10^{15}$ in epochs 3 and 4 and the Sign layer's above $10^{12}$; (b) the (true, 0) arm's kept compose cost of $1.93 \times 10^{10}$ at sentence 7 of ideas epoch 1, a greedy compose with no departure, with $1.37 \times 10^{10}$ into the understanding group. Seed both runs so the trajectory is reproducible, then record per sentence the loss of each objective and each operator's contribution, and the norms of every parameter before and after each optimizer step from two sentences before each event to two after. Name the parameter or operation whose value grows first and the step that grew it, and repair it at the producer under the numerical rule: no clamp, no mask, no guard that hides the value. A certificate that fails on the frozen source and passes on the repair, and both runs repeated to completion with their largest loss and gradient norm stated in §41. The kept costs of all four arms go in §41 as a per-epoch maximum, so a reader sees a divergence where it occurs.
60. The paired proposal, repeated. After item 59, item 48's pair on four further initializations, seeded, same protocol and fields, with the median and the maximum of the kept $R + A$ per setting beside the mean.
61. The XOR pass rate. Run the maintained MM_grammar XOR test on twenty initializations of the repaired source, seeded per run so each is reproducible, and report the pass count, the epoch of each pass, and for each failure the per-epoch loss and which parameters receive nonzero output gradient; compare with the passing runs. Say whether the failing initializations never route the grammar's representation to the output, as §39.3's observation suggests. Do not seed the maintained test or change its bar: whether it asserts one unseeded fit, a pass rate over several, or a seeded fit is Alec's decision in §41.
62. The epoch start state. (a) The start state is taken once, at the first training boundary, so memory written before training is restored at every later boundary: a validation `runEpoch` before training leaves five rows and one relation of validation history in every epoch, where `39da97f4f161` cleared them; `runTrial`'s baseline pass under `BASIC_RUN_TEST` reaches the same path. Propose in §41 what the start state holds, for Alec to confirm, and certify it with that probe. (b) Given knowledge supplied between epochs keeps its LTM0 rows through `clear_origin` but loses its LTM1 relations, because LTM1 is restored wholesale; given knowledge stays under §38.1, with its relations. (c) Snapshot rows are restored with their recurrence readings after `reset(keep_learned_readings=True)` has cleared the seats' use and identities; only supplied rows are set to $-1$ (`PresentationMemory.py` 68), and a later regrouping put an orthogonal row into a snapshot row's seat. (d) A vacant automatic relation handle is reused when only its own records are absent (`LTM1.py` 52, `PresentationMemory.py` 171 to 183); a kept given row that names `('sym', h)` then names the next relation to take handle $h$, without an error. Reserve a handle while any LTM0 role reference, seat or `relate_idx` key names it, or raise. A certificate for each, failing on the frozen source.
63. Loose ends. `promote_identity` consults remembered names only when no definition is active (`candidates = active or remembered`, `FamiliarTier.py` 501), so a word with an active definition at another order mints even when one remembered identity exists at the witnessed order: certify the case or correct it. `Interpret.reading` falls back silently to the word or the transient when more than one remembered denotation exists, where `promote_identity` raises: the same rule in both. A checkpoint without the epoch keys loads with no start state and the next boundary freezes the loaded memory as one (`Models.py` 4189 to 4196, 4454): raise, or record why not. In `test_presentation_memory`, the unasserted-part test's remaining byte comparison has no boundary between its two reads, and the checkpoint test's `no_epoch_clear` patch on `begin_presentation` no longer guards anything since `runEpoch` stopped calling it: give each a boundary or remove it. One of five unseeded probe runs on the frozen source stopped in training epoch 1 with `ValueError('bind requires a selected reference site')`; four on the previous source passed: reproduce it seeded or report that it does not recur in twenty runs.
64. The freeze, after items 59, 60, 62, 63 and Alec's decision on item 61: the three arms and the discriminator rerun on the repaired source, `forced_configuration_passed`, the default `make test`, and `make test_all` (the slow-test record is 32 days old).
~~~

## 42. Review of the §41 return (Claude, 2026-11-03, evening): item 59's producers found and repaired; item 62 built; item 61's cause misread; the freeze not offered, and the XOR gate resolved by item 2.0's retirement

The return was checked on the working tree, which hashes to `3c0416a6d8f9` (975 files, equal to `mtm-carrier-source.json`), against the receipts under `addendum-10`, and on the frozen source `6bafd5191962` rebuilt from its external copy. Every number quoted in §41.2 to §41.13 that has a receipt was recomputed from it and matches; the exceptions are named below. Certificates were rerun here: the 27 magnitude, centroid, retained-inverse and percept-inverse cases pass on the working tree and fail on the frozen source (`regressions.json`: 27 failed there, by $2 \times 10^{11}$, 16,384.375 and $2 \times 10^{9}$); the 26 epoch certificates pass here and 17 of 17 of `test_epoch_start.py` fail on the frozen source at their defect lines; `test_mm_xor.py` passes 7, the seeded case in 395 s; the `XOR_grammar.xml` gate passed one run of two here.

**Item 59: found and repaired.** Three producers, each on a seeded trajectory with a certificate that fails on the frozen source and passes on the repair:
- the decomposition teacher's division by the parent's energy (`Language._bounded_binary_reconstruction`), which took the chooser's weight norm from 1.0010 to $3.74 \times 10^{8}$ in one step under a gradient of $3.74 \times 10^{11}$; replaced by the fold's absolute error, after which the same seventh surrogate is 46.93;
- the derivation inverse's transport of a recovered operand straight to the preceding node (`DerivationReconstruction.reconstruct`), where sixteen unary inverses doubled a .5 disagreement to 1,048,575; the repair translates between the consumer's operand and the producer's result in the atanh chart, and the frozen numerator 312,245,129,052,160 becomes 305.819 on the same journal;
- the whole-mean pullback at vanishing mass (`FormCentroids.weighted_mean`), 22,022,041,600 where the exact derivative is zero; `_WholeMean` now differentiates the complete mean, with gradcheck, jvp and the compiled pullback passing.

The two repeats complete 160 sentences: float64 (false, .5) with largest loss 1,004.41 and gradient norm 970.65, bit-identical between `ad9bc4c9e8c1` and `1b93eb3a5e5a`; float32 (true, 0) seed 47 with 14,667 and 916,140. On those trajectories the divergence is gone, and no trained state fails to come back. Two qualifications. The repeats are on `1b93eb3a5e5a`, five source revisions before the proposed source, and one of those revisions (`9e22f09939e3`) changed the inverse they exercise; the later repeats outside the repository were killed at 16 and 25 sentences, so item 64's reruns are the only coverage of the freeze source. And the surviving 916,140 is the exact slope $1/\mathrm{total}$ of a whole read at mass $1.23 \times 10^{-7}$: it takes `primitive_properties.members` from 18.28 to a raw 1,740.76, and the standing $[0, 1]$ clamp (`PerceptProperties.project`, called from `_project_sentence_parameters` after every step) puts it back to 18.2757, with the Adam moment decaying over the next five sentences. §41 reports the gradient and not that the projection absorbs the step. Whether a whole of that mass should be readable at all is a decision (item 66). Smaller: §41.6's 17,169,605 and .000221 are in no retained receipt (a temporary directory); `reconstruction.content`'s relative registration against a vanishing observed mass remains a potential producer (the controlled runs reach $2 \times 10^{9}$ to $8 \times 10^{10}$ on legal input), though the native producer was elsewhere; the repaired-source table has two rows, the other two arms being item 64's.

**Item 62: built.** All four defects are repaired with certificates: the start state is captured before any evaluation (`capture_start`, from `runEpoch` after provisioning and from validation batches); given rows keep their relations and dependencies across two boundaries (`retain_given`); readings reset after the restore and are regrouped; a withdrawn handle stays reserved while a role reference, seat or `relate_idx` key names it. The native receipts hold their numbers (five occurrences and one relation restored at both boundaries on the reviewed source, none on the repair; twenty probes, seeds 3 and 18 stopped on the intuition replay mismatch with pairs (0, 1) and (1, 0) against (0, 0), both completing on the repair), on `ad9bc4c9e8c1`, whose boundary code equals the live tree's apart from checkpoint validation. One contradiction to settle: §41.1's sentence that a retained conversational dependency "keeps its address but loses its own assertion authority" holds for LTM0 occurrences (`clear_origin` withdraws every row of the origin) and not for LTM1 relation children: `retain_given` copies the child's evidence, and the certificate (`test_epoch_start.py` 62 to 73) asserts the row survives byte-equal, so it locks in the opposite reading (item 67). Also: handle reservation is reserve-only and never raises, which is §40's remedy, but a given row then keeps naming a withdrawn relation; and the "unstarted baseline" branch has no certificate of its own.

**Item 61: measured; the cause misread.** The twenty-start numbers are exact (17 of 20, median epoch 21, range 8 to 96; the misses' spreads .0553, .0191 and .000417; exactly three gradient destinations in all 3,239 gradient records), and Alec's seed 41001 is in the maintained test with its streams restored. The cause the item asked for is in the receipts and §41.7 stops one step short of it. §41.7 says "the record reader consumed detached grammar features" and draws no consequence. The consequence: under `separateChooser` true the output loss cannot reach the grammar at all (`reader_features` returns `fixed_input`, which detaches; the slot path is cut the same way; the fit runs `forward` with no sentence optimizer), so only the readout bias, the readout weight and `answer_record_reader.weight` ever receive gradient. The grammar's parse of the four examples nevertheless changes from epoch to epoch, through the boundary re-presentation and the scoped streams, with no gradient. The three misses are the initializations whose parse never yields more than two distinct roots (41002 and 41006) or collapses to one (41014), leaving the features with no XOR direction; the passes are the initializations whose drift happened to produce one. The affine-solution-norm contrast in §41.7 is selective: passes 41005, 41010 and 41013 end at 401, 503 and 837, the order of miss 41006's 1,528. So the seeded regression certifies that the heads fit a grammar drift that happened to carry an XOR direction, not that any objective teaches the grammar XOR. That is exactly what item 2.0's single backward changes, and its metrics must re-measure this case (item 68). Provenance: the panel on `7c1ecd5efe64`, where seed 41001 crashed in `ReferenceClose.preview` before the §41.10 correction, is absent from §41.7's history; and `test_mm_xor.py`'s docstring still says 200 epochs.

**Item 60: measured; no dominant setting.** All 24 figures and the pooled statistics recompute exactly from the kept costs; each pair shares its model, parameters, structure, optimizer and streams, and differs only by the uniform fraction. The mixture's mean is lower on three seeds, by .16 to .62, and higher on 42004 by 3.74; the uniform's median is lower on three; the arms agree to full precision for the first 4 to 10 sentences of every seed; the §40 separation did not recur. Alec's decision stands open as item 60 intended; my recommendation is to keep the mixture at .5, the designed proposal, and let item 2.0's metrics judge. Two notes: no held-out cost was measured, because the protocol has no held-out set; the metric list with it is item 2.0's, and it needs the set defined there. The departure win rate is in every receipt as `departure_acceptance` per kind; chooser variance is retained in the receipts and left out of the table. The killed arms of the `1b93eb3a5e5a` and `7c1ecd5efe64` studies are not stated.

**Item 63: built.** The unasserted-part byte comparison is removed, and its §40 text still passes on the live `bin/`, so nothing is hidden; the item-63 guard now intercepts `Forgetting.model_boundary`; the bind-site exception did not recur in twenty seeded runs and the two stops it found are repaired. §43 should name the item-63 certificates (item 69).

**Item 64: not offered.** No freeze hash is proposed, and no `ports.json` or `ports.patch` exists for this round; `3c0416a6d8f9` is named only as the MTM-carrier capture. No default `make test`, no `make test_all`, no arm and no discriminator has run on `9e22f09939e3`, `8c6f4c51da75` or `3c0416a6d8f9`; the last `make test` (`8c6f4c51da75`) failed two cases. The packing failure is repaired in the tree with a certificate (red on `8c6f4c51da75`, green here, three of three unseeded repeats). The `XOR_grammar.xml` gate is the other. Codex's diagnosis holds: the four true pairs' residuals ($2.3 \times 10^{-9}$ to $2.0 \times 10^{-7}$) sit above the exact tolerance ($9.1 \times 10^{-13}$), three fall to the learned chooser, and its weights 2.37, 4.85 and 7.81 on fit and the two activations favour a repeated word. The replacements exist only as external monkeypatches. Of five unseeded runs on the current source, three failed with identical read-backs ("world world", "there there", "loving world", "there there") and two passed, so this is a systematic chooser outcome with some variance, not a one-in-N draw. The decision Codex asked of Alec is already taken on the item 2.0 side: the decomposition teacher is retired and the chooser returns to its initial weights, the residual argmin ([item 2.0 spec §4](2026-11-03-gradient-projection.md); Alec, 2026-11-03), and Codex's own replay shows those weights recover all four multisets. The recommendation is to land that retirement in this round as the gate's repair, rather than change the selection precedence (item 65). Alec to confirm.

**The test edits hide nothing.** Eleven test files changed and two are new since §40; every hunk was read. None loosens an assertion, widens a tolerance, adds a skip or xfail, or narrows an input. The one seed is Alec's decision, scoped with stream restoration. The two checkpoint-envelope edits answer the new required-envelope error and keep their property assertions. The 36 new certificate cases pass here and fail on the frozen mirror, except `test_two_retained_roles_can_reuse_the_same_leaf_introduction`, which guards this round's first eligibility rule and fails on `7c1ecd5efe64` instead. Two notes: the item-27 oracle fixture now agrees with the production residual, so it is no longer an independent reference for that term (it still is for materialization, choice, gradient, optimizer and allocation); and the sentence-journal admission certificate is a control-flow test, with the native two-root forest covered by the `native-stack13` receipts.

**Decisions for Alec, from this review.** (a) The `XOR_grammar.xml` gate: repair by the decomposition teacher's retirement in this round (recommended), or a precedence change. (b) The uniform fraction: keep .5 (recommended) or return to 1. (c) Whether a retained relation child of a supplied fact keeps its evidence (the certificate's reading) or loses it (§41.1's sentence). (d) Whether a whole of mass below a floor is readable at all, or the clamp projection is the designed bound.

~~~
Read §42. Items 59, 62 and 63 are approved as built; item 61 is measured and its cause is to be restated; item 60 is measured and its decision is Alec's; item 64 is not offered, and the freeze waits on items 65 and 73. Alec's decisions (a) to (d) of §42 come with this text or follow it.

65. The XOR_grammar gate, by retirement, after Alec's (a). Retire `reconstruction.decomposition`: `Models._decomposition_teacher_loss` and its registration go; `DecompositionChooser` keeps its initial weights, the residual argmin, and no objective writes them. Port the tests named in the item 2.0 spec §4: the teacher's own tests go (`test_decomposition_chooser.py::test_teacher_uses_resolved_input_rows_when_the_legacy_word_lane_is_absent`, `test_native_attention_field.py::test_decomposition_targets_follow_actual_reversed_candidate_reads`); tests of the static `teacher_loss` go if the method is deleted; tests of the initial behaviour stay. Then run the maintained gate unseeded five times on the repaired source and record the texts of every run; the contract is unchanged, four of four multisets in 400 epochs, unseeded. The §41.13 diagnostic replacements stay external.
66. The whole read at vanishing mass. State in §43 that the 916,140 step of sentence 4 is absorbed by `PerceptProperties.project` and count, over both repeats, the steps the projection absorbed. Propose whether a whole whose mass is below a floor is readable, for Alec's (d); add no clamp and no guard in the meantime.
67. Assertion authority of a retained relation child, after Alec's (c). If a supplied fact's conversational relation children lose their evidence as its occurrences do, `retain_given` strips the evidence of children that were not themselves supplied, and the certificate at `test_epoch_start.py` 62 to 73 asserts the stripped row. If they keep it, §41.1's sentence is corrected to occurrences only. Either way, a certificate for the unstarted-baseline branch, and a sentence saying that a given row may keep naming a withdrawn relation handle under the reserve-only rule.
68. Item 61's cause, restated in §43: under `separateChooser` true the output loss cannot reach the grammar, since `reader_features` detaches and the slot path is cut; the grammar's parse of the four examples drifts through the boundary re-presentation with no gradient; the three misses are the initializations whose parse never yields more than two distinct roots or collapses to one; the seed-41001 regression certifies a readout fit over such a drift, not grammar learning. Record the `7c1ecd5efe64` panel in §41.7's history, with seed 41001's crash in `ReferenceClose.preview` and its repair; correct `test_mm_xor.py`'s docstring to 900 epochs.
69. Receipts. Retain or strike §41.6's 17,169,605 and .000221; label `pair_diagnostics.weights` in `native-xor-reconstruction/report.json` as the capture's learned weights and record the `pairs.py` invocation in `commands.json`; state the killed arms of the `1b93eb3a5e5a` and `7c1ecd5efe64` studies; name the item-63 certificates.
70. `reconstruction.content`'s relative registration. State the native condition under which the observed mass vanishes; if there is none, record why the controlled $2 \times 10^{9}$ to $8 \times 10^{10}$ costs cannot occur on native input. No guard.
71. The repeats on the freeze source. Item 64's four arms and the discriminator run on the freeze source, with the per-epoch maximum kept $R + A$ of all four arms, each run's largest loss and gradient norm, and item 66's count.
72. Item 60's decision is Alec's (b). Record it in §43 with the uniform fraction that results; no further pairs. The receipts' per-kind `departure_acceptance` is the win rate; held-out cost belongs to item 2.0, which defines its set.
73. The freeze, after item 65 and Alec's (a) to (d): one hash, with `ports.json` and `ports.patch` written for the round; the default `make test` and `make test_all` green on it; item 71's runs; `forced_configuration_passed`. Then the round closes, and item 2.0's Codex text follows from the commit of that tree.
~~~

### 42.1 Alec's answers (2026-11-03, late)

On §42's decisions (a) and (b), the `XOR_grammar.xml` gate and the uniform fraction: "Please refine after more feedback from the run." Item 65 therefore waits; Codex proceeds with items 66 to 73, and the gate and the fraction are decided on item 71's receipts. Decisions (c) and (d) stand open. On the item 2.0 side the same evening: the behavioral chains are re-specified as one memory file and one linking function ([item 2.0 spec §9](2026-11-03-gradient-projection.md)), and no learning-rate ordering is built, the ratio living in the model file.

### 42.2 Text for Codex, revised after Alec's answers (2026-11-03, late)

Alec deferred §42's decisions (a), the `XOR_grammar.xml` gate, and (b), the uniform fraction, until he has the feedback of a run; (c) and (d) stand open. The run therefore comes before those decisions and the freeze after them. This block replaces §42's.

~~~
Read §42 and §42.1; this block replaces §42's text. Items 59, 62 and 63 are approved as built; item 61 is measured and its cause is to be restated; item 60 is measured and its decision is Alec's; item 64 is not offered. Alec has deferred §42's decisions (a), the XOR_grammar gate, and (b), the uniform fraction, until he has the feedback of a run; (c) and (d) stand open. The order:

Now: §42's items 66, 68, 69 and 70 as written, and from item 67 the certificate for the unstarted-baseline branch and the sentence on the reserve-only handle rule.

Then item 71, revised: on the candidate source that results, which is not yet a freeze, item 64's four arms and the discriminator, with the per-epoch maximum kept R + A of all four arms, each run's largest loss and gradient norm, and item 66's count of steps absorbed by the projection; and the maintained XOR_grammar gate unseeded five times, with the read-back texts of every run. Write the candidate's ports.json and ports.patch. Report all of it in §43 and stop for Alec's decisions.

After Alec's decisions: item 65 if he chooses the retirement for (a); the rest of item 67 after (c); item 72 after (b), recording the uniform fraction that results; then item 73, revised: the freeze, one hash with ports.json and ports.patch, the default make test and make test_all green on it, item 71's runs repeated on the freeze hash if the source changed after them, and forced_configuration_passed. Then the round closes.

Work on main. No seed pinning beyond Alec's recorded 41001, no resize, no language named in code, fail loud on construction errors; nothing retired silently. Return through Alec for Claude's review; no commit or publication.
~~~

## 44. Review of the §43 return (Claude, 2026-11-04): the measurements exact; the three repairs approved; eight maintained tests broken; the whole memberships held at the cube's faces by momentum; Alec's decisions (a) to (d) and one rule for evidence

The return was checked on the live tree, which equals candidate `855dc08150d8` in all 978 files of `return-candidate-source.json`'s snapshot, and against immutable copies of `3c0416a6d8f9`, `7e84661d7eca`, `8533ae0d7aa5` and `855dc08150d8`, each of which reproduces its manifest. Every number in §43.7, §43.9 and §43.11 to §43.15 was recomputed from the raw archives and matches. `ports.json` and `ports.patch` for `855dc08150d8` are exact against `6bafd5191962`: the patch applied to the base reproduces all 32 after-hashes. The source changed in three steps since §42: `7e84661d7eca` (the MTM restore after LTM0 and LTM1, and the unstarted-baseline certificate), `8533ae0d7aa5` (`generateBudget`, the structural trace capacity and `CompileGuard`) and `855dc08150d8` (the nested return). Every changed test hunk adds; none loosens an assertion, widens a tolerance, adds a skip, xfail or seed, or narrows an input.

**Approved as built.** Items 66, 68, 69 and 70, with the records of §44.1's item 81. From item 67, the unstarted-baseline certificate (red at `test_epoch_start.py` about 273 on `3c0416a6d8f9`'s `PresentationMemory.py`, 41 cases green after) and the reserve-only sentence. The generation budget of §43.8: read once at construction, it feeds the three free-walk sites, and a recorded derivation keeps its full structural capacity and raises on truncation. The compile guard of §43.8: every `torch.compile` in `bin/` now goes through `CompileGuard.guarded_compile`. The nested-return repair of §43.10, red and green in both dtypes. The projection observer only reads.

**What the return missed.**
1. *Eight maintained cases fail.* They are in three unchanged files on `855dc08150d8`, and no selection ran them. `test_compose_records.py` fails five: its stubs set `_walk_budget`, and `_derivation_program` now calls `_derivation_trace_capacity` (`Models.py` about 12952). `test_sweep_compile_repair.py::test_reconstruction_brick_accepts_more_than_default_eight_shapes` passes nine widths through one phase; the guard now raises at the second, as `test_compile_guard.py` certifies, so Alec's no-recompile instruction reverses this test's contract. `test_util_device.py` fails two, because `util.compile` returns the guard's wrapper. Two comments contradict the guard (`util.py` about 811 to 816, `Models.py` about 12269 to 12271), and `doc/Params.md`'s `generateBudget` row was added without a before-and-after port.
2. *The guard has not met a compiled lesson.* The arms and the discriminator pass `--eager`, so item 73's `make test` is its first broad exposure. Two same-phase variations are visible. `FailOnRecompileLimitHit` is not wrapped. A silent construction-time `except` remains in `_compiled_output_walk`.
3. *Returns without a result.* The nested-return branch also changes every return from a child with no result. `completion_result` returns None for one, so the returned meaning now keeps the child's own pair and incomplete reason, where `materialize` wrote (0, 0) with `('open_reference',)`. This follows the ownership rule, and neither pinned trajectory changed; it is undocumented and uncertified.
4. *The hash seed.* Unseeded runs neither pin nor record `PYTHONHASHSEED`: `run-candidate.py` (about 52 to 63) removes it except for the seed-47 default and the discriminator. The failed `8533ae0d7aa5` jitter run's hash seed was therefore never recorded, and §43.10's restoration covered the torch, NumPy and Python random streams only. Sentence 1's loss is identical in the replay (21.800628662109375), while its gradient norm already differs (43.13696249526144 against 43.13696249515101).
5. *Momentum.* Reconstruction-owned parameters, the primitive memberships among them, are stepped by SGD with momentum .9 and no variance normalization (`Models.py` about 2882 to 2889, learning rate .001). `PrimitiveProperties.project` (`PerceptProperties.py` about 246 to 249) clamps the tensor and leaves the velocity, which reapplies a spike at .9 per update:
   - (true, 0): both trial steps of sentence 4 apply its gradient of about 916,140 (optimizer steps 10 and 11, raw 916.14 then 1,740.67). The velocity then clips all 1,609 coordinates on 96 updates through sentence 52, and 1,606 on 122 more through sentence 113.
   - (false, 0): sentence 70's spike clips 886 to 1,607 coordinates on all 91 remaining updates; the last raw update is 7,986.5.
   - (false, .5): sentence 7's spike clips 119 to 890 coordinates on all 154 remaining updates.
   Most of §43.12's "projection changed" counts are these echoes. Single steps at ordinary gradients also flip memberships from 0 to 1 (32 at (false, .5) step 4, 7 at (true, 0) step 440). In separate-chooser mode both trials are trained, and in 59 of the (true, 0) arm's 160 sentences the two trial losses are bit-identical, so the same derivation is trained twice.
6. *Two more spikes.* (false, 0) sentence 104 carries 9,899,375,078.97 on `compose_binary_ops.8.LiftLayer`, from the reconstruction objective: dense SGD with no projection, a raw step of about $9.9 \times 10^6$, then carried by momentum. (false, .5)'s $2.14 \times 10^8$ at sentence 7 belongs to the chooser objective (`reconstruction.compose_score_function`: chooser norm 214,053,611 against reconstruction 230,697), not to reconstruction. Every event at or above $10^8$ is in a float32 arm with `separateChooser` false; (true, 0) peaks at 916,140 and float64 at 17,479. `separateChooser` false is item 2.0's default.
7. *Provenance.* (true, 0) and the float64 discriminator are bitwise repeats of §43.9's runs: all 480 and 160 losses, gradient norms and projection norms are equal. The (false, 0) and uniform pair is a fresh unseeded construction (paired-start model `f493054e` against `e992366b` on `8533ae0d7aa5`), so its differences from §43.9, the $1.05 \times 10^{11}$ included, cannot be attributed to the repair.
8. *An instrument's scale.* `xor-rate.py` (about 85 to 89) subtracts the echoic bank's score at unit weight, but the bank reaches the prediction at about one half. In the misses, `effective_grammar_xor_contrast` is therefore minus half the echoic contrast. `routes.json`'s `positive_absolute_grammar_contrast_epochs` and `final_grammar_xor_contrast` for the misses measure the echoic bank, not the grammar. §41.7's "nonzero effective grammar-score spread ... in every epoch" is an artifact for 41014, whose grammar-score spread is exactly zero in all 900 epochs.
9. *Roots found by reference.* Every LTM1 relation that references a supplied occurrence becomes a root with its boundary-time evidence (`PresentationMemory.py` about 216 to 221). A history relation written after the supply that references the supplied occurrence, such as a familiar-tier membership (`FamiliarTier.py` about 639 to 645), would survive as given knowledge. This bears on (c).

**Corrections to §42 (Claude).**
- *"Never more than two distinct roots"* is wrong for one epoch: 41002 has three at epoch 302 (contrast .000421, MSE .249994). The inference stands and is sharper than §42 put it. `grammar_feature_contrast` is exactly zero in 2,698 of the misses' 2,700 epochs and at least .144 at every passing seed's last epoch. In the misses the heads' only XOR direction is the echoic bank's:
  - `jacobian_xor_contrast` is .000188 to .0015, against .0883 to .923 in the passes;
  - the Jacobian's third singular value is .00047 to .00285, against .116 to .297;
  - head norm times `jacobian_xor_contrast` is .00012 to .0021, against .232 to .677, where an MSE below .20 needs an XOR output above .211.
  §43.2's "near-zero-error affine solutions" are minimum-norm interpolations on a rank-4 Jacobian, whose norms do not separate the groups (up to 837 in the passes). §42 also called §41.7's effective figures exact and missed finding 8.
- *"The Adam moment decaying over the next five sentences"* is wrong; finding 5 gives the optimizer and the tails. The momentum has been SGD since `bfe6a0d76f` (2026-10-03).
- *"The exact slope 1/total of a whole read at mass $1.23 \times 10^{-7}$"* is a magnitude match. The pullback was never captured: the two captured at that mass have weight derivative exactly zero.
- *"Agree ... for the first 4 to 10 sentences"*: 4, 4, 6 and 8, to a float32 ulp.
- *Item 68's "900 epochs" for `test_mm_xor.py`'s docstring* was Claude's error. The module's bars are MM_xor to .20 within 200 epochs, MM_xor to the .25 floor (threshold .26) within 600, and the seeded MM_grammar readout to .20 within 900.
- *The item 2.0 gradient spec's two Adam statements* assumed Adam for reconstruction-owned units. They are corrected there the same day.

**Alec's decisions (2026-11-04).**

- **(a) Retire the decomposition teacher now.** The inverse chooser stays the fixed joint residual argmin over form and both meaning lanes; there is no precedence or tolerance change and no replacement learning rule in 2.5. A learned top-down policy, if one is wanted later, is item 2.0's, credited only by output.
  - *§43.17 settled.* Supervision by the bottom-up operand identities ends. Selection is top-down by the joint residual, independent of the derivation's operand order, with meaning in the score.
  - *Why the teacher failed.* Its target could not be met. Conjunction and disjunction are commutative, so every target ties its transpose exactly and the cross-entropy never reaches zero. The weights drift onto the form-only activations (2.37 on fit against 4.85 and 7.81), whose additive score peaks at the repeated top word ("there there"). Of fifteen receipted unseeded runs, 8 passed, and their final weights fall into two clusters that separate passes from failures exactly.
  - *Two dependences remain*, noted for item 2.0. The pair shortlist is by form cosine times priming, so meaning arbitrates only among the K form-nearest words per side. The decoder's basis includes a bounded snapshot of the forward's own activations.
  - Alec had confirmed this retirement for item 2.0 on 2026-11-03 ([item 2.0 spec](2026-11-03-gradient-projection.md) §7); it now lands in 2.5.
- **(b) The uniform fraction stays .5.** Seven matched first-epoch pairs give no basis to change it. On the six from sources without the §39.7 divergence:
  - .5 has the lower mean in four, and the signed ranks split 11 to 10;
  - the pooled means are 8.53 and 8.16;
  - the spread between initializations exceeds every difference within a pair.
  All seven ran with `separateChooser` false. .5 is item 4's designed default, and item 2.0 extends $\varphi$ to generation, where 1 would make the walks uniform over large menus.
- **(c), and one rule for evidence.** Alec asked whether withdrawing evidence is psychologically consistent, and decided one rule for every place evidence is lost: **evidence belongs to the source that produced it, and lasts as long as that source is held.**
  - *Within an epoch* the history is held. Forgetting removes whole memories by retention (trust, surprise, use and age) and never zeroes a belief, so a belief outlives the loss of its episode.
  - *Testimony* carries its teller's authority on the outer relation only. Content embedded in a supply enters unverified, and belief in it comes from inference over the claim.
  - *At an epoch boundary* the history is released, so everything it produced goes, in LTM0 and LTM1 alike. A row kept only because given knowledge refers to it keeps what it is: its endpoints or slots, tags and place and time bands. Both its evidence lanes are zero, and its access count and surprise return to their values at the epoch's start. Given knowledge keeps its supply's evidence. The epoch's start, intentional setup included, is not part of the history it re-presents.
  - *After the last epoch* nothing resets.
  So the conversational relation child of a supplied fact keeps its address and loses its evidence. §41.1's sentence stands, and §42's contradiction is resolved in its favour. The LTM0 dependencies' access counts and surprise, which `withdraw_origin` keeps today, go too, and so does finding 9's leak. The psychological reading: the boundary is not a psychological event but the re-presentation of one history to the weights. Its counterpart is amnesia with semantic memory intact, where what one was told survives and what one concluded from the lost episode does not. The rule is recorded in [Architecture](../Architecture.md) ("Evidence belongs to its source"). The alternative, belief persistence at the boundary, is in [Future work](../FutureWork.md), to revisit if necessary.
- **(d) A whole is read at its presence** $p = \min(1, t)$, the convention WholeSpace already uses for the same memberships (`Spaces.py` about 20268 to 20272 and 20345 to 20348). There is no floor and no new constant.
  - *What happens today.* Any whole with $t > 0$ enters `MeaningKernels.centroid` and `split_centroid` at full weight one half (`MeaningKernels.py` about 28 and 34). So a nearly absent whole moves a form halfway to its property code, the forward jumps at $t = 0$, and the weight pullback $(\mathrm{code}_r - \mathrm{mean})/t$ grows as presence vanishes (`FormCentroids.py` about 60).
  - *Why not Codex's argument.* The scale-invariance argument of §43.4 treats the memberships as proportions; they are absolute conjunctive membership degrees in $[0, 1]$.

**For item 2.0** (Claude's specs, corrected the same day):
- the gradient spec's Adam statements;
- its §4, where a learned pair policy is now credited only by output's comparison;
- finding 6's spikes under `separateChooser` false, now a prerequisite of item 2.0's measurements;
- the memory spec's promotion rows, which are history at the epoch boundary (§38.2 and the evidence rule);
- the basic-percepts spec's one WholeSpace, which carries the presence read and must not inherit `PerceptProperties.reverse`'s and `reverse_poles`'s division by a byte sum.

### 44.1 Text for Codex (2026-11-04)

~~~
Read §44. Alec decided §42's (a) to (d) on 2026-11-04 and, with (c), one rule for evidence: evidence belongs to the source that produced it, and lasts as long as that source is held. This block replaces §42.2's paragraph "After Alec's decisions". Approved as built: items 66, 68, 69 and 70; item 67's unstarted-baseline certificate and reserve-only sentence; the generation budget, the compile guard and the nested-return repair. The order: item 81 (records); items 65, 67, 74 and 77 to 80 (source); items 75 and 76 (instruments); item 71's runs; item 72's record; item 73.

65. The XOR_grammar gate, by retiring the decomposition teacher (Alec's (a)). The chooser stays the fixed joint residual argmin over form and both meaning lanes. There is no precedence or tolerance change and no replacement learning rule.
a. Retire reconstruction.decomposition: BasicModel._decomposition_teacher_loss, its call in _sentence_path_cost, its registration in train_trial and DecompositionChooser.teacher_loss go. reconstruction.walk_policy stays until item 2.0.
b. The chooser weight is the constant [1, 0, 0, 0, 0]. It leaves every optimizer group (the extend in SymbolSpace.__init__ goes), and no objective writes it. A checkpoint that carries other chooser values loads with the key dropped and reported; those values are never installed. In _bounded_binary_reconstruction the shortlist, the eligibility masks, the joint residual and the exact-pair branch stay.
c. Correct by exact replacement every text that describes the teacher: DecompositionChooser's module docstring, _bounded_binary_reconstruction's docstring ("initially least residual"), the comments above candidate_features and the exact-pair branch, and test_decomposition_chooser.py's module docstring. In GradientFlow.md: the reconstruction owner row, the list of retired teachers, and the bullet "The decomposition chooser learns the true pair", which becomes a dated note of this retirement.
d. Tests, each with before and after text in the return.
   - The teacher's own cases go: test_decomposition_chooser.py::test_teacher_recovers_true_shortlisted_pair_despite_misleading_priming, ::test_absent_target_is_counted_and_has_no_training_gradient and ::test_teacher_uses_resolved_input_rows_when_the_legacy_word_lane_is_absent.
   - Elsewhere only the teacher's lines go:
     - test_native_attention_field.py::test_decomposition_targets_follow_actual_reversed_candidate_reads keeps its field-read assertion in a teacher-free case;
     - the two pair cases of test_reconstruction_magnitudes.py keep their pair_residual assertions;
     - test_operators_final_b.py::test_one_signless_form_per_word_and_two_independent_recovered_lanes keeps its recovery, recomposition and zero-residual assertions;
     - test_free_reconstruction.py drops the term from its registry set;
     - test_generation_catalog.py and observer_probe.py follow the weight out of the optimizer, and the probe computes its shortlist coverage from the compose program;
     - test/operators_run_audit.py (about 229 and 252, run through test_review14_contracts.py about 166) and fixtures/2026-10-03-operators-attention/review17_decomposition_probe.py stop patching the teacher.
   - test_initial_choice_is_exactly_the_residual_argmin_without_rng stays.
   - test_learned_orientation_wins_among_roundoff_exact_pairs_in_any_bank_order and test_operators_round1.py::test_saved_review22_miss_has_exact_argmin_precedence describe a learned state that production no longer reaches. Say for each whether it stays as a property of the mechanism or goes, and why.
   - New certificate: after native training batches of data/XOR_grammar.xml, the chooser weight is exactly [1, 0, 0, 0, 0], it is in no optimizer group, and reconstruction.decomposition is absent from the registry.
e. On item 71's source, run the maintained XOR_grammar fit eight times: unseeded, independent, 400 epochs each, each recording its hash seed (item 80). From each run record:
   - both cases of the shared training: TestXorGrammarLearnsXor::test_xor_class_accuracy and the TestXorGrammarReconstruction case that the addendum-11 driver ran;
   - every read-back text and the chooser weight;
   - the number of read-backs in input order, as an observation only (conjunction and disjunction are commutative, and the contract counts multisets);
   - per epoch, reconstruction.free_percept and reconstruction.free_bytes, which compare decoded leaves by position.
   The contract is unchanged: four of four multisets in 400 epochs, unseeded, and no retry replaces a failure. If a run fails either case, record its read-backs with each operation's selected and best true pair residuals, as pair_diagnostics does, and stop for review.

67. The evidence rule at the epoch boundary (Alec's (c)). At model_boundary(epoch_boundary=True) the history is released, so everything it produced goes, in LTM0 and LTM1 alike. The epoch's start, intentional setup included, is not part of the history it re-presents.
- A row the history wrote that nothing given refers to is withdrawn, as now.
- A row kept only because given knowledge refers to it keeps what it is: its endpoints or slots, context, source text, tags and percept bands. Both its evidence lanes are zero, and its access count and surprise return to their values at the epoch's start, or to a newly written row's values if the history wrote it during the epoch. This covers an LTM0 occurrence through clear_origin and an LTM1 relation that the supply did not write in a supplied fact's closure. Today withdraw_origin zeroes the lanes but keeps the history's access count and surprise, and retain_given copies the child's evidence (LTM1.py about 77 to 79).
- Given knowledge keeps the evidence its supply gave it. A re-assertion is a supply, so it still raises a row already in the start by the per-lane maximum (LTM1.py about 84 to 90; test_epoch_start.py about 102 to 117). A given row's access count and surprise return to their epoch-start values, or a newly written row's, like every other product of the history.
- A relation carries a supply's authority only if the supply wrote it. Record which relations a supply's closing created, when it writes them, under clause_assertions, on the symbol-root path (ClauseRow.write_clause about 548 to 549) and on the occurrence path (today collected at the boundary, PresentationMemory.py about 64). A root found by reference to a supplied occurrence (PresentationMemory.py about 216 to 221) keeps its evidence only if its supply wrote it. A history relation that references a supplied occurrence, such as a familiar-tier membership (FamiliarTier.py about 639 to 645), is history: withdrawn, or kept as an address with zero evidence if given knowledge refers to it. If recording on the occurrence path needs a choice this rule does not settle, stop and report it.
Certificates, each red on 855dc08150d8 and green on the repair:
(i) test_new_given_knowledge_retains_its_relation_and_transitive_dependencies writes its claim as its supply writes it, so that it is given. The child stays a history write. After each of two boundaries the test asserts:
   - the child's endpoints, tags and percept equal to the saved row, both its evidence lanes zero, and its access count and surprise at a new row's values;
   - the claim equal to its saved row;
   - the LTM0 dependency's lanes zero, with its access count and surprise at its epoch-start values;
   - a bank.add that re-asserts the child's pair during the epoch is back at zero after the next boundary.
(ii) A write_clause supply whose outer relation names an existing history relation enters that relation with both lanes zero.
(iii) A history relation written after a supply that references the supplied occurrence is withdrawn at the boundary.
Native controls, green on both: a relation asserted by its own supply and later embedded by another supply keeps its asserted pair across two boundaries; an embedded relation the closing creates enters at (0, 0). The epoch-start, epoch-history, presentation-memory, assertion-provenance and nested-retention selections pass. In doc/Architecture.md, Claude has appended "Evidence belongs to its source (Alec, 2026-11-04)" at the end. Move it, by exact replacement, to follow "Epoch start and supplied relations", and change nothing in it but its cross-references. §41.1 stands.

74. The whole read at its presence (Alec's (d)).
- For a whole of total membership t (weighted_mean's second value), presence is p = min(1, t). WholeSpace.compute_word_property_event and compute_stage0_carrier already weight the same memberships this way (Spaces.py about 20268 to 20272 and 20345 to 20348).
- MeaningKernels.centroid moves the parts toward the whole's mean by p/2, computed as a lerp so that its rails stay exact under assert_cube (Layers.py about 101 and 2525). split_centroid inverts the same equation, (parent - (p/2) mean) / (1 - p/2), before the count split. FormCentroids.unary and fit score their candidates by the same two equations.
- At t = 0 nothing changes: no contribution and no gradient through the consumer.
- Compute p times the mean as WholeSpace does, sum_r w_r code_r / max(1, t), so nothing divides by a small mass. The exact pullbacks are bounded: to weight r, (code_r - parts)/2 for 0 < t < 1 and (code_r - mean)/(2t) for t >= 1; to code r, w_r / (2 max(1, t)). Keep _WholeMean's exact-rail forward where the rails need it.
- No floor, clamp, guard or new constant.
Certificates, in float32 and float64: gradcheck, jvp and the compiled pullback. Use the captured sentence-4 whole (addendum-10/native-mean/whole-pullback-1.pt, t = 1.23037e-7) and a whole at the smallest positive normal mass. For each, the composed form is within t/2 of the parts in every coordinate, and the weight pullback is no larger than |code_r - parts| |g| / 2. Red on 855dc08150d8, green on the repair. A test whose expected form changes at 0 < t < 1 is re-derived from this contract and named in §45 with its old and new values; none is loosened. GradientFlow.md and Architecture.md describe the presence read by exact replacement where they describe the whole mean.

75. Attribution of the large gradients, before and after 74.
- What to observe, at every update that brings new gradient to wholeSpaces.2._owned_bases.what.primitive_properties.members or to compose_binary_ops.8.LiftLayer's parameters:
  - the tensor's gradient norm, split by objective (reconstruction, chooser, output, expectation);
  - for each whole read: t, its weight- and code-pullback norms, the cotangent norm arriving at its mean, its consumer (compose centroid, inverse split, candidate search) and the inverse steps above it;
  - the largest per-coordinate gradient, with its row, byte and read.
- What to run, on 855dc08150d8 and on the source with 74:
  - the seed-47 (true, 0) diagnostic to sentence 4, which reproduces exactly (§43.4);
  - (false, 0) to sentence 104 and (false, .5) to sentence 7, from their archived streams under a recorded hash seed (item 80). A replay that departs from its receipt's losses says where, and reports its own largest inflows instead.
- State which reads and objectives produced 916,140, 1.04842e11, 9,899,375,078.97 and 2.14189e8, and whether 74 removes each.
- A spike that 74 does not remove is named with its read, its formula and its objective. It is not repaired or guarded in this round: every event at or above 1e8 occurred under separateChooser false, item 2.0's default, and item 2.0 takes these up first. If one occurs under separateChooser true, stop for review.

76. The optimizer record; no optimizer change.
- Reconstruction-owned parameters, the primitive memberships among them, are stepped by SGD with momentum .9 and no variance normalization (Models.py about 2882 to 2889). PrimitiveProperties.project clamps the tensor but leaves its momentum buffer, and the read's straight-through rail keeps passing the outward gradient.
- Each projection summary adds, at every update, the tensor's gradient norm before the step and its momentum-buffer norm after. It counts updates with new inflow apart from those changed only by carried velocity, and gives the coordinates still re-clamped at the last update.
- On 855dc08150d8: (true, 0) re-clamps 1,606 coordinates through sentence 113; (false, 0) still re-clamps 886 coordinates at its last update; (false, .5) re-clamps 119 to 890 through its last.
- In §43.4 and §43.12, replace the account with this: the projection bounds the parameter, not the velocity, with these tails. §43.4's proposal for (d) is superseded by Alec's decision.
- Report after item 71's runs whether tails remain with 74. As a proposal only, say whether the velocity of the coordinates the projection clips should also be projected.

77. The maintained tests the budget and the guard broke, with before and after text, each red on 855dc08150d8 and green after.
- test_compose_records.py: its two stubs set _derivation_trace_capacity with the values they gave _walk_budget (five cases).
- test_sweep_compile_repair.py::test_reconstruction_brick_accepts_more_than_default_eight_shapes passes nine widths through one phase, which Alec's no-recompile rule now refuses. Rewrite it to assert the guard's RuntimeError naming the callable at the second width, recorded as a reversed contract.
- test_util_device.py::test_compile_tries_successive_backends and ::test_compile_raises_dynamo_recompile_limit_for_codebook_growth port to the guard's wrapper. State whether the 64-specialization limit contract still serves a purpose.
- Correct the two comments the guard contradicts (util.py about 811 to 816, Models.py about 12269 to 12271).
- Record doc/Params.md's generateBudget row as an edited document in the ports, with before and after.

78. The compile guard's first compiled lesson.
- The arms and the discriminator pass --eager, which sets TheCompileBackend to none. Item 73's make test, with MODEL_COMPILE unset (auto), is therefore the guard's first broad exposure. If it raises "Unexpected model recompilation", repair the producer with a width fixed at construction or a declared phase; never admit a shape.
- Two candidates are visible:
  - The answer walk passes basis, constituents, case_bank and whole_bank as either None or tensors. Its walk_bank width is limit without a structural bank and primed plus limit with one. output_walk_phase keys none of these (CompileGuard.py about 25 to 31; Models.py about 9784 to 9797; Generative.py about 256 to 268).
  - percept_arrival_brick has no owner, so its phase omits training, yet it takes training as a Python bool (Expectation.py about 402; ModelAttention.py about 367 to 370).
- Raise a recompile-limit hit (torch's FailOnRecompileLimitHit) as the guard's RuntimeError, naming the callable and the phase.
- Remove the silent construction-time except in _compiled_output_walk (Models.py about 13472 to 13476), so a failed construction raises.
- At Layers.py about 6442, report util.compile's fallback instead of silencing it with verbose=False.

79. Returns without a result. completion_result returns None for a child with no result (ThoughtStream.py about 182 to 184), so every exhausted or concluded child (Models.py about 6224 to 6236, 6322 to 6333 and 6527 to 6531) now takes §43.10's new branch. Its returned meaning keeps the child's own pair and incomplete reason, where materialize used to write (0, 0) with ('open_reference',). This follows the ownership rule. State it in §43.10. Add a control to test_thought_intuition.py: an exhausted child under ask whose meaning has a nonzero pair. Assert the return record's meaning pair and incomplete reason, and that strict replay accepts the episode. Red on 8533ae0d7aa5, green on 855dc08150d8.

80. The hash seed.
- Every run records its PYTHONHASHSEED. A run with none pinned draws one, records it in its command record and passes it to its process; nothing is selected.
- Establish whether training depends on it. On the restored jitter construction stream, run two sentences under two different recorded hash seeds, and twice under one. Report the first differing loss and gradient norm in §45.
- If training depends on the hash seed, name the hash-ordered iteration that feeds a computation and make its order explicit at the producer. No seed is pinned to hide it.

81. Records from existing receipts; no rerun.
- §43.2. Replace the sentence beginning "Sampled frozen full-reader features" with what xor-ad9bc4c9e8c1's epochs.jsonl establish:
  - grammar_feature_contrast is exactly zero in 2,698 of the misses' 2,700 epochs. It is nonzero only at 41002's epoch 302 (.000421) and 41014's epoch 1 (.413, before its collapse to one root), and at least .144 at every passing seed's last epoch.
  - In the misses the heads' only XOR direction is the echoic bank's. jacobian_xor_contrast is .000188 to .0015, against .0883 to .923 in the passes. The Jacobian's third singular value is .00047 to .00285, against .116 to .297.
  - Head norm times jacobian_xor_contrast is .00012 to .0021, against .232 to .677, where an MSE below .20 needs an XOR output above .211.
  - The exact affine fits are minimum-norm interpolations on a rank-4 Jacobian. Their norms (1,528 to 11,600 in the misses, up to 837 in the passes) do not separate the groups.
- The XOR instrument. xor-rate.py (about 85 to 89) subtracts the bank score at unit weight, but the bank reaches the prediction at about one half, so in the misses effective_grammar_xor_contrast is minus half the echoic contrast. Relabel or recompute routes.json's positive_absolute_grammar_contrast_epochs and final_grammar_xor_contrast for the misses. Correct §41.7's "nonzero effective grammar-score spread ... in every epoch": 41014's grammar-score spread is exactly zero in all 900 epochs.
- Provenance, in §43.11 and §43.12:
  - (true, 0) and the float64 discriminator are bitwise repeats of §43.9's runs.
  - The (false, 0) and uniform pair is a fresh unseeded construction (paired-start model f493054e against e992366b on 8533ae0d7aa5), so its differences from §43.9, the 1.05e11 included, cannot be attributed to the repair.
  - Name the second (false, 0) spike: sentence 104, 9,899,375,078.97, on compose_binary_ops.8.LiftLayer, from the reconstruction objective.
  - Name (false, .5)'s sentence-7 spike as the chooser objective's (reconstruction.compose_score_function: chooser 214,053,611 against reconstruction 230,697).
  - Every event at or above 1e8 is in a float32 arm with separateChooser false.
- Separate-chooser mode trains both trials (Models.py about 21786 and 15204). In 59 of the (true, 0) arm's 160 sentences the two trial losses are bit-identical, so the same derivation is trained twice. Record it; item 2.0's single update per sentence removes it.
- §43.10. The jitter replay's PYTHONHASHSEED was unset in both runs, so the failed run's hash seed was never recorded. The restoration covered only the torch, NumPy and Python random streams. Sentence 1's loss is identical, while its gradient norm already differs (43.13696249526144 against 43.13696249515101).
- test_mm_xor.py's docstring states all three bars: MM_xor to .20 within 200 epochs (test_convergence); MM_xor to the .25 floor, threshold .26, within 600 (test_learns_xor_signal); and the seeded MM_grammar readout to .20 within 900. The 900 asked for in item 68 was Claude's error.

71. The runs, on the source with items 65, 67, 74 and 77 to 80:
- the three lesson-13 arms (true, 0), (false, 0) and (false, .5), and the float64 discriminator, each recording its hash seed. Record the per-epoch maximum kept R + A, each run's largest loss and gradient norm by objective, item 75's attribution, item 76's counts, and the bound and exact answers;
- item 65's eight XOR_grammar fits.
No uniform-fraction control (item 72). Report in §45. Stop for review if a run fails, if an XOR fit fails either case, or if a spike of item 75 survives under separateChooser true.

72. The uniform fraction stays .5 (explorationUniformFraction, item 4's B3 default); no source or configuration change. Record in §45 the seven matched first-epoch pairs: §39.7 on the rejected 6bafd5191962; §41.10's 42001 to 42004 on 9e22f09939e3; §43.9 on 8533ae0d7aa5; §43.14 on 855dc08150d8.
- For each pair give its mean, median and maximum kept R + A, its compose departures kept/eligible, and the first sentence at which its arms differ.
- State that the §43.9 and §43.14 pairs are different initializations, so their opposite signs cannot be attributed to the repair.
- State that all seven ran with separateChooser false.
No further pairs.

73. Then the freeze: one hash, with ports.json and ports.patch, documents included; the default make test and make test_all green on it; item 71's runs on the freeze hash, repeated if the source changed after them; forced_configuration_passed. Return for Claude's review. The round closes when Claude accepts the freeze, and item 2.0's Codex text follows from the commit of that tree.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams, no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review; no commit or publication.
~~~

## 46. Review of the §45 return (Claude, 2026-11-04, evening): the repairs built, with a regression in the presence read's inverse and gaps in the evidence rule; fit 5 reads back its last training root against a dictionary that moved after it; the freeze not offered

The return was checked on the live tree, which equals candidate `cf32ffd84391` in all 979 files of its source snapshot. All 43 files of `ports.json` have after-hashes equal to the live files, and every source and test before-text equals the immutable `855dc08150d8`. Four independent checks covered item 74, item 67, items 65 and 77 to 80, and the receipts; each ran its tests in scratch copies of the trees. Claude replayed fit 5 four times. The numbers of §45.6, §45.8 and §45.9 were recomputed from the raw receipts and match, except where noted below. The review's own receipts are in [claude-review45](../benchmarks/2026-11-04-item2-5-claude-review45/reading.md).

**Approved as built.**
- *Item 65.*
  - The teacher's method, call and registration are gone, with `DecompositionChooser.teacher_loss` and the extend in `SymbolSpace.__init__`; `reconstruction.walk_policy` stays.
  - The weight is a non-persistent buffer: no gradient, absent from `state_dict`, written by no code. A checkpoint's chooser keys are dropped and reported in both loaders (`DecompositionChooser.py` about 21 to 29, `Models.py` about 4599 to 4604), and the optimizer's name remap drops their state.
  - Shortlist, masks, residual, exact branch and tolerance are unchanged. The certificate is not vacuous: it reads the flattened optimizer groups and asserts that `walk_policy` is present. Exceptions: finding 6.
- *Item 67*, the rule's main clauses:
  - history rows withdrawn, and the leak of §44's finding 9 closed;
  - a row kept only by reference keeps its structure with both lanes zero and its creation access and surprise, in LTM0 and LTM1 and through nested chains;
  - re-assertion takes the per-lane maximum with no access increment;
  - the closing's writes recorded at write time on both paths, which also corrects the baseline's unreversed pair for a negative supply;
  - embedded content written recursively enters at (0, 0);
  - the new metadata survives a mid-epoch checkpoint.
  Both of Codex's readings are right. Certificate (ii)'s zero is the start copy: the history that wrote the live row is still held. The re-supply fixture's (1, .3) is the decided per-lane maximum over an intentional setup of (1, 0); the baseline's (.6, .3) lowered the start's lane. On the live tree, `test_epoch_start.py` passes 23; the baseline with the new file fails 6 and passes 17, both native controls among the 17. Exceptions: finding 3.
- *Item 74's forward.*
  - The compose centroid is bitwise `lerp(parts, mean, p/2)` at every mass tested, from 0 to 50, in both dtypes. Its pullbacks are the decided ones to $6 \times 10^{-8}$ in float32 and $2 \times 10^{-16}$ in float64; at $t = 1$ both take the $t \ge 1$ branch.
  - At $t = 0$ it returns the parts, with zero whole gradient.
  - No consumer is left on the unweighted read, and no floor, clamp or constant was added. The inductor pullbacks match the formulas and bounds.
  - `test_whole_presence.py` passes 22 on the live tree and fails 14 of 22 on the baseline; the wider selection passes 149.
  Exceptions: findings 1 and 2.
- *Items 75 and 76.* The instruments' neutrality rests on Codex's two-sentence controls; the attribution's numbers recompute from the raw receipts, except finding 9's. The presence read takes the seed-47 optimizer peak from 916,140 to 54.9. On the 104-sentence (false, 0) replay it takes the one event at or above $10^8$ (sentence 14, chooser objective, $2.74 \times 10^8$) to 1,298.
- *Item 77's ports*, the reversed nine-width contract among them.
- *Item 78:*
  - `FailOnRecompileLimitHit` is wrapped;
  - the arrival brick declares its phase from the call's own `training`;
  - the construction-time `except` is gone, and the fallback at `Layers.py` about 6444 is reported;
  - an oversized surface raises.
  Keying a phase on whether an optional bank is present changes the traced program, not a shape, so it is a declared semantic phase. Exceptions: findings 4 and 5.
- *Item 79*, whose control discriminates on the incomplete reason and the evidence meaning's pair.
- *Item 80.* The hash dependence was found and repaired at its producer, `PrependedSymbols.stage`; set membership stays and the order is admission. The A/B/A records are byte-identical. A seeded XOR_grammar run under hash seeds 11 and 22 gives identical epochs and final parameters. Exceptions: finding 7.
- *Item 81's records*, with the corrections of finding 9 and of Claude's errors below.
- *Item 72's draft* has every requested element; all seven pairs were recomputed and match.

**What the return missed.**
1. *The presence read's inverse loses the rails (blocking).* `MeaningKernels.uncenter` computes its forward from the carrier $\sum_r w_r \mathrm{code}_r / \max(1, t)$, not from the exact-rail mean. Where the parts and every member are 1, it returns $2 - \sum_r w_r/t$, which can be an ulp off 1.
   - In a sample of 500 multi-member wholes built from the captured codes, 7,488 of 48,182 common-rail coordinates miss 1 in float32 at $t \ge 1$, and 3,168 exceed it; on `855dc08150d8`, none.
   - `FormCentroids.unary`'s in-chart test then rejects the right candidate. In float32, `test_form_centroid_numerics`' mixture ($t = 1.0001624$, so $p = 1$ and the decision changes nothing) decodes through the free unary step to the parent's own form (error .1) on `cf32ffd84391`, and to $6 \times 10^{-8}$ on `855dc08150d8`.
   - §44.1 asked to keep `_WholeMean`'s exact-rail forward where the rails need it; the centroid does, the inverse does not. An exact-rail forward with the carrier's derivative, built as the centroid's is, has the same gradients, keeps every rail, and at $t \ge 1$ is bitwise $2\,\mathrm{parent} - \mathrm{mean}$ (Claude's probe).
2. *Two presence certificates are weaker than §45.4 says.* The compiled cases compare compiled with eager and check finiteness, never the decided formula or bounds. So `[captured]` passes on `855dc08150d8`, whose compiled and eager pullbacks agree at about $10^6$, and the cases compile with aot_eager, not inductor. The code-pullback assertions use `assert_close`'s default atol ($10^{-5}$ in float32), above the expected values at small mass: a zero code pullback passes in all four cases. The implementation is right there; the certificates do not show it.
3. *Three gaps in the evidence rule.*
   - A supply withdrawn mid-epoch makes the boundary raise `KeyError` (`PresentationMemory.py` about 127 to 128). The withdrawal can come from `store_truths`' resubmission through `clear_origin(ORIGIN_USER)` (`Models.py` about 7089), the stateless prune at load, or `forgetProtectOrigins` excluding given origins. Without the crash, the withdrawn supply and its LTM1 rows would be restored at the boundary, though their source is no longer held.
   - The closing writes companions and identity-bearing nodes with their own pair, `value.evidence` (`ClauseRow.py` about 529 to 539). That is the identification pole, "never the source's authority" (`ClauseJournal.py` about 81), yet `PartClosing` records it and `retain_given` keeps it as the supply's pair. A companion relation at (.5, .1) survives the boundary at (.5, .1), where the baseline withdrew it. An identity-bearing embedded child survives on both trees. Only the outer relation carries the supply's authority.
   - The access and surprise restore sits in the shared `withdraw_origin` (`Layers.py` about 10197 to 10202), so it runs off the boundary too:
     - a mid-epoch withdrawal resets (8, .9) to (1, .2);
     - a row that `_epoch_rows` does not describe raises `ValueError`;
     - the load post-hook runs before `_epoch_rows` is replaced (`Models.py` about 5183 and 4475), so a reused instance raises a `ValueError` that the `except RuntimeError` there does not catch;
     - mid-epoch checkpoints from before the repair raise, as `LTM1.py` about 79 does.
   Two notes go with these:
   - A history row that shares a document with a later supply becomes USER by document. It keeps its history values across the boundary on both trees ((.4, .4), access 6, surprise .9), because `PresentationMemory.py` about 116 to 118 keeps survivors outside the start with their live values. Under the rule that is a leak.
   - Raw `('sym', h)` references keep §43.1's reserve-only rule, so a given row can end naming a withdrawn relation. Under the evidence rule, a relation that a given row names is referred to by given knowledge, and is kept as an address with zero evidence. This follows from Alec's rule and supersedes the reserve-only sentence for given rows.
4. *The byte axis is a sentence's, not a word's.* Item 78 fixed the primed bank's surface axis at the input capacity plus EOS, 4,097. The slab is int64 with a Boolean mask, 36,873 bytes per candidate, over the seen ledger plus 16 candidates.
   - On XOR_grammar it is 14.6 MiB per sentence, against 4.5 KB before.
   - On `data/BasicModel.xml`, the Makefile's default, it is about 355 MiB per sentence, computed from the configuration. It is retained per sentence slot in `_sentence_understandings` (up to 32), re-expanded as int64 in `FormCentroids.candidates` and padded and concatenated again in `walk_bank`.
   - On XOR_grammar, in one-thread eager runs, the byte axis alone adds 7.5% per epoch and the round 11.7%.
   The same missing capacity breaks a ninth maintained case, which §44 did not list: `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_records_truths` raises the guard's error in `_reconstruct_sentences`, because `_ar_target_word_bytes` goes from 5 to 6 bytes in one phase. It fails on `855dc08150d8` too.
5. *An undisclosed change.* On `855dc08150d8`, `backend="auto"` raised `InvalidBackend` when the answer walk was constructed, and the silent `except` then ran it eagerly, so the default never compiled it. It now compiles with inductor, without `util.compile`'s fallback chain. That enlarges item 73's first compiled exposure, and §45.5 does not say so.
   - The guard's limits are not "a 64-entry global cache". In torch 2.15 `recompile_limit` (raised from 8 to 64) applies per code object, and `accumulated_recompile_limit` stays at 256 per code object across instances.
   - With `fail_on_recompile_limit_hit`, either limit now surfaces as "Unexpected model recompilation". That includes a long test worker filling its cache with independent models, with no churn at all.
6. *Ports beyond the teacher's lines.*
   - Five of `observer_probe.py`'s assertions that are not the teacher's changed: containment per band, cover-only sentence rows, `form_containment`, the live percept width, and `identity_recovered`. Its replacement (about 92) repeats the check at about 71, so the identity check is deleted, not ported.
   - The learned-orientation case was retired whole, though its fixed-weight content holds over all six bank orders: the minimum-residual exact pair wins, the multiset is always {b0, b1}, and the perturbed b2 never wins. That is a production property of the multiset gate and has no certificate now.
   - Under fixed weights, review-22's exact branch cannot change a selection, so its name overstates what it tests.
   - `Language.py` about 13714 still says "The pair chooser owns orientation".
   - No test feeds the optimizer's remap a saved state that contains the chooser.
   - `test_compile_tries_successive_backends` dropped its identity check; the mocked eager backend returns the same function, so a sentinel is needed to restore it.
7. *Two hash orders remain.*
   - `WordIdentity.py` about 235 to 243 iterates a set of byte strings into the queue that decides `_mint`'s roles in a collision cascade.
   - `Language.py` about 11741 to 11743 and 11796 to 11812 builds the S-tier host layers in the order of a set of method names. So `syntacticLayer.layers` and `_host_layer_registry` depend on the hash seed. On XOR_grammar this is observed in `named_modules`. It is harmless while those layers have no parameters, but it can reach the penalty sum (about 11255) and the first-match fallback (about 11463 to 11466).
8. *The streams are not archived.* `xor_gate.py` saves each fit's initial streams, which makes every fit replayable, but only under `/private/tmp/item25-review44/`; the receipts neither archive nor hash them. Claude's copies are in [the review receipts](../benchmarks/2026-11-04-item2-5-claude-review45/streams/SHA256SUMS).
9. *Records.*
   - §45.6: 258 chooser calls agree; the first difference is the 259th (index 258).
   - §45.8's seed-47 paragraph names the wrong read. The step-11 peak comes from read 20907, the second trial's read of the same whole; read 17257 is step 10's. Both share the mass, the pullback of 1,724,355.27, the contribution of 916,140.06 and the peak at [0, 63]. Read 20907's normalized-mean cotangent is .0264, against read 17257's 591.34, and the two pullbacks are identical. So the gain is $1/t$ on the cotangent's component along $\mathrm{code} - \mathrm{mean}$, and "591.3445 becomes 1,724,355" does not hold. The baseline and the repair also ran different `attribution.py` versions (c9667b4c and 41c4d53e).
   - The post-repair maximum of 1,298 on the (false, 0) replay is at sentence 63, under the chooser objective. Of it, 9.46 reaches the whole-2 memberships and the rest goes to parameters the receipts do not name.
   - §45.9's "absent from shortlist" is "absent from the eligible pairs": the diagnostic sees only eligible pairs and saves no shortlist.
   - Fit 5's residuals are bit-identical to §41.13's receipt on `8c6f4c51da75`, a different fit, so they belong to the fixture. All eight fits' `epochs.jsonl` are byte-identical: the free terms are constant across epochs and fits and cannot locate a failure. Fits 6 and 8 completed both cases before their cancellation was recorded.
   - Per-fit time fell to 397 to 401 seconds, from per-source means of 497 to 508. Those older runs used `MODEL_COMPILE=eager`, ran one at a time and had no observers, so no pair isolates the source.
   - §41.7's "two passes also finish without four" is three: 41005, 41010 and 41013. Two of them never reach four.
   - Item 72's "first difference" for §43.9 is the first unequal kept R + A (sentence 2); the training loss already differs at sentence 1.
   - The evidence check saw one failure that did not recur: `test_epoch_history.py::test_native_train_keeps_last_grounding_and_recognizes_it_across_ten_epochs` (symbol 14 not grounded after the reload). It failed once in the repository directory under hash seed 0 and passed 21 times after.

**Fit 5, replayed (Claude).** Fit 5 replays exactly. `xor_gate.py` saves each fit's torch, NumPy and Python streams before construction, and once the hash order is repaired nothing else in the process varies. Fit 5's streams were restored under its recorded hash seed in a copy of `cf32ffd84391`. Four such replays each reproduce `epochs.jsonl`, `readbacks.jsonl`, `reading.json` and `pair-diagnostics.json` byte for byte.

*The walk at the final read-back* ([walk summary](../benchmarks/2026-11-04-item2-5-claude-review45/replay/walk-summary.json)):
- "hello world" and "loving world": the disjunction's inverse is exact by the walk's rule (form residual 0; first meaning lane $4.65 \times 10^{-9}$ and $2.5 \times 10^{-13}$, each at most $10^{-8}$). It is the only admissible move, so the walk is forced.
- "hello there": form residual 0, but the first meaning lane misses by $4.08 \times 10^{-7}$ (RMS $6.4 \times 10^{-4}$). Conjunction, disjunction and STOP all stay admissible, and the learned policy decides: STOP $-0.788$, disjunction $-0.867$, conjunction $-1.586$. The approximate STOP emits the nearest admitted terminal, exactly "hello"'s code. One leaf for two words fails coverage, so the sentence is unavailable.
- "loving there": also inexact ($3.74 \times 10^{-8}$); the policy takes the disjunction by .142.
The pair chooser is not involved: it selects the true pair in all four sentences. Nothing is truncated by the budget.

*The teacher never trained this state* ([teacher summary](../benchmarks/2026-11-04-item2-5-claude-review45/teacher/teacher-summary.json)). A second replay records every call of `reconstruction.walk_policy` (`BasicModel._decomposition_walk_teacher_loss`): 800 calls over both trials, all four sentences and all 400 epochs. Of their 3,200 root states, none is inexact; every target operation recomposes its root with residual exactly 0. In the last epoch the target leads STOP by 6.27 to 6.48 at every exploit root. Those large logits come from the menu's exact feature. At the read-back's inexact roots the feature is off, and the policy is close to indifferent.

*The read-back root is the last training root; the dictionary moved after it* ([probe](../benchmarks/2026-11-04-item2-5-claude-review45/probe/probe.json), [comparison](../benchmarks/2026-11-04-item2-5-claude-review45/states/comparison.json)).
- A third replay recomputes the read-back root's menu three ways: as the walk does, without the forward's constituent snapshot, and as the teacher does, with the teacher's bank. All three give the same residuals, so neither the constituents nor the read-back's larger bank cause the miss.
- A fourth replay saves the last training epoch's roots and bank beside the read-back's. The four read-back roots equal the last exploit trial's roots byte for byte, and those equal the compose tree's parents. The explore trial composed three of the four with the conjunction. Both trials saw the same bank, and its word codes moved in the final update: "there" by up to $6.82 \times 10^{-4}$, "loving" $4.32 \times 10^{-4}$, "hello" $2.83 \times 10^{-4}$, "world" $2.6 \times 10^{-5}$.
So the read-back decodes the understanding composed before the last update against the dictionary after it. §41.13 found the same on `8c6f4c51da75`, and these residuals are that receipt's to the bit.

Fit 5 therefore fails because its read-back decodes a root that the moved dictionary no longer recomposes exactly, and at such a root its walk policy, trained only at exact roots, chooses close to a coin toss. The four passes do not show that such roots read back. Which producer carries the root unchanged across the final update is not established.

*Two exactness rules.* The pair search calls a pair exact when its joint residual is at most $(8\epsilon)^2$, about $9.1 \times 10^{-13}$ in float32 (`Language.py` about 13810). The walk admits only the exact pair when form and both lanes are each at most `EXACT_RESIDUAL_TOLERANCE`, $10^{-8}$ (`decoder_eligibility`, about 13913; decided in §26.3). The pair diagnostic's "exact" column is the first rule; the walk uses the second.

**Corrections to §44 (Claude).** These came from Claude's §44 and §44.1, and Codex carried several into §43:
- *(false, 0)'s tail.* "886 coordinates at its last update" (item 76) is wrong: the last update re-clamps 923, and 886 is the minimum (sentences 123 to 130). "886 to 1,607 coordinates on all 91 remaining updates" (finding 5) is also wrong: 90 updates follow the spike, re-clamping 886 to 1,569; 1,607 is the spike's own step.
- *(true, 0)'s tail.* "1,609 coordinates on 96 updates" is 95 after both trial steps. "Through sentence 113" stops too soon: every update through sentence 160 still re-clamps 231 to 301 coordinates.
- *The affine fits' norms.* §44 and item 81 said the norms (1,528 to 11,600 in the misses, up to 837 in the passes) "do not separate the groups". By the numbers given they do: at the final epochs the misses are at least 1,528.32 and the passes at most 837.49. What stands is that a minimum-norm interpolation's size is not a grammar contrast. §45's rewrite ("do not distinguish useful grammar signal") is unsupported too.
- *Two labels.* 9,899,375,078.97 is the all-parameter optimizer norm at step 104; `compose_binary_ops.8.LiftLayer`'s own figure is 9,899,375,028.95, an epoch-2 accumulation. "Chooser 214,053,611 against reconstruction 230,697" are epoch-1 accumulations, and no receipt key is named `reconstruction.compose_score_function`.

**Nothing waits on Alec now.** Item 83 below either finds a stale value that the evaluation should have recomputed, and refreshes it at its producer, or it stops. If it stops, Alec decides how a two-word root whose forms are exact and whose evidence lanes drifted after it was read should be decoded. No tolerance, precedence or eligibility rule changes until then. The walk policy's untrained states remain item 2.0's, which retires its teacher.

### 46.1 Text for Codex (2026-11-04, evening)

~~~
Read §46. Items 82 to 91 come first, in this order: 90 (records), 85, 82, 86, 87, 88 and 89 (source), 83 and 84 (fit 5 and its peers), 91; then §44.1's items 71 to 73 on the repaired source. Approved as built: items 65, 67, 74's forward, 75, 76, 77, 78, 79, 80 and 81, except where §46 names a finding and the items below repair it. Claude's receipts are in doc/benchmarks/2026-11-04-item2-5-claude-review45, called claude-review45 below.

82. The presence read's inverse keeps the rails (item 74).
- MeaningKernels.uncenter computes its forward from the carrier. Where the parts and every member are 1 it returns 2 - sum(w_r/t), which can be an ulp off 1, and FormCentroids.unary's in-chart test then rejects the right candidate. In float32, test_form_centroid_numerics' mixture (t = 1.0001624, so p = 1) decodes through the free unary step to the parent's own form on cf32ffd84391 (error .1) and to 6e-8 on 855dc08150d8 (claude-review45/presence/free_unary_step.py).
- Build uncenter as centroid is built: the forward (parent - (p/2) mean) / (1 - p/2) from the detached exact mean and presence, plus the carrier expression's derivative. At t >= 1 the forward is then bitwise 2 parent - mean (claude-review45/presence/fix_probe.py). No new constant.
- Certificates, red on cf32ffd84391 and green after, in float32 and float64:
  - uncenter and split_centroid at common rails, with multi-member wholes at t >= 1 and at 0 < t < 1;
  - the free unary decode above;
  - gradients equal to cf32ffd84391's.
- test_whole_presence.py: the compiled cases also assert, on the compiled outputs, the decided pullback, the |code_r - parts| |g| / 2 bound and the t/2 forward bound, under inductor as well as aot_eager. The code-pullback assertions use rtol with atol 0, or a tolerance scaled by t; the default atol passes a zero pullback.
- GradientFlow.md: restore, by exact replacement, the _WholeMean derivative and its link to test_form_centroid_numerics.py for the literal weighted_mean read.
- Generative.py about 68 passes whole_bank.read(left), a literal (mean, total). Read the presence carrier there, or correct result.json's statement that every candidate path uses it.

83. Fit 5's read-back (§46).
- Archive the eight fits' initial-random-stream.pt files with their hashes in your receipts; Claude's copies are in claude-review45/streams.
- On fit 5's exact replay, the read-back's four roots equal the last exploit trial's roots byte for byte, while the bank's word codes moved in the final update by up to 6.82e-4 (claude-review45/states). Find the producer:
  - which stored or staged value carries the root, or its word values, unchanged across the final update;
  - which update moves the bank's codes;
  - whether the evaluation composes the sentence at all.
- If the root reaches the read-back through a value staged before the update that the evaluation was meant to recompute, refresh it at that producer. Certificate, on an immutable copy of cf32ffd84391 with only that repair, red before and green after: on fit 5's replay, all four read-back roots exact by the walk's rule and all four sentences read back.
- Otherwise the read-back decodes, by design, a stored understanding against a dictionary that has moved since it was read. Stop and report: how such a root decodes is Alec's decision. Change no tolerance, precedence or eligibility rule.

84. Replay fits 1 to 4 and 6 to 8 from their streams on cf32ffd84391, recording the read-back walk as claude-review45/scripts/replay_gate.py does.
- For each fit and sentence, report whether its read-back root is exact by the walk's rule, and the policy's margin where it is not.
- Complete fit 7's reconstruction case.
This is an observation: no replay is a fit of item 71, and the gate is unchanged.

85. Every run that does not select a seed records its initial torch, NumPy and Python streams with its hash seed, archived and hashed in its receipts, so that a failure can be replayed exactly. Recording selects nothing.

86. A word's surface capacity (item 78).
- The primed bank's surface axis is the input capacity plus EOS: 4,097 int64 bytes and a mask per candidate, over the seen ledger plus 16. That is 14.6 MiB per sentence on XOR_grammar, against 4.5 KB before, and about 355 MiB on data/BasicModel.xml. The slab is:
  - retained per sentence slot in _sentence_understandings (Models.py about 21842);
  - re-expanded as int64 in FormCentroids.candidates (about 214);
  - padded and concatenated again in walk_bank (Generative.py about 261).
- A word's surface gets its own capacity, fixed at construction: an existing constructed word capacity if there is one (name it), otherwise one parameter with a power-of-two default, recorded in doc/Params.md.
  - A longer surface raises.
  - The capacity sizes the primed slab, walk_bank's lexical slab and inputSpace._ar_target_word_bytes.
  - Keep bytes as uint8 where they are bytes, and do not retain or re-expand a slab its consumer does not read.
- The ninth maintained case the guard broke has this producer. test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_records_truths raises in _reconstruct_sentences when _ar_target_word_bytes goes from 5 to 6 bytes in one phase, on 855dc08150d8 and on cf32ffd84391. It turns green with its assertions unchanged.
- Certificates:
  - no recompilation across sentences of different word lengths in one phase;
  - the slab's bytes per candidate and per sentence on XOR_grammar and data/BasicModel.xml, before and after;
  - XOR_grammar's seconds per epoch against cf32ffd84391's, under the same backend and thread count.

87. The evidence rule's gaps (item 67).
a. A supply withdrawn mid-epoch is not restored at the boundary, in either memory, and the boundary does not raise; today it raises KeyError at PresentationMemory.py about 127 to 128. The withdrawal can come from store_truths' resubmission through clear_origin(ORIGIN_USER), the stateless prune at load, or a forgetProtectOrigins without the given origins. Drop both memories' supply snapshots together.
b. Only the outer relation carries the supply's pair. Every other write of the closing enters the start at (0, 0): companions, identity-bearing nodes and reference requests included (ClauseRow.py about 529 to 539). Their own pair is the identification pole (ClauseJournal.py about 81).
c. The access and surprise restore runs only at the boundary (PresentationMemory._begin_rows, about 115); withdraw_origin's other callers, store_truths and the stateless restore, keep what they had. None of these raises:
   - a row that _epoch_rows does not describe;
   - a reused instance's load (the post-hook runs before _epoch_rows is replaced, Models.py about 5183 and 4475);
   - a mid-epoch checkpoint saved before this repair.
d. A history row that becomes USER through a later supply's document does not keep its history evidence across the boundary (PresentationMemory.py about 116 to 118).
e. A relation that a given row names by a raw ('sym', h) reference is referred to by given knowledge. It is kept as an address with zero evidence, not withdrawn; the reservation stays for seats and relate_idx keys. Replace §43.1's reserve-only sentence for given rows by exact replacement.
f. supplied_occurrence's asserted pair becomes required, with no default to the live pair (PresentationMemory.py about 79); port the tests that relied on the default. Remove authority['relations'], which is written and never read, or make it the record the boundary reads.
Certificates:
- one for each of a to e, red on cf32ffd84391 and green after;
- one for a mid-epoch LTM0 dependency reached through an LTM1 endpoint, which certificate (i)'s fixture no longer covers.
The epoch-start, epoch-history, presentation-memory, assertion-provenance and nested-retention selections pass.

88. Ports and records of items 65, 77 and 78.
- observer_probe.py: restore the identity_recovered check, whose replacement (about 92) repeats the check at about 71. For each of the other four changed assertions, restore it or state its port with before and after.
- Port the learned-orientation case's fixed-weight content over all six bank orders: the minimum-residual exact pair is selected, the multiset is {b0, b1}, and the perturbed b2 never wins.
- Rename review-22's case for what it now tests: multisets and exact recomposition on captured roots.
- Correct Language.py about 13714 ("The pair chooser owns orientation").
- Add a test that feeds the optimizer's name remap a saved state containing the chooser.
- Restore test_compile_tries_successive_backends's identity check with a sentinel eager backend.
- In §45.5 and the compiler readings:
  - correct "the 64-entry global cache": in torch 2.15, recompile_limit applies per code object, raised to 64, and accumulated_recompile_limit is 256 per code object across instances;
  - record that the answer walk now compiles under auto with inductor, where 855dc08150d8 ran it eagerly after a silent InvalidBackend.
- Say whether one make test worker can reach the accumulated limit with independent models. If the guard can raise on cache capacity rather than churn, its error names the limit that was hit.

89. The two remaining hash orders (item 80), each ordered at its producer, with a certificate under two hash seeds:
- WordIdentity.py about 235 to 243: a set of byte strings feeds _mint's collision queue;
- Language.py about 11741 to 11743 and 11796 to 11812: the S-tier host layers are built in a set's order, so syntacticLayer.layers and _host_layer_registry follow the hash seed.

90. Records, by exact replacement in §45 and in the sections it corrected:
- §45.6: 258 chooser calls agree; the first difference is the 259th call (index 258).
- §45.8, seed 47:
  - The step-11 peak's read is 20907; read 17257 is step 10's. Their shared numbers stand.
  - Read 20907's mean cotangent is .0264 against 17257's 591.34, with identical pullbacks, so the gain is 1/t on the cotangent's component along code - mean.
  - Correct seed47-reading.md and seed47-paired-attribution.json the same way, and record the two attribution.py versions.
- §45.8, (false, 0): the post-repair maximum is at sentence 63, under the chooser objective; 9.46 of its 1,298 reaches the whole-2 memberships.
- §45.9:
  - "absent from the eligible pairs" for "absent from shortlist";
  - the four residuals are §41.13's receipt's to the bit (8c6f4c51da75, another fit), so they are the fixture's;
  - all eight epochs.jsonl are byte-identical, so the free terms cannot locate a failure;
  - fits 6 and 8 completed both cases before their cancellation was recorded.
- Wherever timing is cited, state that the older XOR fits ran MODEL_COMPILE=eager, one at a time, without observers.
- §41.7: three passes finish without four roots (41005, 41010, 41013); two of them never reach four.
- §43.2: the final affine-fit norms separate the groups (misses at least 1,528.32, passes at most 837.49), but a minimum-norm interpolation's size is not a grammar contrast. §44.1's sentence and §45's rewrite are withdrawn.
- §43.4 and §43.12:
  - (true, 0) re-clamps 1,609 coordinates on 95 updates after both trial steps, and its tail continues through sentence 160 at 231 to 301 coordinates.
  - (false, 0)'s last update re-clamps 923 (886 is the minimum, sentences 123 to 130); after the spike come 90 updates at 886 to 1,569.
  - The Lift figure is the step-104 all-parameter norm; Lift's own, 9,899,375,028.95, is an epoch-2 accumulation.
  - The chooser and reconstruction figures are epoch-1 accumulations; give their receipt keys.
- Item 72's draft: say which measure its first difference is (kept R + A); §43.9's training loss already differs at sentence 1.

91. Item 75's attribution in item 71's runs names every parameter, beside the two monitored tensors.

Then come §44.1's item 71, item 72's record in your return, and item 73's freeze. They run on the source with items 82 and 86 to 89, and with 83's repair if it makes one. Item 71 is the three lesson arms and the float64 discriminator, each recording its streams, then eight fresh unseeded XOR_grammar fits.
- If item 83 stops for Alec, run the lesson arms and the discriminator anyway, and leave the eight fits, item 72's record and the freeze until his decision.
- test_epoch_history.py::test_native_train_keeps_last_grounding_and_recognizes_it_across_ten_epochs failed once in the evidence check and passed 21 times after. If it fails again, capture it with its streams and stop.
- Stop for review on §44.1 item 71's conditions, on item 83's, and if the guard raises in make test.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review; no commit or publication.
~~~

## 48. Review of the §47 return (Claude, 2026-11-04, night): the repairs built; item 83 decided by Alec, an MTM identity is computed from its references; the panel stopped on an observer defect; a headless composition takes identity 0; the freeze not offered

The return was checked on the live tree, which equals candidate `9a82e07d458e` in all 984 files of its source snapshot. Against `cf32ffd84391`, 53 files changed: 16 in `bin/` (`WordSurfaces.py` new), the two model files in `data/`, and 35 in `test/` (four new). Claude read the source diff in full and reran two selections on the live tree under `PYTHONHASHSEED=0`, torch 2.15.0:
- items 82, 86, 88 and 89: `test_whole_presence`, `test_form_centroid_numerics`, `test_word_surface_capacity`, the ninth case `test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_records_truths`, `test_hash_order_producers`, `test_compile_guard`, `test_decomposition_chooser`, `test_objective_optimizer_resume` and `test_observer_identity`: 92 passed in 63 seconds;
- item 87: `test_epoch_start`, `test_epoch_history`, `test_presentation_memory`, `test_assertion_provenance`, both nested-retention files, `test_forgetting` and `test_ternary_truth_store`: 135 passed in 82 seconds.

The source hash was unchanged after both.

**Approved as built.**
- *Item 82.* `uncenter`'s forward is $(parent - (p/2)\,mean)/(1 - p/2)$ from the detached exact mean and presence, plus the carrier expression's derivative (`MeaningKernels.py` about 38 to 49), as specified. `native_recompose`'s unary path reads `read_presence` (`Generative.py` about 68). GradientFlow carries both contracts again.
- *Item 84*, as observations. Fit 6's walk was checked against its summary (finding 1).
- *Item 85.* The recorder installs no state, refuses to overwrite, and requires a recorded hash seed. The default runner draws and passes each worker's hash seed (`bounded_tests.py` about 489), so a failed case's setup-entry streams replay exactly.
- *Item 86.* One capacity, $64 = 2^6$, with a separate EOS position: 130 bytes of `uint8` and mask per candidate, against 36,873. The target table no longer follows the longest word, so the ninth case passes unchanged. The bank's byte table no longer truncates a long surface silently (it cut at `raw[:n]` before; it raises now). Two observations, with no change now:
  - The capacity is read twice, in `Models.py` about 2163 and `Spaces.py` about 13119.
  - Item 2.7's corpus will raise at 64. Across `data/corpus/assembly-01` and the other text fixtures in `data/`, 144 percept runs exceed 64 bytes:
    - 10 ASCII letter runs, the longest 126 bytes;
    - 134 punctuation or high-byte runs, the longest 453 bytes (UTF-8 CJK text, one primitive run under `Meronomy.percept_spans`).

    Digit runs reach 41. The corpus round either constructs 512 or splits such runs. This is recorded in todo for item 2.7.
- *Item 87*, except finding 3. The supply ledger rebuilds both memories from the base and the held sources, and the five clauses hold:
  - (a) withdrawal and compaction revoke a supply in both copies (`Layers.py` about 10204 and 10306);
  - (b) only the outer root takes the asserted pair;
  - (c) access and surprise roll back only at the boundary;
  - (d) a shared document grants no authority;
  - (e) a raw `('sym', h)` reference keeps the relation at zero evidence.

  `set_origin` has one production caller, which passes its pair, so the required `asserted` is safe. The 49 fixture ports pass the live pair explicitly, which is what those tests asserted before. The given-role case asserts the decided retention. The `SymbolSubSpace` deferral fix is right. One cost note: each given closing copies every relation row (`ClauseRow.py` about 461), and each supply rebuilds both memories (`refresh_supplies`). That is negligible at today's fixtures, which supply at most three truths each. It is to be measured when item 2.7 supplies a corpus's truths.
- *Item 88.* Read in the diff:
  - the observer's identity check restored;
  - the six bank orders;
  - the sentinel backend;
  - the cache account;
  - `compose_choice_phase` declaring the closing;
  - the capacity error naming its limit.

  The other ports are as §47.6 lists them, and their selection passes. On the accumulated limit, see finding 5.
- *Item 89.* `WordIdentity.admit` queues in admission order, and `grammar_S_methods` keeps declaration order (`dict.fromkeys`).
- *Item 90.* Spot checks against §46.1's list match: §41.7's three passes, §45.6's index 258, §45.9's eligible pairs and timing caveat, and §43's tails and keys.
- *Item 91*, on its receipts' neutrality controls. Claude did not rerun them.

**Findings.**

1. *Item 83: Codex took the right branch, with the wrong citations.* The pending-name operand contract is §37.4's `test_reading_curriculum` case, approved in §38 ("the pending-name case, which the STM plan's bind contract supports"). §40 withdrew item 46, which concerned §37.7's jitter wording, not this. Evaluation keeps the last epoch's memory by Alec's confirmation in §41: "The last epoch remains available at the end of training."

   The mechanism, read in the code:
   - The epoch boundary restores MTM to its setup (`PresentationMemory.py` about 513).
   - `MediumTermMemory.named` (about 292) gives a name its token at its first sight in the epoch. The token holds the value it is given then. Afterwards `named` returns the token without changing its value.
   - Each kept close moves the dictionary through `MereologicalCodes.place_rows`.

   Evaluation follows epoch 400 with no boundary, so it composes from epoch 400's first-sight values. By the same mechanism, a name's later mentions inside any epoch compose from its first-sight value.

   The residuals belong to the fixture. Fit 6 also chose disjunction, and its walk has fit 5's four residuals bit for bit. It reads back only because its policy margins are positive: .378 and .592, against fit 5's -.079 and .142. The policy decides at roots it never trained on: the teacher's 3,200 states were all exact (§46). As built, a fresh fit that chooses disjunction passes or fails on that margin.
2. *A headless composition takes identity 0.*
   - `ChunkJournal.apply` builds the parent record as `torch.where(unary or headed, head, 0)` (about 181). A headless binary's record therefore holds identity 0, and about 252 writes the record's identity into the composed chunk.
   - STMChunks reads any identity of at least 0 as an identity: `refresh` (about 185) and `bind_menu` (about 204).
   - Codex's deep trace of fit 5 shows the effect after every root composition, in both trials of the last training epoch and in evaluation ([trace](../benchmarks/2026-11-04-item2-5-review46/fit5-producer/deep/raw/deep-summary.json)). Before the disjunction, the composed chunk held identity 18; after it, it holds identity 0. `refresh` then returns off-stack chunk 4 in its place: identity 0, a RELATION reference, the previous headless root. Between 6 and 14 columns differ, by up to 1.0.
   - In XOR only the STOP choices after the root read the substituted value. In a longer sentence, a later composition or bind can read the previous root instead of its own operand, and `bind_menu` offers that root as an antecedent.

   Codex recorded the observation and, rightly, left it outside item 83. It is item 94.
3. *Fallbacks where a raise belongs.*
   - `PresentationMemory.initial_observation` (about 68) used to raise for a row that lacks its creation observation. It now returns the row's live access and surprise. The new `_initial_relation` (about 119) does the same for a relation. A live count includes the history's reads, which the rule excludes.
   - In almost every case the stamp exists. Creation stamps a new row while the start exists (`Layers.py` about 9769, `LTM1.py` about 273). The capture holds every earlier row. `import_checkpoint_observations` stamps an older envelope's rows at load (`Models.py` about 4480).
   - One path can still lack a stamp: a live row that `withdraw_supplies` drops from the base and that a later supply refers to. Record its start observation where it is dropped, and raise for a missing one (item 95).
   - Codex added the third fallback in `LTM1.retain_given` (about 57). That method has had no caller since item 87 replaced its two ClauseRow calls. Retire it, recorded.
4. *The panel stopped on an observer defect, as §46.1 required.* The carrier hook reads `gradient.dtype` (`attribution.py` about 158). Autograd calls a tensor hook with `None` when the node runs but that output's gradient is absent. Sentence 7's output cost was 0. Codex's proposed repair is approved, with the node and output index recorded (item 93). The partial record shows no spike: the largest pre-step optimizer norm in six sentences is 20.9, at sentence 1.
5. *The full suite has not run since `8c6f4c51da75`* (§41), before item 77's compile guard and every repair since. Codex confirms that one worker can reach `accumulated_recompile_limit` (256 per code object) inside a module, since Dynamo is reset only at module boundaries (`conftest.py` about 74). `make test` therefore runs next, before the panel (item 97), and not only at the freeze.

**Decided (Alec, 2026-11-04, night).** On item 83:

> The MediumTermMemory maintains a set of references to STM and LTM. Identity can be computed from those references (e.g. "Alec went shopping and then he ate dinner" creates one identity in MTM that has two references to STM0 sentences/clauses with subjects "alec" and "he". it probably makes sense for each reference from MTM to have both the STM0/LTM0 .when for addressing as well as the concept that is referenced within that idea (to disambguate subject/object, etc), which in this case are "alec" and "he".

On its scope, Alec chose the first of three options:
- In this round, an identity with no event of its own computes its meaning from its referenced concepts each time it is read, with today's join. This repairs fit 5.
- The full reference form goes to item 2.0's memory spec ([its §9](2026-11-03-item2-0-behavioral-chains.md)).

The options declined were the full form in this round, and leaving everything to item 2.0. Under the rule, a held name reads its concept's current meaning. Fit 5's sentences are then composed from current meanings and decode exactly against the current dictionary. Like §44's evidence rule, it makes what MTM holds depend on its source rather than on a copy taken from it.

**The freeze is not offered.** Items 92 to 97 come first, then item 71's runs on that source.

### 48.1 Text for Codex (2026-11-04, night)

~~~
Read §48. Alec decided item 83: MTM holds references to STM and LTM, and an identity is computed from them; this round applies that to identities without an event of their own (item 96), and the full form is item 2.0's. Your receipts are in doc/benchmarks/2026-11-04-item2-5-review46, called review46 below. Items 92 to 97 come first, in this order: 92 (records), 93 (the observer), 94 (identity 0), 95 (raises), 96 (item 83), 97 (make test); then §44.1's items 71 to 73 on that source. Approved as built: items 82, 84, 85, 86, 87, 88, 89, 90 and 91, except where §48 names a finding and the items below repair it.

92. Records, by exact replacement in §47.8, review46/readings.md, review46/fit5-producer/readings.md and review46/work-state.json:
- The pending-name operand contract is §37.4's test_reading_curriculum case, approved in §38. §40's withdrawal of item 46 concerned §37.7's jitter wording; remove it from the argument.
- Evaluation keeps the last epoch's memory by Alec's confirmation in §41: "The last epoch remains available at the end of training."

93. The attribution observer's absent cotangent (§47.11). Your proposed repair is approved.
- tensor_hook's carrier branch (attribution.py about 158) records an absent cotangent as absent, with its read ID, its objective, the hooked tensor's grad_fn and output index, and that objective's cost at the call. It makes no zero vector and no effective mean for it. The tensor branch and its return are unchanged.
- Certificates:
  - the old callback raises on None;
  - the repaired one records the absence;
  - finite tensor callbacks give the same records and native gradients as before;
  - a native control in which a node runs with one output's gradient absent.
- Before items 94 to 96, on 9a82e07d458e with the repaired instrument, (true, 0)'s archived streams under hash 0 reproduce the stopped run's records for sentences 1 to 6 byte for byte and complete sentence 7. Report each absence's read and node. If the replay differs, stop for review. This replay checks the instrument; it is not a run of item 71.

94. A headless composition takes identity 0 (§48 finding 2).
- ChunkJournal.apply builds a headless binary's parent record as zeros (about 181), and about 252 writes its identity, 0, into the composed chunk. STMChunks.refresh (about 185) and bind_menu (about 204) read any identity of at least 0 as an identity.
- In fit 5's deep trace (review46/fit5-producer/deep/raw/deep-summary.json), refresh returns the previous headless root, chunk 4 with identity 0, in place of the composed chunk. This happens after every root composition, in both training trials and in evaluation, with 6 to 14 columns differing by up to 1.0.
- Give a headless record the journal's own empty values at that producer: -1 for identity, and the journal's empty value for every other field whose empty value is not 0 (name them).
- Certificates, red on 9a82e07d458e and green after:
  - a headless composition's chunk carries no identity;
  - refresh of a newly composed headless chunk returns its own value;
  - in a sentence of at least three words whose headless result is a later operand, the later composition reads that result;
  - bind_menu does not offer a headless root as an antecedent.
- If a specified rule gives a headless composition an identity, cite it, leave the code, and stop for review. Port any test that relied on the alias, with before and after.

95. Raises for a missing creation observation (§48 finding 3).
- PresentationMemory.initial_observation (about 68) and _initial_relation (about 119) raise for a row that lacks its creation observation, instead of using its live access and surprise.
- Where withdraw_supplies drops a still-live row from the base, it first records that row's start observation in initial_observations. import_checkpoint_observations at load stays the one path for older envelopes.
- LTM1.retain_given (about 57) has had no caller since item 87: retire it, recorded.
- Certificates:
  - a row without its creation observation raises, red on 9a82e07d458e and green after;
  - a row withdrawn mid-epoch and referred to by a later supply keeps its start observation;
  - both legacy checkpoint certificates pass.

96. Item 83, Alec's decision (§48, Decided).
- An identity with no event of its own computes its meaning each time it is read, from the concepts it refers to:
  - for each of its names and mentions with a reading (MediumTermMemory.names), that concept's current meaning in the dictionary;
  - for a mention without a reading, the meaning its occurrence row holds.
  They are joined as MediumTermMemory.open and bind join today (elementwise maximum). No copy of an earlier reading stands in for that meaning.
- An identity with an event keeps its timeline's stored state. Item 2.0 replaces stored states with references.
- Name every reader of an identity's value, at least PrependedSymbols.symbolize's meaning block, the deposited chunk, STMChunks.refresh and bind's identity value, and state which of them now read the computed meaning.
- Certificates:
  - red on 9a82e07d458e and green after: a name read in two sentences of one epoch, with a kept close's placement moving its word between them, composes the second sentence from the word's current meaning;
  - red on 9a82e07d458e and green after: a bound identity with two references (for example "alec" and "he") computes the join of both current meanings;
  - a control, green on both: an identity with an event keeps its state.
- On fit 5's and fit 6's archived streams and hashes, all four read-back roots are exact by the walk's rule and all four sentences read back. These are observations of recorded initializations whose trajectories now change, not fits of item 71. If either fit does not read back all four, stop for review.
- Port test_reading_curriculum's pending-name case if it changes, with before and after. MTM's state layout is unchanged.
- Architecture.md, by append: before its first event, an MTM identity's meaning is computed from its references.

97. The default make test on the source with items 93 to 96, before the panel. The suite last ran on 8c6f4c51da75, before item 77's guard and every repair since. Stop for review if a case fails, with its recorded streams, or if the guard raises.

Then come §44.1's item 71, item 72's record and item 73's freeze, on the source of item 97.
- Item 71 is the three lesson arms and the float64 discriminator, each recording its streams, then eight fresh unseeded XOR_grammar fits.
- Item 73 runs make test again if the source changed after item 97, then make test_all and forced_configuration_passed.
- If test_epoch_history.py::test_native_train_keeps_last_grounding_and_recognizes_it_across_ten_epochs fails again, capture it with its streams and stop.
- Stop for review on §44.1 item 71's conditions and on those of items 93, 94, 96 and 97.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review; no commit or publication.
~~~

## 50. Review of the §49 return (Claude, 2026-11-05): items 92 to 96 built; the suite, last run on 3 November, fails 56 cases from four rounds; Alec's decisions on the slow tier, the memory ceiling, guard raises, what an identity refers to and when MTM associates; the freeze not offered

The return was checked on the live tree, which equals candidate `d56a8955def9` in all 987 files of its source snapshot. Against `9a82e07d458e`, 12 files changed: 8 in `bin/` and 4 in `test/`. Two review workflows ran, with every finding checked by independent skeptics. The first reviewed items 93 to 96 and the round's readiness (35 agents). The second triaged every failure of the suite below by cause (44 agents). Each reproduced its cases in scratch copies of the immutable trees `8c6f4c51da75`, `855dc08150d8`, `cf32ffd84391`, `9a82e07d458e` and `d56a8955def9`. Their receipts are in Claude's scratchpad; their findings are below.

**The suite.** Item 97's stop was a documentation fault outside Codex's work. Alec's commit `e0c5795695` had moved the todo items above 2.5 into `doc/DevelopmentHistory.md` with their root-relative `doc/` and `bin/` links. Claude rebased the 146 links (`4502b1c566`); the link test passes. Claude then ran `make test` on `d56a8955def9`:
- ten workers: the 28 GiB aggregate ceiling stopped the run at 5,507 of 7,492 cases (peak 30,284,958,976 bytes), with 5,222 passed, 46 failed and 2 skipped;
- six workers: one worker crossed its 8 GiB limit at 3,446 cases;
- the 1,984 cases not yet run: 1,893 passed, 10 failed, 1 permitted XPASS, 1 skipped.

So 56 cases fail. One certificate crosses the worker limit, and one of its cases has not run to completion. The suite had not run since `8c6f4c51da75` (§41), so the failures come from four rounds:
- 35 from item 96 (§49);
- 6 from item 86 (§47): 2 from its byte axis, still marked dynamic, and 4 from its tables in a test fixture;
- 7 from items 65 and 78 (§45);
- 2 compile-guard exposures that predate this round;
- 6 from a configuration leaking between test modules.

**Approved as built.**
- *Item 92*, the record corrections.
- *Item 93*, the absent-cotangent record. The replay's 48,873 sentence 1–6 records match byte for byte. The stopped run's partial sentence-7 records are an exact prefix of the replay's. The four absences are on reads whose objective had zero cost, as expected.
- *Item 94*. A plain headless binary's chunk carries identity -1, which every reader skips. Exception: its certificates' operands carry no identity, so they would also pass an implementation that inherits the left operand's identity (item 103).
- *Item 95*, except finding 2.
- *Item 96*'s certificates and the fit 5 and fit 6 observations: all eight disjunction roots have residual exactly (0, 0, 0), and both fits read back four of four. The rule itself needed Alec's decisions below, and its implementation has the defects of finding 1.

**Findings.**
1. *Item 96 in the suite: 35 cases.*
   - *1 case.* The meaning band starts at 0 whenever a chart has no meaning pairs (`MediumTermMemory.py` about 315). In `MM_math_chain`'s chart (form 352, location 8, meaning 0) every eventless read therefore replaced form and location with the join of the word rows. The native worked document then charts a parent outside the cube. The fixed rule starts at `percept_event_width` in every mereology chart and keeps the stored value where there is no meaning band.
   - *2 cases.* A reading is read as a raw codebook row. The reading path instead reads the concept the word denotes now (`Interpret.reading`, about 502 to 546), so a defined word read the wrong row. Both modal-timeline cases fail on this.
   - *25 cases* name a word whose legacy codebook row is random and signed, or a key with no row.
   - *1 case*, the pooled description, needed Alec's decision.
   - *1 case.* A kept thought's write to an eventless identity (`ThoughtStream.py` about 505) was silently lost, and `MediumTermMemory.update` accepted it.
   - *5 cases* assert the superseded stored-copy contract: identity rows, the native bind, the first bind, review 10's merge and orders-learned's setup.
   - Further defects found by review:
     - the ingestion pool (`Models.py` about 7181) has no owner, so the rule silently does not apply during `provision_ltm` and `store_truths`;
     - concept reclamation does not protect a held name's reading, which now raises;
     - an alternative timeline's starting state is replaced by the actual's join at the first event;
     - `accessible` recomputes each token's value once per held use.
2. *Item 95: a row held in the start only through a supply's closure.* When the supply is withdrawn or compacted, `refresh_supplies` drops the row without recording its start observation. A later supply that reaches it then raises (`PresentationMemory.py` about 264 to 271, 315 to 337). Also:
   - `import_checkpoint_observations` stamps a missing observation from live counters on every load, not only for older envelopes;
   - an older envelope's `supplied_rows` are stamped from live values rather than their saved state.
3. *Item 86: the constructed byte axis is still marked dynamic.* `Models.py` about 14528 to 14536 marks `_ar_target_word_bytes` and its mask with the part range [3, `outputShape[0]`]. The compiled XOR_grammar gate fails at setup with "64 not in range [3, 6]". Item 86's focused selections ran with `MODEL_COMPILE=none`.
4. *Compile guard exposures, present since `855dc08150d8`, where the guard landed.*
   - `_stm_set_all_slots` updates STM's host high-water mark inside the compiled whole-sentence body (`Spaces.py` about 13421).
   - Even with that repaired, the sparse-active whole-sentence body recompiles 249 times inside admitted phases: its sparse store, inventory and percept parse grow at run time. It recompiled silently 446 times on `8c6f4c51da75` and runs three times faster eager on XOR_exact.
   - The native word bricks `stage_cs_sub` and `stage_cs_sym` recompile on the second word of the default compile mode, because their carried state gains autograd history. `MM_corpus_orders` and `MM_math_chain` reach this under the default backend; the receipted arms ran `--eager`.
5. *Item 78's constructed candidate axis, and memory.* At `MM_math_chain` geometry the axis is 136 rows, where `855dc08150d8` used 42 to 44. `CandidateWholes._weights` (`FormCentroids.py` about 163) reads it as a dense [8, 136, 4096, 256] float32 masked minimum, 4.25 GiB, 93 to 95 times per test. `test_forced_ordinary_pending_premise_in_both_orders` rose from 4.1 to 7.2 GiB and crossed the worker limit. A blocked read is bitwise identical and brings it to 2.8–3.0 GiB. The time stays at about 110–127 seconds, against 55–71, because the eager read processes every constructed row; accepted this round. The `[False]` case also failed 2 of 13 unseeded runs on sources from `cf32ffd84391`, at its line 461, with streams not recorded. The native worked document cannot be replayed at all, because `MathProblemGenerator(seed=None)` draws OS entropy (`math_chain_corpus.py` 61).
6. *Tests the earlier rounds should have ported.*
   - `test_candidate_search_storage` builds SGD over item 65's parameterless chooser.
   - `test_percept_expectation_final`'s compile stub rejects item 78's `initial_phase`. Its faithful port, through the guard, exposes finding 4's word bricks.
   - `test_prepared_answer_boundary`'s explicit bank keeps item 86's 112-row tables beside fewer rows, and they broadcast silently.
7. *Test isolation.* `test/conftest.py` restores the device per module, but not `TheXMLConfig`. Item 89's new certificate and `test_review14_contracts` load the XOR configuration. Bare layers in later modules then take its meaning layout: the heat-reverse cases lose their heat keyword arguments, and the review-20 candidates lose their activations. Production is unaffected.
8. *Harness and records.*
   - Codex's suite wrapper stops on the permitted non-strict XPASS.
   - The item-75 stop reads only the collected objectives.
   - §49.3's "its clause reference is also -1" describes no change: `ChunkJournal.apply` never outputs `parent[:, :2]`.

**Decided (Alec, 2026-11-04 night to 2026-11-05).**
- *Slow tier:* "Drop it from 2.5." The freeze gates on the default `make test`; `make test_all` remains the gate of items 0 and 1. It cannot be green under the default runner: its workers are capped at 8 GiB, two slow tests need 24 and 28 GiB (one runs 48 hours), and 98 slow failures are carried to items 0 and 1.
- *Memory ceiling:* "Allow six workers, recorded." If the ten-worker default still crosses its aggregate ceiling after finding 5's repair, the gate runs at six workers, and both runs are recorded.
- *Guard raises:* "Repair and continue." A further compile-guard raise in `make test` is repaired at its producer under item 78's rule (a width fixed at construction or a declared phase, never admitting a shape, red before and green after), recorded, and the suite runs again.
- *Individuals:* "Yes, join the individual." An identity opened for a known individual also refers to that individual; its current meaning joins the mentions' by elementwise maximum, as one more LTM reference the identity holds. Claude reads a promoted identity's symbol the same way, as the concept it holds in LTM. *Alec corrects this reading if it was meant otherwise.*
- *Thoughts:* "One more reference." A kept thought's written idea joins the references of an identity that has no event of its own, so its meaning joins in.
- *Descriptions and association:* Alec:
  > This suggests that the MTM actually needs to check before associating; the overlap between the red ball and the blue ball is zero. The overlap between "alec" and he is a part/whole.

  Asked where that rule lands, Alec chose:
  > Both now. The object being identified is the noun phrase, so identity may not need to explicitly maintain that representation, since it can be recovered by decoding the references. Although since that decoding is nontrivial for an untrained model, I expect identity (and object permanence) not to function well until the objects themselves have stabilized (which was the problem we ran into before).

  and added: "(So maximal for one of its members)." In this round:
  - a mention whose reading is a derived kind, not a word, reads its own idea, the noun phrase held in its occurrence row;
  - only words read the dictionary, a defined word through what it denotes now;
  - MTM associates a mention with an identity only where their overlap is maximal for one of them, that is, where one meaning contains the other and so one of `IdentityRows.meet_ratios`' two coverages is 1.

  Exact containment gives exactly 1. The witnessed-incompatibility exclusions stay. Weaker binding while objects are unstable is expected, not a fault.

**The freeze is not offered.** The repairs and ports below come first, then a green default `make test`, then item 71's runs, item 72's record and item 73's freeze on one source. Claude's readiness estimate for those gates is 6 to 8 hours of wall time: the suite about 0.3 to 0.5 hours, the four lesson runs 5 to 6.5 hours, and the eight fits up to 0.75 hours.

### 50.1 Text for Codex (2026-11-05)

~~~
Read §50. This round closes item 2.5. Alec's decisions are in §50: make test_all is not part of 2.5; the default make test may run at six workers, recorded, if ten cross the aggregate ceiling after item 105; compile-guard raises are repaired at their producer under item 78's rule and the round continues; and what an identity refers to and when MTM associates are as items 100 and 101 state. Order: 98 (records), 99 (instruments), 100 to 107 (source and tests), 108 (make test), then §44.1's items 71 to 73 on that one source. Approved as built: items 92 to 96, except where §50 names a finding and the items below repair it. Claude's suite results are output/tests/20261104-172809-b9e8bb, 20261104-173643-b187a2 and 20261104-174206-9df447.

98. Records, by exact replacement.
- §49.3: "its clause reference is also -1" describes no change (ChunkJournal.apply never outputs parent[:, :2]).
- §49.6: the stop's 146 links came from Alec's commit e0c5795695, and Claude rebased them in 4502b1c566.
- Item 71's report and item 72's record go in your return section.

99. Receipt instruments, outside bin/, test/ and data/, each with model-free controls.
- run_suite.py stops only on 'failed' and records 'xpassed'. A strict XPASS is already 'failed'.
- lesson.py's item-75 stop also reads the all-parameter pre-step optimizer norm. It stops when that norm, or any collected objective's actual gradient norm, reaches 1e8 under separateChooser true.

100. What an identity refers to (item 96, with Alec's decisions in §50).
a. A word reading reads the concept the word denotes now.
   - Add InterpretLayer.referent(word), the frozen resolution of reading() with no write: the first live denotation; else the single remembered object that no recorded raise supersedes (ambiguity raises as reading() does); else the word itself.
   - It reads no MTM token and no learning mode. reading() resolves through it, so the rule exists once.
   - In value_of, row = owner._csw_row_of(owner.interpret.referent(reading)), raising with both the reading and the concept.
b. A reading that is not a word reads its mention's own occurrence row, as a mention without a reading does: its own idea, the noun phrase. This covers a derived kind or a concept that ReferenceClose recorded for a wordless mention (about 1251 and 1260). A word is a symbol the allocator types 'word'.
c. An identity opened for a known individual (Token.native), or promoted to a symbol (Token.symbol), also refers to that concept. Its current row joins the references.
d. The meaning band starts at percept_event_width in every mereology chart. Where there is no meaning band (meaning_pairs 0, as in MM_math_chain), value_of returns the stored value.
e. An alternative timeline keeps the state its producer gave it: value_of returns its stored value, and event() computes only the actual timeline's starting state.
f. A kept thought's write to an identity with no event (ThoughtStream.py about 505) adds its written row to the identity's references, as a mention without a reading. MediumTermMemory.update raises for an identity with no event.
g. The ingestion pool (Models.py about 7181) is constructed with its owner reference and the owner's n_vectors and field_width. value_of raises for a pool that holds names or mentions and has no owner; standalone numerical pools keep their explicit value API.
h. ConceptForgetting.protected_concepts protects every held name's reading.
i. ReferenceClose.accessible reads each token's value once per call; index names by token.
Certificates, red on d56a8955def9 and green after:
- a defined word reads its denotation's current row, including after that row moves, and after an epoch boundary with one remembered object;
- a description keeps its distinguishing coordinate;
- a known individual named by a word that does not denote it joins the individual's meaning;
- a promoted identity joins its symbol's meaning;
- a chart without a meaning band reads its stored value;
- an alternative's starting state survives the first event;
- a kept thought's write joins the identity's meaning;
- update on an identity with no event raises;
- the ingestion pool reads current meanings;
- reclamation keeps a held reading.
Item 96's ten certificates, test_identity_threading and test_reading_curriculum pass. Fits 5 and 6 on their archived streams again read back four of four with exact roots; these are observations, not item-71 fits.
Port, with exact before and after text citing §50, the fixtures and assertions these rules supersede:
- test_reference_arrivals' world() gives its bare word 'x' the native chart's zero meaning;
- the pending-alias test's dictionary holds the content its named sites carry;
- test_math_chain_repair's forward name is an admitted concept, not the key 17;
- orders_learned records an event before its explicit state;
- every assertion whose expected value changes under a to f (identity_rows, review10's merge, layout_removal's kept thought and others): its new value is derived from the rule, and every other assertion is kept.

101. MTM associates only on maximal overlap (Alec, §50).
- Wherever a mention is joined to an identity by content, a candidate is offered only when one of the two coverages that IdentityRows.meet_ratios gives for it and the mention, on meaning coordinates, is exactly 1. That is, one meaning contains the other.
- This covers at least ReferenceClose.score's candidates and STMChunks.bind_menu's identity chunks. Name every other path. Joins by identity (a name's own token, a merge after re-identification, re-entry by stored identity) are not association by content; say so for each.
- The witnessed-incompatibility exclusions stay. An empty meaning still contains nothing.
Certificates:
- 'the red ball' and 'the blue ball', the same kind with exclusive colours and no witnessed exclusion, never associate; red on d56a8955def9 if they do today;
- 'he', whose meaning is a part of Alec's, associates with Alec;
- 'the ball' associates with 'the red ball'.
Alec expects identity and object permanence to work poorly until objects stabilize. A lesson's falling binding counts are an observation, not a stop. Port a maintained test whose association relied on partial overlap, citing §50, with its fixture made part/whole where the test's subject is a correct binding.

102. Item 95's dropped rows (§50 finding 2).
- refresh_supplies records the start observation of every still-live row and relation it drops from both memories, before replacing either; withdraw_supplies' two special branches then go.
- import_checkpoint_observations applies only to envelopes that predate the creation-observation map, identified by an explicit marker. On a current envelope, a live row with neither a snapshot entry nor a stamp raises.
- An older envelope's supplied_rows are stamped from their saved state.
Certificates, red on d56a8955def9 and green after, in both memories: a dependency held across one boundary only by a supply's closure, the supply withdrawn mid-epoch, then a later supply reaching it, keeps its epoch-start observation across two boundaries; the same for compaction; and the envelope rules.

103. Item 94's certificate: add a case whose left operand carries an identity with an off-stack field entry. Assert identity -1, a refresh to the chunk's own value, and exclusion from bind_menu. It is red on 9a82e07d458e and on an inherit-left variant.

104. Compile (§50 finding 4).
a. _stm_set_all_slots neither reads nor writes STM's host high-water mark while compiling. _forward_per_stage syncs it from each STM's live depth at the eager boundary, on the whole-sentence path (not word_brackets) only.
b. Remove _ar_target_word_bytes and _ar_target_word_mask from the part-axis mark_dynamic loop (Models.py about 14528 to 14536). Their axis is the constructed wordSurfaceCapacity. Correct the [B, W, P] docstrings.
c. When concept_order_limit > 0 and not word_brackets, enable_compiled_step leaves _forward_body and _forward_head eager and says why: the sparse store, inventory and percept parse grow at run time. This is declared host work, not an admitted shape.
d. CompileGuard.carried_state_phase(args, kwargs) returns any(value.requires_grad for value in args[4]). stage_cs_sub and stage_cs_sym compile with it (Models.py about 22816), and Architecture states the phase.
   - Certificate: an undeclared brick raises on carried state; the declared brick admits exactly the fresh and the carried graph; a third pattern raises.
e. Delete test_explicit_dimensions.py:202's MODEL_COMPILE line. It is dead after util fixes the backend at import, and it leaks into later subprocess tests; the compiled gate stays.
Certificates: XOR_exact's two CLI cases and the two TestXorGrammar cases pass under the default compile mode, red on d56a8955def9.

105. CandidateWholes._weights (FormCentroids.py about 163) reads where(support, properties, 1).amin(-1) in blocks of 256 properties, concatenated. Leave SpanWholes alone.
- Certificate: bitwise equality with the dense read at P in {0, 9, 256, 4096}, both dtypes, eager and inductor, including inside torch.while_loop, and for every call of a real run.
- Record the pending-premise case's peak and time before and after. Its remaining time against 855dc08150d8 is accepted this round; record eager reading of available rows only as a proposal.

106. Test harness and ports, each with exact before and after text.
a. test/conftest.py: a module-scoped _restore_xml_config restores TheXMLConfig's _data, _sources and _requirements, as _restore_process_device restores the device.
b. test_candidate_search_storage: assert that the chooser has no parameters and that operators have parameters exactly for Product and Adverb. Step only operator parameters, when there are any. Assert that chooser.weight stays [1, 0, 0, 0, 0]. Cite item 65.
c. test_percept_expectation_final's stub, and its sibling in test_sentence_compose.py, compile through guarded_compile with initial_phase. Every assertion is kept.
d. test_prepared_answer_boundary's _with_identity_bank carries surface_present and primitive_support for its own rows, sets forms from its codes and form_wholes to None. PrimedSymbols.__post_init__ raises when a candidate table's leading shape differs from rows. form_wholes is exempt: walk_bank rebuilds it after the join.

107. Replayability.
- MathProblemGenerator(seed=None) (math_chain_corpus.py 61) draws OS entropy, so no recorded stream replays the native worked document. Draw its seed from the process's recorded Python stream instead.
- Then run test_forced_ordinary_pending_premise_in_both_orders[False] under recorded streams until a failure at its line 461 is captured, or 20 runs pass. Repair a captured failure at its producer. Stop for review if it needs a decision.

108. The default make test on the source with items 98 to 107.
- If ten workers cross the 28 GiB aggregate ceiling, run six and record both (Alec).
- For a failed case, repair the producer, or port a test whose contract a recorded decision superseded (cite it; before and after). Record it, then run the suite again.
- For a compile-guard raise, repair it at its producer under item 78's rule (Alec), certify it, and run the suite again.
- Wait for the suite lock; never run a suite beside the panel.

Then come §44.1's items 71 to 73, on the source of item 108.
- Item 71: the three lesson-13 arms and the float64 discriminator, each recording its streams; then eight fresh unseeded XOR_grammar fits.
- Item 72's record is in your return.
- Item 73: the freeze, one hash.
  - ports.json and ports.patch against d56a8955def9 (/private/tmp/item25-review48/source-item96-d56a8955def9), and the cumulative patch against git HEAD.
  - Every document that differs from HEAD.
  - make test green on the freeze hash, rerun if the source changed after item 108.
  - Item 71's runs on the freeze hash, repeated if the source changed after them.
  - forced_configuration_passed.
  - test_doc_links after the last document edit.
  - make test_all is not part of 2.5 (Alec).
Pre-authorized, not stops:
- documentation-only fixes;
- receipt-instrument fixes with model-free controls;
- keeping the machine awake for the long runs (caffeinate) and freeing disk space first.
Stops for review:
- §44.1 item 71's conditions: a run fails, an XOR fit fails either case, or an event at or above 1e8 under separateChooser true (item 99);
- a failure that the decisions do not settle;
- item 107's captured failure if it needs a decision;
- test_epoch_history's ten-epoch case if it fails again, captured with its streams.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review of the freeze; no commit or publication.
~~~

## 52. Review of the §51 return (Claude, 2026-11-05): items 98 to 106 built; make test completes with 24 failures; item 107 settled by the recorded single-writer contract; trust becomes item 2.3; no freeze offered

The return was checked on the live tree, which equals Codex's `e0fc4a92719e` in all 990 files of its source snapshot. Against `d56a8955def9`, 46 files changed: 15 in `bin/` and 31 in `test/`. Claude read the source diff in full. A review workflow of 20 agents, every finding checked by skeptics, covered:
- items 100 and 101;
- item 107;
- the other repairs and ports;
- what remains.

Claude then ran `make test` on `e0fc4a92719e` (output/tests/20261105-022039-ff4dd3):
- all 7,570 cases ran, in 911 seconds at ten workers;
- the aggregate peak was 19.0 GiB, against the 28 GiB ceiling;
- 7,222 passed, 323 skipped, 24 failed, and one case gave the permitted XPASS;
- all 56 failures of `d56a8955def9` now pass;
- the forced pending-premise case (`test_forced_ordinary_pending_premise_in_both_orders`) passed in both orders, at 129 and 99 seconds.

**Approved as built.**
- *Items 98 and 99.*
- *Item 100*, a to i. The referent is the frozen resolution, and `reading()` resolves through it. Non-word readings read their occurrence rows, and a held individual or symbol joins as a reference. Alternatives keep their states, and only the actual's start is frozen at the first event. A kept thought binds its written row, and `update` without an event raises. The ingestion pool has its owner, an ownerless pool with references raises, held readings are protected, and `accessible` reads each token once.
- *Item 101* in `ReferenceClose.score` (closing, thought and arrival), `STMChunks.bind_menu` and the read-time merge. The other content matches (`match_candidate`, `witnessed_match`, field activation) cluster candidates or refresh access, and associate no mention.
- *Items 102 to 106*, the replayable problem generator, and item 72's record (§51.6).
- *Item 107's capture.* Codex stopped exactly where §50.1 required.

**Findings.**
1. *Item 107: definitions have one recorded writer.*
   - `Lexicon.md` (lines 64 and 65) and `Interpret.py` (about line 370) record that `InterpretLayer.define` is the only writer of definitions. `DefinitionIndex.add` (`Definitions.py` about 46 to 76) tests only a row's type: it indexes any object→word Part row once the word has a description, whatever the row's evidence or writer. `PartClosing` writes such rows as generic chains, so the close's chain over `' '` became a second denotation.
   - Separately, `add` marks the index current (line 70), although its rule reads allocator state that the LTM1 revision does not track. The incremental index and its rebuild therefore disagree on the same rows: [75] after `define`, [14, 75] after a rebuild.
   - The repair restores the recorded contract. A row defines its word only when `define` named its pair (`ConceptAllocator.named_identities`, already saved in checkpoints), and only a rebuild marks the index current.
   - This is not a new semantic choice. It is also the narrowest case of Alec's trust proposal below: structure over a word not yet learned does not become what the word means.
2. *A description with a lexical head reads its head word.* Every close member that is not determined records its head leaf's word as its reading (`ClauseJournal.py` line 617; `ReferenceClose.py` lines 1228 and 1229). So the identity of "a red ball" reads "ball" (or "red", under a headless parse), and item 101 cannot tell the red ball from the blue one. Completing Alec's decision ("the object being identified is the noun phrase") means only a mention of one leaf records its word; a longer description reads its own occurrence. §50.1's item 100b covered only descriptions with no word.
3. *24 maintained cases fail on the return.* They are in 8 modules outside Codex's 20-module selection, and each passed on `d56a8955def9`:
   - 20 follow item 101's containment;
   - 3 follow item 100g's owner checks (a standalone test pool, and an owner stub in both parametrizations);
   - 1, `test_reverse_traversal`, has no producer found yet.

   Also not ported:
   - the arrival frame-scope test now passes through the containment gate rather than through its frame filter;
   - a weekly certificate cannot pass under item 101, because distinct unit vectors never contain each other.
4. *A chart without a meaning band associates by identity only.* In `MM_math_chain` (form 352, location 8, meaning 0) every coverage is 0, so item 101 offers no candidate by content, and permanently. Names still join through their own tokens, and the default suite passes. The phase certificate `test_live_pending_premise_closes_in_both_orders` (skipped: no lesson produces its `pending_premise_learning` artifact) cannot pass on that chart. Recorded, and carried to item 0 with the choice of giving that lesson a meaning band.
5. *The aligned serial reader admits whitespace and punctuation as lexical words* (`Models.py` about 11587 to 11615). This contradicts `Architecture.md` lines 921 to 923 and `Params.md` line 636 (`whitespaceUnits`, false by default). In attempt 11, `' '` became identity 75 and `'.'` identity 77. It predates this round and does not cause item 107. Recorded for item 2.3, which decides what a learned word is.
6. *Smaller.*
   - The witnessed exclusions are subsumed once one meaning contains the other; record that, and certify their remaining uses.
   - Codex's suite wrapper stops at the first failure and misreads the runner's compile-cache retry.
   - The suite and the panel can race for the lock.
   - Both baselines live under `/private/tmp`, which a restart clears.
   - `_stm_shift_and_push` still reads the host depth mirror while compiling.

**Decided (Alec, 2026-11-05).** On item 107, Alec:
> It seems like we are learning relations over words, the words are not fixed, and the relations come to encode false statements.
> My guess is that we need a trust value in words, and a trust value in the relations over them ( using trust here is a misnomer, but we already have that value for relations learned from external sources). So first of all, our words will have a low trust until they are learned, which prevents a relation over them from having a high composite degree of trust (which requires that both operands and operator have a high degree of trust, so maybe multiply).
> That would prevent this situation, and perhaps also ensure the learning order that we proposed making a part of the data, so that we could lean words, then operations one by one (we just need to specify the order of the gates and the criteria for trusting in an operation).
> This will make testing difficult, since it requires a trained model, so probably we need a run_trained flag in edition to the run-slow flag

On its scope: "Trust in 2.3, but it will move many of the grammar tests to the gate I mentioned." Then, refining it:
> Actually, wait. Conceptual trust is already measure by their magnitude, so only operators need a certainty or trust value.
>
> Unless those operators exist somewhere as vectors also, in which case we can use their magnitude.

What the code has ([item 2.3 notes](2026-11-05-item2-3-trust.md) §3):
- a word used as an operator is a concept, so its magnitude serves;
- the fixed relations `part` and `implies` are constant hashed codes, whose length records nothing;
- the grammar operations are `GrammarLayer` weights, with no concept vector.

So the operations need their own trust value. The trained-model tier exists in part as the phase certificates: all 16 skip, because no lesson produces their artifacts.

Trust is item 2.3, after 2.5 in the countdown and before 2.0. Claude reads the answers as closing 2.5 with finding 1's repair, the recorded writer contract, which is trust's narrowest case for definitions. Finding 4 stays as recorded, and finding 5 goes to 2.3. *Alec corrects this reading if it was meant otherwise.*

**No freeze was offered.** Codex stopped at item 107's capture, as §50.1 required. The text below closes 2.5. It runs the eight fits before the lesson arms, so that a failing fit stops the round before the long runs.

### 52.1 Text for Codex (2026-11-05)

~~~
Read §52. This round closes item 2.5. Alec placed trust in a new item 2.3, after 2.5 (§52: a concept's trust is its magnitude, and operators need their own); this round adds no trust and no new test gate. Order: 109 (definitions), 110 (descriptions), 111 (the 24 failures and other ports), 112 (records), 113 (instruments and baselines), 114 (make test until green), 115 (item 107's twenty runs), then §44.1's items 71 and 73 as amended below. Approved as built: items 98 to 106 and item 72's record, except where §52 names a finding and the items below repair it. Claude's suite on e0fc4a92719e is output/tests/20261105-022039-ff4dd3: ten workers, 911 s, peak 19.0 GiB, 24 failures, one permitted XPASS.

109. Definitions have one writer (item 107; §52 finding 1). Lexicon.md lines 64 and 65 and Interpret.py about line 370 record that InterpretLayer.define is the only writer of definitions.
- DefinitionIndex.add (Definitions.py about 46 to 76) indexes a row only when define named its pair: obj in allocator.named_identities.get(word, ()), beside the existing type test.
- add no longer marks the index current (delete self._revision = store.revision, line 70). Only rebuild does, so the index always equals its rebuild.
- Interpret.define records named_identities before it writes and adds its row.
- A checkpoint whose word descriptions predate named_identities raises in _restore_allocator_extras, before anything changes.
- The load's backfill of named_identities from the rebuilt index (Models.py about 4487) is retired, recorded.
Certificates. The first three are red on e0fc4a92719e and green after:
- for every write order, the incremental index equals a fresh DefinitionIndex over the same rows;
- the three late-descriptor lanes, (0,0), (0,1) and (1,0), give only the genuine object, before and after a rebuild;
- attempt 11's archived stream (pending107-11, PYTHONHASHSEED 277256384) gets past the ambiguity raise;
- test_part_closing's nested-unknown case stays green.

110. A description reads its noun phrase (Alec, §50; §52 finding 2).
- ReferenceClose.py line 1228: a mention records its head word as its reading only when it has one leaf, `and len(item['leaves']) == 1`: a name, a pronoun or a bare noun. A description of more than one leaf records no reading and reads its own occurrence row.
- FamiliarTier.promote_identity (line 496) names a token by its reading only when that reading is a word.
- Certificate, red on e0fc4a92719e, through the ordinary closing: 'a red ball' opens an identity whose value keeps its red coordinate; 'the blue ball' is not offered against it at ReferenceClose.score; a mention contained in it is.

111. Claude's 24 failures, each green on d56a8955def9. Port each with exact before and after text, citing §50; keep every other assertion; where the test's subject is a correct binding, make its fixture part/whole.
- Item 101 (20 cases): test_bind_thought's G2 (12 cases) and G3[k-True] (4 cases); test_item25_completion::test_sign_question_reads_the_latest_named_token_and_no_concept_for_an_unnamed_token; test_live_references::test_public_reading_fuses_the_reference_before_capture_and_write; test_reference_mentions::test_later_fill_releases_the_menu_without_an_expectation_loss; test_reference_reconstruction::test_unpriced_endings_and_covering_candidates_have_zero_reference_cost[empty].
- Item 100g (3 cases): test_reading_round::test_named_alias_follows_merge_restore_and_horizon_expiry (its standalone pool), and test_step2b_storage::test_truth_ingestion_restores_caller_recall_and_isolates_each_source[False] and [True] (its owner stub).
- test_reverse_traversal::test_free_walk_reports_missing_leaves_without_underflow: find its producer. Repair the producer, or port the test under a recorded decision, with the evidence.
- test_reference_arrival_review::test_new_document_cannot_refine_a_same_stream_site_outside_its_frames now passes through the containment gate. Port it so that only its frame filter prevents the refinement: world(pending_atom=0), described(atoms[0], x), equality(..., subject=atoms[0]).
- The weekly test_math_chain_section14::test_ordinary_initial_binding_distribution_includes_every_retained_candidate cannot pass under item 101, because distinct unit vectors never contain each other. Port its fixture so that each retained identity is a part of the mention: parts = eye(width)[:count]; the mention is their sum.
Where your reading of a case differs, give the evidence. Another recorded rule or a certified producer repair settles it without a stop.

112. Records, by append to Architecture.
- In a chart without a meaning band (meaning_pairs 0, as MM_math_chain) no coverage can be 1. MTM there associates by identity only: a name's own token, reading_identity, kept definitions and promotion. The phase certificate test_live_pending_premise_closes_in_both_orders cannot pass on that chart and is carried to item 0 with this record.
- The witnessed-incompatibility exclusions are subsumed for a mention's own meaning once one meaning contains the other. Certify their remaining uses: an asserted constraint and the grouping pass.
- The aligned serial reader admits whitespace and punctuation as lexical words (Models.py about 11587 to 11615), contrary to Architecture lines 921 to 923 and Params.md line 636. It is a known defect for item 2.3, not repaired here.

113. Receipt instruments and baselines, outside bin/, test/ and data/, each with model-free controls.
- run_suite.py runs the suite to completion and classifies outcomes from the native result.json after the runner's compile-cache retry; a compile_cache_retry is not a failure. It records every failure with its streams.
- run_panel.py refuses to start without a green run_suite result for its own source_sha256. Suites and the panel launch strictly in sequence.
- Pass the identify_source record explicitly to both. Keep durable copies of the d56a8955def9 and e0fc4a92719e baselines outside /private/tmp. Run the gates with ~/.cache writable and at least 10 GiB free.

114. make test through run_suite.py at ten workers, repeated until green. Claude's e0fc4a92719e run peaked at 19.0 GiB. If a ten-worker run crosses the memory ceiling, record the crossing and use six workers (Alec, §50).
- Repair or port every failure under a recorded decision, then run again. Nothing is retried.
- A compile-guard raise is repaired at its producer under item 78's rule (Alec). _stm_shift_and_push's host depth read under compile is a known candidate.
- Stop, with the case captured with its streams, if:
  - a failure has no recorded decision that settles it;
  - an unseeded learning certificate fails (TestXorGrammar, MM_xor, the constituent lesson, exploration coverage, surface grammar);
  - the forced pending-premise case (test_forced_ordinary_pending_premise_in_both_orders) fails;
  - test_epoch_history's ten-epoch case fails.

115. Item 107's twenty runs, on the source of item 109, counted from zero (the earlier ten do not count): run until a failure is captured or twenty pass. Repair a captured failure at its producer under a recorded decision, or stop. Repeat the series only if bin/ or data/ change afterwards.

Then, on the green source of item 114, identified:
- Item 71:
  - first, the eight fresh unseeded XOR_grammar fits: four workers, MODEL_COMPILE=none, as in §45.9;
  - then the three lesson-13 arms and the float64 discriminator, in sequence, each recording its streams;
  - report, beyond §44.1: the projection tails and the velocity proposal, and every event at or above 1e8 under separateChooser false, named by read, formula and objective.
- Item 72's record stands as returned in §51.6.
- Item 73: the freeze, one hash.
  - ports.json and ports.patch against e0fc4a92719e and against d56a8955def9, and the cumulative patch against git HEAD;
  - every document that differs from HEAD;
  - make test green on the freeze hash, and item 71's runs on that hash; repeat both if the source changes after them;
  - forced_configuration_passed, from report_runs.py with --require-panel, the green suite's result and its certificate directory;
  - test_doc_links after the last document edit.
Pre-authorized, not stops:
- documentation-only fixes;
- receipt-instrument fixes with model-free controls;
- a run that fails only in a receipt instrument is repaired and replayed from its archived streams and hash seed, and the failed attempt is kept. Replaying a recorded stream is not a retry. A failure in the model stops.
- caffeinate, and disk cleanup.
Stops for review:
- §44.1 item 71's conditions: a run fails in the model; an XOR fit fails either case (recorded with pair diagnostics); an event at or above 1e8 under separateChooser true;
- item 114's stops;
- item 115's captured failure, if it needs a decision.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review of the freeze; no commit or publication.
~~~

## 54. Review of the §53 return (Claude, 2026-11-05, night): items 109 to 115 built; Claude's suite on the freeze catches an older defect in the forced pending-premise case, a forward reference read as its own mention; the freeze not accepted

The return was checked on the live tree, which equals the offered freeze `8e9fe263470939dea6a30c66800b1d142ce8c1825c9628ad174178c5f13c1237` in all 994 source files and in the 8 documents of its manifest. Against `e0fc4a92719e`, 8 files changed in `bin/` (about 90 lines) and 22 in `test/` (4 of them new). Claude read all of it, and the receipts behind each claim of §53.

Claude then ran `make test` on the freeze (output/tests/20261105-135927-c0b82a):
- all 7,626 cases ran, in 907 seconds at ten workers, with a peak of 19.0 GiB; the 16 cases beyond Codex's 7,610 are document links that §53 added;
- 7,304 passed, 323 skipped, and one case gave the permitted XPASS;
- one failed: `test_forced_ordinary_pending_premise_in_both_orders[True]`, at line 461 ("x must bind to three or retain its named open referent"), one of §52.1 item 114's stops.

The source was unchanged before and after. The capture and Claude's replays are in `doc/benchmarks/2026-11-05-item2-5-review54/claude-capture/` (local).

**Finding 1: a forward reference read as its own mention.** This stops the freeze.
- *The capture.* The failing case's streams have SHA-256 `ff2a73b6a91d`, with `PYTHONHASHSEED` 460798160. The worker's initial stream has the same hash, so collection consumes no randomness, and a replay from process start (`replay_check.py`) is exact.
- *It reproduces on both sources.* It fails at line 461 on `8e9fe2634709`, and on `e0fc4a92719e` from Codex's durable baseline tree, on the same row with identical dumps. So the defect is older than this round. Item 115's twenty runs covered only the `[False]` order (as item 107 named it), and the earlier suites drew passing initializations.
- *What the read does.* Row 32 (`('ltm', 'address-v1', 3339473413110316094)`, kind `question`) holds x's forward reference `(0, 13)` open.
  - `ClauseRow.read_reference_bindings` (about lines 717 to 755) reads that reference through the MTM token at the row's own occurrence, which is the row's own mention of x. The token has kind `None`, symbol 14 (x's named object) and is not pending.
  - The guard there, "A forward word names the missing referent. Its lexical form cannot fill that role merely because it now has a DEF row" (about line 737), tests only the token's kind.
  - So the row's referent becomes its own mention, and the forward reference is dropped.
  - The test's strict-xfail classifier does not fire, because the token's symbol is one of x's names. This is not the recorded deviation, which `UnsupportedReversePremiseBinding` restricts to a retained kind that is not x's (§34).
- *Also:* `read_reference_bindings` accepts `exclude` and never reads it, at HEAD, at round A's landing and at item 2.7's close. `ThoughtClosing` (about line 28) passes `exclude=(index,)` for "The row now closing cannot use its own complement".
- *Recorded rules settle the repair; no new decision is needed.*
  - A state update "does not alone discharge a named open referent; selected binding and kept thought remain the owners of that transition". The pending-premise assertions keep their "bind-correctly-or-stay-open condition" ([layout and removal](2026-10-23-layout-and-removal.md) §7, item 2).
  - The rule the guard states, which `ReferenceClose.finish_thought_bind` also records (about lines 575 and 576): a name "does not supply the missing referent merely by being read".

**Approved as built.**
- *Item 109.* `DefinitionIndex.add` indexes a row only when `define` named its pair, and only `rebuild` marks the index current. `define` records the pair before it writes. A checkpoint with word descriptions and no `named_identities` raises before anything changes, and the load's backfill is retired. A named pair stays indexed whatever its evidence: the index gains no evidence filter, which would be trust (item 2.3).
- *Item 110.* A head word is a reading only for a one-leaf mention, and promotion names a token by its reading only when that reading is a word. The red-ball certificate runs the native journal and the ordinary closing under both head parses.
- *Item 114's closing repair.* After item 110, a multi-leaf sentence root that closed as an open relation no longer borrowed its first word's occurrence, so its relation handle became its MTM mention address. The closing now gives such a root's mention an occurrence of its own:
  - untimed, with zero evidence, and counted in the capacity preflight;
  - written by `PartClosing._write_occurrence`, extracted unchanged from `PartClosing._write`.

  A root closes as a relation only when it has two or more children, so only multi-leaf descriptions get the extra row. One-leaf roots and embedded mentions keep the occurrences they had, and Architecture records the rule. This is item 110's "a longer description reads its own occurrence row", and needs no new decision.
- *Item 114's compile repair.* `_stm_shift_and_push` no longer advances the host depth mirror while compiling. This matches the per-word body's own pushes (`Models.py` about 19577 and 19694), which already skip it. The eager whole-sentence boundary re-pins the mirror from tensor depth (`Models.py` about 24741).
- *Item 111.* The 24 ports, each with its evidence. The reverse-traversal failure's producer was the test's own assumption, that a bias towards STOP makes STOP legal; semantic eligibility (§11) can forbid it. Claude's archived stream shows this, and the existing walk-control oracle isolates the test. Approved with finding 2 below.
- *Items 112, 113 and 115*, the last with twenty fresh passes counted from zero.
- *Item 71.*
  - The eight fresh XOR_grammar fits pass both cases.
  - The three lesson-13 arms and the float64 discriminator complete, with no event at or above 1e8 in any audited class, and `forced_configuration_passed` is true.
  - One sentence of the jitter arm was not admitted ("Who is here?", with no derivation within its budget). The Reader's repair for it is a receipt fix: certificate (vi) covers kept derivations (§32.4).
- *Item 73's manifest and ports*, against `e0fc4a92719e`, `d56a8955def9` and HEAD, and the document links (928 passed).

**Recorded and carried.** None of these stops the freeze.

2. *An elided role binds nothing under containment.* A missing role carries no content, so item 101 offers it no candidate. G3's port supplies the intended individual's description at the site (`test_bind_thought.py` about 232 to 238), and the native closing produces no such content. So G3 now certifies binding when the content is given. Carried to 2.0 round 5 with the full reference form: an elided role needs content, for example from the expected role its witnesses established.
3. *A substitution that no closing reaches.* `ReferenceClose.commit` substitutes a one-leaf mention's word occurrence when the mention's address is a symbol (about line 1216), and `Models` passes `word_occurrences` for it (about line 21002).
   - Its one production caller passes the rows of `write_clause`, where every mention now has an occurrence of its own.
   - Before item 110, only the defective multi-leaf root reached the substitution.
   - The substitution and the argument are retired in 2.0 round 1, with a record. `commit_read_words` stays.
4. *The output gradient does not reach understanding under the shared backward.* In both `separateChooser` false arms and the discriminator, in every epoch, the output gradient reaches the percept-to-word readout (`conceptualSpaces.0.layers.4.coefficients`, up to 6.8 an epoch) and the answer readers. It reaches none of the codes, grammar operators and definitions, as §34 recorded. Round 3 (one gradient) starts from this.
5. *The projection's tail.* Primitive memberships are re-clamped on 318 of 320 updates in the separate arm, and on 159, 157 and 156 of 160 in the others. The last updates move by momentum alone (gradient norm 0 in three arms). Codex proposes projecting outward momentum at clipped coordinates. Round 3 decides on it with the momentum-carried projection, which the 2.0 text builds "only if the receipts call for them".
6. *The lesson's answers.* In any arm, at most 1 of 80 is bound or exact. Item 71's gate is numerical; learning the lesson is not 2.5's criterion.
7. The runner warns that the weekly slow record is 34.9 days old. `make test_all` is the gate of items 0 and 1 (Alec, §50).

**The freeze is not accepted.** The text below repairs finding 1 and offers the freeze again. The repair changes `bin/`, so make test and item 71's runs are repeated on the new hash, as §52.1 requires.

### 54.1 Text for Codex (2026-11-05)

~~~
Read §54. The freeze 8e9fe2634709 is not accepted. Claude's make test on it (output/tests/20261105-135927-c0b82a: ten workers, 907 s, peak 19.0 GiB) fails test_forced_ordinary_pending_premise_in_both_orders[True] at line 461, one of §52.1 item 114's stops. Everything else in §53 is approved as built (§54). Order: 116, 117, then make test, item 71's runs and the freeze. Write receipts to doc/benchmarks/2026-11-05-item2-5-review54/, where Claude's capture is in claude-capture/.

116. A forward reference read as its own mention (§54 finding 1).
- The capture: claude-capture/worker-061-c96357c9fd39f1f8-streams.pt (SHA-256 ff2a73b6a91db658a554cde1bae002e302dad5cb14db2a48d761a0b115ad1b82), PYTHONHASHSEED 460798160, the runner's one-thread environment. Collection consumes no randomness, so replay_check.py from process start reproduces it exactly, on 8e9fe2634709 and on e0fc4a92719e; the command is in claude-capture/README.md.
- What it shows. Row 32 (kind question) holds x's forward reference (0, 13) open.
  - ClauseRow.read_reference_bindings (about 717 to 755) reads it through the MTM token at the row's own occurrence, the row's own mention of x: kind None, symbol 14 (x's named object), not pending.
  - The guard "A forward word names the missing referent" (about 737) tests only the token's kind.
  - So the referent becomes the row's own mention, and the forward reference is dropped.
- Also: read_reference_bindings accepts exclude and never reads it. ThoughtClosing (about 28) passes exclude=(index,) for "The row now closing cannot use its own complement".
- Repair at the producer under the recorded rules:
  - a read does not alone discharge a named open referent; selected binding and kept thought own that transition, and the pending-premise assertions keep bind-correctly-or-stay-open (doc/plans/2026-10-23-layout-and-removal.md §7, item 2);
  - a forward word's name does not supply its referent merely by being read (the guard; ReferenceClose.finish_thought_bind about 575).
- Make exclude do what its caller records, or retire it, recorded. Certify that finish_thought_bind cannot select the same token, or repair it under the same rule. Record the repair in Architecture.
- Certificates:
  - the captured stream, red on 8e9fe2634709 and green after;
  - a direct case: a forward reference whose only candidate is its own mention's token, named by the word's object, keeps its referent open.
- Where your reading of the producer differs, give the evidence. A certified producer repair under these rules settles it without a stop.

117. Twenty runs of each order, counted from zero, on the source of 116: test_forced_ordinary_pending_premise_in_both_orders[True], then [False] (bin/ changes after item 115). Run each until a failure is captured or twenty pass. Repair a captured failure at its producer under a recorded decision, or stop.

Then, on the green source, as §52.1 states them:
- make test at ten workers until green (item 114, with its stops);
- item 71's runs: the eight XOR_grammar fits first, then the three lesson-13 arms and the float64 discriminator;
- item 73's freeze, with ports against 8e9fe2634709, e0fc4a92719e and HEAD.

Pre-authorized steps and stops are as in §52.1.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review of the freeze; no commit or publication.
~~~

## 56. Review of the §55 return (Claude, 2026-11-06): item 116's repair approved; item 117, the suite, the XOR fits and the three float32 arms pass; the discriminator's construction failure follows the repair's change of trajectory; no freeze offered

The return was checked on the live tree, which equals Codex's candidate `a7851dde8a41` in all 995 source files. Against `8e9fe2634709`, two production files changed (`bin/ClauseRow.py` and `bin/ReferenceClose.py`), one test file was added and Architecture gained one record. Claude read all of it, and checked the receipts of the suite (output/tests/20261105-170621-ddd93f: 7,650 of 7,650 cases, exit 0, 699 seconds, peak 20.3 GiB) and of the forty runs.

**Approved as built.**
- *Item 116.* `ReferenceClose.is_forward_placeholder` is shared by the read and the thought bind: a token whose referent is only the forward word or that word's own objects does not discharge the name. A resolved kind takes precedence over an older naming symbol, so a selected native arrival still reads. `exclude` now does what `ThoughtClosing` records. Claude's captured stream is red on `8e9fe2634709` and green after, and the thirteen direct cases are red and green on the two producer files alone.
- *Item 117:* twenty fresh passes in each order, counted from zero.
- The default suite, the eight XOR_grammar fits, and the three float32 lesson arms, with no event at or above 1e8.

**Finding 1: the discriminator's construction failure (§55.9).** This stops the round.
- *The same run diverged.* The discriminator ran with the same command, construction stream (`7b339728410f`) and `PYTHONHASHSEED` 0 as in §53.9, where it completed all 160 sentences on `8e9fe2634709`. Claude compared the two runs' trial observations: identical through sentence 16, they first differ at sentence 17, the first of ideas epoch 2. So item 116 changed the trajectory, and the failure comes 35 sentences later.
- *What the capture shows.* At "What is that?" the root bind (leaves 0 to 3) carries `compose_bind` 3. That handle's origin is a journal-local action (tag 3, action 0). Before the root, the preview had mapped only handles 4 and 5, so `handles.get(3)` is `None` and `ReferenceClose.preview` raises (about line 875). `pending_names` is `(14,)`.
- *Two code facts bound the search.*
  - `ReferenceClose.sites` gives a word-bearing node a site only when it is an addressed introduction, a participant, or a sign or no-op leaf (about lines 700 to 702).
  - `ClauseJournal` registers a tag-3 introduction only under the conditions at about lines 309 and 310.

  So a compose bind can name a local introduction that no site writes.
- *Not yet known.* The capture cannot tell whether item 116 produces this state (for example a forward name it now keeps pending, whose introduction no site writes) or whether the new trajectory reaches an older one. A diagnostic replay can: float64 with the replayed stream is deterministic, as the identical first sixteen sentences show.
- *The recorded invariant settles the repair:* "Scoring and admission must enumerate the same sites. Introductions precede their uses" (`ReferenceClose.sites`, about line 790). Item 116's rule stays.

**Carried, unchanged:** §54's findings 2 to 7.

**No freeze was offered.** Codex stopped where §54.1 required. The text below repairs finding 1.

### 56.1 Text for Codex (2026-11-06)

~~~
Read §56. Item 116's repair, item 117 and the passed gates of §55 are approved as built. The float64 discriminator's construction failure (§55.9) stops the round. Order: 118, 119, then make test, item 71's runs and the freeze. Write receipts to doc/benchmarks/2026-11-06-item2-5-review56/.

118. A compose bind names an introduction that no site writes (§56 finding 1).
- Diagnose with the failed run itself. Run its exact command again on a7851dde8a41: the approved discriminator stream, PYTHONHASHSEED 0, float64. Add read-only probes on that sentence's identity transactions and on ReferenceClose.preview's site enumeration. This is a diagnostic replay of a captured model failure, not a retry. It must fail at attempted sentence 52 with the captured values; if it does not, stop.
- On 8e9fe2634709 the same stream completed all 160 sentences. The two runs' trial observations are identical through sentence 16 and first differ at sentence 17.
- Find which action introduced handle 3, and why no site before the root maps it. Find whether item 116's rule produces that state (for example a forward name it now keeps pending, whose introduction no site writes), or whether the new trajectory reaches an older one.
- What the capture shows:
  - the root bind (leaves 0 to 3) has compose_bind 3, origin tag 3 (journal-local action 0);
  - before the root, the preview mapped only handles 4 and 5;
  - pending_names is (14,);
  - ReferenceClose.sites (about 700 to 702) gives a word-bearing node a site only as an addressed introduction, a participant, or a sign or no-op leaf;
  - ClauseJournal registers a tag-3 introduction only under its conditions at about 309 and 310.
- Repair at the producer under the recorded invariant: scoring and admission enumerate the same sites, and introductions precede their uses (ReferenceClose.sites, about 790). A compose bind never names an introduction that the closing does not write. Item 116's rule stays: a forward name's own word or object does not discharge it. Stop if the repair needs a new decision.
- Certificates: a direct case of the condition, red on a7851dde8a41 and green after. Item 71's discriminator, from its unchanged stream, then certifies sentence 52 natively.

119. Twenty runs of each order of test_forced_ordinary_pending_premise_in_both_orders, counted from zero, on the source of 118, as item 117.

Then, on the green source, as §52.1 states them:
- make test at ten workers until green (item 114, with its stops);
- item 71's runs: the eight XOR_grammar fits first, then the three lesson-13 arms and the float64 discriminator;
- item 73's freeze, with ports against a7851dde8a41, 8e9fe2634709, e0fc4a92719e and HEAD.

Pre-authorized steps and stops are as in §52.1.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review of the freeze; no commit or publication.
~~~

### 56.2 Revised after Codex's diagnosis (Claude, 2026-11-06)

Before §56.1 reached it, Codex located the producer in `ChunkJournal.capture`, and Claude confirmed the reading in the code:
- `capture` selects journal rows with `columns.clamp_min(0)` (about line 321), so an action without a journal column, such as a SHIFT (kind 0), aliases row 0.
- The introduction search (about lines 369 and 370) scans every action. `by_column`, two blocks earlier (about line 334), keeps only folds and rewrites (kinds 1 and 2).
- In the capture, handle 3 records action 0, a SHIFT, while its introduction is action 5 (`lower_unary`). `ClauseJournal` (about lines 309 and 310) then registers no introduction at the real parent, no site writes it, and `ReferenceClose.preview` raises, as it should.
- When no introduction is found, the search falls back to action 0 without a word.

The producer is older than item 116, whose change of trajectory reached it. Item 116's rule stays. The text below replaces §56.1, whose diagnostic replay this reading makes unnecessary.

~~~
Read §56 and §56.2; this text replaces §56.1. Item 116's repair, item 117 and the passed gates of §55 are approved as built. The float64 discriminator's construction failure (§55.9) stops the round; your diagnosis of its producer is confirmed (§56.2). Order: 118, 119, then make test, item 71's runs and the freeze. Write receipts to doc/benchmarks/2026-11-06-item2-5-review56/.

118. An introduction's recorded origin (§56 finding 1, §56.2).
- ChunkJournal.capture selects rows with columns.clamp_min(0) (about 321), so an action without a journal column aliases row 0. The introduction search (about 369 and 370) scans every action, and falls back to action 0 when it finds none. In the capture, handle 3 records action 0, a SHIFT; its introduction is action 5, lower_unary.
- Repair at the producer:
  - search for an introduction only among folds and rewrites (kinds 1 and 2) with a valid journal column, as by_column already does (about 334);
  - raise at the producer when no introducing operation is found, in place of the action-0 fallback.
- Certify that no other reader of the clamped selection takes row 0 for an action without a journal column: the events table (about 410, masked only by kind > 0, so transposes included) and the timeline. Repair any that does under the same rule.
- Keep ReferenceClose.preview's raise. Item 116's rule stays.
- If the new raise fires for an identity that a recorded rule says has no introducing operation in its journal, stop with the case captured.
- Certificates:
  - a direct case in which a SHIFT aliases row 0: the recorded origin is the real introducing action, and the closing writes its introduction; red on a7851dde8a41, green after;
  - the discriminator's exact command with --limit 52, on the repaired source from its unchanged stream: sentence 52 completes. This checks the repair before the long runs; it is not item 71's run.

119. Twenty runs of each order of test_forced_ordinary_pending_premise_in_both_orders, counted from zero, on the source of 118, as item 117.

Then, on the green source, as §52.1 states them:
- make test at ten workers until green (item 114, with its stops);
- item 71's runs: the eight XOR_grammar fits first, then the three lesson-13 arms and the float64 discriminator;
- item 73's freeze, with ports against a7851dde8a41, 8e9fe2634709, e0fc4a92719e and HEAD.

Pre-authorized steps and stops are as in §52.1.

Work on main. No seed pinning beyond Alec's recorded 41001 and the approved diagnostic streams (replaying a recorded stream is one), no resize, no language named in code; fail loud; nothing retired silently. Return through Alec for Claude's review of the freeze; no commit or publication.
~~~
