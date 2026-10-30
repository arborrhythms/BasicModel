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
