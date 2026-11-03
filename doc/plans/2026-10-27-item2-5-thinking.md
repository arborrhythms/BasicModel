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
