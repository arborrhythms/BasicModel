# Item 3.0 step 3: STM as eight shared chunks, MTM symbols prepended, the overlay retired (Claude, 2026-10-25; for Codex, after step 2's review)

**Status.** The third round of item 3.0 ([hand-off](2026-10-24-handoff-3-0.md),
"Item 3.0, in order", 3). It realizes the STM Alec decided in step-2 spec
[§72](2026-10-21-step2-one-ltm.md#72-decided-alec-2026-10-23-the-four-open-points-of-71)
("STM: what is open" and the decided paragraphs after it) and the
identity spec's ["Integration by prepending"](2026-10-23-identity-threading.md)
(§2), on the layout step 1 landed (ideas in LTM0, chains in LTM1, the
journal as the chunk record) and the placement step 2 landed (promoted
identities placed by the surface). Step-2 spec
[§39](2026-10-21-step2-one-ltm.md#39-standing-rule-amended-alec-2026-10-22-codex-continues-documenting)
governs. Code names are read from BasicModel `4f136f76fa` and reconciled
with steps 1 and 2 where they renamed or removed them.

## 1. Decided (Alec; the record this round builds)

1. **STM is eight shared chunks**, each over a few ideas (as LTM0 holds
   them) and relations (as LTM1 holds them): one working set for the open
   sentence's constituents, recalled chains and recent rows together
   (step-2 §72, points 1 and 4, decided).
2. **Eviction:** the least recently read chunk that is not an operand
   still on the parse stack leaves when a ninth arrives; attention chooses
   what is read, recency what stays (§72, decided).
3. **Prepending:** MTM symbols are prepended to the symbol array that
   conceptual space reads, before the order rungs, so identities and
   recalled chains are leading entries of the compose window beside the
   words, available to compositions at every order; bind composes two
   field positions; the transitional identity overlay retires (identity
   spec §2; Alec: "The overlay is a transition implementation").
4. **Working values are never placed on the priming surface before
   promotion;** MTM writes nothing into a concept; a prepended identity's
   symbol id is its token key, a reserved transient id, until promotion
   gives it `symbol` (identity spec §2).
5. **An alternative timeline is a second prepended entry** for the same
   identity with that timeline's value; a modal clause's continuation is a
   bind to that entry (identity spec §2; iteration spec §4).
6. **A chunk's stamp:** an STM row has the LTM0 form `[form | meaning]`;
   a concept entering STM is stamped now in its percept `.when`; a
   recalled entry keeps the stamp it was stored with; an expectation's
   estimate is stamped ahead (layout spec §1 items 10 and 11).
7. **Trials own their chunks**, as the fork owns the stack today; the kept
   trial's chunks are what the close writes (§72, point 5).
8. **The thought loop's serial slot** becomes a chunk with the thought
   design, after this round (§72, point 6). Not built here.
9. **A mention's reading and an identity's role evidence live in MTM**
   (§72, points 1 and 3); a recalled chain brings its identities back
   through `query()`.

## 2. What exists, in code names

Three structures hold working content today:

- **The parse stack**, the six-tensor STM state `FunctionalPeerSTM`
  carries through `torch.while_loop`: `buffer [B, K, D]`, `depth [B]`,
  `orders [B, K]`, `grammar_orders [B, K]`, `concept_rows [B, K]`,
  `concept_activations [B, K]`, with `K = <stmCapacity>` (8;
  `Language.py` "DEFAULT_CAPACITY=8"). `choose_operation` gathers the two
  newest occupied slots as the compose `window` (`source`, `n = 2`) and
  `prepare_operands(window, identities, flags, positions, orders=...,
  live=..., active=...)` (`ReferenceContext.py`) enumerates the operation
  variants, returning `binary_refs [B, N-1, R, 2]`, `unary_refs`, the
  `left`/`right` operand slabs and the validity masks the compose layer
  reads (`Language.py` ≈7339, `reference_data['binary_refs']`).
- **The identity overlay**, `MTMOverlay` (`bin/MTMOverlay.py`): `stage`
  reads the accessible frames at the host boundary
  (`ReferenceClose.accessible(frames)` over `disc.situation_references(b)`,
  then `individuals(records)`), puts their values at indices `0 .. base`
  of a values tensor `[B, base + words + journal, D]` with `valid`,
  `stack [B, K, 7]`, `origins`, `actions`, and the language state carries
  it as `lang[26:identity_end]` (`identity_end` 31, or 38 with
  `TimelineOverlay` when a verb action exists; `Models.py` ≈19675,
  ≈20831 `lang = (*lang[:26], *overlay)`, ≈21832
  `identity_overlay.push(current_cs_lang[26:case_start], ...)`). `menu`
  builds the unary `bind` variants, one per location plus the dummy
  (`prepare_operands`, the `method_name == 'bind'` branch); `refresh`
  substitutes bound values into the window; `apply` opens a bind only at
  order 1 (`opening = (bind & (choice.order == 1)) | determined`).
  `ReferenceClose.commit` applies the kept trial's choices to `mtm` once.
- **The recent rows**, an index of addresses: the expectation predictor's
  `_inter_context[b]` and `pending.frames` (`SituationAddress`,
  `Layers.situation_references`, window `_inter_chain_window`), and
  thought's `memory.retrieved_frames` (`Thoughts.py` ≈329; `Models.py`
  ≈5867), read by bind's `accessible`.

The per-word symbol arrays the loop reads are InputSpace's
`_ar_word_concept_ids`, `_ar_word_concept_rows`, `_ar_word_concept_orders`,
`_ar_word_object_ids/rows/orders`, `_ar_word_symbol_where`,
`_ar_word_symbol_when`, `_ar_concept_lookup_rows`, all `[B, W]` (or `[B,
W, 4]` for the bands), written by the staging at `Models.py` ≈11354 and
read per word by `run_word_loop` / `word_concept_rows_at(p)`.
`_prime_sentence_symbols(sid)` bumps `_ar_concept_lookup_rows`.
`SentenceFork` reservoir-samples the detached `(stm, lang, feedback)`
state per row, so a trial already owns the stack and the overlay
together; `WalkTrials` compares the walks.

## 3. The chunk array

**One structure.** The six-tensor STM state becomes the chunk state, still
one tuple of preallocated tensors `[B, 8, ...]` (`<stmCapacity>` names the
chunk count; 8 is the Miller limit Alec set) carried by
`FunctionalPeerSTM` through the compiled loop:

- `buffer [B, 8, D]`: the chunk's vector, `[form | meaning]` as an LTM0
  idea is; `orders`, `grammar_orders`, `concept_rows`,
  `concept_activations` as today, per chunk.
- `chunk_refs [B, 8, 2, 2]` (long): two tagged references `(tag, value)`,
  tags `SYMBOL` (a concept or word id), `IDENTITY` (a token's transient
  id), `ACTION` (a journal column of this reading), `RELATION` (an LTM1
  handle, for a recalled chain head), `OCCURRENCE` (an LTM0 address key,
  for a recalled idea); zero invalid. A compressed chunk has none.
- `chunk_identity [B, 8]` (long): the bound token's transient id or `-1`.
- `chunk_timeline [B, 8]` (long): the identity's timeline index, 0 main.
- `chunk_stack [B, 8]` (long): position on the parse stack, `-1` off it;
  `depth [B]` stays the stack's depth.
- `chunk_read [B, 8]` (long): the loop tick of the chunk's last read.
- `chunk_where [B, 8, 4]`, `chunk_when [B, 8, 4]`: the percept bands,
  stamped by decision 6 (`RowTime` for a chunk stamped now).

**Allocation and eviction**: `FunctionalPeerSTM.allocate_chunk(state,
gate)` returns the index of a free chunk (`buffer` zero and `chunk_stack ==
-1`), else the chunk with the smallest `chunk_read` among `chunk_stack ==
-1`, clearing it (decision 2). With every chunk on the stack it returns
`-1` and the shift is refused, as a full stack refuses a shift today: the
chooser reduces. No spill structure (Claude's call under §39; §72's
early-write recommendation is recorded, not built).

**Reads refresh the tick**: a chunk is read when it is a compose operand
(the two newest stack chunks the window gathers), when the chooser keeps
a bind to it, when `refresh` substitutes its value into the window, and
when `query()` or thought returns it; the menu's offering a chunk is not
a read (attention chooses what is read).

**The stack is an order over chunks.** `choose_operation`'s `source`
maps the two newest stack positions through `chunk_stack` to chunk
indices; `window = buffer.gather(...)` as today. A reduce writes its result
into the head operand's chunk, frees the other operand's chunk (its
content is in the journal and in the result's `chunk_refs` where it did
not compress), and renumbers `chunk_stack`. A shift takes a chunk by
`allocate_chunk`.

**Compression during the parse** (§72 point 2, Claude's recommendation,
adopted as the chunk rule): a reduce whose operands are both resolved (a
concept symbol, a bound identity with a value, a compressed chunk) writes
its value and no references; otherwise it keeps `(left, right)` as
`ACTION` references to the journal columns of its operands, so the close
(`ClauseRows.write_clause`, step 1) finds the chain in the journal exactly
as it does now. "V O becomes a V" is this rule inside the sentence.

**Recent rows and recall.** At the kept close the sentence's top chunk
stays, `chunk_stack = -1`, with `chunk_refs` set to the written entry
(`OCCURRENCE` of its idea or `RELATION` of its chain's top row) and its
stored stamp; the other sentence chunks are freed. `query()` loads a
returned entry into a chunk by `allocate_chunk` with the entry's
references, vector and stored `.when`, and its identities re-enter MTM
(decision 9). `ReferenceClose.accessible` reads the chunks with
`chunk_identity >= 0` or an `OCCURRENCE` reference to a row whose subject
has a token, in place of `frames`; `memory.retrieved_frames` and
`situation_references` stay the expectation predictor's and thought's own
context (not STM) until the thought design (decision 8).

## 4. Prepending MTM symbols

**The array.** The staging at `Models.py` ≈11354 writes the per-word
arrays for `base + W` entries, `[B, base + W]`, with the first `base`
entries the accessible MTM symbols and the words after them; `base` is
the preallocated prepend capacity `<mtmPrepend>` (default 8, the chunk
count; written beside the buffers), `-1` padded. The entries, in order of
their token's most recent event (`Token.events`, latest first), are the
tokens `ReferenceClose.accessible` returns for the stream's chunks and
MTM's live pool, one entry per token, plus one more entry per alternative
timeline of a token that has one (decision 5). A token's entry carries:

- `_ar_word_concept_ids`: the token's **transient id**, from a
  preallocated pool `MediumTermMemory.transient_ids [mtmCapacity]` of ids
  `nVectors + i`, assigned at `open`, released at expiry or promotion;
  `_csw_row_of(transient)` is `None`, so the entry has no codebook row,
  is never bumped by `_prime_sentence_symbols` and never placed (decision
  4). A promoted token's entry carries its `symbol` instead, with its row.
- `_ar_word_concept_rows`: `-1` for a transient id; the symbol's row when
  promoted.
- `_ar_word_concept_orders`: 1 (an identity), or the promoted symbol's
  recorded order.
- `_ar_word_symbol_where`: the token's last mention's bracket, kept on the
  token as `Token.where [4]` at `open` and `bind` (new field, written from
  `_ar_word_symbol_where` of the mention).
- `_ar_word_symbol_when`: stamped now (`RowTime`, decision 6).

**The symbol's two parts** (`symbolize` for a prepended entry): the form
block is the token's name's form when the token has a reading
(`MediumTermMemory.names`, the word's row through the symbol index), else
the centroid (`MeaningKernels.centroid`) of its mentions' forms read from
their LTM0 rows; the meaning block is `Token.value` (or the timeline's
value for a timeline entry). The loop deposits each prepended entry as a
chunk with `chunk_identity` the token's transient id, `chunk_timeline` the
timeline index, `chunk_stack = -1`: in STM, readable, not on the parse
stack. When more entries exist than free chunks, `allocate_chunk` evicts
by recency, so the least recently read identities yield to the words.

**Bind as a compose over two field positions.** `BindLayer` moves from
`_compose_unary_rules` to `_compose_binary_rules` with `head_role 1`: its
left operand is the newest stack chunk (the mention), its right operand
one chunk off the stack with `chunk_identity >= 0`. `prepare_operands`
enumerates, for the `bind` rule, one variant per such chunk plus the
dummy (right operand the zero vector, `binary_refs[..., 1] = 0`), writing
the chunk's transient or promoted id into `binary_refs[..., 1]` and the
chunk index into `binary_choices`; this is `MTMOverlay.menu`'s
enumeration moved onto the chunk array, with `exclusions` (the witnessed
incompatibilities) applied as today. `BindLayer.forward(left, right)`
returns the identity chunk's current value for a chosen chunk and the
mention's own content for the dummy, as today's forward does by location.
No order restriction: the variants are offered whatever `choice.order`
is (decision 3's constraint; the `opening = bind & (order == 1)` guard
goes). The journal records the choice (`binding_choices`,
`ClauseJournal` formation) and `ReferenceClose.commit` applies the kept
trial's binds to `mtm` once, as today.

**The verb acts on the identity**, unchanged: the subject operand is
whatever the newest stack chunk holds, which after a bind is the identity
chunk's value; the close publishes the VP result to the token (`update`)
at the clause boundary, and the same write refreshes the identity chunk's
`buffer` row so the next clause reads it; a timeline entry's chunk is
refreshed from its timeline's value.

**Attention and expectation.** The attention mask over `.where` covers the
chunk bands (`chunk_where`), so prepended identities are in the field;
priming is unchanged (a transient id has no row); the expectation
predictor reads the chunk vectors it already reads as ideas and chain
heads (step 1), with no new predictor.

## 5. Retirement

- `MTMOverlay` (`stage`, `empty`, `read`, `write`, `push`, `refresh`,
  `menu`, `rule`, `apply`) and `TimelineOverlay` retire; `lang` is
  `lang[:26]` plus the case state (`case_start` follows 26);
  `identity_end`, `trace._identity_state` and the `[26:...]` slices go
  (`Models.py` ≈19676, ≈20831, ≈20867, ≈21832, ≈22315).
- `ReferenceClose.accessible(frames)` becomes `accessible(chunks)`;
  `MTMOverlay.stage`'s host read of `individuals(records)` becomes the
  prepend staging of §4.
- `SituationAddress` and `SituationFrame` stay for the expectation
  predictor and `TimelineRead`; `memory.retrieved_frames` stays for
  thought until decision 8's round.
- The identity spec's `Lifetime` paragraph holds: `values`, `names` and
  the transient id pool serialize with the pool in the codebook's vocab
  extras (`vocab_extras`, version bumped); a checkpoint without the pool's
  transient ids is rejected by version.

## 6. Trials and gradient

`SentenceFork` already samples the detached `(stm, lang, feedback)` per
row; with the chunks inside `stm` and no overlay in `lang`, each trial
owns its chunks with no further mechanism (decision 7); `WalkTrials`
compares as today; the kept trial's chunk state is what `write_clause` and
`commit` read. Chunks are the compiled loop's differentiable state: a
recalled chain's vectors and a prepended identity's value enter detached
at the host boundary (as `stage` stages values today); within a sentence
the VP output is a function of the identity chunk's value, so output
prediction's gradient reaches the VP and the bind through the derivative,
and the committed value is detached at keep (identity spec §2, "Tensors
and gradient").

## 7. Ports

The identity file's overlay tests (`MTMOverlay.menu`, `apply`'s opening
rule, `lang[26:]` slices, `TimelineOverlay`), the bind-menu tests that
enumerate overlay locations, and any test asserting the three-structure
STM (stack depth as capacity, `retrieved_frames` as bind's source) are
ported to the chunk array with exact before and after text in the
receipt; the modal's timeline tests port to the second-prepended-entry
form. `<stmCapacity>` keeps its name and meaning (8); `<mtmPrepend>` is
new. A port that changes a decided behavior stops for Alec.

## 8. Gates (sized to the change; one frozen hash)

1. **Identity spec §4's four gates** pass with the overlay gone: the
   second VP receives the first VP's result through the subject identity;
   change of state keeps the latest place; two aliases read one value and
   a description mention joins; both-trial isolation with opposite keep
   outcomes.
2. **Bind over two field positions.** With one mention and two prepended
   identities, `prepare_operands` offers the `bind` rule with exactly
   three variants (two chunks and the dummy), `binary_refs[..., 1]` naming
   the transient ids; the kept choice's token receives the mention in
   `commit`; the variants are offered identically at a mention of order 1
   and of order 2.
3. **Eviction.** With eight occupied chunks, a ninth arrival evicts the
   chunk with the smallest `chunk_read` among those off the stack and
   never one on the stack; a chunk read as an operand or kept as a bind
   has a later tick than one merely offered.
4. **The shared set.** After a close, the sentence's top chunk remains
   off the stack with its written reference and stamp and is bind-
   accessible in the next sentence; `query()` on a stored chain loads its
   head into a chunk with a `RELATION` reference and its identities into
   MTM; the words of the next sentence take chunks by eviction order.
5. **Prepending.** A transient id has no codebook row, is not bumped by
   `_prime_sentence_symbols`, and `placement_witnesses` at no row changes
   for it; a promoted identity's entry carries its `symbol` and row and is
   placed at its next registered sentence (step 2).
6. **Timelines.** A token with an alternative timeline yields two
   prepended entries; a modal clause's continuation binds to the timeline
   entry; iteration 2's alternative-timeline gates pass in the ported
   form.
7. **Retirement.** `lang` has 26 columns before the case state; no module
   imports `MTMOverlay` or `TimelineOverlay`; a checkpoint without the
   transient id pool is rejected by version.
8. **Unchanged sets on the frozen hash:** the layout gates as steps 1 and
   2 left them, G1–G3, the pending-premise certificates, the reference,
   bind-thought and identity files, the XOR standing gate and both
   decomposition arms. The sweep is the item's one, at the end of 3.0.

## 9. Not in this round; Claude's calls; Alec's decisions

**Not in this round.** The thought loop's serial slot as a chunk, and
descend/return over a chain's references (decision 8, the thought
design); retire mode (step 4); orders learned (step 5); certainty.

**Claude's calls under §39:** no spill structure when all eight chunks are
on the stack (the shift is refused and the chooser reduces); prepend
order by the token's latest event; the expectation predictor's
`_inter_context` and thought's `retrieved_frames` kept as their owners'
context rather than folded into the chunks this round; `ACTION`
references to journal columns as a chunk's uncompressed references;
`<mtmPrepend>` default 8.

**Alec's decisions:** none open for this round at this writing. If step
1's or step 2's landing changes a name or a shape this section relies on,
Codex reconciles under §39 and records it in a dated section here.


## 10. Incoming-source review and prerequisite repairs — Codex, 2026-10-25

The 888 source/config/test paths match the approved placement receipt's source
`1e94238b57` before these changes. Alec owns the commit; this round uses that
working tree without staging, committing or publishing. The local step-3
receipt retains the incoming manifest, archive and this spec as received.

**Recurrence bound (§21 finding 1).** ConceptualSpace's `recurrenceCapacity`
default is its resolved `nVectors`. The five recurrence summary slabs and the
first-identity sidecar use that bound; `recurrence_readings` remains one entry
per LTM0 episode row. A standalone LTM0 with no concept owner uses its explicitly
constructed row bound unless its caller supplies `recurrence_capacity`; the
production SymbolSpace constructor always supplies the ConceptualSpace bound.
Exhaustion remains loud. The recurrence checkpoint is version 2, LTM truth
schema 13 and semantic extras 9; old layouts and changed construction bounds
are rejected rather than resized.

**Unforced two-stream repair (§21 finding 2).** The first diagnostic run
preserved the original native-ID failure (47 passes, one failure). It showed
that the timeline's selected unary replay passed a two-position CaseView to
one selected operand: `[2,1,128]` meaning versus `[2,2,128]` case output. The
replay now gathers that operand's case view by the same selected position as
its form. A per-stream CaseView also restores the singleton position axis
before combining it with a unary value. The four new prerequisite checks and
the case-view, alternative, consolidation and native-ID files pass: 48 cases.
The native-ID fixture is already unforced in source and remains so. Its
regression restores exactly the placement receipt's first failing RNG capture
`6d431a2258f041da4e82-30485.rng.pt` (SHA256
`6aae781f62a90b228ca400923febf6e665ce061ad620f6d0963fb829de0b46d4`),
without choosing a seed or substituting an operation. Both attempts are in
`doc/benchmarks/2026-10-25-item3-0-stm-chunks/development/`.

**Reconciliation for the chunk implementation.** `FunctionalPeerSTM` is a
class in `Models.py`; LTM0 and the runtime STM owner are in `Layers.py`.
The grammatical action journal is already temporary, and the 26-column
language tuple carries clause scope and operation references. Grammar
provenance stays in that scope/journal; working identity and timeline values
move into the same eight chunks as sentence constituents. No second live
value bank replaces the retiring overlays. A reduce's grammatical stack
effect must distinguish its second field operand from a second parse-stack
operand: binary bind reads an off-stack identity and rewrites one stack
position. The other binary rules continue reducing two stack positions.

## 11. Chunk integration decisions — Codex, 2026-10-25

The construction now owns thirteen STM tensors; physical chunk indices and
parse positions are separate. The existing 26-column language tuple keeps
integer grammar provenance in its scope and operation-reference journal:
14 integers per scope position and 36 per journal action. These replace the
overlays' transaction metadata without another live identity/timeline value
bank. Numerical before/operand values remain in the existing action journal.
Binary bind has one parse-stack operand and one field operand. Its forest,
form-count, case-flow and inverse readers use that stack effect; the inverse
retains the original mention as an occurrence witness. Its thought face keeps
the independent unary contract until the thought round.

The symbol fields are construction-bounded at `mtmPrepend + W`. The existing
inverse lookup bank already contains both word and object blocks, so that
bank is `mtmPrepend + 2W`. Word-indexed grammar/reconstruction readers take the
word suffix; priming reads the full bank after prepending. Tests that indexed
word masks against the old unprefixed arrays are ported to that suffix.

A trial reserves transient ids in its own chunks and integer journal, using
the free-seat snapshot of the constructed MTM pool. The pool changes only at
the kept close. The close maps the journal's identity handles to the actual
assigned pool seats, including when simultaneous streams reserved the same
candidate, then refreshes the retained identity chunks. `Token.where` is
written from the kept mention's bracket. Sentence resets release parse-stack
positions while retaining recent and recalled chunks for recency eviction.

Bind returns the chosen identity's current value, as specified. Publishing
the kept bind joins the mention into that working identity before a following
operand refresh; the close applies the same kept membership once. This
preserves the identity certificate's description-join behavior while moving
its source from the overlay to the shared chunks.

The recorded native two-stream initialization and the new prerequisite/chunk
checks pass together (13 tests, development `chunk-wiring-09.log`). These are
development checks, not the source freeze or the round's measurement receipt.

## 12. Lifetime and integration review — Codex, 2026-10-25

The development review found and repaired the following within §1's decisions.

* **Transient reuse.** An identity chunk's second reference records its MTM
  origin as an OCCURRENCE. Staging resolves that origin before interpreting the
  transient id, invalidates expired identities and remaps promoted ones. The
  same reserved id can therefore be reused without an old chunk becoming an
  alias of the new token. A recent sentence keeps its own first reference and
  its token origin in the second; prepending does not overwrite that recent
  entry merely because it shares an identity. No extra epoch or value slab is
  introduced.
* **Bind at every order.** The binary menu alone was insufficient: the closing
  reader also had an order-1 filter. Explicit recorded bind requests now pass
  through that reader at their actual order. The dummy opens a trial identity
  at any order; a selected entry records the token and timeline. The chooser
  scores bind's mention occupation without borrowing an adjacent stack word's
  occupation. Thought bind retains its separate contract.
* **Journal recovery.** The integer journal record has **37 columns**, updating
  §11's initial 36: one publication-kind code distinguishes the ordinary result,
  an equality's complement and a part join. It recovers an evicted identity's
  exact published value from the existing numerical operands. Shift sources use
  negative leaf-column encoding (`-word-1`); operation sources use the existing
  one-based action-column encoding. ACTION references keep those journal source
  codes until compression. The scope/journal remains provenance, not another
  live working-value store.
* **Pool and trial isolation.** Free-id reservations include evicted
  introductions still named in the journal. The host stages the bounded pending
  id list once; a trial does not consult or mutate Python pool state. At the
  kept close, reservations map to the pool's actual seats before publishing
  chunks. This accommodates two streams reserving the same free id independently.
* **Timeline continuation.** The old overlay's automatic branch substitution
  retires. A continuation explicitly chooses the second prepended entry. A later
  actual event is not rolled back when an earlier event receives a post-modifier.
* **Staging and stamps.** Native input uses the largest constructed serial-word
  bucket; mixing input uses its constructed perceptual word slots. Both allocate
  the leading field at construction. The lexical inverse uses the suffix because
  a transient entry has no independent spelling; priming reads the leading field
  too, ignoring row `-1`. Shifted concepts and prepended identities receive the
  current percept stamp in both buffer and sidecar. Recalled heads retain the
  stored stamp. A relational close retains its head vector even when `Clause.point`
  is absent. Metadata setters preserve the full constructed batch and reject growth.
* **Access and recall.** The live pool accompanies the chunks, so an empty old
  situation-frame fixture no longer hides live identities. The G2 fixture selects
  its intended dummy explicitly. Query loads returned heads into the shared set
  under the existing work meter and returns their identities to MTM. Predictor
  and thought frames remain their owners' context; the thought serial slot does
  not become a chunk in this round.

New field certificates cover both bind orders and a kept membership write,
recall with eviction and stored time, transient expiry/reuse and promotion,
priming/placement exclusion, and checkpoint retirement. The existing identity
and modal tests exercise the actual chunk/journal path, including derivatives
and opposite kept-trial outcomes. Development failures and fixture repairs are
retained in the local receipt; they are not frozen-source measurements.

The measurement adapter retains every original decomposition parameter by its
catalogue identity. Binary bind now occupies a new binary row; it no longer
occupies the unary list. The original seeds, snapshots, budgets, question rows
and proof assertions stay fixed. The footprint observer reports raw changes;
its permitted working changes include the thirteen constructed STM members
loaded by query, in addition to step 2's nonpersistent form-validity cache and
work counter. Parameters, durable row values, placement and witnesses remain
in the original comparison. The final receipt will carry exact test/helper
before-and-after text and the source/measurement hashes.

## 13. Reader and measurement reconciliation — Codex, 2026-10-25

**Percept coordinates and membership maps.** Entering chunks carry the current
`.where/.when` in their buffer as well as their sidecars. Signed location
coordinates cannot enter a `[0,1]` membership chart. Composition and inverse
reconstruction therefore zero the declared percept band before their content
kernels; the chunk and the kept reading retain their stamps. The original
packing comparisons remain numerical and unchanged: all reconstructed content,
roots, actions and byte costs are still compared. This repairs the development
failure on a negative location coordinate without clamping membership values.

**Trial names when the pool is small.** An open reading can introduce more
identities than `mtmCapacity`; committing it can evict earlier introductions.
The fixed pool reserves only its own free ids. Beyond those, the existing
integer journal uses private provenance names starting after the pool, bounded
by `W + W*(3 + 2*stmCapacity)` (one shift and the constructed action columns per
word). These names never become MTM tokens or codebook ids; the close maps each
surviving introduction to its actual pool seat. Their allocation reads the
maximum name already present in provenance, not an enlarging candidate slab.

**The conditional bind audit.** Its former retained-frame menus of 1, 4 and 16
candidates become shared-chunk menus of 1, 4 and 7, beside the mention in the
eighth chunk, at orders 0, 1 and 2. The ordinary unforced document reads and the
unchanged `[.9,1.1]` content-probability band remain. The dummy and each physical
chunk are distinct variants; two chunks may carry the same token (a recent row
or another timeline), so the audit records both chunk choice and identity id.
It does not deduplicate the menu by token id or reintroduce implicit mint/bind
variants on unrelated rules.

**Measurement construction.** Packing cases run one per bounded worker: three
cases together exceeded the 8 GiB development worker bound, despite the first
case passing alone. All three ambient initializations remain. The original
decomposition snapshots map every surviving action row by grammar declaration,
including repeated declarations of the same numerical operation; `list.index`
would alias those rows after binary bind is inserted. The original seeds,
budgets and proof assertions remain. These repairs occur before the one smoke
and source freeze; development attempts are retained separately.

## 14. Shared-field lifetime and reader repairs — Codex, 2026-10-25

**Current state and historical uses.** `accessible(chunks)` admits only tokens
reached through the chunks. Its bounded MTM mention history supplies the
individual uses and their role/predicate evidence. Each record exposes the
current token value separately from its historical mention value; grouping can
therefore use the witnessed distinctions without resurrecting a past working
state. `include_pool=True` is the prepend staging boundary, not the ordinary
closing reader. Old fixtures that supplied only an unpooled situation address
now explicitly recall it first. A recalled promoted individual retains its
native identity, with separate working tokens for separate streams. A direct
bind fixture's dummy opens its own occurrence at any order, just as the recorded
binary bind does. This refines §12's access paragraph; live pool entries do not
silently make an otherwise empty closing context accessible.

**Publication and recent rows.** A recent sentence and a prepended identity can
share an identity id. Publication, operand refresh, case-object reads and journal
capture now prefer the actual field entry (first reference SYMBOL or IDENTITY)
over the recent sentence's OCCURRENCE/RELATION entry. Otherwise a VP could update
one seat while the next operation read the other seat's stale value. The forced
decomposition fixture's former MTM-location choice now explicitly chooses that
field seat; the original seeds, snapshots, budgets and proof assertions stay.
The recent sentence keeps its written reference and stored stamp. Newly
published identity chunks inherit the mentioning head's percept bands; refreshing
an existing field retains its entry stamp and refreshes its read tick. Buffer and
sidecar stamps agree, including relational recent entries.

**An event is not a surface attachment.** The prior timeline owner published an
ordinary state change at the predication. It carried that event through ordinary
surface attachment without publishing another actual change. The first chunk
implementation mistakenly published and suppressed the prior event at each such
attachment. The journal now carries that provenance; only a post-modifier or a
wrapper of an alternative republishes the alternative. Publication-kind 3 in the
existing 37th integer column identifies an opening, so capture does not mistake
an ordinary inherited head for a new bind. No column or value slab is added.

**Unpublished alternatives and occupied zeros.** An uncommitted trial alternative
is cleared off the stack at close; a top chunk retains its written question but
loses the unpublished identity/timeline. A zero-valued deposited chunk is still
occupied: `chunk_read == 0` distinguishes an unused seat, avoiding accidental
eviction of a recently read zero value. These are implementation reconciliations
of the existing lifetime decisions, not additional trial mechanisms.

**Content at numerical readers.** A legacy equal-width model has no separate
meaning chart. Prepending preserves its token content instead of replacing it
with a lexical name's form; paired models use the specified form/meaning split.
Event witnesses read by `Queries._with_components` remove the declared percept
bands, matching the operands consumed by content kernels. `IdeaEvaluation.py`
and `BasicModel.begin_presentation` remain byte/function-text identical to the
approved source. No numerical proof tolerance, work budget or inverse search is
changed.

**Clock certificate port.** The nine recorded clock presentations retain their
original seed, presentation count, selected readings, native identity uniqueness,
and exact presented-content/reference comparisons. The old certificate compared
all forecast-derived values and final answers as if time could not affect the
predictor. Entering chunks now carry time in their vectors (§1.6); the unchanged
predictor can consequently change its forecast, omission evidence, surprise,
recurrence weights and answer readout. The port records all of those raw series
and checks their finiteness, while preserving exact equality for presented
content and the remaining unaffected state. It is a presented-content invariance
certificate, not a claim that the whole predictor or answer is clock-invariant.
Exact prior/current assertions are included in the receipt's port ledger.

## 15. Final integration repairs before measurement — Codex, 2026-10-25

**An MTM staging copy is an arrival.** Reading a live token into the next
sentence's prepended field refreshes that chunk's recency, including when its
physical seat is reused. All entries copied at that boundary receive the same
final tick; an entry already copied is protected from being treated as stale
while the rest of the field is deposited. Merely enumerating the bind menu
still changes no tick. Preserving the prior sentence's tick while copying and
stamping a fresh entry allowed a just-staged target to be evicted before its
mention. The original two-link decomposition certificate exposed this defect.
The regression tests staging, untouched seats, the menu and the next eviction
separately. This applies §1.2 and §1.6 without another value store.

**A dummy is an explicit compose choice.** Binary bind's recorded dummy opens
its own mention at every order. An abstract unresolved bind with no word and
no recorded dummy remains open, as the unchanged thought certificates require.
The initial broad `mode == bind` admission confused these cases. The direct
complete-mention fixture now records a dummy when its chooser selects no
antecedent; unresolved thought fixtures do not. This makes §14's phrase
"direct bind fixture's dummy" precise. The role-2 context fixture deposits the
recalled object participant in its own shared chunk, instead of treating an
old frame activation as a subject identity. The cover fixture retains its
native symbol and verifies that the new mention joins the same recalled token.
The existing G1–G3 assertions and pending-premise proof assertions are unchanged.

**Reduction stamps and construction fixtures.** A reduce writes content into
its head chunk and restores that head's percept bands in the buffer from the
chunk sidecars. Content kernels continue to exclude those bands. A direct
regression checks both representations. The structural chooser fixture now
constructs eight chunks and its declared rule heads; the verb-teacher fixture
uses the journal instance that owns the percept-band declaration. The `.when`
reader inventory explicitly includes the bounded MTM event-history reader.
These are fixture ports, not new thought behavior.

The first failed reverse pending-premise initialization from development
`core04` is retained as `c96357c9fd39f1f89019-47470.rng.pt`, SHA256
`becf69d8a0c29608e187dc402bc77132f3210fa05ec450192bb8930545ceeb72`.
Its unchanged certificate passes when replayed after these integration repairs
(`pending26.log`). The replay is a regression on the observed failure, not a
selected passing initialization. Final scoped measurement captures fresh RNG
states, except for the required recorded compound-case regression and the
original decomposition arms' standing snapshots/seeds.

The reduction-stamp regression includes an empty inactive stream beside an
active reduction. The original two-link arm exposed a missing safe gather index
for that inactive row; the write remains gated, so every inactive chunk tensor
stays exactly unchanged. Development `core05` was explicitly interrupted when
this defect was found (154/1,059 cases completed), rather than measuring a
source being edited. The single formal freeze has not yet occurred.

## 16. Packing and field-audit fixture reconciliation — Codex, 2026-10-25

The three ambient packing initializations exposed an intended difference in
the root's eight percept coordinates: the second packed sentence has a later
input offset and arrival stamp, while its separately presented counterpart
starts a new input field. The old comparison was valid while these coordinates
were zero. The port retains all content, actions, inverse, byte-cost and
truncation comparisons at their original tolerances. It checks each raw root
stamp against that layout's staged head fields and still compares the owned
program root exactly against the raw root. Both raw stamps are recorded.
No comparison of meaningful content is dropped and no clock or input bracket
is changed to make layouts coincide. This refines §13's packing description
for the now-stamped reductions in §15.

The associated source audit found two old root reads using `buffer[depth-1]`
after physical chunks ceased to be stack positions. The differentiable read
uses the stack view. After the kept close removes the parse positions, the
batch-end read uses the retained per-sentence root. The packing test also
checks the published answer root against its retained completed field, so a
recent chunk at physical seat zero cannot silently become the answer.

The conditional distribution fixture supplied chunks but omitted the reference
scope that enables the actual field menu. On an initialization whose ordinary
reads offered no retained identity, this left all three audit sources empty.
The controlled context now supplies its two-column scope and an empty predictor
bank while retaining the same untrained chooser. It must observe all nine
count/order combinations (1/4/7 candidates at orders 0/1/2), including the dummy.
It forces no selected operation or answer. The ordinary document reads and the
unchanged [0.9,1.1] conditional probability band remain.

The XOR standing gate found an additional construction-bound defect: installing
a smaller active batch truncated the clause-scope owner, then a later larger
batch could not reset it. The metadata owners now retain their full constructed
stream extent, like the thirteen chunk members. Clause scope reserves fourteen
integer provenance columns (a two-column legacy view remains valid), whole
provenance reserves five (three indices plus two form-span endpoints), and unary
history reserves one entry per chunk plus one depth per stream. Active-prefix
writes preserve the inactive suffix; width/device validation never grows a
slab. This is the same no-resize contract, not a change to XOR or its thresholds.

## 17. Native lesson reader repair before the freeze — Codex, 2026-10-25

The first short native lesson failed in `_condition_answer_on_question` before
the source freeze. Unlike the already reconciled composition/inverse readers,
that reader applied the cube membership chart to the chunk's signed percept
coordinates. It now conditions the content on either side of the declared
percept band and carries that band's position and time unchanged. Dense event
conditioning retains its existing linear law. A direct regression verifies
signed stamps, the first occupied role, content gradients and the absence of
conditioner gradients into the percept band. `IdeaEvaluation` and the training
presentation boundary are unchanged.

The failed smoke and its RNG remain in the receipt's
`development/native-smoke-first`. The replay uses that exact initialization,
SHA256 `60b0e1bedf3adf343c382d67a7e65e04778c15359f624dbe65d3ba62fd8e8395`;
it does not search seeds. This is a repair/replay of the requested short lesson,
not an additional lesson or a source freeze. The final measurement will report
both attempts explicitly.

## 18. Frozen decomposition outcomes and review flag — Codex, 2026-10-25

The single source freeze is SHA256
`b21d4068cd803e4e9dd45270a9537a4884660cbc6ad1966adff2327436fbb85a`.
Both original decomposition arms pass their one- and two-link standing
certificates on that source. The test file is byte-for-byte unchanged from the
approved placement source (SHA256
`f53bfb85ff80ca978c93d2085d8ac1d59f03f0e41c01a039084094f9abab2c5f`).
It explicitly defines an existence demonstration, retaining every companion
outcome rather than requiring an accuracy bar. Each passing proof retains its
bound referent, exact inference/descend/return counts, final conclude, work
budget and durable-write checks.

**Review flag:** companion outcomes are **4/8 correct**, down from placement's
**8/8**: one of the two questions in each length/arm. The first stream supplies
the required proof; the second has different kept premise readings. In the
one-link run, for example, its assignment has no resolved equality references
and its license closes as a single symbol, rather than the two equalities the
first stream keeps. The standing fixture leaves the explored suffix and keep
decision live, so restoring the old companion score by forcing those choices
would change the measurement. No seed, budget, proof assertion or keep decision
was changed. This is not reported as eight successful answers and is not
silently assigned to a later item; Alec should include it in Claude's review.
The receipt's `decomposition-outcomes.json` preserves the approved and current
question records, current premise readings and departure records together.

## 19. Frozen measurement and handoff for review — Codex, 2026-10-25

All five scoped groups completed on §18's single frozen source: **1,070 passed,
seven existing prerequisite skips, zero test failures**. Core is 1,054 passes
and seven skips; packing is 4/4; XOR is 8/8; both original decomposition arms
are 2/2. The native lesson's same-initialization repair replay passed before the
freeze; its original failure remains recorded. No source or protocol changed
after freezing, and no full sweep ran.

The receipt at
`doc/benchmarks/2026-10-25-item3-0-stm-chunks/README.md` contains the source archive,
73-path delta from approved placement, protocol hashes, all formal logs,
placement's two deterministic series and exact before/after text for 88 ported
test bodies. It also records 22 new bodies, zero removed or renamed tests, and
the explicit retirement of `MTMOverlay` and `TimelineOverlay`. Seven standing
prerequisites remain skipped (copular identity one, pending-premise learning
two, MTM reidentification four). `IdeaEvaluation.py` and
`BasicModel.begin_presentation` retain their approved source hashes.

This is a review candidate, not a landing or approval. Alec should return it to
Claude with §18's **4/8 companion-answer review flag**; passing the unchanged
decomposition proof certificates does not restore placement's 8/8 outcome.
Alec owns commits. No commit or publication was made. Steps 4–5, the thought
serial-slot design and the one end-of-item sweep remain outside this round.

## 20. Claude's review of step 3 (source `b21d4068cd`) — 2026-10-25: not yet approved

Reviewed: §§10–19, the [receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/README.md),
`summary.json`, `verification.json`, `frozen-source.json`,
`decomposition-outcomes.json` and both arms' `progress.json` against the
placement baseline's, `ports/manifest.json`, the clock certificate's before
and after text in `ports/tests-before-after.md`, and the new text of
`STMChunks.py`, `PrependedSymbols.py`, `BindLayer`, `MediumTermMemory`'s
transient pool, `ReferenceClose.accessible(chunks)` and the config schema.

**Verified.** All 893 manifest paths hash to their entries and the canonical
map hashes to `b21d4068cd…`; nothing changed after the one freeze. The five
groups: core 1,054 passed with the seven known skips, packing 4, XOR 8,
both decomposition arms 2 and 2 on their original seeds, snapshots and
budgets, the standing decomposition test byte-identical to placement's
(`f53bfb85ff…`). `IdeaEvaluation.py` and `begin_presentation` match their
approved hashes; no language is named in any added code or fixture and no
seed is added; 88 ported bodies, 22 added, none removed or renamed;
`MTMOverlay.py` and `TimelineOverlay.py` are gone, `lang` ends at column 26
before the case state, `accessible` reads chunks, the transient id pool is
`nVectors + i` over `mtmCapacity`, `recurrenceCapacity` defaults to
`nVectors` (§21 finding 1 of the placement spec), and the `CompoundUnaryLayer`
two-stream case is repaired with a regression on placement's recorded
failing initialization (§21 finding 2). The chunk state is the thirteen
preallocated tensors of §3 with eviction, read ticks, stack-as-order,
compression and recent rows as specified; prepending is construction-bounded
at `mtmPrepend + W`; bind is binary with one field operand; the new gate
files cover §8's eight gates.

**Findings.**

1. *Blocking: the second stream loses its readings.* §18 reports companion
   answers 4/8 against placement's 8/8 and attributes it to "different kept
   premise readings" in the second stream. The certificates say more. In
   every one of the four runs the lost answer is document 1's, and its
   `progress.json` shows that stream's first sentence with no reading at all
   (`(1, 0): [null, null, null]`) and its third with only a bare symbol
   (`(1, 2): [[sym 38], null, null]`), where placement's baseline had full
   readings on both (`(1, 0): [ltm, sym, sym 20]`, `(1, 2): [ltm, sym, sym
   39]`). Stream 0 is intact in all four. The seeds, snapshots, budgets and
   keep decisions are unchanged, so this is not the explored suffix: a
   second-stream mechanism defect in the new chunk, prepend or staging
   path drops the kept readings, and the decomposition certificate passes
   only because it needs one stream's proof. Candidates the round's own
   history suggests: the staging-copy recency repair of §15 (a just-staged
   target evicted before its mention), `allocate_chunk` or `at()` on a
   second stream after the first stream's allocation, `_sentence_seen_rows`
   or the pool snapshot read for stream 0 only. Repair, with a regression
   on the two-stream reading loss, a new freeze and the five groups again;
   8/8 restored, or the cause shown to be a decided behavior, which it is
   not from here.
2. *For Alec: the clock certificate was weakened.* `test_clock_independence`
   asserted the answer byte-identical across the nine clock translations;
   the port keeps presented content, references and programs exact and
   asserts only finiteness of the answers and forecasts, because entering
   chunks now carry their `.when` in the vector (§1.6) and the predictor
   reads it. Carrying time is decided; letting the answer change under a
   pure translation of the clock is not, and 5.5 defined tense as the
   relation between a stamp and now, which is translation-invariant. This
   port changes a decided behavior and under §7 stops for Alec. Recommended:
   the predictor and the answer reader see stamps relative to the current
   tick (the sinusoidal `RowTime` band rotated by now before those readers,
   content kernels already excluding the band), and the original answer
   equality assertion is restored. The alternative is to accept the weaker
   certificate and record that answers depend on absolute clock.
3. *The percept certificate is absent*, through no fault of this round: the
   FutureWork entry adding it to step 3's gates postdates the Codex text. It
   goes into step 4's text as an unchanged-set file.
4. *Fixture choices made explicit* (the G2 dummy, the decomposition
   fixture's field seat, the conditional audit's scope) are setup repairs
   within §39 and are disclosed; none changes an assertion on a decided
   behavior. Noted, not objected to.

**Verdict.** Not approved. The mechanism is the spec and the measurement is
honest, including the flag, but finding 1 is a regression in the model's
reading of a second stream and lands nothing until it is repaired on a new
freeze. Finding 2 is Alec's; its disposition joins the same repair round.
No commit of this round until then; steps 1 and 2 remain approved and
uncommitted in the same tree.

*Finding 2 decided (Alec, 2026-10-25): B, absolute stamps with now as an
input.* "Please do B, since it will give a time to the LTM entries that are
saved." Entries keep their absolute `.when` as written, so recall by time and
the stored stamps are untouched; the predictor is given the current tick as
an input beside the chunk vectors (the `RowTime` encoding of now, four
coordinates, appended to `ActivationPredictor`'s context at a fixed,
construction-sized position) and stamps its estimate ahead of it. Clock
translation invariance of the answer is then learnable, not structural: the
ported `test_clock_independence` stands with presented content, references
and programs exact and the forecast series recorded, and the invariance
claim moves to item 0's learning measurement. Recorded here; option A
(relative stamps at the predictor's input) is withdrawn.

## 21. Repair round stopped at the percept certificate — Codex, 2026-10-25

**Status: not approved; no commit.** The requested repair round began on
`b21d4068cd803e4e9dd45270a9537a4884660cbc6ad1966adff2327436fbb85a`.
All 893 paths in that rejected source still match their recorded hashes.
The only added source file is `test/test_percept_transitivity.py`; no
production code or existing test was edited. The separate
[repair receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/repair-01/README.md)
retains the rejected evidence, the diagnostic replay and the failed percept
certificate. It does not replace §§18–20 or the original receipt.

**Finding 1: partial diagnosis, not a disposition.** The zero-arm,
length-1 replay uses seed `4199722147`, the original `0.pt` initialization,
budget 512 and the original proof assertions. Read-only instrumentation
records `SymbolEntries`, both streams' seen rows and free pool seats,
`allocate_chunk`, `at`, read recency, the bind menu, `accessible`, both
trial observations and commit. The first sentence has no prepended identities:
both stacks are empty, both staged word-id rows and seen rows agree, and
all 64 transient seats are free. Initial allocation returns chunk 0 for
each stream. `accessible` is empty for both streams at staging and at both
trial closes; the §15 staging-copy refresh has no target on this sentence.

Both exploit programs end in `equal` and carry `_equality=True`. Stream 1's
explored program instead ends in `conjunction` (binary operation 19 rather
than 2), with mask `[true, true, false]` and no equality binding. The keep
vector is `[false, true]`; the admitted observation retains that difference.
This locates the missing equality before commit admission, but does **not**
establish why the losing semantic path is kept or rule out a chunk-path
defect. The cost and candidate/writeback diagnosis remains unfinished.
There is no regression fix, no new two-stream regression and no claim that
the 4/8 result is chooser variance or a decided behavior. The original
8/8-to-4/8 companion records are preserved, with no repair result invented.

**Finding 2: decision B stands; implementation deferred by the stop below.**
Absolute entry stamps, time recall and content-band exclusion remain as
specified in §20. The required four-coordinate current `RowTime` input at
a fixed construction-sized context position, future estimate stamp and
same-chunks/two-ticks regression have not been implemented. The existing
clock certificate is unchanged; its equality assertion has not been restored.

**Finding 3: the read-only certificate fails; stop as instructed.** The new
file audits the two kept readings from the existing short native occupation
lesson. It reads the staged identity-part rows and the separately preserved
`native_indices` / `native_part_spans` byte-atom witnesses from tower 0,
and the staged property rows/spans from tower 1, through `_native(t).lookup_rows`.
It takes every strict span edge from `where_containment_edges`, calls
`Ops.partOf`, checks `_interval` for every defined symbol, and records each
tower-1 whole's Euclidean distance from the all-ones corner. Model tensors
are compared exactly before and after the audit and are unchanged.

The first two attempts exposed certificate setup defects: word-wide identity
spans alone supplied no strict edges; treating an identity's internal key as
observed bytes was invalid. Those attempts remain in the receipt. The
completed certificate uses the stem's actual byte rows and recorded spans,
with the first attempt's captured initialization replayed exactly, not a
new seed or substituted codes. It reports:

- **152/152 failed containment edges:** 120 within tower 0 and 32 from
  tower-0 atoms to tower-1 wholes, over two kept readings.
- **6/6 empty definition intervals.** Symbol 11, for example, has parts
  `[14, 15, 16]` and wholes `[0, 8]`, all with evidence 1. Its lower bound
  exceeds its upper bound at coordinates `[0, 15, 28, 36, 44, 63, 64]`, with
  maximum violation 1.0. This independent interval failure does not depend
  on assigning byte atoms to identity spans.
- A cross-tower counterexample: stream 0, atom row 3 at `[2, 3]` and whole
  row 0 at `[2, 5]`, maximum coordinate violation `0.24595165252685547`.
  Whole-code distances from all ones range from `8.713382720947266` to
  `8.727951049804688`; the receipt includes every whole separately.

The user's instruction was: **“if it fails, stop and report, do not port it.”**
No tower repair, projection, tolerance, weakened assertion or alternate
initialization is applied. `IdeaEvaluation.py` and `begin_presentation`
retain their approved hashes. There is no repair smoke, new freeze, formal
five-group rerun, full sweep, commit or publication. This is development
evidence of a blocking failure, not the requested completed measurement.

**Return through Alec to Claude.** Recommend resolving the concrete empty
interval and cross-tower counterexamples against the FutureWork construction
claim before resuming this round. Keep the certificate and its recorded
initialization unchanged. Then finish finding 1's diagnosis and regression,
implement B and its regression, and perform the requested smoke, one new
freeze and scoped groups. Neither §1 nor §20 is reversed here.

## 22. Respecified percept certificate — Codex, 2026-10-25

Claude resolved §21's stop in FutureWork, **“Measured (Codex, 2026-10-25;
Claude's reading)”**; Alec instructed this repair to resume. The first
certificate and all its counterexamples remain in `repair-01`, unchanged.
Its 120 byte-span edges were not WordIdentity parthood. Its 32 cross-tower
failures and six empty intervals remain evidence of the cross-tower state;
the sorting item owns that requirement, not this round.

`test_percept_transitivity.py` now reads the kept words through WordIdentity
and verifies their actual tower-0 rows exactly. It checks every
`part_atoms(raw)` atom against the word, every recorded nested part against
the word, and every pair of kept words whose atom sets nest, using
`Ops.partOf`. It records each encountered tower-1 whole's Euclidean distance
from all ones without asserting a tower-1 order. It compares model tensors
before and after the audit exactly. No production code is changed for this
certificate.

The original captured initialization is replayed, SHA256
`d12e208a73149b71dda310ddb5b0833c9e3b9872a04d018dbd5b62d5ab6c93df`.
Development passes: 40 atom edges and six word-pair edges, with no failures.
This lesson has no recorded nested parts and its six qualifying word pairs
are reflexive; it supplies no nontrivial nested-part witness. The exact
before/after test and the initial resumed fixture error (a duplicate Python
argument name, fixed without changing the certificate) are retained in the
separate `repair-resumed` receipt. A semantic failure of this respecified
certificate still requires a stop, as instructed.

## 23. Second-stream diagnosis and content readback repair — Codex, 2026-10-25

The extended read-only trace starts from rejected `b21d4068cd`, seed
`4199722147`, the original length-1 snapshot and budget 512. It retains §21's
staging, allocation, `at`, recency, accessibility, program and admission
records and adds each trial's reconstruction terms and detached numerical
readback inputs. Both streams' first sentence has the same six staged word
ids, the same seen rows 11–19, no prepended identity, and all 64 transient
seats free. Each allocates chunk 0 in its own batch row. The §15 staging
copy has no identity to refresh and `accessible` is empty for both rows.
There is no first-stream allocation alias or second-stream gather drop.

The defect is in `LanguageSpace.decoder_eligibility`. `_reverse_one` uses
the operation's `content_operand`, excluding the percept band; eligibility
then compared those children against unmasked parents and symbol banks.
Chunk location and time could therefore make an otherwise exact native
word unreadable. The resulting reconstruction cost incorrectly rewards the
stream-1 conjunction over the equality: free-byte cost is 2 versus 4/3
(normalized values 0.3606737554 versus 0.2404491752). Commit faithfully
retains that already selected non-equality. It is not chooser variance.

Eligibility now applies the same declared content projection to the parent,
candidate/terminal banks and recovered children before all singular/pair
comparisons. No content kernel, numerical tolerance, keep law, seed, budget
or proof assertion is changed. This is the missing reader of the existing
percept-band exclusion rule. The recorded first-sentence comparison then
ties at R = 0.036067377775907516 for both trials in both streams, keeping
both greedy equalities.

The new `test_recorded_two_stream_readings` reuses the original snapshot and
seed through an explicit fixture adapter and runs the unchanged standing
proof. It additionally requires full kept references for both streams'
three premise sentences. Applied to an extracted copy of `b21d4068cd`, it
fails at `(1, 0, [None, None, None])`; after the reader repair it passes all
six premise checks. No draw or choice is forced by this regression. The
fixture retains every surviving parameter and grammar row; B's four new
predictor columns retain their original-seed construction values.

**A separate companion residual remains for review.** Development restores
all premise readings, but the length-1 second document still fails its
companion question in both arms; length 2 answers both documents. Its
question's explored program replaces `surface` (operation 16) with `whole`
(operation 1), producing a different referent. The recorded actual R+A
comparison is greedy R = 0.23423752188682556, explored R =
0.20923753082752228, A = 1 for both. The 0.025 improvement is the existing
coverage charge (1 versus 0.75, weight 0.1); field error and iteration cost
agree. E is recorded and excluded as the decided comparison requires.
`SentenceCredit.comparison` keeps explore at 1.2092375755310059 rather than
1.234237551689148. This is the concrete existing R+A behavior, separate
from the repaired premise-reader defect; it is not an 8/8 restoration or a
claim that the earlier regression was acceptable. The receipt preserves
both programs, all components, kept references and the companion outcomes
for Claude to assess. Changing the objective or forcing the question's
greedy path is not part of this repair.

## 24. Decision B: current-time context and forecast persistence — Codex, 2026-10-25

`ActivationPredictor` has a construction-sized context width **4D + 4**:
mean and latest observed STM contribute 2D, the two observed field lanes
contribute 2D, and the four-coordinate `RowTime(now)` begins at offset 4D.
The derivation is beside the linear layer and the fixed empty-time buffer.
Other shared predictor consumers use that fixed zero input; every
`expect_next_meaning` reads current model time and supplies its encoded four
coordinates. The cached prior can be reused only at the same tick. At a
later tick it retains exactly the same observed sources and re-evaluates
with the new current time. Forecasts do not become their own context.

The estimate envelope is stamped at `tick + 1`, with a loud horizon check.
Both durable estimate writers pass that explicit stamp through
`append_estimate`; a later close does not overwrite it with arrival time.
Previously written entries and time recall are unchanged. The new regression
keeps identical chunks at ticks 7 and 19, compares the actual linear-layer
context inputs (first 4D exact; final four different), and checks stamps 8
and 20. It then writes at tick 31 and verifies source stamp 2, estimate
stamp 20 and observation stamp 31. The chunks remain bit-exact throughout.

Two existing boundary fixtures require explicit ports. Streaming now supplies
the layout step-1 fused idea, like the packed close, and the durable pair is
identified by its recorded link rather than physical row adjacency. The
truth-ingestion test keeps observed context inputs exact but checks a finite
forecast with the new current-time stamp rather than requiring a forecast
to remain unchanged after time advances. The already ported
`test_clock_independence` is untouched; no forecast/answer equality assertion
is restored. Exact prior/current bodies are in the receipt. A no-store
fixture also exposed an unconditional `store.when_start` access in
`_clause_end_state`; the guard now permits its existing no-LTM path.

The original decomposition parameter adapter copies every old context
column exactly into the wider layer and records the four new columns.
This is an explicit measurement port, not a production checkpoint migration.
The source/measurement archive includes the fixture and adapter. Neither
`IdeaEvaluation.py` nor `BasicModel.begin_presentation` is edited. No storage
bound is resized, no new trial mechanism is introduced, and no test is
silently retired. Formal results and the new source freeze follow separately.

The wider predictor also brings its retained-estimate tests into the scoped
check. They exposed unported pre-layout fixtures: three-role direct LTM0
writes and a version-5-to-6 semantic migration. The retention fixture now
supplies one fused point and compares only its occupied row in the returned
chain; source links, estimate/observation separation, capacity atomicity,
non-authority and exact restore checks remain. The migration file now checks
the already decided rejection of version 5, exact round-trip of version 9
with all four native-symbol prior entries, and fingerprint rejection of a
corrupted current estimate. Its two old test names have explicit successors
in the port ledger. The withdrawn behavior is acceptance of a pre-layout
checkpoint, already superseded by layout step 1 and this round's checkpoint
versions; no production migration is reintroduced. All setup failures,
including the intermediate occupied-row shape and constructor-argument
corrections, remain recorded.

## 25. Frozen clock check and next-stamp cursor repair — Codex, 2026-10-25

The first repair freeze, `0b7391c17d38e7e48a4e5e6ecb08aa89088af8db9b1a9363c04155dc5193084a`,
followed a passing native smoke. Its core group exposed a real bookkeeping
failure in the unchanged `test_clock_independence`: the largest written
stamp was 3 while the store's next-stamp cursor remained 3 rather than 4.
The explicit future-stamp writer added for B bypassed the cursor advance
performed by the ordinary arrival-stamp writer. The test is correct and
remains byte-identical; no assertion or invariant is weakened.

`LTM0.append_estimate` now advances only the durable next-stamp cursor to
`max(cursor, decode(written_stamp) + 1)` after writing a timed estimate.
The entry's absolute stamp, the model's presentation clock and
`begin_presentation` are untouched. Re-witnessing a past estimate cannot
move the cursor backwards. This is B's necessary writer bookkeeping, not
an alternative to its future stamp or an advance of presented time.

An isolated copy of the first freeze passes the unchanged clock certificate
at all nine recorded clocks with this correction. The new current-time
regression additionally writes an estimate stamped 20 while the cursor is
19 and requires cursor 21; it still checks later storage at 31, unchanged
source stamp 2 and identical chunk tensors. No different seed is selected.

The failed first frozen run and its smoke remain in `repair-resumed`; the
replacement source and all five groups are recorded separately under
`repair-resumed/final`. A replacement smoke and freeze are necessary because
the production writer changed. This is explicitly an additional failed
freeze followed by a repaired freeze, not a claim that the initial one
passed or that only one freeze occurred. No source is edited while its
measured core run is active. The §39 instruction to fix and continue applies;
neither §1 nor §20 is reversed and no commit is made.

## 26. Repair measurement and return for review — Codex, 2026-10-25

**Ready for Alec to return to Claude; not approved.** The replacement source
SHA256 is `4b7fa2aaf7e5657bd486ced39d632b8a4a2545cca7e1500cced647be2b17c84f`.
The [final repair receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/repair-resumed/final/README.md)
records a passing native smoke, the replacement freeze, all five groups on
that hash, exact ports and the retained failed attempts. It is separate from
rejected `b21d4068cd`, the first percept stop and the failed clock-cursor
freeze `0b7391c17d38`; none is overwritten or counted as a pass.

The frozen groups pass **1,113 cases**, with **seven standing prerequisite
skips and zero test failures**: core 1,097/7/0, packing 4/0/0, XOR 8/0/0,
and each decomposition arm 2/0/0. The smoke is one additional pass. The
original selector inventory is retained; this round adds the percept and
two repair regressions plus five predictor/persistence files affected by B.
The new regressions pass, as does the unchanged clock certificate. The
percept certificate passes 40 atom edges and six reflexive word-pair edges,
with no nested-part or nonreflexive word-pair witnesses in this lesson.
Every tower-1 whole's distance from all ones remains recorded.

**Finding 1's premise loss is repaired; the companion flag remains explicit.**
All **32/32** premise readings across both lengths, streams and arms have
full references on the final source. The new strict two-stream
regression fails on the verified 893-path rejected source and passes here.
The raw companion comparison remains placement **8/8**, rejected source
**4/8**, repair **6/8**. Both residual misses are document 1's length-1
question. In both arms its explored `whole` transformation replaces the
greedy `surface`; the recorded existing R+A comparison keeps the lower
coverage-cost program as detailed in §23. The final trace records the same
costs and keep. No chooser draw, seed, budget, proof assertion or objective
is forced to restore eight companions. A standing existence-certificate
pass is not substituted for a correct companion.

**Recommendation for review:** accept or reject the premise-reader repair
from its failing-before/passing-after regression and complete traces;
assess the remaining question-program miss separately against the decided
R+A objective. Keep 6/8 visible as a review flag. Changing that objective or
forcing its keep would require a separate decision and is not done here.

Eight existing test functions have explicit before/after ports and the two
migration names have named successors; no test is silently retired. Both
original decomposition parameter snapshots and seeds are verified, with
all surviving rows/columns exact. The four new time-input columns retain
original-seed initialization. The source archive contains 897 paths and
21 frozen protocol files. Source/protocol checks hold across every group;
`IdeaEvaluation.py` and `BasicModel.begin_presentation` retain their approved
hashes. No stage, commit, publication or full sweep. Steps 4–5 and the one
end-of-item sweep remain; Alec commits nothing until approval.

## 27. Claude's review of the repair (source `4b7fa2aaf7`) — 2026-10-25: approved with two items before the commit

Reviewed: §§22–26, the [repair-resumed](../benchmarks/2026-10-25-item3-0-stm-chunks/repair-resumed/README.md)
and [final](../benchmarks/2026-10-25-item3-0-stm-chunks/repair-resumed/final/README.md)
receipts, `summary.json`, `verification.json`, `frozen-source.json`,
`companion-records.json`, `diagnostic-traces.json`, the port manifest, and
the new text of `decoder_eligibility`, `ActivationPredictor`,
`append_estimate` and `test_percept_transitivity.py`.

**Verified.** All 897 manifest paths hash to their entries and the canonical
map hashes to `4b7fa2aaf7…`; nothing changed after the replacement freeze.
Core 1,097 passed with the seven known skips, packing 4, XOR 8, both
decomposition arms 2 and 2 on their original seeds, snapshots and budgets;
the standing proof file and `test_clock_independence` are byte-identical to
the rejected source's and pass; `IdeaEvaluation.py` and `begin_presentation`
match their approved hashes; no language named, no seed added; eight ported
functions and two named successors, nothing removed. The failed first repair
freeze (`0b7391c17d`, the cursor) is kept apart, as is `b21d4068cd`.

**Finding 1 is repaired, and the diagnosis is right.** The second stream
lost its readings because `decoder_eligibility` compared children whose
percept band the inverse had removed against parents and banks that still
carried it, so a chunk's own stamp counted as unexplained content and the
non-equality reading cost less. The repair projects parent, banks and
children through the same `content_operand`; the first-sentence trials now
tie in both streams; the two-stream regression fails on the verified
rejected source at `(1, 0, [None, None, None])` and passes here; 32/32
premise readings have full references on the final source. The reader was
the missing consumer of the percept-band exclusion the layout round
decided; no draw is forced.

**The percept certificate** is as resolved in FutureWork: 40 atom edges and
six reflexive word-pair edges pass over `WordIdentity`'s mereology, tower-1
distances reported, no cross-tower assertion; the lesson supplies no nested
part, which the receipt says.

**Decision B** is built: `ActivationPredictor`'s context is `4D + 4` with
`RowTime(now)` at offset `4D`, every `expect_next_meaning` supplies it, the
estimate is stamped `tick + 1`, and the regression shows the same chunks at
two ticks giving two contexts.

**Two items before Alec commits.**

1. *The companion residual (6/8) needs one replay, not a repair.* The
   length-1 question of document 1 is lost in both arms because the
   explored program replaces `surface` with `whole` and the decided R+A
   comparison keeps it: its `free_bytes` is 0.271 against 0.361 and its
   coverage charge 0.75 against 1.0, so it reads as the better
   reconstruction while answering the wrong referent. That is the existing
   objective acting on an untrained chooser, and the certificate is an
   existence proof, so it is not a gate. But placement answered this
   question on the same seed, and the receipt cannot say why: it has no
   term trace for that question on the placement archive. Replay the
   length-1 question on the retained placement archive
   (`step2-baseline.tar.gz`) with seed 4199722147 and record the same
   terms. If the greedy `surface` program's coverage was 1.0 there too, the
   difference is the explored path and the residual is learning-side,
   carried to item 0 with the 6/8 flag. If the greedy program's coverage
   was lower there, the chunk field has made the decoder miss a word it
   found before, which is a defect to fix on a new freeze. No source
   changes for the replay itself.
2. *For Alec: what tick does an estimate occupy?* `append_estimate` now
   advances the next-stamp cursor to `written + 1`, so an estimate stamped
   `tick + 1` makes the next arrival stamp `tick + 2`: the prediction and
   the sentence it predicts no longer share a time, and every estimate
   consumes a tick of the arrival clock. This was done to satisfy
   `test_clock_independence`'s cursor invariant (`model_clock ==
   max(stamp) + 1`), which was written when only arrivals were stamped.
   Recommended: the estimate shares the stamp of the arrival it predicts
   (`tick + 1`, which is the next arrival's tick), the cursor tracks
   arrivals only, and the clock test's cursor invariant is ported to say
   so, with before and after text. The alternative is to keep the reserved
   tick and record that an estimate is an event of its own in the stream's
   time.

**Verdict.** The repair is approved: the regression was a reader defect, it
is fixed with a failing-before, passing-after regression, B is built as
decided, and the certificate is honest about what it witnesses. The commit
of steps 1 to 3 follows item 1's replay and Alec's answer to item 2; if
either changes source, one more freeze and the five groups on it, then the
commit. Step 4 then begins. No publication until the end of 3.0.

*Item 2 decided (Alec, 2026-10-25): an estimate carries no time.* "The
estimate should not be estimating time, only content. We can feel free to
stamp 0 on the estimate to make the idea that is written to conceptual
space carry the time exactly. That will save the time as stamped into LTM,
which is what we want." Confirmed by Claude: the zero `.when` band is the
model's mark for "no time" (inner chain rows, the pre-layout migration), the
`estimate` kind is already excluded by recall by time, forgetting's age and
the document segments, so the arrival's idea is the one entry stamped at its
tick and LTM's saved time is the arrival's. Consequences: `append_estimate`
writes a zero stamp and no longer advances the cursor (the §25 cursor
repair is removed); `test_clock_independence` holds unchanged with no port;
decision B keeps its input (now in the predictor's context) and drops the
`tick + 1` output stamp; the B regression asserts a zero stamp on the
estimate beside the arrival's exact stamp; the layout spec's "an
expectation's estimate has a percept part stamped ahead" (§1 item 10's
decided paragraph, "Alec: That will allow thoughts of the three times") is
superseded in that one respect: the future in STM is the estimate's
content, dated by nothing. The withdrawn alternative, the estimate sharing
the arrival's tick, is recorded above in the example tables.

## 28. Closing replay: the companion residual is learning-side — Codex, 2026-10-25

Following §27 item 1, replayed the length-1 question on the retained
`step2-baseline.tar.gz`, source `1e94238b57`, in both zero and enabled arms.
All 888 archived source paths match before and after each run. Both use
seed **4199722147**, the original parameter snapshot SHA256
`dbd7315f8a49f581f5f2bb0f837e0075ef9dbada63cc94cc9db6c0d1ee367d34`,
the existing grammar-identity parameter port, budget **512**, and unchanged
proof assertions. The placement cache observer is the one accepted in its
receipt. The added observer records programs, decoded values, every cost
term and the R+A comparison without consuming a draw or modifying state.

In both arms, document 1's greedy `surface` trial already has coverage
**1.0** and normalized `free_bytes` **0.3606737554**. Placement's explored
trial also has coverage 1.0 and the same free-byte report; its extra
iteration costs 0.01. Actual R+A is **1.2132812738 greedy versus
1.2232812643 explored**, so greedy is retained and both companions pass.
The approved chunk repair instead explores `whole` at the disputed reduce;
coverage is 0.75, R+A **1.2092375755 versus 1.2342375517 greedy**, so explore
wins. The free-byte report is not trained in this field objective.

This meets §27's learning-side branch: greedy coverage did not regress.
No reader, objective, chooser, seed or keep decision is changed. Carry the
**6/8** companion flag, including both length-1 document-1 questions, to
item 0 with these budgets and traces. The source-level second-stream
reading repair remains approved, with all 32 premise readings complete.
The [closing receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md)
keeps the placement replay apart from the `4b7fa2aaf7` measurement and the
new closing measurement.

## 29. Untimed estimates, current-time inputs retained — Codex, 2026-10-25

Implemented Alec's §27 item-2 decision. `append_estimate` accepts no stamp,
writes zero `.when` (also on a re-witness), and never reserves an arrival
tick. The §25 cursor advance is removed. Both durable callers,
`ClauseRows.write_clause` and `LTM0.append_expectation_pair`, pass no stamp.
The ordinary arrival writer retains its absolute time and advances its
cursor in the existing way. Recall by time and content-kernel exclusions
are unchanged.

`ActivationPredictor` still receives current `RowTime` in the four
construction-sized coordinates at offset `4D`, on every prediction call.
The returned estimate's band is cleared, including when its selected key
has a percept band. There is no `tick + 1` output stamp. The valid input
range now includes `RowTime.HORIZON`, consistent with the ordinary stamp
writer, since no future tick is required. No buffers or capacities change.

The B regression is explicitly renamed from
`test_same_chunks_supply_current_time_and_future_estimate` to
`test_same_chunks_supply_current_time_and_untimed_estimate`: identical
chunks at ticks 7 and 19 still yield distinct current-time context inputs;
both forecasts and both durable estimate writes have zero time, the early
write leaves cursor 19, the source stays at 2, and an arrival at 31 keeps
31 and leaves cursor 32. The truth-ingestion test's forecast-stamp assertion
is ported to the same zero-band decision; its retained source-input
assertions remain exact. Both complete before/after bodies are in the
closing receipt. `test_clock_independence` is byte-identical, with no port.
The layout spec's §1 item-10 decided paragraph now records the dated
supersession of "stamped ahead". These changes reverse no remaining Alec
decision in §1, §20 or §27; §27 supersedes only the estimate's output time.

## 30. Closing measurement and return through Alec — Codex, 2026-10-25

One short native smoke passed, then one new source freeze:
**`cd49a1e84534d6c92f074276e2cfc57fccd1226a876bad4dce5905c6d6b3107d`**,
897 source/config/test/fixture paths and 24 protocol files. The five scoped
groups passed on that hash: core **1,097 passed, seven standing prerequisite
skips**; packing **4 passed**; XOR **8 passed**; decomposition **2 passed
per arm**. Total **1,113 passed, seven skips, zero test failures**, plus the
one smoke. The same selector inventory is retained with B's explicit
renaming. No full sweep was run.

Both regressions pass; `test_clock_independence` is byte-identical and
passes unported. The percept certificate remains byte-identical and passes
40 atom edges and six reflexive word-pair edges, with no nested or
nonreflexive witness in this lesson; tower-1 distances remain diagnostics.
All **32/32 premise readings** are complete. Companions remain **6/8**,
with the same two length-1 questions and R+A decisions. §28's untouched
placement replay assigns that residual to item 0; no reader, objective,
seed, snapshot, budget or proof assertion changed to remove the flag.

The [closing receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md)
keeps `b21d4068cd`, the failed percept certificate, `0b7391c17d`, approved
`4b7fa2aaf7` and this measurement separate. It records the two complete
test ports, all prior evidence hashes, source delta, initialization checks,
companion records and exact question terms. A post-run aggregation assertion
looked in the wrong initialization directory; that reporting failure is
retained and an external post-processing copy corrects only the path. The
frozen protocol and all measurements remain unchanged; no gate was rerun.

Return through Alec for Claude's closing review. On approval, Alec commits
steps 1–3 as one commit naming frozen sources **`34848e154f`** (layout),
**`1e94238b57`** (placement) and **`cd49a1e845`** (STM chunks, repaired and
closed). No commit or publication by Codex. Steps 4–5 and the item's one
full sweep remain after this round.

## 31. Claude's closing review of step 3 (source `cd49a1e845`) — 2026-10-25: approved

Reviewed: §§28–30, the [closing receipt](../benchmarks/2026-10-25-item3-0-stm-chunks/closeout/README.md),
`summary.json`, `verification.json`, `frozen-source.json`, the placement
replay traces (`placement-replay/zero/question-trace.jsonl`,
`question-comparison.json`), `ports/manifest.json`, and the new text of
`append_estimate`, the estimate path in `expect_next_meaning` and
`ActivationPredictor`.

**Verified.** All 897 manifest paths hash to their entries and the canonical
map hashes to `cd49a1e845…`; nothing changed after the freeze. Core 1,097
passed with the seven known skips, packing 4, XOR 8, both decomposition arms
2 and 2 on their original seeds and snapshots; `test_clock_independence`
byte-identical (`4e4aad65…`) and passing unported; `IdeaEvaluation.py` and
`begin_presentation` at their approved hashes; the four-path delta is two
production files and two tests; two ports with before and after text, two
named successors, nothing removed; no language named, no seed added.

**Item 1 is settled on the learning-side branch.** On the placement archive
with the same seed, snapshot and budget, document 1's greedy `surface` trial
already had coverage charge 1.0 and `free_bytes` 0.3607, and its explored
trial had the same coverage and free bytes with one more iteration, so
greedy was kept (R+A 1.2133 against 1.2233). On the chunk source the
explored trial is `whole`, coverage 0.75, and the decided R+A rule keeps it.
The decoder did not lose a word it found before; the explored path changed
under the shared field, and the objective prefers it. The 6/8 companion
result is a learning-side residual, carried to item 0 in todo with its
seeds and budgets, with placement's 8/8, the rejected 4/8 and the repair's
6/8 kept apart in `companion-records.json`.

**Item 2 is built as decided.** `append_estimate` writes `when=False` and
re-witnessing writes the zero stamp again; the §25 cursor advance is gone;
`expect_next_meaning` zeroes the returned estimate's band
(`roles[0, when_start:when_start+4] = 0`) and supplies `RowTime(now)` as
four coordinates to `ActivationPredictor`, whose context stays `4D + 4`;
the B regression, renamed to `..._untimed_estimate`, holds the same chunks
at ticks 7 and 19, distinct contexts, zero estimate stamps, cursor 19
after the early write, arrival 31 with cursor 32. The layout spec carries
the dated supersession under §1 item 10; the original "stamped ahead"
sentences in §2 stand as history beneath it.

**Verdict.** Step 3 is approved: the eight shared chunks, prepending, binary
bind, the retired overlays, the transient pool, the recurrence bound, the
compound-case repair, the second-stream reader repair, untimed estimates
with the current time as the predictor's input, and the percept certificate
over the identity's mereology, on one verified frozen hash with every gate
green and the residual named. Alec commits steps 1 to 3 as one commit
naming `34848e154f` (layout), `1e94238b57` (placement) and `cd49a1e845`
(STM chunks). The live checkout is one commit behind origin
(`aefbc1f06d`, the image-modality FutureWork entry): bring it in before
committing and keep both appends to FutureWork at the merge. Step 4,
retire mode, begins on the committed source with its spec section written
first. No publication until the end of 3.0.
