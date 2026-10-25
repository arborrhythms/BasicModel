# Line 6, context wholes: the document and the stream as individuals in the field (Claude, 2026-10-24; for Codex)

**Status.** Iteration 2 of 3.5's remaining work is lines 6 and 7. This spec
covers the part of line 6 that had none: documents and streams as context
wholes. The rest of line 6 has its specs already and §4 only points at
them: the surface operators (occurrence spec §13.1), tense and aspect on
the verb's phases (§§4–5), nested admission (§12) and their gate (§13.2);
line 7 is parts-of-speech §3. Under step-2 §39 Codex reviews this spec
against the code, fixes what it contradicts and continues, documenting
each decision; only reversing a decision in §1 comes back to Alec. The
gate rule is the iteration spec's §6 with §19's addition: a lesson smoke
before each freeze.

## 1. Decided

1. **The document and the stream are context wholes**: individuals minted
   at the document's first sentence and bound by continuity, whose
   activation is in the field while their sentences are read, so that
   attention's region returns them with the other wholes and the
   sentence's concept carries their trace through the fold, to the degree
   the fold keeps it; "what was said in *d*" and re-witnessing a document
   are content matches (two truths §1, 2026-10-20; occurrence spec §0
   head, §3.4, §12).
2. **Situation wholes nest** — sentence, the group read in parallel,
   document, conversation — and are identities, never clock values;
   indexicals take their referent from them; they seed the domain of
   discourse and are the widest bracket of one attention (occurrence
   §3.4). Every text configuration interleaves (§3.5).
3. **Order is the stream's history; recency is store order**; a
   re-witnessed row keeps its place; when context wholes replace the stream
   column, the chain is the stream whole's rows in `.when` order, the same
   order (Alec, 2026-10-17 and 2026-10-22).
4. **Context wholes do not carry word adjacency** (Alec, 2026-10-22).
5. **Every mental event has a perceptual and a conceptual part; a whole's
   percept part is the centroid of its parts' forms** (Alec, 2026-10-23).
6. **MTM holds identities and timelines; conceptual space has access to
   MTM symbols; prepending them to the symbol array makes them available to
   all orders; the overlay is transitional** (Alec, 2026-10-23).
7. **Today's storage stays through 3.5.** The four address columns and the
   `stream` column are retired as columns in item 3's layout round (two
   truths §1). This line replaces their *readers*; the columns are still
   written and go in item 3.
8. **A context whole is a WholeSpace type, not an MTM identity** (Alec,
   2026-10-24: "wholes and parts are in different spaces, in general").
   Parts and their identities live in PartSpace and MTM; a whole is a type
   over a run of the field, in WholeSpace, as a word is over its letters.
9. **No exact readers** (Alec, 2026-10-24: "I don't see a reason to have
   exact readers"). Membership of a row in a document or stream is read
   from its content, by the whole's trace, and the readers that used the
   `stream` column and 2c's document sidecar switch to that read in this
   line; the `stream` column and the sidecar go now, since nothing reads
   them. The four address columns stay until item 3, because the writer
   still addresses a re-witnessed sentence by them.
10. No language in code: no indexical or marker is named.
11. **Wholes come from WholeSpace, parts from PartSpace; the centroid is
    the center of that space** (Alec, 2026-10-24): the average over a
    position's parts and wholes, compensating for their numbers, parts
    over their count and wholes over theirs. Run rows for the sentence,
    the document and the stream are not wholes: "we don't need indices
    for sentence, document and stream." §2a's run inventory and its blend
    into word reads are withdrawn.
12. **Batch indexing and `.when` do the segmenting** (Alec, 2026-10-24):
    LTM and the STM structures are indexed by batch row, and rows carry
    their `.when`; the batch index segments the store by stream, and the
    per-batch structures (the batch's LTM view, MTM's per-stream pools,
    STM per row) stand as they were.
13. **A document starts with a cover in its LTM stream** (Alec,
    2026-10-24): "LTM streams are prefixed by a 'cover' to indicate the
    start of a document, which can be conceptual or perceptual, but
    perceptual LTM requires episodic memory that we don't have yet, so
    let's make that future work and just prepend a per-document concept
    that we create automatically." A perceptual cover is FutureWork.

## 2. The first mechanism — withdrawn 2026-10-24

The first draft made a context whole an MTM identity keyed by a context
address, bound each sentence row to it, prepended the two wholes to the
symbol array, and replaced the stream-column readers by token membership.
Alec: "wholes and parts are in different spaces, in general." Withdrawn;
the record is in the git history of this file. Nothing of it is built.

## 2a. The second mechanism — withdrawn 2026-10-24

*Alec, on §2a and its build: "This definition of wholes is completely
wrong. Wholes come from whole space, parts come from part space." No run
rows for the sentence, the document or the stream, and no blend of run
forms into word reads. Kept as the record; nothing of it is built. §2b
replaces it.*

## 2b. The mechanism, in code names: batch indexing, `.when`, and a cover row

**Batch indexing.** LTM0 keeps the batch index of the row that wrote each
row: the `stream` column returns, as what it always was, exact per-batch
ownership (`witness_stream` at `append_meaning`), and its readers return
to it unchanged: `get_stm_chain` and `stream_rows` (the batch's rows in
store order), `Forgetting`'s age by position within the batch's rows,
`MereologicalCodes.begin_forward`'s stream count, and `ReferenceClose`'s
two antecedent guards. STM is per batch row by construction. `.when` is
the row's model-time stamp (`RowTime`), already a column.

**The cover.** At each document's start the presentation cursor, which
knows the boundary (`source_at`, `_thought_document_ends`), mints a
per-document concept automatically (`owner.new_concept()` and the
allocator's ordinary seat, as `MemoryConsolidation` mints a promoted
identity; order 1, an individual; its code initialized as a zeroth-order
concept is, at random, and placed by the priming surface later) and
writes a **cover row** into that batch's LTM stream: an LTM0 row of kind
`cover` whose one slot holds the concept's code, stamped with the current
`.when`, before the document's first sentence. The cover is conceptual;
a perceptual cover, the title page as a percept in LTM, waits for
episodic memory ([FutureWork](../FutureWork.md#perceptual-cover-episodic-ltm-alec-2026-10-24)).

**Document membership is a segment.** A row belongs to the document whose
cover is the latest cover row before it in its batch's stream, by `.when`
order; the document ends at the next cover or the stream's end. No content
match and no membership column: segmentation by batch index and `.when`,
as Alec says. The readers that 2c's provenance sidecar served read the
segment: `MemoryConsolidation.consolidate` counts a reading's distinct
documents as distinct covers among its rows' segments and a token's
distinct sentences as distinct rows; `MediumTermMemory.close` and
`end_document` retire, at a document's boundary, the entries whose
introducing row lies in the ended segment; "what was said in *d*" is a
bounded read of *d*'s segment under the work meter, reached by the cover
concept as a reference, in the shape of `TimelineRead.read`. The
`_consolidation_sources` sidecar and `document_digest`'s provenance use
stay removed; the four address columns stay for addressing and
re-witnessing until item 3.

**What the cover is for composition.** It is a row in the batch's recent
frames (`situation_references`), so the resolver can offer it as a
referent like any recent row and an indexical can bind to it by exposure;
nothing names it. Nothing is folded into word reads for it: no blend.
Re-reading a document mints a new cover; identifying it with an earlier
reading is the perceptual cover's job, in FutureWork.

**The centroid, corrected (Alec, 2026-10-24).** "This centroid applies
to all concepts order zero. Higher-order concepts don't have parts and
wholes, so it's just a pure centroid calculation (although with positive
and negative evidence and weighted over those)." Two rules, then:

- *Order 0.* A form composed from parts is the center of the space of its
  parts and its wholes: the parts (PartSpace) averaged over their count and
  the wholes (the WholeSpace rows active over the span) averaged over
  theirs, the two averages then centered. Codex applies this to the swap's
  form centroid (`MeaningKernels.centroid`, today parts only) at
  composition, and records where the active WholeSpace rows for a span
  are read; with no active whole it is today's leaf-weighted centroid.
- *Order 1 and above.* A concept's content is the pure centroid of its
  members' content, weighted by their evidence in the two lanes, positive
  and negative, never a max-union or a min-intersection. This replaces the
  coordinatewise maximum in `MemoryConsolidation._fold` for promoted
  identities and readings, and governs Orders §9's centroid caches; the
  identity-basis question in FutureWork is answered by it for content.
  Codex lists the 2c promotion assertions this changes as ports.

**Verified against the code (Claude, 2026-10-24, at Alec's request).**
Centroids are for concepts, stored on the symbol associated with the
concept: the shared index keeps one cached row per symbol/concept pair,
`[form | meaning]` in `similarity_codebook.W`, and a concept's centroid is
written there (promotion copies its fold into `W[row]`; a composed idea's
form rides in the node until its row is written; `centroid_forms` caches
by row). Percepts are stored on the two towers and have a different
mereological definition: `MereologicalCodes._interval` reads a symbol's
parts from tower 0 (PartSpace) and its wholes from tower 1 (WholeSpace)
and returns bounds, not means, the lower bound the maximum over the parts'
codes and the upper bound the minimum over the wholes' codes scaled by
their degree, and `derive` gives a symbol its lower bound; the exact word
forms and atoms live in PartSpace's percept store (`WordIdentity`). So the
order-0 centroid rule above is the rule for a concept's form, the
composed node's and the minted concept's, and leaves the towers' interval
mereology as it is. Concepts in conceptual space are arranged by context:
today by `occurrence_terms`, the mean of a concept's occurrence contexts
written into its meaning block by `derive`; by the registered priming
surface once item 3 lands the placement decision, with zeroth-order
concepts starting random and moved by the surface.

**Clarification (Alec, 2026-10-24, understanding only, no change now):**
"centroids: that applies to percepts and symbols, not concepts." The
centroid is a form-side notion, computed over symbols' forms and stored
on the symbol associated with a concept: at order 0 over the symbol's
parts and wholes, at higher orders over the member symbols' forms weighted
by their evidence. A concept's meaning is not a centroid of its members'
meanings; it is arranged by context, today the occurrence-context mean,
in item 3 the priming surface. The paragraph above says "content" where
it should say the symbol's form, and `MemoryConsolidation._fold` as built
applies the evidence-weighted mean to the meaning coordinates; Alec asked
that nothing change now, so this is recorded as a discrepancy for item 3's
layout round, when the promoted row's two blocks are defined.

**The cover concept (Alec, 2026-10-24):** an order-1 individual with a
random code, minted at the document's start as a promoted identity is,
placed by the priming surface later, written as the cover row's one slot,
and a referent in the batch's recent frames.


**What a context whole is.** A type row in WholeSpace (`Spaces.WholeSpace`,
the learned property table evaluated at every live position) that is
constant across a run of the field and changes at the run's boundary, so
that the run is a maximal constant-type run in the sense already used for
segmentation (`stage_analysis_spans`; `RunStructureLayer` for contiguity
and containment). Three levels this line, nested by their runs:

| whole | its run | begins | ends |
| --- | --- | --- | --- |
| sentence | the words of one clause | the sentence's first word | its close |
| document | the sentences of one document | the document's first sentence | the document boundary (`_thought_document_ends`) |
| stream | one subject's whole history | the stream's first sentence | never, while the stream lives |

The parallel group of `interleave:N` is the presented unity that
WholeSpace's analysis divides; it gets no row of its own in this line.

**Minting and content.** A whole's row is minted when its run begins and
its code is the leaf-weighted centroid of its parts' forms
(`MeaningKernels.centroid`; for a word, `MereologicalCodes.derive` already
takes the form from its parts), updated as parts arrive: a document's form
is the centroid of its sentences' forms so far, the stream's of its
documents'. Its `.where` is the run's bracket, its `.when` the entry
stamp. Its concept, the fold over the meanings of its parts, is minted on
recurrence as any symbol's is and forgotten by value; nothing is placed on
the priming surface before then. The same run machinery that makes a word
whole from its letters (item 6.8) makes these, one level up each.

**Presence in the field.** The whole's activation holds at every position
of its run, so each word's conceptual read in the parallel prelude
(`_per_word_prelude`, `ConceptualSpace.forward(subspace, word_subspace)`
reading `wholeSpace.subspace`) sees the sentence, document and stream
wholes with the other wholes, and the sentence's composed concept carries
their trace to the degree the fold keeps it (§3.4: the parallel read is
where concepts see every part and whole). Attention's mask over `.where`
includes the run: the widest bracket. The exact path from the WholeSpace
row to the word's code is Codex's to confirm against the code and to
state in its review; the requirement is that the trace is in the written
row's content, so that membership can be read from it.

**Membership by content.** A row belongs to a whole when its content
carries the whole's trace: the whole's code covers the row's content
above a threshold, the same meet that binding uses (`IdentityRows.
meet_ratios`), with the threshold a model.xml knob, `contextMembership`.
Every former reader of the `stream` column and of 2c's sidecar reads
this instead: `LTM0.get_stm_chain` (the stream's rows by membership, in
store order), `Occurrence.stream_rows` and `recency_weights`,
`Forgetting.forget`'s stream read, `ReferenceClose`'s two stream guards;
`MemoryConsolidation.consolidate` counts a reading's distinct document
wholes and a token's distinct sentence wholes by membership;
`MediumTermMemory.close` and `end_document` take the document whole in
place of a digest and retire the entries whose introducing row belongs to
it. `witness_stream`, the `stream` column, `_consolidation_sources`
(`ClauseRow.write_clause`, the `Spaces` extras) and `document_digest`'s
provenance use go. Exactness is not promised: a row whose trace fell
below the threshold is not the stream's for these readers, which Alec
accepts.

**Re-witnessing and queries.** Reading a document again re-forms the same
type, since the centroid of the same sentences recurs, and its promoted
concept recognizes it by content, as a word is recognized; "what was said
in *d*" is a bounded read of the rows that carry *d*'s trace under the
work meter, in the shape of `TimelineRead.read`. Indexicals take a whole as
referent not by `bind`, which resolves among things, but as a word takes
an object: by exposure the word's interpretation names the whole's symbol
(`Interpret.define`), and the whole's recent rows are the frames the
resolver already reads (`situation_references`), which is "they seed the
domain of discourse." The current sentence whole is `position`'s default
object for tense, §4's "now."

## 3. Not in this line

Word adjacency (dropped); MTM entries, prepended symbols, run rows and
blends for wholes (withdrawn); a perceptual cover (FutureWork); retiring
the address columns, LTM0 one slot and LTM1 chains (item 3); prepending
the identities and retiring the overlay (item 3); the §72 sentence-level
stamp and recall by `.when` (item 3).

## 4. The rest of line 6, and line 7, by pointer

- **Surface** (occurrence §13.1, §12): `surface` is absorb on either side
  of the head plus `transpose` on the two newest constituents; the unary
  face supplies the absent marker (`SurfaceLayer`, `SurfaceUnaryLayer`,
  `MarkerUnaryLayer`, `TransposeLayer` exist). The remaining English
  analyzer and its table of forms go; the §8 test 10 reverse is exact.
- **Tense and aspect** (§§4–5): `position` selects among the verb phrase's
  phases by its modifier; the phases now exist, since the verb line gives
  the identity's state before (`Event.before`), the change (the diagonal)
  and after (`Event.after`); R is the current sentence whole or a
  referenced event, never a time read from a row.
- **Nested admission** (§12): one marker pattern for one rule at a time
  (`WordIdentity.observe_nested`, `_marker_pattern`, lesson-scoped, off by
  default); the type criterion; irregulars whole.
- **Gate** (§13.2): `test/constituent_corpus.py`'s variants, named by what
  the generator did, one configuration, nothing named; the phase questions
  of §4; fixed budgets and seeds; failures reported to item 0.
- **Line 7** (parts-of-speech §3): the distributional categorizer over
  direct operands; output typing deferred (§4 there).

## 5. Certificates for the cover and batch indexing

1. A batch row reading two documents writes a cover row at each document's
   start, of kind `cover`, holding a freshly minted concept, stamped before
   the document's first sentence; the two covers' concepts are distinct.
2. Batch indexing: `get_stm_chain`, the stream guards and forgetting's age
   give exactly the results they gave before the §2a build, on a fixture
   with two batch rows.
3. Segments: every row's document is the latest cover before it in its
   batch's stream by `.when`; a row written at the boundary belongs to the
   new document; 2c's recurrence and promotion certificates pass with the
   sidecar gone, counting distinct covers.
4. Document end: `end_document` retires the same MTM entries by segment
   that it retired by digest on the 2c fixture; the next document's cover
   starts a new segment.
5. The cover is a referent: a fixed-operation fixture binds a leaf to the
   cover row through the ordinary menu, no word named; the cover concept
   is not on the priming surface until placed.
6. The centroid: a composed form equals the centered average of its parts'
   average and its active wholes' average on a fixture with known parts
   and wholes; with no active whole it equals today's leaf-weighted
   centroid.
7. Unchanged: G1–G3, both decomposition arms, XOR and the pending-premise
   certificates; the bind menu gains the cover row only as any recent row
   is gained, so no twelve-draw series.

## 6. Code review and implementation adaptations (Codex, 2026-10-24)

Claude's specification and accompanying identity/todo edits are committed
unchanged at `0f90e9b2d3`. This review is against that source, whose code
is still the reviewed iteration-1 source. Iteration spec §20 reviews the
existing specifications for the rest of lines 6–7; §21 records release of
this part. The following are implementation adaptations under step-2 §39,
not reversals of §1.

1. **Membership is not identity equivalence.** `MediumTermMemory.bind`
   currently detaches an occurrence from its earlier token, and `names`
   maps a mention to exactly one identity. Calling it twice unchanged
   would move the sentence from its document to its stream. Context
   membership must append to both wholes while preserving any ordinary
   subject/name binding. Give context tokens a declared context role and
   make their membership path non-exclusive; ordinary mention bindings
   keep their existing exclusive rule. Neither context is a verb's
   subject merely because it contains the sentence.
2. **Retained membership survives token expiry.** The current recurrence
   implementation rebuilds from `_consolidation_sources`, while ordinary
   MTM expiry deletes all the token's membership. A one- or two-sentence
   document is not promoted before its end; losing its memberships then
   would make three-document recurrence impossible. Retain the document
   and stream bindings in each occurrence's existing semantic record,
   including its position in the document whole. These are whole
   memberships, not a second source-coordinate sidecar or new LTM rows.
   The live token's occurrences provide the working view; retained
   bindings provide the historical view after expiry and at restore.
   Every constituent of one close shares its source sentence's place,
   rather than its encoded constituent ordinal. This preserves 2c's
   distinct-sentence test. Address columns remain available as durable
   row handles and continue to be written; they no longer supply the
   replaced provenance/stream decisions.
3. **Migration and authentication.** Existing checkpoints and standalone
   LTM test fixtures have no context binding. Import their old stream
   membership and consolidation provenance once at the restore/write
   boundary, then read the new memberships. Authenticate those bindings
   with the existing semantic fingerprint. Do not silently ignore a
   malformed new membership. Pool restore must allow a context key without
   a mention row and overlapping context members, while retaining the
   ordinary-token uniqueness checks. Context promotion associations must
   survive document-token expiry so a re-read can take the existing native
   path. No column is removed in this line.
4. **A digest needs a tensor carrier.** `document_digest` returns 32 bytes;
   `Token.restore` and the compiled origin arrays currently carry int64
   occurrence keys. Use a domain-separated context address and a checked
   int64 transient carrier, retaining the full digest in owner metadata.
   A context carrier is not a concept seat or a word. Resolve it through
   its staged context mapping; do not send it to a codebook row lookup or
   use it as an output word. A promoted symbol follows the native path.
5. **Stage before the compiled prelude.** Despite its name,
   `_per_word_prelude` runs inside the captured forward on the production
   path. It cannot open tokens or inspect the pool. Stage the two values,
   identities, brackets, stamps and counts at the eager sentence boundary;
   prepend tensors in the field/compose readers. A preview stages new
   context values, and a kept close commits them once. Keep percept word
   targets and their source offsets separate from these two non-word
   symbols: adding them as output words would corrupt reconstruction.
   Ordinary identities keep their existing overlay; contexts enter the
   field first and the existing bind menu from that field.
6. **Store order takes precedence over the explanatory clock equivalence.**
   Multiple closes can share one model tick, as §19 of the iteration spec
   records. Therefore `.when` ordering is not an equivalent implementation
   of §1.3. Read retained members in store order; re-witnessing adds no
   new position. `get_stm_chain` already delegates to `stream_rows`.
   `Forgetting.history_age` additionally enumerates `store.stream.shape[1]`,
   so replacing only `forget` would miss a stream-column consumer.
   Both `ReferenceClose` guards need a membership-presence test as well
   as the selected stream's membership, preserving shared/unknown rows.
7. **Lifetime is still bounded.** The description of an ordinary context
   token as subject to capacity and the later phrase “never retired while
   the stream is live” cannot mean a capacity exemption. Keep the stable
   stream identity and refresh it at each close, with ordinary capacity
   eviction. If its working entry is lost, its retained membership lets
   the next access rebuild that same identity; no new stream is invented.
   A document boundary removes its working document value and associated
   ordinary tokens but does not end the stream. Promotion folds actual
   occurrences only, as iteration 1 requires.
8. **Queries need a context-specific dispatch, not event inference.**
   `TimelineRead` matches subject and verb histories. A context membership
   read instead enumerates that whole's retained occurrences under the
   same work meter, excludes the query's own row, and reports truncation.
   It must not interpret the context key as a sentence address or infer
   membership from vector similarity. The promoted content match remains
   the native query path.

The rest-of-line pointers contain two stale code facts: the named analyzer
and layers are already gone, and event snapshots plus a diagonal are not
yet a selectable VP-phase representation. The paired compound currently
bypasses `CaseSelection`; iteration spec §20 gives the required bridge,
including the two evidence lanes and the separation from modal `adverb`.
The older distinct-type statistic was explicitly superseded by lesson
scoping in the 5.5 plan; this line retains that decision and its assertions.

Implementation follows with development logs, a short native lesson smoke
before the single line-6 freeze, then §5's certificates and sized gates.
The twelve-draw series runs once on that freeze because context locations
change the bind menu. Line 7 gets its own smoke, freeze and scoped receipt.
No gate result or test assertion is changed by this review. Claude reviews
the whole iteration through Alec before publication.

*Note (Claude, 2026-10-24): §6 reviews the withdrawn §2. Its adaptations
that concern MTM entries, context addresses, menu locations and pool
expiry lapse with §2; those on store-order recency, bounded context
queries, checkpoint migration and eager staging before the compiled
prelude carry to §2a where they apply. Claude's rewrite briefly dropped
this section by mistake; it is restored verbatim.*

## 7. Review of §2a and disposition of development work (Codex, 2026-10-24)

Claude's replacement and the restored §6, including its amendment, are
committed unchanged at `23eed97d20`, with the identity and todo edits.
This review supersedes §6's implementation instructions. There is no
context address, MTM context entry, retained exact membership, new bind
location, or twelve-draw series in this implementation.

The withdrawn development source and every run are retained under
`doc/benchmarks/2026-10-24-item3-5-iteration2/development/` as
`context-mtm-development.patch`, full tracked/new-file snapshots, logs and
`context-development-status.json`. None of its membership, expiry, menu,
checkpoint or field-prefix implementation remains in the active source.
Its arithmetic for weighted forms can inform the new run owner; it is not
evidence for §2a. Independent surface-inverse tests and neutral wording
edits remain development changes. No line has frozen.

1. **The native read is not the path named in the prose.** The current
   aligned sentence driver perceives each word once before either trial.
   `stage_cs_sub` reads `WholeSpace.compute_word_property_event`, then
   `ConceptualSpace.reduce_aligned_percept_peers`. WholeSpace currently
   affects evidence for the word's indexed dictionary decode, not an
   additional conceptual value. `stage_cs_sym` decodes the indexed word
   again. `_per_word_prelude` itself runs inside the captured body and
   `_sentence_prelude` is a no-op. Therefore putting rows in WholeSpace,
   or publishing its subspace alone, would not put their trace in LTM.
   Stage run boundaries eagerly, materialize their active values in the
   tensor read, and preserve that contextual read through symbol
   promotion and the captured reconstruction source. Test the written
   point, not just presence metadata. Raw percept targets remain words.
2. **Run rows need their own row family in WholeSpace.** The primitive
   property basis in BasicModel has nine rows, all already occupied, and
   is explicitly frozen. Its byte predicates cannot represent a document
   run. Put the run inventory under WholeSpace, with native form centroids
   and brackets, rather than borrowing character-property rows or MTM
   entries. This does not resize a fixture or reserve conceptual seats.
   Concept promotion still uses the existing conceptual allocator and
   ordinary capacity checks. Before promotion no priming symbol exists.
3. **A trace is content, not a provenance encoding.** No digest, row ID,
   batch index or address is converted into a context vector. The run
   form is computed from its actual parts. Repeated equal content can be
   indistinguishable, and membership can be lost as a run's centroid
   changes. Distinct-document/sentence counts count recognized whole
   types, not recovered source identifiers. Fixtures formerly separating
   equal vectors solely by a source key must supply distinguishable
   whole content; those setup ports will be listed individually.
4. **Membership and migration.** Use `IdentityRows.meet_ratios` on the
   common content carrier and the request-coverage ratio (the whole
   covers the row), strictly above `contextMembership`. Start the knob
   at 0.5; tests establish the boundary explicitly, not by tuning a seed.
   An absent or empty trace supplies no membership evidence. Old stream
   bits and document sidecars are discarded on load; they cannot recreate
   content that an old forward never wrote. Do not rewrite historical
   rows to manufacture a migration witness. Four address columns remain
   solely for addressing and re-witnessing.
5. **The reader inventory has two additional consumers.** Besides §2a's
   named sites, `MereologicalCodes.begin_forward` enumerates
   `store.stream.shape[1]`, and `Forgetting.history_age` does the same.
   Both must enumerate the active stream-whole reads instead. Retained
   row exports and `SituationFrame` must also cease exporting the column.
   The two ReferenceClose guards must distinguish unknown membership
   from a positive match to another stream. All ordering stays store
   order; a shared model tick is not an ordering key.
6. **Boundaries do not supply membership.** The presentation cursor still
   says when a document run ends, and the writer still obtains source
   addresses. Neither supplies a row-to-whole lookup. MTM expiry reads
   the introducing occurrence's content against the ending whole; the
   sixteen-close clock remains a close count. Whole queries are bounded
   content scans and never traverse a retained member list. The bind
   menu remains the resolver's existing frames and candidates.

The remainder of line 6 follows iteration spec §20. Its phase gap is real:
event snapshots and verb diagonals are not yet a selectable VP phase
representation, and the paired compound path bypasses `CaseSelection`.
The existing lesson-scoped nested-admission decision remains in force.
Development logs precede a native lesson smoke, one line-6 freeze and its
scoped gates; line 7 then has its own smoke and freeze. Review of both
lines goes through Alec before publication.

## 8. Development decisions and retained failures (Codex, 2026-10-24)

`WholeSpace.context_runs` owns a bounded row family (`WholeRuns.Whole`):
level, native form centroid, leaf count, bracket, entry stamp, whether the
run is open, recurrence count and an optional promoted symbol. There is
no list of occurrences or source keys. Its capacity follows the existing
LTM capacity; the three simultaneously active levels are the minimum
inventory. This does not enlarge either conceptual or primitive-property
seats. Closed inactive types are retired by recurrence value, then age.

The eager sentence boundary stages only the currently presented clause.
The native PS/WS read and symbolic promotion run once as before. CS then
reads the three active WholeSpace forms as additional native form sources,
with an equal-weight centroid, before either grammar trial. The form of a
word interpreted as an object receives the same read. Neither operation
changes meaning lanes or the bind menu. The captured leaf and reconstruction
source receive that same form read; raw percept targets stay words. The
ordinary composition can still lose a trace. Nothing restores it at close
or writes an exact membership alongside it. The native certificate checks
different document forms, different captured leaves and the written point.
This read makes no claim that a learned abstraction retains context.

Membership compares the common native **form** coordinates. WholeSpace's
unpromoted type has a form and no conceptual meaning yet; comparing a
padded zero meaning against all of a sentence's meaning would ask for
coverage the whole never supplied. This clarifies §7.4's “common content
carrier.” The meet and threshold are unchanged. Promotion uses the fold
of recognized actual rows' meanings, adding the whole's own form to its
conceptual cache. Equal re-formed types count once even during the new
run, before its end coalesces the retained type. A word can name the
promoted symbol through `Interpret.define`; `WholeRead` scans membership
under the existing shared work meter. Unpromoted rows add no priming symbol.

The old MTM document-digest cache is removed, including checkpoint export.
Schema 6 rejects that retired field; schemas 1–5 discard it at migration.
Expiry reads the introducing row at the boundary. `GrammarCheckpoint`
also drops old stream tensors before strict model loading. The address
writer and its self-reference exclusion retain their address checks.

Development logs 08–17 are retained beside the earlier withdrawn logs.
The first new run tests used a non-weak-referenceable fixture owner; its
setup was corrected without changing assertions. A native test initially
looked for programs on `SentenceUnderstanding`; it now captures the real
sentence-observation entries. The first lesson invocation skipped its
slow marker, and the next used an unsupported option; neither is evidence
of a pass. With `RUN_SLOW=1`, the native lesson passed (log 12). Log 14
has 28 passes for the context and consolidation files. These are development
results, not frozen receipts.

The first diagnosis of the wider run treated the printed promotion tensor
as empty. `_fold` now declines an empty conceptual fold before allocation,
since recurrence supplies frequency rather than missing meaning. That was
not the full diagnosis: logs 18–19 and the capture described in §9 show a
nonempty fold and excessive content matches. The failed path is the small
`XOR_grammar` fixture, not the eight-seat MM certificate. Log 16 remains
retained; its truncated tensor was not sufficient evidence for the earlier
empty-meaning claim.

Test ports currently applied under §39 (none of these retire a test):

- `test_memory_consolidation.py`: the distinct-document, three-sentence
  promotion and document-expiry fixtures now supply WholeSpace rows and
  actual native context content. Every existing assertion remains verbatim.
- `test_forgetting.py`: the age fixture supplies distinguishable native
  contents and a third whole covering only the first row instead of
  calling `witness_stream`. The common full-store fixture supplies its
  stream whole; the restore fixture restores the separate whole inventory.
  Every existing assertion remains verbatim.
- `test_reference_arrival_review.py`: the scoped-name fixture gives the
  otherwise empty name a native form trace, as production perception does.
  Its two stream-column assertions become active-whole/membership checks;
  the unscoped-row assertion becomes absence of a matching stream whole.
  The new-stream index comes from the active whole runs. Antecedent,
  arrival, row and bank assertions remain verbatim. The scoped test's
  identifier says `whole` in place of `column`.
- `test_operators_round4a0.py`: the legacy stream-bit restoration assertion
  becomes absence of the retired column and state key. Address/reference
  migration assertions remain verbatim. The exhaustive when-reader
  inventory also needs the new WholeRuns readers and the already-landed
  modal teacher's correspondence read; its set-equality assertion stays.

The full changed-assertion/fixture inventory will be generated from the
source diff at the freeze. No new failure has been removed by assertion
loosening. Line 6 remains unfrozen while consumer and native-path checks
and the remaining phase work continue.

## 9. Content-read diagnostics and lifetime repairs (Codex, 2026-10-24)

Alec noted that LTM can retain context but does not itself thread a context
over documents and batches. The implementation keeps that distinction:
WholeSpace maintains the active run and stages its form at each sentence;
LTM retains only the result of the ordinary composition. A later reader
cannot recover a trace that that composition has lost. No context column,
source-derived vector or exact membership is added as a recovery path.

**Initial threshold.** `contextMembership` now defaults to **0.75** rather
than §7's provisional 0.5. This is a development choice under §39, not an
Alec decision being reversed. In log 19's retained
`consolidation-failure.pt`, a row's coverage by its own document is 0.88889;
other documents cover 0.45478–0.67442. At 0.5 the latter can count as three
document types and prematurely promote a word reading. The captured plan
has three nonzero meaning coordinates, total mass 0.31307, so rejecting
empty folds does not fix that failure. The stricter default requires three
quarters of the native form to be covered. Log 21 replays the captured
initial torch RNG and passes the unchanged reread/clock/address test.
This is one mechanism replay, not a selected fresh draw or a learning
claim; approximate matches and missed memberships remain possible.

**Lifetime and restore.** A promoted whole is readable from its native
concept cache after its bounded run row retires. A re-formed type recognizes
that cached form without allocating another concept. Equal sentence and
document types promoted in the same close share a symbol; their run rows
remain distinct. Growing a live run invalidates a symbol that no longer
describes its form. New tests exercise capacity retirement with additional
live streams, bounded reads after retirement, and recognition with the bank
unchanged. Restore validates exact form width, live-run ownership, nesting,
brackets, counts and serials before replacing any live state. The real model
checkpoint round trip passes; the first attempt accidentally used the tiny
fixture's startup-autoload stub and is retained in log 26.

**Native write test.** Two separate setup corrections were necessary, with
all assertions retained. First, admission is not the selected trial's
numerical source: `_commit_sentence` restores the selected frozen values.
Second, the unconstrained chooser can keep an open relation, whose writer
stores canonical roles rather than the fused top of the parser stack. The
new test now captures the selected clause before commit and uses its
production point/role representation. Logs 16 and 28's failures remain;
passing another draw was not treated as a repair. The test keeps a failure
capture containing the initial RNG, chosen clauses, leaves, actions, whole
state and LTM state. The new capacity test also initially exercised eviction
of a lower-value recent type, not retirement of the promoted type; its setup
now supplies additional active streams without changing capacity or any
assertion (logs 28–31).

**Additional ports.** `test_item55_mechanisms.test_stream_chain` supplies
native bounded form coordinates in place of the scalar row-number vectors.
Its ordered payload assertion compares those values, and its upsert call
uses the same values. The two exact stream-bit checks become restored
content-read agreement and native threshold agreement. Its shared-row,
order, address, time and witness-count checks remain. The contextual
code-bank fixture supplies common earlier form content and distinguishable
later content for both whole rows and observations; all its assertions
remain. `test_ltm_consolidation` drops `stream` from the checkpoint-key
assertion loop and asserts its absence; three chain fixtures supply a whole.
`test_existence_provenance` supplies a whole for its two-role chain, and
`test_expectation_retention` supplies one at an explicit 0.25 threshold for
its legacy magnitude-two carrier. Their existing assertions remain.

`development/assertion-ports.json` lists exact old/new assertions and the
changed checkpoint-key loop; `test-ports.diff` retains every fixture edit.
No test is retired. Development groups pass in logs 20 (120 passed, 2
skipped), 25 (21 passed, 2 skipped), 27 (102 passed, 4 skipped, including
G1–G3) and 31 (35 passed). Each result belongs to its development source,
not a future frozen hash. Every failed invocation and diagnostic is retained.
The line remains unfrozen: VP-phase composition, reference-content selection
and the six-variant phase gate still need completion, followed by the scoped
gates and review through Alec. Item 3.0 has not been reached.

The subsequent native lesson smoke passes in log 32 (19.65 seconds).
Log 33 passes all 50 whole/MTM/consolidation tests, including a close that
obtains its active document whole from the run owner rather than requiring
an explicit argument. `development-source.json` hashes the current changed
source files; it is a development snapshot, not the line's frozen source.

## 10. Claude's review of the §2a build, and the book cover — 2026-10-24

Reviewed: §§7–9, iteration spec §22, the working tree (`WholeRuns`,
`WholeRead`, 33 changed files, unfrozen), the development logs 01–33 and
`consolidation-failure.pt`'s numbers. Alec's framing of Codex's finding:
wholes are not a good place for document context unless some percepts are
saved to LTM, because a document's identity in a whole is not transmitted
by concepts, which abstract over it; so save the document's identity in
LTM as (1) a perceptual "book cover" and (2) some episodic memory in LTM;
and since documents are aligned in batches, the document mechanism is in
addition to the per-batch LTM, MTM and STM.

**What Codex built, and what of Alec's two parts it is.** `WholeRuns`
keeps, per stream, an open run row for the sentence, the document and the
stream, each with a leaf-weighted centroid of its parts' forms, a bracket
and a stamp; `contextual_form` folds the three active forms into every
word's form read with equal weight (one quarter word, three quarters
context), after symbolic promotion and before the grammar trials, and the
same read reaches the pushed slab, the atoms and the reconstruction
source; so the written row's form block, its episodic part, carries the
trace. That is part (2), diluted: episodic memory in LTM as a blend in
each row's percept. Membership is the meet coverage of a row's form by a
whole's form above `contextMembership` (0.75), and every former reader of
the stream column and the sidecar reads it: the recency chain, the stream
guards, forgetting's age, consolidation's distinct-document and
distinct-sentence counts, MTM's close and document-end teardown, and a
query by whole (`WholeRead`). Promotion at recurrence three mints a
concept whose form is the whole's centroid; a re-formed type is recognized
by exact equality of centroids. Part (1) is absent: there is no cover.

**Three findings.**

1. *The trace drifts, so membership drifts.* The document's form is a
   running centroid, so the trace written into sentence 1 was made with a
   different form from the one in sentence 20, and membership is tested
   against the final form. Early rows fall out of their own document as it
   grows; Codex notes it. Two documents of similar vocabulary have similar
   centroids: in the captured failure a row's own document covers it at
   0.889 and other documents at 0.455–0.674, which is why the threshold
   went from 0.5 to 0.75. A fixed cover removes both: a distinctive
   percept, set at the run's start and never changed, carried unchanged
   into every row, so that membership is a stable match and distinct
   documents stay distinct however alike their sentences are.
2. *The blend is heavy and universal.* Three quarters of every word's form
   read is context, in every configuration, since `stage_sentence` runs
   wherever word atoms exist. The XOR standing gate and the decomposition
   arms, both on configurations without a meaning block, have not been run
   on this source, and word identification by form (`readback_scores`)
   now sees blended leaves against a bank of pure word forms. They must
   run before any freeze; and the sentence whole adds nothing to context
   (it is the sentence's own words) while costing a quarter of each form.
3. *The stream whole does the per-batch bookkeeping by content.* The
   stream's centroid drifts over everything the stream ever read, so
   stream isolation of antecedents, the recency chain and forgetting's age
   now rest on the weakest trace. Alec's framing separates these: the
   stream is a per-batch structure, and the document mechanism is added
   to it. Reading that literally, the stream's view of LTM is exact
   per-batch ownership, which the stream column was, and only the document
   mechanism is content-based.

**The book cover, as a design.** The document whole has two forms: its
*cover*, fixed at the run's start as the form of its first sentence (the
title, the first page), stamped and saved to LTM as the document's own
percept row, so it survives the run row; and its *type*, the running
centroid, kept for recognizing the kind of document and for promotion.
Each word's form read carries the cover at a small, configurable share
(`contextTrace`), not a quarter; membership is the row's match to the
cover; a re-read of the document matches its cover row in LTM, across
streams and batches, which is the cross-batch identity Alec asks for; and
promotion makes the cover the concept's form. The stream whole is dropped
from the blend; the sentence whole stays as the clause bracket, tense's
"now". The §2a readers split: document readers (consolidation's
distinct-document count, MTM document-end teardown, the query by document)
read the cover; stream readers (the chain, the guards, forgetting's age)
return to per-batch ownership if Alec confirms finding 3.

**Verdict:** not freezable as built; the direction is right and the cover
completes it. Decisions put to Alec below; the remaining line-6 work (the
VP-phase gap of §20.2, the surface inverses, nested admission, the §13.2
gate) proceeds meanwhile.

**Alec's answers to §10 (2026-10-24), under discussion.** On the cover:
"Discuss first." On the stream: "What is the advantage of a stream if we
have batch indexing and document separation?" On the trace share: "What
are the three active wholes? Parts are also significant here, I guess we
should take wholes/numWholes as the contribution, which keeps the centroid
centered." Claude's answers: the three active wholes are Codex's sentence,
document and stream runs; the stream has no advantage over the batch index,
which is what the old column recorded at write, so the stream whole goes
and per-batch ownership stays as bookkeeping; the contribution rule is
taken as stated: a word's form read is the centroid of its own form and
the mean of the active wholes' forms, one term however many wholes, which
with the stream gone and the sentence whole kept only as the clause
bracket is the document's cover alone. The cover itself is discussed in
the chat before §2b is written.

## 11. §2b code review and withdrawal decisions (Codex, 2026-10-24)

Claude's five documents were committed unchanged at `47a2216e0b`.
The complete rejected development tree, untracked files and port inventory
are retained in `development/withdrawn-2a` under the iteration-2 benchmark.
All its runs remain development records; none supplies evidence for §2b.
`WholeRuns`, `WholeRead`, the run-form reads, the membership knob and the
run fixtures are removed. The independent surface/inverse certificates and
language-neutral documentation corrections remain. The old paragraph
beginning “What a context whole is” below the new §2b is withdrawn §2a
history, not a second mechanism to implement.

1. **Batch ownership is restored literally.** `OccurrenceRows.stream_rows`
   and `witness_stream`, LTM0's stream buffer and writer/export/restore,
   `Forgetting.history_age`, `MereologicalCodes.begin_forward` and both
   `ReferenceClose` guards return to the pre-§2a implementation. The towers'
   `_interval` and `derive` are unchanged.
2. **Presentation and membership have different owners.** The eager
   sentence driver has `source_at` before perception and
   `_thought_document_ends` at teardown. It opens a cover once at each
   presentation boundary before either compose trial, using the ordinary
   allocator, and inserts its address into the predictor's recent frames.
   This does not supply a target or name, or create an MTM token for a
   whole. Cursor state detects boundaries only; historical membership is
   rebuilt from cover rows and the exact stream column.
3. **Equal stamps and rereads.** `RowTime.stamp` deliberately permits
   several writes at one model-present tick. Segment ordering uses decoded
   `.when`, with existing store order breaking equal stamps. A new cover
   precedes its newly appended sentence rows at that tick. Existing
   occurrence upserts retain their address, `.when` and store position:
   a reread opens a fresh cover but does not relocate historical rows or
   count an old occurrence as new recurrence evidence. No provenance
   sidecar compensates for that immutable-row contract.
4. **Pool expiry and recurrence.** Document end is passed the cover's
   occurrence, not a document digest. A token's introducing row is tested
   against that segment; the sixteen-close access rule stays. Reading
   recurrence counts distinct cover occurrences; token recurrence counts
   distinct retained rows, as §2b says. Legacy rows before any retained
   cover have unknown document membership and cannot invent a document
   witness. A forgotten cover must not silently reassign its segment to
   its predecessor; the retention pass must protect covers while their
   segment has retained rows.
5. **Covers are ordinary bounded storage.** `cover` is appended to
   `LTM0.KINDS` so existing kind numbers are unchanged. Cover writes have
   one occupied slot and an order-1 individual reference, no derivation.
   Allocation/capacity errors remain loud. A query naming the cover
   traverses its segment under the existing record meter. Covers do not
   themselves enter the recurrence learner as sentence observations.
6. **Centroid correction.** `GrammarLayer.compose_blocks` currently calls
   the parts-only `MeaningKernels.centroid`; its reverse subtracts the
   known part with `split_centroid`. Both faces must receive the same
   span's WholeSpace witness, including candidate reconstruction. The
   native source is WholeSpace's property table and its staged primitive
   counts, already read by `compute_word_property_event`, not invented
   run types or a modification of a lexical conceptual read. Higher-order
   promotion's `_fold` currently uses `amax`; the new fold averages
   members with their positive/negative evidence and keeps absent
   evidence unknown. The exact fixture ports and the span-read wiring
   are recorded with their implementation below.

### 11.1 Development findings and ports (2026-10-24)

Development runs 34–39 are retained. Run 36 passed 36 checks and found:
three consolidation fixtures missing their new cover/evidence inputs, a
new cover-menu fixture assuming native grouping instead of the ordinary
occurrence grouping, and two native capacity errors. The corrected cover
and consolidation unit group passed 27 checks in run 38. Run 37 was an
invalid diagnostic environment (MPS selected before pytest setup); run 39
records the actual CPU allocation failure.

The allocation capture has 64 physical seats, 15 occupied, and a four-seat
initial order-1 partition already occupied by three ordinary word seats
and one cover. The next cover incorrectly reported physical exhaustion.
Under §39 the existing unpartitioned tail is now available after the
initial higher-order partition fills, just as it already is for order 0.
The configured inventory and all fixtures are unchanged; no seat is
reserved and no live row is recycled. Grouped consolidation preflight uses
the same placement rule. The physical-capacity failure remains loud.

Ports so far: `test_memory_consolidation` writes cover rows at the fixture's
presentation boundary, replacing the implicit digest membership; the
three-document reading fixture now writes explicit positive evidence;
the token-fold fixture gives its actual sentence positive evidence and
expects `a0 + (a1 + a2)/3` instead of the old max-union `a0 + a1 + a2`.
Its two numerical checks use that same expected centroid. All other
assertions in that file remain verbatim. The new cover-menu fixture uses
the ordinary occurrence key, then binds through that candidate; it does
not install a lexical name for the cover.

### 11.2 Native centroids, inverse and bounded reads (Codex, 2026-10-24)

FormCentroids.stage reads existing WholeSpace property coefficients,
codes and staged per-word primitive counts outside the compose body.
SpanWholes.read tests those properties over the occupied word span's
primitive support and averages active rows by membership count. This is
where a composed span reads WholeSpace. The provenance stack carries its
span beside producer correspondence. No run row, identity, lexical blend
or membership threshold is created. MereologicalCodes._interval and
derive remain unchanged.

MeaningKernels.centroid centers the leaf-weighted parts average with that
whole average only for an order-0 result, including NP + VP lowering and
modal withholding. The temporary AnswerProgram retains the same witness
and counts for its inverse. A unary shadow read must gather that unary's
position, rather than both window positions; run 47 caught this error
and 49 verifies the repair. Answer generation receives the same native
whole candidate bank as reconstruction.

The free inverse receives neither an input span nor a saved tree.
CandidateWholes gets primitive support from candidates matching dictionary
leaf forms and reads the same WholeSpace table. An unspelled candidate
supplies no whole witness. Without a parent stamp, fit compares the
parts-only and centered form equations and fits the count ratio from
candidate forms. This inverse remains approximate. Higher-order promotion
weights the two evidence lanes separately without changing member rows.

Cover lookup now uses existing leaf/occurrence indexes, without scanning
unmetered meanings. Every segment payload is charged before reading.
Remaining query roles still constrain its match. A cover is protected
while any segment member survives; losing its last member releases the
boundary for a later pass.

Additional fixture ports: test_modal_timelines.close opens one cover and
writes to stream 0; its two expiry calls supply that cover and store,
replacing a document digest. Its promotion fixture supplies positive
evidence on the actual occurrence. Existing assertions stay verbatim.
The when-reader inventory adds DocumentCover.segments and the previously
missing ModalLearning.observe entry.

New-test corrections in retained runs 51–54: the cover fixture now uses
production's configure_model_index. The committed-bind check uses the
audit's reference and resolves its occurrence key through the store,
correcting an invented selected field. The new forgetting test initially
requested target zero while requiring a protected cover; its setup now
targets one, retaining the cover/member assertions and removing the
incorrect warning expectation. No existing forgetting test was changed.
Run 49 passed 55 checks including the native lesson smoke. Run 51 passed
131 and skipped 4 alongside retained failures. Run 54 passes all 13 cover
cases and the existing native tensor-loop/backward check. Run 48 used the
wrong interpreter and remains an invalid invocation.

## 12. Preflight stops requiring Alec's decisions (Codex, 2026-10-24)

This is still a working tree on `47a2216e0b`, without a line-6 freeze,
receipt, review verdict or publication. Claude's five requested files
were committed unchanged at that hash first. The withdrawn §2a source
and logs remain in the iteration's development/withdrawn-2a directory.
These findings meet §39's stop rule because their apparent remedies
reverse explicit gate or fixture constraints. Review remains through Alec.

### 12.1 The original decomposition draw no longer supplies a proof

Preflight used the original snapshots and seeds. XOR passed all eight
cases. Both decomposition arms passed one successor and failed the
unchanged two-successor `assert proved, questions`. These development
results are not promoted evidence for a later source.

The premise/keep probes locate the failure before decomposition. In the
first companion document, a live compose departure keeps a field instead
of "three plus one is four"; the second loses "four plus one is five".
Each chain lacks one arithmetic licence. The equality has R = 0.0360673778
and the competitor R = 0.0288539026, entirely from reconstruction.free_bytes.
A and reference cost are zero. The lower-cost reading is correctly kept.
Both thought trials lack a link: R ties at zero and A at 0.5019048452.
Work is not exhausted. decomposition-diagnostic-01 retains both premise
trials and correspondences.

decomposition-diagnostic-02 changes only a diagnostic plugin: restore
global RNG immediately after each random cover draw, retaining the covers,
codes and menu exposure. The same original zero-arm snapshot then passes
two successors. This isolates the added draws' effect on the departure.
It is **not a gate pass, source fix, seed selection or proposed production
RNG guard**. The intervention, manifest and working patch are retained;
production consumes its draws normally. Both failed arms and every
companion outcome remain.

**Question and recommendation for Alec/Claude:** keep the cover mechanism,
normal RNG advancement, keep law, budgets and proof assertions. Review
whether the architecture certificate's licensed-premise precondition can
be separated from live premise-keep behavior, with the latter finding
carried to item 0. This would amend the instruction that this certificate
pass unchanged and could change its forced scope; neither is authorized
or applied here. Do not choose another seed or promote the diagnostic.
A different mechanism diagnosis from these captures would instead justify
an ordinary §39 repair.

### 12.2 The existing forgetting fixture needs a twenty-fifth row

test_forgetting.test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads
fills 24 rows with eight prediction/observation/absence triples, then
calls boundary forgetting. The required cover adds a row. Run 51 fails
before the boundary; run 53 captures occupancy/capacity 24, zero forgetting
passes, and 1 cover + 8 estimates + 8 observations + 7 inferences, with
one more inference waiting. The capture/state are retained in
development/cover-capacity-failure.

**Recommendation requiring Alec's narrow exception to "no fixture
resizing":** capacity 24 → 25 in both constructors, its capacity assertion
→ 25, and fractional settings 8/24, 6/24 → 8/25, 6/25. Absolute water marks
stay eight and six, with the same input, interval, one boundary pass and
every behavioral assertion. The exact proposed diff is retained as
development/proposed-cover-capacity-port.patch, not applied or run.
Do not suppress the cover, reserve a slot, discard an inference or add
early forgetting to fit the old capacity. Production capacity policy
has not changed for this failure.

VP-phase wiring, six generated variants, final line-6 smoke/freeze/gates,
then line 7 and iteration review remain. Item 3.0 has not been reached.

## 13. Claude's review of the §2b build and the two stops — 2026-10-24

Reviewed: §§11–12, iteration spec §23, the working tree on `47a2216e0b`
(`DocumentCover`, `FormCentroids`, 29 changed files; `Occurrence` and the
towers untouched), the development logs 34–54, the two diagnostics and
the proposed capacity port.

**The build matches §2b.** The stream column and its readers are back
exactly (`Occurrence.py` is unchanged from HEAD). A document's start
mints an order-1 individual with a random code and writes a `cover` row
(`LTM0.KINDS` gains `cover` at the end, so no kind number moves) into the
batch's stream before the first sentence; membership is the segment after
the latest cover by decoded `.when` with store order breaking ties;
consolidation counts distinct covers and distinct retained rows; MTM's
close and document end read the segment; a query naming the cover reads
its segment under the meter; forgetting protects a cover while its segment
has members; the cover enters the recent frames as any row; nothing is
blended into word reads and no MTM whole or sidecar exists. The centroid
rules are implemented as decided: `MeaningKernels.centroid` centers the
parts' average with the span's WholeSpace average for an order-0 result
only, with the same witness carried to the inverse and a candidate-side
estimate for the free search; `MemoryConsolidation._fold` is the
evidence-weighted centroid per lane, an unwitnessed lane staying unknown.
The language-naming cleanups ("English" to "text") are right.

**Three notes for the receipt, not stops.**

1. *Exclude `cover` where `estimate` is excluded.* `premise_evidence`,
   `TimelineRead` and the other thought readers skip estimates and
   questions by kind; a cover row carries a concept code with evidence
   (1, 0) and must not be matched as a proposition. Codex lists the
   readers and adds the kind.
2. *Cover before boundary forgetting.* `open_cover` raises when the store
   is full, and the cursor opens the cover at the new document's first
   sentence. The boundary pass that frees rows runs at the document end;
   the receipt should show that the pass precedes the next cover under
   pressure, so that a full store at a document start forgets before it
   raises. The 24-row fixture (12.2) is the same question in a test.
3. *The span read is one interpretation.* `SpanWholes.read` takes a
   WholeSpace property as active over a span to the extent its
   coefficient holds over every primitive the span supports (the minimum),
   and averages the active rows. That is a reasonable reading of "the
   wholes active over the span"; §5.6 fixes it in a certificate, and it
   is recorded here as Codex's choice.

**12.1, the decomposition arms.** The cause is exact: `open_cover` draws
the cover's code with `uniform_` from the global generator, one draw per
document, so the fixed-seed arms take different live departures and a
cheaper reading drops a licensed premise; restoring the generator after
the draw makes both arms pass on their original snapshots. The project
already has the rule for this: an operator's initialization is
global-RNG-neutral (`_make_lex_gate`'s local generator, "constructing the
operator must not shift the seeded draws of fixtures built after it";
`MeaningAction`'s object readouts initialized "without consuming RNG").
Recommended: the cover's code comes from a dedicated generator owned by
the store, seeded once from a constant and advanced per cover, so the
code is random in the needed sense and the global stream is untouched.
That is an ordinary §39 repair under an existing convention, not a seed
chosen to pass and not an amendment of the certificate; Codex's proposal
to split the certificate's precondition from live keep is not needed.
Alec decides, since Codex framed it as his.

**12.2, the forgetting fixture.** Eight triples fill 24 rows exactly and
the cover is a twenty-fifth. Recommended: accept the exact proposed port,
capacity 24 to 25 with the fractional marks 8/25 and 6/25, the absolute
marks eight and six, the input, the interval, one boundary pass and every
behavioral assertion unchanged. It is a narrow exception to "no fixture
resizing" for a row the design adds, with the alternative, suppressing or
deferring the cover, rejected.

**Decided (Alec, 2026-10-24), the two stops.** 12.1: the cover draws its
code from a dedicated generator owned by the store, seeded once from a
constant and advanced per cover; the global stream is untouched and the
decomposition certificate is unchanged. 12.2: Alec, "resizing breaks
compilation, so that's not allowed", then "just preallocate more": a
capacity changed at runtime is a resize and breaks compiled shapes; a
larger capacity fixed at construction is preallocation and is fine. So
the fixture preallocates a comfortably larger store at construction,
32 rows, with the fractional marks 8/32 and 6/32 so the absolute marks
stay eight and six; the document keeps its eight sentences, and the
interval, one boundary pass and every behavioral assertion stay; the
capacity assertion reads 32. Alec also asked what the
principle behind a forgetting fixture is and whether access counts were
dropped; answered in the chat and recorded in §14.

## 14. The forgetting fixture's principle, and access counts (Claude, 2026-10-24, at Alec's question)

**The fixture.** `test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads`
is the one end-to-end check of forgetting on a real, tiny model: a
document of eight sentences, each close retaining about three rows
(prediction, observation, absence), fills the store; at the document's
end one forgetting pass runs at the configured marks (high water eight,
low water six, absolute, with the interval preventing a second pass),
and the fixture asserts that the pass fired once, reduced the store
toward six or left `forget_pending`, and reported; then a second document
is read and written, the store is checkpointed and restored, and the
restored occupancy, addresses, counters and every row's meaning agree.
The cover is one more row in that count, hence 25.

**Access counts were not dropped; they were never a retention term.**
The forgetting spec of 2026-09-16 (§3) defines a row's value as
`V = w_t·trust + w_u·utility + w_l·luminosity − w_a·age`: provenance
trust, utility as one minus deducibility (the row's `surprise` for ideas,
one relation-reader step for part and implication rows), luminosity as
one minus dissonance (or coverage gain), and age as the row's distance
from the tail of its stream's history, with the explicit rule
"re-witnessing does not move a row; a read does not refresh it." The
access count (`witness_count`, incremented at each re-witness, renamed
"access count" in the layout) is a column the spec keeps, read today by
identity definedness (`Occurrence.identity_code`), placement's occurrence
roots, the reference resolver's witnesses and the components' witness
reads, but not by `retention_values`. MTM, by contrast, is refreshed by
reference (2c's sixteen-close rule), so working memory forgets by disuse
while long-term memory forgets by low value and age in the stream. If
Alec wants frequency of access to count toward retention, it is one term
added to `V` and an amendment of the spec, not a repair.

## 15. Applying Alec's final preallocation and §13 notes (Codex, 2026-10-24)

§13 supersedes both §12 stops. The forgetting fixture preallocates 32 rows
in both constructors and checks capacity 32. Its fractional high/low marks
are 8/32 and 6/32; the absolute marks remain eight and six. The eight
sentences, interval, one boundary pass and all behavioral assertions are
verbatim. This is construction-time preallocation, not a runtime resize.
The earlier 25-row proposal is retained as superseded development material.

`LTM0` owns a CPU generator seeded once with the constant zero, together
with a checkpointed `cover_rng_state`. `open_cover` draws from that stream
and advances its saved state. It neither draws from nor resets the global
generator. Older checkpoints begin this newly introduced stream at its
initial state. Consecutive covers differ, checkpoint continuation reproduces
the next cover code, and the global RNG state remains identical. The
decomposition fixtures and their original snapshots and seeds are unchanged.

The three review notes are resolved as follows:

1. Proposition readers exclude covers: `MemoryIndex.cued_rows`,
   `ThoughtLTMCapability.end_evidence`, `best_match` and `premise_evidence`,
   both `TimelineRead` paths, `DocumentCover.read`'s segment payloads, and
   the generic LTM attention pool. `accept_fact` also refuses a cover.
   Consolidation already excludes it. Stream history, ordinary reference
   access and occurrence placement retain the cover; those read context
   objects, not proposition evidence. No cover's (1, 0) evidence certifies
   a proposition through an addressed or concept-cued read.
2. Production's streaming loop completes the preceding document's backward,
   buffer flush and resets, then calls `post_tick_compact`, before the next
   `runBatch` can open a cover. A full-store certificate follows this exact
   hook order: one boundary pass frees rows, then the next cover opens
   without a second pass. No compaction is inserted into a live forward
   graph, and interval/protection rules are unchanged. If a boundary cannot
   free space under those rules, exhaustion remains loud. The real-model
   fixture separately checks this sequence through the next document and
   checkpoint reload.
3. A span activates a native WholeSpace row to the minimum of its
   coefficients over all primitives supported by that span. The active
   rows are averaged by their total coefficient; that average is centered
   with the leaf-weighted parts average for an order-0 result. The known
   parts/wholes certificate fixes this interpretation and its inverse;
   no run type or contextual word blend is introduced.

Development run 55 passes all 49 cover/forgetting tests, including the
unchanged eight-sentence behavioral checks. This is development evidence,
not a freeze or receipt. XOR and both original decomposition arms are
rerun in `preflight-02` with the production generator and full captures;
the diagnostic RNG intervention is not loaded. All earlier failures and
withdrawn builds remain retained. Line 6's remaining work and the single
freeze, line 7 and the iteration review through Alec are unchanged.

## 16. Preflight and the remaining inverse/fixture repairs (Codex, 2026-10-24)

`preflight-02` completes on the working source: XOR standing 8 passed,
the original zero-budget decomposition arm 2 passed, and the original
enabled arm 2 passed. Both decomposition arms use their retained initial
snapshots and seeds, with complete arrival and keep captures. No diagnostic
RNG intervention is loaded. Run 57 also passes 104 reader, consolidation,
modal and thought checks (4 prerequisite skips). These are development
runs, not evidence promoted to a frozen source.

The free unary inverse lacked the whole-center removal already present
in the retained-program inverse. `FormCentroids.unary` now compares the
plain and centered form equations against the independent candidate bank,
then removes the selected whole center before reversing the unary meaning
kernel. It receives no compose witness or target word. Plain and centered
cases pass alongside the binary inverse certificates; meaning is untouched
by this form adjustment.

`test_item55_mechanisms._stream_code_banks` assumed a free read of a bare
word produced one native object. With covers available as ordinary objects,
the captured reads instead bound that word to two covers. Selecting one
cover cannot repair the fixture: a native identity's cache intentionally
does not vary across streams. The fixture now admits the lexical form
through `Interpret.lookup_word`, opens a cover in each stream, and writes
its two explicit contrasting observations through the existing clause
writer. The cover supplies the preceding history that the production path
has. Every assertion is verbatim, including the different stream codes,
bank correspondence and free inverse. Run 65 passes. Runs 59–64, including
the failed intermediate setups and their full state captures, are retained.
This is a fixture port under step-2 §39, not a change to identity placement
or a restriction on binding to covers.

The obsolete grammar-file comments prescribing a clock shift and a
morphology rewrite layer are removed; no grammar rule changes with those
comments. The VP-phase and six-variant work identified in iteration §20.2
still precedes line 6's smoke and single freeze.

The follow-up native lesson in run 66 found a domain defect in the new
free unary search: a closer centered candidate could unfold outside the
membership chart. The centered equation now requires a finite recovered
form in [0, 1], with the plain equation retained as the other hypothesis;
no value is clipped. Run 67 passes the 19 cover/inverse checks and the
native lesson smoke. Run 69 passes 56 meaning-kernel, retained and free
reconstruction, and surface checks (3 existing slow skips). Iteration §24
records the separate compound/phase contract question submitted to Alec.

## 17. Pointer: presentations and covers (2026-10-24)

Alec decided that memories are deleted between training presentations of
the same source; the cover concept persists per document and its row is
rewritten into the empty store at each presentation. The boundary, its
audit list and the capacity rule are in the iteration spec
(`2026-10-23-verb-and-modal.md` §45).
