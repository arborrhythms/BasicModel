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

## 2. The first mechanism — withdrawn 2026-10-24

The first draft made a context whole an MTM identity keyed by a context
address, bound each sentence row to it, prepended the two wholes to the
symbol array, and replaced the stream-column readers by token membership.
Alec: "wholes and parts are in different spaces, in general." Withdrawn;
the record is in the git history of this file. Nothing of it is built.

## 2a. The mechanism, in code names: wholes as types over runs

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

Word adjacency (dropped); any MTM entry or prepended symbol for a whole
(withdrawn); retiring the address columns, LTM0 one slot and LTM1 chains
(item 3); prepending the identities and retiring the overlay (item 3); a
conversation beyond the stream; the parallel group as a row; the §72
sentence-level stamp and recall by `.when` (item 3).

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

## 5. Certificates for the context wholes

1. A stream reading two documents holds one document whole per document
   and one stream whole as WholeSpace rows constant over their runs; the
   document row's code is the leaf-weighted centroid of its sentences'
   forms, with the run's bracket; the sentence whole's run is the clause.
2. Every word's conceptual read in the parallel prelude includes the active
   wholes; one sentence text read in two documents yields different fused
   points, and re-read in one document refreshes its address as today.
3. Membership by content: with the `stream` column gone, `get_stm_chain`
   returns the same rows in the same order as the column did on a fixture;
   the stream guards and forgetting's stream read agree with the old column
   on that fixture; 2c's recurrence and promotion certificates pass with
   the sidecar gone.
4. Runs end: the document boundary ends the document whole's run and not
   the stream's; `end_document` by whole retires the same MTM entries the
   digest did on a fixture; a re-read document re-forms the same type and
   its promoted concept recognizes it.
5. A word takes a whole as its object through interpretation in a
   fixed-operation fixture, no word named; the wholes are not on the
   priming surface before promotion.
6. Unchanged: G1–G3, both decomposition arms, XOR and the pending-premise
   certificates; the bind menu is unchanged, so no twelve-draw series.

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
