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
8. No language in code: no indexical or marker is named.

## 2. The mechanism, in code names

**A context whole is an MTM identity.** Two per batch stream: the
*document* whole, opened at the first sentence of each document, and the
*stream* whole, opened at the stream's first sentence and never reopened
(it is the conversation). Both are ordinary `Token`s under capacity,
names, expiry and promotion, with one difference of key: a context whole
has no mention row, so its key is a context address `('ltm', 'context',
digest)`, with `document_digest(source_at(model, b, slot)[0])` for the
document (today's provenance key, so a re-read finds the same token while
it lives) and the stream's digest for the stream. Each sentence row the
stream writes is an occurrence of both wholes: `ReferenceClose.commit`
binds the row to them (`mtm.bind(document, occurrence)`, `mtm.bind(stream,
occurrence)`), and a sentence is a description mention of its wholes, so
its meaning *joins* their values ("VP replaces, mention joins"); nothing
acts on a context whole as a verb. A document's occurrences are its
sentences in store order; the stream's include every row it witnessed.

**Percept part.** The whole's form is the leaf-weighted centroid of its
sentences' forms (`MeaningKernels.centroid`, counts from the journal's
leaf counts), its `.where` the bracket from its first sentence's start to
the current sentence's end (`RunStructureLayer`'s contiguous run; one
document is one run of sentences in its stream), and its `.when` is
stamped now when it enters the field, as a concept entering STM is. The
form is updated at each kept close. Nothing is placed on the priming
surface before promotion.

**In the field: the first prepended entries.** The two context wholes are
prepended to the symbol array that conceptual space reads for every
sentence of the document: the per-word arrays `isp._ar_word_concept_ids`,
`_ar_word_concept_rows`, `_ar_word_concept_orders`, `_ar_word_symbol_where`
and `_ar_word_symbol_when`, and the compose window (`stm[0]`, `words`, the
`identities`, `positions` and `orders` of `prepare_operands`), as two
leading positions per batch row, with the whole's current value as content
and a reserved transient symbol id (the token key) until promotion gives
it `symbol`. This is the first consumer of Alec's prepend suggestion,
confined to these two entries; the identity overlay keeps the ordinary
identities until item 3 retires it. Consequences the design wants:

- the parallel prelude (`_per_word_prelude`; interleave groups) folds the
  context entries with the words, which is how each sentence's concepts
  carry the wholes' trace (§3.4: the parallel read is where concepts see
  every part and whole);
- attention's mask over `.where` includes them: the widest bracket;
- the serial chooser can bind a mention to them through the ordinary
  reference-expanded menu (`binary_refs`), which is how an indexical takes
  a context whole as referent, by exposure, with no word named;
- a sentence read in two documents composes to different fused points; the
  same sentence re-read in one document refreshes its address as today.

*Claude's call, reversible:* prepend now rather than stage them in the
overlay, because the decided behavior, presence in the field for the
parallel read, is exactly what the overlay cannot give. The twelve-draw
pending-premise series runs once for this line, since the menu gains two
locations.

**Readers replaced now.** Every reader of the `stream` column reads the
stream whole's occurrences instead: `LTM0.get_stm_chain` (the recency
chain: the whole's occurrences in store order, the same list),
`Occurrence.stream_rows` and `witness_stream` (membership in the stream
whole), `recency_weights` (goes with `PartRead` in item 3; until then reads
the whole), `Forgetting.forget`'s stream read, and `ReferenceClose`'s two
stream guards (a row is this stream's iff it is an occurrence of its
whole). 2c's `_consolidation_sources` sidecar (`ClauseRow.write_clause`,
`MemoryConsolidation.consolidate`, the `Spaces` extras) is replaced by
document-whole membership: a reading's distinct documents are the
document wholes its rows belong to, and a token's distinct sentences are
its occurrences' places in their document whole. `mtm.end_document`
retires the document whole and the entries whose last reference lies in
it, as now; the stream whole is refreshed by every close and lives while
the stream does, under no exemption. The `stream` and address columns are
still written and unread; item 3 drops them.

**Nesting and queries.** Sentence ⊂ document ⊂ stream is read from
membership and brackets; no row is written for it. "What was said in *d*"
is a bounded read of *d*'s occurrences under the work meter, in the shape
of `TimelineRead.read` (a query bound to the document whole), and after
promotion a content match against the document's concept. The current
sentence is the innermost context whole and the default object of
`position` for tense, §4's "now" (§3.4: "tense's 'now' is the current
sentence's whole"); the parallel group is the batch and gets no individual
in this line.

**Lifetime.** A document whole is retired at the document's end; three
sentences promote it to a concept (2c's rule, unchanged), and a later
re-read of the document opens a new token at the same context address and
binds it to that concept through the native path. The stream whole is
never retired while the stream is live.

## 3. Not in this line

Word adjacency (dropped); retiring the `stream` and address columns, LTM0
one slot and LTM1 chains (item 3); prepending the ordinary identities and
retiring the overlay (item 3); a conversation beyond the stream; the
parallel group as an individual; the §72 sentence-level stamp and recall
by `.when` (item 3).

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

1. A stream reading two documents opens one document whole per document
   and one stream whole; every sentence row is an occurrence of its
   document and of the stream; the document's form is the leaf-weighted
   centroid of its sentences' forms with the document bracket, and its
   value the join of their meanings; the whole's percept `.when` is the
   entry stamp.
2. Every sentence's compose window carries the two context entries first;
   the parallel prelude's fold includes them; one sentence text read in two
   documents yields different fused points, and re-read in one document
   refreshes its address.
3. `get_stm_chain`, the `ReferenceClose` stream guards and `Forgetting`'s
   stream read give the same results as before on a fixture; 2c's
   recurrence and promotion certificates pass unchanged with the sidecar
   gone.
4. Document end retires the document whole and not the stream whole; three
   sentences promote the document; a re-read after expiry binds the new
   token to the promoted concept.
5. A fixed-operation fixture binds a leaf to the document whole through the
   ordinary menu; the context entries are never placed on the priming
   surface before promotion.
6. Unchanged: G1–G3, both decomposition arms, XOR, the pending-premise
   certificates, plus the twelve-draw series once for the menu change.

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
