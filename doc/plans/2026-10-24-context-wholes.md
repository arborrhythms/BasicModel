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
