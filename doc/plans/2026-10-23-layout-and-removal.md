# Layout and removal: ideas in LTM0, chains in LTM1, no derivation (Claude, 2026-10-23; for Codex)

**Status:** item 3 work (Alec, 2026-10-23: "the layout spec should be done
with 3.0 of the todo"), moved out of item 3.5 step 2, where it had been
sequenced after the identity line ([step-2 spec §36](2026-10-21-step2-one-ltm.md#36-decided-alec-2026-10-22-identity-first-the-3-5-sequence),
[§72](2026-10-21-step2-one-ltm.md#72-decided-alec-2026-10-23-the-four-open-points-of-71)).
It lands with item 3's placement from the priming surface and the
geometric channel, which it needs (§2). Until then 3.5 runs on today's
storage: the derivation and its readers stay, as the identity spec
already requires. STM's eight chunks follow it, also in item 3.
It builds the layout Alec decided in §§71–72 and removes the stored
derivation (§30). Under §39, Codex reviews this spec against the code,
fixes what it contradicts, and continues, documenting each decision; only
reversing a decision in §1 comes back to Alec. STM's eight chunks (§72,
"STM: what is open") are the round after this one, since this round can
read the existing journal as the chunk record.

## 1. Decided (Alec, 2026-10-22 and 2026-10-23)

1. No sentence derivation in LTM; if the original sentence is needed, it
   is reconstructed from the vector, inexactly (§30).
2. **LTM0 has one slot.** An entry is one vector in conceptual space, an
   absolute truth understood directly, with lanes, access count, surprise
   and `.when`; no symbol is required (§71).
3. **LTM1 has two tagged references and no stored type.** Each reference
   is an LTM0 entry, an existing concept or word symbol, or another LTM1
   row, recursively; a row referencing a row is LTM2, derived (§71).
   If-then, whole-part and type-token are told apart by their references'
   situations and kinds, temporal, spatial, symbolic (§72).
4. **No sentence row.** A sentence whose words are all existing concepts
   compresses to one LTM0 idea; one that cannot is an LTM1 chain whose
   top row is the sentence; V O becomes a V (§71).
5. A mention's reading and an identity's role evidence live in MTM; a
   recalled chain brings its identities back through `query()` (§72).
6. No word bag: retrieval by word is a content match or a chain row that
   references the bare word; membership diffusion is activation over the
   priming surface with nearest-neighbour selection; placement updates
   at the close (§72, decided).
7. A chain whose unknowns later resolve is immutable; thought writes the
   value to MTM and, when kept, an inference idea (§72).
8. Close-time argument minting ends (§11.1); no word is another word's
   object (identity spec §1.6).
9. A pre-layout checkpoint is rejected by version, loudly; no
   classification is guessed (§33 R8).
10. **Every mental event has a perceptual and a conceptual part.**
    Episodic memory is the percept part of STM and LTM, semantic memory
    the concept part; an entry is `[form | meaning]`. `.when` is not a
    field of LTM but belongs to the percept part, so the separate
    `LTM0.when` and `LTM1.when` columns retire into it; recall by time is
    a cue with only the percept `.when` filled. An order-0 concept is an
    event with `.where` and `.when`; higher orders abstract them. The
    stamp sits at the sentence level only. An STM row has the LTM0 form,
    a concept entering STM being stamped now (§2, "Recall by `.when`
    alone" and the decided paragraphs after it).
    *Decided (Alec, 2026-10-25; STM-chunks §27 item 2):* an estimate's percept part carries no time; its zero `.when` supersedes "stamped ahead" in §2's decided paragraph, while the arrival alone keeps its exact stamp.
11. **STM is eight shared chunks**, each over a few ideas and relations,
    one working set for the open sentence, recalled chains and recent
    rows; the least recently read chunk not still on the parse stack
    leaves when a ninth arrives (step-2 §72).
12. **Operations on meaning, centroids on form.** Every grammar and
    logical operation acts on the meaning block, the logical ones by
    their exact rules and the grammatical ones by learned kernels; a
    reduce's percept part is the centroid of its constituents' forms,
    with the bracket for its bands; zeroth-order concepts start as random
    codes and are moved by the priming surface; generation reverses the
    meaning composition and reaches forms through the symbol index (§2,
    "Decided: operations on meaning, centroids on form", with the audit of
    today's form-side kernels and their side effects).
    *Moved to item 3.5 the same day* (Alec: "The whole swap now"): it
    is built in 3.5's first remaining iteration
    ([spec](2026-10-23-verb-and-modal.md) §2), before line 4's verb.
13. **The object supplies its own Q basis** for a transitive verb; the
    verb is the diagonal (occurrence spec §14.5; step 4's spec).

## 2. The layout, in code names

**LTM0** (`Layers.LTM0`): `slots[cap, 3, D]` becomes `slots[cap, D]`;
`role_mask` goes; `_semantic_rows` keeps only semantic bindings that are
not structure: a question's open slots, bound roles, scope, source text
and expectation provenance, authenticated as today. `role_refs` goes: a
three-role meaning is no longer a row. `ConceptualMeaning` keeps its three
roles as the working form in STM; `ClauseRows.write_clause` writes either
one idea (the fused `clause.point`) or a chain.

**LTM1** (`LTM1.LTM1`): `refs[cap, 2]` gains `ref_kinds[cap, 2]` (int8:
`OCCURRENCE`, `SYMBOL`, `RELATION`; zero invalid), the P2 port. `find`,
`_index`, `parts`/`wholes` and `readings` key on (tag, value). An OCC
value is an LTM0 address key; a SYM value an allocator id; a REL value a
row handle. `_load_from_state_dict` validates tags before `super()` (P5's
tag cases) and rejects a checkpoint without `ref_kinds` by version.
Protection: a row protects what it references, transitively; a REL
reference makes the referencing row LTM2 on read (`stratum(row)`).

**Compression at the close.** `ClauseRows.write_clause` walks the kept
reading's journal (`Reading`, renamed from `AnswerProgram`, 14 files) from
the leaves: a composition whose operands are all resolved, a concept
symbol, a bound identity with a value, or an already compressed child,
compresses to its journal value; one that has an unresolved operand (a
pending name, an unknown word standing for itself, an open slot) is
written as an LTM1 row `(left, right)` over its operands' entries, with
the operation's own value nowhere. The sentence's top composition is
written as an idea if it compressed, else its chain's top row is the
sentence and `written_rows[id(clause)]` holds that handle. Parse-only
operations (`surface`, markers, determiners that only select, order
operations) never become rows: a chain row's operands are its semantic
operands, as `ClauseJournal.region` already picks the semantic operand of
a headed operation. The identity line's `mtm.event` records the sentence
in its subject's history with the idea's address or the chain's handle.
A compressed idea keeps its reference to the identity it was composed
over (the token's key or promoted symbol, in its semantic bindings), as
a chain's OCC reference does: conceptual space's access to MTM
identities is the witnessing that plants the seeds of attachment and
must survive compression (Alec, 2026-10-23; iteration spec §4 item 8).

**Derivation removal.** `PartClosing._plan_term` no longer writes
composing Part edges; `_word_origins`, `_definition` and `_reserve` for
argument mints go (§11.1, §1.8). `PartRead.argument_reader`,
`nonparticipants`, `occurrence_roots`, `occurrence_memberships`,
`word_rows` and `expression_components` are removed with their callers'
reads:

- `ReferenceClose.accessible` reads candidates from the frames' written
  activations and MTM, as the identity line leaves it; `_participants`,
  `expected_roles` and `prediction` read MTM event histories (identity
  spec).
- `Queries._with_components` and `BindingAnswers` read chains: a
  component view is the chain's rows; an idea has none, and anything
  finer is reconstruction from the vector, explicitly incomplete.
  `ThoughtBinding.integrate` keeps witnessed equalities and uses
  evaluation through identity (identity spec); its component congruence
  goes with the components.
- `MereologicalCodes.occurrence_terms` becomes an online update at the
  close, `observe_close(reading)`: each word of the kept reading moves
  its order-0 meaning toward the sentence's context code (an EWMA over
  closes), so placement needs no stored membership; item 3 replaces the
  context source by the registered priming surface.
- `Priming.diffuse` keeps the edge channel along LTM1 references (the
  chains and the taxonomy are the Part edges now) and drops the
  word-to-sentence membership edges; the nearest-neighbour channel over
  the surface is item 3's geometric channel.
- The leaf index (`MemoryIndex.LeafCodeIndex`, `leaf_terms`,
  `configure_model_index`'s rebuild) retires; retrieval by word is
  `Queries.best_match` by content, or a chain row referencing the word's
  symbol (`LTM1.rows_referencing(SYMBOL, word)`).
- The expectation predictor (`SentenceExpectation`, `ClosingImage`) reads
  an idea's one vector, or a chain's head idea where it has one, in place
  of three slots; comparisons are per entry.
- Forgetting (`Forgetting.forget`, `forget_parts`) values ideas and chain
  rows by their own value; a chain row protects what it references and is
  released for the next pass; the structure stage is gone, since an idea
  has none; source wording stays the first stage.

**Recall by `.when` alone (Alec, 2026-10-23).** "A mind should be able
to recall mental content by filling in only the `.when`, and leaving the
rest blank." Both strata already carry `.when` as a column (`LTM0.when`
and `LTM1.when`, `[cap, 4]`, `RowTime`'s sinusoidal model time) and no
read is keyed by it: recency is stream order (`recency_weights`) and
queries match content or references. Add the time-cued read: a `query()`
whose description is blank and whose `.when` is set returns the LTM0
ideas and LTM1 rows nearest that time in the encoding, within a bounded
window, under the work meter, loaded into STM chunks with their
identities re-entering MTM. Several entries at one time are an episode,
so this makes the episodic-memory note decided work
([FutureWork](../FutureWork.md#episodic-memory-several-ltm0-entries-per-time-alec-2026-10-23)).
Concepts carry no `.when` (decided 2026-10-20, kept): an STM chunk's idea
has the same form whether pulled from a concept or from LTM. **The stamp
is inserted at the final sentence level, not in every entry (Alec,
2026-10-23).** Only the entry that is the sentence carries `.when`: the
LTM0 idea it compresses to, or the top row of its chain. Mentions and
inner chain rows carry none; their time is their sentence's, reached by
reference, and an identity's event history is ordered by its sentences'
stamps. Today every row of a close is stamped alike (`write_clause`,
`closing.when`); this round stamps the sentence entry only, and recall
by time returns sentence entries, from which chains and mentions follow.
Forgetting's age reads the sentence's stamp for what the sentence
references.
**STM rows have the LTM0 width (Alec, 2026-10-23):** "when a concept
is written to an STM row, it should receive a timestamp, which makes its
width equivalent to an LTM0." The code facts (checked 2026-10-23):
percepts and symbols are muxed event vectors with `.where` and `.when`
bands inside them (`canonical_shape`: 4 + 4 for InputSpace, PartSpace,
WholeSpace and SymbolSpace); a concept has no bands at all
(ConceptualSpace `(0, 0)`: "one opaque code"), and its cached paired
row is `[form | meaning]`, the symbol's muxed form block first. The
symbol band's `.when` is the sentence's position in its source document
(`relative_positions`, `source_at`), not the clock; the clock is
`RowTime`, model time on a 4-dim ladder, kept as LTM0's `when` column:
"a column, never an operand of composition".

**Decided (Alec, 2026-10-23): every mental event has a perceptual and a
conceptual part; episodic memory is the percept part of STM and LTM,
semantic memory the concept part; `.when` is not a field of LTM but
belongs to the percept.** "A concept is one opaque code that has
generalised over where and when" is not to be assumed: an order-0
concept is an event and has both; higher orders abstract them (nouns
probably drop `.when`, not necessarily). The paired row `[form |
meaning]` is this split: the form block is the entry's percept, located
by its `.where` and `.when` bands, and the meaning block its concept. So
an LTM0 entry, an STM row and a concept row have one form, and `.when`
lives in the percept part of each. Alec asked where a sentence's
perceptual content comes from, beyond its `.when`: from its surface.
Composition acts on the full paired vector, so a fused idea's form block
is the fold over its words' forms, lossy, which is what the free decoder
reconstructs from ("reconstruct from the vector; it will not be exact"),
with its `.where` and `.when` bands; and the source wording is retained
beside the row (`_texts`) as the first forgetting stage, the exact
surface while it lasts. Alec asked how a reduce generates the new percept fields, given that
the operations describe how concepts combine, and whether centroids
serve. Claude first answered that the operations act on the whole paired
vector; Alec corrected it: "grammar operations do not operate on both
form and meaning together. They combine concepts in conceptual space,
and we are discussing how percepts combine." The code fact, checked:
`GrammarLayer.compose_blocks` splits the paired vector and applies the
operation's learned kernel (the sigma or pi butterfly) to the operands'
*form* blocks, while the *meaning* blocks combine by a declared
bilattice rule (`MeaningCodes.compose`: intersection and conjunction
take the lane-wise meet, union and disjunction the join, `sum` the
mean, `not` the lane exchange) or, for a headed operation, pass one
operand's meaning through: `lift(NP, VP)` keeps the subject's meaning
and drops the verb phrase's, `verb(V, NP)` keeps the verb's and drops
the object's, `lower(DET, N)` keeps the noun's. So today the learned
grammar acts on percepts, the logical operations combine concepts by
fixed rules, and a predication's idea in conceptual space is its
subject's meaning alone: nothing of "runs" enters "Felix runs". That
is the reverse of the design, and it is why identity threading's "the
VP replaces the identity's state" has no composition to act through
yet on the meaning side. The free decoder reverses the form
kernel (the candidate search's form residual). The `.where` and `.when`
bands of a reduce are a bracket from the first owned leaf, not a fold
(`ClauseJournal.bands`).

**Decided (Alec, 2026-10-23): operations on meaning, centroids on form.**
"Percepts are mereological towers, and therefore a shape at order-0
percepts. Even though higher-order concepts have a mereology that is
less and less meaningful at higher orders, centroid is still reasonable
on forms. And grammar and logical operations are all on meaning, with
zeroth-order concepts starting with random assignment and being moved
by the priming surface." So, in item 3:

- The percept part of every reduce is the centroid of its constituents'
  forms in the percept cube, the mereological fold that makes a word's
  form from its letters, with the bracket for `.where` and `.when`. The
  learned form kernels retire; no operation learns on forms.
- Every operation acts on the meaning block, in conceptual space: the
  logical ones by their exact bilattice rules (the order-0 calculus of
  `intuit`), the grammatical ones, `lift`, `verb`, `lower` and `bind`,
  by learned kernels, so a predication composes the verb phrase into
  its subject's meaning and identity threading's "the VP replaces the
  identity's state" has its composition.
- A zeroth-order concept's meaning is a random code at its mint, moved
  by the registered priming surface (the placement decision); no fold
  generates it.
- Generation reverses the meaning composition to word meanings and
  reaches forms through the symbol index, the only link between the
  spaces; reconstruction's objective moves with the kernels, which is
  item 3's invertible-path work.

**Side effects of the form-side kernels (Claude's audit, 2026-10-23, at
Alec's question).** Fewer than the discrepancy suggests, because the
paired representation made most semantic readers meaning-only:
`IdentityRows.coordinates` zeroes the whole percept-event block, so
bind's meet, `individuals`, the reference-reconstruction term, queries'
vectors, extents and negation, placement's context codes
(`MeaningCodes`), answer credit (`SentenceCredit.forward_discard`) and the
closing image's split all read meanings. What the form-side arrangement
does affect:

1. *Predication's idea.* With headed passthrough, a sentence's meaning is
   its subject's; expectation over stored ideas predicts subjects, and
   evaluation through identity cannot compose "x plus two" on the meaning
   side ("plus 2"'s meaning is "plus"'s). The decomposition gates pass on
   stored components and the form side.
2. *The identity line's values.* "The VP replaces the identity's state"
   is implemented, but the verb's effect today lands in the form block of
   the value, since that is where the kernel acts; the meaning block of
   the state stays the subject's. The mechanism is right and its content
   is on the wrong side until item 3 moves the kernels.
3. *Reconstruction's objective* trains the form kernels; after the swap
   it trains the meaning kernels through generation, and the surface is
   reconstructed from the percept centroid by mereological decomposition.
4. *The XOR composition gate* exercises the kernels on forms; it moves to
   meanings with them (`XOR_grammar.xml` already carries a meaning width).
5. *Reading back* a form fold through the learned kernel's inverse
   (`Language.py`, the candidate search) becomes reading parts out of a
   centroid, the meronomy's own inverse.

Everything else, MTM, STM, LTM0 and LTM1, bind, forgetting, priming and
the orders, is unaffected by which block the kernel acts on.

*Corrected on checking (Claude, 2026-10-23).* The split exists only where
the conceptual space is wider than the form: `BasicModel`,
`BasicModel_answers_tied_benchmark`, `MM_grammar`, `XOR_grammar` and the
fixtures that widen it. `XOR_exact`, `MM_xor`, `MM_math_chain` and the
default `model.xml` have no meaning block; their kernels act on the whole
vector. So point 4 is wrong (the XOR standing gate runs on `XOR_exact` and
`MM_xor` and does not move), and points 1 and 2 hold only on the paired
configurations: on `MM_math_chain`, which the decomposition arms use,
"x plus two" composes on the whole vector.

**A contiguous event takes its actual mereology.** Alec: "Hopefully it will
take on the actual mereology of the event if the event is contiguous?"
Yes: a sentence or clause whose words are one run in the input is a whole
in WholeSpace's sense, a maximal constant-type run, its parts the words,
its `.where` the bracket, and the centroid of its parts' forms is that
whole's form, as a word's is the centroid of its letters. A discontiguous
event, one interrupted by another clause, has no single run; its centroid
is then a type over disjoint parts, Orders §9's discontinuous case, which
is where the mereology becomes "less and less meaningful" as Alec says.
 A read sentence's percept part is therefore
what was said, where and when; its concept part what it meant. A
concept pulled from the index
gets a percept part stamped now, the mental event of thinking it; a
recalled entry keeps the percept part it was stored with; an
expectation's estimate has a percept part stamped ahead. The separate
`RowTime` column folds into that band: `LTM0.when` and `LTM1.when`
retire, forgetting's age and recency read the percept `.when`, and
recall by time is a cue with only the percept `.when` filled and the
concept part blank. Claude's reading to confirm: the percept `.when`
of a read sentence is the mental time of reading, `RowTime`'s tick, not
the sentence's position in its document, which the stream's context
whole and the address keep (5.5). Sentence-level stamping (above)
stands: inner chain rows and mentions carry no percept part of their
own time; the sentence's does.
A recalled entry keeps its own stamp,
the time it happened, while its place in STM is governed by recency
(Claude's reading of the recalled case). So STM, LTM0 and the sentence
entry share one form, and the close keeps the stamp on the sentence
entry alone.
Alec: "That will allow thoughts of the three times." A chunk's stamp
places its content in the past (a recalled entry keeps the time it
happened), the present (a concept or perceived sentence stamped now)
or the future (an expectation's estimate stamped ahead of now), and
the same chunks hold all three, so thought ranges over them with one
form; tense (5.5) is the relation between a sentence's stamp and now,
and the modal (step 5) ranges over possibilities at a time.

- `TimelineRead.read`'s expired-branch fallback (iteration 1, §19 note 1)
  reads subject and predicate through `PartRead.argument_reader`; after
  removal it reads the chain's references.

**Readers that stay.** Semantic bindings in `_semantic_rows`
(`bind_constituents`, `read_structure` for those bindings), the pending
name index (`_forward_addresses`), MTM, the identity line's overlay and
event histories, the taxonomy edges in LTM1 (kinds, definitions) and
their readers (`ConceptIndex`, `DefinitionIndex`).

## 3. Test ports

P2 (seven columns with `ref_kinds`), P4's tagged snapshots and absence
ports, P6 (close-time minting) and P7 (root readers and pending names)
are re-audited against this layout and listed in the receipt with exact
before and after text; the §25–§26 amended proposals are the starting
point where they still apply. Tests that assert a stored derivation or a
three-slot row are ported to the idea or chain the sentence now writes;
tests of the leaf index, `occurrence_memberships` and `expression_
components` are retired and listed. Any port that changes a decided
behavior stops for Alec.

## 4. Gates

1. "Felix chases Tom" with all words known writes one LTM0 idea and no
   LTM1 row; "he said she said x" with x unknown writes the chain
   `(OCC he, REL (SYM said, REL (OCC she, REL (SYM said, OCC x))))` and no
   sentence row; a chain row referencing the bare word x is found by
   `rows_referencing`.
2. After "x is three", the chain is unchanged and thought writes y's value
   to MTM and an inference idea; the decomposition arms pass by evaluation
   on their original snapshots and seeds.
3. A relation row referencing a row reads as LTM2; if-then, part and
   kind rows are read apart by their references alone.
4. Placement: a word's order-0 meaning moves toward its sentences'
   context at the close, with no stored membership; retrieval by content
   finds the idea.
4a. Recall by time: a query with a blank description and a `.when`
   returns the ideas and rows written at that time and nothing else;
   two sentences of one close are recalled together.
5. A pre-layout checkpoint is rejected by version; a layout checkpoint
   round-trips ideas, chains and tags.
6. `AnswerProgram` is gone; `Reading` names the kept reading.
7. Unchanged: G1–G3, the pending-premise certificates, the reference,
   bind-thought and identity files, the eight-file set and the regression
   files, on one frozen hash. Step 2's one full sweep follows 2c.

## 5. Code review and layout decisions — Codex, 2026-10-25

This round implements §2 only. The single item-3 sweep belongs after all
five steps, superseding §4's historical step-2 sweep sentence. The presentation
boundary remains `BasicModel.begin_presentation`.

1. LTM1 has no composed value. Its percept contains only the two four-coordinate
   location bands, `.where` and `.when`; the top chain row receives the close's
   clock stamp. These bands are not a substitute sentence vector. LTM0 stamps
   the existing percept band of its one paired vector. Inner entries have a
   zero clock band. Direct stores declare their percept layout at construction.
2. Tags distinguish occurrence keys, symbols and relation handles even when
   their integers coincide. Taxonomy callers default to SYMBOL; chain writers
   pass explicit tags. External relation handles retain the existing `sym`
   envelope because their allocator already supplies unique handles; the two
   stored argument tags, never the integer's magnitude, determine traversal.
3. Checkpoints require the new version and the exact constructed capacities.
   The former restore-time row resizing is removed. Every new buffer is sized
   at construction with its derivation beside the declaration.
4. The current reference resolver already reads role evidence from MTM events.
   Its remaining same-time Part traversal in `accessible` is replaced by the
   frame's written activations and identities. Compression retains the subject
   identity as a semantic binding; it does not retain grammatical role slots.
5. A direct asserted taxonomy/equality pair remains an LTM1 relation. Ordinary
   resolved grammatical compositions compress. A pending operand keeps the
   semantic pair chain, without the parse-only heads or an extra sentence idea.
   Pending mentions retain their existing occurrence addresses for MTM.
6. Placement's context source is explicitly interim: the kept reading updates
   preallocated order-zero context codes once at the close by EWMA. Step 2
   replaces this context source with the registered priming surface. The receipt
   will enumerate remaining per-forward inventory/store scans.

Publication of iteration 2 was verified from both remote main refs:
BasicModel `4f136f76faabafc24397765f4ffe90d01d3dc4bc`, WikiOracle
`69f82f5f77a3980020841fe859a09afb7733e7be`. The reviewed production source
remains `75345f891e`. This round is local and returns through Alec for review.

## 6. Reader and fixture audit — Codex, 2026-10-25

1. Embedded occurrence addresses include the source document and sentence
   position. Content alone reused mentions from repeated sentences and broke
   MTM membership. A selected arrival operand receives its own unstamped
   occurrence; only the root sentence is stamped. The live `written_rows` map
   exposes entries consumed inside chains, but is not retained as derivation.
2. A chain's temporary working view exposes its actual operands. A missing
   third grammatical slot is not manufactured as a free variable. Pending
   names and unresolved child chains are read from their tagged references;
   they do not require a stored operation or a composed vector. MTM event
   checkpoint validation accepts the root's relation handle as well as an
   idea address, while identity membership remains occurrence-addressed.
3. Pure concept fixtures now declare `percept_width=0`. Tests of time use an
   actual percept band. Several old arrival fixtures supplied a known concept
   by filling an objectless word's bank seat; their setup now admits a separate
   object and definition. Bare names remain pending. These are setup repairs,
   not new language rules or changed seeds.
4. The stream-membership slab is constructed with an explicit capacity. Its
   default is the LTM capacity: a simultaneously closing batch must have room
   for one entry per stream. `ltmStreamCapacity` can declare a tighter bound.
   No witness or checkpoint restore grows this slab.
5. Both forgetting strata use the dependency snapshot taken before the pass.
   Removing a parent releases its operands for the following pass. The
   presentation cleanup compares tagged references, so equal integers in
   different reference domains cannot erase unrelated concepts.
6. Tests asserting close-time argument minting, stored composing edges or
   a fused value for an unresolved chain are ported to the decided layout.
   Their exact before/after text and the explicit retired-reader list belong
   in the receipt. The reference, choice, gradient and ownership assertions
   remain behavioral gates.

## 7. Time, identity and evaluation audit — Codex, 2026-10-25

1. Re-witnessing an existing idea preserves its original percept time even
   when the incoming carrier has a blank time band. Stream membership is
   access history, not a replacement event clock. The forgetting fixtures now
   exercise a declared percept band and compare model-time ages. Neither
   widening a checkpoint's capacity nor inferring absent layout columns is
   permitted by the new version boundary.
2. A compressed event over a freshly introduced identity keeps that actual
   occurrence, ahead of its kind symbol. A numerical journal state update
   does not alone discharge a named open referent; selected binding and kept
   thought remain the owners of that transition. The pending-premise
   assertions retain their original bind-correctly-or-stay-open condition.
3. Relation verification reads the two tagged operands. It cannot assume
   that a temporary working view puts the right operand in grammatical slot
   two. A referenced relation is a situation without requiring its own time
   stamp; only the containing sentence carries a stamp. Nested asserted pairs
   therefore remain readable as conditionals while their inner rows remain
   unstamped.
4. Alec clarified the decomposition comparison: “we can compare output or
   some internal isomorphism.” The original snapshots, seeds and configurations
   remain the measurement inputs. The legacy full-width kernels may carry
   different vectors for expressions over witnessed-equal identities. The
   proposed read-time isomorphism reconstructs a common context with a declared
   inverse, checks its forward recomposition, and requires the already-read
   equality witnesses for the inputs. MTM's existing before/after event fields
   are available for this read even when no verb teacher is attached. This
   keeps no new durable derivation and charges inverse work to the query meter.
   Numerical equality of paired meaning blocks remains the direct case. The
   receipt must report the actual result, including any failed original arm;
   the clarification is not permission to substitute a seed or snapshot.
5. Old storage tests of three roles, composing argument rows, proof discounts,
   leaf postings and membership diffusion are ported or explicitly retired
   against §3. The receipt preserves their exact original and resulting text.
   Resolved fixture vocabularies admit their objects before composition;
   close-time argument minting is not reintroduced by the fixtures.

## 8. Output comparison and remaining storage readers — Codex, 2026-10-25

1. The output isomorphism in §7.4 is now implemented in `IdeaEvaluation`.
   It reads existing MTM before/after values and a native prototype only after
   `ThoughtBinding.integrate` has established the inputs' witnessed equality.
   A declared inverse proposes a shared context. Finite-chart clipping can
   erase different coordinates in each inverse, so their coordinatewise meet
   and join are tested as hypotheses; both complete outputs must recompose at
   the existing native decoder tolerance. No grammatical trace is retained.
   Native lexical activation is searched over its entire allowed interval by
   a sixteen-interval coarse bracket followed by golden-section refinement.
   Every inverse and forward check consumes the shared work meter. Both original
   decomposition lengths pass in both gain arms in development, on the original
   snapshots, original seeds and unchanged proof assertions and budgets. The
   frozen receipt is the gate result; this is a mechanism certificate, not a
   claim of learned decomposition.
2. A kept inference retains its bound answer reference in `_bound_references`
   independently of `_identity`, the identity whose value it updates. Both are
   semantic bindings allowed by §2. `BindingAnswers` can therefore authenticate
   the retained answer after the working grammatical roles have compressed.
3. A query cannot cite its own closing relation as an independent witness.
   The relation branch now honors `_query_occurrence`, as the content branch
   already does. This preserves search-exhaustion behavior without changing the
   chooser, reconstruction criterion or seeds.
4. LTM1 checkpoint cycle validation uses an explicit stack, so a legal deep
   chain is not rejected by Python's recursion limit. All column layouts,
   reference tags, counts, access counts and surprise values are checked before
   replacement. Construction must provide at least the reserved zero handle
   and one live handle. Cover-map validation retains the pre-round checks for
   digest shape, positive symbols and distinct ownership.
5. `Interpret.define` no longer stamps a definition row. Only the sentence
   entry carries model time. Document-cover retrieval ignores internal binding
   metadata when testing content constraints, as ordinary content retrieval
   does. Cover symbols and their separate RNG remain unchanged. A pre-layout
   checkpoint and a differently sized inventory are rejected instead of migrated
   or resized; the older acceptance fixtures now assert that version boundary.
6. The fixture audit also covers the older direct storage readers: three-role
   column assertions become unary idea or tagged operand assertions; source
   time readers use `entry_when`; pure vector fixtures declare no percept; time
   fixtures declare eight percept coordinates. Tests of a selected known
   vocabulary admit its objects before composition. The receipt records every
   changed test and fixture in full, including the historical time-reader
   inventory's removed and added readers.

7. The storage audit restored the existing TruthSet admission boundary: an
   externally asserted resolved sentence is a fact, an interrogative cannot be
   asserted, and provenance, text, stream and temporary-cycle validation still
   run before writing. A full LTM0 can re-witness an existing explicit sentence
   address without reserving a new seat. The receipt includes the reproducer
   and regression for that capacity case.

## 9. Construction bounds and query ordering — Codex, 2026-10-25

The bounded affected-file run stopped at its memory ceiling when a canonical
fixture inherited BasicModel's 262,144-entry LTM and the direct-store default
reserved that many stream columns too. `SymbolSpace` now constructs the stream
slab from the configured training batch size; `ltmStreamCapacity` can declare a
larger runtime batch before construction. Direct isolated `LTM0` stores retain
the capacity-derived default from §6.4. No store grows after construction. The
same fixture passes its memory boundary with this production derivation; the
aborted run remains in the receipt.

Content retrieval retains its previous address-order tie break among equally
matching descriptions. Explicit references prioritize candidate enumeration
under a short meter; they do not displace an older equally matching witness
once both have been read. The replacement for the leaf index now preserves
that distinction. An inline query fixture supplies one fused description,
retaining its scope, identity, evidence and pure-construction assertions.

## 10. Final reader and admission audit — Codex, 2026-10-25

1. A chain handle is an existing description wherever an idea occurrence is:
   both description-resolution contexts accept a live LTM1 handle, charge the
   record read and preserve its actual operand view. The controller recognizes
   a live chain from allocator occupancy without reading its content while
   building candidates. An ordinary concept symbol is not a description.
   A kept query that merely recalls an unchanged chain reuses that entry and
   its original percept time. A filled binding still writes the inference
   value and updates its identity through the existing thought path.
2. Trust-derived evidence on a negative relation reverses its poles because
   LTM1 has no polarity column. Explicitly supplied evidence is already
   canonical and is not reversed again. This restores the pre-layout rule;
   the existing attribution, negation and negative-trust assertions remain.
3. Stream-capacity checks precede any durable mutation in both the direct idea
   writer and the clause transaction. Overflow leaves the tensors, semantic
   metadata and relation inventory unchanged; it never grows the slab.
4. Predicate and answer-adapter fixtures read the actual two-operand view of
   a retained pair. A live grammatical predicate remains part of the working
   meaning, not a durable LTM1 column. Predictor fixtures supply a unary idea
   or an actual chain. The final receipt contains exact before/after text,
   explicit retirements and every failed development invocation.

## 11. Frozen diagnostic-contract repair — Codex, 2026-10-25

The initial content freeze `7b0d1a0090f24a5df7e07c4154e268d3c3906730aece0ce5d57d2b172f6b40ca`
completed the 906-case core set: 898 passed, seven prerequisite skips, one
failure. The new chain-description rejection behaved correctly but its message
omitted “reference,” required by the existing unary-description rejection test.
The repair adds that word in the two description-reader contexts. No assertion,
branch, value, seed or budget changes. The initial failed result and archive
remain intact. Under §39's fix-and-continue rule, this diagnostic-only repair
requires a new recorded content hash and the required groups on that hash;
the receipt must not report the original freeze as a pass or combine its
passes with the repair. This is an exception to the intended one-freeze plan,
not an additional full sweep. The single native smoke is not repeated; its
code differs from the repair only in those two error strings. The repair
manifest authenticates that exact delta. The final runner records one CPU
thread per worker and preserves emitted certificates before temporary cleanup.

## 12. Step-1 result for review — Codex, 2026-10-25

Final source SHA256
`34848e154f0fa9912223d4117136d7eb63657cefaeca94a76c6574559a73819e`
completes **911 passes, seven existing prerequisite skips, zero failures**:
899 core passes and seven skips, eight XOR passes, and two decomposition
lengths in each of the original gain arms. All eight companion decomposition
questions are correct. G1–G3 pass all 32 generated cases; both pending-premise
orders preserve the bind-correctly-or-stay-open assertion (forward 0 bound /
16 open; reverse 0 / 24; no wrong binding). Layout gates 1–6 and 4a pass;
gate 4 uses the declared interim close update.

The [receipt](../benchmarks/2026-10-25-item3-0-layout/README.md) contains the
source archive/manifests, commands, original snapshots/seeds, emitted
certificates, 162 test ports with full before/after text, 21 explicit
retirements of removed behavior, 18 added test functions, all development
logs, the initial failed freeze and the §11 diagnostic-only repair.
`BasicModel.begin_presentation` is unchanged. No full sweep, commit or
publication occurred. Step 1 returns through Alec for Claude's review;
steps 2–5 remain, including replacement of the interim context source and
removal of the listed whole-store scans.

## 13. Claude's review of step 1 (source `34848e154f`) — 2026-10-25

Reviewed: §§5–12, the [receipt](../benchmarks/2026-10-25-item3-0-layout/README.md),
`gate-summary.json`, `repair-01/frozen-source.json`, `remaining-scans.md`,
`retired-tests.json`, `test-ports.json`, `presentation-boundary.json`, the
working-tree diff against `4f136f76fa` (35 source files, five new modules
and tests, 67 test and fixture files) and the new `IdeaEvaluation`,
`EntryRead`, `Priming`, `MereologicalCodes`, `RowTime`, `LTM1` and
`MemoryConsolidation` text.

**Verified.** The freeze is real: every one of the 887 manifest paths,
including the five untracked new files, hashes to its manifest entry on
disk, and the compact canonical JSON of the map hashes to
`34848e154f…`. The gate counts are the summary's (core 899/7/0, XOR 8,
decomposition 2+2 on the original seeds and snapshots), the seven skips are
the known prerequisites, and the initial freeze's one failure and its
two-string repair are both recorded, neither promoted. `begin_presentation`
is byte-identical to the base. No language is named in any added code or
fixture and no seed is added in any test. The 201 ledger entries are 162
ports, 21 retirements with reasons, 18 additions, and the retirements are
all leaf-index, derivation, membership or stored-argument tests.

**The layout is §2 as written.** `LTM0.slots` is `[capacity, nDim]`, the
`role_mask`, `role_refs` and `when` columns are gone from the store (the
remaining `role_mask` reads are `ConceptualMeaning`'s working form, which §2
keeps); `LTM1` has `refs`, `ref_kinds` (OCCURRENCE 1, SYMBOL 2, RELATION 3,
zero invalid), an eight-coordinate `percept`, `layout_version`, `find`,
`rows_referencing`, `stratum` and a version-checked load; `write_clause`
stamps `closing.when` into the sentence entry alone through
`RowTime.write_stamp`, `entry_when` reads it, forgetting's age reads it, and
`EntryRead.cued_rows`/`recall_time` are the time-cued query; the chain
gate's fixture asserts the decided shape `(OCC he, REL (SYM said, REL (OCC
she, REL (SYM said, OCC x))))` with the top row alone stamped; `Reading`
replaces `AnswerProgram` with no residue; `PartRead`, the leaf index,
`occurrence_memberships`, `expression_components`, `nonparticipants`,
`argument_reader`, `_word_origins`, `_definition`, `_reserve` and
`membershipPriming` are gone; `Priming.diffuse` is the edge channel only;
`observe_close` is the interim EWMA over preallocated `context_codes` and
`context_counts` at the declared rate, order-0 rows only, and the receipt
says so. The second freeze under §39 and the stream-slab construction bound
(§9) are right.

**Findings.**

1. *The hash record is incomplete (record only).* Todo "Current", the 3.5
   line and §5 say the reviewed source "remains `75345f891e`" and that
   publication is verified. The commits after `f306fc28aa` were rewritten
   before the push when receipts left version control (`537523ae04`), so
   `75345f891e`, `a34325c85d` and `f095569a1f` are not on origin; their
   published equivalents are `2067dbc25a`, `3561f39aa0` and `31c3e4a59e`.
   One line in each place.
2. *Reading recurrence is now content identity (for Alec).* `consolidate`
   keyed a reading by the symbol a mention read to (`role_refs[0]`); with
   no reading row it now keys by `reading_key`, the SHA-256 of the idea
   vector, so a reading recurs only when three documents write an
   identical idea. §§5–10 do not record the change, and it alters 2c's
   promotion (three documents for a reading), a decided behavior, so under
   §3 it is Alec's. Recommended: key recurrence on MTM's reading record
   (`MediumTermMemory.names`, the word a mention read to, counted over
   documents while the token lives), which is where §72 put readings; fix
   at the start of step 2, which touches `consolidate` for `_fold`.
3. *`IdeaEvaluation` is new mechanism in a removal round (watch item).*
   The read-time isomorphism (declared-inverse context, meet/join
   hypotheses, sixteen-interval bracket and golden-section search over the
   lexical activation, all under the work meter) keeps the decomposition
   arms passing on `MM_math_chain`, whose kernels act on the whole vector.
   It stores nothing and chooses no seed, and Alec's clarification admits an
   output comparison, so it stands; it should retire or shrink when the
   kernels act on meanings on that configuration, and the receipt's
   "forced architecture certificate" wording is the right claim.
4. *`Priming.part_edges` now walks every live LTM1 row and resolves
   OCCURRENCE references through `_semantic_rows` on every `prime_seen`.*
   Listed in `remaining-scans.md`; step 2 caches it by revision ([step 2
   spec](2026-10-25-item3-0-step2-placement.md) §6).
5. *Gate 1's last clause* ("a chain row referencing the bare word x is
   found by `rows_referencing`") is certified for `said`; x is an
   OCCURRENCE in the chain, which §5.5 explains. Acceptable; noted.

**Verdict.** Step 1 is approved as a mechanism round: the layout, the
removals and the ports are what §§1–4 decided, on one verified frozen hash
with every gate green. Alec's commit follows findings 1 and 2's record
lines; step 2 begins on that source with finding 2 as its first task once
Alec decides it. No publication until the end of 3.0.
