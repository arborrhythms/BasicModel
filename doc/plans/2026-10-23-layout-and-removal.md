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
"a column, never an operand of composition". So the stamp is appended,
as Alec says, not written into a band: an STM row is the concept vector
with the current `RowTime` appended as its `when`, the same two parts as
an LTM0 row (`slots[cap, D]` beside `when[cap, 4]`), and the column
stays. A concept pulled from the index has no time and is stamped with
the current `RowTime` on entry, and a recalled entry keeps its own stamp,
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
