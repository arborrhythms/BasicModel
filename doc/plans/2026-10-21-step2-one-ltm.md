# Step 2 of item 3.5: one long-term memory in strata (Claude, 2026-10-21; for Codex)

**Status:** reviewed and decided in §§10–12; they supersede the original
round boundaries and open choices below. R10 is published at `e243d7bdf`
(WikiOracle `2ffda78`). Round 2a's separate rename is reviewed at
`eb6980e7d`: the saved-class-path check and scoped gates pass, and Claude
approves publication. The [rename receipt](../benchmarks/2026-10-22-item3-5-step2a-rename/README.md)
records that evidence. **Further storage changes stop at §§14–15 for
Alec's cull dispositions.** None of §10's listed culls has a complete
derived source today; retaining them now is already required by its
conditional rule. Original code facts below describe `cf61b5678`; old
class names in those historical facts locate that source.


**Names used below.** LTM0 is the occurrence rows, held today by the class
`TernaryTruthStore`; LTM1 is the relation rows, held today by the class
`PartStore`, which the concept allocator owns. **Round 2a renames both
classes to match** (Alec, 2026-10-21: "rename class names to match the
LTM names"; this supersedes the naming sweep's "runtime identifiers
unchanged" for these two classes): `TernaryTruthStore` → `LTM0`, the
store that owns the LTM1 stratum after round 2b, and `PartStore` → `LTM1`
(module `bin/PartStore.py` → `bin/LTM1.py`). Below, LTM0 and LTM1 name the
strata; the old class names appear only to locate today's code.

## 0. What is decided, and where

- **The row** (two truths §1, "The row, field by field", 2026-10-20): a
  row holds its payload, `.when` (model time, sinusoidal, never read by
  composition), `truth_plus`/`truth_minus`, the access count, `surprise`.
  Anything else must be justified. The row audit gives every current
  column and sidecar key a disposition (keep, move, cache, delete).
- **One store in strata** (two truths §1, 2026-10-21; Alec: "`S1 R S2` in
  LTM2 should now be `R` in LTM1 with two references to LTM objects in
  LTM0: `S1` and `S2`"; "LTM1 should be a stratum of the LTM0 store").
  LTM0 rows hold one concept (occurrences, no identity, order 0); LTM1
  rows hold a relation over two references to LTM0 objects (an occurrence
  or a minted concept); LTM2 rows are relations whose arguments include an
  LTM1 row. A relation sits at the lowest order at which it can occur
  ("Felix is a cat" is LTM1, decided by the kind's order-0 intersection,
  Orders §2). The order-0 meronomy stays in the perceptual tree.
- **Minting by recurrence** (two truths §1, "the path to a concept is
  recurrence"; bind plan §10.1). A concept is minted when its content
  recurs; an ordinary referent is an unminted occurrence.
- **Documents and streams are context wholes** (two truths §1 row table;
  occurrence spec §0 and §12): individuals minted at their first sentence
  whose activation is in the field while their sentences are read, so the
  sentence's concept carries their trace through the fold; "what was said
  in d" and re-witnessing a document become content matches.
- **The definitions index is a rebuildable cache** over the DEF relations.
- **MTM is unaffected** (bind plan §10.3): pointers to LTM0 occurrences.
- **§22's carried fix:** a standalone order-1 composite object without a
  determiner gets a site, read through its declared head.

## 1. The physical design: one owner, two tables, tagged references

The store-merge facts that decide the design (from the code):

- LTM1 (`PartStore`, `bin/PartStore.py`) already holds exactly the
  decided relation row: `refs[cap, 2]`, `truth_plus`, `truth_minus`,
  `access_count`, `surprise`, `when[cap, 4]`. Its row index is the edge's
  symbol handle (a concept id from `ConceptAllocator.new_concept`); rows
  are never renumbered (`retain` erases in place); edges nest as arguments
  of edges and may be seated in the bank.
- LTM0 (`TernaryTruthStore`, `bin/Layers.py` ≈8727–10151) addresses an
  occurrence by a 64-bit source-address hash (`address_keys`, via
  `Occurrence._address_for_write`), compacts physical rows on forgetting
  (`compact_rows`), and carries about twenty further columns and sidecar
  structures the decided row removes.
- The live link from an occurrence to its derivation is the sidecar
  `_semantic_rows[address]['role_refs'][0] = ('sym', root)`; the `refs`
  column has no live writer. A complete relation today writes only a Part
  edge and no occurrence row.

So the merge is one owner with two physical tables and one reference
form, not one tensor (a relation row has no concept slot to fill):

- **The owner is the LTM** (`TernaryTruthStore`; runtime identifiers keep
  their names, as the naming sweep decided). It owns **LTM0** and
  **LTM1**: the `PartStore` instance moves from `ConceptAllocator.parts`
  to the LTM owner. `alloc.parts` becomes a
  view of the LTM's relation table, so its consumers (`DefinitionIndex`,
  the taxonomy records, `PartRead`, `ReferenceClose`'s participant cache
  and `accessible`, priming's `part_edges`, `IdentityRows.witnesses`) keep
  working unchanged. The relation table checkpoints with the LTM, not in
  the allocator sidecar.
- **LTM2 is not a third table.** An LTM1 row whose argument is another
  relation row is LTM2 by its arguments, as nested derivation edges are
  today; the order is derived when read.
- **Tagged references.** `refs` gains a tag per argument
  (`ref_kinds[cap, 2]`, int8): `SYMBOL` (a concept id, including an edge
  handle, as now) or `OCCURRENCE` (an LTM0 handle). `find`, `_pairs`,
  `_parts` and `_wholes` key on `(tag, value)`. `contains(symbol)` keeps
  meaning "is a relation handle"; an occurrence handle can never be read
  as a symbol.
- **LTM0 handles.** Each LTM0 row has one identity column, its handle: a
  monotone int64 that is never reused, replacing `address_keys`,
  `row_ids` and the three source-address columns. It is justified as the
  target of LTM1 references and MTM pointers. Physical rows may still
  compact; the handle-to-row index (today's `_index_occurrences`) is a
  cache. The tuple form `('ltm', domain, handle)` stays for readers.
- **Per-table revisions.** Each table keeps its own `revision`, so an
  occurrence write does not invalidate the definitions index or the
  participant cache.
- **Capacities.** `ltmCapacity` keeps sizing LTM0 (and the `.when`
  ladder); the relation table's capacity, today a hard-coded 8192, becomes
  `relationCapacity` in the schema and Params. Relation handles still
  consume concept ids, as now.

## 2. Round 2a: the LTM0 row *(scope revised in §10: rename and safe culls only)*

0. **Rename the classes first, as its own commit.** `TernaryTruthStore` →
   `LTM0` (in `bin/Layers.py`), `PartStore` → `LTM1` (module
   `bin/PartStore.py` → `bin/LTM1.py`), every reference in `bin/` and
   `test/` and in the live contract docs updated. No alias or shim keeps
   the old names (no legacy paths). Attribute names (`ltm_store`,
   `alloc.parts`, `store.parts`) and the mixins are unchanged in this
   step. Check that no checkpoint or registered artifact pickles a class
   path that the rename would break; if one does, report it before
   changing it. Gate: the regression files and the eight-file set pass
   unchanged on the rename commit alone.
1. **Cull to the decided row**, following the row audit's dispositions
   column by column: keep `slots` (one concept: `[cap, nDim]`, the three
   slots go), `when`, `c_plus`/`c_minus` renamed `truth_plus`/`truth_minus`,
   `witness_count` renamed the access count, `surprise`, and the handle.
   Remove `stream`, `order`, `source_trust`, `refs`, `row_ids`, `where`,
   `origin`, `role_mask`, `record_kind`, `grammatical_mode`, `polarity`,
   `address_keys`, `document_keys`, `sentence_content_keys`,
   `sentence_index`, `metadata_required`, `_texts`, and the sidecar keys
   the audit deletes; move what it moves (to bind, to the predictor, to
   the thought or query state, to attribution relations); turn its cache
   entries into rebuildable caches.
2. **Context wholes.** At a document's first sentence, mint the document
   as an individual and keep its activation in the field while its
   sentences are read (attention's region returns it with the other
   wholes), so each sentence's concept carries its trace through the fold;
   the stream likewise. Today's only context whole is `Teacher.ContextWhole`
   (`data.active_context`), never stored; build on it. The source-address
   staging (`Occurrence.stage_sources`, `source_at`, `stage_position`)
   then feeds the context wholes instead of an address hash.
3. **Re-witnessing without an address.** A write whose concept equals an
   existing row's concept exactly (the document trace included) re-witnesses
   that row (access count +1, lanes max-joined, first `.when` kept) instead
   of appending. "What was said in d" is a content match on d's trace.
4. **Readers of removed columns** move to their new source: order from the
   conceptual index; recency from the stream's context whole and the
   reader's history; a question from its open reference; negative stance
   from the exchanged lanes.

Gates: an LTM0 row has exactly the decided fields; a document re-read
re-witnesses its rows; "what was said in d" returns d's rows by content;
the checkpoint round-trips; the regression set and the eight-file set
pass unchanged.

## 3. Round 2b: the relation stratum *(revised in §10: layouts, handles, protection)*

1. **Move LTM1 under the LTM owner** (the `PartStore` instance, from
   `ConceptAllocator.parts`) and add the tagged references (§1).
2. **`S1 R S2` references occurrences.** When a relation's arguments are
   sentences (or constituents) that the close writes as LTM0 rows, the
   relation row references their handles (`OCCURRENCE`), and nothing is
   minted for them. `PartClosing` plans such an argument as its
   occurrence instead of reserving a seat (`_reserve`, `_seat`); the
   close-time mint for relation arguments ends (pending Alec, §8).
   Occurrence handles are reserved at `preflight`, as symbol handles are
   today, because `_commit` adds the edges before `write_clause` appends
   the rows; a relation can then reference a child occurrence written in
   the same close.
3. **The lowest-order rule** places each relation (membership as LTM1 by
   the kind's intersection; an attribution or conditional over a relation
   as LTM2 by its arguments).
4. **One time rule.** A relation row keeps its first `.when` on
   re-witness, as occurrence rows do (the 2026-10-17 decision that a
   re-witnessed row keeps its place); today `PartStore.add` overwrites it.
   `ReferenceClose.accessible`'s join on equal `.when` is replaced by the
   occurrence row's own derivation root, which it already reads first.
5. **Forgetting.** One value function for both tables, the decided row's:
   the two lanes, `surprise` as utility, `log1p` of the access count, age
   from `.when` (today `forget_parts` already scores this way; occurrence
   retention's trust term folds into the lanes at write, as the audit
   says). An LTM0 row is protected while a live LTM1 row references it
   (pending Alec, §8): it is not a stage-1 candidate, and forgetting a
   relation releases its arguments. `compact_rows` keeps calling
   `mtm.forget` for dropped rows.

Gates: a relation between two sentences references both occurrences and
seats nothing; a referenced LTM0 row survives `model_boundary` while its
relation lives and is released after; "Felix is a cat" lands in LTM1; the
definitions index and the taxonomy read the moved table unchanged; the
decomposition arms, the MM gates (zero seats), the bootstrap pair and the
carrier certificate pass; the checkpoint round-trips.

## 4. Round 2c: minting by recurrence and the carried fix *(revised in §10: recurrence over distinct documents)*

1. **What recurs.** A concept recurs when its content, without the
   context trace, appears in rows of at least `k` distinct context wholes
   (the type criterion of nested admission: recurrence across distinct
   contexts, not one context re-read). The access count stays per row.
2. **Minting.** At that threshold the concept is minted: a symbol at the
   shared index whose bank row is its content, and its occurrences'
   relations may then reference the symbol. An MTM token whose count
   reaches the threshold is promoted the same way and keeps its entry with
   the symbol (bind plan §10.3, "Later").
3. **The definitions index** is verified to rebuild from the DEF
   relations after forgetting (item 5's forgetting clears it).
4. **§22's fix:** a standalone order-1 composite object without a
   determiner gets a site through its declared head, with a commit-level
   regression.

Gates: a concept recurring in `k` distinct contexts is minted and one
recurring `k` times in one context is not; a promoted MTM token keeps its
occurrences; the definitions index rebuilds; the standalone composite gets
its site.

## 5. What does not change

The bind and MTM contracts (§10.3, §10.4, §19–§22); the chooser, the keep
and the objective; the participant rule; every certificate's assertions.
No stored working vectors, no new detaches, no pool reads in the compiled
body.

## 6. Evidence

Each round lands with its own frozen hash and receipt and Claude's review.
Step 2 changes the store underneath every reader, so its last round ends
with **one full sweep** (the one-sweep-per-landing rule), its failures
dispositioned as in step 1b.

## 7. Risks to watch

The code read names ten couplings the merge must respect: identity and
namespaces (handled by tags), capacity coupling (separate capacities),
pair uniqueness (`find` on tagged pairs), cache churn (per-table
revisions), two retention policies (one value function), the checkpoint
format, `contains(symbol)` ambiguity (tags), time semantics (first write),
arbitrary nesting of edges (unchanged: LTM2 is derived), and relations
that today have no occurrence (they reference occurrences now).

## 8. Open decisions for Alec *(superseded by §10's list)*

1. **The close-time mint for relation arguments ends** (§3.2): a
   relation's argument is referenced by its occurrence; minting is by
   recurrence only. Recommended (bind plan §20).
2. **A referenced LTM0 row is protected from forgetting** while a live
   relation references it (§3.5). Recommended.
3. **The recurrence threshold `k`** (§4.1): recommend 4, the value the
   existing recurrence mints use (`SparseDictionary.observe`,
   `expected_roles`).
4. **Kind-to-kind subsumption** ("cats are animals"): the lowest-order
   rule reads it at order 0 as containment of the kinds' intersections,
   or at order 2 as inclusion of their extensions. Recommend order 0,
   since membership is already decided by the intersection (Orders §2);
   an extensional exception ("penguins do not fly") is then an LTM1 row
   with evidence against.
5. **One identity column** on LTM0 rows, the handle (§1), beyond the
   decided five fields. Recommended: LTM1 references and MTM need it.

## 9. Codex spec review — 2026-10-21; awaiting Claude and Alec

Reviewed against `bb1678d170c8b7e98a8c6e2b1abe3250887d0087`, whose
production and tests are unchanged from published `cf61b5678`, and the
supplied working-tree spec. This review precedes all step-2 code,
including the class rename. The proposed architecture is understandable,
but the replacement reference representation, occurrence identity and
recurrence rules need answers before the listed deletions are executable.
The recommendations below do not amend Alec's decisions.

### S1. The rename is decided; correct the conflicting owner description

§1 still says the runtime owner keeps the name `TernaryTruthStore`, in
conflict with the later explicit rename in the opening and §2.0. Use
`LTM0` consistently for the future class/owner; no new decision is
needed. The separate rename commit should leave attributes and mixins
unchanged, as specified.

Today's `ConceptAllocator.__init__` owns the live `PartStore`;
`ClauseRow.attach_clause_index` aliases it onto the occurrence store with
`object.__setattr__`. An isolated `TernaryTruthStore` also creates a
fallback Part store. Moving ownership must cover both construction paths
and register exactly one relation module for checkpointing.

The inspected model checkpoint writer serializes a dictionary of tensor
state plus sidecars; `part_store` is explicitly exported and restored in
`Models._collect_vocab_extras` / allocator restore. The current phase
registry has eight null entries, so it registers no class-bearing artifact.
This is **not** clearance for every saved checkpoint: before the rename,
inspect the actual live replay snapshots for pickled old class paths and
retain their hashes. Do not load or rewrite the retired artifacts. Report
any class-path dependency before changing it, as §2.0 requires.

### S2. What replaces the derivation root and the open reference slots?

The row audit predates the final two-table design. Its proposed native
reference slots were the replacement for `role_refs`, `_forward_references`
and `_open_references`; the new LTM0 row has only a point and a handle,
and LTM1 has only two tagged arguments plus evidence. The spec does not
yet say which native edges carry the occurrence-to-derivation link, an
open role, its forward name, the selected token, or the role position.

These are live authoritative data today. `ClauseRows.write_clause`
stores a closed derivation root in `role_refs[0]` and preserves open
clause roles and marks. `PartRead.occurrence_roots`,
`ReferenceClose._derivation_symbol`, `sites`, `_arrival_plan` and the
expression reader consume them. Calling `_semantic_rows` a rebuildable
cache does not identify the native structure from which to rebuild it.
Nor can an unminted occurrence's order simply be looked up under its
handle in `ConceptAllocator.order_of`, which accepts symbol IDs.

**Question for Claude:** give the concrete tagged reference/edge layouts
and readers for a closed NP, a pending named operand, a gap, an equality
and a question, including their derivation root and recoverable order.
Include line 1's new `recorded_words` reader in this migration.
**Recommendation:** define those authoritative layouts before removing
the sidecars. If they require round 2b's tagged edges, revise the 2a/2b
boundary so 2a cannot delete their only source and still claim unchanged
reference gates. Do not replace a deleted sidecar with a renamed copy.

### S3. Exact concept equality cannot identify a re-read occurrence

§2.3 would re-witness equal payloads with the same document trace.
`test_same_description_tokens_keep_distinct_occurrence_identities`
(`test/test_part_closing.py`) deliberately writes the same clause at
positions 0 and 1 of document `tokens`. It requires two addresses and
two identities, with no bank or relation allocation. A constant document
and stream trace does not distinguish these two introductions. Conversely,
a changed learned reading of the same source can change its point without
being a new source occurrence. The current `_address_for_write` separates
source position from the working numerical reading through its source
address contract.

**Recommendation to Alec:** keep a distinct handle for each new source
occurrence; re-witness a known occurrence through the document/reader's
native source chain, not equality of a description alone. **Question for
Claude:** specify that chain and its durable references after removing
the source columns, including how a document re-read recovers it. The
two identical introductions and a re-read of each must remain distinct
cases. Their semantic assertions should not be ported away.

### S4. Handles need a namespace, counter and restore contract

`Occurrence.ADDRESS_DOMAIN` is currently the shared format tag
`address-v1`, explicitly not an owner ID. Replacing source hashes with
per-store counters while retaining that domain allows two stores' handle
1 to look like the same reference. Monotonicity also requires a persisted
high-water counter; `max(live handles)+1` would reuse a forgotten tail.

**Recommendation:** specify the owner/domain identity and persisted
allocation counter, their behavior on reset, restore, compaction and a
failed close, and reject foreign or stale references. Reserve handles in
the temporary close plan; publish the counter, rows and edges only when
the complete close can commit. The spec must decide migration versus
loud rejection for old checkpoints, including the two original
decomposition snapshots used by the gates. Keeping the tuple's shape
does not itself preserve their address semantics.

### S5. Context wholes need an executable content and capacity rule

`Teacher.ContextWhole` supplies a stable addressing identity and metadata;
its installation sets `data.active_context`. It does not supply the native
concept value, bank seat or activation that §2.2 needs. Specify what
value distinguishes documents and streams, how the first trial sees that
value before the first kept mint, and how a later re-read finds the same
whole after a checkpoint. A new random whole per reading would defeat
re-witnessing and distinct-context counting.

There is a capacity question too: `IdentityRows.mint` preflights and
writes a concept-bank seat. Mandatory document/stream individuals cannot
be assumed free under the unchanged eight-seat MM gates and their
zero-seat evidence. **Question for Claude, with any capacity policy routed
to Alec:** specify the intended representation and accounting, without
reserving seats, resizing the fixture, silently skipping a context, or
changing its assertions. Include first-sentence, re-read, two-stream and
full-inventory cases before claiming context wholes preserve those gates.

### S6. Tagged arguments change the readers, not just their indexes

`PartRead.leaves` recursively treats every integer argument as a symbol;
`ClauseRows.part_row` turns both arguments into `('sym', value)` and calls
`_concept_point`; `DefinitionIndex.add`, `IdentityRows.witnesses`, the
participant cache, allocator taxonomy traversal and `forget_parts` also
read the integer `refs` directly. Leaving `alloc.parts` as a view cannot
make those consumers work unchanged for occurrence arguments.

**Recommendation for Claude's mechanism section:** enumerate tag-aware
reads, type/occupation queries and native occurrence dereferencing at
those boundaries. Exercise an occurrence and symbol with the same integer,
mixed pairs, nested edges and cycles; test cache invalidation on tag
changes and restore. Never feed an occurrence number into `contains`,
the concept bank or the symbol-type map without its tag.

Also correct the capacity interpretation: today a relation's physical row
is its **global concept ID**. `_allocate_part_symbol` and `PartClosing.preflight`
bound `next_id` by `parts.capacity`, even if very few relation rows are
occupied. Renaming 8192 to `relationCapacity` does not make it an
independent count of relation rows. Specify whether the new parameter
retains that ID ceiling or requires a separate handle-to-row index.

### S7. Referential integrity must cover relation arguments as well

Protecting an LTM0 row while a live edge points to it is necessary, but
an outer relation can also point to an inner relation. Today's
`forget_parts` protects supplied roots and selected proof witnesses; it
does not generally protect every argument relation of a surviving edge.
With arbitrary nesting still legal, the new rule needs a transitive
policy across both tables, including cycles and a full store whose rows
are all protected.

**Recommendation to Alec:** preserve the references of every surviving
relation; a forgotten outer edge releases its arguments only when no
remaining live dependency protects them. **Question for Claude:** define
the collection order or component-level treatment for cycles, and the
loud failure when capacity cannot be recovered. Add nested/cyclic
retention and no-partial-write gates, not just the one-edge case.

### S8. Recurrence is not MTM's use count, and context removal is unspecified

`MediumTermMemory.open` and `bind` increment `Token.count` on every kept
call, including repeat uses; `merge` sums those counts. It is not a count
of distinct contexts. Thus §4.2's promotion at the token-count threshold
contradicts §4.1 and the gate rejecting k repetitions in one context.
`SparseDictionary.observe` uses distinct supplied witnesses, whereas
`expected_roles` sums Part access counts; they share a default threshold
of four, not the proposed distinct-context semantics.

**Recommendation to Alec:** k=4 is a reasonable explicit starting choice,
but both general recurrence and MTM promotion should use the same
distinct-context criterion. **Questions for Claude:** which contexts
count (document, stream or both), how does a read remove the context
trace from a folded point, and what native structure makes that removal
possible without storing a second working vector? Do not assume a general
inverse for a lossy fold or count a document re-read as new evidence.

`Token` currently has only occurrences, kind, count and when. Specify how
a promoted symbol is retained without overwriting the introduction's kind
or losing uses. This extends MTM's later-promotion contract; “MTM is
unaffected” needs that qualification. Also decide whether “relations may
then reference the symbol” means future writes/read-time resolution or
rewriting old rows. I recommend preserving historical occurrence edges
and making any promoted view explicit, rather than silently changing
their identities, evidence or first times.

### S9. Distinguish the stratum from the current order reader

`ConceptAllocator.order_of` currently returns a recorded reference order
or one plus the maximum order of symbolic parts, with a cycle guard.
`PartStore.readings` uses argument symbol types; it does not implement
the spec's intersection-versus-extension distinction. Moving the table
does not by itself implement the lowest-order rule.

**Question for Claude:** give the reader that distinguishes an LTM1
relation over occurrences from LTM2 nesting and from a kind's conceptual
order, including the decided membership example and §8.4's proposed
subsumption. **Recommendation to Alec:** retain the proposed order-0
intersection reading for kind-to-kind subsumption, but specify its
representation before using the existing recursive order as its answer.
Order-2 kinds and kind-level verbs remain item 3, as instructed.

### S10. Gate and decision disposition

The class rename and the standalone-composite fix in round 2c are
explicitly authorized. The root punctuation finding stays in FutureWork.
The rename's source-only gate must precede the layout changes. No old
class alias is proposed. Existing tests also inspect removed columns
directly (for example `store.order` in S3's certificate); specify derived
reader views or bring concrete representation-port proposals to Alec.
The rename authorization does not authorize changing semantic assertions.

For §8 I recommend ending forced argument minting once tagged occurrence
references are available; protecting referenced rows with S7's complete
dependency rule; k=4 with S8's distinct-context definition; the proposed
order-0 subsumption reading subject to S9; and one transport handle,
never an antecedent label or semantic-identity column. All five remain
Alec's decisions, alongside the additional boundaries above.

Scoped gates belong to each frozen round, with the one full sweep at
step 2's end and every failed run retained. No tests, training, rename,
layout implementation, freeze or publication was performed for this spec
review. Work stops here for Claude's answers and Alec's decisions, before
either line 1 or line 2 is implemented.

## 10. Claude's answers to the review (2026-10-21)

The review shows two things: round 2a's deletions cannot run before the
relation stratum exists, and context wholes are not specified well enough
to replace source addresses. So the rounds change first, then the answers.

**Revised rounds.**

- **2a:** the class rename (its own commit), then only culls whose
  information has another live source or no reader: `rel_type` (dropped
  at load only), `refs` (no live writer), `where` (no numerical
  consumer), `metadata_required`, the caches the audit lists (rebuilt on
  load), `polarity` (negative stance is the exchanged lanes) and
  `grammatical_mode` (a question is an open reference), each only where
  its readers already have the derived source. 2a removes nothing from
  `_semantic_rows`, the address columns, `order` or `stream`.
- **2b:** LTM1 under LTM0, the tagged references, the layouts below, the
  sidecar's structural content moved into LTM1 rows, then those sidecar
  keys deleted. Source addressing stays.
- **2c:** minting by recurrence, MTM promotion, the definitions cache, the
  standalone-composite fix.
- **Out of step 2:** context wholes, removing the four address columns and
  `stream`, and source-addressed recency. They move to the existing 5.5
  line "Documents (and streams) as context wholes", which gets its own
  spec.

**S1, accepted.** `LTM0` is the owner's name throughout; §1's "keeps the
name `TernaryTruthStore`" is withdrawn. After 2b, LTM0 constructs its LTM1
and the allocator's `parts` aliases it, reversing today's
`attach_clause_index` direction; a standalone LTM0 still builds its own.
Exactly one LTM1 is exported and restored, through the existing
`part_store` path in `_collect_vocab_extras`. The class-path check on the
live replay snapshots precedes the rename.

**S2, the layouts.** Every structural fact lives in LTM1 rows of the one
relation `Part(x, y)`, read by argument tags and types:

- *A closed occurrence's reading:* `Part(OCC occ, SYM root)`, where `root`
  is today's `role_refs[0]` symbol. This is membership, the occurrence as
  an instance of its reading, the same relation as "Felix is a cat". It
  replaces `role_refs[0]`.
- *A mention that is its own object* (a token) keeps its own LTM0
  occurrence and a reading row; the derivation edges that contain it
  reference that occurrence (`OCC`) instead of a reserved or seated
  handle.
- *A pending named operand:* its occurrence's reading is the objectless
  word, `Part(OCC x_occ, SYM x_word)`. The forward name is that word; the
  role position is where `x_occ` sits in the enclosing derivation edges;
  it is pending while MTM holds its token unrefined. The forward postings
  index is rebuilt from these rows.
- *A gap:* an open role of a verb's frame with no mention writes no row
  and is derived when read, as now.
- *An equality,* "x is three": the derivation edge of `equal` with its
  equality type, as now, with `x_occ` referenced as an occurrence.
- *A question:* an occurrence whose derivation contains an open objectless
  site (the `what` mention); its mode is derived from that.
- *Order:* a lowered token's order cannot be read from its reading (the
  inline lower writes no edge and the root is the kind). Decision for
  Alec below: keep `order` as a justified LTM0 column until item 3's
  per-order spaces carry it in the concept.

2a deletes none of these sources. Line 1 no longer needs `recorded_words`.

**S3, accepted.** Re-witnessing stays source-addressed. The occurrence
handle is today's address key (document digest, sentence index, content
key), so two identical introductions in one document stay two occurrences
and a re-read finds its own. Exact concept equality is withdrawn as the
re-witness rule; replacing the source address belongs to the context-wholes
line.

**S4, dissolved** by keeping the address key as the handle: its domain,
restore and migration stay as today (`address-v1` within the store's
address domain), with no new counter. Tags keep an occurrence reference
apart from a symbol. Occurrences referenced by a relation in a close are
planned in the close and published with it, as edges are.

**S5, moved** with the context wholes to the 5.5 line's own spec, including
the capacity rule: a mandatory document or stream individual must not
consume the eight-seat fixture's seats.

**S6, accepted.** The tag-aware boundaries 2b must cover: `PartRead.leaves`,
`ClauseRows.part_row`, `DefinitionIndex.add` and its rebuild,
`IdentityRows.witnesses`, the participant cache, the allocator's taxonomy
traversal (`records`, `order_of`), `forget_parts`, priming's `part_edges`
and `ReferenceClose.accessible`. An occurrence number never enters
`contains`, the bank or the symbol-type map without its tag. Gates: the
same integer as an occurrence and as a symbol, mixed pairs, nested edges,
cycles, cache invalidation on a tag change and on restore. On capacity you
are right: a relation's row index is a global concept id and the ceiling
bounds `next_id`. Step 2 keeps that coupling and the existing value;
`relationCapacity` is withdrawn.

**S7, accepted.** Protection is transitive over both tables: every
reference held by a surviving relation protects its target, occurrence or
relation. A row is a forgetting candidate only with no incoming live
reference. A cycle is one strongly connected component, forgettable only
whole and only when nothing outside references it. Removing a relation
releases its targets for the next pass, never within the same pass, so no
cascade runs inside one pass. When a write needs space and nothing is
forgettable, the write fails loudly, as item 5 and §10.1 already do.
Gates: nested, cyclic, a full store with every row protected, and no
partial write.

**S8, accepted.** Distinct contexts are distinct documents, by the
occurrence's document digest (kept in step 2). A concept recurs when its
reading (the same `root` in `Part(occ, root)`) occurs in occurrences of at
least `k` distinct documents; no trace is removed from any point. MTM
promotion uses the same criterion over the token's occurrences, not
`Token.count`. A promoted token keeps its entry, key, introduction's kind
and uses, and gains a `symbol` field. Relations written after promotion
may reference the symbol; historical rows and edges are never rewritten,
and readers resolve an occurrence to its token's symbol through MTM.

**S9, accepted.** A relation row's stratum is read from its argument tags
(an argument that is a relation handle makes it LTM2, otherwise LTM1);
`order_of`, the conceptual order, is separate and unchanged. Kind-to-kind
subsumption at order 0 is an evaluation rule (truth as containment of the
kinds' intersections, read where the taxonomy evaluates truth,
`Queries.relation_evidence`), not storage. Step 2 stores "cats are
animals" as an LTM1 row, as today; the rule lands in a later round if
Alec decides it.

**S10, agreed.** A test that reads a removed column gets a derived reader
view or a port proposal for Alec; no semantic assertion changes without
his authorization.

**Decisions for Alec, replacing §8** (recommendations first):

1. The close-time mint for relation arguments ends: a mention's own
   occurrence is referenced instead. Recommended.
2. Protection is transitive, with cycles as components and a loud failure
   when nothing is forgettable (S7). Recommended.
3. `k` = 4 distinct documents, for both recurrence and MTM promotion.
   Recommended.
4. Kind-to-kind subsumption at order 0, as an evaluation rule in a later
   round. Recommended.
5. The occurrence handle is today's address key; no new column (S3, S4).
   Recommended.
6. `order` stays as a justified LTM0 column until item 3. Recommended.
7. Context wholes and removing the address columns move to the 5.5 line's
   own spec. Recommended.

## 11. Decided (Alec, 2026-10-21)

1. **The close-time mint ends.** It existed only because a relation row
   could reference symbols alone, so an unminted argument had to be given
   a seat at the close. Relations now reference occurrences, so the
   argument's own occurrence is referenced and nothing is minted.
2. **No cycles, by design.** A relation references only rows that exist
   before it (or are planned earlier in the same close), and no row is
   rewritten, so the reference graph is acyclic by construction. A write
   that would reference itself or a later row fails loudly. Protection is
   transitive over that acyclic graph; §10's component handling for
   cycles is withdrawn.
3. **k = 3**, and MTM does not outlive its use. "It shouldn't last long
   beyond last reference." Today MTM evicts only at capacity, oldest
   first, and on row forgetting; round 2c adds expiry: at each kept close
   an entry whose last access is more than `mtmHorizon` closes old in its
   stream is removed, and every entry of a document is removed at that
   document's end. Default `mtmHorizon` = 16, twice the STM index
   (recommended; Alec to confirm the value). Concepts recur across 3
   distinct documents (by their reading, §10 S8). Because a token now
   lives within one document, its promotion counts 3 distinct sentences
   in which it occurs, not documents (recommended; Alec to confirm).
4. **"Cats are animals" is about individuals, not events.** In the Orders
   numbering an individual is order 1 and a kind collecting individuals is
   order 2 (Orders §3: "inclusion (cats ⊂ animals), quantification,
   generic truth"); the code's determiner kind mode gives order 2
   (`ReferenceContext.operation_order`). So the statement is inclusion
   between two order-2 kinds over their order-1 members, not containment
   of order-0 regions. Storage is unchanged: `Part(SYM cats, SYM animals)`
   in LTM1, evaluated by the taxonomy's inclusion as today. §8.4's and
   §10's order-0 recommendation is withdrawn; two truths §1's
   generic-subject paragraph is amended to match.
5. **The handle is an address, not an identity.** Identity lives in MTM.
   The occurrence's address key only says which occurrence a relation or
   an MTM pointer refers to.
6. `order` stays as an LTM0 column until item 3. Approved.
7. **Context wholes get done in 3.5**, on the 5.5 line "Documents (and
   streams) as context wholes", with their own spec first.

## 12. Decided (Alec, 2026-10-21, later)

- **Relations and MTM point to whole LTM0 entries only.** "Relations should
  be pointing to entire LTM0 entries; how can MTM point to a part of an
  LTM0 since it is relatively opaque?" That is the design and today's
  write path: the close writes each noun-phrase mention as its own LTM0
  entry (the journal's order-1 constituents become separate occurrence
  rows), so every `OCC` reference and every MTM pointer names a whole
  entry. A sentence's entry stays opaque; its parts are other entries,
  linked by relation rows. No reference may address a part of an entry.
- **MTM expiry** (`mtmHorizon` = 16 closes past the last reference, and
  removal at the document's end) and **token promotion at 3 distinct
  sentences**: approved.
- **Binding stays optional at the close; a noun phrase becomes an object at
  its determination** (Alec): the own-object outcome is always available,
  and the object is created where the grammar determines the noun phrase.

## 13. Carry from the R10 review — 2026-10-21

Alec's handoff after [reconstruction spec §17](2026-10-21-reconstruction-reads-the-bind.md#17-claudes-review-of-the-r10-stop-decided-alec-2026-10-21)
carries two existing arrival-plan observations into **round 2c**, with no
change in R10 or 2a:

- A same-close joined named site plans an arrival that commit skips,
  because its use is attached after the arrivals run.
- A later read of a still-pending address also plans a preview-only
  arrival. This arrival is outside the reference reconstruction term.

Check both when 2c changes arrivals or token promotion. The R10 review
requests no current mechanism change, and these notes authorize no new
assertion, objective or keep-rule change.

## 14. Codex cull preflight — 2026-10-22; for Claude and Alec

The separate class rename is frozen at `eb6980e7d`, based on R10's
published `e243d7bdf`. The rename-only gates pass: **113 passed, seven
existing prerequisite skips** in the eight-file set and **85 passed** in
the regression files. The [receipt](../benchmarks/2026-10-22-item3-5-step2a-rename/README.md)
retains the class-path audit, exact identifier-only source audit, captures
and reports. No column has been culled and no assertion has changed.

§10 explicitly limits 2a to culls whose readers **already** have another
source. The following preflight corrects code facts in its examples; it
does not reopen the decided layouts, source addresses or order column.

| Proposed cull | Current source and reader | Consequence before a replacement exists |
| --- | --- | --- |
| `rel_type` | Already absent from the row. `LTM0._load_from_state_dict` drops a zero legacy column and rejects active retired relations. | There is no live column left to remove; retain the loud rejection. |
| `refs` | Current closes do not populate it, but `Forgetting._references`, coarsening, `clear_origin` and `ClauseRows._native_reference_indexes` still read it. Coarsening writes the retained column. `test_forgetting.py:139` explicitly supplies an address and checks it is cleared. | "No live writer" needs qualification: no close-time introduction, but there are live maintenance/read paths and a retained fixture contract. A constant empty reader would not preserve that contract. |
| `where` | `ClauseRows.write_clause` copies `Clause.where`; `row()` and `SituationFrame` return it. `test_clause_storage.py:96` requires the exact value. | No numerical consumer does not make the stored value derivable. The current opaque point has no declared reader for this four-coordinate stamp. |
| `metadata_required`, `semantic_fingerprint` | `raw_meaning_of`, `semantic_extras` and `load_semantic_extras` use them to reject missing or mismatched scope. The flag's partial-checkpoint test and the fingerprint's sidecar-corruption tests are live. | Deleting them, or rebuilding the fingerprint from the sidecar it authenticates, changes the restore contract before 2b supplies the structure. |
| Leaf postings and `leaf_complete` | With a configured owner, `reindex_meanings` rebuilds them. Standalone restore can also consume saved postings; existing tests inspect their wire columns, exact retrieval and malformed-column rejection. | The owner-backed rebuild exists, but removing the serialized fields needs the S10 representation ports or an explicitly specified wire view. |
| `adjacent_*` | `Models._commit_sentence` writes identified-code pairs through `witness_adjacency`. Restore reads these columns; `reindex_meanings` does **not** rebuild them. `test_operators_final.py:45` requires their round trip. | These are not yet rebuildable from the retained reading by any live reader. Dropping them loses observed order and positions. |
| `polarity` | `append_meaning` stores it independently when evidence is supplied. Equal positive/negative evidence pairs may accompany either polarity. | Exchanged lanes cannot recover this bit for all accepted rows. |
| `grammatical_mode` | `append_meaning` permits an unverified interrogative or assertive reading with the same closed roles; admission also reads the mode. | Open-reference status alone cannot recover every currently accepted mode. |

The isolated [preflight probe](../benchmarks/2026-10-22-item3-5-step2a-rename/cull-preflight.json)
demonstrates the last three information gaps on `eb6980e7d`: identical
content, roles and evidence with opposite polarities or different modes;
and adjacency lost after omitting its columns and reindexing. It changes
no production state. Its first invocation mistakenly used the default
fact admission for an interrogative and was rejected; that script and log
are retained. The corrected diagnostic uses `kind='unverified'` for all
three rows. This was a new diagnostic setup error, not a failed gate.

**Questions for Claude.** Verify these boundaries and identify any cull
that is safe now without a new representation, test port or changed
admission behavior. In particular, the adjacency columns are not rebuilt
by the reader that rebuilds leaf postings. The two must not be treated as
one cache migration.

**Decision for Alec, with recommendation.** Land the rename on its own
evidence, and leave the listed fields in place until their derived source
and any exact S10 test ports have been reviewed. Keep the undecided culls
explicitly on step 2's line; do not declare the row reduction complete or
remove them from 3.5. Do not add compatibility storage or broaden the
accepted row/restore contract merely to make a deletion pass. If a cull
must precede 2b, its representation and concrete test ports need a decision
first. This is a stop before further implementation, under the handoff's
rule for new spec questions; no gate failed and no sweep has run.

## 15. Claude's review of the rename and culls — 2026-10-22; decisions for Alec

[Claude's read-only review](../benchmarks/2026-10-22-item3-5-step2a-rename/claude-review.md)
of frozen source `eb6980e7d` says **safe to publish the rename**, with no
BasicModel blocker. It confirms the exact rename, saved-class-path audit
and 198 passes / seven existing skips. The requested parent WikiOracle
import search was then checked by Codex: neither old class name occurs
outside BasicModel and the unrelated nanochat submodule. No extra gate or
collect-only pass was run. Publication of the separately authorized rename
does not settle the culls or complete 2a/line 2.

Claude confirms §14's central finding: **none of the listed culls meets
§10's current derived-source condition.** Keeping those fields now is
already authorized, not a new approval request. The future replacement
and disposition choices below still need Alec.

The review adds these code facts: `ConceptForgetting.py:66` also reads
`refs`, as do the row export and the predicate lookup through
`point_of_row`. The coarsening fixture sets the address at
`test_forgetting.py:139`, checks it is cleared at `:156`, and checks the
column snapshot at `:126`. The local `refs` binding at
`Forgetting.py:166`, described in the review as scoring, is unused; it
does not supply a derived representation. `where` is stamped by
`ClauseJournal.py:256–264` before its close-time write. The metadata flag
also feeds `row()['metadata_complete']` and `Language.py:11865`. Without
explicit evidence, zero trust still loses the polarity distinction in the
lanes. `grammatical_mode` also includes legacy `unspecified`, which an
open-reference test cannot recover. The adjacency writer is
`Models._commit_sentence`; §14's initial method-name typo was corrected.

**Claude's recommendations for Alec (not yet decisions):**

| Field or boundary | Recommended disposition |
| --- | --- |
| `refs` | Retire in 2b only when tagged references replace it. Propose the exact coarsening-fixture port to stored role references then; do not change its assertions now. |
| `metadata_required`, `semantic_fingerprint` | Keep through 2b while they authenticate sidecar content. Revisit after the structural sidecar has moved, accounting for any remaining scope/text/provenance. |
| Leaf postings, `leaf_complete` | Keep. Decide the S10 restore contract in 2b before any cull: owner-backed restoration or a specified standalone wire view. |
| `adjacent_*` | Treat as retained data. Before changing it in 2b, decide whether pair wholes become LTM1 rows or remain justified columns. |
| `polarity` | Keep as a justified column. A narrower admission contract that makes lanes sufficient requires its own spec. |
| `grammatical_mode` | Keep until 2b's question layout exists; decide the disposition of legacy `unspecified` rows before deriving it. |
| `where` | Keep beside `when` until the 5.5 context-wholes spec. This changes the original removal disposition and needs Alec's decision. |

After Alec's decisions, amend §10's examples and the carried cull list to
match. These fields remain explicit work on step 2's line until their
disposition is recorded; no cull, representation port or admission change
has been implemented. The two R10 arrival observations remain in 2c.
No learning claim, no second sweep, and item 3.0 has not been reached.
