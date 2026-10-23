# Step 2 of item 3.5: one long-term memory in strata (Claude, 2026-10-21; for Codex)

**Status:** reviewed and decided in §§10–12; they supersede the original
round boundaries and open choices below. R10 is published at `e243d7bdf`
(WikiOracle `2ffda78`). Round 2a's separate rename is reviewed at
`eb6980e7d`: the saved-class-path check and scoped gates pass, and Claude
approved its publication at `804d8adcb` (WikiOracle `dbd9b52`). The
[rename receipt](../benchmarks/2026-10-22-item3-5-step2a-rename/README.md)
records that evidence. **Alec's cull decisions are §16:** 2a removes
`where`, keeps `.when`, and leaves the other columns until their decided
replacement. §17 authorizes the remaining shape-test port. The
[where receipt](../benchmarks/2026-10-22-item3-5-step2a-where/README.md)
records frozen source `d4f316411`, 223 passes / seven existing skips and
Claude's approval to publish the sole cull, completing 2a at `0b030e3eb`
(WikiOracle `baa2e7f`). **§§33–36 answer the restatement review and decide
the sequence: storage changes now, identity threading next, derivation
removal after it. Claude's incoming four documents were committed unchanged
at `cc6407f3b`. §38 approves the revised P5 port and is committed unchanged
at `bb0045d7a`. The scoped implementation has begun; §40 records the next
required test disposition, for the newly retained relative sentence beside
its unasserted operands. Work is stopped before applying that port, freezing
source or running gates.**
Main includes the published documentation handoff at `c39ef9f3a`; all work
now stays on main. Original code facts below describe
`cf61b5678`; old class names in those historical facts locate that source.


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
  *Corrected 2026-10-22 (§§14–16): these examples were wrong about the
  code, and no listed field had a derived source. Alec's dispositions are
  in §16: `where` comes off the row in 2a, polarity is semantic and
  stays, and the rest stay until 2b replaces them.*
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
  Alec below: keep `order` as a justified LTM0 column until item 3
  carries it in the concept (its recorded order and Part edges; one
  conceptual space since 2026-10-22, Orders §10).

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

## 16. Claude's review of step 2a; decided (Alec, 2026-10-22)

Reviewed: frozen source `eb6980e7d`, published with WikiOracle `dbd9b52`;
its receipt; §§14–15; and the delegated
[review](../benchmarks/2026-10-22-item3-5-step2a-rename/claude-review.md).

**The rename is correct and published.** The class-path audit, the
identifier-only source audit and 198 passes with seven existing skips
support it.

**The culls: Codex was right to stop.** §10's examples were mine and were
wrong about the code. `refs` has live maintenance readers; `where` is
stamped at every close; the adjacency columns are data that no reader
rebuilds; and polarity and mode carry information that the lanes and open
references cannot recover. Spot-checked: `ClauseJournal.bands` stamps
`where` from the words' input positions, and `append_meaning` folds
polarity into the lanes only when no evidence pair is supplied.

**Decided (Alec).**

1. **`where` comes off the LTM row in 2a.** Alec: ".where is associated
   with all percepts. .when is associated with LTM." The column holds the
   encoded input position of the sentence's first word, copied from its
   percepts at the close, and nothing computes with it. The row export and
   `SituationFrame` stop carrying it, and `test_clause_storage.py` drops
   its exact-value check of `where`; that port is authorized. Location
   stays with the percepts.
2. **Polarity is semantic and stays.** Alec: "Polarity is semantic; since
   they are always paired, I would expect the first of the pair to be pos
   evidence, and the second of the pair as neg evidence." Polarity is a
   field of the meaning, and the evidence pair is always (positive,
   negative) for the meaning as stated. Live closes already write that way.
   Writes without an explicit pair (`append_estimate`, the observation in
   `append_expectation_pair`, and `append_idea` without evidence) swap the
   pair for a negative meaning; 2b removes that swap. Any existing
   assertion that depends on the swap comes back to Alec.
3. **Kept until 2b replaces them**, as §10's conditional rule already
   requires:
   - `refs`, retired when tagged references replace it; the coarsening
     fixture's port is proposed then;
   - `metadata_required` and `semantic_fingerprint`, kept while they
     authenticate sidecar content;
   - leaf postings and `leaf_complete`, after 2b decides the S10 restore
     contract;
   - `adjacent_*`, retained data, after 2b decides whether word-pair wholes
     become LTM1 rows or stay a justified column;
   - `grammatical_mode`, kept until 2b's question layout exists; 2b also
     decides what happens to legacy `unspecified` rows.

   `rel_type` has nothing left to cull. Where 2b chooses a representation
   above, it comes to Claude and Alec as a dated spec question with a
   recommendation.

## 17. The remaining `where` shape check — 2026-10-22

§16 authorizes removing the exact-value check at
`test/test_clause_storage.py:96`. The same file also asserts at line 23:

```python
assert store.where.shape == store.when.shape == (8, 4)
```

The authorized representation port is exactly:

```python
assert store.when.shape == (8, 4)
```

This removes only the expectation that the deleted column exists and
preserves the entire time-shape check. The
[patch](../benchmarks/2026-10-22-item3-5-step2a-where/proposed-shape-port.patch)
is retained for review. **Alec authorized this shape-check port on
2026-10-22**, in the handoff conversation. It is applied with the original
exact-value removal; every other assertion is unchanged. No existing test
was run or failed before the authorization, and it uses no setup exception.

The constructor in `test_independent_components.py` also supplied
`where=` to `SituationFrame`. Removing that unused fixture argument follows
the decided frame API removal; all of its assertions remain verbatim.
No production or existing-test failure is being treated as a same-round
setup exception.

## 18. Codex's 2b preflight — 2026-10-22; for Claude and Alec

2a is published: BasicModel **`0b030e3eb`**, WikiOracle **`baa2e7f`**.
The cull's frozen source is `d4f316411`; its [receipt](../benchmarks/2026-10-22-item3-5-step2a-where/README.md)
records 223 passes, seven existing skips and Claude's **publish, no
blockers** verdict. The separately published rename and this sole cull
complete 2a. No 2b production or test change has been made, and no 2b gate
has run. These questions are a code review of the published source, not
failed-test diagnoses. The [2b spec-review receipt](../benchmarks/2026-10-22-item3-5-step2b-spec-review/README.md)
retains this review and Claude's response.

The decided owner, tagged layouts, whole-entry occurrence handles,
acyclic graph and transitive protection in §§10–12 are not reopened.
`metadata_required` and `semantic_fingerprint` stay while they authenticate
sidecar content. §16 requires the following three representation decisions
before their implementation; the two test dispositions follow from the
same preflight and the standing assertion rule.

### Q1. Leaf-posting restoration: keep the specified standalone wire view?

`MemoryIndex.LeafCodeIndex` serializes `posting_codes`, `posting_roles`,
`posting_rows` and `leaf_complete`. Its loader validates dimensions,
dtypes, row ranges and unique/sorted postings, then exposes the restored
cache to a standalone store. With a complete owner,
`configure_model_index` calls `reindex_meanings`, replacing that cache from
the retained readings. These are two live contracts:

- `test_accessible_mind.py::test_leaf_column_growth_is_amortized_and_checkpoint_append_is_safe`
  requires the compact serialized column and retrieval from the restored
  standalone store, including subsequent append and reset.
- `test_part_read.py::test_postings_rebuild_from_part_without_inverting_the_narrowed_point`
  supplies a structurally valid stale cache and requires the attached
  owner to rebuild the correct terms. The integrated checkpoint case in
  `test_learning_readiness.py` requires malformed `posting_codes` to fail.

**Recommendation:** retain these wire columns and `leaf_complete` as the
specified standalone cache view in 2b. Keep their current validation;
once the owner and both tables have loaded, rebuild from tagged reading
rows before owner-backed retrieval. The serialized cache does not become
an authority over the loaded derivation. This preserves the current
standalone and complete-owner restore contracts and needs no assertion
port in those cases. Removing the wire cache instead would require an
explicit change to the standalone API and S10 ports. **Decision for Alec:**
keep this specified wire view, or make complete-owner restoration a
prerequisite and authorize its concrete test ports?

### Q2. Adjacency: retained witness columns or a new LTM1 layout?

`Models._commit_sentence` calls `witness_adjacency(row, identified_codes)`.
`MemoryIndex` stores `(left code, right code) -> {(occurrence address,
pair position)}` and serializes it through `adjacent_left`,
`adjacent_right`, `adjacent_address` and `adjacent_position`. Re-witnessing
replaces that address's pair witnesses; compaction removes dead addresses.
*Corrected by Claude's review, §19: codebook remapping updates leaf
postings only, not adjacency keys; the original claim here was wrong.*
`reindex_meanings` does not
reconstruct these observations.

The data is consumed: `MereologicalCodes.centroid_forms` supplies it to
`SymbolCentroid.place`, where distinct witnesses weight adjacent wholes;
`begin_forward` snapshots it for both trials. The keys are word-bank code
addresses, not untagged native concept IDs. The checkpoint, replacement
and compaction assertions in
`test_operators_final.py::test_adjacent_wholes_rewitness_compact_and_checkpoint`
require the exact witnesses, including pair position. A bare
`Part(SYM left, SYM right)` would not preserve that occurrence/position
multiplicity or identify the pair whole.

**Recommendation:** retain the four columns as justified inverted witness
data in 2b, preserving those assertions. Do not describe them as a cache
that the new reading rows already rebuild. Moving them to LTM1 needs a
reviewed pair-whole and witness-position layout, with a code-to-symbol
mapping and no reference into part of an LTM0 entry. **Decision for Alec:**
retain the columns for this round, or specify that layout before 2b?

### Q3. `unspecified` mode is also produced by live writes

`LTM0._load_from_state_dict` gives legacy conversation rows mode
`unspecified`; only the accepted user/provisioned origins get assertive
fact status. More than migration depends on this value:
`ClauseRows.write_clause` constructs a live prediction's estimate with
`mode='unspecified'` (`ClauseRow.py:443–445`). `ConceptualMeaning` permits
all three modes and `raw_meaning_of` returns the stored one. Thus §16's
"legacy `unspecified` rows" understates the boundary. Absence of an open
reference does not establish an assertion, and an open pending name is
not by itself a question.

**Recommendation:** keep `grammatical_mode` as a justified column through
2b for the accepted three-mode API, including live estimates and legacy
unknowns. Implement the decided question layout, but do not default
unknown rows to assertions or infer mode from the presence of any open
site. Retire the column only once all accepted modes have a specified
lossless structural source. No new flag or duplicate compatibility field.
**Decision for Alec:** retain the column on this basis, or specify the
unknown/live-estimate layout and restore rule before deleting it?

### Q4. The implicit-pair decision requires two expected-value ports

§16 already decides the production rule: the pair is evidence for the
meaning as stated; polarity remains a separate semantic field. The shared
implicit-pair branch in `LTM0.append_meaning` currently reverses that pair
for a negative meaning. The existing parametrized
`test_operators_round3a.py::test_trust_to_pair_and_negation` requires the
old reversal in exactly two cases.

**Recommended port for Alec's authorization:** change only these two
parameter tuples:

```python
(.7, False, (0., .7))  ->  (.7, False, (.7, 0.))
(-.4, False, (.4, 0.)) ->  (-.4, False, (0., .4))
```

Every assertion body, the other three cases, the retained trust-derived
pair check and the absence-of-retired-`trust` check stay verbatim. This is
an expected-value change required by §16, not a setup correction; it is
proposed here and has not been applied. New regression coverage will
exercise the affected write paths and retained polarity on the 2b source.

### Q5. The coarsening fixture also tests the old deletion policy

The `refs` representation port has two straightforward locations in
`test_forgetting.py`: replace the removed-column snapshot at `:126` with
the tagged-reference representation, and replace the raw `refs` fixture
write at `:139` with the retained occurrence edge. But the assertion at
`:156` is not the only issue.

`test_idea_coarsens_parent_and_relation_cascades` also requires the child
to be deleted, the idea parent's reference to become `None` and
`coarsened == 1`, or the relation parent to be deleted in the same pass
with `cascade == 1`. `Forgetting.forget` implements precisely this in
`deletion()`: it edits a surviving parent's references or adds its parent
to the removal set. Under §§10–11, a surviving relation protects its
targets, no row is rewritten, and a removed relation releases targets
only for the next pass. Merely porting `store.refs[index, 0] == -1` to a
new reader would retain the wrong policy once the dependency is in LTM1.

**Recommendation for Claude to check and Alec to decide:** port this
case to the already decided protection contract, preserving both
idea/relation variants, capacity, value weights and the content equality
check. With the parent surviving, require the child and its reference to
remain, with zero coarsening and zero same-pass cascade. Add a separate
case that removes an unreferenced relation, proves its target survives
that pass, and permits reclamation on the next pass. Authorize the
corresponding old-policy assertions to change; do not call them mere
column-name changes or keep a second legacy deletion policy. The exact
tagged accessors and snapshot patch will be listed with the 2b source
before freezing; no existing assertion has changed in this preflight.

Work stops before 2b implementation for Claude's code-fact review and
Alec's dispositions. The two R10 arrival observations remain in 2c. The
2a review's `SentenceEndState.where` recall-history observation goes to
the 5.5 context-wholes spec; it does not expand this cull or 2b. No gates
are rerun, and the one full sweep remains at step 2's end. No learning
claim and no transition to item 3.0.

## 19. Claude's 2b preflight review — 2026-10-22; decisions still open

[Claude's read-only review](../benchmarks/2026-10-22-item3-5-step2b-spec-review/claude-review.md)
used review-input commit `d5fc19791`, with production and tests unchanged
from published `0b030e3eb`. It confirms Q1–Q3's recommendations with the
corrections below, verifies Q4's two proposed expected values and confirms
Q5's policy conflict. **These are recommendations, not Alec's decisions.**
There is still no 2b implementation or test execution.

### Restore and reader corrections

- The missing-postings case in `test_accessible_mind.py:381` is a third
  live restore contract and must remain supported beside standalone wire
  restore and complete-owner stale-cache rebuild.
- The loader does not itself prevent retrieval from a stale wire cache.
  Rebuild occurs at `configure_model_index`; "before owner-backed
  retrieval" must preserve that calling convention. In 2b, LTM1 must load
  before `load_semantic_extras` reads the moved structure, or that rebuild
  must be explicitly deferred until both tables are available.
- Add `MemoryIndex._meaning_leaf_terms` and the `read` closure inside
  `configure_model_index` to §10 S6's tag-aware boundaries and gates. The
  latter currently treats every `store.parts.refs` argument as a symbol.
- `ClauseJournal.py:405–406` is another `refs` reader: it treats exported
  entries 0 and 1 as relation operands. Its port must distinguish a
  relation row's operands from the occurrence-role positions used by
  forgetting. `test_live_parent_pins_its_child` retains its behavior, but
  its `role_refs[0]` read must follow the new representation.

**Recommendation for Q1:** keep the current validated wire view and the
owner rebuild, with the restore ordering and additional tagged readers
above. Do not add a new cache authority or silently accept malformed
posting columns.

### Adjacency correction and disposition

§18 originally claimed that remapping updates adjacency keys. It does
not: `MemoryIndex.remap_leaf_codes` remaps `_leaf_postings` only. The
adjacency keys can therefore go stale if word rows are remapped. This is
an existing code gap, found by inspection, not a failed 2b run. The full
review is retained and §18 is corrected in place.

**Recommendation for Q2:** keep the four adjacency columns in 2b as
non-protecting witness data, preserving their current checkpoint and
re-witnessing contract. They must not become a second protection graph.
Give the remap gap a separately scoped follow-up with a failing-before
regression and review; Alec decides its destination before it is moved
out of this open-question list. It has not been repaired or silently
declared a cache. An LTM1 replacement still needs its own pair-whole and
position layout under the whole-entry-reference rule.

### Mode disposition

The review confirms live `unspecified` estimates and the independent
three-mode API. It also identifies both directions of non-equivalence:
the grammar's `AskLayer.meaning_mode` can produce an interrogative without
an open site, while existing assertive rows have open referent slots.
Record kind is not a substitute either. **Recommendation for Q3:** keep
`grammatical_mode` as the existing three-valued column; no extra flag,
default assertion or new word-based classifier.

### Evidence: another swap and the relation-edge convention

Q4's two expected-value ports are correct. The review also corrects two
code facts in the earlier path list:

- `append_idea` always constructs the default positive-polarity
  `from_description` meaning. Although it uses the shared writer, it
  cannot exercise the negative swap through its present API.
- `ClauseRows.write_clause` has a second swap at `ClauseRow.py:468–472`.
  It derives a pair from trust and reverses it for negative polarity.
  The branch runs for an implicit pair and under the TruthSet assertion
  context even if an explicit pair was supplied.

For a completed relation the destination is the shared LTM1 `Part` edge,
which has no polarity field. The existing tests in
`test_lies_can_be_learned.py`, `test_clause_storage.py` and
`test_relative_sentence_codebook_insertion.py` supply explicit negative
counterevidence to that same edge and require both poles to remain. They
pin the canonical relation-edge convention, not the implicit-swap branch;
none is authorized to change. The LTM0 meaning, by contrast, retains its
own polarity, and `Queries` accounts for stored/query polarity when read.

**Recommendation for Alec:** authorize Q4's two parameter-tuple changes
for the decided LTM0 implicit-pair rule. Keep the LTM1 canonical-edge
convention in this round until a relation-polarity layout is specified.
Decide the scope of the shared `write_clause` swap explicitly before
editing it. Claude recommends leaving that site outside this removal for
now. Codex notes that its loop also prepares pairs for LTM0 writes: the
disposition must distinguish the destination and the assertion-context
override, rather than assuming every iteration writes a relation edge.
No swap site or test tuple has changed in this preflight.

### Protection: the policy port needs a complete layout first

Claude confirms that the old coarsening test cannot keep its semantic
assertions under the decided rule. Its parent depends on the child twice
(sidecar `role_refs[0]` and the raw `refs` column); porting just the latter
would leave the old deletion policy live. With the proposed replacement,
the capacity-four fixture's target is two: the protected parent and child
remain, and the two independent rows with trust .7 and .8 are removed.

**Recommended test disposition, subject to Alec:** change the old-policy
assertions to child/parent/reference retention, zero coarsening and zero
same-pass cascade, preserving both variants, capacity, weights and the
content equality check. Also check that the two independent rows are
gone and the target was reached, so protection is not a vacuous no-op.
For the next-pass release case, set actual second-pass pressure (or an
explicit epoch pass with `at_epoch`) and retain `min_interval=0`; expect
the existing warning when protection prevents the first pass reaching
its target. The exact snapshot port at `test_forgetting.py:126` is to
remove `'refs'` from the key tuple, with an explicit new tagged-table
immutability check in the replacement coverage. These test dispositions
are proposed only; existing assertions remain untouched.

**Three layout questions still require answers before that port:**

1. Which tagged relation encodes role 0 of the fixture's LTM0 parent
   pointing to its child? The stated `Part(OCC occurrence, SYM reading)`
   row alone is not this occurrence-to-occurrence participant edge.
   **Recommendation:** express it through the typed derivation's declared
   participant position and an `OCC` child, with an explicit worked layout
   in S2; do not invent an untyped role-edge convention in the code.
2. What preserves an occurrence's reading and derivation rows while that
   occurrence survives? The later reading row points to and protects the
   earlier occurrence; the reverse dependency is not specified. Dropping
   that reading row would coarsen a surviving occurrence without editing
   its opaque row. **Recommendation:** have Claude complete S2/S7's
   retention rule so a surviving occurrence keeps its readable structure,
   with no semantic back-edge that breaks the DAG and no silent coarsening.
   The exact lifetime/protection mechanism remains open for Alec.
3. Is a pass the whole document-boundary call or each table's forgetting
   function? Today `model_boundary` invokes `forget_parts` then `forget`.
   **Recommendation:** one boundary call is one pass across both tables;
   releasing a target must not make it collectible later in that same
   call. Only the next boundary can use the release.

The three field decisions, adjacency-gap destination, evidence scope,
test authorizations and the completed protection layout remain with Alec.
This is the required stop under the handoff's new-spec-question rule,
not a permission request for the already completed 2a work. The next
implementation must start from the published 2a source after these
dispositions. No second sweep, no learning claim, and no item-3.0
transition.

## 20. Claude's answers to §§18–19 — 2026-10-22

**The `where` cull.** Reviewed `d4f316411`, published as `0b030e3eb`: the
column, its write, its export and the frame field are gone, an old
checkpoint drops the column on load, and the new location test covers both.
No finding.

**The three layout questions, completing S2 and S7.**

1. *The participant edge.* A sentence occurrence's role 0 is read through
   its reading row and its derivation, never through a separate role edge.
   `Part(OCC S, SYM root)` names the derivation root; the root's operation
   declares the participant position (for `lift`, `argin1`), and that
   argument is the child occurrence, tagged `OCC F`. Worked layout for
   "Felix runs" after an earlier mention F of Felix: LTM0 holds F and S;
   LTM1 holds F's reading `Part(OCC F, SYM felix)`, the derivation edge
   `e = lift(OCC F, SYM runs)`, and S's reading `Part(OCC S, SYM e)`. The
   forgetting fixture's parent-to-child dependency is that argument: S
   reaches F through e. No untyped role edge exists.
2. *A surviving occurrence keeps its reading.* An occurrence owns its
   reading row. The row lives and dies with the occurrence: it is never
   chosen for forgetting on its own, and it is removed with its occurrence.
   Ownership is a lifetime rule, not a reference, so no back-edge enters
   the graph and it stays acyclic. The rest follows from the decided
   protection: the reading row protects its root, and the derivation
   protects its arguments transitively, so a surviving occurrence keeps its
   readable structure. A derivation symbol shared by several readings
   survives while any of them does. Forgetting therefore values occurrences
   and independent relations, never reading rows. A fact row, an LTM1
   relation that is not a reading, is forgotten by its own value whenever
   nothing protects it.
3. *A pass is one boundary call.* `model_boundary` calls `forget_parts`
   and then `forget`; together they are one pass. A release made during
   that call takes effect at the next boundary.

**Q1–Q5.**

- **Q1.** Keep the validated wire view with the owner rebuild, supporting
  all three restore contracts (standalone wire, complete-owner stale
  rebuild, missing postings), with §19's load ordering.
- **Q2.** Keep the four adjacency columns as non-protecting witness data.
  The remap gap, adjacency keys that `remap_leaf_codes` never updates, is
  an existing bug: fix it as its own small commit at the start of 2b, with
  a failing-before regression, since 2b touches `MemoryIndex` anyway.
- **Q3.** Keep `grammatical_mode` as the three-valued column. Like
  polarity (§16), mode is a field of the meaning, semantic, and not
  derivable from open sites.
- **Q4.** Authorize the two tuple ports. For the sentence writer's swap
  (`ClauseRow.py` ≈468–472), the destination decides, under §16: reverse a
  trust-derived pair only when it goes to an LTM1 edge, which stores no
  polarity, so a negative relation is counterevidence on its positive
  edge. Never reverse it for an LTM0 meaning, which keeps its polarity and
  its pair as stated. The tests that pin the edge convention stay
  unchanged; any existing assertion that pins the LTM0 reversal comes back
  to Alec.
- **Q5.** Authorize the coarsening port as §19 proposes, built on the
  layout above: parent, child and reference retained, zero coarsening,
  zero same-pass cascade, the two independent rows removed and the target
  reached, plus the next-pass release case.

## 21. Alec's answers to §20, and Claude's revisions — 2026-10-22

**Decided: evidence.** Alec approved both parts of Q4: the two tuple ports
in `test_trust_to_pair_and_negation`, and limiting the sentence writer's
reversal to LTM1 edges. An LTM0 meaning keeps its polarity and its pair as
stated.

**Leaf postings and mode.** Alec: "I don't know what they are, so I'm
inclined to remove."

- *Leaf postings* are the inverted index from each leaf code to the
  stored rows and roles that contain it, used to find stored sentences
  that mention a code. The index is derived: with its model attached,
  `configure_model_index` rebuilds it from the rows. Only a store restored
  without its model reads the saved copy. Claude's revised recommendation:
  remove the saved columns and `leaf_complete`, keep the in-memory index
  rebuilt at load, and port the standalone-restore tests. A derived cache
  is rebuilt, not stored.
- *Mode* says whether a stored sentence was a statement, a question or the
  model's own guess (`unspecified`), so a stored question is never read
  back as a fact. It can go once each case has a structural source: a
  question by the `ask` operation at the root of its derivation, which the
  reading row reaches; a guess by being written as an estimate; and a
  legacy row read as unverified. Claude's revised recommendation: remove
  it in 2b with that replacement, porting the tests that read it.

**Adjacency.** Alec: "Probably design layout. Again, why do we need word
adjacency?" Its one use is `SymbolCentroid.place`, reached through
`MereologicalCodes.centroid_forms`: it pulls each word's *form*, the
perceptual half of the `[form | meaning]` pairing, toward what its
neighbouring words' forms share, weighted by how often they were adjacent.
That moves forms by context, which the two-spaces rule forbids: context
places meanings, in conceptual space, where the occurrence-context mean
and item 3's placement already work. Claude's revised recommendation:
remove the four columns and the nudge, leaving each form at its letters'
fold, rather than design a layout. The stale-key bug goes with them.

**Forgetting: held by Alec; §20's answer 2 is withdrawn.** It contradicted
the forgetting spec's decided "coarsening, not cascade, where the gist
survives" and its rule that a pass takes each row one stage further.
Revised rule:

- An occurrence's reading and derivation rows are stages of that
  occurrence's forgetting. They never outlive it and are not chosen by
  their own value, but they can be dropped before it. The occurrence then
  keeps its fused point as its gist and reads thereafter without inner
  structure: coarsening without rewriting a row.
- A child referenced by a derivation stays protected until that derivation
  is dropped, and is released for the next pass.
- One boundary call, `forget_parts` and then `forget`, is one pass.

The coarsening test then ports across two passes. In the idea variant,
pass 1 coarsens the parent (structure dropped, content kept,
`coarsened == 1`) and releases the child, and pass 2 removes the child. In
the relation variant, pass 1 removes the relation parent and pass 2 the
released child. Capacity, weights and the content check stay.

All three remain Alec's decisions; no implementation has started.

## 22. Decided (Alec, 2026-10-22): mode is not stored

Alec: "I'm not sure mode is even needed structurally. If there is a
question, we expect an answer. That is something the teacher should know,
who has access to the data. The model will learn from the form of the
question that they need to provide an answer or not."

So 2b removes the stored `grammatical_mode` column with no structural
replacement; §21's proposed replacement is withdrawn. A stored question
already makes no truth claim: its evidence pair is empty, so it is never
read back as a fact. The readers of the stored column (`raw_meaning_of`
and `accept_fact`) stop using it; any existing assertion that depends on a
stored mode comes back to Alec.

The live parse keeps its mode in this round. It is the form the grammar's
`ask` operation gives the current sentence. Answering inside an episode is
already the chooser's choice (Alec: "it calls ask() or query() when it
wants"); mode's last live use is deciding whether a yes/no question,
which has no open slot, opens an episode at all. Offering the episode at
every close instead is item 3's work
([todo item 3](../../todo.md)).

## 23. Decided (Alec, 2026-10-22): postings, adjacency and forgetting

1. **Leaf postings: remove the saved copy.** Drop `posting_codes`,
   `posting_roles`, `posting_rows` and `leaf_complete` from the checkpoint.
   The in-memory index is rebuilt at load from the stored rows, after both
   tables have loaded (§19's ordering). The tests that restore memory
   without a model are ported; Alec's choice authorizes those ports, listed
   in the receipt.
2. **Word adjacency: removed, with context to come from WholeSpace.**
   Alec: "We don't need to nudge toward neighbors: we need context in the
   WholeSpace, which would achieve the same thing; so maybe we need to make
   wholes of part unions, which would include neighbors in the
   WholeSpace?" 2b removes the four `adjacent_*` columns, the
   `witness_adjacency` writer and the neighbour nudge in
   `SymbolCentroid.place`, so each form stays at its letters' fold. Tests
   that exist only to check the removed mechanism are retired and listed in
   the receipt. The stale-key gap goes with it. *Later the same day Alec
   withdrew the WholeSpace replacement:* a whole storing left/right
   neighbours "is not really characterizing the whole". Nothing replaces
   adjacency: neighbours are distributional information, and a word's
   order-0 meaning is already the occurrence-context mean of the sentences
   it occurs in (`MereologicalCodes.occurrence_terms`).
3. **Forgetting: §21's staged rule is adopted.** A sentence's reading and
   derivation are dropped before the sentence, leaving its fused content as
   a gist with no row rewritten; a referenced child is released for the
   next pass; one boundary call is one pass. The coarsening test is ported
   across two passes as §21 describes, with capacity, weights and the
   content check unchanged.

With §§16, 21 and 22, every 2b question from §18 is decided: postings and
adjacency removed; mode no longer stored; polarity semantic, with the
writer's reversal limited to LTM1 edges and the two tuple ports; and the
staged forgetting rule.

## 24. Test-port audit before 2b implementation — 2026-10-22

The repository handoff is complete. Claude's seven uncommitted documents
were committed **without any content change** as `c39ef9f3a`, with the
requested message, then fast-forwarded with `d5fc19791` and `61860c849`
onto main and pushed. No WikiOracle bump was made for these documents.
All subsequent work is on main. Production and tests still match the
published 2a source; no 2b code or test has changed and no gate has run.

§§20–23 settle the design. This audit does not reopen those decisions.
It identifies existing assertions not covered by the specifically
authorized test ports. §22 explicitly requires an assertion that depends
on stored mode to return to Alec, and the latest handoff says any other
assertion change stops for Claude and Alec. The [port-preflight receipt](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/README.md)
retains the audit and the proposed patch; nothing in that patch is applied.

### P1. One stored-mode assertion must change

`test_nested_retention.py:56`, in
`test_origin_replacement_retains_reachable_content_without_fact_authority`,
requires the restored LTM0 meaning's mode to be `interrogative`. The fixture
writes a question with zero evidence and later clears its origin while a
parent still references it. There is no allowed implementation of §22
that remembers this mode solely to satisfy that assertion.

**Recommended exact port for Alec:** replace only that assertion with
`assert store.row(_index(store, inner_ref))["evidence"] == (0., 0.)`.
It checks §22's reason the retained question makes no truth claim. Keep
all reachability, content, scope, origin, count and idempotence assertions
verbatim. Do not replace it with an assertion that stored mode is
`assertive` or reconstruct it from the retained structure. Other mode
assertions found in the live parse and `WhatInteractionMemory` are outside
this removal and stay unchanged.

### P2. Exact LTM1 column assertions must include the decided tags

§1's `ref_kinds[cap, 2]` adds the argument tags to LTM1. Two existing
checks require precisely the old six-column layout:

- `test_clause_storage.py:40`: the exact `LTM1.columns` tuple;
- `test_part_expectation.py:31`: `len(f.store.parts.columns) == 6`
  (both parametrizations).

**Recommended exact ports:** add `ref_kinds` immediately after `refs` in
the first tuple and change the count to seven in the second. The two
references, poles, access count, surprise and time remain checked. Every
expectation target, residual, source, bank-isolation and restore assertion
stays verbatim. Do not hide the tags outside the row export to keep a
six-column assertion passing.

### P3. A mixed certificate still checks the removed nudge's switch

`test_operators_final_b.py:133`, inside `forced_prefix_certificate`,
asserts both `derived.context_width == 128` and `derived.symbol_centroid`.
The enclosing certificate also checks descent, reconstruction and shared
snapshots, so it is not an adjacency-only test to retire under §23.

**Recommended exact port:** remove only `and derived.symbol_centroid`.
Keep the context-width check and all certificate assertions, including
zero R/residual and both-trial snapshot counts. Retire the now-unused
switch with the mechanism rather than retaining an inert true flag for
the test.

P1–P3 are the four edits in
[proposed-fixed-ports.patch](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/proposed-fixed-ports.patch).
They have not been applied.

### P4. `refs` audit: more than the coarsening case reads the old column

The [complete direct-access inventory](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/ltm0-refs-audit.json)
records the current file, function, line and source for each match. Besides
the already authorized coarsening/snapshot ports, these include:

- empty-column shape/absence checks in `test_clause_storage`,
  `test_clause_acceptance`, `test_part_closing`, `test_sentence_end_state`
  and `test_sentence_retention`;
- immutability and restore snapshots in `test_medium_term_memory`,
  `test_identity_rows`, `test_reference_close`, `test_reference_arrivals`,
  `test_reference_mentions`, `test_reference_occurrence_views` and
  `test_selected_nested_meaning`;
- native-reference and legacy-migration fixtures in
  `test_concept_forgetting` and `test_operators_round4a0`, plus the
  observer in `identity_measurement.py`.

**Recommended port rule for Alec:** replace accesses to the retired LTM0
column with the corresponding tagged reading/derivation rows. Preserve
every antecedent, semantic content, occurrence address, evidence, bank
value, immutability extent, restore outcome and rejection. Snapshot both
`refs` and `ref_kinds` of the corresponding retained LTM1 structure when
checking that binding has not rewritten it; do not delete immutability
coverage. Pure assertions that the old empty LTM0 column exists should
become assertions that it is absent, beside positive checks of the new
tagged layout. Record each exact port in the implementation receipt and
source audit. Any change to these semantic outcomes still stops. This is
not authority to preserve a fabricated all-`-1` compatibility column or
to add an untyped occurrence-role edge.

### P5. A complete-model checkpoint test pins malformed saved postings

`test_learning_readiness.py:188–195`, at the end of
`test_native_progress_and_trained_artifact_evaluation_path`, reshapes the
saved `posting_codes` and requires a load failure naming that column.
Unlike the authorized standalone-memory restore ports, this is the
complete-model path and part of a larger training-progress test.

**Recommended disposition for Alec:** retain the full learning-progress
and evaluation case; replace only this obsolete corruption subcase with
a malformed `ref_kinds` checkpoint under the existing `part_store` path,
requiring loud load rejection. Keep all earlier assertions verbatim. The
exact new key/path will follow the decided one-owner export in the 2b
source; no change has been applied here and no learning run has been made.

### Already authorized changes are kept separate

The standalone-memory posting restore ports, the two evidence tuples,
the coarsening test across two passes and §19's snapshot port, and the
reader change in `test_live_parent_pins_its_child` are authorized by the
handoff. Tests solely of adjacency/nudging can be retired and individually
listed: `test_centroid_moves_without_losing_identity_and_caps_containment`,
`test_adjacent_wholes_rewitness_compact_and_checkpoint` and
`test_absent_adjacent_wholes_leave_the_lower_form`. Their removal does not
authorize retiring mixed certificates such as P3 or P5.

This is an assertion-authorization stop found by inspection, not a failed
run or a same-round setup exception. Claude reviews the exact scope, then
Alec decides P1–P5 before implementation resumes on main. There is no 2b
freeze, gate pass or publication to claim. The two arrival notes remain
in 2c; the one full sweep stays at step 2's end. Item 3.0 is not reached.

## 25. Claude's assertion-port review and amended proposals — 2026-10-22

The actual read-only [Claude review](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/claude-review.md)
of input `76de86f7a` confirms the stop required by the assertion rule.
No production/test edit, test run or implementation freeze has occurred.
The already authorized standalone restore, adjacency-only retirements,
evidence, coarsening/snapshot and named role-0 reader ports are not being
requested again. §§20–23 remain decided.

The [amended proposal](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/amended-port-proposal.md)
contains the concrete dispositions for Alec. It supersedes §24's proposed
ports where they differ:

- P1 checks **question kind and zero evidence**, not just evidence, since
  origin withdrawal zeros the evidence of every row. The review also
  identifies the stored-mode reader in `reasoning.py:170`. The proposed
  implementation keeps `ConceptualMeaning`'s existing default on read and
  removes that redundant mode guard, preserving kind/metadata/evidence
  guards and incoming live-mode admission. It stores no replacement mode.
- P2's seven-column checks are correct. P3 also needs the existing
  vocabulary-cached form snapshot to survive unchanged, with the removed
  adjacency key/switch gone; remove the unused config/schema entries.
- P4 snapshots pre-existing tagged rows **by handle**, declares expected
  additions separately, and compares cross-path graphs through their
  address/structure mapping. This strengthens the old all-`-1` snapshots;
  a failure stops. Two address/migration fixtures and the repeated-role
  fixture need typed derivations, not an untyped occurrence edge. Their
  role, address, timing, evidence and restore checks remain obligations.
- P5 covers the complete-model corruption check and the full-owner stale
  posting test. Require the owning LTM1 validator to reject invalid tag
  values and shape by name; preserve the preceding learning-readiness
  assertions and retained-source membership assertions. Keep
  `leaf_complete` as a non-persistent derived buffer so its live assertions
  need no port.
- The review found an LTM0 row-export `refs` check missed by the first
  inventory and a remaining `where` comparison in
  `test_selected_nested_meaning.py:81`. Port the former to absence and
  remove only `where` from the latter, retaining all `.when` checks. The
  `where` conflict is established by inspection; no 2a failure is claimed
  and no additional 2a run is proposed.

One review question is already decided: **§11.1 ends close-time argument
minting in 2b**, when tagged occurrence arguments arrive. It does not move
to 2c. The amended proposal's P6 supplies the requested limited audit of
tests requiring argument seats, symbol-only reads and rejection solely
because the working bank is full. Recommended ports assert the same
retained content and typed derivation through OCC, no new argument seat,
and successful closing with unchanged bank capacities; actual memory
exhaustion still rejects atomically. These are explicit changed
expectations for Alec, not mechanical ports disguised as unchanged checks.

P7 lists further old derivation-root and pending-name reads, and proposes
injecting a corrupt tagged graph for the defensive cycle-reader test
while leaving its reachable-leaf assertion unchanged. The public writer
still rejects self/forward references. The amendments get a focused
read-only follow-up review before Alec's decision; no design implementation
or test port is authorized by this review record alone.

The implementation receipt will enumerate every final changed assertion,
fixture and retired test. No proposed rule permits weakening semantic
checks, changing capacities, promoting earlier evidence or continuing
after an existing-test/production failure. The two R10 notes remain in
2c, one full sweep remains at step 2's end, and item 3.0 is not reached.

## 26. Follow-up review; two decisions for Alec — 2026-10-22

Claude's actual [read-only follow-up](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/followup/claude-review.md)
reviewed input `a9409901c`. It accepts amended P1–P3, tightens P4–P5 and
the close-time-mint ports, and identifies one genuine design choice:
where an occurrence-only argument's occupation would be recorded after
its forced native mint disappears. No code or test changed and no gate
ran. The [final port proposal](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/amended-port-proposal.md)
incorporates the review's corrections; this section recommends decisions,
not authorization already received.

**1. Authorize the corrected existing-test ports, apart from the
occupation decision below. Recommended.** P1 checks retained question
kind plus zero evidence; P2–P3 port the tag-column layout and removed
switch while preserving the form snapshot; P4 compares pre-existing
tagged structure and ports the retired-column/read-root assumptions;
P5 uses owner-validated bad tags/shapes and a no-saved-postings check;
P6 expresses §11.1's OCC/content/no-seat outcomes; P7 preserves pending
word readings and injects corruption only for defensive-reader fixtures.
The receipt lists the individual cases and the already authorized
retirements separately. No capacities, tolerances or unrelated assertions
change. Corrections made explicit after the follow-up:

- The two address/migration fixtures use
  `e = lift(OCC child, SYM predicate)` and `Part(OCC parent, SYM e)`.
  The repeated-edges fixture uses `e = equal(OCC child, OCC child)` and
  `Part(OCC parent, SYM e)`; its two arguments supply roles 0 and 2.
  Its `records_scanned == 2` remains verbatim: the current implementation
  (`Layers.py:9060–9064`) counts the bounded LTM0 address scan, and the
  fixture still has two occurrence rows. Its node/depth/record limits
  stay, as do the separate query work-meter obligations.
- A full-bank successful close preserves `rows` and `placement`, not
  the whole allocation inventory: relation handles advance `next_id`.
  Record exact new occurrence/relation/handle counts from the declared
  layout. Existing `test_complete_derivation_capacity_failure_has_no_prefix`
  retains the real exhaustion/atomicity obligation. The zero-seat test
  writes once; it does not restore caps and write a second occurrence.
- For MTM occurrences, `_derivation_symbol` must give token kind
  precedence over the immutable word reading; an anonymous token gives
  None. A bound use retains its own `Part(OCC used, SYM x)` reading,
  with the identity link in MTM. The arrival must still read `three`,
  never `x`; no antecedent assertion is ported. The pending reading is
  additionally checked unchanged through arrival.
- Invalid tag values are out of range on occupied rows; shape validation
  precedes PyTorch's loader, and both cases raise `ValueError` naming
  `ref_kinds`. Only occupied pre-2b pairs default to SYMBOL; empty tags
  stay zero. Reading rows never become definitions. Defensive cycles are
  planted directly in buffers, while the public writer rejects self or
  forward references, including a newly assigned handle that an existing
  row already references.

The last two bullets preserve the existing semantics and are implementation
consequences, not requests to reconsider the decided layouts. The same
applies to removing the additional stored-mode guard in `reasoning.py`:
its kind, metadata, shape, mask and content checks stay; no replacement
mode is added. The live incoming-mode admission check stays.

**2. Decide the occurrence-only occupation.** In
`test_nested_terminal_mint_keeps_its_word_and_admitted_occupation`, the
existing count/restore checks refer to a freshly minted symbol.
`PartClosing.py:265–271` keys occupations by symbol and skips bare words.
After §11.1 there is no such new symbol. An OCC cannot enter that map
untagged.

**Claude's recommendation, also recommended here:** record no separate
symbol-keyed occupation for this occurrence-only argument; its typed
derivation already carries the grammatical position. Explicitly retire
that test's stored-count and count-restore expectations and check the
retained word and position through the tagged structure, without a new
definition or seat. **Alternative:** record the occupation on the word,
changing the current bare-word rule. A new tagged occupation map would be
a further layout decision and is not proposed for 2b. Nothing is
implemented before Alec chooses.

This stop comes from the handoff's explicit assertion-change rule and
§22's stored-mode requirement. It is not a failed gate or an application
of the same-round setup exception. The repository handoff remains
published at `c39ef9f3a`; these proposals and receipts are on main pending
Alec. After the decisions, implementation, a source freeze, the requested
scoped gates and Claude's source review still precede 2b publication and
the WikiOracle bump. No fresh series or full sweep has run; 2c keeps its
two arrival notes, and item 3.0 has not been reached.

## 27. Claude's review of the 2b port package — 2026-10-22

Reviewed: §§24–26, the [port-preflight receipt](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/README.md),
the amended proposal, both delegated reviews and the amended patch, on
`4eb955198`. The repository hand-off was done as asked: Claude's seven
documents are `c39ef9f3a` on main, verbatim, and work is on main.

**Verdict: the package is sound and complete enough to authorize.** The
audit found what the §23 hand-off missed: existing assertions that pin the
old six-column LTM1 layout, the stored mode, the removed nudge's switch,
the empty LTM0 `refs` column, the saved postings on the full-model path,
and the close-time mints that §11.1 ends. The two delegated reviews'
corrections are right, and I checked the ones the ports rest on:

- `withdraw_origin` downgrades only facts, so a withdrawn question keeps
  kind `question`; P1's kind-plus-evidence check is the right port.
- `_derivation_symbol` reads the raw role 0 before MTM. With a pending
  name's reading row `Part(OCC x, SYM x)`, it would return the word
  instead of `None`; the follow-up's precedence rule (token kind first for
  pooled occurrences, `None` for an anonymous token) keeps the arrival
  tests' antecedent and is required.
- The `round4a0` fixtures plant an occurrence directly in role 0 with no
  derivation, which the decided layout forbids; the typed rewrites are the
  right port.
- `test_selected_nested_meaning.py:81` compares a `where` buffer that 2a
  removed. It is outside 2a's gate files, so published 2a very likely has
  one failing test there. 2b's port covers it; no 2a rerun.

**One addition: how an occurrence-only argument's position is read.**
`PartClosing` records occupations under an admitted object symbol and
skips a bare word, and `PartRead.argument_reader` classifies a retained
pair from its arguments' occupations, all of which must carry the
operation's position. After §11.1 an argument tagged OCC has no symbol and
no occupations of its own, so the reader would classify nothing. The
recommended disposition (no separate occupation) therefore needs its
other half stated: the tag-aware reader resolves an OCC argument through
its reading row to the word it reads to, and uses that word's grammar
occupations (`PartsOfSpeech`, keyed by word). That is the §10 contract's
own source and needs no new map. The nested-terminal test then checks the
word and its position through that read, and its symbol-keyed count and
restore checks are retired.

**Recommendation on sequencing.** 2b lands as two frozen commits with one
receipt: first the layout (LTM1 under LTM0, tags, reading rows, the
removals, evidence, forgetting and ports P1–P5, P7), then the end of
close-time minting (P6) on that source. The second is a behavior change
with six changed expectations and deserves its own diff. Step 2's one
sweep stays at its end.

**Decided (Alec, 2026-10-22).** P1–P7 are authorized as amended, with
every changed assertion and retired test listed in the receipt; any
further change or failure still stops. An occurrence-only argument's
position is read through its reading row to its word's grammar
occupations; no separate occupation is stored, and the nested-terminal
test's symbol-keyed count and restore checks are retired.

## 28. Reading-row discrimination — 2026-10-22; reviewed by Claude, for Alec

§27 and Alec's handoff authorize P1–P7, including both reviews'
corrections, and decide the occurrence-only occupation read. Those choices
stand. Claude's six incoming documents were committed unchanged on main
as `00bc915a9`. Production and tests still match published 2a
`0b030e3eb`; no production/test edit, gate, failure or 2b freeze has occurred.
The [2b receipt](../benchmarks/2026-10-22-item3-5-step2b/README.md)
retains the static audit and the actual
[Claude review](../benchmarks/2026-10-22-item3-5-step2b/layout-preflight/claude-review.md).

**One missing layout fact.** The decided tags identify what an argument
references, but do not distinguish an occurrence's reading row from an
ordinary relation with that occurrence as its first argument. In §20's
worked layout:

```
rF = Part(OCC F, SYM felix)   # F's reading
e  = Part(OCC F, SYM runs)    # the lift derivation
rS = Part(OCC S, SYM e)      # S's reading
```

`rF` and `e` have the same tag pattern. Excluding the current edge, or
selecting an unreferenced row, can distinguish this small example, but
does not determine the answer when there are further incident relations.
A completed relative close remains an independent LTM1 edge without an
LTM0 sentence row (`ClauseRows.write_clause`, `retained_pair` path).
Its `(OCC, SYM)` arguments do not distinguish it from a reading either.
More decisively, a reading and an ordinary relation with the **same**
tagged arguments coalesce under `LTM1.find`; one row would then have two
different retention obligations.

Today the occurrence's sidecar `role_refs[0]` identifies its root. Removing
that structural pointer requires a recoverable replacement for the
P4/P7 readers, the new OCC-to-word occupation read and the staged
forgetting policy. A derived cache cannot supply a missing distinction
after restore. Word occupations are not exclusive row types; direct
semantic pairs need not have a formal operation, and occupation recording
may be frozen (`PartClosing.preflight`, `PartsOfSpeech.add`). MTM's kind
precedence settles the working token read while its entry exists, but MTM
does not provide the permanent reading-row discriminator.

**Claude confirms the gap and corrects the diagnosis.** This is ambiguity,
not an unavoidable recursive loop: a reader can exclude the edge it is
currently classifying. Also, the existing forgetting code protects a child
while its parent lives, so the example of dropping F's reading while e
survives is not an observed production failure. These corrections do not
resolve multiple eligible rows or a coalesced reading/relation pair.
No failing executable diagnostic is claimed.

**Recommendation to Alec: add `READING` as a third `ref_kinds` value.**
It denotes an occurrence handle in argument 0 of its reading row, with a
SYMBOL argument 1: `Part(READING F, SYM felix)`. Ordinary uses of F remain
OCCURRENCE. No column, identity label, working vector or capacity is added.

- Exactly one live reading row is permitted per occurrence. Re-witnessing
  that same reading remains allowed; writing or loading a conflicting
  reading fails before mutation. With no reading row, the occurrence is
  read as a gist.
- Pair keys include the tag, so a reading and an ordinary relation with
  equal integer arguments are separate rows. Readers and restore recover
  the distinction without time, handle order or lexical guesses.
- For address validity, READING resolves to LTM0 like OCCURRENCE. Its
  owner endpoint is a lifetime association, not a self-protecting edge;
  its SYMBOL root and that root's dependencies remain protected until
  the reading is dropped. This preserves §§21–23's staged policy.
- Reading rows remain excluded from definitions. All tag-aware readers,
  validation and retention paths must explicitly handle the third value.

This adds an explicit reading role to the tags and therefore amends
§10 S2's claim that the reading is indistinguishable membership. It is
**not implemented or authorized by the existing two-tag decision**.
Restoring a root column on LTM0 would be a larger alternative and would
reverse the decided structural move and absence ports; it is not
recommended.

If approved, P2 still has seven columns. P5 rejects invalid tag values and
shapes with the three valid values accounted for. P4/P7 identify readings
through READING; every other authorized semantic assertion stays. New
coverage should check a reading and ordinary relation with identical
arguments, restore, removal of a reading without inventing a substitute,
the same integer in both namespaces, exclusion from definitions, and loud
rejection of a conflicting second reading. A retained occurrence without
a reading must have no claimed word occupations.

**Decision requested:** approve this discriminator, or have Claude revise
the layout with an alternative that uniquely recovers readings and their
retention obligations. Work stops here under the standing rule for a new
spec choice. The two frozen commits and their per-commit scoped gates
remain pending; no earlier evidence is promoted. P1–P7 are authorized,
not awaiting approval again. The two arrival notes remain in 2c, with
step 2's one full sweep at its end. Item 3.0 has not been reached.

## 29. Claude's review of §28 — 2026-10-22

**The gap is real, and it is my layout's.** §20's worked example gives an
occurrence's reading row and an ordinary derivation the same tag pattern,
and §10 S2 called the reading "the same relation as 'Felix is a cat'".
Today `role_refs[0]` on the occurrence's side is the one-way pointer that
names the root; once it moves into LTM1, nothing persistent says which
`(OCC, SYM)` row is the reading, and a reading and a relation with equal
arguments coalesce under `LTM1.find`. Codex was right to stop, and the
delegated review's corrections (no true recursion; today's protection
narrows the coarsening example) are right too.

**`READING` as the third `ref_kinds` value is the right fix.** It is a
property of how argument 0 is referenced, not a new relation type, so the
one-relation design stands. It needs no column, keeps LTM0 at the decided
row, survives forgetting and restore, and separates the coalescing keys.
It also makes §21's staged rule exact: a READING argument is not a
protecting reference (the occurrence owns its reading), while the reading
protects its SYM root and the root's derivation.

**One correction: re-witnessing replaces the reading; it must not fail.**
§28 says writing a conflicting reading "fails before mutation". But a
perceived sentence's address comes from its source text
(`Occurrence._address_for_write`: document, position, source content key),
so the same sentence read again in a later epoch reaches the same address,
and `append_meaning` overwrites its slots and its meaning, root included:
the last reading wins today. As the grammar learns, a re-read's parse
changes, and the proposed rule would turn every such re-read into a loud
failure. So: on re-witness, the new reading row replaces the old one,
matching the overwritten content; the old reading is retired, and its root
is released for the next pass. Two readings for one occurrence in a single
write or a single load remain a loud failure, as corruption.

**Decision for Alec:** approve `READING` with that correction. Every other
§28 consequence stands: the reading's exclusion from definitions, three-
value validation in P5, P4/P7 reads through READING, and the listed new
coverage, to which add one case: a re-read with a different parse replaces
the reading and leaves exactly one.

## 30. Alec: "Sentence derivation should not be in LTM" — 2026-10-22

Asked to approve §28/§29's `READING` tag, Alec answered: "Sentence
derivation should not be in LTM." This withdraws the premise of §10 S2's
layouts, §20's worked layout and §§28–29: there are no derivation edges in
LTM1 to tell a reading apart from. The derivation stays in working memory,
the open sentence's journal, and ends with it. The `READING` question
lapses; 2b stays stopped until the layout is restated.

Readers of the stored derivation today, which the restated layout must
replace or retire:

- `PartRead.occurrence_roots` and `occurrence_memberships`, which feed
  the order-0 placement (`MereologicalCodes.occurrence_terms`) and priming
  diffusion (`Priming.diffuse`): which words a stored sentence contains.
- `PartRead.expression_components`, read by thought queries
  (`Queries._with_components`) and answer matching (`BindingAnswers`): a
  stored sentence's components and head references.
- `ReferenceClose.accessible`, which finds bind candidates inside earlier
  sentences by walking their derivations, and
  `ReferenceParticipants.nonparticipants`, which classifies retained pairs.
- §21/§23's staged forgetting, which drops structure before the sentence.

Claude's reading, for Alec to confirm: LTM keeps whole occurrences, both
sentences and their noun-phrase mentions, as LTM0 entries; each mention's
reading, the word or symbol it reads to; each mention's membership in its
sentence, `Part(OCC mention, OCC sentence)` ("a word is part of its
sentence"); and the relations a sentence asserts, such as equalities and
definitions. A sentence's internal structure, who did what to whom, is
recovered when needed by inverting its stored concept, item 3's invertible
path, not read from stored edges.

**Decided (Alec, 2026-10-22).** "If we need the original sentence, just
reconstruct from the vector. It will not be exact." No derivation is
stored, and nothing replaces it: a stored sentence's structure is
recovered, when needed, by reconstructing from its vector through the
existing reverse path, inexactly.

## 31. 2b restated without stored derivations — Claude, 2026-10-22; for Codex's review

§30 removes the derivation from LTM. This section restates what 2b stores
and where each reader of the old derivation now reads. It replaces §10
S2's layouts, §20's worked layout and §§28–29. Decided items that do not
depend on the derivation stand: the class names, `where` gone, tagged
references, the evidence rule, mode not stored, saved postings removed,
adjacency removed, and P1, P2, P3 and P5.

**What LTM stores.**

- **LTM0**, the occurrences, as today: sentences, and noun-phrase mentions
  as their own rows, each an opaque concept. A mention is already
  addressed under its sentence (`ClauseRows.write_clause` writes it with
  document key `('embedded', document, position)` of that sentence), so
  "this mention is part of that sentence" is read from addresses and needs
  no edge.
- **LTM1, three kinds of row**, with tagged references:
  1. *Word membership*, `Part(SYM word, OCC sentence)`, one row per
     distinct word of the sentence, written at its close from the
     journal's leaves: a bag, with no order and no operation. It replaces
     the derivation walk (`PartRead.occurrence_memberships` through
     `word_rows`) as the source for placement
     (`MereologicalCodes.occurrence_terms`), priming diffusion
     (`Priming.diffuse`) and the leaf-index rebuild at load. This is the
     "a word is part of its sentence" edge the architecture already names.
  2. *Mention reading*, the word or symbol a mention reads to: for names,
     kinds, pending names, the arrival's name match and
     `_derivation_symbol`'s kind for an unpooled occurrence.
  3. *Asserted relations*: what a sentence asserts, such as an equality,
     a part or a definition, as today's relation closes write them.

**The discriminator question survives in one place.** A mention's reading
and an asserted relation about that mention can share a tag pattern: "x is
three" relates x's mention to three, and x's reading relates the same
mention to the word x. Recommended: `READING` as a third `ref_kinds` value
on argument 0 of a mention's reading, as §28 proposed, now needed only for
mention readings. One reading per mention; a re-read replaces it (§29). A
READING argument is not a protecting reference. Alternative: keep each
mention's word in its row's sidecar binding, where the pending name lives
today (`_forward_references`, indexed by `ClauseRows._forward_addresses`),
and store no reading row at all.

**Where each old reader now reads.**

- *Placement, priming and the leaf index* read word membership.
- *Thought queries and answer matching* (`Queries._with_components`,
  `BindingAnswers`) read mention readings and asserted relations first;
  anything finer about a stored sentence is reconstructed from its vector,
  inexactly. Tests that assert exact stored components need ports.
- *Bind candidates inside an earlier sentence*
  (`ReferenceClose.accessible`) are that sentence's mention rows, found by
  address. Mention rows are written only for participant positions, so
  `ReferenceParticipants.nonparticipants` and the pair classification it
  uses (`PartRead.argument_reader`) retire with the stored pairs.
  §27's occupation read through the reading row goes with them.
- *Close-time minting* (§11.1) ends as decided, more simply: there are no
  derivation arguments to mint.

**Forgetting, restated.** §21/§23's staged rule had a sentence's
structure dropped before the sentence. Now a sentence's stored structure
is its word memberships. Recommended: they are its first stage, dropped
before the sentence, leaving the vector as its gist; its mentions are
occurrences in their own right, forgotten by their own value while
nothing protects them. A sentence never protects its mentions, since no
edge points from it to them. One boundary call is one pass, as decided.

**Ports to re-audit.** P1, P2, P3 and P5 stand. P4, P6 and P7 assumed
derivation rows and reading roots; Codex re-audits them against this
layout and lists each changed assertion for Alec. The two arrival notes
stay in 2c; step 2's one sweep stays at its end.

**For Codex's review, then Alec:** the word-membership row as the durable
source; `READING` versus keeping the word in the sidecar binding; the
query fallback to reconstruction; the restated forgetting stages; and any
reader of the stored derivation this list misses.

## 32. Codex's review of §31 against the code — 2026-10-22; stopped for Claude and Alec

**§30 is accepted: no stored sentence derivation and no exact-replay
promise. §31 is not yet a complete replacement contract.** The remaining
questions concern which occurrences exist, how a word reaches a bank row,
the distinction between a reading and an assertion, and consumers that
currently need roles rather than a bag of words. None is implemented here.

Review input is main `e0d474090`, plus Claude's incoming §§29–31. Production,
tests and configuration still match published 2a `0b030e3eb`. The
[static review record](../benchmarks/2026-10-22-item3-5-step2b/restated-review/README.md)
retains the incoming patch, source hashes, reader search and
[exact port proposals](../benchmarks/2026-10-22-item3-5-step2b/restated-review/proposed-ports.md).
This is file/AST inspection, not a diagnostic run or evidence that a test
passed or failed. No implementation, freeze, gate, Claude invocation,
publication or WikiOracle bump occurred. Work stays on main.

### R1. Word membership needs a sentence occurrence that does not always exist today

`ClauseRow.write_clause` (`retained_pair`, lines 461–464, 481–482,
509–513) returns a completed relative close as `('sym', edge)` and writes
**no LTM0 row for that sentence**. Thus `Part(SYM word, OCC sentence)` has
no target for those closes. The older `PartRead.occurrence_roots` explicitly
handles both LTM0 roots and independent LTM1 roots; replacing just its walk
misses the latter. Also, a relative `Clause.point` is normally None
(`ClauseJournal`, final `Clause` construction); the journal composition's
point, not an argument or a guessed bank row, is the available fused value.

**Recommendation:** retain an opaque LTM0 occurrence for each completed S
in a sentence reading, relative closes included, with its journal value.
Retain the asserted relation separately and keep the existing relative
return value for callers that ask for that assertion. The close must also
expose the occurrence in its written-row/address result for membership,
frames and expectation provenance. Direct semantic Part writes with no
sentence reading need no invented sentence or lexical bag. Nested S
occurrences get their own bags; NP mentions do not become extra sentence
contexts. This is a proposed amendment, not a code fact or permission to
change the literal row counts. P4/P6 below list the resulting count ports.

For a re-read, replace the sentence's bag from that reading, not a union
of obsolete parses, and preserve the sentence address and existing witness
accounting. Membership evidence must not duplicate the asserted truth's
authority. Recommend zero truth poles for these incidence rows and using
the occurrence's witness count/recency once per distinct word, rather than
counting both the assertion and its sentence as two contexts. Claude needs
to specify how that ownership survives re-read and load.

"Asserted relations as today" also needs a narrower definition:
`PartClosing._relations` includes `lift` and `verb`, and `ClauseJournal`
promotes an enclosing fold to `relation='operator'` when a child is a
relation (lines 513–530). Retaining every such pair would retain precisely
the derivation §30 removes. Recommend keep the explicitly asserted
relations (including the existing generic inclusion and implication),
but do not retain a lift/verb/operator pair merely to connect its parsed
operands. The P4/P6 layout proposals use that interpretation.

### R2. Specify what SYM word means and how it is read

The word ID is not in general the selected object's ID or its bank seat.
`SentenceFixture.program` carries all three separately (`word_ids`,
`concept_ids`, `word_rows`); `Composition.leaves` contains input positions.
`PartRead.word_rows` currently follows **selected native arguments** to
seats. `Interpret.lexical_identities` and `word_surface_for_row` expose
defined objects as lexical rows; a defined word itself need not have such
a row. `DefinitionIndex.objects(word)` can contain several objects and
`deref` rejects an ambiguous selection. A direct `_csw_row_of(word)` is
therefore not the replacement for today's walk.

**Recommendation:** the durable bag names the actual word symbols carried
by the kept reading, including markers; never infer words from spelling,
occupations, a content match or an OBJECT carrier with no word ID. Keep
membership separate from noun-site eligibility. For the bank-facing view,
explicitly decide whether a word conducts to all its currently seated
lexical interpretations. I recommend that interpretation-independent word
view, with no invented selected sense and no new seats. It preserves the
unique-definition expectations in the authorized P5 fixture; it deliberately
does not preserve historical sense selection for ambiguous words. A new
ambiguous-word/restore regression should state that consequence. It must
not be implemented by silently calling `deref` or changing P5's four
leaf-set assertions.

`MemoryIndex` has **three role postings**. A sentence bag can supply role 0
of its fused occurrence; it cannot be copied into every role of an open
three-slot question/estimate and called complete. Specify these readers
separately, retaining the existing work meter and explicit incomplete
result when there is no exact source. Placement and priming must use the
same occurrence key, recency and witness count; today's synthetic LTM1
root ordering is no longer an independent source of sentence recency.

### R3. Mention-only READING is preferable; bag ownership also needs a rule

Recommend §31's **READING on mentions only**, rather than another general
sidecar structural pointer. `_forward_references` is only pending-name
metadata: it is not the durable reading of every ordinary mention, kind
or unpooled occurrence. The sidecar alternative therefore needs a new
authenticated key, not simply reuse of the existing index. Keeping
`metadata_required`/`semantic_fingerprint` is still necessary for the
remaining bindings, scope, source text and expectation provenance.

The proposed reading target is a pre-existing native head/kind when the
mention has one, or its objectless word when it has only a name; a bare
numerical operand has no fabricated target. It is not a reference to a
sentence derivation. MTM kind takes precedence, including anonymous None,
as already decided in §27. A later refinement does not rewrite the word
reading. One live reading per mention, replacement on re-read, no reading
indexed as a definition, and validation before mutation remain appropriate.
Retiring `PartRead.argument_reader` does not retire the **live**
`ReferenceParticipants.participant_terms`/`clause_participants` readers.

One distinction is still unspecified: a word membership `(SYM w, OCC s)`
can equal an asserted Part with the same tagged arguments. `LTM1.find`
coalesces pairs. If that is one semantic row, deleting it as owned detail
must not erase the independently asserted fact. **Recommendation:** make
the ownership rule explicit, including this coincidence, before choosing
the final tags. Prefer a separately recoverable membership use when its
lifetime differs; do not add a tag by inference in implementation. This
is a bag/authority question, not revival of §28's sentence-reading question.

### R4. The embedded address is derivable, but is not a participant classifier

For an addressed close, the child document digest can indeed be computed
from the stored parent document digest and sentence index:
`document_digest(('embedded', parent_document_digest, parent_index))`.
There is no need to invert a hash. However, §31's claim that these rows
are written only for participants is false today:

- `ClauseJournal.retain_completed` keeps nested S events; its pending
  predicate branch also retains an unindexed predicate (lines 684–685).
- `write_clause` places **all** such children in the enclosing close's
  embedded namespace. It is not an immediate-parent map for a nested S.
- Unaddressed writes use `('direct', kind)`, while
  `LTM0.bind_constituents` uses a separate `('constituents', ...)` scheme.
- A changed parse can leave an older child's different content-addressed
  row in that same namespace. Namespace membership alone does not establish
  that it belongs to the current frame's reading.

**Recommendation:** use the existing frame's written activations for its
accessible mentions, and keep native participant eligibility supplied by
the kept journal. Do not offer every order-1 row or every frame symbol:
a sentence can inherit order 1, and a native predicate can be an individual.
If §31 instead requires rebuilding all accessible mentions solely from
addresses after restore, Claude must specify a durable distinction between
mentions and other embedded rows, current re-read membership, and direct
writes. Do not invent an OCC membership edge, label, or address change to
fill this gap. A bag of all words is not that distinction.

### R5. Reconstruction fallback is an adapter to build, not an existing component reader

`Queries._with_components` calls only `PartRead.expression_components`.
`BindingAnswers._matches` calls it too and also follows meaning references.
Neither currently falls back to generation. `MemoryIndex.unfold_idea`
does invert a value through the current generate path, but returns a flat
`codes`/`operations` result, not the `kind`, ordered `operands`,
`reference` and `head_reference` graph these consumers expect. It needs
current priming (eligible rows exceed 1), a compatible dictionary width,
the stored order and a work allowance; its absence/incompleteness is
explicit. `configure_model_index` currently walks retained references or
matches a unique terminal; it does not call this inverse.

**Recommendation:** specify one bounded transient component view over that
existing reverse path, charging both record reads and unfolds. Do not store
the result, reconstruct during a membership/index rebuild, feed expected
leaves to the inverse, or describe a reconstructed operand as the original
occurrence identity. Direct mention/token identity and explicit asserted
relations remain the exact answer path. An inexact reconstruction may
propose content, but should not itself certify that a particular historical
mention was the antecedent. Preserve explicit incomplete results.

Exact-component tests and their proposed dispositions are P7 below. The
standalone controlled inverse tests in `test_stored_idea_generation` and
the working component fixtures in `test_thought_binding` do not assert a
stored derivation and should stay unchanged. Production cold-start tests
must not acquire a forced inverse merely to reproduce the old tree.

There is a further scope question: stored pending rows and
`LTM0.bind_constituents` retain role-to-occurrence references, including
thought-produced meanings. Removing only `PartClosing`'s pairs would leave
this second structural graph. Conversely, dropping those references also
changes the pending-parent contract, question filling and thought evidence.
Recommend distinguish **semantic bindings/explicit relation arguments**
from parse structure, retaining the former with authentication and no
producing-operation tree. Claude needs to state that boundary before the
raw pending-parent and nested-thought assertions can be ported. They are
listed as held, not silently relaxed, in the port audit.

### R6. Missing binding reader: predicate-role prediction and expected roles

`ReferenceClose._participants` (lines 66–95) derives subject/object
observations from stored pairs marked state/situation.
`prediction` weights them by edge access count; `expected_roles` uses
their recurrence to open missing sites in `ClauseJournal`. `accessible`
also recovers a candidate's predicate from those pairs. These are not
just candidate enumeration. Deleting the pairs makes these features
disappear even if the same mention candidates survive.

**Recommendation:** include these consumers in the bounded transient
reconstruction contract, with unavailable role evidence explicitly unknown.
Do not substitute co-occurrence for a predicate role or install a new
durable role-count table. Claude/Alec must decide the resulting scope and
budget, including whether to retain current-frame role evidence directly
while a frame is live. The four occurrence-view cases require the same
positive/zero predictions and predicate, and bind-thought tests exercise
recurrence: no outcome assertion is proposed for weakening. Any fixture
adapter needed for reconstructed evidence must be approved explicitly.

### R7. Forgetting must distinguish owned detail from protecting assertions

Recommend the stage order **source wording → sentence word memberships →
opaque sentence**, at most one stage for a given occurrence per boundary
call. Source wording is already a stage (`Forgetting.forget`, lines
290–298); §31 does not say whether it disappears. Membership removal
leaves the occurrence's content, evidence and address untouched. A mention
has its own lifetime; the sentence's existence does not pin it. An
explicit asserted relation or live root still protects its targets
transitively; a released target is reconsidered only on the next pass.

Owned detail pointing to its owner must not protect that owner forever.
A reading's target and a membership's word remain protected until the
owning detail is removed, subject to R3's independent-assertion case.
Membership deletion cannot count as deleting the sentence seat. The
old child-first rewrite/cascade routine (lines 300–390) is not this policy:
it rewrites parent references and may cascade through them in one pass.
The old P4 coarsening fixture therefore needs a different port from §21's
derivation-stage port. The proposal separates bag coarsening from explicit
relation protection; its exact assertions are listed in P4 below.

### R8. Remaining readers, writers and restore boundaries

In addition to §31's list, the implementation audit must cover:

- `MemoryIndex._meaning_leaf_terms`, `configure_model_index`, and
  `Forgetting`'s leaf-completeness/utility reads: distinguish symbol,
  occurrence, reading and membership uses; no OCC-to-bank lookup.
- `ClauseJournal`'s stored-relative recovery (lines 399–411),
  `ClauseRows.relation_operands`, `_native_reference_indexes`,
  `_premises`, `_part_witnesses`, `find_relation`, and
  `Queries.premise_evidence`: asserted relations still have exact tagged
  arguments, but a sentence no longer names its derivation root.
- `LTM0.read_structure`, `bind_constituents`, local-reference/sidecar
  validation, origin withdrawal and checkpoint rekeying: R5's semantic
  reference boundary applies here, not only at a sentence close.
- `ConceptForgetting.protected_concepts`, `Priming.part_edges`,
  `DefinitionIndex.rebuild`, allocator record/order reads and
  `IdentityRows.witnesses`: raw integer pairs must not turn an occurrence
  or reading into a symbol, definition, taxonomic edge or bank seat.
- `ReferenceClose.token_value`, grouping baselines, commit/arrival kind
  reads, `ThoughtStream`'s reads of returned frame meanings, and the
  native observer in `identity_measurement.py`: a root, a mention reading
  and a bound token are different sources.

Both tables must load before derived indexes rebuild. An old checkpoint
can contain both composing pairs and asserted pairs; LTM1 does not store a
producing operation. Converting every old pair to SYMBOL tags, by itself,
would keep the forbidden sentence derivations. **Recommendation:** require
an explicit, versioned migration disposition; preserve authenticated
assertions where identifiable, and fail loudly for a legacy layout whose
classification cannot be established. Do not guess from occupations or
silently drop all its Part rows. This question is separate from the
already authorized legacy tag default and class-path compatibility.

Finally, the old two-commit boundary needs to be restated: the
`PartClosing._plan_term` path creates both composing edges and the
argument mints. Removing the former removes the latter's original reason
to run. Recommend Claude name the exact mint path, if any, that survives
the first frozen layout commit. Do not retain gratuitous minting merely
to manufacture a second commit, or test that intermediate source against
the final layout counts. The two-commit instruction is not silently
collapsed by this review.

### P4/P6/P7 re-audit and stop

The [port ledger](../benchmarks/2026-10-22-item3-5-step2b/restated-review/proposed-ports.md)
quotes each proposed changed assertion at this source, defines its exact
replacement, lists fixture rewrites/retirements, and identifies the checks
held for the decisions above. It supersedes only the derivation-dependent
parts of the earlier P4/P6/P7 proposal. It is **not authorization** to apply
those changes. P1, P2, P3 and P5 stand; all unlisted assertions stay.

Claude should answer R1–R8 and correct these code facts; Alec decides the
open choices and exact ports. No implementation, freeze or gate before
that answer. The requested two-commit sequence remains pending that
restatement. The two arrival notes stay in 2c and step 2's full sweep stays
at its end. Item 3.0 has not been reached.

## 33. Claude's answers to §32 — 2026-10-22

**Codex is right on all eight points; §31 was too thin.** Two of them
reach decided behavior, so they go to Alec first; the rest are answered
below.

### Decisions for Alec

**D1. The structure of an asserted expression.** Equality substitution
(`ThoughtBinding.integrate`) proves "y is three plus two" from "y is x plus
two" and "x is three" by comparing the stored components of "x plus two"
and "three plus two": same operation, equivalent operands. Thought queries
supply those components (`Queries._with_components` through
`expression_components`). The 3.5 decomposition gates exercise exactly
this. With no stored structure at all, the operands of an asserted
equality are opaque, and substitution would need an exact reconstruction
that does not exist.

- *Recommended:* keep the composition of an **asserted relation's
  operands**, the expression "x plus two" inside "y is x plus two", and
  nothing else. That structure is the content of what was asserted, not
  the parse of the sentence: no `lift`, `verb`, `surface` or operator pair
  that only connects a sentence's parsed parts.
- *Alternative:* store no expression structure, reconstruct operands from
  the vector inexactly, and move the decomposition gates to item 3 with
  the invertible path.

**D2. Role knowledge.** `ReferenceClose._participants` reads stored
subject/predicate and object/predicate pairs. From them come
`expected_roles`, which opens an elided object (G3's "four role witnesses
open the elision"), and `prediction`, bind's evidence of which individuals
fill a predicate's roles. These are derivation pairs.

- *Recommended:* expected roles come from the predicate word's grammar
  occupations (`PartsOfSpeech`: how often it heads a VP with an object),
  a §10 grammar-only source already recorded; and bind's role prediction
  from stored pairs retires, leaving its expectation term to priming,
  which already feeds it. The knowledge that dogs chase then lives where
  meanings live, in conceptual placement and later `intuit`. The fixtures
  that plant role witnesses, and the tests asserting stored predictions,
  need ports.
- *Alternative:* keep subject/predicate pairs as asserted relations, a
  predication's one retained fact.

### Answers

- **R1.** Agreed: every completed S in a kept reading gets an LTM0
  occurrence, relative closes included, with its journal value; the
  asserted relation is retained beside it. Asserted relations mean the
  explicit ones (equality, part, whole, inclusion, implication), with
  D1's operand expressions; never a `lift`, `verb` or operator pair kept
  only to connect operands.
- **R2 and R3: the bag is owned detail of its sentence, not LTM1 rows.**
  It names the word symbols the kept reading carried, markers included,
  from the journal's word ids; nothing inferred. Store it with the
  sentence's own detail, beside its source wording, authenticated with
  the sidecar: a re-read replaces it, a load restores it, and it can never
  coincide with an asserted Part. This withdraws §31's `Part(SYM word, OCC
  sentence)` rows and settles R3's coalescing. The bank-facing view
  conducts a word to all its seated interpretations, as Codex recommends;
  no sense selection is stored. Only role 0 of a fused occurrence is
  supplied by the bag; open three-slot rows keep their explicit
  incomplete reads.
- **R3.** `READING` on mentions only, with Codex's targets: a
  pre-existing native head or kind, else the objectless word; MTM kind
  first. A later refinement does not rewrite the reading.
- **R4.** Agreed, and §31's claim was wrong: embedded rows include nested
  S and predicates. Accessible mentions come from the frame's written
  activations, as production already does, with participant eligibility
  from the kept journal. No address classifier.
- **R5.** No reconstruction adapter in 3.5 beyond what D1 needs: a stored
  predication's components read as explicitly incomplete. Semantic
  bindings stay: pending names, a question's referent slots, bound roles,
  and thought-produced meanings' references, authenticated, with no
  producing-operation tree.
- **R6.** D2.
- **R7.** Agreed: source wording, then the word bag, then the sentence,
  one stage per occurrence per boundary call. Owned detail never protects
  its owner; asserted relations protect their targets.
- **R8.** Agreed on the reader list. Migration: no phase artifact is
  registered (the rename audit found all eight entries null), so a pre-2b
  LTM checkpoint is rejected loudly by version, with no classification
  guessed. Sequencing: removing the composing edges removes the argument
  mints in the same code path, so **2b lands as one frozen commit**, not
  the two I recommended in §27; no mint path survives to separate them.

P4, P6 and P7 follow these answers and D1–D2; Codex restates the port
ledger after Alec decides.

## 34. Alec's answers to D1 and D2: thread through identity — 2026-10-22

Alec, on D1: "The input looks like two sentences. LTM0 keeps absolute
truths, though, and at least one of those is an LTM1 relation." On D2:
"Verbs operate on subjects, but you might as well thread them through
identity."

Claude's reading. Neither answer keeps stored structure. "y is x plus two"
and "x is three" are two LTM0 absolute truths, and the equalities they
assert are LTM1 relations between whole entries. What connects them is
identity: x's mention in the first sentence is bound to x's token, and the
second sentence's equality refines x's token to three. A predication works
the same way: since NP plus VP lowers, a closed sentence is an occurrence
of its subject, so binding the sentence to its subject's identity records
who did what, and an object is a mention bound to its own identity inside
that occurrence. A verb is then the difference between occurrences of one
identity, which is step 4's construction. D1's recommended keeping of
operand expressions is withdrawn, and so is D2's grammar-occupation source
for expected roles.

**What does not yet exist.** Equality substitution
(`ThoughtBinding.integrate`) compares stored operand components, and bind's
role evidence (`ReferenceClose._participants`, `expected_roles`,
`prediction`) counts stored subject/predicate pairs. The decomposition
gates and G3 depend on both. Their identity-threaded replacements are new
mechanism: recording a closed sentence as an occurrence of its subject's
token, substitution through tokens and equality relations, and role
evidence read from identities' occurrence histories.

**The open choice is sequencing**, Alec's to decide:

- *Recommended:* build identity threading first, then remove the stored
  derivation. 2b lands the storage changes that do not depend on removing
  the derivation readers: tags, mention readings, an occurrence for every
  completed S, the removals, evidence and the forgetting stages. Step 4 is
  brought forward with the identity threading. The derivation edges and
  their readers are removed once the replacements pass the same gates.
- *Alternative:* remove the stored derivation in 2b now, accepting that
  the decomposition gates and role evidence fail until identity threading
  lands.

**Alec (2026-10-22), on what identity threading means:** "Is this for when
the VP acts on identity instead of the identified NP? That's better long
term, because if the verb acts on a pronoun, that is not the right/complete
subject." Yes. Today the verb composes with the subject mention's own
content, so in "it runs" it acts on a pronoun's near-empty content.
Threaded through identity, it acts on the subject's identity, whose content
is the union of all that identity's occurrences (`token_value`), so after
"the black cat sat" it acts on the whole cat. Two stages: first, record a
closed sentence as an occurrence of its subject's identity, which is enough
for role evidence and for substitution; then let the VP compose with the
identity's value, which needs the subject resolved before the verb applies,
at the NP's determination rather than only at the close. The second is
step 4's.

**What MTM holds, and Alec's name/value proposal (2026-10-22).** Today an
MTM entry holds pointers only (`MediumTermMemory`: "no stored working
content"): a token keyed by its introducing occurrence, with the addresses
of the occurrences bound to it, a kind, a count and a time. Its value is
computed on each read as the union of those occurrences' stored contents
(`ReferenceClose.token_value`), and bindings are made only when a sentence
closes. Alec: "I think it is a value store, that contains a copy of the NP
vector, and since there may be multiple, it makes sense to have MTM be a
name/value store," with the test case "alec went to town, and then he was
attacked": he is attacked in town only if the first verb acts on alec and
that value is written to "he" before the second verb acts on it.

Claude's reading. The case needs two timings, not two further pieces of
information: the first verb's result is written to alec's entry before
the second clause reads it, and "he" is bound to alec before the second
verb applies. Today neither holds within a sentence: the second verb
composes with the pronoun's own near-empty content, and the first clause's
result reaches alec's token only as an occurrence pointer at the close.
One refinement to the proposal: key the value by identity, and map names
to identities, so "alec" and "he" read one current value rather than two
copies that can drift apart. The current value is the state after the
latest verb, which also handles change ("went to town, then went home"),
where a union of occurrences would hold both places; the history stays in
LTM0. Storing a value reverses MTM's "no stored working content", which is
Alec's call.

## 35. Decided (Alec, 2026-10-22): MTM stores working content

"MTM should store working content." This reverses the bind plan's §10.3
rule ("no working vector") and `MediumTermMemory`'s "no stored working
content". With §34's refinement, accepted with it:

- **Two maps.** Names map to identities: a binding records that "he" is
  the same identity as "alec". Each identity holds one current value.
  Every name bound to an identity reads that value; no copies are made.
- **The value is the current state**, the result of the identity's latest
  verb. At an identity's introduction it is the introducing noun phrase's
  vector. When a verb acts on the identity, its result becomes the new
  value. The history stays in LTM0, one occurrence per event, so change
  ("went to town, then went home") holds the latest place, not both.
- **The timing the test case needs**, for "alec went to town, and then he
  was attacked": the first verb's result is written to alec's value before
  the second clause reads it, and "he" is bound to alec before the second
  verb applies, so that verb acts on alec in town. Both happen inside one
  sentence, so the value and the binding must be live before the sentence
  closes, which R10's trial-local token already began.
- **Kept from the current design:** keying by the introducing occurrence,
  kind, count and time, the bounded capacity, expiry and promotion (§11),
  and the two trials: a value written during a trial is trial-local until
  that trial is kept, as R10's preview record is.

What this does not yet decide: when within a sentence the write and the
binding happen, which is the subject's resolution at its determination
(§34), and the order question in §34, whether identity threading is built
before the stored derivation is removed.

## 36. Decided (Alec, 2026-10-22): identity first; the 3.5 sequence

Alec chose to build identity threading before removing the stored
derivation. The rest of step 2 therefore splits into three rounds:

**2b, now: the storage changes that do not depend on the derivation.**

- Remove the saved leaf postings (§23.1), word adjacency and its nudge
  (§23.2) and the stored mode column (§22), with the authorized P1, P3
  and P5 ports.
- The evidence rule (§16, §21): the two tuple ports; no implicit-pair
  reversal for LTM0 meanings; the writer reverses a trust-derived pair
  only for an LTM1 edge.
- Retire LTM0's unused `refs` column, with the P4 absence ports for it.
- An LTM0 occurrence for every completed S, relative closes included
  (§33, R1), with the exact count ports listed.
- The staged forgetting rule (§21, §23) over today's storage: a
  sentence's structure goes before the sentence, leaving its vector as
  the gist; a referenced row is released for the next pass; one boundary
  call is one pass; the coarsening test across two passes.

The derivation, its readers and close-time minting stay as they are in
2b; `ref_kinds`, READING and P2, P4's tagged ports, P6 and P7 wait for the
removal round.

**The identity line, next: specified by Claude before it starts.** MTM
stores working content (§35); a closed sentence is recorded as an
occurrence of its subject's identity; equality substitution runs through
tokens and equality relations; role evidence is read from identities'
occurrence histories; the subject is resolved at its determination so a
verb acts on its identity (§34). This brings forward the first part of
step 4. It passes the decomposition gates and G3 on the new source
before the derivation goes.

**Derivation removal, then.** Tags with READING for mention readings, the
word bag as owned detail (§33), the end of close-time minting, the
derivation readers removed, a version-checked load, and P2, P4, P6 and
P7 restated for that layout. Then 2c, and step 2's one full sweep at its
end.

## 37. P5 after the identity-first split — Codex, 2026-10-22; for Claude and Alec

**Input and scope.** Main `cc6407f3b` commits Claude's spec, bind plan,
Architecture and todo edits unchanged. `1db6e011d` retains the preceding
§32 static-review artifacts. Production, tests and configuration still
match published 2a `0b030e3eb`. §36's storage-only scope is accepted;
identity threading, tags, READING, derivation removal and close-time
minting are not reopened here.

**One port no longer has its required column.** §36 authorizes P5 but
defers `ref_kinds`. The [amended P5](../benchmarks/2026-10-22-item3-5-step2b-port-preflight/amended-port-proposal.md#p5-full-owner-checkpoint-checks)
explicitly replaces only the corruption subcase of
`test_native_progress_and_trained_artifact_evaluation_path`
(`test/test_learning_readiness.py:188–195`) with two malformed
`ref_kinds` cases, both expecting `ValueError` matching `ref_kinds`.
Today's LTM1 has no such column. Keeping the original subcase would read
the `posting_codes` checkpoint key that this round removes. Retiring the
subcase or testing a different column changes the authorized assertion.
This is a static authorization conflict; no test or production failure
has occurred.

**Recommendation for Alec: port that subcase to today's LTM1 pair
validation.** The [exact proposed patch](../benchmarks/2026-10-22-item3-5-step2b/storage-preflight/proposed-p5-existing-pair.patch)
is retained but not applied. It uses a fresh copy of the same full-model
checkpoint for each case, through the same `load_model` call and readiness
threshold:

| Case | Checkpoint mutation | Exact expected exception |
| --- | --- | --- |
| Invalid occupied argument | In the exported allocator's `part_store`, set the second `refs` argument of an occupied row to zero. | `ValueError`, matching `Part arguments must be positive native symbol handles`. |
| Invalid pair shape | Replace that store's `refs` with its first column, shape `(capacity, 1)`. | `ValueError`, matching `two symbol references`. |

These are LTM1's existing `refs`, not the unused LTM0 column being retired.
`LTM1._index` already rejects the first mutation. Its load hook already
rejects the second before `super()._load_from_state_dict`; no new column
or validation mechanism is proposed. The checkpoint path is
`structural_extras.conceptual_spaces[*].allocator.part_store`, written by
`Models._collect_structural_extras` and restored by
`Models._restore_allocator_extras`.

Every earlier assertion in the learning-readiness case remains verbatim.
No test is retired. P5's separate `test_part_read` no-saved-postings check,
rebuild after owner restore and non-persistent `leaf_complete` are still
implementable as authorized. The malformed-tag checks return with the
tagged layout in the removal round; that later layout still needs its
own port and source evidence.

**Decision needed:** may the two existing-pair cases above replace P5's
two deferred-tag cases for this storage round? Recommended: yes, retaining
full-model corruption rejection while keeping §36's scope.

The [preflight record](../benchmarks/2026-10-22-item3-5-step2b/storage-preflight/README.md)
contains the unapplied patch and source audit. Per the handoff's rule
against assertion changes beyond the authorized ports, stop for Claude's
review and Alec's decision. No implementation, source freeze, gate,
publication or WikiOracle bump. Item 3.0 has not been reached.

## 38. Claude's review of §37 — 2026-10-22

**Verdict: approve the proposed port.** The corruption subcase in
`test_learning_readiness` exists to show that a damaged structural
checkpoint fails loudly through the full-model load path. Its old target,
the saved postings, goes in this round, and its authorized replacement,
malformed `ref_kinds`, is deferred with the tags. Codex's two cases keep
the obligation on columns LTM1 has today, and I checked both validations
in the code: a zero second argument on an occupied row is rejected by
`LTM1._index` when the loaded store is indexed, and a one-column `refs`
is rejected in `_load_from_state_dict` before `super()`. Each case loads
a fresh copy of the same checkpoint through the same `load_model` call
and readiness threshold; every earlier assertion stays verbatim; no test
is retired. When `ref_kinds` arrives in the removal round, the malformed-
tag cases are added beside these, not in place of them.

One note for the removal round, not this one: the shape message still
says "LTM2 requires exactly two symbol references", a name the rename
retired; the match pattern avoids it.

**Decided (Alec, 2026-10-22).** The §37 port is approved: the two
existing-pair corruption cases replace the posting-corruption subcase for
this round, with every other assertion verbatim and no test retired.

## 39. Standing rule amended (Alec, 2026-10-22): Codex continues, documenting

Alec: "Codex is more capable than you give credit, please allow it to
review spec, also suggest fixes and continue as long as they are
documented." From here on, within an authorized round, Codex reviews the
spec, proposes fixes, applies them and continues. Each decision is a
dated spec section, and the receipt lists every changed assertion, port
and retired test. Claude reviews at the round's end. Only reversing an
explicit Alec decision comes back to Alec first. The hard rules stand: no
seed pinning, no language named in code or fixtures, no learning claims,
every failed run retained.

## 40. Relative occurrences and the operand-only certificate — Codex, 2026-10-22; for Claude and Alec

**Input.** Main `bb0045d7a` commits Claude's §38 edit unchanged. The
approved P5 patch is applied in the working tree. The storage-only §36
implementation is partially written, with the derivation readers,
`PartClosing`, MTM and reference resolver unchanged. There is no source
freeze or runtime evidence yet.

**The new occurrence reaches two existing operand checks.**
`test_relation_closing.py::test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent`
has six cases: one or two unidentified operands, with or without a nested
relative sentence. Its original layout stores only the unidentified
operands in LTM0, so both of these checks are about those operands:

```python
assert not f.store.trust[:len(f.store)].any()
for index in f.store.ideas().tolist():
    assert f.store.row(index)['kind'] == 'unverified'
```

§36 adds the completed relative sentence to LTM0. The outer close in this
fixture supplies `trust=.7`; the current writer's outer occurrence kind
is `observation`, while the operand occurrences remain unverified with
zero trust. Applying only the authorized row-count change would make the
two checks cover the newly retained sentence as well. This is a static
conflict with their original operand-only scope, not an observed test or
production failure. Suppressing the sentence's provenance to satisfy the
old checks is not proposed.

**Recommendation for Alec:** authorize the [exact proposed port](../benchmarks/2026-10-22-item3-5-step2b/storage/proposed-relative-occurrence-port.patch):

1. Restrict the existing zero-trust check to the first `len(unknown)`
   rows, the unidentified operands written before their relative parent:
   `assert not f.store.trust[:len(unknown)].any()`.
2. Use `for index in range(len(unknown))` for the existing `kind ==
   'unverified'` assertion, whose text stays verbatim.
3. Check the new outer sentence row separately: its kind is
   `observation` and its evidence is `pytest.approx((.7, 0.))`.

The existing Part-edge evidence check, operand content comparisons,
child count and relation checks stay verbatim. No test is retired. The
count changes already authorized by §36 are exactly:

| `unknown` | `nested` | LTM0 rows before → after |
| --- | --- | --- |
| `(0,)` | `False` | 1 → 2 |
| `(1,)` | `False` | 1 → 2 |
| `(0, 1)` | `False` | 2 → 3 |
| `(0,)` | `True` | 1 → 3 |
| `(1,)` | `True` | 1 → 3 |
| `(0, 1)` | `True` | 2 → 4 |

Equivalently, the exact count is `len(unknown) + 1 + int(nested)`: the
operands, the outer sentence, and the inner sentence when present.
The proposed patch includes that count change for review but is not
applied, including its newly requested assertion/scope changes.

**Stop under the handoff's assertion rule.** Claude reviews this
disposition and Alec decides it before implementation continues. The
[partial-work record](../benchmarks/2026-10-22-item3-5-step2b/storage/README.md)
retains the input audit, working diff, every applied assertion change and
the three authorized adjacency-only test retirements. Production and test
edits remain uncommitted on main; the patch is review material, not a
frozen source. Other authorized ports and the final source audit remain
unfinished. No test, gate, publication or WikiOracle bump has run. The
identity line still requires Claude's spec; item 3.0 has not been reached.

**Concurrent amendment.** While this section was being written, Claude's
new §39 appeared in the shared spec. It records Alec's instruction to
continue within the authorized round, documenting fixes and ports for
Claude's end-of-round review. The active chat handoff still explicitly
requires a stop for other assertion changes. Codex has asked whether §39
supersedes that instruction; no reply has arrived yet. If confirmed, apply
the proposal above under §39 and continue the scoped implementation.


## 41. Claude, mid-round — 2026-10-22: §39 governs; §40 approved; three notes

**§39 supersedes the chat hand-off's stop.** §40's port is correct: the
two checks were about the unidentified operands, the new sentence row
carries the outer close's provenance, and the port keeps the checks
operand-scoped and checks the sentence separately. Apply it and continue;
from here, document and continue.

**Three notes on the working tree**, for Codex to resolve and record in
the receipt, not stops:

1. `Forgetting.forget` now breaks out of the pass after the first
   coarsening or the first owner advance of a referenced row. The decided
   rule is one stage per occurrence per boundary call, not one stage per
   pass. Either continue the pass over the remaining unreferenced rows
   and unadvanced owners, or state in the receipt why one stage per pass
   is the right reading, with the `forget_pending` carry as its
   justification.
2. `accept_fact` lost its stored-mode rejection. The receipt should show
   that a stored question is still refused as a fact by its record kind,
   which §22 relies on.
3. `LTM0.MODES` is removed; list its remaining consumers and their ports.

The evidence change (occurrence keeps polarity; only a trust-derived edge
pair is reversed for a negative meaning), the relative occurrence beside
its edge, the retired `refs` and mode columns, the non-persistent leaf
index and the dropped nudge all match §§16, 21–23, 33 and 36.


## 42. Codex: storage-only continuation and §41 dispositions — 2026-10-22

Alec's latest handoff confirms that §39 supersedes the earlier stop. The
§40 patch is applied exactly, including the operand-scoped provenance
checks. Work continues on main from `bb0045d7a`; the earlier partial-work
snapshot remains historical. Identity threading, tags, mention READING,
derivation removal and close-time minting are unchanged/deferred.

**Forgetting (§41.1).** Retain the conservative stage barrier offered in
§41: after advancing the first selected structural owner, end that pass.
The priority queue chose its low-value child; deleting a more valuable,
unreferenced substitute in the same pass would defeat that retention
choice and could reach low water before the released child is considered.
`forget_pending` carries the unmet low-water pressure to the next boundary,
including across checkpoint load and when occupancy has fallen below high
water. The configured interval still applies. Wording can be dropped from
several rows together, but each row advances at most one stage; a structure
advance cannot be followed by deletion of that owner or its released child
at the same boundary. This deliberately favors the retention ordering over
maximal progress through every independent row. It uses §41's stated
alternative, with no changed capacity, weight or retention objective.

The coarsening certificate retains its capacity, weights and vector check:
pass 1 coarsens an idea's parent or removes an unfused relation parent;
the referenced child remains. Pass 2 removes the released child, with no
same-pass cascade. A separate checkpoint regression covers the pressure
carry at three occupied rows out of four.

**Stored questions (§41.2).** `accept_fact` still rejects `question` and
`estimate` record kinds. A new regression checks a question before and
after checkpoint restore, empty evidence, absence of the stored-mode
column/enum, rejection, and successful admission of an ordinary row. The
live parse's mode and thought features remain unchanged.

**Mode consumers (§41.3).** An audit of active `bin`, `test` and `data`
finds no remaining consumer of `LTM0.MODES`. Its former uses were LTM0's
raw read, sidecar validation, legacy migration, writer and `accept_fact`;
each now uses the existing record-kind/polarity contract without storing
mode. The live `ThoughtFeatures.MODES` tuple is independent and stays.
The nested-retention assertion on an embedded question's stored mode is
ported to its question record kind and empty evidence (P1). Retired mode
checkpoint keys are discarded; no replacement column is introduced.

**Restore and test-observer repairs.** Rebuild leaf postings at the end
of full structural restore, after the Part owner and occurrence sidecars.
The two isolated index restore fixtures explicitly reattach their forced
numerical reader before rebuilding; the reader is fixture setup, never
saved postings. The former serialized posting-size checks become cache
absence checks; their retrieval, compaction and append assertions stay.
Read-only snapshots of retired LTM0 `refs` now inspect actual LTM1 argument
pairs. Where a close can append edges, compare pre-existing occupied pairs
by stable handle, allowing new rows while retaining the old arguments.
The identity measurement observer reads MTM's existing token mapping (or
its recorded native reference), not the retired all-minus-one column;
it supplies no target or production input and makes no learning claim.

**Other ports.** The §36 relative-occurrence count ports count the
completed sentences beside their existing Part edges. A full-store
rewitness fixture fills only the remaining slots after its first relative
occurrence; capacity and the full-occupancy check stay. The two exact
negative-polarity tuple ports retain evidence on the occurrence and reverse
only the trust-derived Part pair. Concept-forgetting fixtures supply their
live symbol reference in the retained semantic reading at the original
write instead of planting it in unused `refs`. The complete assertion and
fixture ledger will accompany the source diff in the receipt. Three
adjacency-only tests are retired under P3; no P5 test is retired.


## 43. Codex: remaining retired-reference consumers — 2026-10-22

The retained `final/development-01` run found `Models._commit_sentence`
still reading the removed LTM0 row export's `refs` key. Its kept field now
gets occurrence references from the authenticated meaning's `role_refs`;
a Part return still uses its two native arguments. Static follow-through
also removes Queries' fallback to that never-populated column and confines
ClauseJournal's two-argument Part-reading branch to an actual Part handle.
A three-role occurrence does not thereby become a Part. These are reader
repairs required by retiring unused `refs`; they do not remove or replace
the retained derivation. The original failure and initial captures stay.

## 44. Codex: deep participant traversal — 2026-10-22

Both existing 1,200-operation deep-clause tests reached Python's recursion
limit in `ReferenceParticipants.participant_terms`. Convert that traversal
to an explicit stack, preserving depth-first child order, the `(term,
noun)` visited key and every position rule. No fixture, assertion, parse
choice, operation budget or recursion-limit adjustment. The deep tests and
participant regressions gate the repair.

## 45. Codex: pre-existing count and retention-fixture ports — 2026-10-22

The three Part-count failures replay identically with this working writer
and `bb0045d7a`'s writer at their captured initial RNG states. The diagnostic
keeps both pair inventories in `final/preexisting-counts.json`: the compound
VP has six edges (four definitions, verb, lift), nested attribution has
15, and a lowered particular plus predicate has four (three definitions,
lift). Inline lowering was already accepted; PartClosing is unchanged.
Port the stale checks `10 -> 6`, `14 -> 15`, and the lower bound
`len(words)+2 -> len(words)+1`. No mint policy is changed.

Two further occurrence-count ports are direct §36 consequences: the
repeated relative observation plus estimate changes `2 -> 3`; the truth
text fixture's idea indices change `[0] -> [0, 1]`. The narrow-idea test
keeps the existing native reading of the earlier idea rather than expecting
all references absent, and explicitly excludes an implicit occurrence
reference to that earlier row. Derivation removal remains deferred.

The full-inventory atomicity fixture's root disjunction does not need a
seat. Place that composite in a Part argument position using the already
present word `cat`, so the same full inventory is actually asked for a new
argument seat; all refusal and atomicity assertions stay. No capacity or
word inventory grows. The sentence-end storage fixture now presents its
sentence through `What.present` with persistent serving (`stateless=False`)
to exercise a retained native close; all its storage assertions stay.
These are mechanism fixtures, not learning claims.

## 46. Codex: retired reader-side binding assertions — 2026-10-22

Three existing Part-read cases require equality witnesses to bind a pending
name during `meaning_of`. That is the reader-side binder explicitly retired
in bind §10.4; restoring it would reverse Alec's decision. Port those
cases to show that, without a resolver choice, the site keeps its open
referent and forward name, retains its complement, and acquires no thought
witness. The raw-row immutability and closing-question checks stay. The
MTM/arrival regressions still require the actual selected refinement and
its content. This is a port of the retired mechanism, not acceptance of a
wrong antecedent, and introduces no new forcing.

## 47. Codex: time-reader audit fixture — 2026-10-22

The source inventory test finds the already-published
`MediumTermMemory.open` eviction read of `entry.when` missing from its JSON
fixture. Add that function with its existing RowTime tick/oldest-token
purpose. Both exact inventory assertions and all production time behavior
remain unchanged. The audit failure is retained with development-01.


## 48. Codex: deep capture helper — 2026-10-22

After the production traversal repair, the two deep-clause cases reached
the same recursion limit in the historical capture helper, after
`finish_clause` had returned. Add a round-local, iterative flat-graph
capture (`final/storage_probe.py` and `storage_capture.py`); keep the old
probe and failed outputs unchanged. It records the same node/edge fields,
without limiting depth or changing the test. Capture paths include node
IDs to avoid collisions across the expanded module set.

## 49. Codex: native staged-forgetting expectation — 2026-10-22

The native document turnover test reached 15 rows and retained eight when
its first structure stage stopped the pass, rather than deleting to six
at the same boundary. This is the §42 stage barrier, not a capacity change.
Port its exact six-row/target-reached assertions to require actual
turnover, occupancy between six and the original count, and exact agreement
of `target_reached` and `forget_pending` with that occupancy. Keep the
24-row capacity, absolute high/low thresholds, interval, source sentences,
checkpoint-column checks and subsequent-read/write checks unchanged.

## 50. Codex: P5 selects its actual exported owner — 2026-10-22

The full-model fixture exports more than one conceptual-space entry; the
§37 proposal's single-entry unpacking was a wrong code fact. Select the
exported index of `model._concept_owner()` explicitly. The one-space
redesign remains outside 2b. Both corruption cases still use fresh copies
of the same full checkpoint, the same load/readiness call and the exact
approved exception messages; no assertion changes in this repair.

## 51. Codex: public request fixture boundary — 2026-10-22

The shared reasoning test model had a one-row public request still in its
recall history when the training test changed batch shape. Add the existing
hard document-boundary reset before that test's training call. The
production refusal of an unreset batch change, every assertion, policy,
training budget and batch size remain unchanged.

## 52. Codex: active measurement probe port — 2026-10-22

The review14 wiring test invokes an archived October 7 probe that still
reads `WordIdentity.projection`, LTM0 `rel_type` and `timestamp`. Preserve
that archive unchanged and copy its active probe/observer to `test`.
Report all LTM0 rows as occurrences (definitions are in LTM1), export
`when`, and hash the actual fixed form-code tensor instead of the retired
projection. The hash-equality assertion changes its two field names;
other assertions are initially preserved. The wrapper points at the live
copy. This is observation-only instrumentation: no model input, additional
training or optimization term. Every subsequent probe port, if required,
will be listed separately. The original nested failure stays retained.


## 53. Codex: provisioning recall isolation — 2026-10-22

The retained recall diagnostic corrects §51's incomplete diagnosis: even
in isolation, `runEpoch` first provisions two TruthSet texts at batch one,
then starts the configured batch six. Their temporary recall fields
survive provisioning. `_ltm_ingest_truth_texts` now uses an empty temporary
recall history, clears it between distinct source texts, and restores the
caller's original history and batch stamp in `finally`. This preserves the
existing refusal of an unreset caller batch change. No assertion is
weakened, training/configuration changes, or answer target is introduced.
The real six-row training test and new success/error restoration cases
exercise it. The §51 test reset still separates the earlier public-request
case from its training case; it alone was not sufficient.

## 54. Codex: active audit width — 2026-10-22

The copied measurement probe passed its earlier checks, then reached a
hard-coded 128-coordinate closing-image assertion. Compare each closing
image width to the configured concept owner's `outputShape[-1]` instead.
The old archive, dimensions, fixture configuration and all other probe
assertions remain unchanged. No fixture is resized.

## 55. Codex: remaining module fixtures — 2026-10-22

The expanded module development run retains four failures and these ports:

- The three-slot composite-operand closing has two completed sentences;
  its LTM0 count changes `0 -> 2`, while the inner Part containment check
  stays. This is the same §36 count rule.
- The isolated reference-index restore reattaches the fixture's numerical
  reader and rebuilds after tensor/sidecar load; both exact retrieval
  assertions stay. Rename the test to describe the now-derived index.
- The no-numerical-journal probe made valid stop choices on its first
  draw. Requiring an applied operation assumes an unforced random grammar
  must reduce. Check that the chooser was attempted instead; keep all
  dense-value/reference absence assertions. No operation is forced and
  no draw is selected. This tests the no-reader storage contract for the
  actual choices, not learned composition.
- The native-column checkpoint fixture still expected a determiner's
  mint to allocate a native individual, which step 1b retired. Explicitly
  mint the fixture's native concept and give it one witnessed Part edge;
  every ownership, content, order, witness-count and checkpoint assertion
  stays. MTM minting is unchanged and is covered by the resolver tests.

These are within §39; no explicit Alec decision is reversed. All failures
and the original captured states remain in `final/development-07`.

## 56. Codex: journal probe's actual consumer boundary — 2026-10-22

Development-08 refutes §55's journal diagnosis: the final trace has no
attempts either. `_run` enters `_sentence_run`, which enables the closing
reader inside the context and clears `_sentence_ends` on exit. Checking
that flag after exit did not establish that the run lacked a consumer.
Correct the fixture to open the normal sentence context and disable its
closing and reconstruction consumers during the call, as its name and
assertions require. Restore the original applied-operation assertion;
every assertion is now verbatim. No operation choice is forced. Both
failed attempts remain retained. The other three §55 ports passed their
development follow-up.

## 57. Codex: native end-state handle on the frozen draw — 2026-10-22

Frozen `bfa9c94ee` completed all 683 scoped cases: 674 passed, eight
existing prerequisite skips, one failure. The regression and eight-file
groups passed. The native end-state fixture selected a relative Part
reading, then treated its returned handle as an LTM0-only address. The
captured-state replay confirms handle `19`, common location `('sym', 19)`,
no LTM0 index for that symbol, and five retained LTM0 occurrences. This is
the existing relative return contract explicitly kept by §36, not a
failure to retain the sentence.

Port the fixture's lookup from `index_of_row` to the common `location`
reader. Keep its non-null assertion. Compare the exact ending roles to
the Part row's read for a Part handle, or directly to `store.slots[row]`
for an occurrence, with the same zero tolerances. All trace-cleanup,
closed-slot and no-program assertions stay. No production, grammar,
forcing, seed, budget or objective change. The full failed frozen run and
its replay remain under `final/`; the corrected source's complete scoped
gates will be under `final/landing/`. Nothing is promoted from the failed
hash and no full sweep is added.

## 58. Codex: final storage-source validation — 2026-10-22

Final source `3c5cc2a1d5243dd80c2cdcfc3cc2ebcdaa98c56c` completes all
683 requested scoped cases: 675 passed, eight existing prerequisite skips,
zero failures. The eight regression files give 85 passes; the eight-file
native set gives 123 passes and seven skips; the direct changed-module
files give 467 passes and one skip. The pending-premise counts are forward
bound/open `0/8`, reverse `0/16`, with no closure forcing and zero attention
budget. These are mechanism results, not learning evidence.

The [receipt](../benchmarks/2026-10-22-item3-5-step2b/final/landing/README.md)
links the frozen commands, reports, captures, full assertion/fixture
ledger, module coverage and every failed earlier run. Its audit explicitly
includes the three new fixture helpers and their assertion bodies. The
first freeze remains a failed attempt; none of its passing subsets is
promoted. No full sweep or twelve-draw series was added. Claude's read-only
review of this source returned HOLD; §59 records the continuation. The next line
is the requested §§34–35 code review and Claude's identity spec, not an
early implementation of it.

## 59. Codex: repairs from Claude's storage-source HOLD — 2026-10-22

Claude's read-only review of `3c5cc2a1d` returned HOLD, despite its 675
passes and eight existing skips. Preserve that source, gates and review
under `final/landing/`; none is promoted to the repaired source. Under §39,
repair the findings without reversing Alec's decisions, then re-freeze and
run the expanded three scoped groups, followed by another Claude review.

**B1, live mode.** Removing stored mode must not erase the current parse's
mode. `closing_question` applies the live parsed/clause mode to the retained
reference view; `_commit_sentence` applies the clause's mode to its
sentence end state, preserving the answer reader's interrogative guard.
The held-description expectation reader follows an LTM address under the
current `ask`'s query intent, as `_ask` already does. Thought-memory records
still use their live mode. No mode column, sidecar field or structural
replacement is introduced. Four new cases write a zero-evidence yes/no
question and close it, with/without an explicit parsed view and for both
absolute and relative clauses; all must open an episode without an open
referent slot. The existing direct/nested expectation comparison joins the
scoped module set unchanged.

This completes the §41.3 consumer list beyond the removed `LTM0.MODES`
enumeration: `ThoughtClosing`, end-state answer seeds, and held-description
expectation needed live-context repairs. `ThoughtFeatures`' stored reads
now receive the ordinary default mode, while live thought records keep
theirs. `accept_fact` continues to reject the stored `question` record kind;
a row explicitly stored as an `observation` with evidence has no saved
interrogative mode to test. Incoming live TruthSet questions remain refused.

**B2, writer coverage.** Include every direct `write_clause` test file and
the additional `commit_reading` callers identified by the review. Three
known completed-relative count ports change LTM0 `0 -> 1` in
`test_relative_sentence_codebook_insertion`, `test_lies_can_be_learned` and
both observation-boundary cases in `test_selected_relation_meaning`.
Their Part counts, evidence and content checks stay. Any further diagnosis
and exact port is recorded below and in the receipt. This is the requested
changed-module scope, not a full sweep.

**B3, numeric end-state references.** Preserve the retired unused LTM0
column's effective value `(-1, -1, -1)` in an occurrence's end state. Do
not flatten `('slot', occurrence, role)` or `('constituent', index)` into a
number; their typed references stay in `meaning.role_refs`. Native Part
returns keep their two native arguments.

**Follow-ups.** Remove the always-empty native predicate cache and its
unreachable lookup. Sharpen the narrow-idea check to exclude every LTM
operand reference, and compare it with the pre-2b writer in a retained
diagnostic. Explicitly list the removed `where` loop comparison as an
assertion port. The three §46 Part-read cases were already failing after
§10.4, independently of 2b: the published source carried those stale
reader-binder assertions. The first frozen end-state failure belongs to
the module group; “native” describes its production path, not its group.

## 60. Codex: expanded writer checks and exact retained-value ports — 2026-10-22

Development-10 completes 359 cases: 346 passed, one existing stage3 prerequisite skip,
12 failed. The four new closing-mode cases and the unchanged direct/nested
held-question comparison pass. All failures and initial states remain.

The exact count ports required by the completed-S rule are:

| Test | Old published assertion → new |
| --- | --- |
| relative insertion, repeated opposite-polarity claims | LTM0 `0 -> 2`; Part `1` stays |
| lies, conflicting claims | LTM0 `0 -> 2`; Part `1` stays |
| negative expectation, concluded `not` | LTM0 `1 -> 2` |
| replacement contracts, direct and two chain lengths | LTM0 `0 -> 1`, three cases |
| taxonomy policy, nested conclusion | LTM0 `0 -> 1` |

The first two correct §59's static `0 -> 1` proposal: opposite polarities
have distinct native content keys and therefore distinct occurrences,
while sharing the Part edge. The selected-relation observation's `0 -> 1`
port passes in both cases. Evidence, ablation and path assertions stay.
`test_real_packed_runbatch_trains_prediction_and_representation_with_teacher`
now asserts absence of the already-retired `legacy_prediction_enabled`
attribute instead of reading it; all predictor/gradient assertions stay.

Four predicate-argument cases also fail with the pre-2b clause writer,
replayed from their captured initial RNG states. Both writers return the
same Part arguments (`18`, `25`) and no native bank value for either: they
are retained derivation roots, whose content belongs to their LTM0 source
occurrences. Port the fixture's native-bank-value assumption to two exact
checks per argument: the source occurrence's root reference equals that
argument, and its retained point equals the original operand leaf. Keep
both relation types, registry/no-registry paths, two-column layout,
row-free predicate and predicate-not-an-argument checks. This makes the
identity link and full retained content explicit without seating roots or
changing the derivation reader or close-time minting. The retained
`review-repair/predicate-arguments.json` includes both writers' exact reads.

No test is retired, skipped or resized in these ports. The three original
P3 adjacency-only retirements remain the entire retirement list. The
standalone-site fix, identity threading and removal-round readers remain
at their decided later steps.

## 61. Codex: predictor-context observation port — 2026-10-22

Development-11 passes 11 of the 12 retained failures. The joint-objective
case reaches its next stale storage assumption: `_inter_context` now holds
`SituationAddress` values, not only `(depth, tensor, mask)` triples. Read
the same bounded predictor context through `_read_context(b)[0]` before
applying its unchanged `not p.requires_grad` predicate. Do not substitute
the full LTM history or add a detach; this tests the actual predictor view
for each existing stream. All gradient, parameter-update, loss-weight and
observation checks remain unchanged. The first failure and this second
failure stay retained; there is no production or objective change.

## 62. Codex: held content need not be an executable thought — 2026-10-22

The frozen `9c3899f2f` regression and eight-file groups pass. During its
expanded module run, an additional read-only probe catches an edge case
in B1's held-description repair: an `ask` may name an ordinary fused LTM
idea, which has no registered thought VP. Following the address under
live query intent must not then require that content to have a thought
signature. The retained `review-repair/held-idea.json` captures the
`ValueError: thought meaning has no grammatical VP` on this exact source.
No code changed before that probe; this source will not be published.

Repair the mask reader to follow only a valid nested thought signature.
For an LTM description without one, stop following and keep the current
request's existing open-role mask (none for a filled `ask`). This is the
same signature eligibility the live closing-image path already checks;
it does not infer a saved mode or execute the description. Non-LTM live
thoughts retain their prior loud validation and mode guard. Add two cases:
a fused idea and a relation whose VP is an ordinary concept, each read
once on the shared meter with no invented open role. The existing
registered nested-question comparison stays unchanged.

Retain this frozen run and probe. The guarded source gets a new freeze
and all three scoped groups; no partial result is promoted. There is still
no full sweep. Identity and derivation-removal contracts are unchanged.

## 63. Codex: held-description fixture admission — 2026-10-22

Development-13 passes 13 of 14 cases. The new ordinary-relation case fails
before reaching the reader because `append_meaning` defaults to `fact`
and the catalogue fixture supplies a live interrogative mode. This test
is about reading retained descriptions, not accepting them as facts.
Specify `kind='observation'` on its write, matching production observation
admission and the helper's earlier row. Both assertions stay verbatim;
no production change. This is the same-round setup correction permitted
by reconstruction §17 and §39. Preserve the failed run, then rerun both
cases before freezing. The frozen `9c3899f2f` gates meanwhile completed
with 1,023 passes and nine existing prerequisite skips; its §62 probe
still disqualifies that source from publication.

## 64. Codex: a word keeps its own bank form — 2026-10-22

The `1f53af50c` frozen gates complete with 1,024 passes, nine existing
skips and one production failure: the joint-objective test's second
presentation trips the rung-zero identity audit. The first draw and two
unchanged-source captured-RNG replays are retained. A selected object's
word is `2`, but the bank row reached through that word decodes as `plus`.
The word `2` has itself become a referent of `plus`; the bank's spelling
reader follows that second definition and substitutes its form. This is
an ownership defect, not an allowed reconstruction error or a test port.

Under §39, `word_surface_for_row` now reads a word's own admitted
one-form description first. For a non-word object it continues to follow
its unique definition through `Interpret.lexical_form`. No association,
conceptual reading, choice, identity audit, or existing assertion changes.
The native word row and the object it names therefore reconstruct the
same word, even if that word is also another word's referent. The retained
failing initial RNG passes all three presentations with this repair; that
replay is diagnosis only and does not pin any gate initialization.

The new regression explicitly admits two words and their objects, defines
one word as the other's referent, and checks the first word's bytes and
native form both at its own row and its object's row, with both definition
links retained. Its first setup used the surface-snapshot helper and
assumed that `12` had been admitted as a single word; that helper does not
promise this. Development-15 and -16 retain the resulting setup failure.
The setup now uses the existing explicit word-admission fixture. Every
assertion remains verbatim; development-17 passes. This is the
same-round fixture exception, not a new forcing or a resized fixture.

Withdraw `1f53af50c`; retain its complete gate summary and captures. The
next freeze reruns all three scoped groups and adds the form/definition,
word-store, derived-code and reconstruction-bank files for this reader.
There is no fresh series or full sweep. No earlier result is promoted.

## 65. Codex: preserve the bound-word terminal contract — 2026-10-22

Claude's read-only review of `867452a79` returns **HOLD**, despite its
1,100 scoped passes and nine existing skips. N1 is correct: §64's claim
that choices and conceptual reads were unchanged was too broad. Giving
every defined word its own surface also made ordinary bound-word rows
terminal candidates and pinned their derived form coordinates. The old
contract emitted objects and objectless words, not every bound word.
The full review and that frozen run remain retained; nothing is published.

Take Claude's recommended narrow repair under §39. This preserves the
existing bound-word terminal rule and reverses no explicit Alec decision;
it does not choose the alternative broader decoder contract. A word with
an object has no row surface, including when another definition also
names that word. Objectless words and ordinary objects keep their prior
spelling paths. The word's bank form is read from its own perceptual
parts, so another word's definition cannot substitute a different sparse
key. No new terminal, masking policy, bind rule or lexical part of speech
is introduced. The captured failing initial RNG now passes all three
joint-objective presentations with this narrow implementation. It remains
a diagnostic replay, never a gate initialization.

**Exact same-round regression port.** In
`test_word_bank_form_survives_use_as_another_words_object`, replace the
loop's `assert owner.word_surface_for_row(row) == b'12'` for both word and
object with `assert owner.word_surface_for_row(owner._csw_row_of(word)) is
None` and `assert owner.word_surface_for_row(owner._csw_row_of(obj)) ==
b'12'`. Also check the word surface is `None` before the second definition.
The exact native-code decode checks for both rows and both definition-link
assertions stay verbatim. The earlier assertion embodied §64's rejected
broadening; this is a disclosed contract correction under §39, not a setup
fix with unchanged assertions. Development-18 passes. No other test
assertion or fixture changes. Retain the exact patch from `867452a79`.

The next frozen hash reruns all three groups and adds the four direct
reader consumer files named by Claude: `test_radix_layer_reverse.py`,
`test_active_decode_candidates.py`, `test_echoic_decoder.py` and
`test_tied_operator_reconstruction.py`. No fresh series or full sweep.

**Other review declarations.** A completed interrogative Part now keeps
the live clause mode and is excluded from answer seeds; the previous
assertive Part readback allowed it through. `closing_question` prefers
`parsed.mode` when provided, otherwise the clause mode. The current
held-description signature guard treats invalid thought-shaped LTM
content as ineligible too; invalid live thought records still fail
loudly. A future explicit eligibility predicate may make that distinction
clearer. The predicate ports assert the actual occurrence link and content,
not uniqueness among all occurrences. The narrow-idea diagnostic tests a
second sighting of the same row. None is a hidden extra certificate.
The surface-keyed observer no longer receives ordinary word/object
duplicates from §64's broadened terminals. The review's five additional
indirect-consumer files remain covered by step 2's end full sweep; they
are not claimed as gates of this storage round. B1–B3, the §41 dispositions
and the earlier port package are accepted in Claude's review.

## 66. Codex: echoic fixture operands in the declared domain — 2026-10-22

The `85da7c14d` frozen set completes with 1,142 passes, nine existing
skips and two failures, both in the newly included echoic-decoder file.
The eager and compiled `test_same_walk_infers_binary_and_unary_without_a_journal`
cases emit zero leaves instead of two. Preserve the full run and its
captured RNG states; this source is not published or submitted as passing.

The fixture does not call the word-surface reader or any LTM storage
path. It supplies `[[2,1],[1,2]]` as conceptual operands, outside
`decoder_eligibility`'s existing `[0,1]` domain. The binary inverse is
therefore correctly rejected and the reflection keeps changing the root
without reaching an emitting leaf. Seven relevant production functions
(the walk, inverse menu, pair search, binary dispatcher, eligibility,
conjunction kernel and code-direction read) have identical AST hashes at
the unchanged §38 baseline and this source. Both failures reproduce from
the captured RNG; the diagnostic retains the actual roots, operands,
outputs and traces.

Under §39, port only the fixture's numeric operand literal from
`[[[2.,1.],[1.,2.]]]` to `[[[1.,.5],[.5,1.]]]`, with a comment naming the
declared domain. The two directions and the conjunction root are
unchanged. Shape, two-stream/two-operand capacity, budget eight, policy,
operators, eligibility, existing second-walk monkeypatch and every
assertion stay verbatim. This is no added forcing or menu mask. A
captured-state diagnostic with that literal alone passes both eager and
compiled cases, including the exact emitted values, two-leaf counts,
action traces and no-gradient assertions. No production change is needed.
This is an existing-fixture repair under §39, not a claim to the
same-round setup exception.

The next hash freezes this fixture correction and reruns all three scoped
groups with the same 92-file selection. No result from `85da7c14d` is
promoted. N1's narrow production repair remains unchanged; Claude reviews
it and this port together after the new gates. No fresh series or full
sweep; step 2's one full sweep remains at its end.

## 67. Codex: storage-only 2b passes and Claude says LAND — 2026-10-22

Frozen source `6a36b7ceeb5a5da9ae6019f62c59c16d27f7f1ce` completes all
92 selected files: 85 regression passes, 123 native passes/seven existing
skips, and 936 module passes/two existing skips. Total: 1,144 passes,
nine existing prerequisite skips, all 1,153 selected cases completed.
The pending-premise certificate records forward 0 bound/7 open and
reverse 0 bound/16 open. No earlier source's results are promoted; all
failed runs and reviews remain retained. No fresh series or full sweep.

[The source-bound receipt](../benchmarks/2026-10-22-item3-5-step2b/final/decoder-domain/README.md)
contains the exact assertion and fixture ledgers, the three P3 test
retirements, all gate reports and the actual read-only Claude review.
Claude returns **LAND**, no blocking findings. N1's narrow repair and
the §66 fixture correction are accepted. The source and tests are
unchanged since the freeze. Publication is authorized on main, followed
by the WikiOracle bump; no second sweep.

Two nonblocking review notes are declared. First, the exact domain check
is `Language.py:13445–13448`: `in_domain` requires finite coordinates in
`[0,1]` for both operands, and `admissible` combines it with availability.
The final pair mask at line 13474 requires that value. Thus §66's old
operands are rejected before recomposition/residual tests. Second, a
bound word used as another word's object has no emitting surface at that
object row. The second word has no emitting terminal through it. Carry
the ownership question into the identity review/spec; no further decoder
change is part of this storage landing.

The §41 resolutions remain as accepted: a conservative structure-stage
barrier with `forget_pending` carry, stored-question rejection by record
kind, and all stored-mode consumers retired or explicitly ported to the
current clause/query mode. Derivation, its readers, close-time minting
and MTM working values remain for the decided later rounds. Identity
threading needs Claude's spec and the requested §§34–35 code review
next. No learning claim; item 3.0 has not been reached.

## 68. Codex: identity §§34–35 against the landed storage source — 2026-10-22

**Scope.** This is the requested code review for Claude's identity spec,
after storage-only 2b was published at BasicModel `39272e7a6` and
WikiOracle `ade0399`, with reviewed source `6a36b7cee`. It implements nothing. §36 settles identity before
derivation removal; §35 settles one current working value per identity.
The earlier pointer-only/no-working-vector rule is superseded for the
identity line. The questions below concern its concrete contract.

**What the code actually does.** `MediumTermMemory.Token` currently has
only occurrences, kind, count and when. Its two dictionaries map token
keys to records and occurrence addresses to token keys; they are not a
surface-name dictionary. `ReferenceClose.token_value` clones the
introduction's retained point, optionally includes the kind, and takes a
coordinatewise maximum over uses. This is a historical union, with an
existing detach, not the latest state or a differentiable recurrent
value. Moreover, `individuals` uses this path specifically for anonymous
introductions; ordinary groups have their own occurrence/native baseline.
The spec must name every consumer that moves to current value, rather
than only replacing `token_value`. `_derivation_symbol` currently takes
a stored `sym` root before consulting the token kind; changing the kind
alone does not redirect every read to an identity's current value.

`ReferenceClose.preview` runs over an already composed clause. Its
`('new', n)` records let a later site in that preview read an earlier
site's content (R10); they do not let a later VP read a previous VP's
updated state during composition. `commit` attaches the planned noun
sites and arrival refinements, after `write_clause`. It does not in
general attach every closed S to its subject's token. 2b now supplies an
LTM0 address for every completed S, but its existing role 0 can still
name the derivation fold. Neither fact supplies identity threading by
itself. For a completed relative, `write_clause` still returns the Part
handle and puts that handle in `written_rows[id(clause)]`; the new LTM0
occurrence is a separate row. The identity spec must expose that event
address without conflating it with the retained relation's handle. `ReferenceClose._participants/prediction/expected_roles` still
read Part arguments and access counts. `ThoughtBinding.integrate` still
uses component kind/operand congruence as well as witnessed equality edges.

**1. Define “name” as a mention, not a global spelling key.**
Recommendation: retain the introducing occurrence as the token key and
map resolved mention handles to it. A lexical name may denote different
objects in different contexts; two uses of the same form or kind must
not automatically alias. Keep pending-name metadata distinct from a
settled mention mapping. Specify how provisional handles at determination
become the exact durable occurrence addresses at kept close, including
two identical mentions in one sentence and separate batch streams.

The final storage review also identifies a dual-role row: a bound word
used as another word's object has no emitting surface under the preserved
terminal rule, so the second word has no emitting terminal through that
row. Recommendation: specify ownership and separate the form handle from
the identified object explicitly; do not broaden decoder terminals as an
incidental effect of identity threading. The storage landing leaves this
case unchanged beyond the reviewed narrow repair.

**2. Give the trial an explicit read/write sequence.** Recommendation:
each trial starts from the same MTM snapshot and owns an overlay. At the
grammatical determination, the existing resolver menu chooses an
antecedent or dummy; a first mention initializes its own value from its
NP. Resolve the subject before applying the VP, read that identity's
current value, then publish that VP's result to the trial overlay before
a subsequent clause can read it. At keep, commit only the kept overlay
and its occurrence history once. A discarded trial, preview, preparation,
failed close must not leak bindings, values, recency or counts. Specify
whether provisioning contributes an independent identity context or a
temporary overlay; it must not accidentally replace the caller's context. Reuse the current dummy/tie, exclusion and chooser
credit contracts. No new forcing, lexical noun test or ID-to-NP rule is
needed. The spec must identify the actual determination event for a
participant leaf, as well as a determiner result, using the existing
positions, occupations and order stamps.

**3. Separate current state, kind and history.** Recommendation: a VP
update replaces the current state; it does not union it with all history.
Keep kind as introduction/refinement metadata and LTM0 as immutable event
history. The spec must define whether an ordinary compatible mention adds
descriptive information to current state, and how an arrival sets or
refines it without replaying old VPs. Preserve the settled-join rule:
a later assertion cannot move an already joined site or overwrite the
old antecedent's identity. Adding event occurrences must not accidentally
disable the singleton guard on an unresolved introduction, or relax that
guard for genuinely joined mentions. Distinguish event-history membership
from a second independently joined noun use when specifying that guard.

A bound native minted individual currently gets no MTM entry at all.
Its future mutable state must not be a bank-row mutation. Recommendation:
specify an occurrence-keyed working entry for that case, preserving the
native identity as a reference. Confirm how this amends the current
no-entry exception while keeping kind, native concept and current state
distinct.

**4. Define event ownership and role evidence explicitly.**
Recommendation: make each closed S reference its subject identity and
record the corresponding event address in that identity's history.
An object participant remains a reference to its own identity; it must
not become another subject-state update. State the durable evidence that
identifies the verb and participant role after derivation removal.
A token plus an unlabelled fused event point does not by itself replace
`_participants`' subject/predicate/object distinctions. Specify the
counting unit (distinct events, with existing re-witness behavior) and the
G3 expectation query before claiming role evidence follows automatically
from token history. Keep these semantic relations separate from a saved
parse or an exact word sequence. The current writer gives all completed
members of one close the same `RowTime` stamp; define event order inside
that close from the kept operation/occurrence sequence, so the later VP
and the temporal-difference teacher do not depend on a new timestamp.

**5. Preserve witnessed equality and compositional substitution.**
Recommendation: resolve operand references through tokens, but retain
positive equality evidence and its source occurrences in the proof.
Token aliases alone are not evidence of a new equality. The spec must
explain how the two decomposition arms compare expressions containing
the same bound variable after component/operand reads disappear; whole
token equality alone does not implement congruence of compound
expressions. Run the original decomposition snapshots/seeds and G3 on
the identity source while the existing derivation readers still exist.
Only the later removal round removes those readers; no premature exact
component assertion port is implied by this review.

**6. Preserve the meaning of each scored reference choice.**
Recommendation: capture the referent value that the mention actually saw
at determination. The already-landed `reconstruction.reference` term
charges the uncovered share of that mention's own content at weight 0.1;
a later VP update must not retroactively change that earlier comparison.
Keep first mentions free and same-close aliases covered, and state how
arrival/thought exclusions carry forward. The selected bind still receives
sentence or episode R + A score-function credit, with no reintroduced
local `choice_cost` objective. The identity spec should name the scored
snapshot and commit correspondence, as the existing preview/prepare/commit
contract does.

**7. State tensor ownership, gradient and lifetime boundaries.**
Recommendation: stage the bounded identity values and handles into
trial-local tensor state at the host boundary; pass that state through
the compiled operation path, without Python pool reads inside its body.
The spec must distinguish the current detached history read from a
pathwise VP value read, and say where a graph ends between kept
sentences. Do not add a detach simply to make the overlay convenient or
quiet an ownership error. Bind's discrete choice retains score-function
credit; changing a continuous consumer or loss needs an explicit
contract, not an incidental new training path.

Today the pool is a plain object attached to ConceptualSpace; the
resolver checkpoint extra is version-only and ordinary Space resets do
not serialize or reconstruct its entries. LTM0 row reclamation already
calls `mtm.forget`; preserve that existing ownership hook unless Alec
changes its lifetime contract. Specify checkpoint/load,
document/stream isolation, row reclamation and capacity eviction for
values and mappings together. The decided 16-close/document expiry and
promotion policy is still 2c's work. Do not silently implement that
policy early, or let temporary truth provisioning erase the caller's
identity overlay.

**Recommended identity certificates.** Keep all existing binding, G3 and
decomposition semantics, and add a fixed-operation within-sentence
two-clause case where the second VP demonstrably receives the first VP's
result via the subject identity; a change-of-state case that keeps the
latest place rather than their union; two aliases reading one updated
value; both-trial isolation with opposite keep outcomes; stream and
provisioning isolation; event/mention count and joined-site protection;
and immutable historical LTM0 rows after later value changes. Use
mechanism fixtures with no learned-grammar or learning claim.

Claude's spec should answer these questions and name the implementation
seams before identity coding begins. Item 3.0 has not been reached.

## 69. Claude's end-of-round review of storage-only 2b — 2026-10-23

Reviewed: §§42–68, the final receipt under
`benchmarks/2026-10-22-item3-5-step2b/final/`, the delegated HOLD and LAND
reviews, the assertion ledger, and the frozen source `6a36b7cee` at the
points that matter. Published at BasicModel `39272e7a6` and WikiOracle
`ade0399`.

**Verdict: the landing stands.** 92 selected files, 1,144 passes, nine
existing prerequisite skips, no failures, bound to the frozen hash; 55
changed checks, three retired tests (all adjacency-only), four new tests,
every one listed with before and after text; seven frozen sources, each
failed one retained with its receipt and never promoted. §39 worked as
intended: every fix is a dated section, and the two HOLDs were repaired
rather than argued.

**Checked against the code.** `accept_fact` refuses a stored question by
its record kind. No consumer of `LTM0.MODES` remains, and the live parse
keeps its mode where it is needed (`closing_question`, the sentence end
state, the held-description reader). The occurrence keeps its polarity
and only a trust-derived edge pair is reversed for a negative meaning.
The `word_surface_for_row` repair is narrow: a word with an object spells
nothing, as ordinary bound words already did; only the dual-role row
changes. The forgetting pass stops after its first structural advance,
with `forget_pending` carrying the pressure to the next boundary.

**Carried notes, none a stop.**

1. *The stage barrier.* §42's "one structural advance per pass" is
   stricter than §21's "one stage per occurrence per pass", and §49 ported
   an exact six-row certificate to a range for it. Acceptable as
   documented; low water now takes more boundary calls. Step 2's sweep
   should report how many.
2. *Latent failures outside gate scope.* Several tests were already
   failing on published sources (§45's stale counts since inline
   lowering, §46's reader-side binding since §10.4, §66's out-of-domain
   operands). Each was ported with a diagnosis; the one sweep at step 2's
   end is where the rest surface, as decided.
3. *The dual-role word* (a word that is another word's object has no
   emitting terminal through that row) goes to the identity spec, as §68
   says.

**§68 is accepted as the brief for the identity spec**, which Claude
writes next. It answers each of the seven points and names the seams:
mention-keyed names, not spelling keys; a trial overlay with the read and
write sequence at determination; current state separate from kind and
immutable history; events referencing their subject identity with the
verb and roles recorded durably; equality substitution that keeps
witnessed equalities and compound congruence while the derivation readers
still exist; the referent value scored at determination; and tensor,
gradient and lifetime boundaries. The certificates §68 proposes are the
spec's gates.

## 70. Discussion (Alec, 2026-10-23): where a sentence lives; two places that chain

Alec, on the identity spec's "events as the sentence row's three slots":
"The sentence row being three-slot is a solution, but in that case where
do we store that row? LTM3 was what we used to use, and perhaps turning
that into a two-place pointer into LTM0 was a mistake. We might be able
to allow three places in LTM1, which would accommodate clauses and
chaining, and remove any need for a 'sentence row' as separate from the
two LTM structures." Then: "Actually, three places are not necessary as
long as the LTM1 can chain: so the second entry points to another LTM1
relation. That might be even better."

**Where things stand.** The two-truths amendment of 2026-10-19 decided
three positions (argument, kind, argument) in LTM1 and one concept per
LTM0 row; 2026-10-21 narrowed LTM1 to two references; the code has
three-slot LTM0 rows and two-reference LTM1 rows, neither as decided.

**Two places that chain.** An LTM1 row holds two tagged references and
the lanes; either reference may be another LTM1 row, which is LTM2 by
argument, as decided. The storage then mirrors the grammar's binary
compositions: "Felix runs" is `(OCC Felix, SYM runs)`; "Felix chases Tom"
is `(OCC Felix, REL r)` with `r = (SYM chases, OCC Tom)`, VP = VP NP; "he
said she said X" is `(OCC he, REL (SYM said, REL (OCC she, REL (SYM said,
REL X))))`. No kind slot: the row's kind is read from what it references.
No separate sentence row: the top relation row is the sentence.

**What this stores, and what §30 removed.** These rows are the sentence's
semantic composition over bound mentions and verbs, predication and
verb-object application. They are not the parse: no surface, marker,
determiner or order operation is a row, and no operation name is stored.
§33's "asserted relations: never a lift or verb pair kept only to connect
operands" is amended to this: the predication and the verb-object
application are the asserted relations.

**Open, for Alec.**

1. *Does a row carry a vector?* With the one-entry reading (concept, lanes
   and time as the LTM0 stratum; references as the LTM1 stratum; one
   handle), the top relation row has the sentence's fused vector, so
   reconstruction from the vector and the gist on forgetting still work,
   and 2b's occurrence-beside-its-edge collapses into one entry. Without
   it, a sentence is its references and lanes only.
2. *One handle namespace.* Occurrences are addressed by 64-bit keys,
   relation handles are allocator concept ids; one entry needs one key or
   both.
3. *Sequencing.* This is the derivation-removal round's layout. The
   identity line's events then are the top relation rows; building it on
   today's three-slot rows first means migrating twice.

## 71. Decided (Alec, 2026-10-23): ideas in LTM0, binary chains in LTM1, no sentence row

Alec (his "STM0/STM1" read as LTM0/LTM1): "The change from 10-19 to 10-21
was good but incomplete. Since VPs are defined on NP subjects, some
sentences will reduce to 'ideas', which means (in this context) an
absolute truth. Absolute truths are understood directly in the sense that
they do not require references or computation; they are a single vector
in conceptual space, and not necessary in any codebook. So there is only
one slot for an LTM0. LTM1 contains two references, and for simplicity we
do not store a typed relation (these are used for temporal chaining,
ifThen, type and part operations). However, they must now handle the
three-slot clause chains that we are moving out of LTM0 that are not able
to compress into one vector. In order to do that, and still remain a
binary relation, they are able to contain both LTM0 and (recursive) LTM1
relations. These two structures should replace any notion of a sentence
row, since sentences should turn into LTM0 (if they can) or an LTM1 chain
(if they can't). So 'Felix runs' and 'Felix chases Tom' are both LTM0 (as
long as all words are existing concepts). 'She said X' is either LTM0 or
LTM1 (depending on if X is an existing concept). Given an English syntax
where V O become a V, we need two more LTM1: (He) ((said) (she said x))."

**The layout.**

- **LTM0: one slot.** An entry is one vector in conceptual space, an
  absolute truth understood directly, with its lanes, access count,
  surprise and `.when`; no symbol is required. A sentence whose words are
  all existing concepts compresses to one: the VP applied to its subject.
- **LTM1: two tagged references, no stored type.** Each may be an LTM0
  entry (OCC), an existing concept or word (SYM), or another LTM1 row
  (REL), recursively; a row referencing a row is LTM2, derived. The lanes
  and `.when` stay. The relation's use (temporal chaining, if-then, kind,
  part, predication) is not a column.
- **No sentence row.** A sentence is an LTM0 idea if it compresses, else
  an LTM1 chain whose top row is the sentence. V O becomes a V: "he said
  she said x" with x unknown is `(OCC he, REL (SYM said, REL (OCC she, REL
  (SYM said, OCC x))))`, two rows more than "she said x".

**Superseded by this:** 2b's rule that every completed S gets an LTM0
occurrence beside its edge (§33 R1); the identity spec's events as "the
sentence row's three slots"; §33's narrow list of asserted relations; the
READING question of §28 as posed; §70's open point 1 (a chain has no
fused vector, by definition; an idea is one).

**Open after this, for Alec (§72 when answered):**

1. *Role evidence once a sentence compresses.* Bind's `expected_roles`
   and `prediction`, and G3's elision, read who filled which role of which
   verb. In a compressed idea that is inside the vector. Candidates: read
   it from the verb's group structure (the object supplies Q, the verb the
   diagonal, step 4), from MTM's event history while it lasts, or only
   from uncompressed chains.
2. *Telling untyped relations apart.* Recommended: the relation word is
   itself a referenced symbol, as a verb is (`(S1, REL (SYM if-then, S2))`),
   and kind and part are read from the arguments' orders.
3. *A mention's reading.* Where the word a mention reads to is kept: an
   LTM1 row `(OCC mention, SYM word)`, or MTM only. With the copula a
   referenced symbol, "x is three" is `(OCC x, REL (SYM is, SYM three))`,
   so a reading row is no longer confusable with an equality.
4. *Sequencing.* This is the removal round's layout; the identity line's
   events are then LTM0 ideas written as occurrences of their subject's
   identity, and chains for what cannot compress.

## 72. Decided (Alec, 2026-10-23): the four open points of §71

1. **Role evidence comes from MTM while it lasts.** An uncompressed chain's
   identities are brought back into MTM by recalling that memory: `query()`
   brings at least a reference to the chain's head into STM. Alec's
   refinement (2026-10-23), recorded for the thought design: STM is eight
   chunks, each over a small number of ideas (as LTM0 holds them) and
   relations (as LTM1 holds them), so a recalled chain sits in STM by
   its head reference while its identities live in MTM. A chunk is not
   one row: it is a small working set of both kinds.
2. **If-then, whole-part and type-token need no stored type.** The first
   has temporal references, the second spatial, the third symbols; the
   referenced entries' tags and situations (`.when`, `.where`, symbol)
   tell them apart. §71's point 2 recommendation is withdrawn; no
   relation word need be referenced for the kind of the relation.
3. **A mention's reading is kept in MTM as a name/value pair.** No reading
   row in LTM1; the READING tag question is closed.
4. **Sequencing is Claude's call.** Decided: the identity line first, as
   §36 ordered, because role evidence and readings now live in MTM and are
   prerequisites for removing the derivation readers; then one
   layout-and-removal round on §71 (LTM0 one slot, LTM1 two tagged
   references with chaining, no sentence row, the Part-edge tree and its
   readers removed), for which Claude writes the spec after the identity
   line lands; then 2c and step 2's one sweep.

**Aside (Alec, 2026-10-23): the parse-tree cache and STM1.** "I don't know
what cache we are using for the sentence parse tree, but it might have a
good integration with STM1." The cache is the per-sentence operation
journal: `AnswerProgram` (`bin/Understanding.py`) holds the kept reading's
`actions`, `leaves`, `concept_ids` and `word_ids`, the selected
`reference_ids`, and, only while the reading is open, `operation_values`
(each action's two operands and result) and `operation_refs`, gathered at
the close from the language layer's journal columns; `ClauseJournal.
finish_clause` turns it into the clause tree, and nothing of it is stored
beside a row. Claude's reading of the fit: each journal action is a binary
composition with two operands and a result, which is exactly an STM1
chunk's content, relations over ideas. The journal would then be STM1's
live form; a recalled LTM1 chain loads into the same structure; and at the
close, what compresses becomes an LTM0 idea and what does not becomes
LTM1 rows that are the journal's own compositions over bound mentions,
with the parse-only operations dropped. For the layout-and-removal spec.

**STM: what is open (Claude, 2026-10-23, at Alec's question).** Correction
first: each kept reading is a sentence; `AnswerProgram` is that reading's
record and is misnamed (14 files use it). Rename it `Reading` in the
layout-and-removal round. Today: `<stmCapacity>` 8 is the depth of the
parse stack in the language layer; the conceptual working slots are the
open sentence's three roles; the recent rows are an index of addresses
(`SituationAddress`, `memory.retrieved_frames`) read by bind; and the
reading's journal holds every reduce's operands and result. Under Alec's
refinement, STM is eight chunks, each over a few ideas and relations. The
opens, with Claude's recommendations, for the layout-and-removal spec:

1. *One structure or three.* The parse stack, the eight-slot working
   memory and the recent-rows index are three things today. Recommended:
   one, the eight chunks; a chunk is a constituent as it is built (its
   idea vector, its composition's two references, its bound identity, its
   order stamp and `.where`/`.when`), and the stack is the chunks in
   reduce order; the journal is their history.
2. *Compression during the parse.* Recommended: at every reduce, as
   composition already produces a value, a chunk whose operands are both
   existing concepts compresses to its idea and keeps no references;
   otherwise it keeps its two references and is written as an LTM1 row at
   the close. "V O becomes a V" is this rule applied inside the sentence.
3. *Capacity.* Eight chunks is a Miller limit. When a sentence needs more
   (deep clauses), recommended: the completed clause chunk is written to
   LTM1 early and held by its head reference, which is what a recalled
   chain looks like; no spill structure beyond LTM1.
4. *Recall and recent rows.* `query()` loads a chain's head reference
   into a chunk and its identities into MTM; recent rows are chunks that
   outlive their sentence until replaced. **Decided (Alec, 2026-10-23):
   a shared set.** The eight chunks are one working set for the open
   sentence's constituents, recalled chains and recent rows together.
5. *Trials.* Each trial owns its chunks, as the fork owns the stack
   today (`SentenceFork`); the kept trial's chunks are what the close
   writes.
6. *Thought.* The thought loop's serial slot (one three-role meaning
   today) becomes a chunk; descend and return navigate a chain's
   references. Belongs with the thought design, after this round.
7. *Gradient.* Chunks are the compiled loop's differentiable state; a
   recalled chain's ideas are detached reads, except identity values,
   which the identity line makes pathwise within a sentence.

**Remaining opens on STM, MTM and ideas with relations (Claude, 2026-10-23,
at Alec's question).** Three, each with a recommendation:

1. *Which chunk leaves STM when the ninth arrives.* Recommended: the least
   recently read chunk that is not an operand still on the parse stack;
   attention chooses what is read, recency what is kept.
2. *Which words a sentence contained, after derivation removal.* Placement
   (`MereologicalCodes.occurrence_terms`), priming's membership diffusion
   (`Priming.diffuse`) and retrieval by word (the leaf index) read this
   from the derivation today; a compressed idea has no references, a
   chain's mentions are occurrences, and MTM's readings expire.
   Recommended: the sentence entry keeps its word bag as owned detail in
   the sidecar (§33 R2), dropped as its first forgetting stage; not a
   relation, not a derivation.
3. *A chain whose unknowns later resolve.* "y is x plus two" is a chain;
   after "x is three", thought can evaluate it. Recommended: the chain is
   immutable history; the evaluation writes y's value to MTM and, when
   kept, a new LTM0 idea as an inference row, as the thought loop writes
   inferences today.

Implementation consequences carried to the layout-and-removal spec, not
decisions: the expectation predictor reads ideas and chain heads instead
of three-slot rows; forgetting values ideas and chains by their own value
with chains protecting what they reference; a promoted identity (2c) is a
minted symbol whose occurrences are its LTM0 ideas.
