# Item 2.0, memory: one file for STM, MTM and LTM, and MTM as one linking function across time (Claude, 2026-11-03, re-specified the same evening, revised three times on 2026-11-04; spec in code names; every decision closed)

**Status.** Decided in direction by Alec on 2026-11-03 ([item 2.0 spec §8](2026-11-03-gradient-projection.md), [Future work](../FutureWork.md) "MTM at order zero: behavioral chains"), scheduled before item 0, the full training session. A first design of the same day, which added a chain entry with its own states field, continuity rule, gestalt matcher, promotion branch and forgetting roots, was withdrawn by Alec as too complex in light of his first answer below; this is the re-specification. On 2026-11-04 a version that sorted links by the order of what is linked was answered by Alec with the connected path of §1, 7; §§2-5 are revised to it from a survey of the code with an adversarial re-check of every claim, and then to Alec's description of the three stores (§1, 8). Governed by step-2 §39 ([2026-10-21-step2-one-ltm.md](2026-10-21-step2-one-ltm.md)). The Codex text (§8) is held until §5 is decided and joins item 2.0's one Codex text after the review of the §41 return. Code names and line numbers are from the 2026-11-04 working tree, which carries Codex's uncommitted completion round; Codex re-locates them. Late on 2026-11-04 Alec added that MTM holds references, from which identity is computed (§1, 9); §9 records what that changes, and §§3 and 4 are revised to it before the Codex text.

## 1. The record (Alec, 2026-11-03 and 2026-11-04)

1. "We might also have MTM operate on order zero concepts as a way of creating behavioral chains? Animals have that, and we might want to store those in MTM before promotion to chained event concepts."
2. "Behavioral chains should be implemented before step 0. Just as it identifies across nouns currently, it can also be used to identify 'the same' event (via gestalt properties, etc), which might even help in the creation of nouns."
3. On the first design's placement question: "Placement should be a single file called memory that contains STM, MTM, and LTM. Note that MTM is just linking things across time; causality for events, identity for nouns, processes for verbs. It's the same function in each case." On its eight further decisions: "all too complex in light of 1, respec and get back to me."
4. In the same reply, on a question about where the compose lesson sits: "If this is just learning MTM, and we have since resolved that it should operate on order zero also, then it should go in conceptual space as a method called from forward(), and probably ramsified." Read here as the placement of MTM's linking: a method of the conceptual space, called from `forward()`, written once. *Alec corrects this reading if it was meant otherwise.*
5. The constraints kept from the loop: MTM serves thought and action only; reconstruction is unchanged when STM and MTM are emptied; no order-0 identity is opened, since a chain at order 0 is a sequence, not an identity.
6. (2026-11-04) "MTM, when it links order-zero concepts (events), creates a single extended event (sequence). When it links order-one concepts (nouns), creates a single extended noun (identity). It exists as a precursor to concepts and (taxonomic) relations that store the same thing."
7. (2026-11-04, on the version that sorted links by order) "Not sure sorting is necessary, but contiguity of the collection (a connected path) is necessary." And: "a connected path is perhaps the intent of collection in MTM, but no worries if we cant make it an enforceable requirement."
8. (2026-11-04) "D4: yes. STM can be seen as a small idea/relation buffer of length 8. MTM has a time horizon, and knits together LTM and STM entries to form per-order identities that maintain a state of affairs (which are also candidate concepts)."
9. (2026-11-04, night, answering item 2.5's item 83, [thinking spec §48](2026-10-27-item2-5-thinking.md)) "The MediumTermMemory maintains a set of references to STM and LTM. Identity can be computed from those references (e.g. "Alec went shopping and then he ate dinner" creates one identity in MTM that has two references to STM0 sentences/clauses with subjects "alec" and "he". it probably makes sense for each reference from MTM to have both the STM0/LTM0 .when for addressing as well as the concept that is referenced within that idea (to disambguate subject/object, etc), which in this case are "alec" and "he"." On its scope Alec chose: item 2.5 applies it to identities with no event of their own, and the full reference form comes here (§9).
10. (2026-11-05, on the cases item 2.5's suite raised, [thinking spec §50](2026-10-27-item2-5-thinking.md)) An identity opened for a known individual also refers to that individual ("Yes, join the individual"). A kept thought's written idea is one more reference of an identity with no event ("One more reference"). On descriptions: "This suggests that the MTM actually needs to check before associating; the overlap between the red ball and the blue ball is zero. The overlap between "alec" and he is a part/whole." Then: "Both now. The object being identified is the noun phrase, so identity may not need to explicitly maintain that representation, since it can be recovered by decoding the references. Although since that decoding is nontrivial for an untrained model, I expect identity (and object permanence) not to function well until the objects themselves have stabilized (which was the problem we ran into before)." And: "(So maximal for one of its members)."

## 2. What exists, in code names

- **Four files hold the three stores.** `bin/STMChunks.py` (the eight shared chunks: `FIELDS`, `empty`, `keep_close`, `locate`); `bin/MediumTermMemory.py` (`MediumTermMemory`, `Token`, `Timeline`, `Event`; state version 10); `bin/Layers.py` (`class LTM0`, the one-slot occurrence rows with their bands); `bin/LTM1.py` (`LTM1`, two tagged references per row, `OCCURRENCE`, `SYMBOL`, `RELATION`, `add`, `stratum`).
- **MTM links across time, twice.** Identity: `ReferenceClose.commit` applies the kept bind; `MediumTermMemory.open` and `bind` append a mention's occurrence to a `Token`'s `occurrences` and join its `value` by `torch.maximum` (`MediumTermMemory.py` about 278-331). Candidates come from the STM chunks only: the preview reads the chunks (`include_pool=False`, `ReferenceClose.py` about 786), and only `PrependedSymbols` reads the pool (about 166). `score` builds eight features (two meet ratios, role prediction, prime, form region, rank recency, candidate role, request role; about 449-470) for a `policy` initialised to `[0,1,1,1,0,0,0,0]` (about 39), with a zero-logit dummy that wins ties. Process: `_commit_events` (in `ReferenceClose.py`, about 1439-1572) appends `Event(occurrence, verb, role, when, before, basis, after, ...)` to the subject's `Timeline`, and `update` sets the timeline's value to the latest state. Both run at the kept close only; every write is detached.
- **No place or time in the choice.** `IdentityRows.coordinates` zeroes the form, where and when columns before every meet (`IdentityRows.py` about 40-51), `meet_ratios` has no distance term, "recency" is an enumeration rank, and `Token.where` is used only as an exact chunk address (`STMChunks.locate`).
- **Order of members.** Members are kept in link (append) order and nothing sorts them. `merge` appends the source's members after the target's (`MediumTermMemory.py` about 425-433), and `test/test_reference_arrival_review.py` (about 44) certifies a result that is not in time order; `_detach` removes a member from the middle when a bind moves an occurrence to another token (about 392-408). Readers that assume time order: `accessible`'s event windows (degenerate in practice, since mention rows are unstamped and every event falls to the tail use), `promote_identity`'s origin `occurrences[0]`, and VerbLearning and ModalLearning's latest event.
- **Clocks.** `Token.when` mixes three clocks: the model tick (`ReferenceClose.commit`), MTM's close count (`FamiliarTier`) and an old recalled tick (`EntryRead`). MTM's own per-member clock already exists, for an identity's named mentions only: `MediumTermMemory.close` records `refinement['mentions'][member] = (stream, close clock)` for each reference in `self.names` and prunes it to `mtmHorizon` (about 473-479); `merge` unions it; recurrence counts its distinct entries (`FamiliarTier.py` about 685-695).
- **LTM re-entry.** Query work (`Queries.py` about 1359-1367) calls `EntryRead.recall_chunks`, which runs `recall_identities` and loads the recalled heads into STM's eight chunks with their stored `.when` (`EntryRead.py` about 189-230). `recall_identities` (about 148-187) walks each frame's identity chain and opens every LTM reference not yet in MTM as a token of its own (`MediumTermMemory.open`, with the recalled row's old tick as `when`). `open` records no MTM clock and joins nothing, even when an open token already holds the same individual.
- **Contiguity today.** Nothing enforces or measures that members form a connected path. The only temporal limit is expiry: an entry with no access for more than `mtmHorizon` (16) kept closes in every stream is detached (about 480-484), and any access counts (binds, events, field activations), not only new members.
- **Orders.** An identity is one order above its members: `promote_identity` reads the order from its first member (`FamiliarTier.py` about 487-490), and `test/test_epoch_history.py` promotes order-0 members to an order-1 identity. Its members must share one order: the definition path raises when a member's order disagrees with its identity's ("witnessed member ... disagrees with the order of", about 641-642). Identity tokens can hold order-0 members (sign and percept leaves open `NULL_CONCEPT` tokens, `ReferenceClose.py` about 1229-1235), and a sentence's state is order 0 too, so order alone does not separate a noun link from an event link. `doc/Orders.md` §3 defines order 1 as occurrences joined into an individual, with continuity, the where/when trajectory, as the evidence, which is the documents' form of the connected path. It also states "Identity is per order".
- **Recurrence promotes.** `FamiliarTier.promote_candidates` counts distinct kept readings within `horizon` against `concept_mint_count` and sends an identity to `promote_identity`: one `('part', ('occ', origin))` constituent, `reference_orders[cid]` one above the members, `settle`, the concept row, `place_rows`, `pool.promote`. It writes no LTM1 membership rows, but `_resolve_null_token` already writes `(OCCURRENCE, member) -> (SYMBOL, target)` rows with each member's when band (`FamiliarTier.py` about 638-645), as `ConceptIndex` (about 70) does.
- **LTM1 and chains.** LTM1 can store an `(OCCURRENCE, a) -> (OCCURRENCE, b)` row and indexes it both ways (`wholes`, `parts`), but it has no column for the kind of a row, and `PartClosing` already writes occurrence-to-occurrence rows when it composes two leaves (about 296-307). No code writes successor rows. `ChainRead.components` walks relation rows and treats an occurrence as a leaf (about 45-46), so it does not step from an occurrence to a successor; `recall_time` reads any stamped row by its when band (`EntryRead.py` about 121-146). Occurrence references are not cycle-checked on load.
- **Order zero today.** `ChunkJournal.push` and `apply` open an individual only at order 1 ("an order-zero event stays an event"). Nothing links order-0 states to one another.

The second version's statements that occurrences are "in time order", that adjacency comes from `recall_time`'s window, and that chain readers "need no change" were wrong; this version replaces them.

## 3. The design, in code names

### 3.1 One file

`bin/Memory.py` holds the three stores: `STMChunks`, `MediumTermMemory` with `Token`, `Timeline` and `Event`, `LTM0` and `LTM1`, moved with their names and behaviour unchanged; `LTM0` leaves `Layers.py`. The four old modules go and every import is repointed; no compatibility alias, under the no-legacy rule. The file states the one boundary at which all three change: the kept close. The design below adds no field to `Token`, so MTM's state version stays 10.

### 3.2 One link, one rule

In Alec's terms (§1, 8): STM is a small buffer of eight ideas and relations (`STMChunks`, `stmCapacity` 8). MTM has a time horizon (`mtmHorizon` 16) and knits LTM's and STM's entries into per-order identities that maintain a state of affairs and are candidate concepts. LTM holds what promotion stores. MTM has one job, linking occurrences across time, and the intent of each of its collections is a connected path (§1, 7). One method does it, with no branch on order or on store: `ConceptualSpace.link(occurrence, candidates, *, value, where, when, stream)`, called from `forward()` at the kept close in place of `ReferenceClose.commit`'s bind and `_commit_events`. It builds one feature row per candidate, the policy chooses, and the dummy opens a new collection.

- **What links, from STM.** Three callers at the kept close pass occurrences to `link`: reference sites (today's binds, and `commit_read_words`'s familiar candidates, questions included), verb events (today's `_commit_events`), and, new, each kept sentence's order-0 state (not an estimate, a question or a cover; a modal or unlowered idea is never a state).
- **What links, from LTM.** Re-entry, run by query work, is how MTM knits LTM's entries to STM's. It stays deterministic and outside `link`: a recalled entry already names its identity, so no policy chooses. Two changes fit it to the one link.
  - Re-entry records the recalled member's link time as the stream's current close (`closes[stream] + 1`, as `keep_access` does; `recall_chunks` already has the stream). A recalled identity is then inside the window from its recall, and occurrences at the close can link to it. Its row's own stamp keeps its world time; the path runs in MTM's time.
  - A recalled entry whose stored identity is an individual already open in MTM (the same `native`) joins that collection instead of opening a duplicate; any other recalled entry opens its own, as today.
- **The path.** A collection is its members in link order. A link appends at the tail, and MTM's close clock never decreases, so link order is time order and nothing is sorted. `merge` interleaves the two collections by their members' close clocks instead of appending (the expected order in `test_reference_arrival_review.py` flips, ported with before and after); after `_detach` removes a member, its neighbours become consecutive.
- **Enforced: the window.** A collection is a candidate only while its latest member is within `mtmHorizon` kept closes of the stream. The time is read from `refinement['mentions']`, which today records only an identity's named mentions; every link and every re-entry now records its member there, so every collection has the record. A longer gap closes the collection: it takes no further link, stays readable and promotable, and expires by today's rule. This is the part of the connected path the code can check cheaply. One constant, 16 = $2^4$; no new knob, no new field.
- **Enforced: likeness.** A collection's members are alike: of one order, which is identity per order (§1, 8; [Orders](../Orders.md) §3), and of one kind (reference sites, an identity's events, states), both set by the collection's first member. The order part is the rule the definition path already raises on, moved forward to the choice. Kind is needed beside order because a sentence state and a word-level mention are both order 0. This is a filter on candidates, not a branch in the function; it keeps a noun mention out of a chain of sentence states whose value contains it.
- **Scored, not enforced: place and continuity.** Four features join today's eight: the overlap of the where bands with the tail member; the gap over `mtmHorizon`; the first difference against the tail member, which is common fate; and presence in STM's chunks. Their weights start at zero, so at construction today's eight features decide among the candidates as today, and the dummy still refuses a weak match. Adjacency in place cannot be a rule, since mentions of one thing sit at different text positions.
- **Candidates.** For every occurrence, the open collections of its stream that pass the window and likeness, from MTM's pool and STM's chunks alike (D4).
- **The choice.** One `policy` over the one row of twelve features, today's eight weights kept and the four new ones at zero; old checkpoints load with the new weights zero-padded, reported. Chosen at the close and credited by the ordinary departure rule; under item 2.0 §2 that credit is on the output side whole.
- **Values: the state of affairs.** Each collection's value is the state it maintains (§1, 8), by today's two updates: a mention joins its description by `torch.maximum`, and an event sets the latest state (`update`), as a state's link does. The first-difference feature reads the tail member.
- **The judge.** The coverage term `reconstruction.reference` (`ReferenceReconstruction.cost`: a closing mention's description covered by the record it was linked to) becomes `memory.link_coverage` for every link: an occurrence linked into a collection should be covered by the collection's value. Its gradient reaches what the coverage reads; the choice stays discrete ([order-zero loop](2026-11-04-item2-0-order-zero-loop.md) D2).
- **Recurrence.** A matched link is one `record_reading` for the collection, as a mention is for an identity today.
- **Lifecycle.** Collections share `mtmCapacity`, `mtmHorizon` and the transient ids. Collections of states are never prepended to the next sentence's field, and likeness keeps them out of reference binding, so MTM stays out of reconstruction (the loop's acceptance check 2).

### 3.3 Orders

Identity is per order (§1, 8; [Orders](../Orders.md) §3): a collection's members share one order, identities at different orders are never joined, and a collection promotes to a concept one order above its members (`_part_order`). Beyond likeness the order decides nothing in `link`. Joining a course of events' states makes an extended event, and joining mentions of one thing makes an extended noun (§1, 6); these describe the result and are not branches in the code. No identity concept exists at order 0, which is what §1, 5 meant by a chain at order 0 being a sequence. The `NULL_CONCEPT` tokens of sign and percept leaves stay as today.

### 3.4 Promotion

MTM is the precursor of concepts and relations that store the same thing (§1, 6), so promotion is one rule for every collection: `concept_mint_count` recurrences within `horizon` promote it to a concept whose members are its occurrences, through `promote_identity`'s path (`('occ', ...)` parts, the order by `_part_order`, the concept row by `place_rows`, the transient id released). The rows written store the path itself:
- membership rows `(OCCURRENCE, m_i) -> (SYMBOL, cid)` for every member, the taxonomic relation of a member to its concept, by `_resolve_null_token`'s existing writer;
- successor rows `(OCCURRENCE, m_{i-1}) -> (OCCURRENCE, m_i)` along the path, with $m_i$'s bands, witnessed on repetition (`LTM1.add` with `witness=True`).

Both are written for every collection, so an extended noun's mentions are as durable as an extended event's states: $2n-1$ rows for $n$ members. They ground the concept in the history, so the epoch boundary withdraws them, as it does every relation that grounds a minted concept, and keeps the concept with its order and its wiring (thinking spec §38.2; Architecture, "Evidence belongs to its source"). The next epoch's reading grounds it again. A successor row is told from a `PartClosing` composition row by its endpoints, both members of one concept as read from the membership rows, so no LTM1 column is added. A new bounded walk, `ChainRead.path(cid)`, reads the path back from the concept's members over the successor rows between them, with a visited set, since a recurring state can close a cycle; `recall_time` reads the rows by time unchanged. A promoted extended event is the chained event concept, the class of a course of events that recurred.

### 3.5 Nouns from chains

An identity that is the participant of `concept_mint_count` consecutive links of one collection of states receives one mention witness per link, the count promotion already reads, and mints through `promote_identity` with no new count. Whether this helps create nouns is a learning measurement for item 0, recorded null if null.

### 3.6 Not here

Training verbs from an extended event's before and after pairs (the loop's part of item 2.0, its §3.8); re-identification across a closed gap; branching chains; prepending chains; any change to `conceptualPi`, to the literal candidates or to sign tokens.

## 4. Gates (one frozen hash, sized to the change; certificates, no learning claim)

1. **The file.** `bin/Memory.py` holds the four classes; the four old modules are absent, with a certificate; every existing test of STM, MTM, LTM0 and LTM1 passes on the hash.
2. **The link, per caller, on one fixture.** Reference sites: the two-verbs-one-identity certificate (`test/test_identity_threading.py`) through `link`. An identity's events: the timeline certificate (`test/test_modal_timelines.py`) through `link`. States: two consecutive kept closes of one stream link two order-0 states into one collection with their stamps and bands; a third close matching by content and bands is a recurrence; a distant one takes the dummy and opens a new collection; a modal clause adds no link.
3. **The window and likeness.** A collection whose latest member is older than `mtmHorizon` kept closes takes no link and stays readable and promotable until it expires; every collection's members share one order; a mention never joins a collection of states. Certificates for each.
4. **The path.** After `merge`, members are in close-clock order (`test_reference_arrival_review.py` ported with before and after); the receipt reports how many gaps `_detach` left on the fixtures.
5. **Re-entry.** A recalled identity records its link time and is a candidate at the next close; a recalled entry of an individual already open joins it, with one token for the individual; the recall certificates (`test/test_document_covers.py`, `test/test_layout_removal.py` and the users of `test/chunk_fixtures.py`) pass, ported with before and after where the join changes them.
6. **Promotion.** A collection of states recurring `conceptMintCount` kept closes promotes once: membership and successor rows with bands, one concept one order above its members, the id released, `ChainRead.path` reading the path back and stopping on a cycle, the epoch boundary withdrawing the rows and keeping the concept and `reference_orders`. An identity promotes the same way, with its mention path.
7. **Nouns.** An identity that is the participant of three states of one collection promotes through `promote_identity` with no new count.
8. **Absence.** No concept promoted from MTM at order 0 (every promotion one order above its members); collections of states absent from `stage`'s leading field; the input reconstruction unchanged with STM and MTM emptied.
9. **The judge.** `memory.link_coverage` replaces `reconstruction.reference` in the registry, scored on every link.
10. **Unchanged, on the same hash:** the MTM, identity, timeline, familiar-tier, forgetting and reference test files, and the core unchanged sets; one native lesson smoke; `make test` at the end of item 2.0. Under D4 a certificate whose identity choice changes because a candidate came from MTM's pool is named in the receipt with its before and after and the reason.

Rules, as in every round: preallocate at construction, never resize; no language named in code, fixtures or tests; no seed selected to pass; existing tests ported with before and after text in the receipt, nothing retired silently; do not read or merge the item 1.5 sandbox branch. Receipt under `doc/benchmarks/2026-11-XX-item2-0-memory/`.

## 5. Decisions

- **D1 The move** (§3.1): decided by Alec on 2026-11-03, "a single file called memory that contains STM, MTM, and LTM". `LTM0` leaving `Layers.py` and the old modules going with no alias follow from the no-legacy rule. Closed.
- **D2 One policy over one row**, with no order input and no policy per kind (§3.2). This follows from "not sure sorting is necessary" (§1, 7) and replaces the second version's one policy per order. Closed unless Alec meant otherwise.
- **D3 Processes are paths too** (§3.2): an identity's `Timeline` is the path of its events, kept apart from states by likeness. This replaces the second version's merged kind and the first version's process promotion. Closed with D2.
- **D4 Candidates from MTM's pool, not only STM's chunks** (§3.2), for every caller: decided by Alec on 2026-11-04, "D4: yes". A thing mentioned again after it has left STM, within the window, links to its identity instead of opening a duplicate. This is the one change to today's identity linking.

## 6. Where

After the review of the §41 return, inside item 2.0's one Codex text, from the commit of the reviewed 2.5 tree, on main. The move into `bin/Memory.py` comes before item 1.5 (Alec, 2026-11-03: the countdown order stands), whose lookups are then ported onto it from the 2026-10-30 sandbox branch. Files: `bin/Memory.py` (new), `bin/Spaces.py` (`ConceptualSpace.link`, called from `forward()` at the kept close), `bin/ReferenceClose.py` and `bin/Models.py` (the bind and event paths routed through `link`, `score`'s row and `policy`), `bin/FamiliarTier.py` (`promote_candidates`, `promote_identity` and `_resolve_null_token`'s membership writer for every collection), `bin/ChainRead.py` (`path`), `bin/EntryRead.py` (re-entry's link time and the join of an open individual), `bin/MediumTermMemory.py` (the link-time record for every collection), `bin/ChunkJournal.py` (unchanged, certified), `test/test_reference_arrival_review.py` (the merge order); `doc/Architecture.md` by append (the one memory file and the one link function) and by exact replacement of its line-10 sentence that calls the LTM layout "specified for a later round"; `doc/FutureWork.md`'s entry gains the pointer; `doc/Orders.md` by append: order 1's continuity, applied to every MTM collection as its connected path.

## 7. Decided (Alec, 2026-11-03 and 2026-11-04)

On 2026-11-03: one memory file for STM, MTM and LTM; MTM one linking function across time, the same for causality, identity and processes; the first design's eight further decisions withdrawn. On 2026-11-04: MTM makes an extended event of events and an extended noun of nouns, the precursor of concepts and taxonomic relations (§1, 6). Sorting by order is not needed, and the intent of each collection is a connected path, enforced where the code can check it (§1, 7). This version therefore drops the order dispatch, enforces the window and likeness, scores place and continuity, and writes every path at promotion. D1 to D3 are closed by those answers. Later the same day: "D4: yes"; STM a buffer of eight ideas and relations, MTM knitting LTM's and STM's entries over its time horizon into per-order identities that maintain a state of affairs and are candidate concepts (§1, 8). Re-entry from LTM records MTM's time and joins an open individual, likeness includes the order, and the values are the state of affairs. Every decision is closed. Late on 2026-11-04: MTM holds references to STM and LTM, and an identity, its state of affairs included, is computed from them rather than stored (§1, 9; §9).

## 8. Text for Codex

§5 is decided; the text joins item 2.0's single Codex text, written after the review of the §41 return.

## 9. MTM as references (Alec, 2026-11-04, night)

Alec's answer to item 2.5's item 83 (§1, 9) describes what MTM holds. Item 2.5 applies it to identities with no event of their own: their meaning is computed each time it is read from the concepts they refer to, with today's join ([thinking spec §48.1](2026-10-27-item2-5-thinking.md), item 96). The full form comes here. It is read as follows; *Alec corrects this reading if it was meant otherwise.*
- **A member is a reference.** It has two parts: the `.when` of the STM0 or LTM0 entry, which addresses it, and the concept it refers to within that idea, which says which role it fills (subject, object and so on). In Alec's example, "Alec went shopping and then he ate dinner" makes one identity with two members: the first clause with "alec", and the second with "he".
- **Values are computed, not stored.** An identity's value and its state of affairs are computed from its members when they are read. `Token.value`, the timelines' stored values and the join in `open` and `bind` retire. The events an identity takes part in are among its members, as the clauses that state them.
- **The rest keeps its role.** `link`, the window, likeness, the path and promotion (§§3.2 to 3.4) work over members of this form.
- **Checkpoints.** A checkpoint carries the members and no values, so MTM's state version changes from 10.

§§3 and 4 are revised to this before item 2.0's Codex text. One question stays open for that revision: how a state is computed from a run of event members, which item 2.5 does not exercise.

**Added 2026-11-05 (§1, 10).** Item 2.5 applies these now ([thinking spec §50.1](2026-10-27-item2-5-thinking.md), items 100 and 101); §§3 and 4 take them over:
- *What a member reads.* A word reads the concept it denotes now. A mention without a word reads its own idea: the noun phrase it identifies, held in its occurrence. An identity's known individual, or the concept it was promoted to, is one of its references too. A kept thought's written idea is a member.
- *Likeness is containment.* MTM associates a mention with an identity only where their overlap is maximal for one of them, so that one meaning contains the other: part/whole, as "he" in "alec". Zero overlap, or a partial one such as two balls of exclusive colours, does not associate. In code, one of the two coverages of `IdentityRows.meet_ratios` is exactly 1. This is `link`'s likeness test (§3.2), beside the window.
- *Expected behaviour.* Until objects stabilize, decoding the references is hard for an untrained model, so identity and object permanence are expected to work poorly early in training. That is an observation, not a fault to repair by loosening likeness.

**Decided (Alec, 2026-11-05).** The open question above is deferred: "Just as some nouns share an identity, some events are mereological neighbors (in world-space, not .where, although we can infer one from the other for words, if not what they represent). We can defer this to future work." Item 2.0's memory round therefore computes no value for a run of event members: an extended event keeps its path (membership and successor rows) without a computed state, and how events relate goes to [Future work](../FutureWork.md#events-as-mereological-neighbours-alec-2026-11-05). *Alec corrects this reading if it was meant otherwise.*

**Decided (Alec, 2026-11-05): MTM at the basic level of perception.** "MTM can be used at the perceptual level to create basic percepts", and "In 2.0, 5 and 6 merged, but MTM only operative for basic level percepts (not parts and wholes)." MTM's link also runs over basic-level percepts, and promotion writes a basic percept ([basic-percepts spec §9](2026-11-04-item2-0-basic-percepts.md)). This revises §1, 5: MTM no longer serves thought and action only. The memory round and the basic-percepts round are one round of item 2.0.

**Decided (Alec, 2026-11-05, later): runs form concepts at every level.** This replaces the deferral above: "We might just say that event runs are the mechanism that is currently used to form concepts at any level; it would work the same as for identity, but only order lower." Read as follows; *Alec corrects this reading if it was meant otherwise.*
- An extended event is an MTM collection of order-0 states, as an extended noun is a collection of order-1 mentions. Its state is computed from its members when it is read, as an identity's value is computed from its references (§9; `MediumTermMemory.value_of`), one order lower. No `update` sets it.
- The run stays a sequence (§1, 5): its members keep their path order.
- Runs are how concepts form at every level. A recurring run promotes by the one rule (§3.4), whether its members are basic-level percepts ([basic-percepts spec §9](2026-11-04-item2-0-basic-percepts.md)), order-0 states or order-1 mentions.
- Events as mereological neighbours in world-space stay in [future work](../FutureWork.md#events-as-mereological-neighbours-alec-2026-11-05) as the fuller account.


## 10. Shared STM/LTM strata and decoding from an idea (Alec, 2026-11-08)

This amendment records Alec's direction after the round-2 interim return and
supersedes earlier readings that require a saved input derivation for decoding.
It extends the memory design; it is not a claim that the shared strata or the
new decoding contract are implemented.

### 10.1 Decided: short-lived instances of the same stores

Alec: "STM0 and STM1 should be short-lived versions of LTM0 and LTM1, sharing
their implementation. Add that to the plan so reconstruction, conditioning and
MTM use the same stored structure."

- STM0 holds live concept/idea entries using the same storage and read
  implementation as LTM0. STM1 holds binary branches using the same storage
  and read implementation as LTM1. Their short lifetime and retention policy
  distinguish them from LTM; a parallel parser-specific tree layout is not the
  intended permanent architecture.
- Reconstruction, input-derived conditioning and MTM read this common stored
  structure. Current temporary `Reading`/`Composition` records and chunk
  references must be accounted for when moving to it; a second authoritative
  tree or hidden per-operation side table would not fulfil the consolidation.
- Shared structure does not make its historical presence a prerequisite for
  decoding. A single retrieved LTM0 idea must also be usable after the original
  input STM and derivation are gone. Any working decomposition created while
  decoding is a fresh proposal, not recovery of a required saved input tree.
- Addressing must distinguish an entry from a reused physical slot. The MTM
  reference form in §9 remains applicable: entry time plus the concept
  referenced within the idea. Shared implementation must preserve entry
  lifetime, reference validity and the kept-close ownership boundary.

Alec also clarified that concept-conditioning context is learned from input.
As its grammar is composed, the composition can supply learned top-down context;
it should not condition syntactic categories too strongly before that grammar
is reliable. Architecture-generated sentences need no separate context-learning
pass. No numerical strength schedule is decided here.

### 10.2 Decided: a syntactic inverse need not recover its original operands

Alec: "the inverse of a syntactic operation is not expected to be fully
invertible" and "we should be able to unfold a surface sentence from only a
single LTM0 idea". The decoding contract is therefore generative matching:
use the idea, active concepts and the verb's available inverse to propose
constituents, and judge their composition against the idea. Exact original
wording and the original syntax tree are not promised by a compressed idea.

An exact conditional inverse, when its witnesses happen to be available, is
one proposal or a diagnostic. It must not impose a saved-object-value,
saved-group or input-journal requirement on ordinary idea decoding. Unknown or
lost operands leave a search problem; they do not themselves mean that decoding
is broken. A numerical failure in a supported calculation still propagates.
For an object-conditioned verb hypothesis, its forward check must use the
hypothesized object's semantics consistently with that hypothesis; substituting
an unrelated identity-basis forward map would test a different action.

This corrects the interpretation behind the main plan's §12.2 routing blocker.
Recorded replay can continue to verify a particular known forward application;
its certificate does not establish the architecture's general decoding ability.
The exact free-decoder search/scoring changes still need implementation review
against this contract; no new schema or search algorithm is declared implemented.

### 10.3 Open representation question: operations in a two-reference branch

The proposal as first discussed is retained below; §10.5 records Alec's subsequent
endorsement of operator concepts and ordered binary branches.

Alec asks whether the argument parts of speech can determine the operation,
retaining the simple binary structure instead of adding operation fields. The
following are design analysis and a proposal, not additional decisions:

1. **Categories can constrain an operation without uniquely naming it.**
   Verb/object and subject/verb-phrase roles may identify a compatible
   application. Coarse category pairs alone need not distinguish, for example,
   conjunction from disjunction between clauses when the connective is absent.
   Lexical identity, operand meaning, order and the parent idea provide further
   evidence. Multiple compatible operations can remain candidates.
2. **The current formal names do not prove identifiability.**
   `PartsOfSpeech.formal_arguments` constructs operation-specific names such as
   `verb_argin1`; `observe_kept` learns participation from the selected grammar
   operation. Reading its name back as an operation is circular unless the
   actual occurrence's category membership independently resolves it. Learned
   memberships can overlap. Any category-only proposal needs an ambiguity
   certificate on actual memberships, not a count of distinct role names.
3. **A possible representation without new row fields:** retain an operator
   as an ordinary concept operand where its identity is semantically needed,
   with nesting made entirely of the existing binary references. Schematically,
   `pair(pair(operator, left), right)` uses two binary nodes; a unary application
   can use `pair(operator, argument)`. These are graph illustrations, not new
   edge opcodes or a specified storage encoding. A reader would interpret the
   operator concept and roles; merely writing a part-of pair does not execute
   an arbitrary operation. This needs a concrete mapping and certificates for
   the actual grammar before adoption. Its cost is more nodes and reads, not a
   wider row. Existing verb/connective concepts should be reused where possible.
4. **No distinction can be recovered from nothing.** If different operations
   produce the same retained representation, their historical selection is
   unrecoverable. General decoding may choose a meaning-compatible realization.
   Where a distinction matters to meaning, conditioning or MTM, retain it as
   concept content/structure rather than silently assuming category uniqueness.

No new operation field is authorized by this question. No representation is
claimed biologically established merely because it uses fewer columns.

### 10.4 Current implementation discrepancy and proposed certificates

The live code is not a bare pair-only LTM1: `Memory.LTM1` registers `operator`
and `grammar_operation`, `PartClosing` writes them, and `refresh_effective`
uses them to limit authority through concept/operation trust and gates. This is
an existing implementation fact, not a decision to perpetuate those fields or
a solution to storing a complete grammar. The shared-store design must audit
these dependencies before choosing how operation distinctions are represented;
removing them without preserving their justified function would change trust.

The implementation review should cover: the same data/read implementation for
STM0/LTM0 and STM1/LTM1; common reads by reconstruction, conditioning and MTM;
reference validity across short-term expiry; competing operations with
compatible argument categories; and surface generation from one retained LTM0
idea with the input journal and input STM absent. That last certificate judges
meaning-compatible realization, not a uniquely recoverable original sentence.
The existing order-zero loop's separation of perception and thought remains;
this amendment does not insert a mandatory grammar-history dependency into
perceptual reconstruction. Scheduling the storage change must be reconciled
with the existing memory/loop rounds rather than silently expanding the
round-2 repair freeze.


### 10.5 Operator ideas in STM0; ordered binary branches in STM1 (Alec, 2026-11-08)

Alec: "STM0 contains operators as well as words, so that STM1 relations can
continue to be (ordered) binary relations. That sounds promising, but to be
consistent we should represent the operations as ideas/concepts: probably those
have to be created a priori."

**Representation direction.** Operations are ordinary ideas/concepts. STM0 can
hold occurrences of operator concepts beside word/concept occurrences; STM1
continues to hold ordered pairs of references, including nested pairs. The
operator is present in the referenced content, not supplied by a new operation
column on every branch. Establishing a reusable operator concept and holding
one of its applications in STM are distinct: the concept need not be recreated
for each sentence, while its working occurrence has the short-term lifetime.
This also makes it available through the shared LTM representation when needed;
its identity is not a parser-private opcode.

**A priori construction: working proposal following Alec's suggestion.** Seat
concept identities for the primitive operations declared by the configured
grammar before reading begins, and connect those identities to their native
operation implementations. A reader must be able to apply an operator concept;
merely assigning it a vector does not define an executable operation. The
construction is analogous in placement, not identical in semantic role, to the
existing pre-built formal argument concepts. It is not an instruction to create
an operation concept for every word or every future compound action.

A priori availability does not by itself decide a fixed concept direction,
full initial confidence or frozen operation parameters. The suggested split is
to declare identity and the primitive execution interface at construction,
while learning the concept's participations, relations, applicability and any
already-learnable numerical action parameters from experience. Preserve the
existing fixed-versus-learned operation contracts until explicitly revised.
This is not permission to feed textual operator names into lexical examples;
operator concepts need no surface spelling unless the model learns one.

**Concrete encoding remains to be specified.** The schematic
`pair(pair(operator, left), right)` preserves order with two binary branches;
it illustrates the direction, not a finalized application convention. Unary
operations, partial applications, reference lifetimes and the relationship
between an operator concept and its formal argument concepts need a consistent
mapping using the shared stores. Any replacement of today's `operator` and
`grammar_operation` authority fields must preserve their trust/gating function
through the chosen representation and readers, with migration evidence.
No executable storage or grammar change is made by this plan amendment.

### 10.6 Decided: the tree collapses at the end of the pinned sentence; no new rows (Alec, 2026-11-08, evening)

Alec, on a review that counted one LTM1 row per application: "If you mean
storing operators in LTM, I would like to collapse the tree in STM at the end
of the current pinned sentence, so that most STM sentences (and all LTM
sentences) still have only (compacted) ideas in XXX0 and relations in XXX1."
And, on a choice between a membership row and a nested pair: "we should not be
adding rows, here or anywhere, without a lot of justification."

**What this fixes.** The derivation tree of §10.5, with its operator
occurrences in STM0 and its ordered binary branches in STM1, is the working
structure of the one sentence on the STM stack (the pinned sentence; `pin` is
the stack flag of `ThoughtChunks.put`). At the kept close it collapses: the
sentence's compacted idea stays in STM0 as a recent entry, the relations the
close writes today (`PartClosing._write`'s `parts.add` rows) stay in STM1 and
LTM1, and the branches and operator occurrences are freed. LTM0 and LTM1 hold
compacted ideas and relations only; an operator is in LTM only as an a priori
concept seat (§10.5), never as an occurrence or an application row. The
decoder unfolds an idea by generative matching (§10.2), not by reading a
retained tree. In today's code the collapse is `STMChunks.keep_close` (the
kept top becomes an off-stack recent entry and its stack peers are freed); the
tree it frees is held today not in STM but in the `ChunkJournal` record and
the `Composition` objects, the parser-private layout §10.1 retires, so the
change is where the pinned sentence's tree lives, not whether it collapses.

**Rows.** A membership row tying an application to its operator, and a
persistent nested pair, are both rejected: neither adds a row to LTM1, here or
anywhere, without a lot of justification. Within the pinned sentence the
operator occurrence is justified as the record of the chooser's choice that
credit reads before the close; after the close it is not needed. Sizing is by
the sentence-length constant, preallocated: a sentence of $n$ words has at
most $n-1$ binary applications, so the pinned region holds at most $n$ word
occurrences and $n-1$ operator occurrences in STM0 and, under the nested
schematic of §10.5, at most $2(n-1)$ branches in STM1, separate from the
`stmCapacity` recent entries. Whether the nested schematic or a flatter
working form is used inside the pinned sentence is an implementation choice
for the round that builds the shared strata, under the same rule.

**Open (for Alec).** `LTM1.operator` and `LTM1.grammar_operation` are today's
collapsed provenance: two integers on a relation row, read by
`refresh_effective` to re-scale a row's authority when a word's trust or a
gate changes after the close. They add no row. Claude's reading: they can stay
as the collapse's residue until the round that moves authority to the operator
concepts' trust; removing them at the collapse would fix a row's authority at
its close. Alec decides whether they stay.

**Decided (Alec, 2026-11-08: "keep them until the round that moves operation
authority onto the operator concepts' trust").** `LTM1.operator` and
`LTM1.grammar_operation` stay as the collapse's residue until that round: two
integers on a relation row, no row of their own, read by `refresh_effective` as
today. That round retires them, with the gate-reopening function preserved
there; no earlier round touches them. The open
point above is closed; §10.5's open items are now the application convention
inside the pinned sentence and the initialization policy of the seats.
