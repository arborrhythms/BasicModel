# Item 2.0, memory: one file for STM, MTM and LTM, and MTM as one linking function across time (Claude, 2026-11-03, re-specified the same evening; spec in code names, with decisions for Alec)

**Status.** Decided in direction by Alec on 2026-11-03 ([item 2.0 spec §8](2026-11-03-gradient-projection.md), [Future work](../FutureWork.md) "MTM at order zero: behavioral chains"), scheduled before item 0, the full training session. A first design of the same day, which added a chain entry with its own states field, continuity rule, gestalt matcher, promotion branch and forgetting roots, was withdrawn by Alec as too complex in light of his first answer below; this is the re-specification. Governed by step-2 §39 ([2026-10-21-step2-one-ltm.md](2026-10-21-step2-one-ltm.md)). The Codex text (§8) is held until §5 is decided and joins item 2.0's one Codex text after the review of the §41 return. Code names are from the 2026-11-03 working tree, which carries Codex's uncommitted completion round.

## 1. The record (Alec, 2026-11-03)

1. "We might also have MTM operate on order zero concepts as a way of creating behavioral chains? Animals have that, and we might want to store those in MTM before promotion to chained event concepts."
2. "Behavioral chains should be implemented before step 0. Just as it identifies across nouns currently, it can also be used to identify 'the same' event (via gestalt properties, etc), which might even help in the creation of nouns."
3. On the first design's placement question: "Placement should be a single file called memory that contains STM, MTM, and LTM. Note that MTM is just linking things across time; causality for events, identity for nouns, processes for verbs. It's the same function in each case." On its eight further decisions: "all too complex in light of 1, respec and get back to me."
4. In the same reply, on a question about where the compose lesson sits: "If this is just learning MTM, and we have since resolved that it should operate on order zero also, then it should go in conceptual space as a method called from forward(), and probably ramsified." Read here as the placement of MTM's linking: a method of the conceptual space, called from `forward()`, written once for the three kinds of link. *Alec corrects this reading if it was meant otherwise.*
5. The constraints kept from the loop: MTM serves thought and action only; reconstruction is unchanged when STM and MTM are emptied; no order-0 identity is opened, since a chain at order 0 is a sequence, not an identity.

## 2. What exists, in code names

- **Four files hold the three stores.** `bin/STMChunks.py` (the eight shared chunks: `FIELDS`, `empty`, `keep_close`, `locate`); `bin/MediumTermMemory.py` (`MediumTermMemory`, `Token`, `Timeline`, `Event`); `bin/Layers.py` (`class LTM0`, the one-slot occurrence rows with their bands); `bin/LTM1.py` (`LTM1`, two tagged references per row, `OCCURRENCE`, `SYMBOL`, `RELATION`, `add`, `stratum`).
- **MTM already links across time, twice.** Identity: `ReferenceClose.commit` applies the kept bind, `MediumTermMemory.open` and `bind` join a mention's occurrence into a `Token` whose `occurrences` are in time order and whose `value` is joined by `torch.maximum`; candidates come from the chunks and the pool (`accessible`, `individuals`), scored by `score` (meet ratios, role prediction, expectation prime, form region, recency, roles) through a learned `policy` with a zero-logit dummy, chosen by the chooser with the ordinary departure credit. Process: `_commit_events` appends `Event(occurrence, verb, role, when, before, basis, after, ...)` to the subject's `Timeline` and `update`s its value. Both run at the kept close only; every write is detached.
- **Recurrence promotes.** `FamiliarTier.promote_candidates` counts distinct kept readings within `horizon` against `concept_mint_count` and sends an identity to `promote_identity`: `('occ', ...)` parts, `reference_orders[cid]` one above the members, `settle`, the concept row, `place_rows`, `pool.promote`.
- **Order zero today.** `ChunkJournal.push` and `apply` open an individual only at order 1 ("an order-zero event stays an event"). MTM's order-0 contents are familiar-field literal candidates and sign tokens; nothing links order-0 states to one another, and nothing writes temporal links to LTM1 (the layout is built, version 14; it has no such writer).
- **Location and time.** Every occurrence carries `.where` and `.when` in its percept bands; the sentence entry alone is stamped; `IdentityRows.coordinates` zeroes the bands out of content comparisons; `recall_time` reads by `.when`.

## 3. The design, in code names

### 3.1 One file

`bin/Memory.py` holds the three stores: `STMChunks`, `MediumTermMemory` with `Token`, `Timeline` and `Event`, `LTM0` and `LTM1`, moved with their names and behaviour unchanged; `LTM0` leaves `Layers.py`. The four old modules go and every import is repointed; no compatibility alias, under the no-legacy rule. The file states the one boundary at which all three change: the kept close.

### 3.2 One function: linking across time

MTM has one job, linking occurrences across time, in three kinds:

| Kind | Links | Entry today | Order |
|---|---|---|---|
| identity (nouns) | mentions of one thing | a `Token` | 1 and above |
| process (verbs) | the events of one identity | its `Timeline` | 1 and above |
| causality (events) | consecutive states of one course of events | new | 0 |

One method does all three: `ConceptualSpace.link(kind, occurrence, *, value, where, when, stream)`, called from `forward()` at the kept close, in place of the three separate paths (`ReferenceClose.commit`'s bind for identity, `_commit_events` for process, nothing for causality). It is written once, with no store-specific names inside it ("ramsified"): it takes a kind, the accessible entries of that kind as candidates, one feature row per candidate, and returns the chooser's choice, including the dummy, which opens a new entry. The kinds differ only in which occurrences are eligible candidates and which promotion follows (§3.4).

- **The entry.** The existing `Token`, with one new field `link` naming the kind. For causality the token's `occurrences` are the kept sentences' order-0 ideas in time order, its `value` the latest state, its `when` and `where` the latest state's; no new entry type, no states field. An occurrence is eligible for a causality link when `store.order_of_row(row) == 0` and the row is not an estimate, a question or a cover; a modal or unlowered idea is never a state.
- **The feature row**, the same for every kind: the content meet of the new occurrence against the entry's current value (`IdentityRows.meet_ratios`); continuity, the overlap of the where bands and the adjacency of the when stamps (`recall_time`'s window); and for the two ordered kinds, process and causality, the first difference of the new occurrence against the entry's last link, which is common fate. The bind's remaining features (role prediction, expectation prime, roles) stay on the identity kind's row.
- **The choice.** One small `policy` per kind, of one shape, with the zero-logit dummy. The identity policy is today's. Chosen by the chooser at the close, credited by the ordinary departure rule; under item 2.0 §2 that credit is on the output side whole.
- **Recurrence.** A matched link is one `record_reading` for the entry, as a mention is for an identity today.
- **Lifecycle.** Entries of every kind share `mtmCapacity`, `mtmHorizon` and the transient ids; an entry no link has touched for `horizon` kept closes expires. No new knob. Causality entries are never prepended to the next sentence's field and never bind targets, so MTM stays out of reconstruction.

### 3.3 Order zero

The identity kind links at order 1 and above, as today, and no order-0 identity is opened; the causality kind links at order 0; the process kind joins an event to its subject's identity. MTM therefore operates at order zero through the causality link only, which keeps "a chain at order 0 is a sequence, not an identity" exactly true.

### 3.4 Promotion

One rule for every kind: `concept_mint_count` recurrences within `horizon` promote the entry to a concept whose members are its occurrences, through `promote_identity`'s path: `('occ', ...)` parts, the order derived by `_part_order` (1), the concept row placed by `place_rows`, the transient id released. The rows written are membership rows `(OCCURRENCE, occurrence) -> (SYMBOL, cid)` for every kind, and for the ordered kinds successor rows `(OCCURRENCE, s_i) -> (OCCURRENCE, s_{i+1})` with the later state's bands, witnessed on repetition (`LTM1.add` with `witness=True`). A promoted causality entry is the chained event concept, the class of a course of events that recurred; its readers need no change (`ChainRead.components`, `recall_time`). A process entry is not promoted on its own: its events already belong to the identity's history and its verb has its concept.

### 3.5 Nouns from chains

An identity whose events are the subject or object of `concept_mint_count` consecutive links of one causality entry receives one mention witness per link, the count promotion already reads, and mints through `promote_identity` with no new count. Whether this helps create nouns is a learning measurement for item 0, recorded null if null.

### 3.6 Not here

Training verbs from a causality entry's before and after pairs (the loop's part of item 2.0); re-identification; branching chains; prepending chains; any change to `conceptualPi`, to the literal candidates or to sign tokens.

## 4. Gates (one frozen hash, sized to the change; certificates, no learning claim)

1. **The file.** `bin/Memory.py` holds the four classes; the four old modules are absent, with a certificate; every existing test of STM, MTM, LTM0 and LTM1 passes unchanged on the hash.
2. **The link, per kind, on one fixture.** Identity: the existing two-verbs-one-identity certificate (`test/test_identity_threading.py`) passes through `link`. Process: the existing timeline certificate (`test/test_modal_timelines.py`) passes through `link`. Causality: two consecutive kept closes of one stream link two order-0 ideas into one entry with the entries' stamps and bands; a third close matching by content and bands is a recurrence; a distant one takes the dummy and opens a new entry; a modal clause adds no link.
3. **Promotion.** A causality entry recurring `conceptMintCount` kept closes promotes once: membership and successor rows with bands, one concept at order 1, the id released, `ChainRead.components` reading it back, the epoch boundary keeping `reference_orders`.
4. **Nouns.** An identity linked through three states of one causality entry promotes through `promote_identity` with no new count.
5. **Absence.** No order-0 identity token except `NULL_CONCEPT`; causality entries absent from `stage`'s leading field; reconstruction unchanged with STM and MTM emptied.
6. **Unchanged, on the same hash:** the MTM, identity, timeline, familiar-tier, forgetting and reference test files, and the core unchanged sets; one native lesson smoke; `make test` at the end of item 2.0.

Rules, as in every round: preallocate at construction, never resize; no language named in code, fixtures or tests; no seed selected to pass; existing tests ported with before and after text in the receipt, nothing retired silently; do not read or merge the item 1.5 sandbox branch. Receipt under `doc/benchmarks/2026-11-XX-item2-0-memory/`.

## 5. Decisions for Alec, with recommendations

- **D1 The move.** `bin/Memory.py` as §3.1, with `LTM0` leaving `Layers.py` and the old modules removed. Recommended: yes.
- **D2 The policies.** One small policy per kind, of one shape (the identity policy being today's), rather than one policy with the kind as an input. Recommended: one per kind, since the candidate sets differ.
- **D3 Process promotion.** Out of scope: a process entry is not promoted on its own. Recommended: yes.

## 6. Where

After the review of the §41 return, inside item 2.0's one Codex text, from the commit of the reviewed 2.5 tree. Files: `bin/Memory.py` (new), `bin/Spaces.py` (`ConceptualSpace.link`, called from `forward()` at the kept close), `bin/ReferenceClose.py` and `bin/Models.py` (the bind and event paths routed through `link`), `bin/FamiliarTier.py` (`promote_candidates` and `promote_identity` on every kind), `bin/ChunkJournal.py` (unchanged, certified); `doc/Architecture.md` by append (the one memory file and the one link function) and by exact replacement of its line-10 sentence that calls the LTM layout "specified for a later round"; `doc/FutureWork.md`'s entry gains the pointer; `doc/Orders.md` by append: order 1's continuity relation applied to events at order 0, as threading, not identity.

## 7. Decided (Alec, 2026-11-03)

One memory file for STM, MTM and LTM; MTM one linking function across time, the same for causality, identity and processes; the first design's eight further decisions withdrawn. §5 to decide.

## 8. Text for Codex

Held until §5 is decided; it joins item 2.0's single Codex text.
