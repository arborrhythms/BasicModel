# Item 3.0 step 2: placement from the registered priming surface, and the geometric channel (Claude, 2026-10-25; for Codex, after step 1's review)

**Status.** The second round of item 3.0 ([hand-off](2026-10-24-handoff-3-0.md),
"Item 3.0, in order", 2). It builds the decided placement objective
([FutureWork, placement](../FutureWork.md#placement-from-the-registered-priming-surface-one-metric-for-sbow-diffusion-and-bind-alec-2026-10-20-proposed-with-the-objective-to-decide),
decided 2026-10-22; [Orders §10](../Orders.md#10-orders-are-learned-definitions-raise-three-compositions-lower-alec-2026-10-21-decided-built-in-item-3),
"one conceptual space"; [Architecture](../Architecture.md), the loop
guard), replaces step 1's interim context source in
`MereologicalCodes.observe_close`, adds the geometric channel the
[layout spec](2026-10-23-layout-and-removal.md) §2 names ("the
nearest-neighbour channel over the surface is item 3's geometric
channel"), removes the per-forward whole-store scans step 1 lists, and
resolves the promotion fold's meaning-side centroid
(`MemoryConsolidation._fold`). Step-2 spec
[§39](2026-10-21-step2-one-ltm.md#39-standing-rule-amended-alec-2026-10-22-codex-continues-documenting)
governs: Codex reviews this against the code as step 1 left it, fixes and
continues, documents each decision in a dated section of this file; only
reversing a decision in §1 comes back to Alec. The code names below are
read from BasicModel `4f136f76fa` (origin main, the hand-off) and are to be
reconciled with step 1's landing where it renamed or removed them.

## 1. Decided (Alec; the record this round builds)

1. **The objective.** A symbol's meaning code is moved to raise its meet
   ratio with the codes the registered surface activates and to lower it
   against negatives, in conceptual space's own metric, never the percept
   cube's; the negatives are the brake on collapse and the collapse
   diagnostic is the measurement (FutureWork, decided 2026-10-22).
2. **One conceptual space.** One surface places every concept of every
   order with the same objective; the per-order strata and the
   stratified read are withdrawn; strata remain the structure of LTM
   (Orders §10, 2026-10-22).
3. **The read point.** Placement reads the surface after the seen bumps
   and the edge diffusion and before the geometric (conceptual) channel;
   the chooser and attention read it after. Codes never read what was
   computed from codes (Architecture, guard against the feedback loop).
4. **No word bag; no stored membership.** Placement is the online update
   at the close from the registered surface, not a recomputed mean over
   stored memberships; membership diffusion is activation over the
   surface with nearest-neighbour selection in conceptual space (step-2
   spec §72, decided).
5. **Centroids on form; meaning by context.** A concept's meaning is not a
   centroid of its members' meanings; it is placed by context. Centroids
   are computed over symbols' forms and stored on the symbol's row
   (context-wholes spec §2b, clarification of 2026-10-24; layout spec §1
   item 12). Zeroth-order concepts start as random codes and are moved by
   the surface.
6. **Working values are never placed before promotion** (identity spec
   §2, "Integration by prepending"); MTM writes nothing into a concept.
7. **Preallocate at construction, never resize; a checkpoint that lacks a
   decided column is rejected by version, loudly** (layout spec §1 item 9).

## 2. The surface, in code names

**The object.** `ConceptualSpace._priming_boosts`, `[B, V]`, one row per
batch stream, neutral `1.0`, on the canonical holder
(`_priming_target()`; per-stage towers delegate to it). Its columns are
the canonical codebook rows, the same index as `similarity_codebook.W`
(`_priming_dim()`, `_csw_row_of(symbol)` maps a symbol to its column).
`priming_weights(batch)` and `relevance_weights(batch)` return the
bounded consumer view `_bounded_priming(surface)` (`primingMaxBoost`),
never the stored energy.

**Allocation.** `_priming_surface(V, batch)` allocates on first use and
reallocates when `B` or `V` changes. This round allocates it once at
construction, `[nStreams, nVectors]` (the configured stream count and the
codebook's `nVectors`, written beside the buffer), and the resize branches
become a loud `RuntimeError`; a surface narrower than the live inventory
is a construction error, not a silent grow.

**Write channel 1, SEEN** (`prime_seen(rows, bump, active_rows)`), in
this order inside one call: decay toward neutral
(`surface * d + (1 - d)`, `primingDecay` 0.9), `Priming.diffuse(owner,
surface, edges, spread, active)` of the standing energy along
`_priming_edges()` (`primingSpread` 0.05; the edge channel only, and
from this round the **taxonomy edges only**: `LTM1` rows whose two tags are
both `SYMBOL`, the kind, part and definition rows `ConceptIndex` and
`DefinitionIndex` read; a chain row, one with an `OCCURRENCE` or `RELATION`
tag, does not conduct (decided, §12); the word-to-sentence membership pairs
and `membershipPriming` are gone), then the bump `+1` at `rows`, clamp at
0.
Callers: `Models._prime_sentence_symbols(sid)` before either trial is
composed, with the sentence's concept rows
(`isp._ar_concept_lookup_rows` for that sentence, each row once), and
`Models._prime_seen_step` after the forward with the fired rows
(`_cs_level_rows`, once per sentence under `_sentence_priming_written`).
The rows `_prime_sentence_symbols` bumps for stream `b` are kept for the
close as `owner._sentence_seen_rows[b]` (a preallocated `[nStreams,
seenCap]` long buffer, `-1` padded, `seenCap` the sentence slot capacity,
written beside it): the registered sentence, which §4 places.

**Write channel 2, DESIRE** (`prime_desire(rows, valence, bump)`):
thought's desire channel, signed, floor 0. Unchanged.

**Energy.** Decay, the conservative diffusion ledger in `diffuse`
(`_last_priming_diffusion`), the feasible-fraction guard and the bounded
read are unchanged. `project_priming_to_towers` (the CS to PS/WS heat
projection) is a consumer and reads through §5's reader.

## 3. The read point: `placement_context(batch)`

A new reader on `ConceptualSpace`, `placement_context(batch)` returns the
stored excess energy of stream `batch`, `(_priming_boosts[batch] -
1).clamp_min(0)`, `[V]`, read from the canonical target's stored surface
and never from `priming_weights`, `relevance_weights` or any value that
§5's geometric term has touched. This is `Taxonomy.heat_mask`'s formula
applied to the concept surface. It is read once per kept close, in
`ReferenceClose.commit`, for the kept trial only, after the sentence's
`prime_seen` (decay, diffusion, bump) and before `consolidate`. Since §5
never writes the geometric term into `_priming_boosts`, the guard holds
by construction, and the gate in §10 checks it byte for byte.

The **activated set** for a placed row `r` is every column `i != r` with
excess `a_i > 0`; the **context codes** are those rows' placed meaning
directions (§4's `placement[i]`, the for-lane), weighted by `a_i`. These
are codes of other symbols, as SBOW's context words are; what placement
never reads is energy that was moved by code overlap (§5).

## 4. The write: `MereologicalCodes.observe_close(reading, batch)`

**State, preallocated in `MereologicalCodes.__init__`:**

- `placement[nVectors, meaning_pairs]`, float, the for-lane direction of
  each codebook row's meaning (unit norm or zero); `nVectors =
  len(initial)`, `meaning_pairs` as today.
- `placement_witnesses[nVectors]`, long, the closes that have placed the
  row; `defined_code`'s witness count (`MeaningCodes.definedness`,
  recurrence 4), today `len(rows)` of the occurrence scan.
- `placement_dirty[nVectors]`, bool, set by `observe_close` and by a
  promotion write, cleared by `begin_forward` (§6 item 5).

Both tensors ride in the codebook's checkpoint extras (a
`placement_extras()` / `load_placement_extras()` pair beside
`ramsification_extras`, version 1); a checkpoint without them is rejected
by version with a message naming this round (decision 7). The against
lane is not placement's: it stays as evidence writes it, zero from
placement, as `occurrence_terms` left it ("on the for pole, with no
against evidence").

**The random start** (decision 5). A row whose `placement_witnesses` is 0
has no direction yet. At its first placement its direction is
`identity_code(key, meaning_pairs, meaning_ones)` (`MeaningCodes`; the
private SHA-256 generator, no global RNG) with `key` the symbol's content
key: `word_surface_for_row(row)` bytes for a word, the allocator's content
key for a minted concept (the same key `IdentityRows` and the promotion
use). Its magnitude is `definedness(0) = 0`, so an unplaced row reads
zero, as today; the first close gives it `1/5` of its direction.

**Which rows a close places.** The registered sentence: the rows
`_prime_sentence_symbols` bumped for stream `batch`
(`owner._sentence_seen_rows[batch]`, §2), which are the kept `Reading`'s
interpreted concepts (`Reading.concept_ids` through `_csw_row_of`, an
object where the word has one, the word's own row where it stands for
itself). After step 3 this set contains the prepended promoted identities
of the window; an unpromoted MTM value has no row and is never placed
(decision 6). Thought's inference writes (`ThoughtStream`,
`write_clause(kind='inference')`) have no reading and place nothing.

**The step, per placed row `r`**, with `q = placement[r]` (the for-lane,
nonnegative):

- Context: `C`, the activated set of §3 with weights `a_i`, restricted to
  rows with `placement_witnesses > 0` (an unplaced row has no code to
  meet).
- Negatives: `N`, `placementNegatives` rows (default 5) drawn without
  replacement from the rows with excess 0 and `placement_witnesses > 0`,
  excluding `r`, by a private `torch.Generator` seeded from the
  sentence's content key (`Occurrence.sentence_key` of the kept reading's
  words, as `identity_code` is seeded); no global RNG, no seed chosen to
  pass. Fewer eligible rows than requested means fewer negatives, never a
  repeat.
- Objective: $J(q) = \sum_{i \in C} a_i \cdot \mathrm{meet}(q, c_i) /
  \mathrm{mass}(c_i) \;-\; \lambda \sum_{n \in N} \mathrm{meet}(q, n) /
  \mathrm{mass}(n)$, with `meet` and `mass` as `IdentityRows.meet_ratios`
  computes them (coordinatewise minimum summed; candidate coverage), and
  $\lambda$ = `placementNegativeWeight` (default 1.0).
- Update: one subgradient step, $q \leftarrow q + \eta \, \partial J /
  \partial q$, where $\partial \mathrm{meet}(q, c) / \partial q_j = 1$
  where $q_j < c_j$ and 0 otherwise; $\eta$ = `placementRate` (default
  0.1); then `clamp_min(0)` and unit normalization (`F.normalize`, the
  direction `defined_code` scales). `placement_witnesses[r] += 1`;
  `placement_dirty[r] = True`.
- The comment at the write states the technique: one subgradient step of
  the meet-ratio objective over the registered surface, negatives as the
  brake on collapse; nothing else.

**Readers.** `derive(indices, b)` writes the meaning block of each
requested row from `defined_code(placement[row],
placement_witnesses[row])` into the two lane slices (`start =
percept_event_width`; the for lane at `start : start + meaning_pairs`, the
against lane at `start + reserved_pairs : ... + meaning_pairs`, as today),
for every row including the retained ones (`cache_rows`: concepts,
individuals, form regions), whose form block alone keeps its cached value
(§7). `_contexts`, `_unseen_context`, `_context` and the per-stream
`occurrence_terms` calls in `begin_forward` go; `context_audit` reports
`placement_witnesses` and the direction norm per row. The interim EWMA of
step 1's `observe_close` is replaced, not kept beside this.

**Where it is called.** `ReferenceClose.commit`, kept trial only, after
`mtm.close(...)` and before `consolidate(...)`, with the kept `Reading`
and the plan's stream; `consolidate` then places the rows it promotes
(§7) with the same registered surface, so a promoted identity is placed
at the close that promotes it.

## 5. The geometric channel: `relevance_weights(batch)`

The conceptual channel of Architecture ("the overlap of placed codes,
computed on read against the active field, never stored"), the
nearest-neighbour selection of step-2 spec §72 ("some selection of
nearest neighbours; we had some matrix method for this"). The matrix
method is the one `Taxonomy.build_semantic_heat` and `_active_heat_set`
already implement for the symbolic heat path: with `r` the excess energy
`[V]` of a stream and `S` its active rows, `z = r_S @ A_S` is the heated
context and `s = A @ z` every row's overlap with it, `A` the placed
for-lane directions `placement` (`[V, meaning_pairs]`, read, not
rederived).

`relevance_weights(batch)` (and `priming_weights`, which it aliases)
returns `_bounded_priming(surface + geometric(surface))`, where
`geometric` is computed on each read and never written back:

- `r = (surface - 1).clamp_min(0)` per stream; `S = r > 0`.
- `s = placement @ (placement[S].T @ r[S])`, zero at `S` itself and at
  rows with `placement_witnesses == 0`.
- The `geometricNeighbours` (default 8) rows with the largest `s` per
  stream receive `geometricSpread` (default 0.05) of `r_S`'s total,
  divided in proportion to `s`; the same amount is taken from `S` in
  proportion to `r`, so the term conserves energy as `diffuse` does. A
  stream with no active row or no placed neighbour adds nothing.
- `prime_seen` and `prime_desire` return `relevance_weights(batch=B)` so
  the reconstruction basis in `_prime_sentence_symbols`, the pyramid's
  priority read (`_assemble_relevance_priority`), the reading scope and
  `project_priming_to_towers` all read after the channel; `placement_
  context` (§3) is the one reader before it. `_priming_boosts` is not
  changed by any read.

This is the replacement for the membership channel step 1 dropped from
`diffuse`: second-order context by code overlap instead of by shared
sentence membership.

## 6. Whole-store scans this round removes

Each is listed in the receipt with its before and after call count per
forward on the short native lesson.

1. `MereologicalCodes.begin_forward`'s `occurrence_terms(b)` per stream
   per forward (every LTM row through `PartRead.occurrence_roots`, every
   Part node through `occurrence_memberships`): gone with §4. If step 1's
   interim `observe_close` kept any of these readers alive,
   `occurrence_roots`, `occurrence_codes`, `occurrence_memberships` and
   `PartRead.word_rows` retire here with their tests listed.
2. `Priming.diffuse`'s membership scan (`occurrence_roots` plus
   `occurrence_memberships` on every `prime_seen`): step 1 removes it;
   this round confirms it absent and retires `membershipPriming`.
3. `Priming.part_edges(owner)` rebuilt on every `prime_seen` (two per
   sentence per stream), today walking every live `LTM1` row and resolving
   `OCCURRENCE` references through `_semantic_rows`: it becomes the
   `SYMBOL`/`SYMBOL` rows only (§2, decided), so no occurrence or chain
   resolution remains, and the edge list is kept in preallocated buffers
   `_priming_edge_src[2 * partsCap]`, `_priming_edge_dst[2 * partsCap]`,
   `_priming_edge_count`, with `partsCap` the store's row capacity
   (`parts.refs` is already `[cap, 2]`), rebuilt only when a symbol row is
   written (`ConceptIndex`, `Definitions`, consolidation increment
   `parts.symbol_revision`; the cache carries the revision it was built
   at).
4. `project_priming_to_towers`'s Python walk over
   `_word_definition_ids()` and `word_descriptions` on every
   `_prime_seen_step`: the bridge (`src` rows, `dst` parts and wholes) is
   cached by the definitions' revision in the same way, in preallocated
   `[bridgeCap]` buffers sized from the vocabulary capacity.
5. `begin_forward`'s whole-codebook rederive (`cb.W[clear] = 0` then
   `lookup_rows` over every `_tensor_row_keys`, each `derive` walking all
   of `layer.features._index` in `_definitions`): rederive only
   `placement_dirty` rows and rows whose form changed (the
   `centroid_forms` revision already tells); the receipt reports rows
   rederived per forward before and after. If the form side's revision
   cannot be read per row without a new index, Codex implements items 1
   through 4, reports item 5's count, and records the decision.
6. Listed, not removed: `MemoryConsolidation.consolidate` and
   `DocumentCover.segments` walk every LTM row once per kept close (not
   per forward); they stay, with their counts in the receipt.

*Reconciled with step 1's receipt (Claude, 2026-10-25).* Step 1's
[remaining-scans.md](../benchmarks/2026-10-25-item3-0-layout/remaining-scans.md)
is the authoritative inventory. Of its rows, this round removes the
per-forward ones: `begin_forward`'s derive of every allocated row (item 5),
`centroid_forms`' per-forward revision enumeration (cache the word list by
`identity.revision` alone), the `_definitions` walk for unchanged rows
(item 5), `IndependentComponents.begin_forward`'s enumeration of every
order-0 row and LTM0 observation (keep a revision-stamped witness cache,
preallocated), and `Priming.part_edges`' per-`prime_seen` walk of every
LTM1 row (item 3; the walk through `_semantic_rows` `_identity` bindings is
part of the cached build). Lazy rebuilds keyed by a revision (`LTM1._index`,
`DefinitionIndex`, `_forward_addresses`, `slot_of_reference`) and the
metered query paths stay as they are; the per-close scans
(`consolidate`, `segments`, `exclusion_pairs`) are listed with counts.

## 7. The promotion fold: `MemoryConsolidation._fold`

As built, `_fold` returns the evidence-weighted mean of the occurrences'
`coordinates` (meaning only, per lane) and `consolidate` copies it into
`W[row]`: the promoted row's meaning is a centroid of its members'
meanings, which decision 5 says it is not. Resolution:

- `_fold(owner, store, rows, order)` returns the **form block only**: the
  evidence-weighted mean (weight `c_plus + c_minus` per row) of the
  occurrences' form coordinates `[:percept_width]`, with the location and
  time bands `[percept_width : percept_event_width]` zero (a promoted
  concept abstracts `.where` and `.when`, layout spec §1 item 10) and the
  meaning block zero. For a promoted reading (a symbol recurring across
  three documents) this is that symbol's form; for a promoted identity
  token, the centroid of its mentions' forms, which the identity spec
  gives a prepended identity as its percept part.
- The promoted row's meaning is placed, not folded: `consolidate` writes
  `W[row] = [form | 0]`, leaves `placement_witnesses[row] = 0`, and after
  its writes calls `observe_close`'s step (§4) for the promoted rows with
  the registered surface of the close that promoted them, so the row's
  direction starts from its random code and takes one step toward its
  context at birth. From then on it is placed when it is in a registered
  sentence (a name read as its object, or, from step 3, a prepended
  identity).
- Any other writer of a row's meaning coordinates is listed in the receipt
  with its disposition (`IdentityRows.mint`'s fold-cache copy if step 1
  left it; the identity line already retired `expose_form` and
  `absorb_form`). `W`'s meaning coordinates are a cache of `placement`
  that `derive` overwrites.
- The 2c promotion assertions that the context-wholes round ported to
  "evidence-weighted centroid per lane" are ported again: the form
  assertion stays a centroid, the meaning assertion becomes "direction
  equals the random code moved one step toward the close's context,
  witnesses 1". Each port is listed with before and after text.

## 8. Configuration

ConceptualSpace space parameters, as `membershipPriming` was, with their
defaults written beside their reads: `placementRate` 0.1,
`placementNegatives` 5, `placementNegativeWeight` 1.0,
`geometricSpread` 0.05, `geometricNeighbours` 8. `membershipPriming`
is removed from the schema and `model.xml`. No language is named in any
fixture this round adds.

## 9. Ports

Tests of `occurrence_terms`, `context_audit`'s occurrence counts, the
per-forward context mean (`test_operators_final_b.py`'s `context_audit`
assertion and its kin), the membership channel's `_last_occurrence_priming`
fields, and `_fold`'s meaning centroid are ported to `placement`,
`placement_witnesses`, the edge-only diffusion record and §7, each with
exact before and after text in the receipt; tests of the retired
`PartRead` readers are retired and listed. A port that changes a decided
behavior stops for Alec.

## 10. Gates (sized to the change; one frozen hash)

1. **Placement moves toward context.** On a short native lesson of a few
   sentences repeated (the layout spec's gate-4 fixture), a word's
   for-lane direction after the last close has a higher mean meet ratio
   (`meet_ratios`, candidate coverage) with the codes of the words it
   occurred with than after its first close; its `placement_witnesses`
   equals its closes; no stored membership exists (the `PartRead` readers
   are gone).
2. **Collapse diagnostic.** `MereologicalCodes.collapse_audit(rows)`
   reports, for the placed rows, the mean meet ratio against the current
   registered context and against `placementNegatives` rows drawn by a
   private generator from a fixed audit key. Over the lesson the context
   ratio rises and the random ratio does not rise (equality within
   `1e-6`); the receipt records both series.
3. **The read point.** Across one kept close, `placement_context(b)`
   equals `(_priming_boosts[b] - 1).clamp_min(0)` exactly; a
   `relevance_weights(b)` read leaves `_priming_boosts` byte-identical;
   with two placed rows that overlap and one of them active, the
   geometric term is nonzero at the other and zero at every unplaced row;
   total energy is conserved to the ledger's tolerance.
4. **No whole-store scan per forward.** Over the lesson with no Part
   write between two sentences, `part_edges` is rebuilt 0 times and the
   tower bridge 0 times; `occurrence_roots` is not importable; the
   rederived-row count per forward is reported (item 5 of §6).
5. **Promotion.** A promoted identity's row has the centroid of its
   mentions' forms in its form block, bands zero, and a meaning direction
   with `placement_witnesses == 1` that is its random code moved one step
   toward the promoting close's context; a checkpoint without
   `placement` is rejected by version.
6. **Unchanged sets on the frozen hash:** the layout spec's gates 1–3,
   4a, 5 and 6 as step 1 left them, G1–G3, the pending-premise
   certificates, the reference, bind-thought and identity files, the XOR
   standing gate and both decomposition arms. The sweep is the item's
   one, at the end of 3.0.

One short native lesson smoke, one source freeze, then the gates on that
hash, in a receipt under the local `doc/benchmarks/2026-10-2X-item3-0-placement/`
(receipts are outside version control since `537523ae04`; the receipt's
hash and counts go in the dated spec section and todo). No learning
claim: gates 1 and 2 are mechanism checks over a handful of closes with
no RNG; the learning measurement is item 0's.

## 11. Not in this round; Claude's calls; Alec's decisions

**Not in this round.** `embed.py`'s SBOW/CBOW pretraining (phase 1) keeps
its sampled negatives; `observe_close`'s step is exposed as a function
(`MereologicalCodes.placement_step(q, context, weights, negatives)`) so a
later round can read the registered surface there, with the corpus
(item 3) or at item 0. STM chunks and prepending (step 3), retire mode
(step 4), orders learned (step 5). Certainty gets no code.

**Claude's calls under §39** (Codex may refine and document; not
Alec's): the subgradient step as the write; negatives by a private
generator from the sentence key; the placed set equal to the SEEN set;
the geometric term by the heat path's matrix method with a top-k
neighbour selection; the preallocated edge and bridge caches by revision;
the default values in §8.

**Alec's decisions, with recommendations** (asked 2026-10-25; the Codex
text for this round waits on them):

1. *Default state of the geometric channel.* Recommended **on** at
   `geometricSpread` 0.05: it replaces the membership channel that was
   live by default, at the value Alec set for the spread. Off until item
   0 is the alternative.
2. *`embed.py` out of this round.* Recommended **yes**: the pretraining
   path is the corpus work and is measured at item 0; the shared
   `placement_step` keeps one objective.
3. *A checkpoint without `placement`.* Recommended **rejected by
   version**, as the layout round's rule says; the alternative loads it
   with every row unplaced (witnesses 0), which guesses nothing but hides
   the mismatch.

## 12. Decided (Alec, 2026-10-25), after step 1's review

1. **The edge channel is the taxonomy.** Alec, asked whether `LTM1` and the
   taxonomy are one format: "They are close enough that they might be the
   same class. Regardless, priming energy can spread purely via taxonomy
   edges (and just note that it does not spread on LTM1 edges)." In the
   code they are one class already: `LTM1.LTM1` is the one store
   (`Layers.py` ≈5867, "LTM1 and the symbol taxonomy, one owner"), and the
   two kinds of row are told apart by their tags, a taxonomy row having
   `SYMBOL` on both sides and a chain row an `OCCURRENCE` or `RELATION`.
   So `Priming.part_edges` reads the `SYMBOL`/`SYMBOL` rows and nothing
   else; a sentence chain conducts no priming energy. §2 and §6 item 3 are
   amended above.
2. **The geometric channel is on by default**, `geometricSpread` 0.05.
3. **`embed.py` is out of this round, and is dormant on the production
   path** (checked 2026-10-25): `train.py`'s phase 1 runs `embed.py train`
   only when embeddings are absent and skips it for the byte lexer;
   `model.xml` has an empty `embeddingPath`; `Models` returns no InputSpace
   SBOW loss under the byte lexer ("perceptual SBOW replaces it"); and
   `MereologicalCodes.derive` overwrites every row's meaning, so pretrained
   vectors would not survive. The word-level SBOW pretraining is a legacy
   path; `placement_step` is the one live objective. Whether to retire
   `embed.py train` is a housekeeping line (todo item 2), not this round's.
4. **A checkpoint without `placement` is rejected by version.**
5. **Reading recurrence by angle** (step 1 review §13 finding 2; Alec:
   "hashes of the idea are too brittle, because the idea will change a bit
   over time. Angle and some theta tolerance"; "promoting after three is
   fine"). `MemoryConsolidation.consolidate` replaces `reading_key` with a
   tolerance match: an idea's content (its `slots` vector with the time
   band zeroed) joins the reading whose representative is within
   `<recurrenceAngle>` of it by cosine, else starts a new reading;
   representatives, their document counts and their first idea's `_identity`
   binding live in a preallocated slab `recurrence_codes[readingCap, D]`,
   `recurrence_documents[readingCap]`, `readingCap` the LTM0 capacity,
   checkpointed with the store; three documents promote as before; the
   representative is the evidence-weighted mean of the ideas it has
   admitted. Default angle 15 degrees (cosine 0.966), written beside its
   read; it is a config, not a seed. This is the first task of the step 2
   round, before its freeze.
6. **The hash record** (review §13 finding 1): one line each in todo
   "Current", the 3.5 line, iteration spec §62 and the local receipt
   README, naming `2067dbc25a`, `3561f39aa0` and `31c3e4a59e` as the
   published equivalents of `75345f891e`, `a34325c85d` and `f095569a1f`.


## 13. Code review and angular recurrence — Codex, 2026-10-25

The starting source is step 1's approved content SHA256 `34848e154f`.
The four history notes and todo links were updated before source work.
§12.5 replaces the unrecorded exact-hash grouping first. LTM0 now owns fixed
reading seats and an assignment for each retained idea; compaction carries the
assignment with the idea. A close rebuilds each assigned group's representative
from its retained ideas, weighted by `c_plus + c_minus`, then admits new ideas
by the closest cosine strictly above `cos(recurrenceAngle)`. Equal scores choose
the earliest seat. Reclosing an idea cannot count its evidence twice. The first
idea's identity binding stays attached to its reading seat and is checkpointed
in a fixed-length sidecar; promoted symbols are a slab column too. Document
counts deduplicate the retained cover's document key, so another presentation
of one cover is not another document. Unpromoted empty groups release their
seats; an empty promoted group retains its symbol association. Capacity
exhaustion is loud. LTM0 tensor schema 12 and semantic schema 8 reject older
recurrence layouts. This refines the retained-cover rebuild already used by
consolidation; it changes no admission threshold or presentation boundary.

The baseline archive and protected function hashes are retained in the local
placement receipt. Its baseline scan observer initially mishandled a scalar
`derive` argument; that observer failure and correction are retained. The
corrected baseline lesson passes without changing source or selecting a seed.

## 14. Placement, cache ownership and fixture ports — Codex, 2026-10-25

The registered surface is allocated on each Space at construction. A smaller
active batch uses a prefix view of the configured stream slab; an oversized
batch, changed inventory width or mismatched device is an error. Hard reset
fills neutral energy in place. This keeps ordinary tail batches without a
resize. The sentence registration uses `serialWordCapacity` (the declared
input slot count in nonserial fixtures). Taxonomy edges have a dedicated
revision changed by SYMBOL/SYMBOL insertion, removal or restore; chain writes
and rewitnesses cannot invalidate it. Seating a previously unseated symbol
also invalidates the projection. The tower bridge has fixed buffers bounded
by conceptual vocabulary seats times the declared native tower inventories,
and invalidates on definition/description or seat changes. Overflow is loud.

Placement takes simultaneous updates from a snapshot of the placed directions,
so the order of sentence seats does not affect a close. It reads the surface
once after MTM closes; promotion reuses that read. The legacy non-component
close uses the same ordering. Inference writes have no Reading and do not
place a sentence. Direct consolidation fixtures can supply the context and
content key explicitly. The five controls use the stated defaults; geometry
is a read-only energy transfer and the bounded consumer normalization remains.

There was no allocator content-key column for minted concepts in step 1.
A fixed 32-byte key per codebook seat now captures a minted concept's content
before its meaning cache is cleared; lexical rows use their existing word
bytes directly. This is checkpointed with placement, and does not depend on
a row id, model time or a chosen seed. `IdentityRows.mint` retains the form
only on paired layouts; its meaning remains unplaced until a registered
close. Full-width legacy layouts have no separate meaning block and keep
their existing native vector behavior. `IdeaEvaluation` is unchanged.

Form caches have fixed shape. Native feature/value or percept-identity
revision changes invalidate them; otherwise `derive` reads cached forms and
placement, including for retained identities. `begin_forward` refreshes only
dirty allocated rows. There is no per-row optimizer revision, so a changed
native parameter revision invalidates the form cache as a whole; actual rows
recomputed are reported. The centroid word enumeration is cached by percept
identity revision, with explicit invalidation when a new native seat appears.
The independent-component dictionary seats and LTM observation witnesses have
fixed revision caches. Revision rebuilds, kept-close scans and metered query
paths remain visible in the receipt.

Ports replace lazy-reset `None` with in-place neutralization, occurrence
context counts with placement witnesses, and meaning centroids with a form
centroid plus one placement step. The old graph fixture constructed a one-seat
LTM1 and relied on resizing; it now constructs the saved capacity. The large
edge fixture supplies SYMBOL tags explicitly. These are same-round fixture
repairs. Every exact before/after text will be in the receipt; nothing is
retired silently.


## 15. Pre-freeze regression findings — Codex, 2026-10-25

The placement certificates first passed the context-rise, collapse, read-point,
cache and checkpoint checks; promotion exposed a local variable shadowing the
sentence key, corrected before freezing. Independent-component witness caches
must still filter newly submitted innovations against their cached retained
set on each forward. They now do so without re-enumerating LTM observations.
The ordinary checkpoint front-door check advances with LTM0's schema 12;
new checkpoints must not be rejected as pre-layout by that earlier check.

The other paired-cache writers are `DocumentCover.open_cover` and forgetting:
a new cover keeps its existing private random draw and form, pins its content
key, and clears bands and meaning until placement; reclaiming a native seat
also clears its placement, witness and content-key columns. This prevents a
reused seat inheriting a forgotten symbol's meaning. The legacy full-width
paths are unchanged. Post-close leaf tests compare their captured form with
the retained form, and separately certify the new placement witnesses; the
captured pre-close meanings are zero and the live post-close meanings are now
placed, as §§3–4 require. Two-stream unit fixtures now declare two streams at
construction. No fixture chooses a passing seed.

## 16. Pre-freeze integration and cost-certificate ports — Codex, 2026-10-25

The scoped development run completed 971 cases (942 passed, seven standing
prerequisite skips, 22 failures). The failures and their original RNG states
remain in the receipt. A kept Reading argument in `ReferenceClose.commit`
was shadowed by a local identity id; the kept value now has its own binding.
The native attention heat reader requests the active batch prefix of the fixed
slab. `SymbolSpace` deliberately bypasses the Space constructor and owns no
codebook; its existing no-surface consumer contract is explicit rather than
falling into the new fixed-slab reader. Its native conceptual heat still belongs
to ConceptualSpace. Reading a lexical description must not advance its bridge
revision; only changes to that description advance it.

Presentation reset clears the new recurrence evidence just where it cleared
`_recurring_readings`, including when given rows survive; learned placement and
promoted concepts survive. `BasicModel.begin_presentation` is unchanged. The
relevance fixtures now check neutral constructed buffers, raw energy decay plus
the bounded consumer, live heat before the existing document reset, and the
current native attention reader in place of the already removed hard-coded
scope method. The packed-versus-single fixture uses separate owners from the same captured
ambient initialization: closes are now learning writes.

One old snapshot assertion used zero reconstruction cost as a proxy for a
shared code snapshot. Replaying its first recorded failing RNG on the approved
step-1 archive passes; on placement it exposes `reconstruction.content` alone.
The forced prefix fixture still recovers all four word multisets and has zero
composition residual. Its two distinct negative word meanings can now have a
zero meet under disjunction, losing meaning while retaining recoverable form;
the existing content loss correctly charges that loss. Restoring zero cost
would hide the loss or constrain the decided placement. The port directly
checks identical bank objects, bank codes, placement and witness counts for
both training trials. The forced certificate keeps exact text and residual
assertions, checks zero cost before placement, and checks that its later cost
is exactly the content charge, positive on the losing pair. No production
reconstruction objective or operator was changed. Exact before/after text and
the two-source replay are retained in the receipt.

## 17. Remaining fixture defects and sentence-entry ownership — Codex, 2026-10-25

The packing parity fixture was not in step 1's scoped gate set. Its first
recorded initialization fails the nontruncation assumption on both the approved
archive and this round, with identical untrained projection programs. The port
compares truncation status between layouts rather than assuming successful free
decoding; it retains all reference, action, root, recovered-value, byte-cost and
owned-program equality assertions, and records each truncation. Because an
untrained contextual proposal is not a fixed program across packing boundaries,
the final fixture declares a source-order read and sum proposal at the existing
chooser seam, using eager native loop bodies. It tests the packing driver and
ended-state ownership, not learned choice or compilation. Both arms construct
separate owners from one captured ambient initialization, repeated for the three
original ambient cases; no seed is selected. The decoder success certificates
remain the unchanged XOR and decomposition gates.

Running the surrounding item-5.5 fixture file exposed its remaining retired
`when`-column read. It now checks clock translation in the percept time band,
with every other coordinate still compared. Located values can change native
content addresses when their time changes; references are compared by physical
row correspondence and their semantic metadata is checked under that mapping.
The predictor now sees percept time, so its numeric prior logits are not a clock
invariant; their identities, orders, keys and links remain invariant. Programs
and the actual answer remain byte-identical across all nine original clocks. Its interleaving probe
exposed a real closing defect: a projecting wrapper could reuse an already
written embedded mention as the sentence entry. That entry was unverified and
had no sentence stamp, so linking the prediction to the external observation
failed. The close's memo now applies to embedded uses only; a sentence writes
its own kind/address/time, and preflight counts that entry separately. This
preserves semantic compression and introduces no composing edge. A focused
sentence/mention regression is added. This is a repair of the existing closing
contract, not work on mode retirement.

Angular recurrence identity extras also accept JSON-list representations of
local occurrence references, matching the existing tuple/list load contract.

## 18. Frozen measurement protocol ports — Codex, 2026-10-25

The native smoke passed once (2.33 s), then source SHA256
`1e94238b57c99ba2b1003de2467e6f6acf543b68dc425263bed708509677d55e`
was frozen. No source/test/config file changed after that freeze. The initial
core worker was stopped at its 8 GiB per-worker bound while retaining the
packing fixture's model owners. It had completed 545/998 cases, with zero
assertion failures. Completed cases are retained; the 453 unfinished cases
resume on the same hash, with packing cases isolated one per worker. Already
started unfinished cases restore their recorded RNG, rather than resampling.
This is a process-layout repair, not a new source freeze or a sweep.

The original decomposition footprint observer reports two new transient fields
as unexpected: `MereologicalCodes._form_valid` (a nonpersistent validity buffer)
and `form_rows_derived` (an integer diagnostic counter). A measurement-only
adapter records the full raw differences and removes exactly those two fields
from the learning-mutation proof. It verifies the buffer is nonpersistent.
Placement, witnesses, keys, form values, all parameters, old-row digests, source
budgets, descend/return/conclude checks and original initialization snapshots
remain in the unchanged proof. The first two failures and exact observer
before/after text remain in the receipt. The adapter has its own protocol hash;
the original frozen files and protocols remain intact.

The scan receipt reports actual calls rather than claiming global absence of
scans. In the cold native lesson, feature-index walks are 169 → 10 and native
word enumerations 170 → 2. Boundary rows requested are 11 → 11; a warmed gate
requests zero unchanged rows. The retained exclusion scan is called 23 times,
including candidate planning; it is not literally one call per kept close.
Kept-close consolidation remains two calls and cover segmentation six. Cached
reads and permitted revision/query inventories remain listed.

## 19. Two standing-fixture measurement ports — Codex, 2026-10-25

The resumed core completed its remaining 450 cases: 444 passed, four standing
prerequisite skips and two failures. Both exact initial RNG states were replayed
on the approved step-1 archive and the frozen placement source. The original
failures, replays and the following measurement-only before/after text are in
the placement receipt; the source archive, including its original test files,
has not changed.

The native-ID capture fixture can select `CompoundUnaryLayer` in an untrained
two-stream forward. Its form is `[2, 1, D]` while its case meaning broadcasts to
`[2, 2, D]`; the same error occurs on step 1 at the recorded initialization. This
is not a placement regression, nor is that broad unforced operator path claimed
repaired. The ID-ownership certificate now declares source-order attention and
a sum/stop proposal through the existing chooser seam, preserving every native
ID, allocator-row distinction, later-staging and detached-ownership assertion.
The port does not select a seed or claim learned choice. The separate native
compound-case broadcasting defect is retained explicitly for review.

The cached-mention fixture used the sentence's address as the mention's token.
The §17 closing repair correctly gives the sentence its own external entry.
Its durable `_identity` is still the selected token; the port asserts that
binding equals the token and asks the kind reader about that identity. The
original embedded-mention answer and reassignment assertions remain. This
checks the compressed idea's retained identity rather than depending on the
sentence/mention memo collision. The first adapter attempt used pytest's
original callable despite the replacement fixture signature; that setup error
and its correction are retained separately as well.

## 20. Frozen measurement and review handoff — Codex, 2026-10-25

Implementation and scoped measurement are complete for Alec to return to
Claude. This is a review candidate; Alec commits. No commit, push, publication
or full sweep was performed. Step 3 remains the separately authored STM-chunks
round. The one native smoke passed, followed by one source freeze:
`1e94238b57c99ba2b1003de2467e6f6acf543b68dc425263bed708509677d55e`.
All 888 source/test/config files and the original protocol hashes still match
that freeze. The step-1 comparison has 41 changed files: 21 implementation/config
files and 20 test/helper files. `IdeaEvaluation` and `BasicModel.begin_presentation`
remain byte-identical to the approved source.

The final explicit protocol has **1,003 passes and seven standing prerequisite
skips** across 1,010 unique cases: 998 core, eight XOR and two decomposition
lengths in each of two arms. G1–G3 pass 32/32, the forced pending-premise
certificates 2/2 and the layout file 19/19. Both original decomposition arms
keep their seeds, initial snapshots and 512/768 budgets; all eight companion
questions are correct. Their footprint proof uses §18's observer port.

These counts include §19's two explicit fixture ports. The unported core itself
records **989 passes, seven skips and two failures**; the receipt keeps those
results separately, including the pre-existing unforced compound-case shape
defect. No original failure is relabeled as an original-test pass. The initial
worker memory stop, both initial decomposition observer failures and the first
fixture-adapter setup failure also remain. No source file changed after the
freeze and no random initialization was selected to obtain a pass.

On the eight-close mechanism lesson, mean context meet ratio rises
`0.229750529 → 0.884333044`; the fixed audit-negative ratio falls
`0.325180680 → 0.094838891`. The receipt prints both complete series, including
their intermediate increases and decreases; no per-close monotonicity or
learning claim is made. Each registered word has eight witnesses. Warmed
taxonomy/bridge rebuilds, unchanged boundary-row requests and native form
derivation counts are zero. The cold native lesson and all retained scans are
reported separately, including 23 exclusion reads during candidate planning.

The [local receipt](../benchmarks/2026-10-25-item3-0-placement/README.md) has
README SHA256 `a2342de738bd35354ed5ad26c2ffd996c024226e8887e2643b83fb3c31ec8dc2`.
It links the exact before/after text for 24 source-test ports, nine added tests
and 14 helper changes, plus the two standing-fixture measurement ports and one
observer port. No test was retired in this round. `gate-summary.json` reconciles
every selected case with its original and final result; `remaining-scans.md`
and `scan-counts.json` retain the before/after inventory. Todo Current and the
item-3 block point here and to step 3.

## 21. Claude's review of step 2 (source `1e94238b57`) — 2026-10-25

Reviewed: §§13–20, the [receipt](../benchmarks/2026-10-25-item3-0-placement/README.md),
`gate-summary.json`, `frozen-source.json`, `scan-counts.json`,
`remaining-scans.md`, `standing-test-ports.json`,
`measurement-test-ports.json`, the config schema, and the new text of
`Priming.part_edges`, `Priming.geometric`, `Spaces._priming_surface`,
`placement_context`, `relevance_weights`, `MereologicalCodes.placement_step`,
`place_rows`, `observe_close`, `begin_forward`, `derive`,
`MemoryConsolidation.recurrence_groups`, `_fold` and `consolidate`.

**Verified.** All 888 manifest paths hash to their entries on disk and the
canonical map hashes to `1e94238b57…`; no file changed after the freeze.
The core's 998 cases are the union of the memory-stopped first worker (545
completed, no failure) and the resumed 453 on the same hash; XOR 8, both
decomposition arms 2 and 2 on their original seeds, snapshots and budgets
through the disclosed observer port. The unported core is 989 passed, 7
known skips, 2 failures, and both failures reproduce on the approved step 1
archive at the same recorded initialization, so neither is a placement
regression; the two measurement ports are separate protocols, the frozen
test files untouched, before and after text recorded. No test retired, no
language named, no seed added; `IdeaEvaluation.py` and
`begin_presentation` match their step 1 hashes. The hash notes of §12.6 are
written in all four places.

**The round is the spec.** `_priming_boosts` is allocated at construction
and `_priming_surface` raises on any resize; `part_edges` reads only rows
with `SYMBOL` on both tags, cached in fixed buffers by `symbol_revision`
(§12.1); `placement_context` returns the stored excess, and both priming
writers return `relevance_weights`, whose `geometric` term is the heat
path's `(r @ A) @ A.T` overlap over `placement`, a conservative transfer
computed on read and never written back, on by default at 0.05 with eight
neighbours (§12.2); `place_rows` takes one simultaneous subgradient step of
the meet-ratio objective from a frozen snapshot of the directions, random
start by `identity_code` of a 32-byte content key kept per seat
(`placement_keys`, Codex's §14 addition, needed because step 1 had no
allocator key column), five negatives from a private generator seeded by
the sentence key, for-lane only; `derive` reads
`defined_code(placement, witnesses)` for every row including retained ones;
`begin_forward` rederives dirty rows only; the extras are versioned and a
missing layout is rejected (§12.4); `_fold` is the evidence-weighted form
centroid with bands and meaning zero, and `consolidate` places the promoted
row at its promoting close; recurrence is by cosine within
`recurrenceAngle` 15 degrees over fixed reading seats with document counts
and the first identity binding, three documents promote (§12.5). The read
order in the native close trace is
`commit → mtm.close → observe_close → placement_context → consolidate`.
Gate 2's series: context meet ratio 0.230 to 0.884, audit-negative ratio
0.325 to 0.095, not monotone and not claimed so. Scan counts: feature-index
walks 169 to 10, native word enumerations 170 to 2, warmed rebuilds zero;
`derive` calls rise 169 to 415 because cache-backed reads replaced the
whole-codebook rederive, which the receipt says.

**Findings.**

1. *Recurrence slab size (for Alec).* §12.5 said `readingCap` is the LTM0
   capacity; Codex built it so. At production (`ltmCapacity` 262,144, the
   muxed LTM0 width) `recurrence_codes` is on the order of a gigabyte,
   doubling LTM0's own slab for a table that can never hold more live
   readings than the concept pool promotes plus the pending groups.
   Recommended: a `<recurrenceCapacity>` parameter defaulting to
   ConceptualSpace's `nVectors` (65,536), loud on exhaustion as now; first
   task of step 3, which rebuilds the fixed slabs anyway.
2. *A pre-existing compound-case defect (carry).* In an untrained
   two-stream forward `CompoundUnaryLayer` broadcasts its case meaning to
   `[2, 2, D]` beside a `[2, 1, D]` form; it fails on the step 1 archive
   too. Not placement's; Codex says so. Step 3 reshapes the compose window
   and should repair it there, with the native-ID fixture restored to its
   unforced form once it does.
3. *Reconstruction's content charge now sees placement (watch).* The old
   bootstrap-snapshot proxy of zero reconstruction cost is gone: with
   placed meanings, disjunction over two distinct placed codes can lose
   meaning while the form stays recoverable, and the existing content term
   charges it. The port asserts the exact charge rather than zero. This is
   the first point where placement meets the reconstruction objective; item
   0's learning measurement will see it, and the invertible-path work
   (layout spec §2, "reconstruction's objective moves with the kernels")
   is where it is resolved.
4. *A close repair rode in (accepted).* §17's sentence-versus-embedded-
   mention memo collision is a repair of the existing closing contract
   with its own regression; it introduces no composing edge and belongs to
   no later step.
5. *Commit hygiene.* Step 1 was approved but not committed before step 2
   began, so both rounds sit in one working tree and the step 1 hash is no
   longer the tree. The step 1 archive is retained
   (`step1-baseline.tar.gz`), so either one commit naming both frozen
   hashes or two commits from the archive are possible; one commit is the
   simpler record and loses nothing the receipts do not keep.

**Verdict.** Step 2 is approved as a mechanism round: the registered
surface places the codes with the decided objective, the geometric channel
is the chooser's and attention's and never placement's, the taxonomy alone
conducts, the whole-store scans named in §6 are gone or cached, and the
promotion fold is form-side. Alec commits; step 3 begins on this source
with findings 1 and 2 as its first tasks. No publication until the end of
3.0.


## Item 2.9 amendment (2026-10-27)

Section 12 decision 1 (taxonomy edges conduct, default on) is superseded by
[item 2.9, section 8](2026-10-26-priming-location-diffusion.md#8-decided-alec-2026-10-26-and-the-design-as-amended).
Only placement conducts priming. The retained directed membership cache
supplies net positive evidence weights for class centroids. Placement reads
only the separate seen ledger, never propagated or desire energy.
