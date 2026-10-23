# Step 2c on today's storage — 2026-10-23

Implementation is ready for a frozen scoped pass. No gate result or review
is claimed yet. Review is mediated through Alec; no direct Claude request
will be sent. Publication and step 2's one full sweep follow that review.
Item 3.0 has not been reached. This is mechanism evidence, not a learning
claim.

The [step-2 specification](../../plans/2026-10-21-step2-one-ltm.md)
§§73–81 records the code review, decisions and each development repair under
§39. The published base is BasicModel `148953f9b` (identity source
`65ab7fbe9`, Claude LAND in identity §34), WikiOracle `175c101`.
[Publication record](publication-identity.json). The required identity
receipt sentence was added; no further identity gates ran.

## Implemented scope

- Three distinct source documents make a reading eligible for a concept.
  Already seated non-word concepts are reused; an unseated reading gets
  the union of its retained occurrence content. Re-observation in one
  document never supplies another document.
- Three distinct source sentences promote a token. Its key, kind, members,
  names, count and current VP state remain; `symbol` identifies its new
  concept, whose content is the union of its immutable mention rows. A
  native individual needs no second seat. Future occurrence reads use the
  promoted symbol, and occurrence activations keep the same token identity.
- Per-stream clocks advance only at kept closes. A reference refreshes its
  access; age greater than sixteen closes expires it after other stream
  accesses also expire. The cursor's document-end teardown removes that
  document's pool entries, names and histories. LTM rows remain.
  Refreshes follow the actual committed mentions, refinements and events;
  several closes can share a model timestamp without refreshing unrelated
  tokens. A token evicted within its own close loses its event history;
  a determined introduction that never closed still fails loudly.
- Today's embedded addresses encode constituent ordinals, not original
  sentence positions. The writer retains the actual close's source pair
  in the consolidation sidecar, without changing row addresses or values.
  This evidence and the reading/symbol association checkpoint with the
  owner; missing legacy evidence is acquired on kept re-observation.
- The existing definitions cache is tested after DEF forgetting and
  restoration; it rebuilds from the Part revision. No second cache exists.
- A standalone order-1 composite gets a site through its declared head.
  Completed S/VP and relation nodes, including surface-wrapped relations,
  are excluded from that exception.
- The two R10 notes are checked explicitly. A same-close alias and a later
  pending self-arrival keep the same anonymous token; a self-choice has no
  refinement to apply. Existing settled-site/no-merge rules and the
  reference-term exclusion stay. The original named-alias cases still
  check nonzero features, matching content and zero reference cost.
- Mention re-closing detaches membership without erasing another retained
  token's event. Capacity, expiry and document-end retirement do the same;
  row reclamation alone removes another token's event witnesses. An
  event lacking factored refs uses its captured reading; an unseated verb
  word with one DEF resolves to that object for role evidence. An ambiguous
  unavailable lexical reading supplies no invented role evidence.

The derivation, its readers, close-time minting, three-slot rows, chooser,
objective and keep rule stay. The new LTM0/LTM1 layout, meaning/form kernel
change and shared STM chunks remain item 3. No new forcing, answer targets,
seat reservation, menu masking, fixture resizing or MTM-read detach.

## Exact test-port ledger

1. `test_medium_term_memory.unchanged` gains an optional `promoted_rows`
   argument. Its exact tensor comparison still covers all LTM/Part tensors
   and every bank coordinate outside explicitly newly seated promotion
   rows. Its default compares the entire bank as before.
2. `test_medium_term_memory.keep` records the pre-commit seat set and calls
   `unchanged_after_consolidation`. That helper asserts every new seat
   belongs to a token/reading promotion before applying the comparison.
   This is the sole port of the former full-bank invariant at promotion;
   no antecedent, count, content or partition assertion changes.
3. The local `close` fixture in
   `test_reference_review10.test_later_ball_mention_preserves_the_boxs_key_members_and_count`
   uses the same explicit promotion comparison (both parameter cases).
   Its key, member, count, exclusion and chosen-reference assertions stay.
4. New event fixture corrections: supply an unseated native word and its
   DEF edge, including the object's declared type, instead of the lookup
   path that seats a word. Every assertion in the new case stays verbatim.
5. `reading_fixtures.commit_reading` returns the actual `registry.space`
   from `_concept_owner`, retaining its separate trust/LTM-switch namespace.
   Its users in the selected-nested, selected-relation, expectation-default,
   item4-native, existence-metadata, joint-objective, sentence-expectation
   and compiled-expectation fixtures keep every assertion. No timestamp
   inventory port is needed after the production refresh repair.

No existing test is retired. `final/source-03/test-ports.patch` and
`assertion-audit.json` give the exact changes, including all added tests.

## Retained development runs

Every directory contains the source diff/hash ledger, initial RNG and
semantic captures, run log and failures. Fixed-operation unit failures
have no paired trial or keep; capture diagnoses say so rather than invent
one. None of these runs substitutes for frozen-source gates.

| Run | Passed | Skipped | Failed | Diagnosis |
| --- | ---: | ---: | ---: | --- |
| foundation-01 | 60 | 0 | 9 | Embedded addresses miscounted as source sentences; overbroad arrival/root guards; new event fixture seated its word. |
| foundation-02 | 66 | 0 | 3 | Two legitimate promotion bank writes need the scoped helper port; new DEF fixture lacked its object type. |
| foundation-03 | 192 | 4 | 15 | Nine relative handles used as row indices; four promoted identity display splits; two local bank-helper ports. |
| foundation-04 | 206 | 4 | 1 | Unnecessary global native-token lookup crossed the frame-access boundary. |
| foundation-05 | 30 | 0 | 0 | Targeted repaired contracts, R10 notes and malformed metadata. |
| expiry-clock-01 | 0 | 0 | 1 | Several closes share one model tick; the original refresh rule kept an unmentioned token alive. |
| capacity-01 | 0 | 0 | 1 | Keep probe: both trials tie in R + A and keep greedy; actual same-close capacity evictions precede the event error. |
| capacity-unit-01 | 0 | 0 | 1 | Four determined objects in a two-entry pool reproduce the event error through the real close. |
| foundation-06 | 150 | 0 | 0 | Corrected owner fixtures, expiry, capacity, and the unchanged missing-introduction rejection. |
| capacity-02 | 0 | 0 | 1 | Driver advances past capacity; standalone recovery mistakes a surface-wrapped relation for an object. |
| standalone-relation-01 | 0 | 0 | 1 | Direct reproduction of the surface-wrapped relation error. |
| foundation-07 | 36 | 0 | 0 | Standalone and participant contracts, expiry, capacity, and loud rejection. |
| capacity-03 | 1 | 0 | 0 | Original driver assertions pass for 1/2/3/8 streams; all paired closes and private generator states retained. |
| event-retention-01 | 0 | 0 | 2 | Capacity and expiry incorrectly remove another retained token's event while its row survives. |
| event-retention-02 | 80 | 0 | 0 | Corrected ownership, MTM, consolidation and identity contracts; row reclamation still removes witnesses. |

## Frozen evidence and review

`final/source-01` retains the first frozen source `253bec194`: 1,650 passed,
twelve existing skips, 24 failed. Its [diagnosis](final/source-01/diagnosis.json)
accounts for all failures: 22 partial-owner fixtures, the timestamp reader,
and the multi-stream capacity case. The capacity probe restores the process
RNG, but the original private corpus-generator RNG was not saved. Its new
presentation and both trials are retained; it reproduces the failure class
and is not claimed as an exact presentation replay.

`final/source-02` retains `babd4f466`: all scoped cases completed with
1,678 passes and twelve existing skips. The later ownership diagnostic
finds the two failures recorded in §81, so this source is superseded.
No result from it is promoted.

`final/source-03` holds the final source audit and selections: the prior scoped
groups, both decomposition arms at their original snapshots/seeds, and
new/direct changed-module files. Gates are run once by source. Every
failure is retained and diagnosed before source changes. No full sweep
runs until this source is reviewed through Alec and published.
