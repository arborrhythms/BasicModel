# Item 9b: September 26 corrections

This is the review handoff for the four corrections in
[todo](../../../todo.md) and the latest
[9b plan](../../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md).
It supersedes the September 25 follow-up's interpretation default, training
context pass and integer-only address description. Changes remain uncommitted
for Claude's review.

## Resulting behavior

The parallel context pass runs before the corresponding serial sentences and
takes no gradient or optimizer step. It primes the shared inventory through
ordinary admission, participation and attention. Serial reading supplies the
training steps; there is no separate naming/read-back pass.

`interpret` first uses the object already associated with a word. If *cat*
already names the stored cat kind, an ordinary read returns that kind. The
order-1 default applies only when the word has no object and one must be
created. A grammar request selects among multiple associations or specifies
the order of a new object; it does not create a particular beside a sole
existing kind. Both operator-created and witnessed word associations are
covered.

Each spatial address now travels in the `.where` band itself: two sine/cosine
pairs, one for range and one for resolution. A model owns one encoding shared
by InputSpace, PartSpace, every WholeSpace and SymbolSpace. Construction assigns
their disjoint ranges and derives the periods from the entire allocation.
Captured programs carry these bands. An integer address is decoded when
needed; it is not stored as a competing coordinate beside the program.
Grouped percepts retain a band for each member.

Integer modulo happens before float32 conversion. Thus an address above
2²⁴ does not lose its adjacent-address distinction before its fine phase is
encoded. The round-trip probe covers a 512,808,192-location registry, including
adjacent addresses at its upper end. `decode_index` returns the integer address;
`decode` remains available for continuous floating positions. Copying a model
preserves the shared encoder rather than creating a second registry ladder.

One shared `.when` ladder covers the configured LTM capacity. Everything
observed in one field receives the same stamp; byte offsets no longer advance
time. The exact model clock remains alongside the band. Resetting one field
clears its time band without changing other fields. The later item 7 seal
still owns adding these field coordinates to the LTM row schema.

Content-only spaces still use their complete vector width. Their generic
encode/decode helpers check the space's local band widths before calling the
shared encoders. A review probe reproduced the otherwise silent overwrite of
eight concept coordinates and now guards both directions.

The former per-space period and rung-ratio XML knobs are retired. Capacity is
fixed at construction, and exceeding it stops with an actionable error. Item 3
now budgets approximately 200,000 word forms, 200,000 associated objects and
room for non-verbal concepts. At 1,048,576 rows × 1,032 float32 values, the
concept dictionary is 4.33 GB (4.03 GiB), without Adam moments on its
rotation-owned rows. PartSpace needs a few hundred thousand rows; WholeSpace
occupancy still needs measurement. The 65,536/32,768 reserves are small-run
settings.

## Validation and measurements

The 663-file source snapshot has aggregate SHA-256
`86c9aa27bed5345a57e253a3f85668066789499555ba145fca0af960173f17d6`.
It is available as a [file hash map](source-manifest.json), a
[source archive](review-source.tar.gz), and a
[patch against the reviewed follow-up](changes-since-followup.patch).
The associated [item 9 protocol](../2026-09-26-item9/PROTOCOL.md) fixes
three seeds, four training conditions and 64 updates before interpreting any
learning results.

The final default sweep completes **4,920 unique cases: 4,595 passed,
324 skipped and one existing expected failure**, with no unexpected failures.
It takes 1,880.10 seconds, with a maximum worker footprint of **7.19 GiB**
under the unchanged 8 GiB cap. All twelve new correction regressions pass.
The explicit slow reconstruction selection passes **4/4** in 164.87 seconds,
peaking at 6.66 GiB. No new skip, expected-failure marker, selected seed or
relaxed tolerance was introduced.
Final documentation-link verification passes **96/96**.

The [validation summary](validation-summary.json) links these results through
their exact source map. Raw evidence is preserved in the
[full result](full/result.json.gz), [full source manifest](full/source-manifest.json.gz),
[full worker logs](full/workers.log.gz),
[slow result](slow/result.json.gz), [address probe](address/result.json),
and [measurement manifest](measurements/manifest.json). The
[diagnostic index](diagnostics/index.json) retains failed, interrupted and
superseded checks; earlier affected passes are diagnostic evidence, while
this full sweep covers the final test-fixture source too.

The first complete sweep reached all 4,914 cases and exposed an obsolete
per-space-encoder assertion and an ARM Inductor C++ error when padding a fused
boolean reconstruction mask. A focused reproducer failed with the same mixed
`VecMask` types. Padding the mask as integers, then restoring its boolean type,
preserves the decoder's values and gradients and avoids that compiler error.
No reconstruction assertion or compiler backend was relaxed. Additional red
probes caught encoder sharing lost during copying and a stale time band after
per-row reset.

An intermediate full rerun was deliberately stopped after 1,690 of 4,918
cases when the content-only helper probe exposed that last regression. Its
completed cases had no failures; it is retained as an interrupted diagnostic,
not a full passing receipt. Final checks use the corrected helper source.

A subsequent sweep stopped at the unchanged 8 GiB worker limit after 3,731
of 4,920 cases, with no assertion failures. That worker combined eight files
and reached 8.54 GiB while constructing another STM fixture. The isolated
seven-case file passed, peaking at 7.20 GiB. The final sweep therefore uses
one file per worker batch, at most 32 cases, with the same 8 GiB cap. No test,
assertion, seed or capacity limit changed to address this stop.
All seven STM cases pass in that final sweep, peaking at 7.19 GiB.

The explicit reconstruction selection passes **4/4**: all three ambient
initializations and the excluded-candidate value/gradient check. The spatial
probe recovers **6,144 addresses** and the temporal probe **4,096 times**
exactly on both CPU and MPS; the spatial CPU check also runs through a full
Inductor graph. The registry has 512,808,192 locations, with periods
536,870,912 and 512. The LTM probe has 1,048,576 rows, with periods 2,097,152
and 256. Both probes and the reconstruction checks match the source above.

### Serial baseline and packed parity

The source-matched seven-update serial run has reconstruction cost
**.1505906619 before training, .1347484022 during training and .1362805218
after training**. The previous follow-up measured .1005906649, .0949082300
and .0882481802. The coordinate correction changes the event bands consumed
by the reconstruction objective, so it also changes the training trajectory.
This is the new comparison baseline, with no improvement claim. The serial
run does not use the parallel context pass.

Packed and single-sentence layouts still have exactly equal roots, saved
program roots, references, recovered leaves, actions and candidates. Their
mean byte cost is **.6839025616645813** in both layouts. This agreement is a
layout check, not a claim that every word has been reconstructed correctly.
The [raw serial phases](measurements/baseline.json) and
[layout comparison](measurements/comparison.json) preserve the measurements.

The [item 9 learning receipt](../2026-09-26-item9/README.md) keeps quality
gates separate. In particular, the repeated parsed-text curriculum fails
on five of 56 held-out wordings after its unchanged 9,000 updates. That slow
learning assertion remains red and is not part of a passing default sweep.

Earlier packed-answer quality numbers, including the September 16 batch-2
supplied-answer runs, are invalid: those saved programs contained zero roots
instead of the completed sentence state. The warning remains in
[Testing](../../Testing.md); it does not invalidate unrelated timing results.

The older compose-gradient/cache probe remains assigned to item 1. Its
nonzero-gradient assertion failed before the cache check on both the committed
and patched parity runtime; this work does not claim that cache check passed.
