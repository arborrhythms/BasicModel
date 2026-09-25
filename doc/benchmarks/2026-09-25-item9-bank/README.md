# Item 9 review follow-up: require a usable reconstruction bank

This follows the [packed/single parity review](../2026-09-25-item9-parity/README.md).
The user asked why a reconstruction request could proceed with an empty
codebook or silently use a uniform guess. This preserves the source before
the [accepted host-sync correction](../2026-09-25-item9-bank-sync/README.md).

## Diagnosis and fix

The persistent concept inventory and the temporary candidate bank are different
objects. The bank holds this batch's WORD and OBJECT concept vectors. WORD
concepts own spellings; an OBJECT reaches a spelling through its current WORD
association. The forward stem admits first-seen words before taking this
snapshot. A cold-vocabulary probe confirms admission on the first presentation,
growth of the codebook's active prefix, and reuse of the same identities on a
second staging. There is no need to warm the vocabulary to obtain candidates.

The defect was acceptance of missing staging state. The old inverse and surface
helpers manufactured an invalid one-entry bank when their inputs were absent;
missing sentence ownership disabled every candidate. A bank could also be
populated for the first sentence while a later sentence had no usable surfaces.
The fallbacks could yield a constant uniform-byte loss; some partly missing
snapshots instead failed later through unrelated shape checks. Seven boundary
fault probes were reproduced red before the fix; ordinary first-sight admission
already passed.

The direct [before/after probe](fallback_probe.py) removes only sentence
ownership after an ordinary packed forward. Both runs have **14 populated
candidates**, next concept id **19**, physical capacity **4096**, and **no
refused admissions**. The reviewed parity source continues with byte cost
**5.5451774597 = log(256)** and maximum end-state gradient **0**, reporting
truncation. The revised source raises `tied reconstruction bank lost its
sentence ownership`. This demonstrates a failed use of a populated codebook,
independently of any capacity or cold-start question. Both runs use ambient
initialization, fresh processes, an 8 GiB cap and a 600-second deadline.
[Before result](fallback-probe/before.json); [after result](fallback-probe/after.json).

The eager stem now checks the bank after admission and surface staging. It
requires concept rows/vectors, matching sentence ownership, and WORD surfaces
with validity masks. Each active sentence must have at least one usable surface
candidate. Missing candidates raise an error naming the affected row/sentence
and admission statistics. An eager inverse checks the same contract; compiled
inverse entry retains structural checks without host diagnostics inside its
loops. Missing inverse/surface snapshots and ownership no longer have a fallback.

An individual unknown can still be guessed from that sentence's available
candidates. The byte scorer retains its existing uniform background probability
alongside those candidates, preserving its uncertainty floor and the calibrated
single-candidate objective. Production reconstruction may not use absence of
candidates as a uniform-only substitute. Input targets never supply replacement
candidates, and reconstruction itself does not allocate concepts.

The existing physical `nVectors` ceiling is unchanged. Logical growth activates
preallocated rows and preserves parameter/optimizer ownership. If admission is
refused at that ceiling and a sentence has no candidates, the reconstruction
request now fails visibly instead of claiming a scoreable reconstruction.

## Validation

The [final full receipt](full-summary.json) completes **4,839 cases**:
**4,514 passed / 324 skipped / 1 existing expected failure**, with no red
outcomes. It finishes in **1,178.8 seconds** using eight workers, with an
**11.82 GiB** aggregate peak under the unchanged 20 GiB aggregate / 8 GiB
per-worker limits. There are no compiler-cache retries.

The [affected default selection](affected-summary.json) completes **193 cases**:
**125 passed / 68 skipped**. The [explicit slow selection](slow-summary.json)
passes **24/24**, including all three ambient parity cases, compiled traversal,
retained-gradient reads, empty-row handling and repeated training with live and
detached encoder credit. The per-worker cap remains 8 GiB.

The [fresh measurement](measurements/comparison.json) preserves exact equality
of every measured seal, program root, retained/recovered leaf and byte cost.
Both packed and single means remain **.6839025616645813**. The
[serial baseline](measurements/baseline.json) remains **.08916187100112438**
after the fixed seven-update workload. The measurement keeps historical seed
42; the regression tests and direct fallback probe select no seed.

The first full attempt stopped at its per-worker memory cap after **3,127/4,839**
cases, with no assertion failure. Eight independent fault fixtures retained
their model reference cycles until module teardown; the worker reached 15.05 GiB
before the monitor terminated it. The affected selection had recycled workers
near its cap, which had concealed this accumulation. The test module now releases
cycles and compiler caches between cases. Its unchanged eight assertions pass
together in one worker at **4.77 GiB**, below the unchanged 8 GiB cap.

Only that [test cleanup](test-cleanup.patch) changed after the affected/slow
receipts and measurements. The patch's original content was verified against
the earlier receipt's source hash. Production sources match exactly; the final
receipt records the sole test-file delta rather than claiming identical test
trees. The full run and [eight-case cleanup check](cleanup-summary.json) match
all **650 source files** in the [review manifest](review-source.json) and
[source archive](review-source.tar.gz), whose manifest hash is
`4054a9492f4d1210cc0ca20f2044dd8ed522c65e89c491a1198ac01f93f2010a`.
The [validation summary](validation-summary.json) retains each measured hash
and source delta. Earlier parity source archives and the interrupted full
attempt remain preserved.

The inherited slow reconstruction-cache probe remains assigned to item 1: its
compose-gradient assertion fails on both the committed and patched baseline,
before the cache assertion. This change makes no new backward-cache claim.

Documentation-link verification passes **87/87** on the same source manifest.
This receipt was recorded before Claude's review; the accepted correction above
supersedes its source for publication.
