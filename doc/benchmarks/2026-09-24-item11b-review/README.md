# Item 11b review: located parts, pervading wholes

The September 24 correction is: **pervasion is necessary for a whole, not
a part**. An order-0 concept reads located PartSpace parts and covering
WholeSpace properties. A part can occupy a proper subspan; its ordered
constituents need not each fill the subject. The definition is sigma over
pi of percept memberships. The extent readout still unions occurrences.

This closes the review of [the first 11b landing](../2026-09-23-item11b/README.md).
Item 10's graded membership-fold evaluation has not started.

## Implementation

At the sentence boundary, a witnessed word's staged PartSpace group is
fused into one native radix row with the existing synthesis initializer.
Its canonical bytes remain in PartSpace. Membership reads the ordered
composition at its location, independently of whether the current encounter
retained one fused row or several constituent rows. Absence requires an
observed span without that part; a missing observation cannot deny it.
WholeSpace continues reading a property's membership, or its complement,
throughout the occurrence.

The word writer references that single fused part and its whole property.
Later witnesses with different parts or wholes supply alternative
conjunctions, using ordinary order-0 concept rows and the existing sigma
matrix. Repeated witnesses reuse their definition without overwriting
learned weights. The original conjunction remains intact. Location
knitting no longer appends the fused word's letters as word-level features.
Generic asserted refinements retain their existing conjunction behavior.
Meronomic refinement and pruning operate within each alternative; the
accumulated references across witnesses cannot retire the union as an
oversized conjunction.

Both symbols of each percept conjunction are products. Positive support
unions alternatives; negative support multiplies their complement products.
There is no De Morgan dual inside a percept conjunction. Distributed codes
follow these definitions at the boundary; an SBOW update to a stored
order-0 code is therefore inert as a lasting update. The existing
definition-sparsity penalty remains the only definition regularizer.

The normalization residue is closed: only assigned provisional disjunctive
rows receive maximum normalization; discovered and seal-written rows retain
their exponent scale. No new parameter, projection read or admission floor
is introduced. Object testimony, sequence concepts and the JOINT concept
remain item 7.

## Failing probes and correction

The [initial failing receipt](red-result.json.gz) records two failures for
the previously forbidden order-0 alternative connection and the smoke
workload's zero positive field after a training boundary.
The [coverage probe](pervasion-red-result.json.gz) independently fails when
a required PartSpace row occupies only half the candidate span, while the
WholeSpace property covers both positions. The former read incorrectly
requires the part to pervade the span.

The corrected tests require positive word evidence at position, extent and
symbol scope after one smoke epoch, preserve letter order and repetition
across native tilings, check containment and missing observations, and
round-trip alternative definitions through the model's structural
checkpoint. A native "love" word definition has one fused PS feature;
its letters remain inside the PS meronomy.

A [further failing probe](alternatives-red-result.json.gz) found that six alternatives triggered the old
over-collection retirement. The correction preserves their separate
conjunctions through refinement and pruning. The slow optimizer fixture
now witnesses actual native percepts instead of invented ids in an empty
inventory, and checks ownership of the feature weights as well as the
concept-to-concept weights.

An [interrupted full attempt](interrupted-full-result.json.gz) is retained
as incomplete: source changed while the fixture and lifecycle corrections
were being completed. An [affected run with eight-case batches](affected-memory-result.json.gz)
hit its 8 GiB worker cap after 184 of 186 cases; no assertion failed.
The same selection passes with two-case batches: **186 passed**, including
the explicit slow smoke and optimizer checks.

The [affected-file receipt](affected-result.json.gz) and
[source manifest](affected-source-manifest.json) record the bounded checks.
The original [workers](red-workers.log) and corrected
[workers](affected-workers.log) preserve their assertions and diagnostic
output without selecting a seed.
All [six unseeded native XOR runs](xor.json), pool sizes 4 and 8 with three
runs each, pass at four conjunction hypotheses with MSE
`2.842170943040401e-14`. Their unrelated-content controls are exactly zero
at position, extent and symbol scope.

## Measurements

The [preserved measurement driver](../2026-09-23-item11b/run_measurements.py)
replays the serial baseline with `conceptualPi` off and on, the frozen
primitive-membership ablation, and the prior/segmentation comparison.
Seed 42 fixes this measurement; it is never used to select a passing test.
The [comparison](final-source/comparison.json) is unchanged from the first
11b landing:

| Serial reconstruction mean | `d4dc385` | Current, off and on |
| --- | ---: | ---: |
| Before training | 0.1005932558 | 0.1005932558 |
| Five measured updates | 0.0927930698 | 0.0949083805 |
| After training | 0.0923267286 | 0.0924575571 |

The final cost remains 0.142% above `d4dc385`; the update mean is 2.28%
above it. Frozen primitive memberships reproduce the reference exactly.
The [prior comparison](final-source/priors.json) retains the eight prior
rows over all 256 bytes with zero error and identical segmentation on all
32 sentences, 160 runs. No reconstruction improvement is claimed here.

The [measurement manifest](final-source/manifest.json) matches the final
638-file source snapshot. Its SHA-256 over the sorted JSON source map is
`1e91b282935cb2bb88e137f3584f2f6fcd3e159534c2b2170d917ec3485011b2`.

## Full validation

The [full receipt](full-result.json.gz) completes **4,782 unique cases**:
**4,451 passed, 330 skipped, one existing expected failure**, exit 0.
The expected failure remains
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
No failure is waived. The [six fresh XOR runs](xor-full.json) pass again.
The [summary](full-summary.json) records timing, bounded memory and the
unchanged source digest; the [source manifest](full-source-manifest.json)
matches the affected and measurement receipts across all 638 source files.
The run uses ten one-thread CPU workers, eight-case, one-file batches,
and 8/20 GiB worker/aggregate caps. Documentation is recorded separately
and its links are checked again after the receipt and todo are complete.

All 77 documentation-link cases pass after completing the receipt and todo.
Source and prose are checked for whitespace errors; preserved raw logs and
generated measurement XML retain their original bytes.

[Receipt metadata](receipt-info.json) records the implementation commit.
All 638 committed source blobs match the validated snapshot byte for byte.
