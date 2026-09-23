# Item 11 review corrections: paired conceptual evidence

This landing implements the September 23 amendments to
[Architecture](../../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)
and [two truths §1.1](../../specs/2026-09-16-two-truths-ideas-and-relations.md#11-both-is-a-compositional-fact-decided-alec-2026-09-23).
The WholeSpace redesign and its learned XOR gate remain item 11a. The
two-truths seal remains item 7; its scalar trust collapse is not a carrier
for the new conceptual evidence.

## Representation and computation

The live field is `[S, batch, occurrence, 2]`, where S is the sum of the
taper caps. Its last axis stores independent positive and negative evidence.
No evidence is `(0, 0)`; heterogeneous evidence can be `(1, 1)`. The four
corners are derived only when read. The dictionary stores one code per
concept; the symbol leg interleaves its two activated symbols. Occurrences
stay separate through every rung; only the final symbol read unions them.
This supplies the scope needed by item 11a without claiming that the current
towers have learned its OR and AND features.

The snap reads normalized projections in hypercube-diagonal units and
admits each pole with `relu(±p − τ)/(1 − τ)`. `conceptEvidenceFloor` in
`model.xml` is 0.005. A kind's positive channel is the log-complement union
and its negative channel the log conjunction. A whole uses the dual pair.
Nonnegative exponents weight both channels; a negated part addresses the
other source column. Empty and all-zero definitions assert neither. The
taper ranks the stronger pole and admits both together. Exact Boolean rails
are retained with finite log floors for differentiation.

The transpose distributes each pole in its own chart. Thought effects use
that transpose, including negated literals, in a separate occurrence. The
serial context reader receives each pole separately. Checkpoints retain the
paired field and its code owner; a restored field can produce symbols even
without a new percept event. Located symbol inputs preserve their position
band on both symbol rows.

## Discovery and checkpoint ownership

Context is a dense CPU `S × S` tensor, allocated only when the parallel pool
is used. A 1M dictionary with the production eight-slot taper still uses a
15 × 15 context tensor. Matching uses matrix–vector products. No host
promotion-candidate dictionary remains.

Alternative support is normalized by its row maximum after observation,
projected optimizer updates and discovery. Discovery clears `managed` and
fixes participation at one, so idle discovered symbols stay available.
Exponents are projected nonnegative after model optimizer updates; polarity
belongs to the source column.

The structural sidecar owns part values, context and pool metadata exactly
once. The module registration remains for optimizer identity, but contributes
no duplicate part values to `state_dict`; loading restores topology once.
Old signed part checkpoints migrate polarity into source columns and reverse
the corresponding Adam first moments. Second moments are unchanged.

## XOR disposition

The earlier zero-initialized, identically connected conjunctions received
zero conjunctive gradient for all 900 balanced XOR updates. That symmetry
was the cause of the null. No selected seed or tie-break is added. The
four- and eight-conjunction learning assertions retain their MSE < .1 bar
and are strict expected failures, as decided in review. Empty definitions
now assert neither and receive no invented gradient direction. Passing
composition checks do not count as learning.

The next learning gate is item 11a's parallel path: WholeSpace's grounded
`A ∨ B`, PartSpace's grounded `A ∧ B`, and the concept's
`(A ∨ B) ∧ ¬(A ∧ B)`, read at one occurrence/extent with roles or positions
inside it. Observed zero is counterevidence; an unobserved input remains
unknown. The raw byte positions of A and B need not coincide.

## Measurement protocol

[run_measurements.py](run_measurements.py) runs bounded CPU processes with
one Torch thread, an 8 GiB footprint cap and a 600-second deadline. Seed 42
fixes measurements only. [evidence.py](evidence.py) trains the real parallel
`MM_sparse_concept.xml` model for 16 native updates at batch four, then
measures its validation field. Sixty-four feature permutations calibrate an
unrelated-alignment control; a different 64 check it. They preserve feature
marginals while destroying dictionary alignment. They are synthetic
controls, not independently labeled semantic absences.

The floor is the calibration maximum rounded upward to .001. A separate
run trains with the configured floor and measures both that floor and the
new empirical candidate. Controls are unioned at 8, 64 and 256 slots;
64 and 256 are accumulation stress cases, not larger live batches. This
is a calibration for this bounded smoke workload, not a universal noise
bound for arbitrary dictionaries, training stages or slot counts.

The serial `d4dc385` reconstruction measurement is repeated with pi off
and on. It does not execute the parallel pyramid. Packed/single parity
remains item 9's first landing.

## Numerical results

[The production-snap regression](background.py) reproduces the failure on
`868fba2`: with 2, 4 and 8 inactive parts, the old kind reads **.75, .9375
and .99609375**. Its zero-presence assertion exits 1
([record](background-red.json), [process](background-red-process.json)).
The paired implementation reads **0 in all three cases**
([record](background-green.json), [process](background-green-process.json)).
The tests also retain both from opposite supporting slots, distinguish
unobserved inputs from observed zero, and preserve a foreign occurrence's
independence in composition and reverse.

On the trained model with admission disabled, the calibration control's
largest absolute projection is **.0042405231**, giving **τ = .005** after
upward rounding; the independent control maximum is **.0041762721**.
With τ = 0 its strongest positive control symbol rises from **.0109323**
at eight slots to **.0793480** at 64 and **.2516721** at 256. With τ = .005,
both poles are exactly zero in every measured control scope. The strongest
live projection is **.01314144**, so admission retains some live evidence.

Training a fresh model with the configured .005 floor gives an independent
control maximum **.0025126822**. Its empirical candidate is .003; the
configured .005 floor rejects all these controls too. The live paired field
is **15 × 4 × 8 × 2**, its symbol leg **4 × 30 × 1024**. Eight positive
symbol readings remain nonzero across the batch, the strongest **.00885842**;
all negative readings are zero. **None exceeds the .5 discovery use floor.**
The empty higher rungs remain silent; pi off/on are identical here because
this batch's definitions contain no conjunctive edges. The separate
composition/reverse tests exercise those edges. These numbers are a live
measurement, not successful concept discovery or learned XOR.

The last of the 16 native updates records reconstruction **.0001285936**
and answer loss **.1721265**; validation answer loss is **.1747819**. Native
validation reports reconstruction zero because that cost is not enabled
there; it is not evidence of perfect reconstruction. Each calibration run
uses at most **3.777 GiB**. The full distributions, live switch readings,
source maps and process receipts are in the
[calibration record](final-source/calibration.json),
[configured-floor record](final-source/evidence.json) and
[manifest](final-source/manifest.json).

| Serial reconstruction mean | `d4dc385` | Pi off | Pi on |
| --- | ---: | ---: | ---: |
| Before training | .10059325583279133 | .10059325583279133 | .10059325583279133 |
| Five timed training batches | .09279306977987290 | .09279306977987290 | .09279306977987290 |
| After training | .09232672862708569 | .09232672862708569 | .09232672862708569 |

The differences are exactly zero. This preserves the serial baseline while
the live parallel measurement above covers the changed snap and pyramid.

## Validation

An initial affected selection completed 187/187 cases
([receipt](affected-result.json.gz)). The first full selection exposed the
definition-sparsity penalty's old single-bias assumption: it counted the
positive bias as a part and exempted a real part from shrinkage. It was
interrupted at 1,352/4,743 cases for correction
([receipt](bias-interrupted-result.json.gz)). The penalty now excludes both
bias poles and ranks sparse edges by row, avoiding a dense inventory-sized
square. Its negated-part test uses explicit source polarity.

The first explicit slow selection exceeded its 8 GiB aggregate cap with
two workers accumulating model fixtures
([receipt](live-memory-interrupted-result.json.gz)). Reducing it to one
case per fresh worker and one worker retains the cap and every assertion.
That run was interrupted at 23/37 when the full suite's bias defect required
a source change ([receipt](live-source-interrupted-result.json.gz)). Neither
interrupted run is counted as final validation.

The corrected [affected receipt](affected-final-result.json.gz) completes
**247/247** cases: **219 passed, 28 skipped**, exit 0. The explicit
[slow receipt](live-final-result.json.gz) completes **37/37**:
**35 passed, two strict expected failures**, exit 0, with a largest worker
footprint of **3.325 GiB**. [Both unseeded XOR widths](xor-summary.json) report MSE **.5** after
900 updates; the original **< .1** bar is unchanged.

The [serial comparison](final-source/comparison.json), both trained parallel
measurements and these receipts match the same **633 source files**, whose
sorted compact source map has SHA-256
`e6adf6efca88de6a508d0cd2edf5040be3acd4ef9f83c023537be8f76eb7138b`.
The final [full receipt](full-result.json.gz) completes **4,743/4,743**
cases: **4,410 passed, 332 skipped and one existing expected failure**,
exit **0**, in **1,061.76 seconds**. There is no waived failure or
compiler-cache retry. Peak aggregate footprint is **17.003 GiB** under the
20 GiB cap, with 8 GiB per worker. The
[source manifest](full-source-manifest.json) and [summary](full-summary.json)
retain the same source map. The default slow skips do not count as learning
passes; the two explicit XOR nulls are recorded in the separate slow receipt.

The final [documentation-link receipt](doc-links-result.json.gz) completes
**72/72** cases, all passed, exit 0, after the implementation and receipt
documentation updates.
