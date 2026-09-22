# Item 11: concept parts, witnessed context and provisional rows

The implementation builds on local commits `26c4fa8` through `c1543b1`.
Those commits are preserved and published with this landing. The settled
[design](../../Architecture.md#decided-in-direction-a-concept-is-sigma-over-pi-alec-2026-09-21)
and [context account](../../Philosophy.md#where-is-context-particulars-universals-and-the-two-axes-2026-09-22)
govern the implementation. Item 10 and item 9 are not part of this change.

## Computation

Concepts share one row inventory. `ConceptualAttentionLayer` owns disjunctive
parts `W_sigma`, conjunctive parts `W_pi`, and sparse context weights `where`.
The default rung is the union, `-expm1(sum(abs(w) * log1p(-presence)))`.
`conceptualPi=true` first computes the conjunctions from lower orders, then
lets the union read those conjunctions at the same order and the admitted
lower-order activations. A concept can have both kinds of part. There is no
additive-hop runtime alternative or lateral competition.

Exponents added without evidence start at zero. Positive exponents read a
part's presence; negative exponents read its complement. At exactly zero,
the derivative is the symmetric directional derivative of the two signed
charts. An ordinary `abs` derivative would leave unknown edges unable to
learn. Log arguments have a machine-epsilon floor; a straight-through floor
keeps a finite slope at an endpoint, including a newly initialized
conjunction's presence of one. This numerical convention does not prove
that the learning gates pass.

The transpose normalizes each output row's exponent magnitudes and shares
its chart evidence. The union's own conjunct has an implicit exponent one.
The reverse of a unit-weight conjunction of presence .81 supplies .9 to
each of its two parts; a union of presence .84 supplies .6 to each of two
alternatives. These balanced reconstructions have tolerance `1e-6`; they do
not recover which alternative caused a union. Thought effects use this
reverse when the conceptual pyramid is active. Taxonomic queries continue
walking the same concept records.

[rungs.py](rungs.py) measures the actual layer, with eight parts at each of
four orders. One part has presence one and the others zero. With the switch
off and on, presence is **.9999998808 at every order** (the fp32 log floor),
within `2e-6` of one. Finding 10's old-hop reference is
`.7615941763 → .6420149803 → .5662699938 → .5126146674`.
The reference uses the old `tanh` formula and its zero background; the new
presence-zero background is signed -1. It is not an interchangeable signed
input experiment.

For **32** disjunctive parts of presence **.001**, all admitted at unit
exponent, the measured union is **.0315085351**. Its excess above the
strongest part is below the declared **.031** bound, `(K-1) * .001`.
The count is bounded by the admitted rows. This is a specified weak-presence
case, not a claim that every possible exponent or real activity distribution
avoids saturation.

## Discovery and persistence

`attentionPromotion` now enables a row pool, controlled by the documented
[parameters](../../Params.md). It retains no host candidate dictionaries or
significance score. Each order reserves `conceptPoolSize` provisional rows;
fixed-inventory exhaustion is reported, and discovered identities are never
recycled. The default size is one. Larger experiments can set the size
explicitly within their inventory.

A witnessed concept's leave-one-out context lives on its `where` row.
Matching contexts on different occasions assign the first pair as
disjunctive parts immediately. Matching alternatives subsequently join that
concept. Object concepts and word/object META concepts do not supply
witnessed category evidence. Object kinds remain testimony's work in item 7.
With the switch on, an unmatched co-present set assigns conjunctive parts;
a different or partial set cannot silently dilute that definition.

Participation is a detached EWMA of raw activation above `conceptUseFloor`.
It gates both the presence read by subsequent rungs and the signed evidence
read by the taper and symbols. Part weights learn by EWMA support and ordinary
gradient through that gate. Context never becomes an extra membership edge.
At the discovery threshold the same row acquires its permanent concept id
and a normalized evidence-weighted part code; weak exponents are pruned.
Below the recycling threshold, the least-used provisional row can be
reassigned. A discovered row is not recycled even if its use declines.

Both part matrices, context weights, gate values, row assignments and
observation counters survive checkpoints. Row metadata has one durable copy
in the structural sidecar; its runtime buffers are non-persistent in
PyTorch's state dictionary. The loader restores dynamic
topology before checking registered parameter shapes. Checkpoints carrying
the deleted host promotion cache fail explicitly rather than installing it
alongside rows. Sparse growth and pruning preserve surviving Adam moments;
pending gradients follow allocation identities, so recycling cannot give a
new part an old part's credit. Frozen rows retain their gradient barrier
through growth.

`conceptualize_chain`, `chain_idx`, and the JOINT concept remain for item 7.

## XOR: representation passes; learning is null

The four-conjunction truth-table check uses declared signed parts and
reconstructs XOR within `3e-6`. It verifies representation, not discovery or
learning.

The separate learning gates start all exponents at **exactly zero** and run
**900 Adam updates at .03**, once at **four** conjunction concepts and once
at **eight**, with `conceptualPi=true`. They do not set a random seed, retry
an initialization, seed a sign, or change the MSE **< .1** bar. Both record
**MSE .5**. They remain failing slow assertions, without expected-failure
markers. The zero-initialized conjunctions have identical connectivity;
no initial sign or symmetry-breaking feature is supplied. The mechanism
checks do not establish learned XOR. This null remains in the countdown for
review.

## Reconstruction baseline

[run_measurements.py](run_measurements.py) repeats `d4dc385`'s native seven
training batches and its packed/single tied reconstructions, with the switch
off and on. Seed **42** is only the established measurement seed. Compilation
and checkpoint writes are disabled as in the original probe; each workload
gets a fresh CPU process, one Torch thread, an 8 GiB cap and a 600-second
deadline. No learning assertion is made reproducible by selecting this seed.

| Native reconstruction loss | `d4dc385` | Pi off | Pi on |
| --- | ---: | ---: | ---: |
| Before training | .10059325583279133 | .10059325583279133 | .10059325583279133 |
| Five timed training batches | .09279306977987290 | .09279306977987290 | .09279306977987290 |
| After training | .09232672862708569 | .09232672862708569 | .09232672862708569 |

The means are identical, not rounded matches. Packed and single-sentence
tied-byte costs retain their means **.7866926491** and **.6838697642**;
all four complete parity payloads (two workloads under each switch setting)
match the original records.

The [numerical comparison](final-source/comparison.json),
[native off record](final-source/off-baseline.json),
[native on record](final-source/on-baseline.json) and
[manifest](final-source/manifest.json) retain all seven process receipts,
configurations and probe hashes. Every process exits zero, the largest
measured footprint is **1.423 GiB**, and the six reconstruction source maps
match the affected receipt's **631 files**. The
[actual rung record](final-source/rungs.json) supplies the depth and
weak-presence values above. Process success does not turn the parity or XOR
learning null into a pass.

The native `MM_ladder.xml` workload is serial: it does not execute the
parallel conceptual pyramid. Preservation of this baseline therefore does
not establish the reconstruction behavior of a trained sparse pyramid.
The separate rung, gradient and pool checks exercise the changed mechanism.
Packed/single parity remains item 9's first landing; it is not corrected or
reinterpreted here.

## Receipts

The [initial four mechanism regressions](mechanism-red-result.json.gz)
fail on the old source. Development checks exposed a float/bool admission
mask mismatch, pending-gradient ownership across sparse growth, frozen-row
credit after replacing a parameter, and dynamic topology restoration before
checkpoint shape checks. The
[checkpoint regression](checkpoint-red-result.json.gz) records the last
failure before its correction. The corrected
[development affected selection](development-affected-result.json.gz)
completes 84/84 cases (75 passed, 9 skipped); the later
[preflight](preflight-result.json.gz) completes 53/53 (51 passed, 2 slow
learning checks skipped). These are development source receipts, not the
final source's proof.

Two preliminary full selections were explicitly interrupted for the
pending-gradient/device corrections and the zero-default sparse edge API:
[751/4,709 cases](interrupted-first-result.json.gz) and
[756/4,714 cases](interrupted-second-result.json.gz), both exit 130. Neither
is a completed validation result. Their partial measurement attempts live
in `output/item11-measurements` and `output/item11-final-measurements`; no
partial attempt supplies the reconstruction comparison above.

A subsequent full selection stopped at **824/4,715**, exit 125, because
`test_dimensional_governance` created and removed
`data/_tmp_cs_ws_mismatch.xml` while the runner hashed source files
([receipt](source-hash-race-result.json.gz)). The
[source-integrity regression](fixture-red-result.json.gz) fails while that
scratch configuration exists. It now lives in the system temporary directory,
as does the same scratch-file pattern in `test_truth_ideas_routing`; canonical
schema/default lookup and every model assertion are unchanged. The runner's
source guard is unchanged.

The next diagnostic selection reached **2,116/4,716** and exposed two
strict-reload failures in `test_generation_catalog`
([receipt](checkpoint-inactive-red-result.json.gz)). An inactive model had
not registered the sparse module before saving; early topology restoration
then introduced eight metadata keys absent from its state dictionary.
Those buffers now persist once, through the structural sidecar. Both
inactive and pooled model reloads retain their existing assertions. This
selection was stopped explicitly for that correction (exit 130), not counted
as a passing full receipt.

A separate pruning probe then found nonzero gradient on an unrelated
frozen row after promotion replaced the shared parameter. The
[two-switch regression](prune-red-result.json.gz) records gradients
**.5128228** and approximately **.2183**, against the required zero. Promotion now
re-arms frozen-row barriers after pruning either part matrix. The concurrent
[full selection](pruning-interrupted-result.json.gz) was interrupted at
**2,835/4,716**, with no reported assertion failure; it is superseded by the
final source receipt.

The final [affected receipt](affected-result.json.gz) completes **140/140**
cases: **133 passed and 7 skipped**, exit 0. It covers both strict reload
paths, sparse edges and allocation, growth/pruning credit, pool lifecycle,
dimensional fixtures and supervised generation.

The final [explicit XOR receipt](xor-result.json.gz) completes 3/3 cases,
exit 1: one representation pass and two learning failures. Its
[source manifest](xor-source-manifest.json) and
[summary](xor-summary.json) retain the outputs and unchanged learning bar.

The final [full receipt](full-result.json.gz),
[source manifest](full-source-manifest.json) and [summary](full-summary.json)
complete **4,717/4,717** cases: **4,384 passed, 332 skipped and one existing
expected failure**, exit **0**, in **1072.14 seconds**.
There is no waived failure and no compiler-cache retry. Peak aggregate
footprint is **10.63 GiB** under the 20 GiB cap. The full, affected and
explicit XOR receipts and all six reconstruction records match the same
**631-file** validated source map, SHA-256 of its sorted compact JSON:
`b289442fecf0af783383cebe5560e6d26d22f803ff7f078eca45034f8e478596`.
[Landing metadata](receipt-info.json) records the preserved baseline commits
and the source checks. The default suite's two new slow skips are the explicit
XOR learning failures recorded separately above.

Reproduce the bounded full selection and explicit learning selection from
BasicModel with:

```sh
.venv/bin/python test/test_report.py --batch-size 8 --max-files 1 --workers 10 --memory-gib 20
RUN_SLOW=1 BASICMODEL_DEVICE=cpu .venv/bin/python test/test_report.py test/test_concept_sigma_pi.py -k xor --batch-size 1 --max-files 1 --workers 1 --timeout 180 --suite-timeout 600
.venv/bin/python doc/benchmarks/2026-09-22-item11/run_measurements.py --out output/item11-reconstruction-repeat
```

The full run uses 8 GiB per worker and a 20 GiB aggregate cap, leaving a
separate 8 GiB reservation for the concurrent measurement process and the
usual 8 GiB for the machine. The measurements use CPU fp32; the small XOR
checks also use CPU. The explicit learning command is expected to report the
recorded failures until the learning defect is resolved.

Final [documentation links](doc-links-result.json.gz) pass **71/71** with
the same validated source map.
