# Item 11a: learned primitive properties and grounded XOR

The implementation follows the [property design](../../Architecture.md#item-11a-primitive-properties-and-grounded-extents-september-23)
and the [item 11a brief](spec.md).
WholeSpace owns learned memberships over byte primitives. Its runtime reads
and acquired predicates use those parameters; former tag definitions enter
only through old-checkpoint migration. Constant-signature cuts keep the
ordered input witness. Parallel symbolic cutover maps each independently
coded tower into the conceptual chart and folds its evidence within the same
input extent. Negation exists only for concepts, not percepts (or symbols).

The conceptual snap uses unit-ball units, retains position pairs, and reads
both poles independently. Co-presence supplies positive parts. Zero candidate
edges may subsequently learn a negative part; witnessing and priming cannot
make an unwritten negative nonzero. Candidate growth preserves frozen rows.
Property-row priming uses direct surface references. Checkpoints retain the
learned properties, mapped reads and located evidence with one owner each.

## Learning protocol

The native parallel model receives ASCII byte pairs `00`, `01`, `10`, `11`
in one two-position extent. An initially unnamed property receives examples
of the two byte memberships. The independent tower maps and codebooks receive
900 supervised updates naming those primitive observations at their own
positions. Runtime OR and AND come from the native tower extent folds;
their truth tables are never installed as conceptual activations.

Four positive initial conjunction definitions then receive 600 paired XOR
updates: the observed OR, the observed AND, their co-presence, and standing
presence. Their prospective negative edges start at zero. The optimizer
learns the negated AND part; participation remains an EWMA of use. Every
conjunction must classify all four examples correctly and the combined
paired MSE must be below .1. There are three unseeded runs at each pool size
4 and 8. No seed is set or selected. This is supervised primitive naming
and concept learning; it does not establish unsupervised concept discovery.

This replaces the old strict-expected-failure pyramid learning assertions.
Their zero-definition null remains in the [item 11 record](../2026-09-23-item11/README.md#xor-disposition),
and the empty-definition/no-invented-gradient invariant remains tested.

## Calibration and reconstruction protocol

The [evidence probe](evidence.py) trains a live parallel model for 16 native
updates at batch four. Calibration uses 64 feature permutations with random
generator 1042; independent evaluation uses 64 with generator 2042. The
candidate floor is 1.25 times the calibration maximum, rounded upward to
.001. A canonical meronomy model and the existing mixed smoke config are
measured, with admission disabled and then with the selected .112 floor.
Control scopes of 8, 64 and 256 slots expose accumulation. These are synthetic
alignment controls, not labeled semantic absences or a universal bound.

The serial `d4dc385` reconstruction workload is repeated with conceptual pi
off and on. This serial control does not exercise the new parallel cutover.
Packed/single parity remains item 9's first landing. The bounded full-suite,
slow-path and measurement receipts are recorded below.

## Prior reproduction and scope

The prior check compares all **8 × 256** learned memberships with the old
teaching ranges: maximum error **0**, with **0** differing byte signatures.
The ladder idiom fixture's **32** sentences produce the same **160**
constant-signature runs. This covers the four requested properties and the
four supplied refinements. Boundary-policy learning and the acquired-property
checkpoint round trip are also covered by the affected tests.

The parallel symbol index audit covers `ConceptEvidence.decode`,
`SymbolSpace.forward_concept_to_symbol`, the conceptual context reader and
thought effects. Parallel leg row `j` addresses dictionary row `j // 2`;
its pole is `j % 2`. Serial reference rows already address concepts.
Located evidence travels through symbol readout and checkpoints; inferred
thought extents have sentinel brackets and no invented input positions.

The normalization lifetime was open in the todo. For review, the implemented
post-optimizer maximum normalization applies to assigned provisional
**disjunctive** rows; discovered and seal-written definitions retain scale.
Witness assignment continues to normalize observed support. The frozen-row
barrier and no-gradient participation law are tested separately.

## Measurements

The final source records are in [the measurement manifest](validated-source/manifest.json).
The prior/segmentation comparison is in [priors.json](validated-source/priors.json).
All measurements use one Torch thread, CPU/eager execution, an 8 GiB
process cap and a 600-second deadline. Seed 42 makes these measurements
reproducible; the XOR assertions never set a seed.
Raw logs and generated XML inputs are retained byte-for-byte, including
their original whitespace, to preserve the recorded configuration hashes.

| Reconstruction phase | `d4dc385` | Learned memberships, pi off/on | Memberships frozen for diagnosis |
| --- | ---: | ---: | ---: |
| Before training | .1005932558 | .1005932558 | .1005932558 |
| Five measured training batches | .0927930698 | .0949083805 | .0927930698 |
| After training | .0923267286 | .0924575571 | .0923267286 |

Pi off and on give identical serial measurements. The initial reader error
raised the after-training cost to .1007210538: the serial reference bank
indexed byte counts as property rows and consequently supplied zero property
evidence for ordinary words. The corrected bank evaluates the live primitive
memberships before gathering by property id. Before training, it now matches
the baseline exactly. Afterwards the increase is **.0001308285 (0.142%)**;
the training-window increase is **2.28%**. Freezing only the new memberships
restores all three old means exactly. This isolates the residual difference
to learning those definitions, not to a broken serial reference or reverse.
The frozen run is a [measurement-only ablation](reconstruction_ablation.py),
not another runtime mode. See the [comparison](validated-source/comparison.json)
and [ablation record](validated-source/frozen.json).

| Parallel workload | Calibration maximum | Independent maximum | Largest live projection |
| --- | ---: | ---: | ---: |
| Canonical, admission disabled | .0889045 | .1242888 | .7506586 |
| Canonical, trained with .112 floor | .2712737 | .2889889 | .6487308 |
| Mixed control, admission disabled | .0881007 | .0764134 | .1221441 |
| Mixed control, trained with .112 floor | .0881007 | .0764134 | .1221441 |

The two admission-disabled calibration candidates are .112 and .111;
`model.xml` uses **.112**. The fresh independent control slightly exceeds the
canonical calibration range. Training with the floor also changes the
canonical representation, so the calibration is not a stationary noise
bound. At the live eight-slot scope its strongest unrelated positive /
negative readings are **.19931 / .13467**, below the .5 discovery gate.
At 64 slots they become **.35890 / .25121**, and at the 256-slot stress scope
**.77270 / .62627**: long unions can invent strong support from weak admitted
alignments. The mixed control is zero at every measured scope. These limits
remain visible for the fold evaluation; increasing the number of unioned
positions requires a new calibration.

The canonical configured run retains **36** symbol-pole readings above the
use floor, unlike the old incorrectly scaled snap. This is a live magnitude
measurement, not labeled evidence of correct concept discovery. The separate
unseeded native XOR gate supplies the supervised learning evidence.


## Validation record

The [affected receipt](affected-result.json.gz) completes **187/187** cases:
**178 passed, 9 skipped**, exit 0. The [explicit slow receipt](slow-result.json.gz)
completes **12/12**, all passed with `RUN_SLOW=1`. It includes boundary
learning, acquired-property checkpoint restoration, priming and the native
symbolic reference transaction. The six unseeded native XOR runs in the
affected receipt have paired MSE between **.00002098 and .00002674**; each
of their four conjunctions classifies all four inputs correctly. Their
[outputs, learned parts and gates](xor-affected.json) are retained.
The six fresh runs in the larger-batch full attempt also pass, with paired
MSE from **.00002318 to .00003320** ([individual results](xor-memory-full.json)).
The final bounded full run passes all six again, from **.00001935 to
.00002694** ([individual results](xor-full.json)).

The initial 400-update primitive-naming schedule passed five of six runs;
one missed the declared .08 tower-evidence tolerance. The fixed schedule
was increased to 900 for every run, with no seed selection. The
[initial learning receipt](initial-xor-result.json.gz) retains that failure.
The [initial prerequisite probe](initial-red-result.json.gz),
[first four-conjunction attempt](four-conjunctions-red-result.json.gz)
and [serial-reader regression](serial-reader-red-result.json.gz) retain the
other failing probes. The bias-only conjunction exposed an out-of-range
witness-strength read; standing presence now uses its own constant support.

The first full run was [interrupted at 1,760/4,766 cases](interrupted-full-result.json.gz)
to replace the remaining test import of the deleted signature lookup and
correct the serial reader. It is not a passing full receipt. Initial
reconstruction/calibration records are retained in
[before-reader-fix](before-reader-fix/manifest.json); the fixed-source records
are in [validated-source](validated-source/manifest.json). The failed serial
comparison and the zero-effect frozen-membership ablation on that broken
reader remain visible there.

The second full run completed all 4,766 cases with two failures: an assertion
still expected 64 independent occurrences instead of one extent retaining
64 positions, and a handcrafted canonical WholeSpace fixture lacked learned
primitive memberships. The [failed full receipt](full-red-result.json.gz)
records both. Updating those two tests gives **25/25** completed,
**19 passed and 6 skipped**, in the [contract receipt](contracts-result.json.gz).
No runtime or configuration changed: the [source delta](validation-source-delta.json)
identifies the two test-only changes after the affected, slow and measurement
receipts. Their source manifests remain unchanged as executed.

The third full attempt used 32-case batches. It stopped at **2,964/4,766**
when a `test_reasoning.py` worker exceeded its 8 GiB cap; its sampled peak
was 13.93 GiB. No assertion failed, but this [resource stop](memory-full-result.json.gz)
is not a passing receipt. The source was unchanged, and the final run
returns to the eight-case batches that completed the second full selection.
All attempts, including the accidentally disabled slow selection and the
invalidated collection, remain in the [attempt summary](validation-attempts.json).

The final [full receipt](full-result.json.gz) completes **4,767/4,767** cases:
**4,436 passed, 330 skipped, one existing expected failure**, exit 0, in
**1,083.18 seconds**. No failure is waived and no compiler-cache retry ran.
Ten one-thread CPU workers use eight-case, one-file batches with an 8 GiB
worker cap and 20 GiB aggregate cap; peak aggregate use is **16.99 GiB**.
The extra selected case is this landing's preserved brief, checked by the
documentation-link parametrization. Counts are unique cases; four repeated
passing phase reports remain in the raw receipt and [summary](full-summary.json).

The [full source manifest](full-source-manifest.json) matches all **636**
committed source files. SHA-256 of the sorted compact validated-source map:
`19e362842fde393680faeb47c027dbac060a6d9a38e0689c9534a13a13b6913a`.
The contract receipt has the same source map; the earlier affected, slow and
measurement receipts differ only in the two corrected test files listed
above. Runtime code and configuration are identical across those receipts.
The final [documentation-link receipt](doc-links-result.json.gz) passes
**74/74** cases. [Receipt metadata](receipt-info.json) records the landing
source, validation counts, reconstruction delta and review residue.
