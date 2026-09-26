# Item 9b — shared fields, interpretation and fixed capacities

Claude accepted this implementation for commit, with a follow-up. This receipt
preserves the reviewed source and its original validation. The follow-up changes
interleaving to parallel-first and removes label read-back; the interleave
measurements below are **void for that new schedule**. See
[the follow-up receipt](../2026-09-25-item9b-followup/README.md) for current validation.
The governing plan is [the September 25 plan, including §4e](../../plans/2026-09-25-item-9b-mode-sharing-and-interpret.md).
The [starting receipt](../2026-09-25-item9b-start/README.md) preserves the
original red probes against the published item 9 runtime.

## Contract changes

A concept is an identity and a definition, with no individual spatial or
temporal coordinate. A field carries one convex where bracket and one when
interval, alongside native percept presence and observed-complement poles.
Perception retains its events for exact inverse attribution. Parallel attention
reads every supported definition; serial attention stays focused on the word.

`interpret` owns word-to-object testimony and its lexical inverse. Its tensor
face is mandatory before serial composition. Grammar supplies particular or
kind resolution; spelling belongs to the word's lookup and inverse.

`modeSchedule` accepts `serial`, `parallel` and `interleave:N`. Interleaving
runs a native parallel pass over each group of N serial sentences, followed
by label read-back, using the same model and optimizer owners.
Read-back replaces knowing with the labels' attributed native support before
the reconstruction objective is evaluated. The measured effect therefore
includes this feedback, rather than recording it as a disconnected diagnostic.
Native admission does not synthesize a serial sentence chain at a parallel
boundary. The former implicit right-branching parse mixed concept orders;
the native pass now leaves sentence composition to serial grammar and exact
ordered units to PartSpace.

## Physical memory decision (§4e)

`nVectors` is the physical capacity. Logical admission writes existing rows;
no codebook Parameter is replaced and no optimizer state is migrated. Exhaustion
raises an error naming `nVectors`. Construction assigns stable where-space
slices to input positions, PartSpace, WholeSpace and symbols. Identity remains
the row index, never its where coordinate.

The production PartSpace remains 32,768 × 136 fp32: 17,825,792 bytes for W,
53,477,376 bytes including two Adam moments. The old dense ConceptualSpace
setting of 1,048,576 × 1,032 really would allocate 4,328,521,728 bytes for W
and 12,985,565,184 bytes including moments. Production configurations now
choose 65,536 physical rows (32,768 initially active): 270,532,608 bytes for W
and 811,597,824 bytes if trained with two moments. The production conceptual
dictionary instead uses context-owned rotation and has no Adam moments;
the latter figure is a comparison, not its actual optimizer allocation.
These are tensor-storage estimates, not process peak measurements; sparse
definitions and other owners add memory.

Checkpoint admission cannot resize a live percept codebook. Restore into the
matching declared physical capacity; increasing that capacity is a construction
choice, not a mid-run optimizer migration. Legacy nonempty per-concept location
definitions are rejected with an instruction to re-teach the exact conjunction
as a native fused percept. Empty legacy location metadata does not block a load.
Cross-mode tests use the same declared inventory and preserve its ids,
definitions, dictionary and native poles in both directions.

## Fixture migrations

The former concept-position assertions now inspect native percept events and
the field's one bracket. Exact-zero and inverse assertions remain.
The old located-conjunction review probe retains its original `10`-only truth
table by declaring that ordered unit in native PartSpace containment. Simply
removing its locations would ask the independent-pole conceptual reducer to
perform an exact ordered match, which §4c assigns to perception.

XOR now learns its exact cases from ordinary PartSpace fusion. The old fixture
explicitly disabled that fusion and therefore contradicted §4c. It now admits
recurring parts and declares its former capacity ceiling as physical nVectors.
Each native occurrence contributes one participation observation. The witness
count is derived from the configured EWMA and admission threshold, rather than
counting duplicate conceptual occurrences. No seed is changed or selected.

Physical growth tests now exercise logical admission, preserving
Parameter, optimizer ownership, existing row contents and moments. These old
resize assertions describe the API explicitly retired by §4e.
Byte-lexicon fixtures now declare enough physical rows for their initial
alphabet; the BPE fixture needs 257 (256 bytes plus its null entry). The
word-binding fixture uses a real serial operator owner because its assertions
request word/object testimony. Separate native-boundary coverage rejects both
serial interpretation and implicit sentence-chain construction in parallel.

Testimony fixtures now require the naming word as the object's sole initial
literal, replacing the old ATOM/UNIVERSE bounds. Recurrence changes the
provisional object's participation instead of strengthening a duplicate META
weight row. Ordered native word references identify a word even without a
surface cache. These assertion changes follow plan §5; they are not seed or
tolerance adjustments. The new generic-kind seal test supplies the selected
grammar's resolution and checks the actual sealed references; it does not
claim that an untrained grammar parses English generics correctly.

An intermediate broad audit found eight remaining fixture failures in three files.
The history-projection test now follows word heat into the weighted object
row, retaining the original heat bound and empty-input controls; it also
requires the structural META to have no weighted row. The META structure test
checks that missing row directly. Six decoding cases now declare a 256-row
physical reserve for their initial 128-character alphabet; their decoding
assertions are unchanged. These are the only source changes after the passing
slow, erosion and fixed-seed measurement jobs. All runtime files remain exact.
The additional slow field-publication test also expected the retired fixed
15-row taper instead of the gathered field's actual width. Its shape check
now uses the field's published partition; the exact-zero and unchanged-codebook
assertions remain. This last assertion-only change is in the same test file,
is skipped in the default full sweep, and passes its explicit slow rerun.
The receipt verifies that exact one-line difference.
The first final-fixture attempt was stopped by the 8 GiB guard after 14 of
46 cases. Those partial outcomes are not counted as completed validation.
The rerun isolates each case in a fresh worker under the same 8 GiB cap;
no assertion, seed or resource limit changes.

Field knowing and native percept events may have different stage owners.
The checkpoint saves each independently, so restoring a terminal conceptual
reading also preserves the native reader's event evidence for attribution.
One field where/when pair and per-word symbol occurrence coordinates travel
with captured programs; item 7 still owns the LTM seal's durable row columns.

The former refinement-learning fixture started by learning a missing negative
pole. Native observations now already supply that pole, making its initial
answer exact. It instead learns the particular's initially unwritten link to
that native predicate. The initial-loss, pure-field, scattered-support and
raise assertions remain, with no seed or tolerance change.

The full sweep also found a wrong gradient on written fractional feature
weights: surrogate incidence credit was added to their exponent derivative.
Incidence credit now applies only at zero (insertion); written edges use the
power derivative. Forward independent-pole min/max semantics are unchanged.
The shared-mode learning probe checks an actual reduction in native loss.

A vector output head retains its declared width by selecting the strongest
emitted concepts or symbols. The complete native field and its captured
reconstruction evidence remain intact; a head configured with concept IDs
continues to read those IDs directly.

## Erosion findings

All nine runs complete: seeds 0, 1 and 2, each with serial, parallel and
interleave:2 training from an identical per-seed starting model. Every run
sees 24 inputs once: the four XOR strings and one complete sentence from each
of the 20 launch documents. Readout uses four XOR probes and 68 distinct word
forms; FineWeb category labels are the native orthographic property sets.
This is a small architecture measurement, not semantic category learning or
a throughput benchmark. The CPU harness executes the native tensor recurrence
bodies with an eager loop dispatcher; compiled recurrence is tested separately.

The predicted ordering is a finding, not an assertion to tune into passing:

| Ordering | XOR seeds agreeing | FineWeb seeds agreeing |
|---|---:|---:|
| Parallel native CP below serial | 0/3 | 3/3 |
| Parallel native within-category distance above serial | 3/3 | 3/3 |
| Parallel admits more alternatives per concept | 0/3 | 0/3 |
| Interleaved read-back CP above parallel | 0/3 | 3/3 |
| Interleaved read-back within-category distance above serial | 3/3 | 0/3 |
| Interleaved read-back CP nearer serial than parallel is | 0/3 | 0/3 |
| Taking symbols offline lowers CP (each mode) | 0/3 | 0/3 |
| Largest CP benefit from symbols in serial | 0/3 | 0/3 |

Native XOR CP is zero throughout. FineWeb native CP falls under parallel
training (serial 3.994–5.613; parallel 1.207–1.496), but label read-back lowers
it in every condition, contradicting the predicted restoration. Structural
alternatives (positive W_sigma edges) per admitted order-0 concept are 0.184524 in serial/interleaved
and zero in parallel. The denominator includes admitted concepts without
alternatives. The probe's short training and incomplete lexical coverage are
recorded alongside each score; unseen labels have zero attributed readings,
and are not silently dropped. These results do not establish the full erosion
mechanism. They do not replace the separate native XOR learning gate.

The [per-run table](erosion-table.md), [full JSON](erosion/results.json), corpus hashes, exact inputs and configuration are packaged with
the final source-matched validation below. No seed, duration, margin or result
was selected to produce a preferred ordering.

## Validation

The [full sweep](full-summary.json) completes **4,888 cases: 4,562 passed,
325 skipped and one existing expected failure**, with no red outcomes. It
takes 974.2 seconds, peaking at 8.23 GiB aggregate across two 8 GiB workers.
The [explicit slow selection](slow-summary.json) passes **24/24** in 1,200.2
seconds; [erosion and native-boundary checks](erosion-summary.json) pass **3/3**
in 89.2 seconds. The [final affected selection](affected-summary.json), including
the explicit slow field-publication check and all final fixture corrections,
passes **46/46** in 205.4 seconds with a 3.16 GiB peak. The preceding 141-case
slow audit passed 140 and exposed the retired fixed-width assertion; that
failure and the earlier 53-case checkpoint integration are retained as
diagnostics alongside the development receipts.

The [review source](review-source.json) contains 660 files with aggregate
SHA-256 `1c8edd4342c9739de5c2634e1b9d1af63871a55c06c4bb4ebf0cb427de026d51`;
the [source archive](review-source.tar.gz) preserves their contents. The
[validation summary](validation-summary.json) records source differences:
the full sweep differs only in the one skipped shape assertion subsequently
verified by the 46-case run. The slow, erosion and measurement runs precede
the three documented test-fixture corrections; none of those fixtures is
selected by those checks. All runtime and configuration files match exactly.
The [measurement source delta](measurement-source-delta.json) lists each file.
The independent item 1 reconstruction-cache probe still fails its nonzero
compose-gradient precondition on the previously committed source; item 9b
must not claim that backward-cache behavior has been verified.

The [fixed-seed serial comparison](baseline-comparison.json) is:

| Phase | Published item 9 | Item 9b |
|---|---:|---:|
| Before training | .1006144527 | .1005906649 |
| Seven training updates, mean | .0879226878 | .0949082300 |
| After training | .0891618710 | .0882481802 |

This reports all phases, rather than claiming numerical identity: the mean
training cost is higher and the post-training measurement is lower. The
published `99207a3` receipt reports a comparison without a numerical tolerance
gate, so no new threshold is invented here. [Packed and single sentence
reconstruction](measurements/comparison.json) remain exactly **.6839025616645813**, including equality of
captured roots, programs and recovered values. All 34 changed XML configurations
pass [schema validation](xml-validation.json).
The final documentation-link check passes **91/91**; `git diff --check` is clean.

## Reproduction

Each test worker is capped at 8 GiB. The full sweep uses two workers (16 GiB
aggregate); the explicit slow selection uses one. They run concurrently with
a maximum combined cap of 24 GiB on the 32 GiB host. Source edits stop for
the duration of every bounded run. Documentation is packaged after validation.

```sh
BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py --workers 2 --memory-gib 16 --timeout 1200 --suite-timeout 7200 --batch-size 128 --max-files 8 --run-dir output/item9b-full-rerun
RUN_ITEM9B_EROSION=1 BASICMODEL_DEVICE=cpu MODEL_COMPILE=eager .venv/bin/python test/test_report.py test/test_item9b_erosion.py test/test_item9b_interpret.py::test_native_parallel_pass_and_boundary_do_not_interpret_words --workers 1 --memory-gib 8 --timeout 1200 --suite-timeout 1500 --run-dir output/item9b-erosion-rerun
.venv/bin/python doc/benchmarks/2026-09-25-item9b/run_measurements.py --out output/item9b-measurements-rerun
```

Run the latter commands after the full and slow jobs finish to retain the
aggregate bound. The packaged result files retain every selected node ID for
the affected and slow selections. `package_receipt.py` verifies each final
source map against the live tree and preserves the earlier development failures.
