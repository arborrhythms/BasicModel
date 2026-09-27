# Item 7.5 accepted landing

The September 27 round-3 review and Alec's acceptance authorize the two fixture
patches drafted in the [pressure receipt](../2026-09-27-item7-5-pressure/README.md).
An externally injected full STM now expects the admission assertion; a
unary-preferring user sentence now expects a completed sentence and one LTM row.
No production source, configuration, gate, threshold or seed changes here.

## Validation

Both complete fixture files finish **65 cases: 60 passed, five skipped**, exit
0, in 42.53 seconds. [Fixture summary](fixtures-summary.json).

The single source-matched full sweep completes **4,989 cases exactly once:
4,666 passed, 321 skipped, one failed and one expected failure**, exit 1,
in **5,674.50 seconds (94.57 minutes)**. It uses the existing three-worker,
8 GiB per worker / 24 GiB aggregate limits, with 1,800 seconds per worker and
10,800 seconds overall. Peak worker / aggregate memory is **6.16 / 14.19 GiB**;
there is no retry, continuation or resource-limit stop. No million-sentence
training campaign is part of this landing.

The sole failure is the unchanged
`TestDepth3RelativeEndState::test_first_trained_read_reaches_depth3_end_state`:
the observed depths remain **[1, 1, 1, 1]**, with no depth-three end-state.
It is recorded red under the accepted disposition; the assertion stays intact.
Both patched fixtures pass in the full sweep as well as their file rerun.
The observation, conditioner and shared-operator report regressions also pass.
[Full result](full/result.json.gz), [validation summary](validation-summary.json).

The [selection delta](selection-delta.json) contains only the approved user-truth
fixture rename and the added link check for this receipt. The frozen 671-file
source fingerprint is
`a98b0f4e98b6959c3c60f01567c87dd419a09842c047f6e304a752d230171bf7`.
It differs from the measured round-3 fingerprint
`76d50ec49eb0f978c36c132a6c0790583a9f46b4aa390efbbf2dc05eeedd8d16`
only in `test_compiled_word_chunk.py` and `test_ltm_consolidation.py`, exactly
as the two reviewed patches specify.
[Exact approved patch](changes-since-pressure-review.patch),
[source manifest](source-manifest.json), [source archive](landing-source.tar.gz).

## Measurements and preserved gates

The [round-3 measurements](../2026-09-27-item7-5-pressure/reconstruction-comparison.json)
are carried forward on identical production/configuration source; they are not
rerun or represented as measurements of the two edited test files.
[Source-bound carry-forward record](carried-measurements.json).

| Measurement | Result |
| --- | --- |
| Serial reconstruction before training | .11755186505615711 |
| Serial reconstruction during training | .10999541729688644 |
| Serial reconstruction after training | .09888161532580853 |
| Packed and single byte reconstruction | .18002260848879814, exact parity |
| Warm-window throughput, including epoch tails | 1.5262840033 sentences/s |

All packed/single sentence records and initial fingerprints match; every
truncation flag is false. The unchanged timing protocol has two warm-up and five
measured training batches. Its first training batch took 551.47 seconds; the
previous receipt's measured window included late capture and this one did not,
so the throughput difference is not a clean architecture speed comparison.
Per-batch timing components and row winners remain in the pressure receipt.

The preceding explicit selection still records **five passes and both unchanged
XOR CLI capacity failures**. MM passes that selection; its historical **.21757
after 900 epochs against a <.20 bar** remains a failure, preserved in the
[seed audit](../2026-09-21-item10/README.md#validation-and-limits).
No learning gate, seed or threshold is tuned or waived. The operator report's
tiny reconstruction score-path residual is retained as measured; it is not
evidence of balanced objectives or learned utility.

## Accepted scope

The accepted default training temperature of zero and a possible sentence work
term are deferred to later specifications, as stated in the
[acceptance](../../specs/2026-09-26-one-operation-per-round.md#accepted-alec-2026-09-27--with-two-deferred-decisions).
Restating the shared-operator diagnostic assertion around actual seal
contributions and renaming the native context-read `exploration_trial` flag are
also follow-up work. This landing applies only the two accepted fixture patches
on top of the reviewed implementation.

## Committed source verification

Implementation commit `69067272d823096a101b543ae0af714d12acc67e` contains exactly
all 671 files in the frozen source manifest, with every blob hash matching the
full sweep. The [verification record](committed-source-verification.json) and
[verification driver](verify_commit.py) bind the commit to this receipt.
The implementation is accepted with the depth-three campaign red; the separate
follow-up record contains publication metadata and the reconciled task list.
