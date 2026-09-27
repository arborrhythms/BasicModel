# Item 7.5: reduction pressure and gradient-report corrections

Runtime feedback implemented; stopped for Claude review, uncommitted.
This implements review round 2 and decision 7
in the [spec](../../specs/2026-09-26-one-operation-per-round.md), following the
[sentence-seal receipt](../2026-09-27-item7-5-seals/README.md). That receipt and
its ten full-sweep failures remain preserved as the reviewed baseline.

All nine failures addressed by this feedback now pass. The single full sweep
completes **4,988 cases exactly once: 4,663 passed, 321 skipped, three failed
and one expected failure**, exit 1. The failures are two newly exposed obsolete
fixtures and the unchanged depth-three campaign. The fixture-only follow-ups
below are drafted for review and remain unapplied. Nothing is committed or pushed.

Declared before measurement: `reducePressure=1.0`; every binary logit receives
`reducePressure * (d/a + max(0,d-a)/r)`, zero on an empty stack. The whole STM
depth supplies `d`, even though an operation uses a two-slot window. `r`
includes the current round. Online allowance is `K-1`; seal allowance is one
for an absolute sentence and three for a relative sentence. A pre-push depth
of `K-1` becomes `K` when the word arrives; its final online round must reduce.
Unary and STOP candidates are masked when required reductions exhaust the
remaining rounds. Overflow is an assertion, not an incomplete memory record.

The pressure is a fixed part of model logits and credit; sampling temperature
still affects only the hard draw. This form and default were not selected
against the reconstruction results or learning gates. No million-sentence
training ran. The corrected source receives affected checks, a source-matched
full sweep and a review stop before commit.

The report correction captures each weighted objective's gradients before the
exploit and explore seal graphs are released. It sums detached gradient vectors
by their actual shared parameter, including the cached perception pullback,
then adds the batch-end output/expectation gradients before computing norms and
cosines. This describes the actual training directions across parameter versions;
it is not a single gradient evaluated at the batch-end weights. Sparse gradients
remain sparse, and observation does not write optimizer gradient buffers.

The four evaluation interleave fixtures now expect the native context call where
needed and one serial exploit. The deliberately infeasible one-round compose
fixture still reports incomplete, but records the binary reduction compelled by
its deadline. Observation, gradient-contract, depth-three and XOR/MM assertions
are unchanged from the reviewed source.

New mechanism checks cover the last online round through the real two-slot
window, overflow assertion, completion despite unary preference, packed LTM
recording, pressure monotonicity and credit, one/three-slot allowances, early
STOP in a small relative STM, and compiled forward/backward. Gradient checks
cover vector cancellation across updates, sparse accumulation, and the exact
read-only perception pullback. Development red/green attempts are retained.

The short measurement driver and timing profiler are copied unchanged from the
sentence-seal receipt. The seed, warm-up, measured batches, validation batches,
parity corpus and resource limits are unchanged. Each measurement runs alone.
The full sweep orders expensive files using prior measured timings and uses
the same three-worker, 8 GiB per-worker / 24 GiB aggregate limits, with
30 minutes per worker and three hours overall. It preserves every selected case.

The feedback regression selection completes **35 cases: 34 passed, one failed**.
All three unchanged observation tests, both output-gradient modes, the unchanged
operator-gradient test, all 16 interleave tests, nine sentence-seal tests and
three compose-pair tests pass. The depth-three campaign remains red with
`[1, 1, 1, 1]`; its assertion is preserved. This selection takes 1,045.52 seconds.

The [operator-gradient sample](operator-gradient-samples.json) reports nonzero
reconstruction/output overlap on `operator.CS.surface`: reconstruction norm
`1.9233796976e-13`, output norm `1.7302997847`, cosine `-0.0550582804`, and
output/reconstruction norm ratio `8.9961425028e12`. Restoring the measurement
does not establish balanced objectives or useful learning. These values are
recorded without changing weights, seeds or thresholds.

The explicit selection completes **seven cases: five passed, two failed** in
235.47 seconds. Full-graph runtime lengths, packed operand provenance, packed
reconstruction and two-epoch graph release pass. Both unchanged XOR_grammar CLI
gates still exhaust the fixed WholeSpace codebook at row six during reset/autobind.
MM passes this run; its historical **.21757** failure after 900 epochs against
the **< .20** threshold remains visible in the
[seed audit](../2026-09-21-item10/README.md#validation-and-limits). No gate,
threshold or seed was changed, and there was no million-sentence training.

Serial reconstruction and timing on the unchanged seed-42 CPU protocol:

| Measurement | Reviewed sentence seals | Reduction pressure |
| --- | ---: | ---: |
| Reconstruction before training | .1065397672355175 | .11755186505615711 |
| Reconstruction during training | .1001331090927124 | .10999541729688644 |
| Reconstruction after training | .10013789683580399 | .09888161532580853 |
| Measured sentences/s, including epoch tails | .9189261372761699 | 1.5262840033446774 |
| Mean measured batch call, seconds | 2.1674882418 | 1.3014878498 |
| First training batch, including epoch tail, seconds | 555.77970 | 551.46904 |
| First measured batch, including epoch tail, seconds | 5.65698 | 1.23495 |

The fixed window has two warm-up and five measured training batches, plus four
validation batches before and after training. The prior first measured batch
included late graph capture; the current one does not. Thus the throughput
increase is not a clean steady-state architecture speed comparison. Neither
window was trimmed or given extra warm-up.

Mean measured component times, in seconds, are exploit/explore compose
**.231542517 / .234326183**, their backwards **.274909783 / .270737167**,
batch backward **.000327075**, sentence scoring **.111398617**, snapshot
**.000000692**, restore **.000273617**, and other work **.177972200**.
Explore wins **4/14** training row-sentences, including **1/10** in the measured
window. Both candidate costs and all winners are retained. The complete serial
process takes **739.71 seconds**, peaking at **5.97 GiB** under the unchanged
8 GiB / 1,200-second guard.

Packed and single byte reconstruction are exactly **.18002260848879814**, versus
**.6496902331709862** in the reviewed receipt. Initial parameter and dictionary
fingerprints and all four complete per-sentence records match exactly. Both
layouts now report **false** for every truncation flag; the prior measurement
reported true throughout. These are short reconstruction measurements, not
evidence of learned language utility. Packed/single take **286.27 / 479.28
seconds**, with **4.98 / 4.98 GiB** peak process memory under the same guard.

All corrected regression checks, explicit checks and measurements match the
671-file source fingerprint
`76d50ec49eb0f978c36c132a6c0790583a9f46b4aa390efbbf2dc05eeedd8d16`.
The full sweep completes **4,988 cases** on that source in **5,631.21 seconds
(93.85 minutes)**. Peak worker memory is **6.08 GiB** and peak aggregate memory
is **13.89 GiB**, within the unchanged 8 / 24 GiB limits. There is no cache
retry, continuation or cap stop. All 18 new pressure and gradient mechanism
cases pass in the full sweep. The earlier 54-case combined mechanism receipt
also passes, with only the subsequent root-README change in its source delta.

The full sweep exposed two additional fixtures that still assert the
superseded behavior. `test_tensor_peer_word_loop_rejects_external_full_stm`
expects `_compose_overflow` and an incomplete forest after injecting a full
stack; decision 7 now requires the assertion that fires instead.
`test_incomplete_user_truth_does_not_write_an_ltm_row` forces unary preference
and expects no memory row; the deadline now completes that sentence. These
failures remain visible alongside the unchanged depth-three campaign.

The [overflow fixture follow-up](overflow-fixture-followup.patch) and
[user-truth fixture follow-up](user-truth-fixture-followup.patch) make the
proposed expectation changes concrete for review. **They are not applied or
reported as passing.** The measured and swept source remains unchanged for
the requested single source-matched sweep and review stop.

Review artifacts:

- [Patch since the sentence-seal review](changes-since-seal-review.patch),
  [source manifest](source-manifest.json), [source archive](review-source.tar.gz)
  and [exact reviewed spec](reviewed-spec.md).
- [Feedback regressions](feedback-regressions/result.json.gz),
  [explicit checks](explicit-verified/result.json.gz) and
  [protected contracts](unchanged-contracts.json).
- [Reconstruction comparison](reconstruction-comparison.json),
  [batch timing and row winners](measurements/batch-timing.json) and
  [measurement resource accounting](measurements/processes.json).
- [Validation summary](validation-summary.json) and
  [full-sweep result](full/result.json.gz), with its
  [raw receipt](full/raw-receipt.tar.gz) and
  [supervisor log](full-supervisor.log). Every development attempt retains
  its raw receipt archive and outcomes.
