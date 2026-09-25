# Item 9, first landing: packed/single reconstruction parity

Prepared September 25, 2026 against BasicModel `615e4cd` (WikiOracle
`e3d4c9c`). This preserves the source submitted for review. This is item 9's
separate parity landing; the expectation-learning comparisons remain open.
The [bank-contract follow-up](../2026-09-25-item9-bank/README.md) records the
subsequent review fix and its validation; the source receipts below are retained
as the original parity review state.

## Diagnosis

The [historical measurement](../2026-09-21-item10/README.md) recorded mean tied
byte cost **.7866926491 packed / .6838697642 single** at `d4dc385`. On the
current architecture, the same four presentations and recorded measurement
seed 42 give **.7984843403 / .6839025617** before this fix. Neither measurement
makes an optimizer update. Both warm the same vocabulary first, use identical
parameter and dictionary values, and include the joining space in the first
sentence of each row in both layouts.

There are two reconstruction dependencies on the surrounding pack:

1. The bounded inverse searches the entire row's WORD/OBJECT prototype bank.
   Additional occurrences in a neighboring sentence affect candidate selection
   and the soft candidate mixture, even when the sealed idea and trace are
   identical. Repeated identities in another sentence also change multiplicity.
2. Byte assignment normalizes over that entire bank, including the neighboring
   sentence's surfaces. This changes the score independently of the inverse.

The [captured-input audit](diagnosis.json), reproduced by
[diagnose_captures.py](diagnose_captures.py), holds the pre-fix recovered ideas
fixed and changes only byte-bank validity. Costs change for every sentence,
but remain above the single-sentence costs. Scoping byte scoring alone does
not repair the inverse's already-different recovered leaves.

The reported later-sentence seal difference is a separate capture defect.
The peer loop filled the live root bank at intermediate seals only; the final
seal was returned separately. Packed `AnswerProgram` capture read the empty
final bank slot, while single capture read the actual final seal. Both later
packed programs therefore held zero roots. Capturing the arguments actually
passed to `_reconstruct_sentences` establishes **exact equality of all four
sealed states and depths before the fix**. Composition divergence is not
established by the historical program-root comparison. This correction also
matters to consumers of the saved answer program, independently of byte cost.

## Inputs that differ

| Input | Difference and effect |
| --- | --- |
| Sealed slots and depths at the inverse boundary | Identical for all four presentations before the fix. Intermediate states come from the bank; final states come from the final seal. |
| Saved program end state | Packed final slots were empty. The final seal now completes the bank before program capture, including ragged rows whose last slots differ. |
| Retained leaf codes, signed activations, native rows and parameter/dictionary values | Identical after vocabulary warmup. No parameter or contextual dictionary update occurs in this comparison. |
| Grammar actions and operand occurrence relationships | Same actions after translating physical word/trace offsets to sentence positions. Intermediate seal records occupy a different trace region from final seal records. |
| Inverse prototype bank | The pack includes both sentences' WORD and OBJECT occurrences. Each entry now retains its sentence id; inverse candidates are masked to the sentence being unwound. |
| Byte candidate codes, spellings and validity | The same larger bank previously changed softmax competition. The identical sentence mask now applies to byte assignment. Masked codes are zeroed before dot products, so an excluded candidate cannot poison a gradient. |
| Byte targets | Same complete word spellings and valid bytes, including word-end scoring. These remain scoring targets, not inverse candidates. |
| Sentence ids, word indices, end positions, active masks and traversal bounds | Physical packed addresses and the number of loop iterations differ. The inverse uses sentence ids and occurrence-addressed traces to isolate state and scoring. |
| Part offsets and embedded position bands | Later sentences start at larger byte offsets in the packed input. These locate the input in the row and remain unchanged. Their differing values do not change the actual sealed states in the declared workload. |
| Property membership evidence and word/object identities | Identical per sentence. Absolute retained property spans translate with the input offsets. |
| Host reset schedule | Packed intermediate sentences use the row-local tensor reset; single sentences use the outer reset. Both use the existing flush/reset/compaction protocol. The actual seals and traces agree in this no-update workload. |

The old parity XML contains the retired `conceptualWidth` element and fails
current schema validation. [parity.xml](parity.xml) is derived from current
`MM_ladder.xml` with only word capacity/bucket 8 → 16 and tied reconstruction
enabled. The initial schema failure and both the raw and actual-boundary
captures are preserved under [diagnostics](diagnostics/). The compatibility
baseline is the current `99207a3` implementation (serial after-training
**.0891618710**); `d4dc385` remains the historical gap record.

## Validation

The declared parity tolerance is **atol 1e-6, rtol 1e-5**, applied to every
sentence's sealed state, retained leaves, reconstructed leaves and byte cost.
Saved program roots must also equal the actual seal. The regression uses
ambient model initialization, repeated words and ragged packs. No passing
seed is selected. The initial harness missed its already-cached reconstruction
call; after fixing that harness, the unmodified runtime fails on recovered
leaves (maximum difference **.0620452**). The first implementation passes it.

The initial affected selection stopped at its 16 GiB aggregate cap. Fewer
workers retain the same aggregate/per-worker limits. The next selection found
an inherited binary operand-provenance fixture failure, reproduced on an
isolated archive of committed `615e4cd`: ordinary unary rewrites erased every
leaf identity before the seal. The fixture now explicitly retains leaves until
binary consumption. It keeps all original assertions, including a mixed
leaf/composite intermediate pair, and makes no learned-routing claim. This
fixture correction does not change production grammar behavior. That affected
attempt was interrupted before changing the fixture; it is diagnostic only.
That binary provenance probe no longer exercises a unary rewrite followed by a
binary fold, because it now explicitly preserves the leaf identities it checks.

A later eight-case worker exceeded its unchanged 8 GiB cap. The interrupted
slow selection and the remaining checks run individually together cover all
101 selected cases. After correcting the second trace fixture, **100 pass and
one fails**; [slow-audit.json](slow-audit.json) retains every outcome. The
word-store trace fixture now demands a fold, because an unaligned router can
legitimately COPY without committing a grammar transform. Its original
assertions remain intact. Both fixture failures were reproduced on the
committed runtime before correction; neither correction selects a seed.

The unresolved explicit slow failure is
`test_separate_reconstruction_does_not_retrace_backward_every_step`: its
assertion that some compose-operator gradient is nonzero fails on both this
patch and the committed runtime. The input-state gradient is finite and
nonzero; the assertion occurs before the final backward-cache check, so this
run does **not** establish that cache property. No seed retry, assertion
weakening, or expected-failure waiver is applied. The separate retained-buffer
gradient check passes. Baseline compiler diagnostics used archived production
sources from `615e4cd`, with only the unrelated word-store trace fixture
experiment present in that archive; their source manifest is preserved.

The broad slow audit predates the final word-store fixture edit. Its manifests
and [source deltas](slow-audit.json) identify that sole test-file difference;
all production sources match the review tree. The corrected word-store check,
default affected selection, final measurements and full default receipt use
the exact final source snapshot.

The final [measurement](measurements/comparison.json) demonstrates parity.
Every sealed state, saved program root, retained leaf, recovered leaf and byte
cost is **exactly equal**, stronger than the declared tolerance. Both layouts
have identical effective candidate occurrences and grammar actions.

| Sentence | Packed byte cost | Single byte cost |
| --- | ---: | ---: |
| 9 plus 1 | 0.6138069630 | 0.6138069630 |
| 14 plus 1 | 0.7259159684 | 0.7259159684 |
| 2 plus 1 | 0.6725914478 | 0.6725914478 |
| 3 plus 1 | 0.7232958674 | 0.7232958674 |
| Mean | **0.6839025617** | **0.6839025617** |

The native serial baseline after its fixed seven-update workload remains
**.08916187100112438**, identical to the accepted `99207a3` implementation.
[Native baseline](measurements/baseline.json); [source and process manifest](measurements/manifest.json).
This metric uses `MM_ladder.xml`'s existing objective; the parity table explicitly
enables tied byte reconstruction. Neither is an expectation-learning result.

A development comparison also compared flattened program roots with 3-by-D
boundary roots; the final probe records both in the same shape and captures
program roots independently. Its corrected comparison above passes every
ownership check. Earlier attempts remain diagnostic records.

The final [full default receipt](full-summary.json) completes **4,829 cases**:
**4,504 passed, 324 skipped, 1 existing expected failure**, with no unexpected
failures. It took 1,098.7 seconds with ten workers, a 20 GiB aggregate cap and
an unchanged 8 GiB per-worker cap; peak aggregate memory was 12.88 GiB. There
were no compile-cache retries. The [default affected selection](affected-default-summary.json)
completes **62 passed / 39 skipped**, and the corrected
[word-store trace check](word-trace-final-summary.json) passes explicitly.
The full default suite retains its standard slow skips; it does not waive or
erase the failing slow-enabled audit described above.

All 649 source files match
`ffcb8cb2c80b7c7d42078e6358c98fb5e4f19b8bd38303d865538c170db409e4`
across those final checks and the final measurements.
[Source manifest](review-source.json), [delta from the preceding reviewed landing](source-delta.json),
[source archive](review-source.tar.gz), [tracked-source patch](tracked-source.patch),
and [validation summaries](validation-summary.json) make the original
landing reviewable. The archive also contains the new regression file, which
is not part of the tracked-source diff. Final documentation-link verification
passes **85/85** checks. These receipts were recorded before review and publication.


Repeat the fixed measurement from BasicModel:

```sh
.venv/bin/python doc/benchmarks/2026-09-25-item9-parity/run_measurements.py --out output/item9-parity-repeat
```

The driver refuses an existing directory, runs fresh processes under an 8 GiB
cap and 600-second deadline each, and checks source stability. Seed 42 is
retained for comparison to the historical measurement; it is not a test seed.
The final comparison reports a null if any declared check fails.

## Scope and limits

This establishes layout parity at equal frozen model/dictionary state and
matched input units. Different optimizer/update schedules, fresh vocabulary
allocation orders, changed sentence boundaries, or different retained evidence
are different inputs and are not equated. Absolute source locations are not
rebased or removed. Learned prediction, discrimination and thought-work utility
still require item 9's subsequent declared multi-seed experiment.

Affected runtime files: [Models.py](../../../bin/Models.py) (bank ownership,
scoped inverse/byte assignment, final root publication) and
[Spaces.py](../../../bin/Spaces.py) (candidate provenance lifecycle).
The [native parity regression](../../../test/test_packed_reconstruction_parity.py)
checks per-sentence values and saved programs; the
[existing operand-provenance fixture](../../../test/test_reverse_traversal.py)
now supplies its required leaf precondition. Measurement scripts and this
receipt retain the diagnosis; the countdown keeps the remaining learning gate.
