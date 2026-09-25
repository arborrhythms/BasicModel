# Item 9 accepted review: reconstruction validation without host reads

Alec accepted Claude's review and requested this correction before publication:
keep the [reconstruction bank safeguards](../2026-09-25-item9-bank/README.md),
remove their per-batch device-to-host checks, and flag the historical packed
answer measurements in plain language. Publication is authorized after validation.

The warning in Testing now says that the September 16 supplied-answer runs at
batch size 2 reconstructed from zero-filled saved states instead of the actual
sentence end states. Those answer-quality numbers are unreliable and must not
be cited. The forward state existed; saving the final reconstruction root was
the broken step, repaired in the parity landing.

## Change and regression

Shape, dtype and missing-tensor checks still use Python metadata. Sentence-id
bounds and candidate coverage use three `torch._assert_async` checks, with no
tensor `bool`, `item`, `nonzero`, or `tolist` in the validator. The same checks
still reject a populated first sentence concealing an empty later sentence.
CPU invariant failures raise immediately; accelerator failures are reported
asynchronously at subsequent device work or synchronization. Error messages
name the violated invariant rather than copying sentence indices to the host.
Admission and surface staging retain their existing host work; this correction
removes the host reads introduced by bank validation.

Two new probes failed before the correction. Fake tensor execution rejected
`aten._local_scalar_dense`; a real MPS profile recorded `aten::item` and
`aten::_local_scalar_dense` inside the validator. The corrected tests require
neither operation. The MPS test synchronizes only outside its measured boundary
to surface device errors. CPU fault cases retain the existing admission/surface
checks and add invalid input/candidate sentence-id checks.

The runtime uses PyTorch's native device assertions, including its
[MPS implementation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Assert.mm).
The host profile verifies this machine's installed runtime, rather than assuming
that an operation's name alone guarantees asynchronous behavior.

The initial synthetic probe used fake CUDA tensors. The repaired validator's
boolean inversion reached an unsupported CUDA device guard in this macOS build.
The final fixture uses fake CPU tensors with the same forbidden-data-read
assertion; both final probes were rerun red against the archived reviewed
production source before rechecking the fix. The initial affected failure and
interrupted slow run are preserved as diagnostics, not passing receipts.

## Validation

The [full receipt](full-summary.json) completes **4,844 cases: 4,519 passed,
324 skipped and 1 existing expected failure**, with no red outcomes or
compiler-cache retries. It finishes in **1,276.5 seconds**, using six workers
at a **10.37 GiB** aggregate peak. Its 12 GiB aggregate budget plus the concurrent
slow run's 8 GiB budget keep their combined reservation at 20 GiB; the per-worker
cap remains 8 GiB.

The [affected selection](affected-summary.json) completes **197 cases: 129 passed /
68 skipped**. The [explicit slow selection](slow-summary.json) passes **24/24**,
including native ambient parity, compiled retained-gradient reads, and live
and detached encoder-credit training. The [red receipt](red-summary.json)
preserves the two final probes against the preceding reviewed implementation.

The [fresh packed/single measurement](measurements/comparison.json) agrees
exactly in every measured final seal, saved program root, retained/recovered
leaf and byte cost: both means remain **.6839025616645813**. The
[serial baseline](measurements/baseline.json) remains **.08916187100112438**
after seven updates; every measured reconstruction and answer step matches the
preceding receipt. Historical measurement seed 42 is unchanged; no regression
test chooses a seed.

All **650 source files** match across the affected, slow, full and measurement
receipts, the [review manifest](review-source.json), and the
[source archive](review-source.tar.gz). The manifest hash is
`08ea1dc59989ae9dac0f71b1d1c75d41b0a3bd6cfdf957776275ba01d6aa6cdc`.
The [validation summary](validation-summary.json) records the individual limits
and outcomes; [source deltas](source-delta.json) identify the two files changed
since the preceding bank review.

The inherited reconstruction-cache compose-gradient failure remains open under
item 1, as documented in the original parity receipt; it is not waived here.

Documentation-link verification passes **88/88** on the same source manifest.
The implementation and the 9b planning documents are included in Alec's
authorized full-tree publication.

Implementation `0e70001` includes all changes in the reviewed tree, including the
latest 9b plan with section 4e. All 650 committed source blobs match the review
manifest ([committed-source verification](committed-source-verification.json)).
Item 9b is next; the expectation-learning gates remain open.
