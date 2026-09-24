# Item 11b: recurrent parts and the attended concept field

This corrects the accepted `db73581` / `440af47` review landing. The untracked
11c plan is excluded. The 11b negative-pole formula, witnessing, location,
normalization and primitive-input XOR gate remain the contract.

## Correction

A first word witness retains one ordered PS group literal, including repeated
ids. Neither the word writer nor the location writer interns that witness.
Containment reads the canonical tiling's ids and spans, with no byte
reconstruction. Radix recurrence records id ancestry; the literal then
addresses the promoted row without changing its exponent or reading.
A part formed in a turn cannot aggregate further during that turn.
Definition groups follow id ancestry when later overlapping formations change
the longest-match tiling; observed positions are never re-expanded.

Persistent sparse definitions retain their concept ids. Each sentence binds
its own bounded attended field, including independent bindings within a batch.
Temporary sparse views gather the original parameters, so their gradients and
optimizer state remain attached to the inventory. A missing required source
contributes unknown evidence; its exponent is not silently omitted.
Contexts use sparse inventory coordinates, avoiding a dense inventory square.

Code snapshots, checkpointed evidence and packed symbols carry concept ids.
The logical symbol addresses remain `2*cid` and `2*cid+1`; packed slot offsets
are not identities. Priming resolves ids to inventory addresses, and thought
uses the ids in its retained field, even after another turn reuses the slots.
The order-0 edge guard documents sigma admission and conjunctive rejection.

## Probes and validation

The [initial failing receipt](red-result.json.gz) reproduces first-witness
interning, byte expansion at the read, and permanent exhaustion after eight
order-0 definitions. [Workers](red-workers.log) preserve the failures.

The [location-writer probe](location-red-result.json.gz) separately fails when
an anonymous location flattens a repeated ordered group into individual
features; its [worker](location-red-workers.log) records that failure.
The [overlapping-formation probe](retiling-red-result.json.gz) fails when
an earlier `(a, bc)` definition does not follow the later canonical `(ab, c)`
tiling. Its [source](retiling-red-probe.py) and
[worker](retiling-red-workers.log) are retained.

The corrected probes cover ordered groups and repetition, recurrence
continuity, formation stopping aggregation, more than eight distinct words,
twelve independent sentences in a batch, checkpoint and thought identity,
and an unattended required conjunct. The twelve-word smoke reads every word
through the actual native writers with eight order-0 slots.

An [affected run](affected-memory-result.json.gz) hit its aggregate memory
cap; its [workers](affected-memory-workers.log) are retained. A subsequent
[run was interrupted](affected-interrupted-result.json.gz) during the final
caller audit. Neither is a success receipt.

The [caller-audit run](affected-caller-audit-result.json.gz) completes 224
cases with 221 passes and three failures. Its corrections are explicit:

- The native word-definition expectation now checks one ordered group
  literal, including its constituent ids, instead of independent PS columns.
- The training smoke uses the existing native-property fixture and a real
  optimizer epoch. The former raw-span fixture supplies WholeSpace spans
  outside its word extents; first-witness fusion had masked that mismatch
  by adding an occurrence. Runtime locations are unchanged. The separate
  twelve-word smoke exercises both serial and independent batch bindings.
- Checkpoint key pins were stale in accepted `440af47` itself, as the
  [baseline failure](baseline-pin-result.json.gz) records. The complete
  [baseline](baseline-checkpoint-keys.json) and
  [current](current-checkpoint-keys.json) key lists are identical: 554 XOR
  keys (`6761889f00108ece`) and 749 grammar keys (`a46523768706c5c9`).

The [final affected receipt](affected-result.json.gz) passes **44/44**,
including explicit slow training and optimizer checks, the failing callers,
overlapping recurrence, checkpoint restore and the twelve-word field.
It uses two CPU workers with one-case batches. Its
[source manifest](affected-source-manifest.json) covers **639** files;
the [changes since the caller audit](validation-source-delta.json) remain
recorded. No failed assertion is waived.

The [first full run](full-before-fixture-result.json.gz) completes all
4,794 cases with 4,462 passes, 330 skips, one existing expected failure,
and one failure: the lightweight word-binding fixture omits the inventory
lookup called by refinement. Its [failure log](full-before-fixture-failure.log)
and [source manifest](full-before-fixture-source-manifest.json) are retained.
The fixture is given the real lookup methods and inventory size; its
assertions are unchanged. A separate [correction probe](stubfix-probe-result.json.gz)
passes all ten word-binding checks before that correction is applied.
That fixture is the [only source change](full-source-delta.json) between
the first full run and the final affected run. The
[preceding 34-case affected receipt](affected-before-fixture-result.json.gz)
is retained separately.

All [six fresh unseeded native XOR runs](xor-full.json) pass in the full
suite: pool sizes four and eight, three runs each, at four conjunction
hypotheses. Their MSE is `2.842170943040401e-14`. The landed truth-corner
assertions, composition check and exact-zero unrelated-content controls
remain unchanged.

## Measurements

The [preserved driver](../2026-09-23-item11b/run_measurements.py) replays
the serial comparison with `conceptualPi` off and on, frozen primitive
memberships, and the prior/segmentation comparison. Seed 42 fixes this
measurement only; no assertion selects a seed. The
[comparison](final-source/comparison.json) is unchanged from accepted 11b:

| Serial reconstruction mean | `d4dc385` | Current, off and on |
| --- | ---: | ---: |
| Before training | 0.1005932558 | 0.1005932558 |
| Five measured updates | 0.0927930698 | 0.0949083805 |
| After training | 0.0923267286 | 0.0924575571 |

The final cost remains 0.142% above the reference; the update mean is 2.28%
above it. Frozen primitive memberships reproduce all three reference means
exactly. The [prior comparison](final-source/priors.json) retains all eight
prior rows over 256 bytes with zero error and unchanged segmentation for
32 sentences and 160 runs.

The [measurement manifest](final-source/manifest.json) matches all 639
source files in the final affected receipt, SHA-256 of the sorted JSON map:
`15201722241d80122b33e50a87fc1dddcf6cdfc8bb40caf5f8334ec7ba33f2d7`.

## Full validation

The [final full receipt](full-result.json.gz) completes **4,794 unique
cases: 4,463 passed, 330 skipped, one existing expected failure**, exit 0.
The expected failure remains
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
No failure is waived and no compiler-cache retry ran. The
[six fresh XOR runs](xor-full.json) pass again.

The [summary](full-summary.json) records timing and bounded memory. The
run uses ten one-thread CPU workers, eight-case, one-file batches, with
8/20 GiB worker/aggregate caps. All **639** files in the
[source manifest](full-source-manifest.json) match the final affected
receipt, measurements and working tree byte for byte, SHA-256
`15201722241d80122b33e50a87fc1dddcf6cdfc8bb40caf5f8334ec7ba33f2d7`.
Documentation is recorded separately and checked after the receipt and
todo are complete. Preserved raw logs and generated measurement XML keep
their original bytes.

[Receipt metadata](receipt-info.json) records the implementation commit.
All 639 committed source blobs match the validated snapshot byte for byte.

All **79 documentation-link cases** pass after completing the receipt and
todo ([receipt](doc-links-result.json.gz)). Source and prose pass whitespace
checks; raw logs and generated measurement XML retain their original bytes.
