# Six fixture failures in the full sweep

The full sweep completed with **4,793 passed, 6 failed, 322 skipped and one
expected failure**. Every collected case completed once. All six failures
passed in round 4; all three round-4 failures and thirteen round-3 failures
now pass ([comparison](comparison.md)). At the review stop, nothing below had been repaired or
ported. The [accepted landing](../landing/README.md) now records the three
authorized ports and the six passing cases; the runtime is unchanged.

The [failure analysis](failure-analysis.json) retains each traceback and the
complete unchanged bodies of the tests and shared helper. Their source hashes
match the incoming round-5 candidate and the final tested manifest.

## Five inventory-only fixture failures

| Case | Failure before the protected assertions |
|---|---|
| `test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0.8-0.7-BOTH]` | `terminal_model_index` asks for the predicate's inventory row |
| `test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0.8-0.1-TRUE]` | Same |
| `test_replacement_kernel_contracts.py::test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees[0-0-UNKNOWN]` | Same |
| `test_taxonomy_policy.py::test_unrelated_true_episode_cannot_establish_the_next_parent_relation` | Same |
| `test_accessible_mind.py::test_normal_what_effect_enters_recency_and_detached_knowing` | Its local `terms` fixture also calls `_existing_row` on every role |

The operand references have inventory rows. AK gives `part` one canonical
identity and point without an inventory row, as §22 requires. These fixtures
still treat the middle predicate as an inventory concept; `_existing_row`
therefore raises `ValueError: query concept reference is unavailable`.
The evidence and recency assertions have not run in these five cases.

The apparent next port is to represent the predicate according to its canonical
identity in the index fixture while retaining inventory lookup for operand
concepts. It must preserve the existing conflicting-evidence, unrelated-episode,
recency and detachment assertions and must not mint a predicate inventory row.
This was the proposed port at review; its implementation and verification
are now recorded in the accepted landing.

## One renaming assertion

`test_arithmetic_isolation.py::test_renamed_native_vocabulary_preserves_checked_relation_answers`
requires every role reference to differ between two independently allocated
vocabularies. The operand identities differ, but their `part` predicate is now
the same canonical identity. That explicit assertion fails. The following
fixture loop would also try to allocate and overwrite an inventory row for
the predicate.

A port would need to distinguish renamed operands from the shared predicate,
copy only operand content, and retain the oracle-poisoning, equal numerical
meaning and checked forward/reverse answer assertions. At review the protected test was
unchanged. The accepted port changes only the predicate-sharing expectation
and fixture copy loop; its other assertions remain intact.

## Coverage and limits

The [coverage audit](coverage-audit.json) lists every skip and its reason.
All 322 skips are identical to round 4. These include optional slow runs,
unavailable GPU paths, missing optional dependencies/data/checkpoints and
previously retired fixtures. The required slow XOR proofs ran separately in
the [explicit XOR table](../final-xor-comparison.md); a default-sweep skip is
not a passing learning result. The prior depth-three campaign remains red,
and the mature FineWeb checkpoint is still unavailable.

The one existing expected failure remains
`test_stm_recon_from_cleared_cache.py::test_topk_recovered_words_overlap_input`.
There are no missing, duplicate-completed or unreported selected cases. The
flags test emits four unittest subtest reports plus its parent report; the
summary counts its single collected node once.

No guard stops, diagnostics or compiler-cache retries occurred. The sweep
took 5,378.6 seconds (89.6 minutes), peaking at 5.263 GiB per worker and
19.169 GiB in aggregate, under the unchanged 8 GiB / 24 GiB limits.
