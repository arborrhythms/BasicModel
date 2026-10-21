# Eight-file gate on the third frozen source

`3d453e6bb6927e5da1492c25106a4ef107b87bca`: 110 passed, three failed,
one existing copular prerequisite skip; all 114 cases completed. Every
failure remains in `native-final`, whose manifest records the actual
825 source files. No earlier-source result substitutes for this gate.

Both unchanged merge certificates pass and end with one working group.
Their MTM pools still contain two and four separate entries respectively;
the read-time merge is not written to the pool. The c/e closings case,
G1–G3, episode case, reference-close and reference-mentions tests pass.

## Occurrence witness error

`test_ordinary_initial_binding_distribution_includes_every_retained_candidate`
fails while its observer locates a thought witness. `ReferenceClose.bind_site`
line 444 wraps every non-None context identity as `('sym', identity)`.
An MTM or unpooled occurrence identity is already an `('ltm', domain, key)`
address, so the result is an invalid nested symbol reference.

`witness-repro-04` reproduces the same TypeError with one native occurrence
and an ordinary bind. The selected antecedent is correct; its witness
cannot be located. This is a mechanism error, not an assertion port.
The focused reproduction is in `diagnostics/test_witness_repro.py`; it
does not edit the production source or an existing test. The candidate
repair is to preserve the occurrence address and wrap only integer native
identities, consistent with `ReferenceClose.occurrences`.

`diagnostics/runs/witness-repair-05` applies exactly that one-line patch
in an isolated copy of 3d453e6bb. The new reproduction and all four
unchanged occurrence-view cases pass. The [patch](witness-repair.patch)
and checkout manifest are retained; this development check is not a
frozen-source pass.

## Reverse pending premise

The separate reverse draw fails D2 at line 403 after its first close
explicitly mints x. The order-0 three complement does not become an
accessible candidate for its next frame. Its full capture and both-trial
keep tables are retained; see [the matching first-draw diagnosis](reverse-3-failure.md).
This failure is not called a passing port or attributed to an MTM eviction.

## Bootstrap reconstruction

`test_both_training_trials_share_bootstrap_snapshot` again fails its
unchanged zero-R assertion at line 165. This run has full initialization
and both-trial state from the outset in `native-final-bootstrap-state`.
Batch 1 has R = 0 throughout. Batch 2 has R = .00700668478384614 for
explore rows 0 and 2, with greedy R = 0; the other rows have R = 0.
Component reading is disabled and MTM is empty in both batches.

The [earlier captured diagnostic](../2026-10-21-item3-5-anonymous-bind-v2/bootstrap-failure.md)
isolates the same nonzero content-reconstruction term and reproduces it
identically on accepted MTM source eac2fe87b. That comparison is diagnostic
evidence, not a pass promoted to this source. The zero-R assertion,
reconstruction objective and keep law are unchanged. This failure remains
open for review and disposition.

After review, `bootstrap-reviewed-source-06` replays that captured
diagnostic on this exact 3d453e6bb source and fails identically. Initial
parameters and every comparable trial field match accepted eac2fe87b;
the explicit-bank capture and all batches match the 823f157cd diagnostic
in full. The earlier baseline serialized `primed` as a string, so no
equality is claimed between that representation and the later explicit
fields. [The comparison](bootstrap-reviewed-source-comparison.json)
records those limits. There are no bin/test/data differences between
823f157cd and 3d453e6bb, including ClauseJournal. This strengthens the
diagnosis; it does not waive the failed gate.
