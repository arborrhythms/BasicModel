# Follow-up to Claude's review of 3d453e6bb

Intermediate repair source: `a93f93c0e0f74a7fa51cc2b656a1c337be17ebfb`.

Alec's response to the forcing/objective question is: "Let's ask at review
time." No direction has been selected by that response. Keep, forcing,
objective, assertions and budgets stay unchanged while these questions are
carried into review.

The complete frozen gates belong to `3d453e6bb6927e5da1492c25106a4ef107b87bca`:
10/12 fresh first draws pass; the eight-file set is 110 passed, three
failed and one existing prerequisite skip; 48 focused cases, both original
decomposition arms and both original eight-seat MM draws pass. All failures
and exact probes are in the [frozen receipt](../2026-10-21-item3-5-anonymous-bind-v3/README.md).
The production changes in this follow-up have only the development checks
listed in its README. No frozen result is promoted to them.
The 20 arrival/review cases pass after the scope repair. On committed
`a93f93c0e`, the 14 MTM cases and four occurrence views pass; all 826
validated source files match that commit. The pooled ownership reproduction
remains a failure.

## Repairs to review

Occurrence witnesses remain locatable addresses; only integer native
identities receive a `('sym', id)` wrapper. Native arrivals without an
occurrence refine through their existing anonymous introduction, without
materializing a working vector in a synthetic LTM companion. A missing
arrival selection is reported explicitly. Admission copies the arrival
records before commit adds audit fields. Historical arrival sites must
belong to tokens accessible through the frames' occurrence records.

The native-arrival positive fixtures now cue the pending token as well as
the native candidate. Their existing assertions and capacities are
unchanged. A separate before/after reproduction verifies that an uncued
older site's kind, occurrences, count and time stay unchanged across a new
document in the same stream. All commit-isolation checks retain exact row
and bank comparisons.

## Questions for review

1. **Deterministic named-site certificate.** Both failed reverse draws
   first keep a nameless reading: R ties at .0360673778; named reference
   credit makes A=.4079441726, versus A=0 for the nameless mint/field
   reading. The [captured decomposition](../2026-10-21-item3-5-anonymous-bind-v3/reference-cost-breakdown.md)
   reconciles both trials in both batches of both failures; no premise
   answer target was supplied. Applying the arrival bind cannot refine a
   pending site that the kept reading never created. What contract should
   Alec choose to make the named site deterministic? Extending forcing
   would change its current scope; repricing reference operations would
   change the objective. Neither is implemented or implicitly authorized.

2. **Pooled-candidate ownership and content.** Selecting candidate B's
   introduction to refine anonymous token T transfers that introduction
   through `mtm.bind(T, B, ...)`. `forget(B)` deletes B's whole entry and
   its earlier uses' mappings. The retained regression creates B with
   content `three + distinguishing detail`, binds an earlier use to B,
   then selects B for T. T gains the full content but the earlier use
   loses B. A kind-only refinement preserves B's membership but loses the
   distinguishing detail in T. What ownership/read rule preserves both
   the chosen occurrence's full content and B's earlier uses, while keeping
   one owner per occurrence, introduction-address keys, no stored working
   vector, and no durable revision merge? The failing test is retained in
   [diagnostics/test_remaining_findings.py](diagnostics/test_remaining_findings.py);
   it is not marked xfail and its semantic assertions are not relaxed.

3. **Bootstrap zero-R failure.** The unchanged bootstrap test remains
   red. An exact captured replay on `3d453e6bb` matches the accepted MTM
   baseline's initial parameters and all comparable trial fields; see
   [the source comparison](../2026-10-21-item3-5-anonymous-bind-v3/bootstrap-reviewed-source-comparison.json).
   Both trials share their primed bank, have no MTM entries and no
   reference credit, yet the decoder returns smaller meaning magnitudes
   than the target leaves. The baseline reproduction isolates this from
   anonymous binding; it does not waive the failed gate. Review should
   distinguish a decoder mechanism repair from any proposed change to
   the existing assertion or objective.

Claude's suggested per-token deduplication and overwriting an existing
kind are not applied: Alec explicitly requires one rebind per pending
site and setting kind only when None. Arrivals use the same score,
choice_cost and chooser credit; their absence from ordinary delayed
mentions/answer supervision is declared, without adding a premise target.

## Publication boundary

This is an intermediate repair source, not the final frozen gate source.
After the review questions and remaining mechanism failures are resolved,
freeze the next source and run all twelve first draws with full captures
and failure probes, the eight-file set, both original decomposition arms,
and both original eight-seat MM draws on that hash. Then review the source
and receipts before both-main publication and the WikiOracle bump. No
second full sweep. Every failed run remains in the record.
