# Step 1b §10.4: closing bind and the decomposition open-name seam

Production is unchanged from `823f157cd`. This freeze also connects
`ForcedDecompositionGrammar`'s existing `open_names=('what',)` rule to
`ReferenceClose.select`, so an arrival bind cannot prefill its question.
The original snapshots, seeds, thought requests, reference preferences,
assertions, budgets, objective and live keep remain unchanged. Length 1
passes its original zero-arm snapshot in the development replay.

The [previous source's full ledger](../2026-10-21-item3-5-anonymous-bind-v2/README.md)
retains twelve fresh passes, 48 regression passes, the eight-file failure
and all four decomposition failures. Nothing is promoted to this hash.
Every required gate runs again here; the eight-file set now also captures
the bootstrap case from initialization, without changing its behavior.

The separate [bootstrap discrepancy](../2026-10-21-item3-5-anonymous-bind-v2/bootstrap-failure.md)
is unresolved: its captured draw fails identically on `823f157cd` and
accepted MTM source `eac2fe87b`, with component reading disabled and an
empty pool. No assertion or objective change is authorized or applied.
Any new pass will not erase that failure. The frozen hash, complete gate
ledger, source audit and Claude's review belong here before publication.
