# Shared selected-query work accounting

Implementation reference, September 18.

QueryWorkBudget is one transient integer allowance for a selected query tree.
Its read-only spent, remaining, and per-kind counts report actual debits. It
owns no meaning, checkpoint state, child allowance, reset, learned parameter,
or objective. A debit is atomic: an overlarge request fails without partly
consuming the allowance. QueryContext.work carries the same object through
preparation, execution, and nested calls.
[Meter](../bin/QueryWork.py#L15),
[context](../bin/Queries.py#L19).

| Cost kind | Charged before |
| --- | --- |
| operation | Calling a fully validated selected executor |
| reference | Reading a selected native VP or operand address |
| payload | Reading an addressed conceptual payload |
| node | Inspecting a taxonomy node or codebook candidate |
| record | Reading an LTM/thought row, taxonomy constituent, or prediction-context record |
| expansion | Using a captured taxonomy edge in a proof or neighbor result |

These units are explicit bounded work, not time or FLOPs. Local node, record,
depth, and expansion limits can only tighten the allowance. Taxonomy capture
reserves a share for proof traversal, so a snapshot cannot always consume the
entire meter before a direct proof. Capture, traversal, and nested work still
debit that one object.
[Capture allocation](../bin/QueryWork.py#L57),
[taxonomy capture](../bin/Taxonomy.py#L120), and
[traversal](../bin/Taxonomy.py#L69).

## Validate, charge, then read

Phase permission plus declared argument, domain, and width validation occur
before an operation debit. Invalid calls therefore cannot spend work or invoke
an executor. The grammatical registry also charges selected VP validation and
native operand preparation. Description-valued operands resolve through their
existing occurrence owner, so preparation cannot read a full description before
it shares the allowance.
[Checked call](../bin/Queries.py#L98),
[registry dispatch](../bin/Queries.py#L573), and
[occurrence resolution](../bin/Queries.py#L410).

Exist records actual inspected facts and keeps partial signed evidence when
work ends. Lookup rows, native vectors, thought occurrences, and taxonomy
records use the same meter. Quantize does not materialize a basis after
exhaustion. ARMA reserves all prior-context record cost before predictor
execution and leaves the pending external estimate unchanged. A nested
what(Q) callback must receive the original meter rather than start a new
allowance.
[Fact evidence](../bin/reasoning.py#L156),
[quantize](../bin/Queries.py#L289), and
[ARMA read](../bin/Layers.py#L10369).

Exhaustion reports explicit work_budget incompleteness; it neither proves
falsity nor permits a later uncharged read. Evidence acquired within the
allowance remains available. A partial failed occurrence resolution is still
recorded in the meter even where an older result field cannot expose its
intermediate scan count.

## Gradients, ownership, and remaining controller work

The meter uses host integers and creates no parameter, loss, or ordinary
derivative. Charging does not detach a live input payload or a live
thought-occurrence meaning; durable LTM descriptions retain their existing
detached boundary. Hard reference matching and taxonomy traversal remain
nondifferentiable, so accounting does not supply policy credit or make query
utility learned. See [Gradient flow](GradientFlow.md).

Standalone audited readers can omit work and retain their explicit local
limits. The normal learned controller must instead create one meter from its
episode allowance, forward it through every selected reader and callback, and
record its final cost exactly once in ordinary history. This module does not
add that controller, reconcile the older history budget automatically, or
change the existing cutoff-drain bound.

## Evidence and limits

The rebased reviewer run was red before implementation in
output/tests/20260918-021028-e3efa7: collection failed because QueryWork did
not exist. After the fix, all 13/13 reviewer cases passed in 20.1 seconds at a
0.40 GiB peak in output/tests/20260918-021708-e98757. They cover atomic
overdraw, validation before work, selected VP/payload reads, partial fact
evidence, occurrence resolution, nested callbacks, prediction reservation,
quantize, and taxonomy capture/proof sharing.

The broader non-overlapping affected selection passed 372/372 in
output/tests/20260918-021926-07eb2a, using three fresh bounded workers (189.5
seconds at 1.05 GiB, 89.6 seconds at 0.50 GiB, and 19.1 seconds at 0.40 GiB).
It includes checked query/phase, taxonomy, expectation, thought-history,
retention, LTM, existence, and truth-store regressions.

Per the user-directed no-rerun policy, this is focused and affected-file
evidence rather than a fresh single-snapshot global-suite receipt. The
historical default composite remains in [Testing](Testing.md). Normal
selected-meaning/controller integration, residual policy credit, expectations,
generation ownership, and learned utility remain separate gates; the
separately queued two-truths and forgetting work remains out of scope.
