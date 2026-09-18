# Query execution phases

Implementation reference, September 18.

## Permission follows completed sentence rows

`BasicModel.resolveAnswer()` opens checked query execution only while it
resolves an owned `Understanding`. With tied input reconstruction enabled, that
understanding must already own its completed reconstruction. Its nonempty
captured answer programs identify the permitted rows; missing or padded rows
cannot execute a query. A program-less dense compatibility topology can still
prepare an answer, but supplies no row permission for a checked executor.
Held understandings retain their own readiness after later staging.

The scope restores permission on success and exceptions. A nested resolution
can narrow an outer set of rows but cannot enable another row. This
administrative permission neither selects a query nor certifies a proposition.

## Sentence work masks execution

Input `forward()`, explicit compiled-state forward, supplied input executors,
input-reconstruction completion, `reverseReconstruct()`, and `reverseOutput()`
all mask checked query execution. `understand()` and `what()` use the same
input-execution wrapper. Direct signature calls and grammatical-VP dispatch
check permission before an occurrence read or executor effect. Legacy
`answer_query()`, `reason_about()`, and `think_about()` reject calls from a
sentence path as well.

Pure grammatical formation remains pure: composing or realizing a question
does not execute it. Standalone evidence readers without a `BasicModel` phase
guard retain their explicit read API.

## Compilation and gradients

The mask and completed-row set are transient host state, not tensors, semantic
features, parameters, losses, or checkpoint fields. Query tracing fails
directly; the host wrapper covers compiled numerical execution and eager
islands without relying on attribute mutation from a captured graph. The
published 21-value compiled forward contract is unchanged.

The guard does not detach any current-step payload. Existing input
reconstruction and prepared-answer gradients are preserved; hard query choice
still requires the later controller's explicit policy credit. See
[Gradient flow](GradientFlow.md).

## Evidence and limits

The rebased reviewer probes first failed in
`output/tests/20260918-010057-1c5b25` (16/16 completed with the expected
unmasked-execution failures). After the fix, the expanded preserved reviewer
set passed **23/23** in 166.9 seconds in
`output/tests/20260918-011031-938ed0`, including real fullgraph forward and
backward across word counts plus trace/eager-island negatives. The broader
affected selection passed **280/280** in
`output/tests/20260918-011401-fd5b1c`; fresh workers bounded allocator state
at a 1.56 GiB peak.

Per the user-directed no-rerun policy, this item has focused and affected-file
evidence rather than a new single-snapshot global suite receipt. The earlier
default-suite composite remains recorded in [Testing](Testing.md).

This guard does not implement normal selected linguistic meaning, the ordinary
thought controller, nested retention, shared query work, residual policy
credit, or learned causal utility. Those remain separate integrated-spec gates.
