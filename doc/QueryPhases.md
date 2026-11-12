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

## Limits

This guard grants and withdraws permission only. Selected linguistic meaning
and the thought controller ([SelectedMeaning](SelectedMeaning.md)), nested
retention ([NestedRetention](NestedRetention.md)) and shared query work
([QueryWork](QueryWork.md)) are separate mechanisms, and learned causal
utility is unproven.
