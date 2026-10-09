# Review notes for the frozen math-chain candidate

These notes explain the saved observations; they do not change the measured
source or authorize a replacement training. The final counts are in the
receipt summary.

## Training failures

The document-preserving driver groups documents by length and keeps every
tail. As a result, the batch can change from eight streams to two or three.
`ModelAttention.stage_input` reads the concept owner's retained priming
weights before those weights have been resized for the new batch. The
codebook retrieval prior then multiplies a current-batch similarity tensor
by the previous eight-row priming surface. The saved exception originates
at `Attention.py`'s `_codebook_retrieval_prior`. `Spaces._priming_surface`
already has batch-size handling, but this read occurs before that handling.
A repair needs to establish the current document/batch ownership of the
priming surface before the attention read; broadcasting stale rows would
not establish that ownership.

The zero-budget control bypasses that attention read and reaches the LTM
limit. `MM_math_chain.xml` sets `truthMaxEntries` to 65,536, but it inherits
the separate, explicit `ltmCapacity` of 1,024 from `model.xml`. Language's
store construction gives `ltmCapacity` precedence. The run therefore raises
at 1,024 rows, before one complete epoch. A future configuration must account
for the actual retained occurrence/constituent rows over the whole declared
training, and must preserve any successor facts required by the verifier.
Simply increasing `truthMaxEntries` does not do so.

The sixth and tenth zero-budget attempts instead reach
`ValueError('a relation requires three resolved row references')` in
`ClauseRow.write_clause`'s preflight. Their saved traces come through the
ordinary `_commit_sentence` and `_append_observed_meaning` path. The store
correctly refuses an unresolved relational clause, but this training cannot
continue. The receipt does not capture the rejected clause's contents, so
it does not identify which of its three references was unresolved. This is
a separate failure from capacity exhaustion and from the repaired thought
fill's constituent ownership certificate.

The ninth paired start, with answers and with answers withheld, reaches
`ThoughtFaces.is_part` and then `Taxonomy.part_of`. Its second operand is
rejected by `concept_reference`, which requires a positive symbolic concept
reference. The proposal menu's existing signature validation has therefore
not enforced this executor's complete operand contract. The saved trace
locates the boundary but does not retain the rejected operand. This is an
additional runtime failure, not a failed held-out binding.

## Evidence that still needs to be established

The saved `questions.jsonl` records distinguish explicit math questions from
episodes on other sentences. Many question meanings are closed (`open: []`)
and open no episode; the zero-budget control also records open references
with no work spent. The mechanism certificates construct open references
and pass; they do not show that every question in this corpus's ordinary
sentence path constructs the needed open reference. Resolving the runtime
exceptions alone would not demonstrate that requirement. The unforced
sentence-to-question path needs its own construction case before another
declared learning measurement. The final receipt reports the observed
counts separately for each condition.

The equality certificate establishes a gradient to the verb's change
column for a constructed selected equality/VP. The partial training records
must separately establish that those selections occur on the counting
facts. Some partial runs do record this gradient, while others record none.
Neither observation establishes learned successor accuracy without the
completed training and evaluation.

The observer records thought credit calls, cost ties, operation counts and
raw chooser gradients as they occur. The driver records final chooser
parameter movement only after training and evaluation complete. A failed
attempt therefore has no final movement measurement. Initial snapshots and
partial gradients cannot substitute for that missing measurement.

Held-out and beyond-range evaluation occur after the configured training.
If a run fails first, its accuracy is unavailable, rather than zero percent.
The same distinction applies to both ablations. Every declared outcome is
retained; there is no replacement initialization or training retry.
