# Retained objective-contract preparation attempt

The first full sweep reported a new failure in
`test_free_trial_uses_no_reference_or_offsets_and_keeps_identity_audit`:
its exact list of reconstruction objectives omitted the requested content
term. The learner itself was not changed. The contract was updated to
include the term and assert that it participates in training.

The sweep and the concurrent early lexical warm-up were interrupted before
editing that test, retaining their logs, source, plan, and partial results
here. No branch had started. The experiment's source/corpus check is true;
the restart uses the identical learner, seed, budgets, and corpus. This
attempt is not a completed measurement or the final full receipt.
