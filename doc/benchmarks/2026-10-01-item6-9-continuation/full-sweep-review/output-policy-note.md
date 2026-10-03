# Output-policy probe failure, observed during the frozen full sweep

The first failure was `test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers`.
It stopped at `_policy_training_probe.backward_probe`, line 281, asserting
`m._output_policy_cost is None` during a sentence backward. The stack reaches
this from line 369, the *supervised warm-up* call. It did not reach the later
`data.has_supervised_outputs = False` phase.

Step 5a now calls `reverseOutput` to score each supplied-answer trial with the
understanding graph intact. That call records an output-policy cost. The shared
probe predates that call and expects no recorded cost at sentence backward.
This failure alone does not demonstrate that a sentence without an answer
trains the generation policy; its no-answer phase was not executed.

The remaining gradient assertions have not yet run past this assertion. They
must not be assumed green. Review must decide whether the transient policy-cost
record should be scoped out of the trial preview or the probe should be ported
to recognize it while preserving the policy-gradient, zero-weight, row-mask and
no-answer invariants. No assertion or source code was changed mid-sweep, and
no second full sweep was launched. The failed worker log and exact source are
retained in `../full-sweep/run/worker-005.log` and the source manifests.
