# Full-sweep findings while the frozen run is in progress

The completed sweep retains exactly these three failures; see the
[final findings](failures.md). The initial observation below is preserved.

This is a partial observation, not a completed sweep result. The source remains
the final 711-file snapshot; no repair or test port is applied during the run.

The first three failures all passed in the round-3 sweep:

| Case | Saved report |
|---|---|
| `test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[forward]` | [Worker 110](run/worker-110.json) |
| `test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[packed]` | [Worker 110](run/worker-110.json) |
| `test_item9b_interpret.py::test_selected_generic_grammar_ends_the_interpreted_kinds` | [Worker 123](run/worker-123.json) |

In each failure, the operand references agree and predicate slot 1 differs:
the stored row has `('sym', 4010033338316264258)`, while the expected meaning
has `('sym', 1)`. The first value is exactly `predicate_identity('part')`,
introduced for Z's grammar-predicate identity without an inventory row.
The second comes from the thought registry's `form('part', ...)`; the generic
grammar case compares the store against `Language.program_meaning`.

Thus clause recovery and the registry-based meaning path expose different
addresses for this predicate. These are recorded as failures, not dismissed
as fixture-only differences. No protected reference assertion is weakened.
