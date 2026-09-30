# Three remaining failures in the completed sweep

The frozen source-matched sweep completed every one of its 5,100 cases:
4,774 passed, three failed, 322 skipped and one expected failure. There are
no missing cases, duplicate completions, resource stops or diagnostic reruns.
All thirteen failures from round 3 now pass ([comparison](comparison.md)).

All three new failures passed in round 3. They share one predicate-address
mismatch; their original assertions remain unchanged.

| Case | Saved assertion failure |
|---|---|
| `test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[forward]` | [Worker 110](run/worker-110.json) |
| `test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[packed]` | [Worker 110](run/worker-110.json) |
| `test_item9b_interpret.py::test_selected_generic_grammar_ends_the_interpreted_kinds` | [Worker 123](run/worker-123.json) |

The operand references agree. Predicate slot 1 in the stored row is
`('sym', 4010033338316264258)`, exactly `predicate_identity('part')` from
Z's grammar-predicate identity without an inventory row. The expected slot
is `('sym', 1)`, supplied by the thought registry's `form('part', ...)`.
The generic-grammar case obtains the expected meaning through
`Language.program_meaning` before comparing it with the completed row.

Clause recovery and the registry-based meaning path therefore expose different
addresses for the predicate. This is a remaining contract mismatch, not a
numerical tolerance failure. The assertions after the failed reference
comparison do not execute in these three cases. No test is retired, weakened,
or ported to make this sweep green. Any subsequent repair must preserve Z's
no-inventory ownership and the operand, prediction and gradient checks.

The candidate is handed back for review with this failure cluster unresolved.
[Complete machine-readable summary](summary.json).
