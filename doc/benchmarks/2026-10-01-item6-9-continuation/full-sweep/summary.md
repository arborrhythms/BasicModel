# Final source-matched sweep

All **5,193 cases** completed in **113.5 minutes**: **4,849 passed, 21 failed, 322 skipped, 1 xfailed**, plus four passed subtests. The four subtests belong to `test_use_flags.py::TestOrthogonalFlags::test_flags_match_expected`; they are not four additional collected cases.

There were no missing cases, duplicate completed cases, resource stops or compile-cache retries. The 8 GiB worker guard and 24 GiB aggregate budget were unchanged; measured aggregate peak was 17.84 GiB.

Round 5 had 5,122 cases in 89.6 minutes. This sweep has 71 more cases and took 23.9 minutes longer. The case sets and early failing paths differ, so this is a wall-time comparison, not a controlled speed benchmark.

All six accepted October 1 repair cases pass. The 21 remaining failures are preserved below and in [the complete failure reports](failures.json). No repairs or assertion changes were made after the source was frozen. No second full sweep was run.

| Case | Seconds | Failure family |
|---|---:|---|
| `test/test_compiled_expectation_boundary.py::test_explicit_boundary_retains_factored_roles_and_skips_masked_rows` | 0.013 | private_sentence_tuple_fixture |
| `test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[packed-cat-1]` | 0.007 | private_sentence_tuple_fixture |
| `test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[packed-tree1-3]` | 0.008 | private_sentence_tuple_fixture |
| `test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[pending-cat-1]` | 0.015 | private_sentence_tuple_fixture |
| `test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[pending-tree1-3]` | 0.008 | private_sentence_tuple_fixture |
| `test/test_expectation_defaults.py::test_packed_ltm_ignores_masked_slots_even_with_retained_storage` | 0.013 | private_sentence_tuple_fixture |
| `test/test_item7_end_state_storage.py::test_closing_discards_operations_without_changing_other_live_rows` | 0.000 | private_sentence_tuple_fixture |
| `test/test_joint_objectives.py::test_packed_prediction_teardown_records_each_observation_once` | 0.011 | private_sentence_tuple_fixture |
| `test/test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers` | 863.609 | output_policy_preview |
| `test/test_output_walk.py::test_runbatch_generate_policy_masks_rows_without_supplied_answers` | 873.062 | output_policy_preview |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[0.0]` | 903.370 | output_policy_preview |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[1.0]` | 898.512 | output_policy_preview |
| `test/test_selected_nested_meaning.py::test_10_packed_pending_and_eager_ends_write_identical_rows` | 0.009 | private_sentence_tuple_fixture |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[batch_eval]` | 0.008 | private_sentence_tuple_fixture |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[forward]` | 0.009 | private_sentence_tuple_fixture |
| `test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[packed]` | 0.009 | private_sentence_tuple_fixture |
| `test/test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[forward]` | 0.009 | private_sentence_tuple_fixture |
| `test/test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[packed]` | 0.008 | private_sentence_tuple_fixture |
| `test/test_sentence_expectation.py::test_streaming_and_packed_boundary_use_same_full_roles[False]` | 0.033 | private_sentence_tuple_fixture |
| `test/test_sentence_expectation.py::test_streaming_and_packed_boundary_use_same_full_roles[True]` | 0.014 | private_sentence_tuple_fixture |
| `test/test_tied_operator_reconstruction.py::test_missing_absorbed_operand_uses_bounded_compose_candidates` | 0.268 | hard_inverse_assertion |

The four output-policy cases fail on the preview bookkeeping assertion before their later policy-gradient checks; the no-answer case fails during its supervised warm-up. The sixteen tuple fixtures omit the two new private CSLang fields. The remaining inverse assertion expects an average, while step 6 now chooses a least-residual pair. These diagnoses do not establish that the unreached assertions pass.

[Output-policy diagnosis](../full-sweep-review/output-policy-note.md); [fixture diagnosis](../full-sweep-review/fixture-note.md); [inverse diagnosis](../full-sweep-review/inverse-probe-note.md).
