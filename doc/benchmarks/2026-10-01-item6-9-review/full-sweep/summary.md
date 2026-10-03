# Final source-matched sweep

5218/5219 cases completed in 122.077 minutes.
Case outcomes: {'passed': 4895, 'skipped': 322, 'xfailed': 1, 'timeout': 1}. Extra passed subtest reports: 4.
Round 5: 5,122 cases, 89.6 minutes. Change: +97 cases, +32.477 minutes.
Runner time: 118.591 minutes; continuation preparation between runner segments: 3.486 minutes (included in wall time).
Peak aggregate memory: 17.380 GiB. Worker guard: 8 GiB; aggregate guard: 24 GiB.
Result: resource_stops. Coverage details: `combined-coverage.json`.
Guard-stopped cases: 1; interrupted sibling attempts: 3.
The original collected list was continued after a worker guard stopped dispatch. Completed cases and the case stopped by its own guard were not rerun; the original three-hour suite deadline and memory limits remained.

| Guard-stopped case | Reason | Seconds |
|---|---|---:|
| test/test_item9b_schedule.py::test_native_interleave_supplies_context_then_reads_the_same_sentences[True] | timeout | 1801.036 |

| Required regression/port | Outcome |
|---|---|
| test/test_item7_unindexed_relations.py::test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows | passed |
| test/test_output_walk.py::test_compiled_understanding_captures_explicit_sentence_products | passed |
| test/test_query_phase_fullgraph.py::test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths | passed |
| test/test_item7_definition_integrity.py::test_new_definitions_do_not_alias_later_sentence_identities | passed |
| test/test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_idempotent_clear_then_record | passed |
| test/test_runtime_split_ingestion.py::TestRuntimeSplitIngestion::test_store_truths_records_truths | passed |
| test/test_compiled_expectation_boundary.py::test_explicit_boundary_retains_factored_roles_and_skips_masked_rows | passed |
| test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[packed-cat-1] | passed |
| test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[packed-tree1-3] | passed |
| test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[pending-cat-1] | passed |
| test/test_existence_metadata.py::test_actual_observation_writers_keep_roles_without_asserting_world_facts[pending-tree1-3] | passed |
| test/test_expectation_defaults.py::test_packed_ltm_ignores_masked_slots_even_with_retained_storage | passed |
| test/test_item7_end_state_storage.py::test_closing_discards_operations_without_changing_other_live_rows | passed |
| test/test_joint_objectives.py::test_packed_prediction_teardown_records_each_observation_once | passed |
| test/test_selected_nested_meaning.py::test_10_packed_pending_and_eager_ends_write_identical_rows | passed |
| test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[batch_eval] | passed |
| test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[forward] | passed |
| test/test_selected_nested_meaning.py::test_nested_observation_writers_retain_children_without_certifying_them[packed] | passed |
| test/test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[forward] | passed |
| test/test_selected_relation_meaning.py::test_observation_boundary_uses_selected_relation_before_prediction_and_ltm[packed] | passed |
| test/test_sentence_expectation.py::test_streaming_and_packed_boundary_use_same_full_roles[False] | passed |
| test/test_sentence_expectation.py::test_streaming_and_packed_boundary_use_same_full_roles[True] | passed |
| test/test_tied_operator_reconstruction.py::test_missing_absorbed_operand_uses_bounded_compose_candidates | passed |
| test/test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers | passed |
| test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[1.0] | passed |
| test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[0.0] | passed |
| test/test_output_walk.py::test_runbatch_generate_policy_masks_rows_without_supplied_answers | passed |
