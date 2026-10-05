# Remaining failures at the review checkpoint

The frozen default suite completes all 5,043 selected nodes. Outcomes: {'passed': 4722, 'skipped': 285, 'failed': 35, 'xfailed': 1}. 30 failures reproduced at published HEAD; 5 remain outside that reproduced set. None is waived.

## Unresolved candidate failures

- `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[False]`
- `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[True]`
- `test/test_ops_lift_lower.py::TestConjunctionDisjunctionForwarders::test_conjunction_bitonic_equals_lower_soft`
- `test/test_output_path_supervised.py::test_native_answer_uses_owned_ideas_without_dense_symbol_state[False-development]`
- `test/test_output_path_supervised.py::test_native_answer_uses_owned_ideas_without_dense_symbol_state[True-development]`

## Previously reproduced failures

- `test/test_generation_catalog.py::test_declared_generation_catalog_shares_one_optimizer_owner[False]`
- `test/test_generation_catalog.py::test_alias_catalog_reordering_preserves_shared_identity_without_rng_or_state`
- `test/test_gradient_factorization.py::test_normal_batch_logs_owned_optimizer_gradients`
- `test/test_grammar_separator.py::test_mixing_separator_is_perceived_but_not_pushed_or_read_back[False-False]`
- `test/test_grammar_separator.py::test_mixing_separator_is_perceived_but_not_pushed_or_read_back[False-True]`
- `test/test_grammar_separator.py::test_mixing_separator_is_perceived_but_not_pushed_or_read_back[True-False]`
- `test/test_grammar_separator.py::test_mixing_separator_is_perceived_but_not_pushed_or_read_back[True-True]`
- `test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser`
- `test/test_joint_objectives.py::test_answer_path_operators_are_independently_owned_and_keep_learning`
- `test/test_output_path_supervised.py::test_native_checkpoint_restores_active_answer_widths[False-development]`
- `test/test_output_path_supervised.py::test_native_checkpoint_restores_active_answer_widths[True-development]`
- `test/test_output_walk.py::test_walk_unreduces_a_rule_stamped_top_and_emits_its_constituents`
- `test/test_output_walk.py::test_walk_reports_truncation_when_work_is_pending`
- `test/test_output_walk.py::test_walk_handles_mixed_output_lengths_per_row`
- `test/test_output_walk.py::test_generate_policy_returns_sampled_action_credit_without_imitation`
- `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[1.0]`
- `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[0.0]`
- `test/test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers`
- `test/test_output_walk.py::test_runbatch_generate_policy_masks_rows_without_supplied_answers`
- `test/test_output_walk.py::test_output_rule_inventory_comes_from_generate_even_without_compose_rule`
- `test/test_output_walk.py::test_generate_walk_with_no_declared_rules_only_emits`
- `test/test_prepared_answer_boundary.py::test_prepared_answer_generates_more_words_than_the_captured_input`
- `test/test_readback_operand_unaries.py::test_saved_roots_follow_operand_unaries[False]`
- `test/test_readback_operand_unaries.py::test_saved_roots_follow_operand_unaries[True]`
- `test/test_reader_trial_ownership.py::test_generation_lesson_merge_keeps_its_sentence_row`
- `test/test_reconstruction_scope.py::test_missing_packed_sentence_does_not_dilute_the_owned_reconstruction`
- `test/test_sentence_end_state.py::test_numerical_journal_is_fullgraph_and_keeps_the_gradient_path`
- `test/test_thought_answer_adapters.py::test_typed_results_survive_resolve_and_reverse_without_execution[set]`
- `test/test_thought_answer_adapters.py::test_typed_results_survive_resolve_and_reverse_without_execution[code]`
- `test/test_thought_answer_adapters.py::test_typed_results_survive_resolve_and_reverse_without_execution[subgoal]`

Complete messages are in [remaining-failures.json](remaining-failures.json).
The intersection change is provisional; the other candidate failures concern native output distinction and answer-conditioner gradients. These are failures to repair, not ports justified merely by a changed implementation.
