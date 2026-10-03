# The 25 slowest cases: before and after

All 25 cases passed with their subjects and assertions retained. Before values are the prior sweep’s case-call seconds (one timeout). After values include cProfile overhead; setup is separate because two cases share one trained fixture. The final full sweep will supply unprofiled times on the final source.

| Case | Before call (s) | After call (s) | After setup (s) |
|---|---:|---:|---:|
| `test/test_item9b_schedule.py::test_native_interleave_supplies_context_then_reads_the_same_sentences[True]` | 1801.036 | 2.727 | 0.000 |
| `test/test_negative_expectation.py::test_native_unlabelled_batch_trains_the_same_chooser` | 1628.005 | 1.306 | 0.000 |
| `test/test_compiled_word_chunk.py::test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks` | 1390.340 | 1059.741 | 0.000 |
| `test/test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer` | 1150.582 | 2.787 | 0.000 |
| `test/test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers` | 1030.009 | 6.060 | 0.000 |
| `test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser` | 917.033 | 1.550 | 0.003 |
| `test/test_output_walk.py::test_question_conditioner_optimizer_steps_both_widths` | 902.683 | 2.306 | 0.000 |
| `test/test_arithmetic_isolation.py::test_supervised_update_cannot_use_exact_arithmetic_or_fallback_codes` | 902.172 | 1.700 | 0.000 |
| `test/test_output_walk.py::test_runbatch_generate_policy_masks_rows_without_supplied_answers` | 881.906 | 2.301 | 0.000 |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[1.0]` | 878.732 | 2.353 | 0.000 |
| `test/test_output_path_supervised.py::test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row` | 878.152 | 2.015 | 0.000 |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[0.0]` | 873.612 | 2.356 | 0.000 |
| `test/test_gradient_factorization.py::test_normal_batch_logs_named_shared_operator_gradients` | 777.435 | 1.927 | 0.000 |
| `test/test_surface_grammar.py::test_normal_batch_trains_supplied_grammar_lessons` | 770.411 | 1.814 | 0.000 |
| `test/test_expectation_defaults.py::test_native_future_and_other_row_changes_do_not_change_first_estimate` | 628.543 | 2.695 | 0.000 |
| `test/test_sentence_compose.py::test_real_packed_ends_train_before_the_next_sentence[True]` | 595.938 | 599.057 | 0.000 |
| `test/test_meronomy_ladder.py::test_utility_counts_accrue_once_per_presentation` | 556.259 | 0.001 | 2.736 |
| `test/test_meronomy_ladder.py::test_epoch_report_carries_the_word_unit_fraction` | 550.427 | 0.001 | 0.000 |
| `test/test_joint_objectives.py::test_real_intermediate_and_final_ends_have_same_canonical_roles` | 464.307 | 1.321 | 2.269 |
| `test/test_output_walk.py::test_packed_recall_observes_captured_sentence_programs` | 432.954 | 1.377 | 0.005 |
| `test/test_compiled_word_chunk.py::test_no_grad_fallback_retains_eager_stm_depth_semantics` | 406.261 | 0.319 | 0.002 |
| `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[True]` | 375.244 | 1.716 | 0.005 |
| `test/test_sentence_comparison.py::test_sentence_trials_keep_their_own_perception_pullbacks` | 328.771 | 1.279 | 2.308 |
| `test/test_sentence_compose.py::test_legacy_event_reporting_does_not_zero_the_sentence_objective` | 314.127 | 1.345 | 2.256 |
| `test/test_expectation_defaults.py::test_native_runtime_reports_pairs_without_accumulating_or_updating` | 307.798 | 1.479 | 1.699 |

The real K2 graph and packed compiled sentence remain compiler tests. They still dominate the measured time; the other cases exercise the same numerical and gradient behavior in eager loops. The earlier interrupted compiler profiles and failed small-field fixture probes remain saved.

The total is 19,742.7 before seconds versus 1,712.8 profiled after seconds. These are summed case times, not sweep wall time; concurrency, collection and process starts are separate. The 25-case selection includes the former 1,801-second timeout, now passing in 2.73 call seconds. Compiling was not its subject: its unchanged assertions check native interleave context and the subsequent serial readings. The K=2 graph and packed fullgraph cases remain compiled and account for most of the remaining time.

The separate weekly MPS validation exposed a missing fixture anchor, then a compiled closing input alias and an inactive pipeline journal address. Their failures and complete repairs are saved in `mps-staging-anchor-repair.json`, `mps-closing-alias-repair.json` and `journal-drain-repair.json`. They are additional correctness checks, not repeats used to improve this performance table.
