# Closing candidate measurements

One unseeded attempt per declared run, candidate only. The accepted item-7 baseline remains 44/49, 12/15 exact round trips and MM_grammar median ending MSE .1066. The October 1 receipt rule removes fourteen repeated exact trials, and the explicit trim removes the SPNN smoke case: 49−14−1=34 named cases. No failed run is retried.

The unchanged XOR_grammar fixture inherits lexicon synthesis. The explicit old-reading-mode exemption leaves tied reconstruction disabled for these default gates; the reconstruction-enabled stage-1 copy is recorded separately. No migration or configuration override was applied here.

| Group | Completed measurements | Passes at unchanged bar |
|---|---:|---:|
| class | 10/10 | 9/10 |
| reconstruction | 10/10 | 0/10 |
| sum | 10/10 | 0/10 |

Answers are in the input order saved in `summary.json`. The contrast is y(hello world)+y(loving there)−y(hello there)−y(loving world).

| Run | Four answers | MSE | Read-backs in the same order | Contrast |
|---|---|---:|---|---:|
| class 1 | 0.1626337, 0.8587227, 0.8413896, 0.255486 | 0.034209837 | hello hello; there hello; there world; there there | -1.2819926 |
| class 2 | 0.01692393, 0.7747709, 0.9637247, 0.1996937 | 0.02305201 | world there; there there; world loving; loving there | -1.5218779 |
| class 3 | 0.1929136, 0.6965196, 0.7719203, 0.1333138 | 0.049777229 | world hello; hello hello; world loving; loving hello | -1.1422125 |
| class 4 | 0.1729903, 0.9806475, 0.8933251, 0.009626478 | 0.010443091 | world world; there loving; world there; there world | -1.6913558 |
| class 5 | 0.1727692, 0.9070039, 0.8121417, 0.06618172 | 0.019542056 | hello world; there hello; world world; there there | -1.4801947 |
| class 6 | 0.2802141, 0.7257557, 0.7006046, 0.0993329 | 0.063308639 | world world; hello world; world hello; hello there | -1.0468132 |
| class 7 | 0.09647909, 0.9271157, 0.8930531, 0.118129 | 0.010003111 | world world; there world; loving world; loving there | -1.6055607 |
| class 8 | 0.08418351, 0.7756287, 0.8578414, 0.120307 | 0.023028052 | world world; there loving; world world; hello there | -1.4289795 |
| class 9 | 0.09239393, 0.7244996, 0.9914709, 0.1344135 | 0.025644208 | world hello; there hello; world world; world there | -1.489163 |
| class 10 | 0.1742206, 0.8298452, 0.8142068, 0.2451915 | 0.038485858 | world world; there world; loving world; there loving | -1.22464 |
| reconstruction 1 | 0.2247711, 0.804115, 0.7823763, 0.2232096 | 0.04651889 | world world; there hello; world hello; hello world | -1.1385106 |
| reconstruction 2 | 0.07399315, 0.9191888, 0.9468697, 0.06783116 | 0.0048573335 | world world; there world; loving world; loving there | -1.7242342 |
| reconstruction 3 | 0.1799371, 0.7669507, 0.8556524, 0.1310434 | 0.03117449 | hello loving; there hello; there hello; there world | -1.3116226 |
| reconstruction 4 | 0.08800173, 0.897598, 0.9827852, 0.2504138 | 0.020308477 | world hello; world there; world hello; there there | -1.5419676 |
| reconstruction 5 | 0.3706096, 0.8445709, 0.8224383, 0.1796745 | 0.056330187 | world loving; loving there; world world; there loving | -1.1167251 |
| reconstruction 6 | 0.2237007, 0.8556409, 0.8517368, 0.1098925 | 0.02623497 | world world; there world; there world; loving there | -1.3737845 |
| reconstruction 7 | 0.2859884, 0.9835483, 0.8841652, 0.05411941 | 0.024601659 | world hello; there hello; there world; there there | -1.5276057 |
| reconstruction 8 | 0.1629155, 0.8466649, 0.89948, 0.1188371 | 0.018569911 | world loving; there loving; hello world; there hello | -1.4643923 |
| reconstruction 9 | 0.1046105, 0.8836542, 0.9721203, 0.2446471 | 0.021277294 | world hello; world there; world there; there there | -1.5065169 |
| reconstruction 10 | 0.1123236, 0.9684374, 0.8970714, 0.06841218 | 0.0072218329 | loving loving; loving there; there hello; hello there | -1.6847729 |
| sum 1 | 0.5018044, 0.5007592, 0.5032402, 0.5021951 | 0.25000477 | loving loving; loving loving; hello hello; hello hello | 0 |
| sum 2 | 0.5009769, 0.500721, 0.5010005, 0.5007447 | 0.25000077 | there there; world world; there there; world world | 0 |
| sum 3 | 0.4996161, 0.5009937, 0.5000042, 0.5013819 | 0.25000077 | there there; world world; there there; world world | 2.9802322e-08 |
| sum 4 | 0.4992089, 0.5004169, 0.4971407, 0.4983487 | 0.25000292 | world world; loving loving; loving loving; hello hello | -2.9802322e-08 |
| sum 5 | 0.502298, 0.4994361, 0.501425, 0.4985631 | 0.25000241 | loving loving; there there; hello hello; world world | 0 |
| sum 6 | 0.5011557, 0.5007605, 0.501084, 0.5006889 | 0.25000089 | world world; hello hello; there there; world world | 0 |
| sum 7 | 0.4977329, 0.4974688, 0.4982524, 0.4979883 | 0.25000465 | there there; world world; there there; world world | 2.9802322e-08 |
| sum 8 | 0.498683, 0.4990615, 0.4986202, 0.4989988 | 0.25000137 | there there; world world; world world; there there | 2.9802322e-08 |
| sum 9 | 0.5000869, 0.503058, 0.500342, 0.5033131 | 0.25000513 | there there; world world; loving loving; there there | 0 |
| sum 10 | 0.4992765, 0.4995592, 0.4987151, 0.4989979 | 0.25000083 | there there; world world; loving loving; there there | 2.9802322e-08 |

The sum comparison retains the prior float32 `torch.allclose` defaults. Its raw maximum deviations are in the JSON. Class and reconstruction bars are evaluated independently for every observed run.

| Named table group | Cases | Outcomes |
|---|---:|---|
| 0 | 6 | {'passed': 6} |
| 1 | 10 | {'passed': 10} |
| 2 | 7 | {'passed': 7} |
| 3 | 1 | {'passed': 1} |
| 4 | 1 | {'passed': 1} |
| 5 | 1 | {'passed': 1} |
| 6 | 1 | {'failed': 1} |
| 8 | 1 | {'passed': 1} |
| 9 | 1 | {'passed': 1} |
| 10 | 1 | {'passed': 1} |
| 11 | 1 | {'passed': 1} |
| 12 | 1 | {'passed': 1} |
| 13 | 1 | {'passed': 1} |
| 14 | 1 | {'passed': 1} |

Named-table outcomes: {'passed': 33, 'failed': 1}.

| Named case | Outcome |
|---|---|
| `test/test_grounded_xor.py::test_native_unseeded_xor[0-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[0-8]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[1-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[1-8]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[2-4]` | passed |
| `test/test_grounded_xor.py::test_native_unseeded_xor[2-8]` | passed |
| `test/test_concept_output.py::test_concept_output_resolves_each_batch_identity_and_preserves_unknown` | passed |
| `test/test_concept_output.py::test_reconstruction_without_owned_evidence_cannot_read_the_latest_field` | passed |
| `test/test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence` | passed |
| `test/test_concept_output.py::test_parallel_field_never_dispatches_grammar_lift_or_lower` | passed |
| `test/test_concept_output.py::test_serial_grammar_never_executes_field_sigma_pi_or_not` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0]` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1]` | passed |
| `test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2]` | passed |
| `test/test_concept_output.py::test_contained_ordered_group_reconstructs_at_its_extent` | passed |
| `test/test_concept_output.py::test_property_reverse_attributes_only_written_members` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_a_forward_runs` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_convergence` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_forward_keeps_continuous_symbols` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_has_conceptual_symbolic_spaces` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_learns_xor_signal` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_mm_grammar_learns_xor_signal` | passed |
| `test/test_mm_xor.py::TestMMXorConvergence::test_model_is_mental` | passed |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_output_mse_is_crisp` | passed |
| `test/test_explicit_dimensions.py::TestXorExactCliReconstruction::test_at_least_50_pct_inputs_reconstruct` | passed |
| `test/test_explicit_dimensions.py::TestXorGrammarLearnsXor::test_xor_class_accuracy` | passed |
| `test/test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` | failed |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor]` | passed |
| `test/test_config_matrix.py::test_config_builds_runs_and_reconstructs[xor_noraise]` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_loss_is_live` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_recon_grads_flow` | passed |
| `test/test_reconstruction_roundtrip.py::test_xor_percepts_tile_words` | passed |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_roundtrip_at_harness_budget` | passed |
| `test/test_reconstruction_roundtrip.py::test_mm20m_xor_exact_roundtrip` | passed |

MM_grammar completed 10/10 full 900-update runs. Median ending training MSE: 0.06250000000415867 (accepted item 7: .1066).

| MM run | Completed epochs | Ending training MSE | Process |
|---|---:|---:|---|
| 1 | 900 | 5.5067062021407764e-14 | exit/0 |
| 2 | 900 | 0.25 | exit/0 |
| 3 | 900 | 0.125 | exit/0 |
| 4 | 900 | 5.329070518200751e-15 | exit/0 |
| 5 | 900 | 4.0234482412415673e-13 | exit/0 |
| 6 | 900 | 0.25 | exit/0 |
| 7 | 900 | 0.125 | exit/0 |
| 8 | 900 | 5.773159728050814e-15 | exit/0 |
| 9 | 900 | 0.125 | exit/0 |
| 10 | 900 | 8.317346811281823e-12 | exit/0 |

Source matched: True. Measurement campaign wall time: 43.81 minutes. The final full-sweep receipt is recorded separately in `full-sweep/receipt.json`.

## Final source-matched sweep

**5090/5090 cases completed**, 5090 attempted; 110.70 minutes wall time. Previous: 5219 cases / 122 minutes. Case-count change: -129; wall-time change: -11.30 minutes.

Case outcomes: `{'passed': 4760, 'skipped': 318, 'failed': 11, 'xfailed': 1, 'process_failed': 0}`. Source matched: `True`. Largest worker: 7.240 GiB. The unchanged worker ceiling is 8 GiB and deadline is 30 minutes. Two workers shared 16 GiB alongside the independent weekly ordinary worker's 8 GiB. The previous runner allowed ten workers, 256 selectors per batch and 16 files per batch; this run used two workers, eight selectors and one file per batch. Scheduling differs, so this is an observed wall-time comparison.

This sweep completed in one segment with no process stops and no compilation-cache retries. No measured case was retried.

The raw runner counts report events. One flags test emits four passed subtests plus its parent report; these are one collected case. `case-accounting.json` preserves both counts and those five reports. No test was rerun to resolve this accounting difference.

Weekly coverage warning: Latest weekly slow-test run did not complete on matching source..

| Failure | Phase | Evidence |
|---|---|---|
| `test/test_compiled_word_chunk.py::test_tiny_canonical_detached_reverse_stops_at_root` | call | test/test_compiled_word_chunk.py:955: in test_tiny_canonical_detached_reverse_stops_at_root     assert model.detached_reverse E   assert False E    +  where False = BasicModel(\n  (loss): ModelLoss(\n    (output_criterion): MSELoss()\n  )\n  (inputSpace): InputSpace(\n    (subspace): Sub...t(\n    (0-3): 4 x _BodyStage()\n  )\n  (where_encoding): WhereEncoding()\n  (when_encoding): WhenStartDurationEncoding()\n).detached_reverse |
| `test/test_compiled_word_chunk.py::test_tiny_canonical_detached_reverse_train_step_is_finite` | call | test/test_compiled_word_chunk.py:982: in test_tiny_canonical_detached_reverse_train_step_is_finite     before = [p.detach().clone() for p in chooser.parameters()]                                           ^^^^^^^^^^^^^^^^^^ E   AttributeError: 'NoneType' object has no attribute 'parameters' |
| `test/test_compose_deadline.py::test_shared_stack_contract_allows_unary_until_the_deadline` | call | test/test_compose_deadline.py:18: in test_shared_stack_contract_allows_unary_until_the_deadline     test_shared_operation_writes_the_caller_owned_stack(_xor_model) test/test_subspace_what_stm_contract.py:1309: in test_shared_operation_writes_the_caller_owned_stack     assert bool((out[0][:, 0].abs().sum(-1) > 0).all()) E   assert False E    +  where False = bool(tensor(False)) E    +    where tensor(False) = <built-in method all of Tensor object at 0x120a01ae0>() E    +      where <built-in method all of Tensor object at 0x120a01ae0> = tensor([0.], grad_fn=<SumBackward1>) > 0.all E    +        where tensor([0.], grad_fn=<SumBackward1>) = <built-in method sum of Tensor object at 0x1209e6300>( |
| `test/test_generation_lesson.py::test_sentence_generation_lesson_keeps_its_weighted_output_gradient` | call | test/test_generation_lesson.py:28: in test_sentence_generation_lesson_keeps_its_weighted_output_gradient     cost, *_ = BasicModel._sentence_path_cost( bin/Models.py:20911: in _sentence_path_cost     errors.merge(self._grammar_lesson_errors[name], prefix='grammar.',                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^ E   AttributeError: 'types.SimpleNamespace' object has no attribute '_grammar_lesson_errors'. Did you mean: 'grammar_lesson_weight'? |
| `test/test_query_phase_fullgraph.py::test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths` | call | test/test_query_phase_fullgraph.py:40: in test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths     assert int(torch._dynamo.utils.counters["stats"]["unique_graphs"]) == 1 E   assert 2 == 1 E    +  where 2 = int(2) |
| `test/test_sentence_compose.py::test_disabled_sentence_prediction_leaves_adam_momentum_unused` | call | test/test_sentence_compose.py:223: in test_disabled_sentence_prediction_leaves_adam_momentum_unused     cost, *_ = BasicModel._sentence_path_cost( bin/Models.py:20922: in _sentence_path_cost     prediction_errors, contrast_errors = disc._sentence_prediction_errors                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ E   AttributeError: 'types.SimpleNamespace' object has no attribute '_sentence_prediction_errors'. Did you mean: 'sentence_prediction_cost'? |
| `test/test_sentence_compose.py::test_legacy_event_reporting_does_not_zero_the_sentence_objective` | call | test/test_sentence_compose.py:252: in test_legacy_event_reporting_does_not_zero_the_sentence_objective     assert model._d3_active and not model.reconstruct_in_loop E   assert (False) E    +  where False = BasicModel(\n  (loss): ModelLoss(\n    (output_criterion): MSELoss()\n  )\n  (inputSpace): InputSpace(\n    (subspace): Sub...es=29, out_features=16, bias=False)\n  )\n  (question_conditioner): Linear(in_features=29, out_features=16, bias=False)\n)._d3_active |
| `test/test_tied_operator_reconstruction.py::test_word_scoring_honors_existing_nul_termination` | call | test/test_tied_operator_reconstruction.py:37: in test_word_scoring_honors_existing_nul_termination     torch.testing.assert_close(unknown, torch.tensor([256.]).log()) E   AssertionError: Tensor-likes are not close! E    E   Mismatched elements: 1 / 1 (100.0%) E   Greatest absolute difference: 5.545177459716797 at index (0,) (up to 1e-05 allowed) E   Greatest relative difference: 1.0 at index (0,) (up to 1.3e-06 allowed) |
| `test/test_tied_operator_reconstruction.py::test_byte_targets_survive_whole_word_and_prefix_promotion[alphabet]` | call | test/test_tied_operator_reconstruction.py:447: in test_byte_targets_survive_whole_word_and_prefix_promotion     torch.testing.assert_close(cost, torch.tensor([256.]).log()) E   AssertionError: Tensor-likes are not close! E    E   Mismatched elements: 1 / 1 (100.0%) E   Greatest absolute difference: 5.545177459716797 at index (0,) (up to 1e-05 allowed) E   Greatest relative difference: 1.0 at index (0,) (up to 1.3e-06 allowed) |
| `test/test_tied_operator_reconstruction.py::test_byte_targets_survive_whole_word_and_prefix_promotion[alph]` | call | test/test_tied_operator_reconstruction.py:447: in test_byte_targets_survive_whole_word_and_prefix_promotion     torch.testing.assert_close(cost, torch.tensor([256.]).log()) E   AssertionError: Tensor-likes are not close! E    E   Mismatched elements: 1 / 1 (100.0%) E   Greatest absolute difference: 5.545177459716797 at index (0,) (up to 1e-05 allowed) E   Greatest relative difference: 1.0 at index (0,) (up to 1.3e-06 allowed) |
| `test/test_tied_reconstruction_objective.py::test_student_checkpoint_migrates_shared_weights_and_adam_by_name` | call | test/test_tied_reconstruction_objective.py:75: in test_student_checkpoint_migrates_shared_weights_and_adam_by_name     assert student is not None E   assert None is not None |

## The 25 profiled cases on final source

Final values are pytest call times from this sweep, without cProfile. A missing call is shown as unavailable, with its actual skip/process outcome; the earlier cProfile figures remain separately preserved.

| Current case | Prior sweep seconds | Trim profile seconds | Final call seconds | Final outcome |
|---|---:|---:|---:|---|
| `test/test_interleave_schedule.py::test_native_interleave_supplies_context_then_reads_the_same_sentences[True]` | 1801.036 | 2.727 | 1.669 | passed |
| `test/test_negative_expectation.py::test_native_unlabelled_batch_trains_the_same_chooser` | 1628.005 | 1.306 | 0.921 | passed |
| `test/test_compiled_word_chunk.py::test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks` | 1390.340 | 1059.741 | 849.321 | passed |
| `test/test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer` | 1150.582 | 2.787 | 3.099 | passed |
| `test/test_output_walk.py::test_runbatch_does_not_train_generate_policy_without_supplied_answers` | 1030.009 | 6.060 | 5.802 | passed |
| `test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser` | 917.033 | 1.550 | 1.068 | passed |
| `test/test_output_walk.py::test_question_conditioner_optimizer_steps_both_widths` | 902.683 | 2.306 | 2.756 | passed |
| `test/test_arithmetic_isolation.py::test_supervised_update_cannot_use_exact_arithmetic_or_fallback_codes` | 902.172 | 1.700 | 1.847 | passed |
| `test/test_output_walk.py::test_runbatch_generate_policy_masks_rows_without_supplied_answers` | 881.906 | 2.301 | 2.351 | passed |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[1.0]` | 878.732 | 2.353 | 2.586 | passed |
| `test/test_output_path_supervised.py::test_mixed_supplied_numeric_and_automatic_text_trains_only_supplied_row` | 878.152 | 2.015 | 2.173 | passed |
| `test/test_output_walk.py::test_runbatch_trains_generate_policy_only_with_nonzero_weight[0.0]` | 873.612 | 2.356 | 3.208 | passed |
| `test/test_gradient_factorization.py::test_normal_batch_logs_named_shared_operator_gradients` | 777.435 | 1.927 | 1.419 | passed |
| `test/test_surface_grammar.py::test_normal_batch_trains_supplied_grammar_lessons` | 770.411 | 1.814 | 1.221 | passed |
| `test/test_expectation_defaults.py::test_native_future_and_other_row_changes_do_not_change_first_estimate` | 628.543 | 2.695 | 1.850 | passed |
| `test/test_sentence_compose.py::test_real_packed_ends_train_before_the_next_sentence[True]` | 595.938 | 599.057 | 136.417 | passed |
| `test/test_meronomy_ladder.py::test_utility_counts_accrue_once_per_presentation` | 556.259 | 0.001 | 0.000 | passed |
| `test/test_meronomy_ladder.py::test_epoch_report_carries_the_word_unit_fraction` | 550.427 | 0.001 | 0.000 | passed |
| `test/test_joint_objectives.py::test_real_intermediate_and_final_ends_have_same_canonical_roles` | 464.307 | 1.321 | 1.069 | passed |
| `test/test_output_walk.py::test_packed_recall_observes_captured_sentence_programs` | 432.954 | 1.377 | 1.269 | passed |
| `test/test_compiled_word_chunk.py::test_no_grad_fallback_retains_eager_stm_depth_semantics` | 406.261 | 0.319 | 0.201 | passed |
| `test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[True]` | 375.244 | 1.716 | 2.031 | passed |
| `test/test_sentence_comparison.py::test_sentence_trials_keep_their_own_perception_pullbacks` | 328.771 | 1.279 | 0.876 | passed |
| `test/test_sentence_compose.py::test_legacy_event_reporting_does_not_zero_the_sentence_objective` | 314.127 | 1.345 | 0.472 | failed |
| `test/test_expectation_defaults.py::test_native_runtime_reports_pairs_without_accumulating_or_updating` | 307.798 | 1.479 | 1.212 | passed |

## Longest completed calls on final source

These are observed call times, without an in-process profiler. They exclude setup, collection and unfinished calls; process stops remain in the failure table. The full 25-case list and worker elapsed sum are in `final-runtime-hotspots.json`. Two long cases were also externally stack-sampled for one second each while running; their elapsed times include that observation. See `full-sweep/stack-samples/README.md`.

| Case | Call seconds | Outcome |
|---|---:|---|
| `test/test_compiled_word_chunk.py::test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks` | 849.321 | passed |
| `test/test_expectation_defaults.py::test_expectation_off_keeps_every_packed_observation_in_ltm` | 767.686 | passed |
| `test/test_selected_relation_meaning.py::test_forward_anchor_capture_survives_text_and_grammar_changes[False]` | 685.560 | passed |
| `test/test_output_percept_readout.py::test_model_checkpoint_preserves_readout_and_adam[False]` | 634.866 | passed |
| `test/test_selected_relation_meaning.py::test_forward_anchor_capture_survives_text_and_grammar_changes[True]` | 607.013 | passed |
| `test/test_word_interpretation.py::test_every_serial_word_runs_interpret_before_composition` | 434.759 | passed |
| `test/test_output_walk.py::test_held_answer_idea_ignores_later_staging_and_memory_context` | 419.350 | passed |
| `test/test_output_percept_readout.py::test_model_checkpoint_preserves_readout_and_adam[True]` | 379.856 | passed |
| `test/test_interleave_lifecycle.py::test_interleave_reads_native_context_before_any_serial_word[False]` | 358.392 | passed |
| `test/test_interleave_lifecycle.py::test_interleave_reads_native_context_before_any_serial_word[True]` | 345.343 | passed |
