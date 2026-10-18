"""Central RUN_SLOW gate, including historical decorators and MPS opt-ins."""
import ast
from functools import lru_cache
import inspect
import os
import textwrap
import pytest


# Exact parameterized cases from Testing.md, October 16 tier decision.
WEEKLY_CONTRACTS = frozenset({
    'test/test_math_chain_section14.py::test_ordinary_initial_binding_distribution_includes_every_retained_candidate',
    'test/test_trial_policy_ownership.py::test_trial_and_batch_answer_leave_the_decoder_to_reconstruction[True-explore]',
    'test/test_trial_policy_ownership.py::test_trial_and_batch_answer_leave_the_decoder_to_reconstruction[True-exploit]',
    'test/test_math_chain_repair.py::test_ordinary_driver_two_three_eight_stream_groups_after_single_write',
    'test/test_output_walk.py::test_walk_compiles_fullgraph_and_matches_eager',
    'test/test_bounded_test_recycling.py::test_time_pressure_recycles_after_complete_cases_without_dropping_coverage',
    'test/test_bounded_test_recycling.py::test_live_worker_receipt_keeps_completed_cases_when_a_later_case_times_out',
    'test/test_math_chain_repair2.py::test_eight_corpus_sentences_train_at_distinct_fork_rounds',
    'test/test_arithmetic_isolation.py::test_supervised_update_cannot_use_exact_arithmetic_or_fallback_codes',
    'test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[True]',
    'test/test_output_walk.py::test_output_ignores_input_derivation_on_conceptual_slots',
    'test/test_interleave_schedule.py::test_one_native_read_retains_shared_owner[True]',
    'test/test_compiled_operator_diagonal.py::test_free_candidate_search_compiles_backward_through_native_operators',
    'test/test_tied_operator_reconstruction.py::test_snapshot_owns_values_across_same_shape_dictionary_update',
    'test/test_sentence_journal.py::test_no_numerical_frames_without_a_sentence_closing_reader',
    'test/test_negative_expectation.py::test_native_nonzero_composition_is_bit_identical_at_every_stance',
    'test/test_category_em_smoke.py::test_mlp_chooser_sized_and_context_built',
    'test/test_echoic_decoder.py::test_same_walk_infers_binary_and_unary_without_a_journal[True]',
    'test/test_compiled_word_chunk.py::test_tensor_peer_word_loop_rejects_external_full_stm',
    'test/test_category_em_smoke.py::test_codebook_enables_and_em_populates_on_real_model',
    'test/test_bounded_test_recycling.py::test_recycle_on_memory_pressure_keeps_every_case_once',
    'test/test_output_path_supervised.py::test_legacy_parity_memory_cannot_change_an_owned_conceptual_answer[development]',
    'test/test_negative_expectation.py::test_native_unlabelled_batch_trains_the_same_chooser',
    'test/test_output_path_supervised.py::test_native_checkpoint_restores_active_answer_widths[False-development]',
    'test/test_prepared_answer_boundary.py::test_prepared_concept_handoff_cuts_answer_state_gradient[True-False]',
    'test/test_arithmetic_isolation.py::test_numeric_and_renamed_sentence_paths_keep_oracles_out[texts0]',
    'test/test_one_attention.py::test_narrowing_uses_the_shared_operation_chooser_and_compiles',
    'test/test_bounded_test_runner.py::test_real_pytest_coverage_fresh_workers_and_failure_receipt',
    'test/test_unified_thought_controller.py::test_runbatch_credits_each_controller_row_from_its_own_answer',
    'test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser[tie]',
    'test/test_input_byte_addresses.py::test_promoted_percepts_keep_their_original_byte_addresses[stream_smoke]',
    'test/test_expectation_defaults.py::test_native_future_and_other_row_changes_do_not_change_first_estimate',
    'test/test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser[distinct]',
    'test/test_query_native_ids.py::test_real_sentence_capture_owns_native_object_ids_across_later_staging',
    'test/test_compiled_word_chunk.py::test_peer_leg_profiler_executes_real_ps_ws_and_conceptual_bodies',
    'test/test_review14_contracts.py::test_review17_measurement_wiring_on_one_ordinary_batch',
    'test/test_arithmetic_isolation.py::test_numeric_and_renamed_sentence_paths_keep_oracles_out[texts1]',
    'test/test_ltm_consolidation.py::TestSerialForwardAppendsStore::test_forward_grows_store_not_deque',
    'test/test_output_walk.py::test_question_conditioner_optimizer_steps_both_widths',
    'test/test_output_walk.py::test_packed_recall_observes_captured_sentence_programs',
    'test/test_bounded_test_runner.py::test_parallel_workers_share_the_aggregate_budget_and_overlap',
    'test/test_training_device.py::test_all_dispatches_explicit_device_marks_and_keeps_slow_proofs_on_cpu',
    'test/test_decoder_exploration.py::test_single_departure_replays_prefix_and_is_greedy_after_it[True]',
    'test/test_joint_objectives.py::test_real_intermediate_and_final_ends_have_same_canonical_roles',
    'test/test_generation_catalog.py::test_normal_supervised_output_respects_gradient_contract[False]',
    'test/test_training_device.py::test_parallel_pool_keeps_one_accelerator_training_lane',
    'test/test_relation_closing.py::test_unaligned_mm_forward_can_select_a_relation_without_native_word_rows',
    'test/test_stored_idea_generation.py::test_compound_candidates_are_not_word_emissions[True]',
    'test/test_review11_contracts.py::test_supported_compound_cannot_stop_even_with_a_chunk_code[True-False]',
    'test/test_bounded_test_runner.py::test_test_failure_does_not_cancel_remaining_worker_coverage',
    'test/test_gradient_factorization.py::test_normal_batch_logs_owned_optimizer_gradients',
    'test/test_ltm_consolidation.py::TestRuntimeUserIngestion::test_user_rows_coexist_with_conversation_rows',
    'test/test_surface_grammar.py::test_normal_batch_trains_supplied_grammar_lessons',
})

DEFAULT_CERTIFICATES = frozenset({
    'test/test_item55_mechanisms.py::test_clock_independence',
    'test/test_item55_mechanisms.py::test_offset_independence',
    'test/test_item55_mechanisms.py::test_stream_chain',
    'test/test_item55_mechanisms.py::test_interleaving',
    'test/test_item55_mechanisms.py::test_preposition_idempotence',
    'test/test_item55_mechanisms.py::test_surface_inverses',
    'test/test_item55_mechanisms.py::test_nested_wholes',
    'test/test_item55_mechanisms.py::test_absences',
    'test/test_math_chain.py::test_native_worked_document_trains_unforced_without_numeric_oracles',
    'test/test_math_chain_section14.py::test_forced_ordinary_answer_fills_committed_question_without_its_own_episode',
    'test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[True]',
    'test/test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[False]',
    'test/test_math_chain_section14.py::test_forced_ordinary_bound_declaratives_open_no_episode',
    'test/test_math_chain.py::test_zero_budget_loads_and_trains_the_ordinary_vp_without_attention',
    'test/test_learning_readiness.py::test_native_progress_and_trained_artifact_evaluation_path',
    'test/test_grounded_xor.py::test_native_unseeded_xor[0-8]',
    'test/test_grounded_xor.py::test_native_unseeded_xor[2-8]',
    'test/test_grounded_xor.py::test_native_unseeded_xor[1-8]',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-1]',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-0]',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[8-2]',
    'test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[1]',
    'test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0]',
    'test/test_concept_output.py::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[2]',
    'test/test_math_chain_section14.py::test_forced_c_e_closings_exercise_empty_search_mint_and_question_storage',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-0]',
    'test/test_grounded_xor.py::test_native_unseeded_xor[2-4]',
    'test/test_grounded_xor.py::test_native_unseeded_xor[0-4]',
    'test/test_grounded_xor.py::test_native_unseeded_xor[1-4]',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-2]',
    'test/test_word_admission.py::test_33_grounded_xor_with_word_boundary[4-1]',
})


@lru_cache(maxsize=None)
def _switch_gated(obj, decorators_only=False):
    """Recognize the existing gate itself, never a prose mention of slow tests."""
    if obj is None:
        return False
    try:
        tree=ast.parse(textwrap.dedent(inspect.getsource(obj)))
    except (OSError,TypeError,IndentationError,SyntaxError):
        return False
    definition=tree.body[0]
    nodes=list(getattr(definition,'decorator_list',()))
    if not decorators_only:
        nodes.extend(n.test for n in ast.walk(definition) if isinstance(n,ast.If))
    for outer in nodes:
        for node in ast.walk(outer):
            if isinstance(node,ast.Name) and node.id in ('RUN_SLOW','_RUN_SLOW','RUN_MPS_SLOW'):
                return True
            if isinstance(node,ast.Call) and node.args and isinstance(node.args[0],ast.Constant):
                function=node.func
                if (isinstance(function,ast.Attribute) and function.attr in ('getenv','get')
                        and node.args[0].value in ('RUN_SLOW','RUN_MPS_SLOW')):
                    return True
    return False


def is_slow(item):
    if item.nodeid in DEFAULT_CERTIFICATES or item.get_closest_marker('learning_certificate'):
        return False
    if item.nodeid in WEEKLY_CONTRACTS:
        return True
    return (item.get_closest_marker('slow') is not None
            or any('RUN_SLOW' in str(marker.kwargs.get('reason',''))
                   for marker in item.iter_markers('skipif'))
            or _switch_gated(getattr(item,'obj',None))
            or _switch_gated(getattr(item,'cls',None), True))


def pytest_configure(config):
    config.addinivalue_line(
        'markers','slow: long training/quality/compilation check; opt in with RUN_SLOW=1')
    config.addinivalue_line('markers', 'device(name): explicitly select a bounded worker device')
    config.addinivalue_line('markers', 'learning_certificate: default-tier learning gate regardless of runtime')
    config.addinivalue_line(
        'markers', 'shared_training(name): keep selected consumers of a module fixture in one worker')


def pytest_collection_modifyitems(items):
    enabled=os.environ.get('RUN_SLOW')=='1'
    skipped=pytest.mark.skip(reason='slow -- set RUN_SLOW=1')
    for item in items:
        if item.nodeid in WEEKLY_CONTRACTS:
            item.add_marker(pytest.mark.slow)
        if is_slow(item):
            marked=item.get_closest_marker('slow') is not None
            if not marked:
                item.add_marker(pytest.mark.slow)
            # Historical gates keep their own exact opt-in conditions (for
            # example RUN_MPS_SLOW alone). Add metadata, not a second gate.
            if not enabled and marked:
                item.add_marker(skipped)
