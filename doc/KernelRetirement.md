# Frame-kernel test migration

September 20 item 1 retirement audit against BasicModel `08eebe5`.
`test_thinking_kernel.py` had 67 tests: six remain (five TruthInterval numerical
contracts and the grammar depth campaign), and all 61 removed tests appear below.
“mechanics, dead” names behavior intentionally removed with its controller;
it is not a passing behavioral test. The replacement receipts are recorded in
[Testing](Testing.md#selected-meaning-and-one-controller-september-20).

## Closure status contract

The normal controller preserves two explicit support degrees. At the existing
0.5 posture threshold, conflicting support on both sides is `BOTH`. The old
`mixed` categorical closure is dropped: one supported side yields `TRUE` or
`FALSE`, while the weaker opposing degree remains visible. Both weak sides
remain `UNKNOWN`. A budget cutoff with no supported result is `UNKNOWN` plus
`work_budget` incompleteness and a cutoff record; acquired support is not erased
by later exhaustion. The old `bounded_unknown` label and frame closure enum are
not part of the public result. TruthInterval remains a numerical compatibility
value only. These choices are exercised by the fact-conflict and cutoff probes
below. An early conclude cannot invent a verdict.

Speculative proof execution never materializes a world lemma. Retaining a
completed nested grammatical description is a separate occurrence-write
boundary: embedded content remains unverified or a question, never a fact.

## Every removed test

| Former test | Replacement or disposition |
|---|---|
| `TestLookup.test_stored_idea_is_true` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestLookup.test_empty_store_is_unknown` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestLookup.test_no_chaining` | mechanics, dead — the separate direct-only kernel lookup stage is gone; the canonical taxonomy reader traverses a bounded path. |
| `TestLookup.test_direct_taxonomic_edge` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestLookup.test_negative_world_relation_is_not_taxonomic_evidence` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestLookup.test_conflicting_world_relations_are_not_taxonomic_evidence` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestLookup.test_open_binary_query_is_unknown` | [test_open_part_form_executes_a_taxonomy_set_without_an_alias](../test/test_thought_operation_catalog.py#L255) — Changed contract: an open operand returns a typed set, including an empty set, instead of an unsupported scalar lookup. |
| `TestPart.test_up_and_down` | [test_canonical_form_derives_closed_and_open_roles_without_alias_methods](../test/test_thought_operation_catalog.py#L231) — Typed open-role assignment replaces mode/direction strings. |
| `TestPart.test_mode_and_domain_match_the_actual_source` | [test_canonical_form_derives_closed_and_open_roles_without_alias_methods](../test/test_thought_operation_catalog.py#L231) — Typed open-role assignment replaces mode/direction strings. |
| `TestPart.test_bad_mode_or_direction_raises` | mechanics, dead — mode/direction strings are removed; grammar role/domain validation remains in the checked registry tests. |
| `TestCurriculum.test_depth0_lookup_answer` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestCurriculum.test_depth0_direct_edge` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestCurriculum.test_depth1_syllogism_nested_think` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestCurriculum.test_depth2_two_hops` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestCurriculum.test_world_refutation_does_not_close_taxonomy_false` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestCurriculum.test_dead_end_is_unknown` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestCurriculum.test_isolated_leaf_unknown` | [test_complete_fact_lookup_preserves_conflicting_and_mixed_degrees](../test/test_replacement_kernel_contracts.py#L81) — Restated on the normal controller. |
| `TestFramesAndBudget.test_budget_exhaustion_bounded_unknown` | [test_budget_exhaustion_is_bounded_unknown_with_diagnostic](../test/test_replacement_kernel_contracts.py#L49) — Restated on the normal controller. |
| `TestFramesAndBudget.test_stack_pops_clean` | [test_nested_cutoff_drains_the_actual_depth_without_fresh_work](../test/test_unified_thought_controller.py#L195) — Restated on the normal controller. |
| `TestFramesAndBudget.test_child_returns_result_not_scratch` | [test_subgoal_carries_the_typed_child_result_into_the_answer](../test/test_thought_answer_adapters.py#L53) — Restated on the normal controller. |
| `TestFramesAndBudget.test_cycle_terminates` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestFramesAndBudget.test_unknown_op_raises` | [test_structural_operator_without_thought_declaration_is_not_a_boundary_action](../test/test_thought_operation_catalog.py#L92) — Undeclared operations cannot enter the executable menu. |
| `TestInvariants.test_unsupported_assertion_refused` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Restated on the normal controller. |
| `TestInvariants.test_speculation_does_not_write_ltm` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Restated on the normal controller. |
| `TestInvariants.test_success_without_materialize_flag_does_not_write` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestInvariants.test_taxonomy_derivation_does_not_materialize_a_world_lemma` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestInvariants.test_direct_hit_does_not_rewrite` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestQueryTestimony.test_registered_addressee_returns_testimony` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_unknown_addressee_zero_trust` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_testimony_is_not_truth` | [test_prediction_and_observation_occurrences_cannot_certify_their_content](../test/test_existence_ingestion.py#L16) — Unaccepted external/predicted content has no fact authority; the testimony API itself is gone. |
| `TestQueryTestimony.test_incorporate_above_floor_moves_lookup` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_incorporate_below_floor_refused` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_incorporate_false_testimony_negative_row` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestQueryTestimony.test_legacy_incomplete_relation_cannot_certify_taxonomy` | [test_world_refutation_or_conflict_never_closes_taxonomy_false](../test/test_replacement_kernel_contracts.py#L70) — Restated on the normal controller. |
| `TestQueryTestimony.test_arma_addressee_registered_with_model` | mechanics, dead — the addressee registry, testimony object and incorporation API are removed. Explicit fact admission and the checked prediction operator remain separate contracts. |
| `TestTestimonyInLoop.test_reliable_oracle_grounds_true` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_refuting_oracle_grounds_false` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_unreliable_testimony_stays_unknown` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_testimony_in_loop_does_not_write_ltm` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_tensor_testimony_is_content_not_truth` | [test_typed_prediction_result_is_detached_before_history_or_checkpoint](../test/test_normal_thought_controller.py#L171) — Prediction remains a typed estimate with no truth support. |
| `TestTestimonyInLoop.test_consult_gate_off_skips_query` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestTestimonyInLoop.test_each_addressee_asked_once` | mechanics, dead — there is no automatic oracle consultation or frame-local testimony. Only checked evidence may support an answer; no testimony-derived truth is retained. |
| `TestRewardsAndTraces.test_success_earns_terminal` | mechanics, dead — verifier-derived terminal bonuses are removed; the one policy uses supplied-answer quality and actual work. |
| `TestRewardsAndTraces.test_step_costs_charged` | [test_policy_charges_shared_episode_work_not_the_number_of_choices](../test/test_unified_thought_controller.py#L132) — Restated on the normal controller. |
| `TestRewardsAndTraces.test_valid_unknown_is_not_failure` | [test_dead_end_and_cycle_terminate_unknown_without_speculative_ltm](../test/test_replacement_kernel_contracts.py#L38) — Unknown is valid; no separate verifier success/failure reward exists. |
| `TestRewardsAndTraces.test_unsupported_assertion_is_failure` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Unsupported closure cannot assert true; its old terminal-penalty compiler is removed. |
| `TestRewardsAndTraces.test_trace_examples_only_from_grounded` | mechanics, dead — baseline trace cloning and its exporter are deleted; no imitation labels are manufactured. |
| `TestNextOpPolicy.test_featurize_and_logits_shapes` | [test_normal_controller_policy_sees_all_mandatory_roles_and_gets_credit](../test/test_normal_thought_controller.py#L346) — Restated on the normal controller. |
| `TestNextOpPolicy.test_untrained_head_is_neutral` | [test_untrained_chooser_exactly_reproduces_baseline_at_every_level](../test/test_replacement_kernel_contracts.py#L101) — Restated on the normal controller. |
| `TestNextOpPolicy.test_next_op_loss_trains_the_head` | [test_runbatch_credits_each_controller_row_from_its_own_answer](../test/test_unified_thought_controller.py#L147) — Real optimizer membership and a nonzero parameter update from distinguishable row-local policy loss; no claim of held-out utility. |
| `TestNextOpPolicy.test_empty_examples_returns_none` | [test_policy_masks_other_rows_and_drops_failed_episode_graphs](../test/test_unified_thought_controller.py#L215) — No eligible completed records means no policy loss or baseline update. |
| `TestNextOpPolicy.test_head_prefers_stop_short_circuits` | [test_concluding_without_evidence_cannot_assert_truth](../test/test_replacement_kernel_contracts.py#L59) — Restated on the normal controller. |
| `TestNextOpPolicy.test_head_preferring_explore_keeps_baseline` | [test_untrained_chooser_exactly_reproduces_baseline_at_every_level](../test/test_replacement_kernel_contracts.py#L101) — Restated on the normal controller. |
| `TestModelIntegration.test_knob_parsed` | [test_gates_on](../test/test_reasoning_cde_model.py#L33) — Restated on the normal controller. |
| `TestModelIntegration.test_truthset_provisions_rows` | [test_truthset_provisions_source_rows](../test/test_reasoning_cde_model.py#L39) — Restated on the normal controller. |
| `TestModelIntegration.test_off_returns_none` | [test_public_disabled_budget_never_opens_an_episode](../test/test_replacement_kernel_contracts.py#L115) — Restated on the normal controller. |
| `TestModelIntegration.test_think_about_honest_unknown` | [test_reason_about_returns_honest_posture](../test/test_reasoning_cde_model.py#L105) — Unnamed vectors are rejected before a native taxonomy request is formed. |
| `TestModelIntegration.test_syllogism_over_live_taxonomy` | [test_direct_and_depth_one_two_chains_keep_native_premises](../test/test_replacement_kernel_contracts.py#L23) — Native one-, two- and three-edge proofs; premise ablation and no world-lemma write. Old policy-induced depth is no longer required for a reader path. |
| `TestModelIntegration.test_thinking_loss_head_built_and_in_optimizer` | [test_runbatch_credits_each_controller_row_from_its_own_answer](../test/test_unified_thought_controller.py#L147) — Restated on the normal controller. |
| `TestModelIntegration.test_thinking_policy_loss_graceful` | [test_policy_masks_other_rows_and_drops_failed_episode_graphs](../test/test_unified_thought_controller.py#L215) — Restated on the normal controller. |
| `TestModelIntegration.test_answer_query_kernel_attachment_gates` | [test_public_answer_accepts_the_completed_meaning_without_surface_dispatch](../test/test_unified_thought_controller.py#L36) — One selected summary; no kernel attachment or detector shim. |

## Additional parity-test migration

The full September 20 receipt also exposed two old presentation-loop tests in
`test_what_spacetime.py`. Their causal-child and bounded-return contracts are
restated by `test_child_context_can_refine_before_returning_its_causal_result`
and `test_nested_cutoff_drains_the_actual_depth_without_fresh_work` in
[test_unified_thought_controller.py](../test/test_unified_thought_controller.py).
The old `test_thinking_uses_completed_subquestion_context_and_restores_parity`
and `test_thinking_limit_forces_every_nested_question_closed` now test the
single-presentation wrapper and rejection of an unfinished presentation without
fabricated closure. Its old repeated-forward and forced parity-close mechanics
are dead. Standalone LTM slot/lifecycle tests remain.
