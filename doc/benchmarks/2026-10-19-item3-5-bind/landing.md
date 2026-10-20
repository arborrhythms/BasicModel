# Step 1 landing checks, 2026-10-20

The accepted mechanism candidate is `f4f11aa5b`. `8b294b3ad` folds the
supplied-answer kind correction and the user's bind-plan §6–§6.2 edits.
The removal of three `answer_mode` assignments in `bin/data.py` was also in
the source tree throughout the sweep and is included in the landing.
There is one full sweep; subsequent runs are scoped repairs and native gates.
All failures remain recorded. These checks make no learning claim.

## Did the complete sweep finish?

| Run | Completed | Passed | Failed | Skipped | XFAIL | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| `answer-kind` | 62/62 | 58 | 4 | 0 | 0 | 237.26 |
| `answer-kind-fixtures` | 24/24 | 24 | 0 | 0 | 0 | 8.75 |
| `landing-sweep` | 5907/5907 | 5536 | 48 | 322 | 1 | 1359.07 |
| `landing-repairs` | 455/455 | 427 | 7 | 21 | 0 | 121.78 |
| `landing-repairs-final` | 55/55 | 54 | 1 | 0 | 0 | 42.14 |

All 5,907 selected cases finished. No timeout is counted as a pass.
The sweep's 48 failed cases are preserved in `landing-sweep-failures.json`;
worker reports include their tracebacks. Case totals count each selected
case once, rather than counting an extra passing teardown as a case.
The original bootstrap paired-reconstruction invariant passed. The usual
model-construction tests exercised the common-constructor PerceptMoments.
The absent copular lesson remains its declared prerequisite skip.

## Does the answer judge follow the supplied answer?

| Question | Repair and evidence |
|---|---|
| Referent or sentence? | `WhatAnswer.kind` owns the distinction. The corpus emits a typed referent target; mixed batches select the referent cost per answer. Ordinary answers retain the surface cost and surface readers retain their training. The data flag is removed. `answer-kind-fixtures` passes all 24 checks, including the old-flag-ignored regression. |

## Does admission preserve forms without inventing objects?

| Question | Cause and repair |
|---|---|
| Can a known form be read without a DEF object? | Native attention, frozen interpretation, the percept inverse and location reuse consulted only DEF rows. They now read the existing admitted-form view, including pending forms. A later word's discovering-field eligibility also excludes those lexical forms. |
| Can a shared whole activate an absent form? | The lexical evidence gate included only DEF objects. It now gates both admitted forms and witnessed objects by the input's actual parts. The absent-word certificates retain their assertions. |
| Why did object fixtures fail? | They assumed every new form silently allocated an object. Tests that require an object now supply a native definition explicitly; bare-reading tests assert there is no invented DEF. |
| Why did the forgetting boundary not run? | The fixture counted four obsolete synthetic DEF rows toward its eight-row threshold. It now supplies eight real observations with the existing high/low watermarks and boundary assertions. |
| Why did the stream-context fixture tie? | Its direct writes lacked the identity-column entry, and the half-strength observation had no exact leaf posting. It now supplies two ordinary clause rows with explicit native identity columns. This is an evaluation fixture, with no extra training. |

## Does a selected antecedent survive the closing journal?

| Question | Cause and repair |
|---|---|
| Can a provisional objectless request override a selected identity? | The early deferred-reference branch skipped an operation's actual selected antecedent. It now defers only while no different native antecedent has been selected. The original public-read/capture/store assertion passes in `landing-repairs`. |

## Can retained artifacts load after deleting the compose bind?

| Question | Cause and repair |
|---|---|
| Do operation weights keep their meaning after a positional deletion? | Grammar checkpoint alignment previously remapped chooser axes but not the operation ModuleLists. It now maps all operation aliases by declared identities, retires removed positions, and carries optimizer moments to the same surviving operation. A focused regression verifies values, aliases, moments and a strict reload. |
| Do occupation counts survive? | Saved formal arguments still included the retired compose operation. The one-time checkpoint alignment retains counts by surviving declared argument, removes retired arguments, and gives new arguments no invented evidence. The new regression verifies exact retained counts. |
| Does the measurement adapter undo a migration? | It attempted to restore the old unscoped resolver-count envelope after the native loader rebuilt role counts. It now checks the unchanged statistics and independently reconstructs the role counts from retained occurrences before retaining the migrated envelope. |

## Which failed sweep cases are covered by the repairs?

| File | Failed cases in the original sweep |
|---|---|
| `test/test_assertion_provenance.py` | `test_provisioning_retains_completed_fields_and_shared_row_ids` |
| `test/test_attention_promotion.py` | `test_interpretation_replaces_the_word_row_without_inventing_a_kind` |
| `test/test_autobind_from_cs.py` | `TestAutobindFiresFromCS::test_autobind_creates_meta_taxonomy_entry` |
| `test/test_concept_capacity_policy.py` | `test_serial_property_autobind_does_not_persist_sentence_chain` |
| `test/test_cs_symbol_table.py` | `test_word_object_testimony_participation_strengthens_on_reoccurrence`; `test_interpret_word_structure`; `test_resolve_identities_does_not_collapse_unspecified_object` |
| `test/test_forgetting.py` | `test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads` |
| `test/test_grammar_object_leaves.py` | `test_grammar_resolves_each_word_to_its_full_object_code[XOR_grammar.xml]`; `test_grammar_resolves_each_word_to_its_full_object_code[MM_grammar.xml]`; `test_grammar_resolves_each_word_to_its_full_object_code[aligned]` |
| `test/test_item55_mechanisms.py` | `test_stream_chain`; `test_surface_inverses`; `test_nested_wholes_from_stage3_artifact` |
| `test/test_lexical_reference_orders.py` | `test_selected_grammar_context_selects_order[event-0]`; `test_selected_grammar_context_selects_order[particular-1]`; `test_selected_grammar_context_selects_order[name-1]`; `test_selected_grammar_context_selects_order[pronoun-1]`; `test_selected_grammar_context_selects_order[kind-2]`; `test_order_three_relation_over_kinds_is_read_from_its_stamp`; `test_form_sets_survive_checkpoint_and_missing_order_stays_unknown` |
| `test/test_live_references.py` | `test_public_reading_fuses_the_reference_before_capture_and_write` |
| `test/test_ltm_consolidation.py` | `TestProvisioning::test_truthset_rows_land_via_real_forward` |
| `test/test_mereology_word_binding.py` | `test_word_spelling_is_ordered_and_binding_is_idempotent`; `test_taxonomy_and_word_resolution_are_native_not_wholespace_forwarders` |
| `test/test_one_attention.py` | `test_native_pole_read_omits_padding_without_changing_evidence` |
| `test/test_operators_round3a.py` | `test_live_mint_rekeys_pending_and_completed_definitions[False]`; `test_live_mint_rekeys_pending_and_completed_definitions[True]`; `test_live_grammar_uses_sparse_index_dense_nonzero_roots_and_exact_rung0` |
| `test/test_order0_attended_field.py` | `test_more_than_eight_words_reuse_the_attended_field` |
| `test/test_parts_of_speech.py` | `test_word_flow_follows_the_catalogue_and_ends_at_a_clause`; `test_artifact_restoration_retains_weights_and_occupation_state_exactly[np_definite]`; `test_artifact_restoration_retains_weights_and_occupation_state_exactly[np_description]` |
| `test/test_review10_contracts.py` | `test_frozen_interpretation_reads_known_without_learning_or_reserving` |
| `test/test_role_collapsed_grammar.py` | `test_operator_coverage`; `test_forward_reverse_pairing` |
| `test/test_sentence_end_state.py` | `test_idea_caches_the_operand_addresses_recorded_before_fusion` |
| `test/test_sentence_protocol.py` | `test_repeated_word_is_not_reminted` |
| `test/test_sparse_concept_e2e.py` | `test_interpreted_object_keeps_the_native_features_in_one_row` |
| `test/test_verb_adverb_grammar_ops.py` | `test_complete_contains_absorbed_role_only_methods` |
| `test/test_word_admission.py` | `test_23_fuse_parts_before_word_admission`; `test_29_shared_whole_does_not_evidence_an_absent_word`; `test_29_each_reading_publishes_only_its_words[MM_xor.xml]`; `test_29_each_reading_publishes_only_its_words[MM_grammar.xml]`; `test_29_each_reading_publishes_only_its_words[XOR_grammar.xml]` |
| `test/test_word_store.py` | `test_word_surface_and_reassociated_object_survive_strict_checkpoint`; `test_missing_surface_snapshot_is_a_staging_error_without_input_fallback`; `test_surface_bytes_belong_to_words_and_objects_follow_their_association` |

## Native close-out

| Run | Completed | Passed | Skipped | Result |
|---|---:|---:|---:|---|
| `stream-context-final` | 1/1 | 1 | 0 | The last affected fixture passes. |
| `native-landing` | 73/73 | 72 | 1 | Whole §14: 12 passed, copular prerequisite skipped; G1 12/12, G2 12/12, G3 8/8; original bootstrap invariant and reference/mention checks pass. |
| `bind-decomposition-zero-landing` | 2/2 | 2 | 0 | Both original lengths pass. |
| `bind-decomposition-enabled-landing` | 2/2 | 2 | 0 | Both original lengths pass. |

[One table per companion question](landing-questions.md) records all eight:
correct bindings, required 1/2 inferences, no remaining open roles, and work
175/178/455/465 in each arm. Both arms retain their original measurement
seeds 4199722147 and 2428440506, all 121 constructor parameters match their
step-1 baselines, and the assertions, forced scope and budgets are unchanged.
No extra training repairs a learning result. The one sweep is not rerun.

The landing includes the current bind-plan §6–§6.3, two-truths row amendment
and todo edits, including the row-audit additions made during this run.
Step 1b follows publication on both mains and starts with G0. Its receipt
must include the §6.3 per-field audit. LTM1/LTM3, the lesson sequence and
other steps 2–6 remain in the plan's order. Item 0's failed native learning
record, graded/shuffled, derivation and long-run collapse measurements,
and the separate 5.5 learning gates remain explicitly open.
