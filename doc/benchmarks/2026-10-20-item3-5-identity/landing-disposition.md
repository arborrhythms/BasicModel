# Landing sweep: failure dispositions

One full sweep completed 5,926/5,926 cases: 5,588 passed, 15 failed, 322 skipped and one XPASS. No timeout is a pass. The original failed sweep is retained. Eleven of its failed cases pass after scoped repairs; four remain failed. A further §14 case fails in final scoped validation. These are mechanism gates, not learning grades.

## 1. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_definition_rows.py::test_22_supplied_object_keeps_its_form_row_and_writes_one_definition` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 2. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_item4_review.py::test_native_activated_word_is_observed_by_handle_with_both_lanes` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 3. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_item6_2_thinking.py::test_event_implication_reads_endpoint_when_order[positions1-False]` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 4. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_item55_mechanisms.py::test_clock_independence` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 5. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_item55_mechanisms.py::test_stream_chain` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 6. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_item55_mechanisms.py::test_nested_wholes_from_stage3_artifact` |
| Sweep | Failed |
| Latest scoped outcome | **failed** in [strict-artifacts-final](strict-artifacts-final/result.json) |
| Disposition | [Strict rejection retained; migration decision required](migration-hold.md). |

## 7. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_math_chain.py::test_native_worked_document_trains_unforced_without_numeric_oracles` |
| Sweep | Failed |
| Latest scoped outcome | **failed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | Cycle eligibility is repaired; a [live cube violation remains](live-cube.md). |

## 8. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_operators_round4a0.py::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 9. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_operators_round4a0.py::test_checkpoint_can_grow_capacity_without_changing_an_address` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 10. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_parts_of_speech.py::test_artifact_restoration_retains_weights_and_occupation_state_exactly[np_definite]` |
| Sweep | Failed |
| Latest scoped outcome | **failed** in [strict-artifacts-final](strict-artifacts-final/result.json) |
| Disposition | [Strict rejection retained; migration decision required](migration-hold.md). |

## 11. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_parts_of_speech.py::test_artifact_restoration_retains_weights_and_occupation_state_exactly[np_description]` |
| Sweep | Failed |
| Latest scoped outcome | **failed** in [strict-artifacts-final](strict-artifacts-final/result.json) |
| Disposition | [Strict rejection retained; migration decision required](migration-hold.md). |

## 12. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_sentence_compose.py::test_real_packed_ends_train_before_the_next_sentence[False]` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 13. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_sentence_identity.py::test_mm_grammar_part_under_disjunction_closes_at_full_inventory` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 14. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_stored_idea_generation.py::test_type_family_snapshot_uses_learned_membership_and_row_local_priming` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 15. Does the failed certificate now pass?

| Measurement | Result |
|---|---|
| Certificate | `test/test_stream_state_shapes.py::test_attention_gist_is_invariant_to_other_batch_rows` |
| Sweep | Failed |
| Latest scoped outcome | **passed** in [sweep-repair-09](sweep-repair-09/result.json) |
| Disposition | [Independent repair and fixture disposition](causes.md). |

## 16. Does the whole §14 file pass on the final candidate?

| Measurement | Result |
|---|---|
| Blocking certificate | `test/test_math_chain_section14.py::test_ordinary_initial_binding_distribution_includes_every_retained_candidate` |
| Final outcome | **failed** in [native-final](native-final/result.json) |
| Cause | The initialization read reaches non-cube request content before the distribution measurement. No optimizer was supplied to that read. |
| Disposition | [Live cube gate remains open](live-cube.md); the earlier pass is not promoted and the assertion/band are unchanged. |
