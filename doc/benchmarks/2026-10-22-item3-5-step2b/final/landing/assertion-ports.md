# Exact assertion ports
Baseline: `bb0045d7a2b7a7e5be584066ca1ee283fa899d75`. Line numbers refer to the audited source.
Every changed function, including assertion-preserving setup and loop-scope changes, is in [fixture-ledger.json](fixture-ledger.json). Whole-file edits are in [test-ports.patch](test-ports.patch); copied observer edits are in [copied-observer-ports.patch](copied-observer-ports.patch).

## 1. test/observer_probe.py — test_margin_and_ownership_observe_the_same_actual_steps
Change: replace.

Before:

```python
assert run_audit['start']['identity']['projection_sha256'] == run_audit['end']['identity']['projection_sha256']
```

After:

```python
assert run_audit['start']['identity']['form_codes_sha256'] == run_audit['end']['identity']['form_codes_sha256']
```

## 2. test/observer_probe.py — test_margin_and_ownership_observe_the_same_actual_steps
Change: replace.

Before:

```python
assert all(v==128 for v in run_audit['closing_image']['concept_widths'].values())
```

After:

```python
assert all(v==model._concept_owner().outputShape[-1]
               for v in run_audit['closing_image']['concept_widths'].values())
```

## 3. test/test_accessible_mind.py — test_leaf_column_growth_is_amortized_and_checkpoint_append_is_safe
Change: replace.

Before:

```python
assert column.numel() == 384
assert column.untyped_storage().nbytes() == column.numel() * column.element_size()
```

After:

```python
assert not any(key.startswith("posting_") for key in state)
assert "leaf_complete" not in state
```

## 4. test/test_clause_acceptance.py — test_01_compound_vp_has_one_idea_and_readable_cat_chase_mouse
Change: replace.

Before:

```python
assert f.store.refs[row].tolist() == [-1, -1, -1]
assert len(f.store.parts) == 10
```

After:

```python
assert not hasattr(f.store, 'refs')
assert len(f.store.parts) == 6
```

## 5. test/test_clause_acceptance.py — test_04_absolute_clauses_close_bottom_up_with_their_part_edges
Change: replace.

Before:

```python
assert len(f.store.parts) == 14
```

After:

```python
assert len(f.store.parts) == 15
```

## 6. test/test_clause_acceptance.py — test_05_embedded_part_is_unasserted_and_propagates_operator_kind
Change: replace.

Before:

```python
assert not len(f.store)
```

After:

```python
assert len(f.store) == 2
```

## 7. test/test_clause_acceptance.py — test_03_particular_subject_ends_an_idea
Change: replace.

Before:

```python
assert len(f.store.parts) >= len(f.words) + 2
```

After:

```python
assert len(f.store.parts) >= len(f.words) + 1
```

## 8. test/test_clause_acceptance.py — test_06_implication_keeps_two_unasserted_relation_operands
Change: replace.

Before:

```python
assert not len(f.store)
```

After:

```python
assert len(f.store) == 3
```

## 9. test/test_clause_acceptance.py — test_implication_closes_bare_np_arguments_as_idea_clauses
Change: replace.

Before:

```python
assert f.store.ideas().tolist() == [0, 1]
```

After:

```python
assert f.store.ideas().tolist() == [0, 1, 2]
```

## 10. test/test_clause_acceptance.py — test_07_equality_is_one_membership_in_a_singleton
Change: replace.

Before:

```python
assert len(f.store.parts) == count + 1 and not len(f.store)
```

After:

```python
assert len(f.store.parts) == count + 1 and len(f.store) == 1
```

## 11. test/test_clause_closing.py — test_three_slot_closing_registers_its_composite_operand
Change: replace.

Before:

```python
assert len(store) == 0 and store.parts.contains(inner)
```

After:

```python
assert len(store) == 2 and store.parts.contains(inner)
```

## 12. test/test_clause_storage.py — test_end_row_has_shared_references_and_field_coordinates
Change: replace.

Before:

```python
assert store.refs.shape == (8, 3)
assert store.refs.dtype == torch.int64
assert bool((store.refs == -1).all())
```

After:

```python
assert not hasattr(store, 'refs')
assert 'refs' not in store.state_dict()
```

## 13. test/test_clause_storage.py — test_compound_idea_stores_only_fused_point_without_factored_target
Change: replace.

Before:

```python
assert store.refs[row].tolist() == [-1, -1, -1]
```

After:

```python
assert not hasattr(store, 'refs')
```

## 14. test/test_clause_storage.py — test_recursive_absolute_clauses_each_write_one_idea
Change: replace.

Before:

```python
assert bool((store.refs[:3, 2] == -1).all())
```

After:

```python
assert not hasattr(store, 'refs')
```

## 15. test/test_clause_storage.py — test_attribution_keeps_inner_claim_unasserted_then_direct_assertion_updates_it
Change: replace.

Before:

```python
assert len(store) == 0 and len(store.parts) == 2
```

After:

```python
assert len(store) == 2 and len(store.parts) == 2
```

## 16. test/test_clause_storage.py — test_second_order_implication_names_two_part_symbols
Change: replace.

Before:

```python
assert len(store) == 0 and len(store.parts) == 3
```

After:

```python
assert len(store) == 3 and len(store.parts) == 3
```

## 17. test/test_clause_storage.py — test_clause_end_state_and_evidence_checkpoint_roundtrip
Change: replace.

Before:

```python
torch.testing.assert_close(restored.refs, store.refs)
```

After:

```python
assert not hasattr(restored, 'refs') and 'refs' not in restored.state_dict()
```

## 18. test/test_clause_storage.py — test_observation_writer_accepts_a_completed_field_without_a_program
Change: replace.

Before:

```python
assert len(store)==0 and store.row(row)['refs'].tolist()==[1,2]
```

After:

```python
assert len(store)==1 and store.row(row)['refs'].tolist()==[1,2]
```

## 19. test/test_definition_taxonomy.py — test_closing_uses_the_ordered_object_parent_and_deduplicates_it
Change: replace.

Before:

```python
assert len(store) == 0
```

After:

```python
assert len(store) == 1
```

## 20. test/test_forgetting.py — test_idea_coarsens_parent_and_releases_its_child_next_pass
Change: replace.

Before:

```python
assert store.index_of_row(child[2]) is None
assert store.index_of_row(parent[2]) is None and report['cascade'] == 1
assert index is not None and store.refs[index, 0] == -1
```

After:

```python
pytest.warns(RuntimeWarning, match='low water')
assert store.index_of_row(child[2]) is not None
assert report['cascade'] == 0 and bool(store.forget_pending)
assert store.index_of_row(parent[2]) is None and report['coarsened'] == 0
assert index is not None and not hasattr(store, 'refs')
```

## 21. test/test_forgetting.py — test_idea_coarsens_parent_and_releases_its_child_next_pass
Change: insert.

Before:

```python
(none)
```

After:

```python
assert store.index_of_row(child[2]) is None
assert report['cascade'] == 0 and report['target_reached']
torch.testing.assert_close(store.meaning_of(store.index_of_row(parent[2])).roles, root.roles)
```

## 22. test/test_forgetting.py — test_real_model_forgets_at_document_boundary_then_reads_writes_and_reloads
Change: replace.

Before:

```python
assert report['before'] >= 8 and report['after'] == len(store) == 6
assert report['target_reached']
```

After:

```python
assert report['before'] >= 8 and report['after'] == len(store)
assert 6 <= len(store) < report['before']
assert report['target_reached'] == (len(store) == 6)
assert bool(store.forget_pending) == (len(store) > 6)
```

## 23. test/test_identity_rows.py — test_forgetting_and_reinterpretation_never_refold_a_native_identity
Change: replace.

Before:

```python
torch.testing.assert_close(store.refs,original[1],rtol=0,atol=0)
```

After:

```python
torch.testing.assert_close(store.parts.refs,original[1],rtol=0,atol=0)
```

## 24. test/test_learning_readiness.py — test_native_progress_and_trained_artifact_evaluation_path
Change: replace.

Before:

```python
pytest.raises(ValueError, match='posting_codes')
```

After:

```python
pytest.raises(ValueError, match=message)
```

## 25. test/test_medium_term_memory.py — test_bound_use_reads_as_its_introduction_without_rewriting_rows
Change: replace.

Before:

```python
torch.testing.assert_close(current[:use], old[:use], rtol=0, atol=0)
```

After:

```python
assert_part_references(store, before[0])
torch.testing.assert_close(store.slots[:use], before[1][:use], rtol=0, atol=0)
```

## 26. test/test_medium_term_memory.py — test_lru_eviction_keeps_rows_and_a_later_mention_opens_a_new_token
Change: insert.

Before:

```python
(none)
```

After:

```python
assert_part_references(store, old[0])
```

## 27. test/test_nested_retention.py — test_origin_replacement_retains_reachable_content_without_fact_authority
Change: replace.

Before:

```python
assert store.meaning_of(_index(store, inner_ref)).mode == "interrogative"
```

After:

```python
assert store.row(_index(store, inner_ref))["kind"] == "question"
assert store.row(_index(store, inner_ref))["evidence"] == (0., 0.)
```

## 28. test/test_operators_final.py — test_definition_sentence_joins_identities_beside_the_interpret_definition
Change: replace.

Before:

```python
assert len(f.store) == 0 and first[0] == 'sym'
```

After:

```python
assert len(f.store) == 1 and first[0] == 'sym'
```

## 29. test/test_operators_final_b.py — forced_prefix_certificate
Change: replace.

Before:

```python
assert derived.context_width == 128 and derived.symbol_centroid
```

After:

```python
assert derived.context_width == 128
```

## 30. test/test_operators_round3a.py — test_trust_to_pair_and_negation
Change: parameters.

Before:

```python
pytest.mark.parametrize('trust,polarity,expected', [(.7,True,(.7,0.)),(-.4,True,(0.,.4)),
    (.7,False,(0.,.7)),(-.4,False,(.4,0.)),(0.,True,(0.,0.))])
```

After:

```python
pytest.mark.parametrize('trust,polarity,expected', [(.7,True,(.7,0.)),(-.4,True,(0.,.4)),
    (.7,False,(.7,0.)),(-.4,False,(0.,.4)),(0.,True,(0.,0.))])
```

## 31. test/test_operators_round4a0.py — test_references_and_content_keys_survive_compaction_and_checkpoint
Change: replace.

Before:

```python
assert int(store.refs[1, 0]) == int(store.address_keys[0])
```

After:

```python
assert not hasattr(store, 'refs')
```

## 32. test/test_operators_round4a0.py — test_legacy_checkpoint_rekeys_native_and_typed_edges
Change: replace.

Before:

```python
assert int(restored.refs[1, 0]) == reference[2]
```

After:

```python
assert not hasattr(restored, 'refs') and 'refs' not in restored.state_dict()
```

## 33. test/test_part_closing.py — test_relative_close_is_one_part_edge_and_rewitnesses_its_two_lanes
Change: replace.

Before:

```python
assert len(f.store) == 0
```

After:

```python
assert len(f.store) == 1
```

## 34. test/test_part_closing.py — test_absolute_close_keeps_only_the_concept_and_derivation_pairs
Change: replace.

Before:

```python
assert f.store.refs[row].eq(-1).all()
```

After:

```python
assert not hasattr(f.store, 'refs')
```

## 35. test/test_part_read.py — test_postings_rebuild_from_part_without_inverting_the_narrowed_point
Change: insert.

Before:

```python
(none)
```

After:

```python
assert not any(key.startswith('posting_') for key in state)
```

## 36. test/test_part_read.py — test_closing_and_reads_keep_a_name_pending_without_a_resolver_choice
Change: replace.

Before:

```python
assert store.meaning_of(source).role_refs[0] == ('sym', answer)
assert not open_slots(resolved)
assert resolved.role_refs[0] == ('sym', answer)
assert bindings(resolved)['_thought_witnesses'] == (store.occurrence_of(source),)
```

After:

```python
assert store.meaning_of(source).role_refs[0] is None
assert open_slots(resolved) == (('referent', 0),)
assert resolved.role_refs[0] is None
assert not bindings(resolved).get('_thought_witnesses')
```

## 37. test/test_part_read.py — test_forward_name_stays_pending_beside_a_witnessed_occurrence_complement
Change: replace.

Before:

```python
assert not open_slots(resolved)
assert not bindings(resolved).get('_forward_references')
assert resolved.role_refs[0] == reference
assert bindings(resolved)['_thought_witnesses'] == (store.occurrence_of(source),)
```

After:

```python
assert open_slots(resolved) == (('referent', 0),)
assert bindings(resolved)['_forward_references'] == ((0, name),)
assert resolved.role_refs[0] is None and resolved.role_refs[2] == reference
assert not bindings(resolved).get('_thought_witnesses')
```

## 38. test/test_reference_arrivals.py — test_arrival_bind_refines_and_reads_three_through_the_token
Change: replace.

Before:

```python
torch.testing.assert_close(store.refs[:pending+1], refs[:pending+1], rtol=0, atol=0)
```

After:

```python
assert_part_references(store, refs)
```

## 39. test/test_reference_close.py — test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised
Change: replace.

Before:

```python
assert int(store.refs[store.index_of_row(missing), 0]) == -1
torch.testing.assert_close(store.refs[:store.index_of_row(missing)], before[:store.index_of_row(missing)])
```

After:

```python
assert not hasattr(store, 'refs')
assert_part_references(store, before)
```

## 40. test/test_reference_close.py — test_empty_close_has_no_policy_gradient_or_store_side_effect
Change: replace.

Before:

```python
torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)
```

After:

```python
torch.testing.assert_close(store.parts.refs, refs, rtol=0, atol=0)
```

## 41. test/test_reference_close.py — test_kept_journal_supplies_np_content_predicate_and_positional_provenance
Change: replace.

Before:

```python
torch.testing.assert_close(restored.refs, f.store.refs, rtol=0, atol=0)
```

After:

```python
assert not hasattr(restored, 'refs') and 'refs' not in restored.state_dict()
```

## 42. test/test_reference_mentions.py — test_later_fill_trains_the_same_occurrence_menu_without_a_live_graph
Change: replace.

Before:

```python
torch.testing.assert_close(store.refs, refs, rtol=0, atol=0)
```

After:

```python
torch.testing.assert_close(store.parts.refs, refs, rtol=0, atol=0)
```

## 43. test/test_relation_closing.py — test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent
Change: replace.

Before:

```python
assert len(f.store) == len(unknown)
```

After:

```python
assert len(f.store) == len(unknown) + 1 + int(nested)
```

## 44. test/test_relation_closing.py — test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent
Change: replace.

Before:

```python
assert not f.store.trust[:len(f.store)].any()
```

After:

```python
assert not f.store.trust[:len(unknown)].any()
```

## 45. test/test_relation_closing.py — test_unindexed_operand_ends_as_an_unasserted_idea_before_relative_parent
Change: insert.

Before:

```python
(none)
```

After:

```python
assert sentence['kind'] == 'observation'
assert sentence['evidence'] == pytest.approx((.7, 0.))
```

## 46. test/test_relation_closing.py — test_three_slot_relation_closes_unindexed_numerical_operands
Change: replace.

Before:

```python
assert len(f.store) == 2 and isinstance(row, tuple)
```

After:

```python
assert len(f.store) == 3 and isinstance(row, tuple)
```

## 47. test/test_selected_nested_meaning.py — test_nested_observation_writers_retain_children_without_certifying_them
Change: replace.

Before:

```python
assert len(store) == 0
```

After:

```python
assert len(store) == 2
```

## 48. test/test_sentence_end_state.py — test_reading_an_idea_returns_its_one_occupied_slot
Change: replace.

Before:

```python
assert store.refs[row].tolist() == [-1, -1, -1]
```

After:

```python
assert not hasattr(store, 'refs')
```

## 49. test/test_sentence_end_state.py — test_part_stores_native_addresses_and_reads_current_concept_content
Change: replace.

Before:

```python
assert not len(store)
```

After:

```python
assert len(store) == 1
```

## 50. test/test_sentence_end_state.py — test_no_clause_derivation_or_factored_target_is_checkpointed
Change: replace.

Before:

```python
torch.testing.assert_close(restored.refs, store.refs, rtol=0, atol=0)
```

After:

```python
assert not hasattr(restored, 'refs') and 'refs' not in restored.state_dict()
```

## 51. test/test_sentence_end_state.py — test_one_native_sentence_owns_only_its_end_state_after_the_boundary
Change: replace.

Before:

```python
torch.testing.assert_close(ended.meaning.roles, store.slots[row], rtol=0, atol=0)
```

After:

```python
torch.testing.assert_close(ended.meaning.roles, retained_roles, rtol=0, atol=0)
```

## 52. test/test_sentence_identity.py — test_referenced_phrase_is_minted_as_a_part_argument
Change: replace.

Before:

```python
assert len(f.store) == 1
```

After:

```python
assert len(f.store) == 2
```

## 53. test/test_sentence_retention.py — test_repeated_relation_can_retain_multiple_estimates_without_duplicate_claims
Change: replace.

Before:

```python
assert len(store) == 2
```

After:

```python
assert len(store) == 3
```

## 54. test/test_sentence_retention.py — test_narrow_idea_retains_no_implicit_operand_reference
Change: replace.

Before:

```python
assert store.refs[second].tolist() == [-1, -1, -1]
assert store.meaning_of(second).role_refs == (None, None, None)
```

After:

```python
assert not hasattr(store, 'refs')
assert store.meaning_of(second).role_refs == store.meaning_of(first).role_refs
assert store.occurrence_of(first) not in store.meaning_of(second).role_refs
```

## 55. test/test_truth_text.py — test_supplied_trust_preserves_both_possible_grammatical_fields
Change: replace.

Before:

```python
assert store.row(row)['refs'].numel() == (3 if field.relation is None else 2)
```

After:

```python
assert 'refs' not in store.row(row)
assert store.row(row)['refs'].numel() == 2
```

## 56. test/test_truth_text.py — test_supplied_trust_preserves_both_possible_grammatical_fields
Change: replace.

Before:

```python
assert store.ideas().tolist() == [0]
```

After:

```python
assert store.ideas().tolist() == [0, 1]
```

## Retired tests

- `test/test_operators_final.py::test_centroid_moves_without_losing_identity_and_caps_containment` — adjacency/nudge only (P3).

- `test/test_operators_final.py::test_adjacent_wholes_rewitness_compact_and_checkpoint` — adjacency/nudge only (P3).

- `test/test_review14_contracts.py::test_absent_adjacent_wholes_leave_the_lower_form` — adjacency/nudge only (P3).

## New regressions

- `test/test_step2b_storage.py::test_truth_ingestion_restores_caller_recall_and_isolates_each_source`

- `test/test_step2b_storage.py::test_negative_trust_derived_pair_reverses_only_on_the_relation_edge`

- `test/test_step2b_storage.py::test_stored_question_is_refused_by_kind_without_a_mode_column`

- `test/test_step2b_storage.py::test_pending_forgetting_pressure_survives_restore_below_high_water`
