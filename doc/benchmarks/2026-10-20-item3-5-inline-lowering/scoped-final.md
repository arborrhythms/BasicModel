# Scoped certificate set on f9faae610

Source `f9faae610f35d5ca660d0ac49249637cb23e7adf`.
Exactly one requested scoped set and both decomposition arms; no full sweep.
This is the pre-carrier-repair source, not the later `67c5b203f`.

| File / gate | Passed | Failed | Skipped |
|---|---:|---:|---:|
| `test/test_identity_rows.py` | 9 | 5 | 0 |
| `test/test_operators_final_b.py` | 1 | 0 | 0 |
| `test/test_bind_thought.py` | 15 | 29 | 0 |
| `test/test_reference_close.py` | 13 | 9 | 0 |
| `test/test_math_chain_section14.py` | 12 | 0 | 1 |
| `test/test_reference_mentions.py` | 4 | 0 | 0 |
| `test/test_part_store.py` | 4 | 0 | 0 |
| `test/test_identity_cube.py` | 12 | 0 | 0 |
| Decomposition, zero, both lengths | 2 | 0 | 0 |
| Decomposition, enabled, both lengths | 2 | 0 | 0 |

Native total: **70 passed, 43 failed, one copular prerequisite skip**.
Decomposition: **four certificates pass; all eight companion questions have
correct bindings and proof chains** ([one table per question](decomposition-final-questions.md)).
The paired zero-cost reconstruction invariant passes.

Every native failure is retained below. No assertion beyond D2 and the three
declared reference-close ports was changed. The ordinary arrival certificates
also pass here, separately from their twelve fresh passes.

## test/test_identity_rows.py::test_individual_axes_in_the_change_residual_exclude_form_and_its_gradient

```text
test/test_identity_rows.py:59: in test_individual_axes_in_the_change_residual_exclude_form_and_its_gradient
    residual = resolver.components.innovation(observation, prediction)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/IndependentComponents.py:370: in innovation
    axes = F.normalize(torch.stack([coordinates(self.owner, point(self.owner, cid))
E   RuntimeError: stack expects a non-empty TensorList
```

## test/test_identity_rows.py::test_native_row_is_the_only_content_and_witnesses_are_live_queries

```text
test/test_identity_rows.py:73: in test_native_row_is_the_only_content_and_witnesses_are_live_queries
    assert witness_count(resolver.components.owner,cid) == 1
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/IdentityRows.py:108: in witness_count
    return sum(int(parts.access_count[row]) for row in witnesses(owner, identity))
                                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/IdentityRows.py:103: in witnesses
    return tuple(rows[(parts.refs[rows] == int(identity)).any(-1)].tolist())
                                           ^^^^^^^^^^^^^
E   TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'
```

## test/test_identity_rows.py::test_forgetting_and_reinterpretation_never_refold_a_native_identity

```text
test/test_identity_rows.py:96: in test_forgetting_and_reinterpretation_never_refold_a_native_identity
    torch.testing.assert_close(point(resolver.components.owner,cid),a[0]+a[1])
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/IdentityRows.py:56: in point
    row = owner._csw_row_of(int(identity))
                            ^^^^^^^^^^^^^
E   TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'
```

## test/test_identity_rows.py::test_loader_permanently_rejects_signed_identity_without_modifying_the_row[False]

```text
test/test_identity_rows.py:187: in test_loader_permanently_rejects_signed_identity_without_modifying_the_row
    native = point(resolver.components.owner, identity).clone()
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/IdentityRows.py:56: in point
    row = owner._csw_row_of(int(identity))
                            ^^^^^^^^^^^^^
E   TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'
```

## test/test_identity_rows.py::test_loader_permanently_rejects_signed_identity_without_modifying_the_row[True]

```text
test/test_identity_rows.py:183: in test_loader_permanently_rejects_signed_identity_without_modifying_the_row
    owner.similarity_codebook.W[owner._csw_row_of(identity),0] = -.125
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Spaces.py:15891: in _csw_row_of
    rows = self._csw_rows_of(concept_id)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Spaces.py:15879: in _csw_rows_of
    r = ly.row_of((ns, int(concept_id)))
                       ^^^^^^^^^^^^^^^
E   TypeError: int() argument must be a string, a bytes-like object or a real number, not 'tuple'
```

## test/test_bind_thought.py::test_episode_has_one_bind_per_site_and_kept_write_preserves_np_content

```text
test/test_bind_thought.py:255: in test_episode_has_one_bind_per_site_and_kept_write_preserves_np_content
    assert result.meaning.role_refs[0] == ('sym', f.identities[target])
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...710400551109))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_reference_close.py::test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them

```text
test/test_reference_close.py:145: in test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them
    assert len({identities[a] for a in addresses}) == 1
E   AssertionError: assert 4 == 1
E    +  where 4 = len({('ltm', 'address-v1', -8286165563797862330), ('ltm', 'address-v1', -6997856705104047482), ('ltm', 'address-v1', -2907652584967542766), ('ltm', 'address-v1', 1934400735842808986)})
```

## test/test_reference_close.py::test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[0]

```text
test/test_reference_close.py:170: in test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding
    assert all(isinstance(ref, int) for ref in plan['items'][0]['candidates'])
E   assert False
E    +  where False = all(<generator object test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding.<locals>.<genexpr> at 0x10cc86880>)
```

## test/test_reference_close.py::test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[1]

```text
test/test_reference_close.py:170: in test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding
    assert all(isinstance(ref, int) for ref in plan['items'][0]['candidates'])
E   assert False
E    +  where False = all(<generator object test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding.<locals>.<genexpr> at 0x10cc86ce0>)
```

## test/test_reference_close.py::test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[2]

```text
test/test_reference_close.py:170: in test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding
    assert all(isinstance(ref, int) for ref in plan['items'][0]['candidates'])
E   assert False
E    +  where False = all(<generator object test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding.<locals>.<genexpr> at 0x11fe22180>)
```

## test/test_reference_close.py::test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[3]

```text
test/test_reference_close.py:170: in test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding
    assert all(isinstance(ref, int) for ref in plan['items'][0]['candidates'])
E   assert False
E    +  where False = all(<generator object test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding.<locals>.<genexpr> at 0x11ffef3e0>)
```

## test/test_reference_close.py::test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised

```text
test/test_reference_close.py:184: in test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised
    assert plan['items'][0]['reference'] is None and len(resolver.components.individual_ids) == 1
E   assert (None is None and 0 == 1)
E    +  where 0 = len(())
E    +    where () = IndependentComponents(\n  (verbs): SparseDictionary(\n    (directions): ParameterDict()\n    (relevance): ParameterDict()\n  )\n  (reference_close): ReferenceClose()\n).individual_ids
E    +      where IndependentComponents(\n  (verbs): SparseDictionary(\n    (directions): ParameterDict()\n    (relevance): ParameterDict()\n  )\n  (reference_close): ReferenceClose()\n) = ReferenceClose().components
```

## test/test_reference_close.py::test_revision_merges_three_accessible_groups_transitively_across_cued_documents

```text
test/test_reference_close.py:239: in test_revision_merges_three_accessible_groups_transitively_across_cued_documents
    assert len({identities[a] for a in addresses}) == 1
E   AssertionError: assert 6 == 1
E    +  where 6 = len({('ltm', 'address-v1', -4098209026200192379), ('ltm', 'address-v1', -3433510135052858015), ('ltm', 'address-v1', -3408...ess-v1', -2847588295512550313), ('ltm', 'address-v1', 4576792239911004056), ('ltm', 'address-v1', 5059024907979507958)})
```

## test/test_reference_close.py::test_kept_journal_supplies_np_content_predicate_and_positional_provenance[0]

```text
test/test_reference_close.py:394: in test_kept_journal_supplies_np_content_predicate_and_positional_provenance
    assert identity in component.individual_ids and f.cs._concept_source_order(identity) == 1
E   AssertionError: assert (('ltm', 'address-v1', 7873483260437102423) in ())
E    +  where () = IndependentComponents(\n  (verbs): SparseDictionary(\n    (directions): ParameterDict()\n    (relevance): ParameterDict()\n  )\n  (reference_close): ReferenceClose()\n).individual_ids
```

## test/test_reference_close.py::test_kept_journal_supplies_np_content_predicate_and_positional_provenance[2]

```text
test/test_reference_close.py:394: in test_kept_journal_supplies_np_content_predicate_and_positional_provenance
    assert identity in component.individual_ids and f.cs._concept_source_order(identity) == 1
E   AssertionError: assert (('ltm', 'address-v1', 7873483260437102423) in ())
E    +  where () = IndependentComponents(\n  (verbs): SparseDictionary(\n    (directions): ParameterDict()\n    (relevance): ParameterDict()\n  )\n  (reference_close): ReferenceClose()\n).individual_ids
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[1-3]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[1-0]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[2-3]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[2-0]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[2-1]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[3-0]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[3-1]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[3-2]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open[0-True]

```text
test/test_bind_thought.py:224: in test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...986955241708))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open[1-True]

```text
test/test_bind_thought.py:224: in test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...380650652402))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open[2-True]

```text
test/test_bind_thought.py:224: in test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...626416340373))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open[3-True]

```text
test/test_bind_thought.py:224: in test_g3_four_role_witnesses_open_elision_then_bind_or_remain_open
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...480415548218))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[0-1]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[0-2]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[0-3]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[1-2]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[1-3]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[1-0]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[2-3]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[2-0]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[2-1]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...8626416340373) == ('sym', ('ltm...687836249189))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5383878626416340373
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[3-0]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[3-1]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g1_definite_generator_referent_with_distractor[3-2]

```text
test/test_bind_thought.py:147: in test_g1_definite_generator_referent_with_distractor
    assert actual == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...3480415548218) == ('sym', ('ltm...195129499938))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: 409633480415548218
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[0-1]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[0-2]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[0-3]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...5986955241708) == ('sym', ('ltm...173476468647))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -5015775986955241708
E     Use -v to get more diff
```

## test/test_bind_thought.py::test_g2_generated_leaf_without_object_uses_same_resolver[1-2]

```text
test/test_bind_thought.py:196: in test_g2_generated_leaf_without_object_uses_same_resolver
    assert result.value.role_refs[0] == ('sym', expected)
E   AssertionError: assert ('ltm', 'addr...7380650652402) == ('sym', ('ltm...328310477316))
E     
E     At index 0 diff: 'ltm' != 'sym'
E     Left contains one more item: -1425517380650652402
E     Use -v to get more diff
```
