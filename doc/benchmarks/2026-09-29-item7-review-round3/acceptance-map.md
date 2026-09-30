# Item 7, September 29 acceptance map

The numbers are the spec's tests 22–34. The final explicit receipt reruns
these on the same source as the measurements and full sweep. Stage A and B
receipts preserve their intermediate outcomes separately.

| Spec test | Permanent coverage |
|---|---|
| 22 — new word replaces one inventory row, one DEF row | `test_22_new_word_replaces_one_inventory_row_and_writes_one_definition` in [definitions](../../../test/test_item7_definitions.py) |
| 23 — fused parts and changed properties remain one word | The two `test_23_*` cases in [word admission](../../../test/test_item7_word_admission.py) |
| 24 — both directions, no scans, checkpoint rebuild | `test_24_index_is_bidirectional_and_rebuilds_from_saved_rows` in [definitions](../../../test/test_item7_definitions.py) |
| 25 — ambiguity and synonyms | `test_25_two_objects_require_a_selection_and_synonyms_have_a_reverse_index` in [definitions](../../../test/test_item7_definitions.py) |
| 26 — inventory or store capacity refuses atomically | `test_26_*` in [word admission](../../../test/test_item7_word_admission.py) and [definitions](../../../test/test_item7_definitions.py); pending-reservation coverage in [integrity](../../../test/test_item7_definition_integrity.py) |
| 27 — fixed `.when`, refreshed recency | `test_27_definition_has_a_fixed_when_and_refreshes_recency_without_append` in [definitions](../../../test/test_item7_definitions.py) |
| 28 — forgotten definition disappears from all lookups | `test_28_forgotten_definition_does_not_survive_in_any_lookup` in [definitions](../../../test/test_item7_definitions.py); decoder coverage in [integrity](../../../test/test_item7_definition_integrity.py) |
| 29 — only read words publish symbols | The shared-whole probe and three unchanged configurations in `test_29_*` in [word admission](../../../test/test_item7_word_admission.py) |
| 30 — every XOR proof | All nine groups in [the explicit XOR runner](run_xor.py), including both XOR_exact CLI gates and all three slow MM cases; XOR_grammar remains a recorded non-condition |
| 31 — no extra configuration rows | Three `test_31_*` cases in [definitions](../../../test/test_item7_definitions.py), plus [configuration hashes](constraints-audit.json) |
| 32 — actual HEAD META checkpoint migration | `test_32_head_meta_checkpoint_migrates_to_definitions` in [definitions](../../../test/test_item7_definitions.py), plus retired-identity and inventory checks in [integrity](../../../test/test_item7_definition_integrity.py) |
| 33 — words exist before field cases are discovered | All six `test_33_grounded_xor_with_word_boundary` cases in [word admission](../../../test/test_item7_word_admission.py); each asserts the four word concepts exist before discovery and records unrelated percept events |
| 34 — definition identity survives code changes | `test_34_learning_codes_cannot_change_definition_identity` in [definitions](../../../test/test_item7_definitions.py) |

The [final port ledger](final-test-ports.json) retains the earliest captured
old body and the actual final body for each port, rename and retirement.
The [final protected-source audit](final-protected-audit.json) covers the unchanged
numerical gates and deferred NonLayer / ConjunctionLayer methods. The
[configuration and seed audit](constraints-audit.json) is separate from test
outcomes.
