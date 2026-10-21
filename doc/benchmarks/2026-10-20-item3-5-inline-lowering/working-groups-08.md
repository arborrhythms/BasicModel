# Retained binding versus working identity group

Source `67c5b203f34e4299703d4c7801b34e06c1045eb6`. Both unchanged tests fail.
The records below were captured from the actual commit and read calls; no
working read was inserted into model execution.

## test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them

| Occurrence | Operation | Selected at close | Raw derivation | Content support | Working group |
|---|---|---|---|---|---|
| `-6997856705104047482` | mint | `['ltm', 'address-v1', -6997856705104047482]` | `['sym', 2]` | `[0]` | `('ltm', 'address-v1', -6997856705104047482)` |
| `-2907652584967542766` | mint | `['ltm', 'address-v1', -2907652584967542766]` | `['sym', 5]` | `[0]` | `('ltm', 'address-v1', -2907652584967542766)` |
| `-8286165563797862330` | bind | `['ltm', 'address-v1', -6997856705104047482]` | `['sym', 2]` | `[0, 4]` | `('ltm', 'address-v1', -8286165563797862330)` |
| `1934400735842808986` | bind | `['ltm', 'address-v1', -2907652584967542766]` | `['sym', 5]` | `[0, 4]` | `('ltm', 'address-v1', 1934400735842808986)` |

[Full captured reads](working-groups-08-evidence/test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them.json).

## test_revision_merges_three_accessible_groups_transitively_across_cued_documents

| Occurrence | Operation | Selected at close | Raw derivation | Content support | Working group |
|---|---|---|---|---|---|
| `5059024907979507958` | mint | `['ltm', 'address-v1', 5059024907979507958]` | `['sym', 2]` | `[0]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `-2847588295512550313` | bind | `['ltm', 'address-v1', 5059024907979507958]` | `['sym', 2]` | `[0, 4]` | `('ltm', 'address-v1', -2847588295512550313)` |
| `-4098209026200192379` | mint | `['ltm', 'address-v1', -4098209026200192379]` | `['sym', 4]` | `[0]` | `('ltm', 'address-v1', -4098209026200192379)` |
| `-3408910645460889097` | bind | `['ltm', 'address-v1', -4098209026200192379]` | `['sym', 4]` | `[0, 4]` | `('ltm', 'address-v1', -3408910645460889097)` |
| `-3433510135052858015` | mint | `['ltm', 'address-v1', -3433510135052858015]` | `['sym', 6]` | `[0]` | `('ltm', 'address-v1', -3433510135052858015)` |
| `4576792239911004056` | bind | `['ltm', 'address-v1', -3433510135052858015]` | `['sym', 6]` | `[0, 4]` | `('ltm', 'address-v1', 4576792239911004056)` |
| `6694524296269860842` | mint | `['ltm', 'address-v1', 6694524296269860842]` | `['sym', 8]` | `[0, 4]` | `inaccessible` |

[Full captured reads](working-groups-08-evidence/test_revision_merges_three_accessible_groups_transitively_across_cued_documents.json).
