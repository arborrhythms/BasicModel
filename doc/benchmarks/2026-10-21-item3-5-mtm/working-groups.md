# MTM replay of working-groups-08

The unchanged merge assertions now produce one working group. The transitive
certificate has only its authorized bank-count port; inaccessible rows remain excluded.
MTM entries stay separate when the working read merges them.

## test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them

Outcome: **passed**; 1 accessible working group; 2 retained pool entries.

| Occurrence | Operation | Selected reference | Derivation | Working group |
|---|---|---|---|---|
| `-6997856705104047482` | mint | `['ltm', 'address-v1', -6997856705104047482]` | `['sym', 2]` | `('ltm', 'address-v1', -6997856705104047482)` |
| `-2907652584967542766` | mint | `['ltm', 'address-v1', -2907652584967542766]` | `['sym', 5]` | `('ltm', 'address-v1', -6997856705104047482)` |
| `-8286165563797862330` | bind | `['ltm', 'address-v1', -6997856705104047482]` | `['sym', 2]` | `('ltm', 'address-v1', -6997856705104047482)` |
| `1934400735842808986` | bind | `['ltm', 'address-v1', -2907652584967542766]` | `['sym', 5]` | `('ltm', 'address-v1', -6997856705104047482)` |

[Full commit/read and MTM capture](working-groups-05-evidence/test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them.json).

## test_revision_merges_three_accessible_groups_transitively_across_cued_documents

Outcome: **passed**; 1 accessible working group; 4 retained pool entries.

| Occurrence | Operation | Selected reference | Derivation | Working group |
|---|---|---|---|---|
| `5059024907979507958` | mint | `['ltm', 'address-v1', 5059024907979507958]` | `['sym', 2]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `-2847588295512550313` | bind | `['ltm', 'address-v1', 5059024907979507958]` | `['sym', 2]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `-4098209026200192379` | mint | `['ltm', 'address-v1', -4098209026200192379]` | `['sym', 4]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `-3408910645460889097` | bind | `['ltm', 'address-v1', -4098209026200192379]` | `['sym', 4]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `-3433510135052858015` | mint | `['ltm', 'address-v1', -3433510135052858015]` | `['sym', 6]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `4576792239911004056` | bind | `['ltm', 'address-v1', -3433510135052858015]` | `['sym', 6]` | `('ltm', 'address-v1', 5059024907979507958)` |
| `6694524296269860842` | mint | `['ltm', 'address-v1', 6694524296269860842]` | `['sym', 8]` | `inaccessible` |

[Full commit/read and MTM capture](working-groups-05-evidence/test_revision_merges_three_accessible_groups_transitively_across_cued_documents.json).

