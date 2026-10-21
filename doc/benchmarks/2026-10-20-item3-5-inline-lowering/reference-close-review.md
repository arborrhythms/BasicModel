# Remaining reference-close evidence after the three declared ports

Source `f9faae610`; `closing-01`: 40 cases complete, 31 pass and nine fail.
All 14 PartClosing tests and all four reference-mention tests pass. Three
declared reference-close ports pass. No other assertion has been changed.

| Existing case | Observed failure | Required distinction |
|---|---|---|
| `test_later_common_evidence_merges_accessible_occurrences_without_rewriting_them` | Four occurrence identities instead of one interpreted group | Mechanism: the current grouping/revision path was built around symbol identities; keep the partition and immutable-store assertions |
| `test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding[0..3]` | Candidate assertion requires every reference to be an integer | Four representation ports to native occurrence addresses; preserve every later identity/partition/content assertion |
| `test_bind_without_access_cannot_mint_and_old_rows_cannot_be_revised` | Requires one entry in `components.individual_ids`; new ordinary token has none | Representation port: count the retained token occurrence and retain the no-access/no-mint/unchanged-old-row assertions |
| `test_revision_merges_three_accessible_groups_transitively_across_cued_documents` | Six occurrence identities instead of one; later assertion also requires four symbol individuals | Mechanism first; only the final symbol-count assertion is a proposed representation port; keep transitive merge and inaccessible-row exclusion |
| `test_kept_journal_supplies_np_content_predicate_and_positional_provenance[0,2]` | Requires the occurrence returned by the audit to belong to `individual_ids` | Two representation ports to the written token occurrence; preserve content, predicate, role and provenance assertions |

The proposed additional representation ports cover eight parameterized
cases; they are beyond the three named ports in the current instruction.
They have not been applied. The two failed partition assertions remain
mechanism evidence and will not be loosened or relabeled as representation
ports. The immutable LTM rule and the new same-description/distinct-address
regression remain in force.
