# Memory retained into the next batch

These are the saved-tensor ownership observations from the unguarded diagnostic runs, not the guarded gate results. The second column follows the support-read and diagnostic-cache repair, before predicate compaction.

| Owner | Before (GiB) | After support/cache repair (GiB) |
|---|---:|---:|
| `saved:activation:Spaces.py:_stage0_carrier/Spaces.py:compute_stage0_carrier/PerceptProperties.py:on_counts` | 2.007820 | 0.070320 |
| `saved:activation:Models.py:compose/Models.py:stage_cs_lang/Models.py:_tensor_record_selected_values` | 0.281983 | 0.000000 |
| `saved:activation:Spaces.py:_stage0_unity_forward/Spaces.py:_stage0_carrier/Spaces.py:compute_stage0_carrier` | 0.263672 | 0.263672 |
| `saved:activation:Models.py:stage_cs_lang/Language.py:choose_operation/Language.py:forward` | 0.047609 | 0.000000 |
| `copy:symbolSpace.subspace.layers.2.predictor.2.weight` | 0.003906 | 0.003906 |
| All saved storage | 2.605323 | 0.338169 |

Raw observations: [before](ag-owner-before-v2.jsonl) and [after support/cache repair](ag-owner-middle.jsonl).

The later allocation trace identifies repeated 1.004 GiB host predicate temporaries. Compacting the present columns removes those expansions while preserving their global identities and exact boundary learning. The final-source [two-epoch native gate](final3-core-gates/group-00/result.json) passes at 7.45 GiB under the unchanged 8 GiB guard. Its original graph-free carrier assertions remain.
