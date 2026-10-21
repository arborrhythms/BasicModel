# Review 3 scoped certificates

Tested source: `47652a9fe826817ffe3d13eaac812d75ced9e41f`.

This is the requested eight-file set plus the unchanged random paired reconstruction
certificate, not a second full sweep. The decomposition file is run in both gain arms.

| File | Passed | Failed | Prerequisite skipped |
|---|---:|---:|---:|
| `test/test_math_chain_section14.py` | 11 | 1 | 1 |
| `test/test_reference_close.py` | 19 | 3 | 0 |
| `test/test_reference_mentions.py` | 4 | 0 | 0 |
| `test/test_bind_thought.py` | 44 | 0 | 0 |
| `test/test_identity_rows.py` | 14 | 0 | 0 |
| `test/test_identity_cube.py` | 12 | 0 | 0 |
| `test/test_part_store.py` | 4 | 0 | 0 |
| `test/test_operators_final_b.py` | 1 | 0 | 0 |
| `test/test_math_chain_decomposition.py` | 4 | 0 | 0 |

Native: **109 passed, 4 failed, 1 skipped**. Decomposition: **4 passed**,
with all **8/8** companion questions correct. The copular artifact is the skip.
All validated production, configuration and test file hashes still match the tested source.

## Question A: why is the live-arrival gate still false?

| Check | Result |
|---|---|
| Native forward arrival | Fails at line 391, row 26, stream 6 |
| Pending reference | Filled with row 51, witnessed by row 52 |
| Actual complement | Row 51 has three slots, with word 14 in role 0 and an open role 2 naming word 59 |
| Cause class | Same unresolved live-explore complement as the exact `fresh-forward-5` → `arrival-21` replay |
| Reverse native case | Passes after the working-read certificate port |
| Six fresh per order | Not accepted: the earlier complete set was 5 passed / 7 failed; it is not regraded by these scoped passes |
| Decision | The original assertion/forced-scope requirement keeps the [contract question](reverse-scope-question.md) open |

## Question C: which reference-close cases still expect a symbol?

| Case | Observed failure | Required work |
|---|---|---|
| `test_mint_uses_the_selected_member_order_instead_of_the_source_word_order` | No newly allocated individual symbol; indexes an empty symbol list | Decided §10.2 token address and selected-order port |
| `test_recent_minted_symbol_returns_at_its_native_conceptual_order[0]` | Ordinary occurrence has no symbol-valued raw role 0 | Decided §10.2 kind reference, token address/activation and fixture port |
| `test_recent_minted_symbol_returns_at_its_native_conceptual_order[3]` | Same old symbol assumption at another source order | Same §10.2 work; preserve native-order assertions |

These three are failures on this source, not passes or waived assertions. The
already-decided §10.2 round supplies their production changes and declared ports
after the present B/A round. It has not started.

## Question D: are both decomposition arms and paired R intact?

| Gate | Result |
|---|---|
| Zero gains, prime on, φ=.5 | Both lengths pass; 4/4 companion questions correct |
| All gains .5, prime on, φ=.5 | Both lengths pass; 4/4 companion questions correct |
| Original seeds and parameter snapshots | Preserved and verified in both arms; [validation](validation-29.json) |
| Work, one/two successors | 235 / 579, within the unchanged 512 / 768 budgets |
| Random paired bootstrap reconstruction | `test_both_training_trials_share_bootstrap_snapshot` passes |

[One table per decomposition question](decomposition-29-questions.md).

Publication remains held. Neither main moved. No new learning run, seed selection,
budget extension or second full sweep was used to change these outcomes.
