# Findings retained for review

The final sweep runs on the same source as the closing measurements. Its
failures remain saved; no post-sweep repair, retry, assertion change or threshold
change is included in this receipt. The final aggregate is in `README.md` and
`sweep-summary.json`: all 5,090 cases completed in 110.70 minutes, with eleven
failures and no process stops. `full-sweep/receipt.json` retains the raw report
event counts; `case-accounting.json` separates four extra subtest reports from
the collected-case counts.

## Findings identified during the sweep

| Case | Observed failure and source inspection |
|---|---|
| `test_compiled_word_chunk.py::test_tiny_canonical_detached_reverse_stops_at_root` | `model.detached_reverse` is false. The fixture is a retained meronomy reading: `_understanding_reconstruction_scope()` now requires live tied reconstruction there, overriding its old detached-student request. This fixture has not been ported. |
| `test_compiled_word_chunk.py::test_tiny_canonical_detached_reverse_train_step_is_finite` | The same fixture has no detached reverse chooser, so `chooser.parameters()` raises. Its optimizer assertions are not reached. This fixture has not been ported. |
| `test_compose_deadline.py::test_shared_stack_contract_allows_unary_until_the_deadline` | The forced unary preference finishes at depth one with finite values, but its final top vector is all zero; the unchanged nonzero-top assertion fails. The root cause has not been established. The test moved unchanged from the item-7 reduction-deadline file. |
| `test_generation_lesson.py::test_sentence_generation_lesson_keeps_its_weighted_output_gradient` | Its `SimpleNamespace` fixture returns raw compose/generate objectives but does not construct `_grammar_lesson_errors`, which the new trained-total registry consumes. It fails on that missing attribute before either of its gradient comparisons. This fixture has not been ported. |
| `test_query_phase_fullgraph.py::test_query_mask_preserves_real_fullgraph_forward_backward_across_lengths` | After successful forward/backward at four words and then seven words, the graph counter is two instead of the required one. The earlier journal-width diagnostic independently demonstrated an extra capture when the longest-sentence record width changes. That is a plausible explanation here; this failing run did not record the precise guard that caused its second capture. The one-graph assertion is unchanged. |
| `test_sentence_compose.py::test_disabled_sentence_prediction_leaves_adam_momentum_unused` | The hand-built discourse object returns scalar costs but lacks `_sentence_prediction_errors`. The new cost path reads that attribute even though both prediction weights in this fixture are zero. The test stops before its no-gradient/no-momentum-update assertions. |
| `test_sentence_compose.py::test_legacy_event_reporting_does_not_zero_the_sentence_objective` | The real training assertions pass: two positive sentence costs, positive trial costs and the expected training-step count. The final assertion requires the legacy D3 path with tied reconstruction disabled, which the retained-meronomy scope decision now overrides. This fixture has not been ported. |
| `test_tied_operator_reconstruction.py::test_word_scoring_honors_existing_nul_termination` | Its unknown-candidate subcase sets every candidate-validity bit to false, then expects the former uniform `log(256)` cost. The decided no-admitted-surface rule now returns zero. This subcase has not been ported; its failure also prevents the later NUL-junk value/gradient comparisons in the loop from running. Those comparisons must be kept in any port. |
| `test_tied_operator_reconstruction.py::test_byte_targets_survive_whole_word_and_prefix_promotion[alphabet]` and `[alph]` | Both variants deliberately pass an empty candidate bank and expect the former uniform cost. The initial raw-byte and bank-width checks pass, but the failure occurs before the promotion iteration and the final constituent-count assertion. A port must retain those target/promotion assertions and apply the decided no-candidate rule separately. |
| `test_tied_reconstruction_objective.py::test_student_checkpoint_migrates_shared_weights_and_adam_by_name` | The fixture requests a detached student on a retained meronomy model; the new scope rule supplies no such student. It fails before constructing the historical checkpoint, so this result does not establish whether checkpoint loading or Adam-state migration is correct. Historical checkpoint coverage still needs a valid old-state fixture. |

The first three failures are in part-00 workers 098, 099 and 100; the generation
lesson failure is in worker 253 and the query graph-count failure is in worker
558. Both sentence-compose failures are in worker 634; the empty-candidate
scoring failure is in worker 769, the two promotion variants in worker 772,
and the student-checkpoint fixture in worker 777. Their complete tracebacks are in those workers'
JSON reports and logs. Source inspection is not a substitute for verifying a
future repair. Any additional failures remain listed in the final aggregate.

## Closing gates and scope

All ten class runs classify all four inputs correctly; nine meet the unchanged
MSE bar. None of the ten reconstruction runs reaches its unchanged word-multiset
bar. The ten sum-only runs have essentially zero XOR interaction (absolute
contrast at most 2.98e-8), but deviate from one half by .00100–.00331 and fail the
settled float32 `allclose` comparison. Those numerical facts do not turn the
control green.

The unchanged XOR_grammar fixture inherits legacy lexicon synthesis, which is
explicitly exempt from automatic tied reconstruction until the deferred mode
migration. Its default gates are therefore distinct from the earlier
receipt-local reconstruction-enabled stage-1 arms. No configuration override
was introduced for the closing gates.

## Measurement limits

The reconstruction timing receipt measures a first numerical production batch,
not a full epoch or compiler capture. Three configurations complete both arms;
eight reach the 8 GiB guard in both arms and one rejects an over-capacity
sentence in both. These nine comparisons have no successful timing difference.
The native production measurement fits its explicitly allowed 24 GiB slow
ceiling, but not the 8 GiB sweep target.

The independent weekly run uses its frozen review-13 source. Its failures and
guard stops are retained, and only never-started cases continue. Its results
must not be presented as validation of the later cost-function source.
