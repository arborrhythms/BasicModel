# Step 2's one full sweep — 2026-10-23

Reviewed source: **fcaff2d19624a581206fed3a17dc1b5b3d752612**, Claude LAND
in step-2 §88. The single default sweep completed all **6,375** selected
cases: **6,034 passed, 17 failed, 323 skipped, one non-strict XPASS** in
540.4 seconds. Before/after source audit unchanged; no compile-cache
retry. [Manifest](source-01/run.json), [results](source-01/summary.json),
[all failures](source-01/failures.json), [workers](source-01/full/result.json).
Initial RNG, model/MTM state where available, failure artifacts, pending
arrival captures and keep probes are retained under source-01. The
default suite retains its slow/device/artifact boundaries; no learning
claim follows from its XPASS or any other result.

Step-2 §89 documents the diagnoses and repairs under §39. Production is
unchanged. This failed sweep will not be relabeled by scoped passes, and
there is no second full sweep.

## Exact ports and fixture ledger

The complete [patch](repairs/test-ports.patch) and
[AST assertion audit](repairs/assertion-audit.json) supplement this list.
No test is retired; one catalogue test is renamed as listed below.

1. `test_aligned_capacity_checkpoint_growth.py`, the atomic-capacity and
   legacy-mixing-handle tests: bind the real `_concept_capacity_window`
   method to each partial owner. No assertion changes.
2. `test_bounded_stm_fold.py::test_compose_single_reduction_space_role`:
   `assert layer.r_apply == 0` becomes `assert layer.r_apply == 1` plus
   `assert isinstance(layer.unary_ops[0].gl, Language.BindLayer)`.
3. `test_expectation_defaults.py::test_expectation_off_keeps_every_packed_closing_in_ltm`:
   observe the writer's `root_occurrence` via its `written_rows` map,
   rather than using a returned Part handle as a row index. No assertion
   changes; both source sentences and all stored-kind/content checks stay.
4. `test_live_concepts_from_percepts.py::_checkpoint_space`: install a
   real empty MediumTermMemory and expose the existing readout parameter
   as the checkpoint restore device. No assertion changes.
5. `test_operators_round2c.py`: the forced walk clears `candidate_round`
   and `bind_round`. The existing forced binary product/sum fixture takes
   the next legal option when the saved departure is STOP at depth one.
   It no longer ignores the excluded STOP. No assertion changes, menu
   masking, production forcing, objective or keep-law changes.
6. `test_parts_of_speech.py::test_kept_readings_alone_supply_occupation_counts`:
   `assert parts.counts and set(parts.counts) - set(expected.counts)`
   becomes `assert parts.counts`. The preceding exact kept-word count
   equality stays, as do annotation isolation and frozen-state checks.
   A random kept reading need not mint an additional non-word symbol.
7. Its `test_word_flow_follows_the_catalogue_and_ends_at_a_clause`:
   `assert 'bind' not in rules` becomes
   `assert rules['bind'].arity == 1 and rules['bind'].head_role == 1`;
   add bind's mask 1 to the existing per-operation flow/count checks.
8. `test_taxonomy_entry.py::test_normal_entry_does_not_verify_generated_vectors_as_taxonomy`:
   `assert not len(cs._ltm_store) and cs._ltm_store.parts.contains(edge)`
   becomes `assert len(cs._ltm_store) == 1 and cs._ltm_store.parts.contains(edge)`
   plus `assert cs._ltm_store.row(0)['kind'] == 'inference'`. Every
   completed S now has its occurrence beside the relation. Existing
   taxonomy truth, domain and edge evidence checks stay.
9. `test_verb_adverb_grammar_ops.py::test_complete_contains_absorbed_role_only_methods`:
   `assert 'bind' not in complete_up | complete_down` becomes
   `assert 'bind' in complete_up and 'bind' not in complete_down`.
10. `test_when_grammar_rules.py`, rename
    `test_bind_has_no_chart_rule_ids` to
    `test_bind_has_one_forward_chart_rule_and_no_inverse`.
    `self.assertFalse(up)` becomes equality checks for one rule, arity 1
    and head role 1. `self.assertFalse(op.forward_rule_ids or op.reverse_rule_ids)`
    becomes exact equality of `op.forward_rule_ids` with that forward
    rule's catalogue index and `self.assertFalse(op.reverse_rule_ids)`.
    The no-down-rule and thought operand/result assertions stay.
11. `test_word_loop_buckets_and_orders.py::test_one_operation_layer_owns_both_arities`:
    `assert shared.r_reduce > 0 and shared.r_apply == 0` becomes
    `assert shared.r_reduce > 0 and shared.r_apply == 1` plus
    `assert isinstance(shared.unary_ops[0].gl, BindLayer)`.
12. No edits to the bounded-runner or test-mode files. Five failures are
    inherited measurement-plugin import errors in child processes that
    replace PYTHONPATH. Both unchanged files pass with capture plugins
    absent: [36 passed, three existing skips](repairs/clean-harness/run.log).

## Capture diagnosis before edits

The compose/MM failure reproduces from its original initial RNG in
[departure-before](repairs/departure-before/run.log) and
[departure-choice-before](repairs/departure-choice-before/run.log).
The latter [probe](repairs/departure-choice-before/departure-probe.json)
records the original draw, forced draw, greedy R/A and readings, legal
actions and final fork flags. All four rows join; three select saved STOP
107 again despite legal bind 9, and only the binary product row takes
its departure. Explore fails before its score, so there is no second
R/A or keep decision to report. This is not a bind-price or keep defect.
The initial hypothesis of stale candidate selectors was disproved in
this capture (all were -1); clearing them is the corresponding forced
walk fixture contract for draws that do select those walks.

The remaining failures are deterministic contract/fixture ports described
above. The packed-close failure is evaluation (`train=False`), so no
paired keep exists. All original failures and replay runs are retained.
## Scoped results on the frozen repair

Frozen **9f550087004bdbe21bbe113ca0e50395e261db56** changes ten test files
and no production/configuration file. Its 909-file source audit is
unchanged across all four scoped runs:

| Run | Result |
| --- | --- |
| All ten changed files, fresh initialization | 82 passed, 4 existing skips |
| Captured compose/MM initialization, departure and keep probe | 1 passed |
| Captured packed-close initialization and private generator | 1 passed |
| Both unchanged subprocess harness files, clean environment | 36 passed, 3 existing skips |

[Scoped log](repairs/scoped-source-01/run.log),
[departure replay](repairs/departure-after/run.log),
[packed-close replay](repairs/packed-close-after/run.log),
[clean harness](repairs/clean-harness-frozen/run.log).
The four skips in the changed files are two slow cases and the absent
np_definite/np_description artifacts; the clean harness's three skips
are slow cases. No new skip or expected-failure marker is introduced.

The repaired departure [probe](repairs/departure-after/departure-probe.json)
shows all four saved departures taken. R is zero for both trials in every
row; the binary product departure lowers A by 0.0000198483 and is kept.
The other three rows tie and keep greedy. Both trials' readings, exact
components and keep decisions are retained. No policy, keep or answer
target changed. All 17 original failures are accounted for by these
scoped results. The original sweep remains a failed run with its exact
counts, not a second green aggregate.

Claude's LAND of fcaff2d19 and Alec's §88 hand-off authorize publication
after the sweep repairs under §39. Step 2 ends here; stage 4e and the NP
continuation follow. Layout/removal/STM chunks stay in item 3.0, which has
not been reached. Publication carries the source, this receipt and all
failed runs on main, followed by the WikiOracle gitlink bump.
