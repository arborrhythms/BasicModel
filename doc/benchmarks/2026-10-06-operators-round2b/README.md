# Operators update, round 2b

**Status: measured candidate; standing gate failed. Held for Claude's review. No commit or push.**

Started from the frozen round-2 manifest
`63d68a9b9c38d14470293e12e7d7e85103c24bfa7572d605babafed0c7e7043a`.
All thirty predeclared, unseeded trainings completed without retry, replacement
or tuning. The delivered source and measurement helpers matched throughout.
The round-2 receipt is unchanged. The plan, Philosophy and the previously
corrected Architecture retain their starting hashes.

## Changes

Only strictly lower reconstruction `R` keeps explore; a tie keeps greedy.
The SCG advantage remains the detached total difference `Δ(R+E+A)`, with the
same registration, reduction, single joint departure and `K·R` correction.
The audit records both trial components, the keep's reconstruction decision,
the advantage sign, and answer-driven credit against the keep. The answer-owned
reader trains on both detached trial roots. Trial publication restores that
trial's own narrowing evidence before its training step.

The handoff preserves the leaf event and activation magnitude, changing only
sign where a field operation reached the word. Tied poles retain the prior
sign. Explicit-pole grammar retains the pair. Reference-slab evidence uses the
signed magnitude, saturated exactly as in the landing; an untouched word has
bitwise-identical evidence to the native scalar conversion. Four consumer
paths are declared and counted below. The concept-only closing image is
unchanged: XOR_grammar and MM_grammar have fourteen form-content coordinates
inside the twenty-two-coordinate carrier, zero concept-complement width and
an identically zero image.

The order-zero containment audit intersects the postings of every word's
positive-net parts to enumerate every distinct containing row. Equal sets are
checked both ways and empty sets against all rows; self pairs are omitted.
Counts and maximum coordinate violations are reported before and after
placement snapshots. The current code remains the join `L`; no centroid pass
or higher-order enforcement is introduced. The focused `a`, `ab`, `ac` fixture
has two nontrivial comparable pairs, zero violations, and incomparable `ab`/`ac`.
It also detects an injected pre-placement violation and a zero post-placement
result. The [catalogue's fold table](../../specs/2026-09-29-operator-catalogue.md#123-operators-update-round-2b-2026-10-06-review-candidate)
states each fold's monotonicity and the conditions of the guarantee.

## Freeze and verification

The [source manifest](delivered-source/source.json) covers 701 files; SHA-256
`d99b205afba2e1db0dbd0197d2506fdb100f71cd9cd3f2b0161bccdcea19ddba`. The [source archive](delivered-source/source.zip),
[diff from frozen round 2](delivered-source/changes.patch),
[measurement-helper hashes](delivered-source/measurement-helpers.json),
[complete old/new test texts](delivered-source/test-ports.json) and
[seed-call audit](delivered-source/seed-port-audit.json) identify the candidate.
No existing test seed changed. The only explicit replay seeds belong to the
separate, requested MM comparison.

The frozen result validator expected an optional `warnings` field absent from
this sweep’s result schema. A [schema-only postprocessing port](postprocessing-port.json)
uses an empty default, preserves the original and replacement text with hashes,
and validates the same saved measurements. The frozen helper is unchanged;
no training or measurement was repeated. [Supplemental helper hashes](supplemental-helper-hashes.json)
cover the report writer and the forward-only consumer inspection.

The [full sweep](full-sweep/result.json) ran once on this delivered source:
**5251/5251 cases completed**, outcomes
{'passed': 4965, 'skipped': 285, 'xpassed': 1}, in 208.6 seconds. Normal slow
exclusions and the weekly slow-record warning remain in the raw result.
[Focused checks](focused-final.log): 52 passed, one skipped.
The [final documentation-link check](final-doc-links.log) passed 295 cases. The initial
freeze was retained under `development/initial-freeze/` when the ownership
report's reader-row wording was corrected before the full sweep; no gate
training used it.

## Gate and all thirty trainings

| Gate | 6.8 landing | Round 2b |
| --- | ---: | ---: |
| XOR class | 7/10 | 1/10 |
| XOR reconstruction | 9/10 | 6/10 |
| XOR joint, reported | 6/10 | 1/10 |
| Sum control | 10/10 | 10/10 |
| MM_xor, live | 10/10 | 10/10 |
| Full sweep | green | green |
| Sentence-path perception gradient | 0 | 0 |
| Code displacement | 0 | 0 |
| Ownership conflicts | 0 | 0 |

[Validation](results-validation.json), [summary](measurements/summary.json),
[process plan](measurements/plan.json) and [campaign log](campaign-process.log)
retain the unchanged bars, budgets, all processes and every failure. Each XOR
training supplies both original bars; class is read from its committed greedy
root. Sum was read before XOR and MM. Bands: MSE below .05 is at zero; within
.02 of .25 is at one quarter; remaining values below/above .25 are between/above.

| XOR run | MSE | Band | Correct | Multisets | Class / reconstruction | Final operators |
| --- | ---: | --- | ---: | ---: | --- | --- |
| [1](measurements/xor-01/run.log) | 0.251814003 | at 1/4 | 2/4 | 0/4 | fail / fail | disjunction |
| [2](measurements/xor-02/run.log) | 0.251494743 | at 1/4 | 2/4 | 0/4 | fail / fail | disjunction |
| [3](measurements/xor-03/run.log) | 0.142467712 | between | 4/4 | 4/4 | fail / pass | conjunction |
| [4](measurements/xor-04/run.log) | 0.301200911 | above 1/4 | 2/4 | 4/4 | fail / pass | conjunction |
| [5](measurements/xor-05/run.log) | 0.255502472 | at 1/4 | 2/4 | 0/4 | fail / fail | disjunction |
| [6](measurements/xor-06/run.log) | 0.0719862255 | between | 4/4 | 4/4 | fail / pass | conjunction |
| [7](measurements/xor-07/run.log) | 0.145739296 | between | 3/4 | 4/4 | fail / pass | conjunction |
| [8](measurements/xor-08/run.log) | 0.0113089128 | at 0 | 4/4 | 4/4 | pass / pass | conjunction |
| [9](measurements/xor-09/run.log) | 0.343918022 | above 1/4 | 2/4 | 0/4 | fail / fail | disjunction |
| [10](measurements/xor-10/run.log) | 0.241413629 | at 1/4 | 2/4 | 4/4 | fail / pass | conjunction |

| Sum run | MSE | Band | Control |
| --- | ---: | --- | --- |
| [1](measurements/sum-01/run.log) | 0.312443972 | above 1/4 | pass |
| [2](measurements/sum-02/run.log) | 0.302676082 | above 1/4 | pass |
| [3](measurements/sum-03/run.log) | 0.310594469 | above 1/4 | pass |
| [4](measurements/sum-04/run.log) | 0.300967783 | above 1/4 | pass |
| [5](measurements/sum-05/run.log) | 0.311156631 | above 1/4 | pass |
| [6](measurements/sum-06/run.log) | 0.283827037 | above 1/4 | pass |
| [7](measurements/sum-07/run.log) | 0.299227834 | above 1/4 | pass |
| [8](measurements/sum-08/run.log) | 0.299772918 | above 1/4 | pass |
| [9](measurements/sum-09/run.log) | 0.304856598 | above 1/4 | pass |
| [10](measurements/sum-10/run.log) | 0.302165449 | above 1/4 | pass |

| MM run | Epochs | Best MSE | Final MSE | Gate |
| --- | ---: | ---: | ---: | --- |
| [1](measurements/mm-01/run.log) | 84 | 0.176207185 | 0.176207185 | pass |
| [2](measurements/mm-02/run.log) | 49 | 0.193171933 | 0.193171933 | pass |
| [3](measurements/mm-03/run.log) | 35 | 0.186596483 | 0.186596483 | pass |
| [4](measurements/mm-04/run.log) | 66 | 0.191083521 | 0.191083521 | pass |
| [5](measurements/mm-05/run.log) | 39 | 0.19561784 | 0.19561784 | pass |
| [6](measurements/mm-06/run.log) | 37 | 0.198059291 | 0.198059291 | pass |
| [7](measurements/mm-07/run.log) | 53 | 0.175134331 | 0.175134331 | pass |
| [8](measurements/mm-08/run.log) | 46 | 0.184140936 | 0.184140936 | pass |
| [9](measurements/mm-09/run.log) | 37 | 0.184525177 | 0.184525177 | pass |
| [10](measurements/mm-10/run.log) | 94 | 0.199038714 | 0.199038714 | pass |

## Credit, reader exposure and consumers

Every run's `run-audit.json` retains the two trial components and decisions,
nonzero departures by walk and action, per-epoch chooser logit ranges, reader
row masks, and final committed training operators with both the selecting `R`
and credited total. The evaluation operators and selecting costs are retained
in each gate observation. The [aggregate audit](aggregate-audit.json) and
[tenth-run audit](measurements/audit-summary.json) summarize the same trainings.

XOR: 5,210 nonzero advantages; 1,237 credits against the keep, including 1,237 where adding the answer changes the credit direction. Reader exposure: 8,000 trial steps and 32,000 active rows.

SUM: 0 nonzero advantages; 0 credits against the keep, including 0 where adding the answer changes the credit direction. Reader exposure: 8,000 trial steps and 32,000 active rows.

| Consumer, calls with a pair available | XOR total | Sum total | MM total |
| --- | ---: | ---: | ---: |
| `_attention_sentence_payload` | 36060 | 36060 | 0 |
| `_pushed_word_slab:poles` | 0 | 0 | 0 |
| `commit_word_reference_slab:per_word` | 40061 | 40060 | 0 |
| `commit_word_reference_slab:whole_slab` | 4010 | 4010 | 0 |

The explicit-pole branch is covered by focused tests; these gates use codes.
The unchanged `MM_xor` gate records zero calls to all four consumers. The
[forward-only call-profile check](consumer-path-check.json) confirms that
`MM_xor` has `word_brackets=False` and no reference evidence, while
`MM_grammar` has `word_brackets=True` and reaches both reference paths.
This contradicts the hand-off’s specific reference-slab attribution to
`MM_xor`. The paired trajectory nevertheless differs: MM remains a live gate,
and that difference is not attributed to an unobserved consumer. The raw MM
observer’s `landing_comparison.live` field is derived only from consumer calls;
the paired-replay result governs the gate’s live status.

XOR containment: 0 comparable pairs across final banks; maximum observed before/after violation 0.0. Zero pairs in a bank are vacuous; the focused containment fixture supplies the nontrivial check.

SUM containment: 0 comparable pairs across final banks; maximum observed before/after violation 0.0. Zero pairs in a bank are vacuous; the focused containment fixture supplies the nontrivial check.

MM_xor has no attached `MereologicalCodes` bank in this configuration. Its
containment audit has no applicable banks; it is not evidence for form order.

## Three paired MM replays

[Plan, construction hashes and trajectory comparisons](paired-mm/result.json).
Seeds 0, 1 and 2 were fixed before execution; both versions ran all 200 epochs
at the unchanged learning rate .01, even after crossing the convergence bar.
All three pairs have identical construction parameters and RNG state, but
all differ at the first forward and every recorded epoch. The 6.8 landing is
`42daf96f454a30a3633054151e4ba8d498cd3b59`. Its source archive, six complete
trajectories, process metadata and logs are retained under `paired-mm/`.
The §5 identity amendment does **not** apply; MM's ten gate trainings count.
These six replay trajectories are separate from the thirty unseeded gates.

## Disjunction residual and narrowing observation

[Fixture results](disjunction-result.json), [old/new handoff bodies](handoff-bodies.json)
and [diagnostic log](disjunction-cause-process.log) were recorded before any
of the thirty trainings. The fixture uses the frozen round-2 configuration,
one captured initialization reused for each variant, no optimizer steps,
a forced disjunction, and the observed `not/and/or/descend` field prefix.
The corrected handoff **does not restore this negative-pole disjunction**:
its reconstruction costs remain .01148–.01549 and it recovers 0/4 multisets.

The residual is downstream: `InterpretLayer.forward` resolves a reference as
`object_atoms * activation`. The preserved negative activation still negates
the object form seen by the binding kernel. For disjunction, the direction
of `−u−v−u*v` differs from the negation of `u+v−u*v`, so positive dictionary
pairs cannot exactly recompose it. Two negated operands cancel their signs
in conjunction's product, which explains why that failure is asymmetric.
A diagnostic that uses activation magnitude only at interpretation recovers
4/4 multisets and zero argmin recomposition residual. It is **not installed**:
interpretation, the binding kernel and pair search remain the frozen-round-2
mechanisms, as this hand-off permits a residual to be reported. The fresh
untrained decoder's policy costs are retained separately from exact pair
recomposition and the grammar multiset readback; no trained landing cost is
claimed for that untrained diagnostic.

The [saved-trace recheck](round2-narrowing-recheck.json) does not confirm §15's
passing-run `divide` observation. All 40 final round-2 evaluation sentences
choose `descend` for the two-word bracket, including every passing run.
Failing disjunction runs have the field prefix, but passing conjunction run 3
has it too. Action 0 is divide and action 1 descend in the frozen implementation.
Divide requires a both-pole bracket; descend is available when the bracket has
parts. The claimed divide/descend distinction therefore has no supported
causal explanation in the saved final traces.

## Documents and review boundary

GradientFlow records the reconstruction keep, total-cost credit, both reader
trials, pole-only handoff, residual and complete consumer census. Catalogue
§12.3 records containment and fold monotonicity. FutureWork and todo carry the
remaining interpretation issue, centroid placement and later rounds. The
round-2 receipt, plan, Philosophy and existing Architecture corrections are
preserved. Source, tests, archives, all measurements and failures are held for
Claude's review; no commit, push or parent submodule bump was performed.
