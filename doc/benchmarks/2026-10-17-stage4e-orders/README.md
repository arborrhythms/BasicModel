# Stage 4e: constructed orders and chooser diagnostic

The reviewed preparation landed on BasicModel main at `d080b632d`, with
WikiOracle bumped at `2338d9c`. No learning claim. This receipt follows the
[first-run review](../../plans/2026-10-17-item5-5-start.md#review-of-the-first-stage-4e-run-claude-2026-10-17--gate-failed-for-a-structural-reason-fix-before-any-rerun).

## A. Constructed reference orders

`ReferenceContext.operation_order` is shared by reference requests and result
stamps, including the tensor compose walk and the kept journal. The
determiner operation constructs its output order; other operations retain
their stamp/delta behavior. No word mapping or maximum order is introduced.

| Operation / check | Result |
|---|---|
| Mint and bind, description stamps 0, 2 and 17 | Requested/result order **1** |
| Kind, description stamps 0, 2 and 17 | Requested/result order **2** |
| Native mint with description stamp 0 | One order-1 reference request, written occurrence and witnessed individual |
| Existing stamp-2 closing, reconstruction and checkpoint assertions | Unchanged assertions pass |
| Higher-order reference and constituent focused checks | **16 passed**; artifact gate excluded pending the run |

The amended gate requires one order-1 occurrence at every lowered NP.
Description stamps are reported, not required to be 2. Missing NPs remain
failures; bracket recovery, occurrence coverage and duplicate-write checks
are unchanged. The first receipt retains its historical 2→1 measurement.

## B. What supplies native cases

`Models._sentence_reference_bank` stages the current words and their existing
interpretations. `CaseSelection.stage_case_bank` follows their stored sigma
edges, reads case memberships through `attribute_presence`, and derives
modifier evidence through `cs_reverse_presence`. It allocates nothing.
`select_cases` intersects the head's cases with the modifier's evidence;
`CompoundLayer` refolds only that intersection.

| Saved-checkpoint probe | Native contents / consequence |
|---|---|
| Sigma store | **0 edges**, hence no written case memberships |
| Staged bank for four sentences | **9 identities**, **0 nonzero case entries**, 9 self-observation entries |
| Adjective NP / adjective intransitive | Both teacher traces stop at `compound` |
| Transitive / prepositional | Complete teacher traces |
| Compound target without cases | No legal numerical fold or downstream teacher state under the current operator contract |

Cases are populated by witnessed definitions and native field discovery:
`_populate_concept_weights` writes new percept alternatives or symbolic
memberships; `_assign_concept_parts` records discovered collections;
`assert_concept_relation` can add asserted membership. Text-only singleton
recognition supplies the word's own definition, not shared object/property
cases. The corrected warm-up therefore does not supply the prerequisite.

The prerequisite lesson is grounded stage-1/2 object/property evidence with
independently varied cases, followed by stage-4b subtyping: the head must have
multiple witnessed cases and the modifier must explain a subset of those
same cases. `ConceptLessons.teach_concept_lessons` demonstrates the existing
primitive/name and target-free case-observation path for grounded tasks;
the natural-language case lesson needed by this corpus is not wired into
the accepted warm-up. A bracket label cannot create those memberships.
Changing the compound definition or inserting cases would be additional
work, so neither is included in this fixed-budget rerun.

## C. Chooser credit on the saved checkpoint

Four native free readings of `output/stage4e-20261017-r3/trained.ckpt`
supplied the teacher leaves. The probe evaluated 15 reachable wait targets
and nine folds, using the real chooser, `Error` registry, objective ownership
and optimizer. A disposable CE update was measured; its state is not used
by the rerun. [Full diagnostic](scorer.json),
[summary and setup note](diagnostic-summary.json),
[probe](../../../test/constituent_diagnostic.py).

| Diagnostic | Result |
|---|---|
| Actual GELU preactivation | **−0.704250 to 0.777013**; no dead negative tail; this scorer has no tanh |
| Learned per-location score gap, maximum | **0.163599** |
| Legal action logit gaps | **0.000857–2.193294**; includes the fixed load/deadline prior |
| Direct and owned CE gradient norm | Both **0.170885**; all nine chooser parameters are optimizer/lesson owned |
| One .003 optimizer step, parameter-change norm | **0.000512656** |
| Relative lesson loss before → after that step | **1.706215 → 1.706128** |
| First-run checkpoint change from warm-up | First MLP weight **0.140084**, output weight **0.091598** in norm |

Neither saturation nor a detached CE path is reproduced. The first run
made small, nonzero updates; flat-looking CE is not evidence of either
suspected defect. The chooser is unchanged. Its common scalar output bias
cancels from softmax; zero question context and the absent reference-choice
features explain the corresponding zero/unused gradients in this syntax
probe. No stronger learning claim follows from this one-step diagnostic.

## Fixed rerun, declared before execution

Use the same paired stages-1/2 warm-up, seed **20261010**, **144 training
presentations**, **72 optimizer updates**, batches of four, rate **.003**,
and **40 free held-out readings**. Corpus and presentation order are
unchanged. No extra case lesson, identity presentation or clock/chain/surface
change. The corrected gates are ≥90% exact brackets, ≥95% singular-NP
occurrences with zero duplicate writes, one order-1 occurrence at every
lowered NP, and no occurrence for unlowered descriptions.

## Rerun outcome and closing repair

The one rerun on `f7edc9eee` **failed at runtime in its first batch**, after
the paired optimizer steps and before committing that batch. This is a new
candidate failure, not a setup abort or a failed learning gate. The source
and [failure](failed-run-failure.json), [protocol](failed-run-protocol.json),
[restore record](failed-run-start.json) and [log](failed-run.log) are retained.

| Measurement | Result |
|---|---|
| Training budget | Four presentations attempted; **0 completed batches**, two optimizer steps before admission failed |
| Held-out brackets, occurrences, lowering and controls | **Not measured: 0/40 held-out presentations reached** |
| Closed meaning norms and held-out kept programs | Not measured |
| Runtime failure | `admission changed the selected clause structure` |
| Cause | `AnswerProgram.detached()` omitted `close_references`; the kept journal was rebuilt under the retired reference contract |
| Isolated defect regression | Fails before repair; passes after preserving the flag |
| First-batch diagnostic with the repair | Two steps, successful commit; old flag-dropping structure matches **3/4**, repaired structure **4/4** |
| Focused tests after repair | **28 passed**, artifact gate excluded |

The first-batch diagnostic used the same warm-up, seed and first four
sentences, then stopped. Its four presentations and two updates are a
separate correctness reproduction, not a continuation or a replacement
learning result. [Structure and actual program diagnostic](commit-regression.json),
[driver](commit-probe.py), [before](detachment-before.txt),
[after](detachment-after.txt). The failure observer now records optimizer
steps and completed presentation counts explicitly.

The full experiment was **not restarted**. No identity run, 5.5 mechanism
round, or learning claim follows. The final default-sweep result is pending.
