# Item 3.5 line 3: stage 4e and NP continuation — 2026-10-23

Base: BasicModel **3624f62045725888c0cc0aa3e41ae6ea62f32118**, WikiOracle
**c9b19d4ec490861e36db812839aa75b3d32186f9**. Both mains were pushed after
Claude's step-2 §88 LAND, the single default sweep and its scoped repairs.
[Step-2 receipt](../2026-10-23-item3-5-step2c/sweep/README.md).
BasicModel was clean; the unrelated parent nanochat checkout was untouched.

Governing decisions: parts-of-speech §5a, the 5.5 plan's NP continuation,
the stage-4e student-state record, and Alec's current hand-off to regenerate
the three retired artifacts from their producing lessons. Their retired
files are never loader inputs. The dated code review at the end of the
[5.5 plan](../../plans/2026-10-17-item5-5-start.md) records the adjustments.

## Protocol and order

1. Stage3, original producer `test/constituent_order.py`: seed 20261010,
   LR .003, two lessons of 64 B4 batches, 512 training presentations,
   256 paired updates, 16 evaluations. Original stages-1–2 warm-up through
   the native checked loader; no compatibility bypass. The original CE
   bar is reported, and the free order gate controls continuation as
   already decided in the stage-4e record.
2. NP lower 6/4, definite 6/4, description 24/12, adjective 24/12,
   definite-adjective 24/12 presentations/paired updates. Same original
   factorial and seed, LR .003, B4, two passes. Successful path 84/44;
   no repeat of the historical discarded branch. Each predecessor must
   pass its held-out behavior gate before the next lesson.
3. Final fourteen held-out readings with no updates; then the copular
   identity lesson and its artifact. Check the latter's producing
   protocol before starting it; no unbudgeted substitute curriculum.

Gates stay §5a: .9 brackets, .95 one order-1 occurrence per singular NP,
zero duplicates/collisions, exact answers. Norms, chosen operators,
lowering stamps and bare-description occurrences are reports. Stop the
lesson sequence at the first failed behavioral gate. No forced choices,
seed rescue, budget increase, premise targets or learning claim.

## Source/port ledger

The initial source has no production changes; source 04 repairs the
occupation-counting defect documented below. The native determiner menu has one
action per operator; recency already follows store order.

- `np_experiment.MENU` gains compose-time `bind`, replacing the binding
  opportunity formerly inside determiner variants. The declaration now
  says one determiner action and a bind menu of locations plus dummy.
- Remove the driver's historical empty-occupation-table prerequisite.
  Regenerated stage3 keeps its learned counts. The native loader's exact
  retention checks and the before/after occupation tables remain.
- `test_np_gates_behaviors_and_reports_the_kept_operator`: its exact MENU
  tuple gains `'bind'`; every behavioral assertion stays.
- `test_np_menu_admits_both_modification_operators_at_the_native_mask`:
  its fixture gains a unary bind beside surface_unary and its exact finite
  logit mask becomes `[True, True, True, True, False, True, False, True,
  False, True]`. Both bind positions are available; verb and surface_unary
  stay outside this existing NP lesson menu. No test is retired.

The measurement wrapper retains source manifests, input artifact hashes,
initial state, each batch's RNG, both trials' credit/reading records,
the most recent pre-batch state and any failed state. Output folders are
unique and ignored while the source stays clean; their receipts are
archived before review. No lesson result or phase artifact is claimed yet.
Claude reviews through Alec before line 3 is published. Item 3.0 has not
been reached; layout/removal/STM work stays there.

## Source 01: scoped check and annotation leak

Frozen `acc916026`: 79 passed, three phase-artifact skips, two failures
in the reading-curriculum file. [Full scoped log](source-01/gates/run.log).
Both failing cases pass unchanged in a fresh process
([isolated diagnostic](source-01/isolated-before/run.log)). Prepending
only the occupation-count test reproduces both exceptions
([causal diagnostic](source-01/annotation-leak-before/run.log)).
The annotating fixtures left `grammar_lessons` on their shared data owner.
The source/text guards correctly refused the stale annotation before
greedy cost completed; no paired keep exists for either exception.

Additional fixture ports: `test_parts_of_speech` and
`test_constituent_lesson` use `monkeypatch.setattr` for `grammar_lessons`,
restoring the previous value after each test. No assertion changes or
retired tests. The dated 5.5-plan section records this §39 repair. No
training run or phase artifact has been produced on source 01.

## Source 02: checks pass; old warm-up rejected

Frozen `8dd3a197a`: [81 passed, three existing phase skips](source-02/gates/run.log).
The first stage3 invocation stops before training because the native
loader rejects the October-15 warm-up (366 unexpected parameter keys,
79 newly initialized keys). [Failure](source-02/stage3.log),
[source/input hash](source-02/stage3-capture/run.json). Initial and failed
states are retained, with no compatibility bypass or architecture change.

The next source regenerates only the producing stages-1–2 schedule:
76 presentations, 38 paired updates, original seed 20261010 and LR .003.
The adapter uses the original fields, order, native `present` and kept
program observer. It does not run the following identity-balanced
curriculum; its historical singleton-cosine gate is not a stage3 loader
prerequisite and is not claimed. See the dated 5.5-plan decision. This
dependency regeneration adds no stage3/NP presentation or retry budget.

## Source 03: duplicate word occupations, captured before repair

Frozen `711df4e841e05ad9630bbc4d9e3cdbb6458f8449`: 80 passed, three existing
phase skips, one failure of the unchanged exact word-count assertion.
[Scoped run](source-03/gates/run.log) and
[captured-RNG replay](source-03/occupation-diagnosis/run.log) retain the
same failure. The replay's `probe.json.gz` records both readings, R/A,
keep decision and the closing plan. Row 0 ties and keeps greedy. Row 1
keeps explore: R ties at 1.05; A is 1.089887 versus .764748. The plan adds
lower input/output and intersection input counts to word handles already
counted by the kept program. It is not a duplicate commit: the second
commit call correctly sees `_committed=True`.

Source 04's production repair restricts additional closing occupations
to nonword handles. It uses kept word IDs and native symbol provenance,
never definitions or a grammatical decision. The existing exact word-count
assertion stays; the new deterministic test also checks the composite
output's occupation. No assertion port or test retirement is required.
The scoped files gain PartClosing's closing/reference/bind consumers.

The original-budget warm-up on source 03 completed 76 presentations and
38 paired updates. Stage3 completed 512 presentations, 256 paired updates
and 16 evaluations. Source order passes 8/8; the original CE bar fails.
Free decoder output is 2/4 plus 3/4, a separate measure. Both artifacts are retained and ineligible for continuation due
to the production defect. Regenerate the original schedules on the
repaired source; no seed rescue, added lesson budget or relaxed gate.

## Source 04: captured failure repaired; new fixture correction

Frozen `9623961f8a4aac29c3e87f023811009d61fc7990` passes the
[captured initial-RNG replay](source-04/occupation-replay/run.log).
The [expanded scoped files](source-04/gates/run.log) give 191 passed,
seven existing phase skips and one failure in the new regression. Its
fixture selected `universal`, the last rule whose method is `lower`,
while its Composition and assertions specify the `lower` formal face.
[Declared-field diagnosis](source-04/fixture-diagnosis.json).

Source 05 selects the rule by `PartsOfSpeech.operation_name`, matching
production and the existing Composition. Every assertion stays verbatim;
no test is retired and production is unchanged. This same-round setup
correction follows reconstruction §17's exception and §39. No lessons
ran on source 04.
