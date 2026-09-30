# Item 7: two truths — implementation and review receipt

Status: **incomplete implementation candidate; uncommitted for Claude's review**.
The full sweep has 84 failures, and the reconstruction requirements are red.
Item 7 is not accepted and is not ready to commit. The documentation-only commits
`1678ee1f` (BasicModel) and `850798f` (WikiOracle) were pushed after 115
link checks; [their record](published-documents.json) names the six files and
co-author trailer. Nanochat is unchanged. This implementation follows the
[two-truths spec](../../specs/2026-09-16-two-truths-ideas-and-relations.md)
and the amendments in [todo](../../../todo.md).

## Acceptance coverage

Mechanism fixtures force grammatical choices; they do not claim that an
untrained chooser parses English or learns identity. The separate §7.15/21
measurement uses identical initial predictor weights and a fixed small corpus.
The [acceptance map](acceptance-map.json) names the tests or measurement for
each of the twenty-one numbered requirements.
Its [28 named mechanism cases](acceptance-outcomes.json) all pass in the full
sweep; §7.15 and §7.21 also have the small measurements below. These targeted
results do not close the integration failures or reconstruction requirements.
The additional [`true` execution check](../../../test/test_item7_storage.py#L293)
loads the declared thought operator and executes it over a sealed clause,
retaining both evidence poles.

## Full sweep and open review items

The one full sweep completed **all 5,016 selected cases exactly once**:
**4,611 passed, 84 failed, 320 skipped, one xfailed**. It took 5,873.17 seconds
(97.89 minutes), with no compile-cache retries. Peak worker memory was
6.33 GiB and peak aggregate memory 14.60 GiB, within the unchanged 8/24-GiB
caps. The selection has 145 added and 118 removed node IDs against the preceding
7.5 landing; [selection-delta.json](selection-delta.json) lists every one.

The failure groups below describe observed errors, not established root causes.
The [complete grouped list](full-failure-groups.json) and
[full result](full/result.json.gz) retain every failing node and traceback.

| Observed failure family | Cases |
|---|---:|
| Observation writer reached without an owned selected clause and shared index | 37 |
| Property analysis requires missing learned primitive definitions | 13 |
| Calls to removed WholeSpace taxonomy/category/fold-support APIs | 16 |
| Provisioned TruthSet kind disagrees with the selected grammatical clause | 3 |
| Other graph, order, layout, catalog, reset and identity assertions | 15 |

The broad forward-path failures show that selected-clause ownership has not
been integrated into every supported entry point. They are open implementation
work, not evidence that the focused fixtures establish end-to-end readiness.
Other review items include two compiled chunk graphs where one is required,
explicit order-1 interpretation of an existing order-2 kind, a native-ID
fixture whose IDs now equal its row numbers, and relative-scope expectations.
Some remaining fixtures still call retired APIs or expect `true` to be absent
from the thought catalog; their migration must preserve the new contract.

The protected depth-3 campaign is **red**. On this source,
`test_thinking_kernel.py::TestDepth3RelativeEndState::test_first_trained_read_reaches_depth3_end_state`
fails during TruthSet provisioning, before it measures trained depths. This
must not be reported as another measurement of the preceding `[1, 1, 1, 1]`
result. Both the current failure and the historical failure remain visible.

The tested source comprises 687 runtime, test and configuration files, with
manifest digest `e0b217e0340c4dc11e65c1d3c51649af77b8242f7cbac324feaaf39ef65a9f41`.
See the [source manifest](source-manifest.json),
[verified source archive](review-source.tar.gz),
[unchanged protected contracts](unchanged-contracts.json),
[receipt driver hashes](receipt-drivers.json), and
[validation summary](validation-summary.json). The final
[documentation receipt](documentation) checks the completed review prose.

| Spec §7 | Coverage |
|---|---|
| 1–9, 12 | [Selected clauses](../../../test/test_item7_acceptance.py), [row storage](../../../test/test_item7_storage.py), [owned derivations](../../../test/test_item7_clause_program.py) |
| 10 | [Eager, pending and packed row parity](../../../test/test_selected_nested_meaning.py) |
| 11 | [Provisioning and external assertion authority](../../../test/test_item7_provenance.py) |
| 13 | [Compiled clause scope and committed STM slots](../../../test/test_item7_clause_state.py), [native two-path commits](../../../test/test_sentence_compose.py) |
| 14 | [Native concept taxonomy and n-ary META](../../../test/test_item7_taxonomy.py), [direct references](../../../test/test_reference_table.py) |
| 15 | [Kind supervision](../../../test/test_item7_expectation.py), [measurement driver](measure_identity.py) |
| 16 | [Truth-row migration](../../../test/test_item7_storage.py), [native index rebuild](../../../test/test_item7_taxonomy.py) |
| 17–20 | [Bounded identity mechanisms](../../../test/test_item7_references.py), [situation context](../../../test/test_item7_context.py) |
| 21 | [Fixed small-corpus measurement](measure_identity.py) |

The selected path carries a tensor-only clause journal inside the compiled
loop. Relative seals have trial-local references until winner admission;
embedded rows receive no assertion authority. The winning host transaction
writes children before their outer clause and publishes one or three STM slots.
Original word leaves and owned actions remain the reconstruction target.

## Validation protocol

Development probes are retained, including failures and the explicitly
interrupted cross-owner probe (157 passed, 33 failed, 19 skipped). Its failures
identified retired WholeSpace APIs. Their fixture replacements exercise native
concept membership or assert that the deleted APIs are absent. Predictor-only
fixtures now inject durable meanings directly; production sentence writes
require an owned clause. No protected XOR/MM fixture or assertion was edited.
The [retired fixture map](development/20260927-item7-retired-fixture-map.json)
records the affected WholeSpace contracts; the final receipt separately lists
every added and removed selector against the preceding full sweep.

The acceptance/mechanism run passed 116 tests. The later focused fix run passed
99 tests with one skip. The 52-file affected run completed 675 cases: 656 passed,
16 skipped and three failures caused by removing two still-used Symbolize test
helpers. Restoring those helpers fixed that file, which then passed all 16
cases. Both receipts are retained.
Two later probes found a missing presented-anaphor form binding and float
roundoff in the straight-through identity factor. Both fixes pass their 21-test
focused rerun. The final affected run completed all 210 selected cases:
208 passed and two skipped. Its [selectors](affected-selectors.json) cover the
clause, reference, taxonomy, checkpoint, expectation and native composition
paths touched by the final fixes. The broader earlier selection is retained
in [affected-files.json](affected-files.json). The measurements, corrected
explicit gates and one full sweep are complete.
The first measurement set is retained under
`development/before-presented-form/measurements`; it is not represented as a
measurement of the final source.

The first explicit-gate invocation inherited `RUN_SLOW=0`, so all seven
cases skipped. That receipt is retained as `explicit` and supplies no gate
evidence. [run_explicit.py](run_explicit.py) then ran those same seven selectors
with the reviewed `RUN_SLOW=1` CPU settings and one 8-GiB worker after the full
sweep. Its [separate receipt](explicit-verified/result.json.gz) completes all
seven cases: **three passed, four failed**, in 240.50 seconds. No tests,
thresholds or seeds changed.

| Unchanged opt-in gate | Current outcome |
|---|---|
| Complete-forward single graph across runtime lengths | Passed |
| Packed pre-fold operand provenance | Passed |
| Packed per-sentence reconstruction | Passed |
| Two-epoch cross-batch graph release | Failed during reset: property analysis requires learned primitive definitions |
| XOR_grammar class accuracy | Failed: class 0 accuracy `0.0`, required `>= .5` |
| XOR_grammar reconstruction | Failed: `0/4` recovered, required `>= 50%` |
| MM_grammar XOR signal | Failed before measuring convergence: observation writer lacks its selected clause/shared index |

The XOR outcomes are current measured failures, distinct from the preceding
row-six capacity error. MM's current error is not a convergence measurement;
the historical `.21757` result against `< .20` remains linked below. The seed
was not searched or selected to make any gate pass.

The last two failing probes and their fixes are directly inspectable:
[presented subject form](development/item7_presented_form_probe.py),
[its failure](development/20260927-item7-presented-form-probe.log),
[hard identity value](development/item7_hard_identity_probe.py),
[its failure](development/20260927-item7-hard-identity-probe.log), and the
[focused rerun](development/20260927-item7-presented-form-fix-r2.log).

The reissued serial reconstruction comparison is **red**; no baseline is reset.
Its loss values are identical to the earlier development run:

| Fixed protocol | Reviewed baseline | Preceding 7.5 | Final item 7 source |
|---|---:|---:|---:|
| Before training | .10059066489338875 | .11755186505615711 | .12243161350488663 |
| During training | .09482414424419403 | .10999541729688644 | .1191720113158226 |
| After training | .09287650510668755 | .09888161532580853 | .12371796928346157 |

The measured warm throughput is 1.4530897228 sentences/second including epoch
tails. The first training batch took 565.462 seconds; the five measured warm
batches averaged 1.366896 seconds. These are observations on the existing fixed
protocol, not a new baseline or a claim of learned utility.

Strict packed/single reconstruction parity is also **red**. Both runs start
with identical parameter and dictionary hashes, retain all four sentences,
and report no truncation. The first three sentence records match exactly.
The fourth differs only in byte cost: packed `0.0005294966977089643`, single
`0.0005295205628499389` (delta `2.3865140974521637e-8`). Its roots, retained
references and dictionary hash match. Mean sentence byte costs are
`0.38411848741816357` and `0.3841184933844488`, respectively. The preceding 7.5
mean was `0.18002260848879814` in both layouts. The identity straight-through
fix did not eliminate this residual difference; it is not rounded to a pass.

### Small predictor measurements (§7.15 and §7.21)

[measure_identity.py](measure_identity.py) uses seed 42, identical initial
predictor weights and 240 updates per condition on the fixed four-case corpus.
It forces the grammatical and identity choices and evaluates those same cases.

| References | Mean role MSE before → after | Predicted verb-effect distance before → after | Correct / wrong content MSE after | Wrong-content rise after |
|---|---:|---:|---:|---:|
| Off | .13560979 → .000215475 | .07831984 → 1.40854502 | .000530221 / .250775635 | .250245415 |
| On | .18292188 → .000093236 | .07831984 → 1.40927863 | .000063000 / .250768036 | .250705036 |

Role targets differ with reference choice, so their MSEs are not a common
target comparison. Both conditions distinguish the verbs; this experiment
does not establish a benefit from references. The wrong-content comparison
uses the other lion's resting frame against the unchanged tired predicate.

On the separate mixed four-idea/two-relation fixture, mean kind BCE falls from
`.6931471825` to `.0022676998`; accuracy on those six training cases goes from
`2/3` to `1`. This demonstrates that the kind head trains on the fixture,
not held-out generalization. Raw values and the source manifest are retained
in [identity.json](identity.json).

The preceding reviewed receipt and
its depth-3 failure remain visible in the
[7.5 landing](../2026-09-27-item7-5-landing/README.md). The historical MM training
failure remains visible in [item 10](../2026-09-21-item10/README.md#validation-and-limits).

The fixed reconstruction protocol is reissued by [run_measurements.py](run_measurements.py)
using the existing item-8/item-10 drivers. The full sweep uses the preceding
file costs and unchanged resource caps via [run_full.py](run_full.py): three
workers, 8 GiB per worker, 24 GiB aggregate, 1,800 seconds per worker and
10,800 seconds overall. Long opt-in training remains excluded by the existing
suite configuration; this is not the million-sentence training campaign.

## Review boundaries

The compiled journal marks clause seals within each speculative derivation;
durable child-before-parent admission waits for the winning sentence path.
This preserves the 7.5 two-path training and winner-only commit rule. The
semantic relation has no point even where the numerical loop retains an
internal carrier for fixed-shape execution.

Location is the selected clause's first owned leaf field. Its `.when` is the
input field's shared timestamp under item 9b; this does not create a new clock
or promise a distinct timestamp for every nested S. The row's evidence poles
remain independent of these coordinates and of its point.

The identity tests cover earlier live nouns, held sentence frames, empty
situations, bounded retrieval and preservation of the current words. They do
not separately demonstrate general anaphora to every newly completed embedded
clause through the current sentence's journal. The small predictor experiment
uses forced structure and compares a wrong resting-lion frame with an unchanged
"tired" predicate target. It measures predicate surprise, not learned English
identity selection or generalization.

The frozen source still has historical comments mentioning the removed
`truthCriterion` gate in Models.py and Layers.py. The executable gate and XML
setting are absent; these stale comments are flagged for review and have not
been edited after the measurement/sweep source freeze. Required current
documentation now describes the shared seal and paired evidence directly.
