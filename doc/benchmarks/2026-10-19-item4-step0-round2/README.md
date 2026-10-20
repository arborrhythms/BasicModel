# Step 0, second repair round

Branch: `codex/item4-step0`, from reviewed candidate `c4e3520a2`.
Landing source: **`bfc834d79359efd6be493d401e92f87732b18acf`**.
The closing receipt follows that source; both mains advance to this landing
before `codex/item3-5-first-pass` starts.
This directory owns this round's diagnostics and gates. The first-round
[receipt](../2026-10-19-item4-step0/README.md) remains the record of its
measurements, including failures. No learning run is resumed or regraded.
Human-readable log/HTML whitespace is normalized where needed; original
bytes remain beside changed files as `.raw.gz`. JSON diagnostics retain
their captured output.

## Gate 3 — Which initial distribution does certificate (b) measure?

| Question | Decision / evidence |
|---|---|
| Production prime | ON, uncapped. The binding-specific `log(1.1)` rescaling is deleted. |
| Measured configuration | `test/fixtures/item4-parity.json`: P/S/R/D zero, prime off, previous proposal law. This is distinct from the two standing-fixture arms, which both keep the prime on and φ=.5. |
| Measured logits | The existing chooser's content score, observed before form/occupation priors and expectation bias. Selection still uses the full ordinary score. Both full and measured conditional probabilities are reported. |
| Band and coverage | [.9, 1.1] unchanged, covering open, mint and every retained candidate. The ordinary document audit and six retained-bank contexts remain. |
| Initialization | Accepted zero candidate-value columns/reference projection retained. Open/mint retain their learned scoring paths. No initialization or seed is changed for the decomposition certificates. |
| Result | `scoped-second`: 5,204 menus, 3,864 retained alternatives; relative-to-uniform range **1.0–1.000000030**. 32 ordinary provisioning documents, eight ordinary questions, six controlled retained-bank contexts. |
| Final source after the sweep repair | `post-sweep`: 3,320 menus, 3,864 retained alternatives, the same **1.0–1.000000030** range. The unseeded ordinary menus differ; the declared provisioning, retained-bank coverage and threshold are unchanged. |

## Reference supervision — How is a known antecedent credited?

| Question | Implementation / evidence |
|---|---|
| Objective | Negative log marginal probability of all known correct antecedents, using the existing factored resolver score. |
| Updated owners | Content weight and form/cue weights. Prediction, recency and role weights condition the distribution with detached weights in this supervised path. |
| Supplied answer | Loss-side label, restricted to the answer's reference roles; it cannot change the already selected menu or action. |
| Later fill | The occurrence retains detached scoring inputs. A fill of that same occurrence and role re-evaluates the existing scorer and registers expectation-owned training. No live graph or second pending-reference cache is stored. |
| Missing label / missing correct candidate | No invented target or supervised update; ordinary credit remains. |
| Evidence | Four `test_reference_mentions.py` cases pass: multiple correct antecedents, gradient ownership, supplied-answer role restriction, later native `fill_slot`, and exclusion of training provenance from thought features. |

## Gate 1 — Why did A favour the wrong one-successor trial?

| Question | Finding / evidence |
|---|---|
| Initialization | Recorded seed **4199722147**, derived from the first-round nonce; exact original constructor parameter comparison. |
| Cause | Sentence A was the surface reader's error on the whole question **before** the closing episode. A smaller surface error could prefer a wrong reading. The completed episode's binding scorer was not the source of the reported 1.92. |
| Evidence-slot hypothesis | Not reproduced: no retired evidence slot causes this comparison. Before admission the journal's semantic references are temporary nulls; the native writer installs the chosen references. The episode runs only after sentence selection. |
| Repair | Binding tasks use the existing referent-cost rule at both judges: unfilled 1, incorrect filled `.5 + .5d/(1+d)`, exact identity 0. The two surface readers retain their output-training graphs. R+A, the assertion, forcing, initialization and budgets are unchanged. |
| Fixed-initialization diagnostic | `answer-scoring-repair/questions.jsonl`: both documents have a correct binding, one witnessed inference, and the required descent/return. Work 208/211. This diagnostic is followed by the unchanged certificates below. |
| Detailed components | [answer-cost-cause.md](answer-cost-cause.md); raw before/after reader and episode traces in this directory. |
| Both decomposition certificates, both gain arms | **4/4 pass**; all eight companion questions have correct bindings and the required witnessed chains. Constructor parameters match the first-round initialization exactly in all four runs. |

## Thought menu — Can an unadvanceable open reference conclude?

| Question | Finding / evidence |
|---|---|
| Original loop | The original document-0 episode exhausts 512 work units with 171 `not` events and no inference. The meter includes controller/executor work; 512 is the budget, not the event count. |
| Mechanism | `needs_episode` excluded conclude even when the only available operation was self-inverse `not`, which cannot supply the missing referent. |
| Repair | Such a menu includes conclude while leaving the question open. This is not evidence of an exhausted search and does not mint an individual. |
| Regression | A budget-512 open question with only negation available concludes in **two work units**, without filling a reference. An initial conclude now terminates the context immediately instead of being selected twice. |

## Landing and explicitly retained work

| Question | Status / owner |
|---|---|
| Final affected boundaries | `native-boundaries`: **475/475 pass**, including the original paired reconstruction certificate, controller/credit checks, all four eager/compiled gain-boundary cases and documentation links. |
| One full landing sweep | Completed **5,861/5,861**: **5,540 passed, 322 skipped, one existing XFAIL, one failure**. Both common-constructor PerceptMoments coverage and the original paired reconstruction certificate pass. Runtime 1,353.24 s; peak worker/aggregate memory 2.88/7.55 GiB within 8/28 GiB limits. Full result in `landing-sweep/result.json`, compact [summary](landing-summary.json). |
| Repair after the sweep | The empty-menu regression below has a one-condition repair; the whole §14 file and affected controller/reference checks complete **113 passed, one copular skip**. Both unchanged decomposition certificates then pass in both original arms, **4/4**. No second full sweep. |
| Publication | Source `bfc834d79` plus this closing receipt: branch push and fast-forward BasicModel main, followed by the WikiOracle submodule bump and fast-forward of its main. All BasicModel working-tree edits, including Alec's newer plan/spec/todo edits, are included. |
| Item 3.5 | Starts only after that publication, on a separate branch, in the plan's listed order. |
| Copular identity certificate | Remains prerequisite-skipped until its producing lesson, after stage 4e in Item 3.5. No one-hop reader shortcut. |
| Item 0 | Graded versus shuffled §9.6, derivation-optimality rate and long-run collapse/exposure remain explicitly carried. |
| Item 5.5 | Separate learning track remains carried; no rescue training here. |

| Landing-sweep finding | Cause / disposition |
|---|---|
| `test_no_legal_thought_operation_finishes_with_the_reference_open` | The new `not advances` condition was also true for an empty operation menu, incorrectly offering a conclude event. Requiring a nonempty menu preserves the existing empty-menu finish (`no_legal_operation`, reference open, one work unit), while retaining the negation-only escape. The original certificate and assertion are unchanged. |

Two diagnostic preflights (`answer-before`, `answer-components-before`) stopped
on diagnostic serialization errors. Their logs remain; neither is a gate result.

`scoped-first` completed 47 cases: 44 passes, two failures and the copular
skip. The new initial-conclude regression exposed the duplicate terminal
event; the c/e certificate also reported a query without its expected
conclude. After the terminal fix and excluding detached mention-training
provenance from thought inputs, `closing-diagnostic` passes both cases and
`scoped-second` completes all 48 cases: **47 passes, one copular skip**.
The first results remain visible and are not replaced.

| Native question | Gains zero: binding / chain / work | P=S=R=D=.5: binding / chain / work |
|---|---|---|
| One successor, document 0 | correct / correct / 208 | correct / correct / 208 |
| One successor, document 1 | correct / correct / 211 | correct / correct / 211 |
| Two successors, document 0 | correct / correct / 560 | correct / correct / 560 |
| Two successors, document 1 | correct / correct / 570 | correct / correct / 570 |

Both arms keep prime ON and φ=.5. The original assertion, live departure,
forced scope, initialization and budgets are unchanged. Complete companion
reports are under `decomposition-{zero,enabled}-reports`; resource/result
receipts are under `decomposition-{zero,enabled}`. The accepted first-round
16/16 XOR measurements in each arm remain in their original receipt.
The final `post-sweep-decomposition-{zero,enabled}` runs reproduce every
entry in this table after the empty-menu repair; all 121 constructor
parameters match the original runs, and all eight companion questions pass.
