# Item 3.5, first pass: bind as thought

Branch: `codex/item3-5-first-pass`, cut after Step 0 publication.
Published bases: BasicModel `d6eb0cf3b` (source `bfc834d79`);
WikiOracle `cb4d718be`. This is step 1 of the
[ordered plan](../../plans/2026-10-19-item4-start.md#next-on-codexs-plate-after-item-4-lands-item-35-first-pass-alec-2026-10-19),
following the [concrete bind design](../../plans/2026-10-19-bind-thought-operation.md).
It is a mechanism candidate, not a learning result or a completed Item 3.5.
Final scoped validation: **334 passed, one prerequisite skip; 335/335 completed**
in 168.54 s. Both unchanged decomposition lengths pass in both gain arms
(4/4 certificates, all 8 companion questions correct). [Question tables](questions.md)
report every generated case and native companion question. Review and landing
remain open under the publish rule.
Human-readable logs/HTML have trailing whitespace normalized where needed;
their original bytes remain in adjacent `.raw.gz` files. JSON diagnostics
retain the captured output. Generated source manifests are not retained.

## Gate 0: Does the existing close retain its scorer?

| Question | Evidence |
|---|---|
| Score ownership | `ReferenceClose.score(content, predicate, cue, role, individuals)` is shared by preview and thought execution. The existing form, content, prediction, recency and role terms remain. |
| Refactor check | `score-refactor`: all 18 existing reference-close cases pass before the new sites are introduced. Their assertions are unchanged. |
| Credit | The existing resolver cost enters the thought trial's A term. R+A keeps and credits the episode; work is reported separately. Only the kept episode registers resolver training. |
| Known referents | Step 0's marginal mention objective remains. No additional scorer or category classifier is introduced. |

## Gate G1: Does a definite site choose the generator's antecedent?

| Population | Native path | Result |
|---|---|---|
| Every ordered pair of distinct kinds from `constituent_corpus`: 12 cases | Definite unary rule → closing request → shared resolver through the thought registry; one distractor | 12/12 match the generator; per-case choices and evidence pairs are in `generated-candidate/G1`. |

## Gate G2: Does an objectless form request a binding?

| Population | Native path | Result |
|---|---|---|
| 12 generated markers, each with two distinct antecedents | Admitted form without a DEF object → per-leaf request → committed NP → `closing_question` → thought resolver | 12/12 match the generator and re-emit their single request leaf; `generated-candidate/G2`. |

The form has a native identity before it has an object. Interpretation returns
that row; it does not allocate an object. The request's semantic content is
empty and its cue is the original form. The existing form index retains the
native identity, and a pending NP retains its forward-reference identity.
The field's pending-discovery path remains separate. The former contextual
bind layer and its compose/generate rules are deleted.

## Gate G3: Does a witnessed frame expose an elided participant?

| Population | Native path | Result |
|---|---|---|
| Four kinds × accessible/inaccessible situation: 8 cases | Four kept role-2 witnesses → role-keyed counts → journal opens role 2 → thought bind or unknown pair | 4/4 accessible cases fill the generator's referent; 4/4 inaccessible cases retain the open role with `(0, 0)`; `generated-candidate/G3`. |

The threshold is the components' existing recurrence value, 4. Role counts
advance only on training commits. Old unscoped association checkpoints are
converted once after semantic rows load, by rebuilding role counts from the
retained occurrence contexts and witness counts; old aggregates are not
read by a second prediction path.

## Mechanism checks: Can execution, reverse and storage share the site?

| Question | Implementation / check |
|---|---|
| Site ownership | One candidate per open referent role; an attempted site is not offered again within its episode, including an unsuccessful attempt. |
| Scope | The existing candidate-frame owner bounds columns to STM/cued frames and the episode's held occurrences. No inventory scan selects a referent. |
| Unknown | No accessible individual returns `(0, 0)` and retains the open role. |
| Evidence | A selected antecedent carries `(p, 1-p)`, its identity and its native witnesses. |
| Reverse | The registry dispatches the bind reverse face over the caller's request leaves; an elided site emits no leaves. No inverse chart rule or retained surface trace is added. |
| Kept write | The original NP is re-tied and its witnesses updated without changing its observed content. Rejected trial execution does not write the store. |
| Choice credit | A forced bind/negation fixture retains native logits, masked departure, R+A comparison and commit, and checks each trial against its actually executed resolver costs, with no work reward. The original paired zero-cost bootstrap certificate remains unchanged. |
| Landing | Candidate review precedes the one full landing sweep and publication. Neither is claimed here. |

## Diagnostic history

| Run | Outcome / disposition |
|---|---|
| `first-integration` | 132 passed, 12 failed. Corrected the recurrence owner, restricted the new leaf path to actual objectless leaves, and updated object-supplying fixtures and the operation catalogue for the intended contract. |
| `second-integration` | 185 passed, 6 failed. Added the reference evidence label to thought records. The elision generator had minted four independent individuals; it now witnesses the same individual four times. The interpretation certificate now checks both defined and objectless words. |
| `third-integration` | 221 passed, 13 failed, one skipped. All failures were fixture construction: the definite unary face needs its binary base; the replacement projection operation takes no width pair. Both constructors were corrected. |
| `candidate-boundaries` | 244 passed, two failed, three skipped. The generated gates, original reference-close cases, native kept write and paired reconstruction pass. The whole §14 file exposed missing pending-name provenance in both premise orders; its assertions and forced scope are unchanged. |
| `provenance-boundaries` | 78 passed, 15 failed, two skipped. The native NP closing exposed the premature subject-to-whole binding described below; the paired fixture also accidentally supplied a second slot when it intended a single idea. The pending-premise failures remained. |
| `formation-boundaries` | 73 passed, two failed, two skipped. All generated gates pass after the descendant-open guard and explicit-formation ownership repair. The c/e and pending-False certificates exposed the lost predicate identity at pending-row storage. |
| `predicate-boundaries` | 47 passed, one failed, two skipped. Keeping the observed predicate fixes the c/e certificate. One pending-False certificate still fails; its failure remains recorded. |
| `pending-population` | Four fresh initializations of the unchanged pending-False certificate pass; RNG states and traces are retained. This did not reproduce the preceding failure and is not a replacement for that failed run. |
| `site-owner-boundaries` | 108 passed, 19 failed, one skipped. The whole §14 file passes 12 cases with the existing copular skip. Eighteen definition fixtures still assumed that a form automatically creates an object; the new site-ownership fixture also used the prohibited temporary zero as a leaf identity. |
| `definition-boundary` | 93/93 pass after supplying the intended object witnesses in those fixtures and placing the temporary open choice in the operation journal. Native-read checks explicitly require an unwitnessed form to remain without a DEF object. |
| `final-scoped` | 319 passed, one failed, one skipped, all 321 completed. The whole §14 file, paired reconstruction and bind gates pass. The frozen not/or reconstruction certificate exposes the missing lexical readback for objectless forms. |
| `lexical-readback` | 59 passed, two failed, all 61 completed. The lexical repair passes that frozen certificate and the new no-allocation readback/teacher-target regression. Two enabled slow gradient certificates hit the observer's old uniform-only scale assertion. |
| `proposal-before` / `proposal-observed` | The first diagnostic invocation omitted `RUN_SLOW=1` and skipped both cases; it is not evidence. With slow execution enabled, both cases reproduce the observer discrepancy at their unchanged initialization; `proposal-before.jsonl` records the numbers. |
| `readback-credit` | 36/36 pass, including both unchanged gradient assertions, the frozen not/or reconstruction certificate, original paired reconstruction, generation lesson and lexical boundaries. |
| `native-final` | 70 passed, one copular-artifact prerequisite skip; the whole §14 file completed. This preceded the operand-cache repair below. |
| `decomposition-zero` | Both cases stopped at the saved Item 4 constructor-equality guard, before certificate execution. Adding the new thought operation changes the catalogue and initial parameter layout; this is not evidence of decomposition execution. |
| `bind-decomposition-zero` | Both unchanged certificates ran under the new catalogue and failed. The separate 3.5 comparison uses the original measurement seeds and matches its zero/enabled constructors; it does not claim equality with Item 4's constructor. |
| `binding-before`, `binding-predicate`, `binding-cache`, `binding-close` | Each reproduces the one-successor failure at seed 4199722147. The progressively narrower closing/identity traces are preserved, including the unsuccessful intermediate predicate and cache repairs. |
| `predicate-role` | 67/67 scoped cases passed while the native decomposition still failed. Passing this scope was insufficient. |
| `predicate-cache` | 66 passed, one new paired-bind fixture failed. It incorrectly assumed that a live departure executes the same bind in both trials. The fixture now compares each trial with its actual R+A components, and the separate original bootstrap tie assertion is unchanged. |
| `alias-boundaries` | 68/68 pass, including the exact unresolved-reference regression, both cached-operand cases and all three generated site gates. |
| `binding-alias` | The unchanged one-successor certificate passes at the original seed 4199722147 after the admitted-alias repair. |
| `candidate-scoped` | 334 passed, one prerequisite skip; all 335 complete. Includes whole §14 (12 passed, copular artifact skipped), original paired reconstruction, all generated bind sites, reference-close and mention-ranking checks, lexical readback and exact/finite-difference chooser gradients. |
| `bind-decomposition-{zero,enabled}-candidate` | Both lengths pass in both arms, with all eight companion questions correct and 1/2 inferences as required. All 121 initial parameters match between arms. Seeds remain 4199722147 and 2428440506; the changed catalogue differs in 47 parameter entries from Item 4, explicitly reported in `bind-decomposition-metrics`. |

The pending-premise regression uses a zero thought budget: no bind executor
can have closed the variable. Objectless forms were absent from the existing
form index used by lexical consumers, and their new NP closing request
dropped the forward identity. Admission now registers the form's own identity,
and unresolved NP requests retain that identity through prepare/commit.
The production correction does not alter the forcing helper.

The native NP closing also revealed two ownership errors. A closed containing
phrase could publish its subject form as a particular while its NP child
was still open. `bind_particular` now checks descendant and companion open
slots before publishing that association. An explicit ordinary `open`
formation could acquire a second automatic NP request, changing both the
stored structure and resolver cost. The explicit site now retains the leaf's
cue and empty content without adding that second request. An explicit mint
or selected antecedent likewise owns its site.

When a pending compound is stored as a relation, its observed predicate
may have a point but no dictionary identity. The journal now ends that point
as an ordinary child occurrence. It no longer converts the observed
predicate into a new unknown relation. The isolated pending-subject regression
checks the stored reference; the original §14 certificates remain unchanged.

## Lexical and credit boundaries

| Question | Cause / repair |
|---|---|
| Can an objectless form emit its own spelling? | The decoder's lexical inverse consulted only DEF objects. `Interpret` now owns the lexical view of both witnessed objects and still-objectless admitted forms. Generation and teacher targets read that view without allocating an object or writing a definition. The frozen not/or certificate keeps its assertions and initialization. |
| Why did the gradient observer reject the proposal scale? | It asserted `W × rounds × K`, the uniform-action law. At the existing initialization it saw **123.9845199584961**, against `2 × 31 × 2 = 124`. Production uses the decided φ=.5 mixture, so the correction is `W × rounds / q(action)`. |
| What changed in that observer? | It captures the actual expectation-free proposal logits, independently reconstructs the uniform/policy mixture, and checks its reciprocal correction. It still checks the exact chooser gradient and the finite-difference derivative. The production proposal, initialization and training budgets are unchanged. |

## Native decomposition: why did the new form path lose the proof?

| Question | Cause / repair |
|---|---|
| What was lost? | The ended VP's first cached operand became `-1`; the other operand also lost its raw alias. The bounded component reader therefore had no complete operands to substitute, even though the episode returned one inference. The final variable remained bound to the unrewritten compound. |
| Why? | A live raw form ID equalled `concept_ids[leaf]`, and its reference-order stamp was nonnegative. The old admitted-singleton alias test therefore replaced it with `reference_ids[leaf]`, even when that object address was still `-1`. Objectless forms make that state ordinary. |
| What changed? | An admitted alias must have a valid native address before it can replace the selected form. Non-reference operands retain known identities when their operation has no binding choice, and a predicate operand is not closed as an objectless participant. |
| What is the regression? | The generated closing test has both a missing binding-choice entry and a real selected form whose object-reference column is `-1`. It checks the stored VP's native operand, not a vector-nearest match. |
| What stayed fixed? | The original decomposition certificate, its assertions, forced requests, budgets, R+A keep rule and measurement seeds. The original Step 0 landing remains separately measured. |

## Explicitly carried work

| Owner | Remaining work |
|---|---|
| This candidate | Claude's review, review repairs and one full landing sweep, then publication and the WikiOracle bump. The scoped native boundaries are complete; they are not a full sweep. |
| Item 3.5 step 2 | LTM1/LTM3 layout, three-position row, recurrence minting and the rebuildable definition cache. |
| Item 3.5 step 3 | Stage 4e → NP continuation → identity lesson → producing artifact and the unchanged copular certificate. The prerequisite skip is not a pass. |
| Item 3.5 steps 4–6 | Verb group-element teacher; tense/aspect variants and second difference/world; remaining 5.5 mechanisms, in that order, each with its own receipt. |
| Item 0 | Graded versus shuffled, derivation-optimality, long-run per-level collapse and the original failed native learning run. No regrade or rescue training. |
| Item 5.5 | Separate learning gates remain separate. |

## Reproduction and handoff

`run_scoped.py` records the affected file scope. `run_native_boundaries.py
--decomposition-only --suffix <new-name>` runs both unchanged decomposition
lengths in both declared gain arms and refuses to overwrite a prior run.
Use the repository virtual environment from the BasicModel root.
`bind_gate.py` records the two configurations and checks matched initial
parameters; the raw closing, component and question diagnostics stay in files.
The earlier failed attempts remain separate evidence.

This commit is the step-1 review candidate. The next action is Claude's code
review required by `todo.md`'s publish rule, followed by repairs, one landing
sweep and publication. Steps 2–6 remain in the ordered plan; no phase lesson
or learning gate is marked complete by these mechanism checks.
