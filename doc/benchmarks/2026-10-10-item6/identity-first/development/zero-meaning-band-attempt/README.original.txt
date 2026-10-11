# Item 6 part 2: first native experiment and review follow-ups M/N/O/I

The preparation landed as `06f6e2b88` after [plan §5](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#5-review-of-the-follow-up-landing-and-the-part-2-preparation-claude-2026-10-10), with WikiOracle bump `56544f1`.
This next candidate is not yet reviewed or committed. This is the single receipt
for the correctness repairs, revised corpus and first matched experiment.
Results below are updated from completed artifacts; an unfinished or failed
run never certifies learning or rule retirement.

## Changes before learning

M: sentence reconstruction uses the trial field's actual admitted leaves.
The prior mask applies only outside that sentence. The committed winner uses
the same mask as its scored trial. The retained passive-context regression
covers all four packed rows, including source order `[4,0,1,2,3,5,6,7]`, during
scoring and after releasing the field. The old mask still reproduces binary
underflow in that test; the trial-owned mask preserves every pushed operand.

N: the [revision 3 corpus](../../../../data/identity_from_data/README.md) uses
`a cat runs . a cat sleeps . the black cat runs . the white cat sleeps .`.
Later properties reuse the original verb forms; there are no participles.

O: object fronting is dropped. There is no claimed topicalization derivation.
Introduction role and first-mention position are confounded in the ordinary
SVO introductions. Refresh mentions still cross recency with introduction role:
four cells, eight presentations each. Duplicate witnesses preserve the accepted
presentation budgets. The three-candidate role marginal and thin document-context
coverage remain limitations. No stage-7 interruption/shared-truth claim is made.

I: the exact-fit set restricts the hard choice, then the decomposition chooser's
learned logits rank its members. Raw residual argmin no longer overrides that
choice. The approximate path retains its existing residual/context score and
teacher loss. No side, grammatical family, word or positional ordering fallback
is added. A permutation regression gives a slightly worse but exact-fit pair
the better learned score and verifies that it wins in three bank orders.

The unchanged [bisect probe](../followup/bisect/probe.py) still gives **0.000**
positional overlap on the cold model ([raw result](bisect/probe.json)). Its
chooser starts with weights `[1,0,0,0,0]`: only residual, no learned orientation.
The repair makes a learned score effective; it does not invent an order for an
untrained symmetric pair. Accepted 6.1 gave 1.000 on this artifact; the adverse
XPASS-to-XFAIL remains open. Removing the live near-copy still gives 1.000.

A [separate frozen-artifact teaching diagnostic](bisect/learned-probe.json)
updates only the pair chooser for 64 presentations (SGD .1). Its CE falls
from 1.837530 to .374767 and the repaired path reaches **1.000** overlap.
Replaying those same learned weights with the old residual override remains
**0.000**. This isolates I's obstruction; it is not the native curriculum's
order-learning certificate. The [driver](bisect/learned_probe.py) preserves
the frozen source artifact and labels its explicit word-pair supervision.

## Declared experiment

[Driver](../../../../test/identity_experiment.py),
[metric checks](../../../../test/test_identity_experiment.py).
Measurement seed 20261010; no seed search or retry to obtain a pass. The first
attempt stopped on a retrieval correctness defect described below. Its entire
receipt remains in `development/taxonomy-underflow-attempt`; a corrected-source
run keeps the same seed, presentation budgets, data and learning parameters.
Stages 1/2 present/re-present words and novel words; stage 3 gives 64
presentations each of two-word and four-word lessons; stages 4a–4d each get
eight sentence and eight supplied-answer presentations. The fixed diagnostic
budget continues after a failed stage, without claiming that stage passed.
There is no fresh disabled-attention training control in this experiment.

All eight branches restore the same warm-up model checkpoint and RNG, then
create fresh optimizers at the same learning rate. Inspection of the staged
checkpoint found `optimizer_state=None`: this harness passes an explicit local
optimizer to `runBatch`, whereas `save_weights` reads the model's `_optimizer`.
The archived driver's opening shorthand "including RNG/optimizer" must not
be read as a claim that warm-up optimizer moments survive; the continuation
corrects that docstring. This is a common optimizer
reset on all branches, not an uninterrupted-optimizer continuation experiment.
The checkpoint smoke check exposed native loader codebook drift; the measurement
adapter records it and restores every differing tensor from the frozen checkpoint,
then checks all tensor hashes. This is measurement isolation, not a checkpoint
retention repair. Selected grammar and references are never overwritten.

Factorial and confounded branches each receive 144 text documents with support
1→2→3. A balanced quarter of the factorial combinations is held out: 48 rows.
Confounded training replaces each object's property with its fixed partner,
with equal document, support and verb budgets. Grading labels never enter a
native presentation. Identification counts a present atom only if a nonzero
column in that row's encoding matches it at absolute cosine ≥ .9. Missing
meanings and zero atom vectors are misses. A fused idea is scored as its completed
point, including its predicate, matching the native component objective; a
relative clause retains the native NP superposition. A regression distinguishes
this from mistakenly scoring the pre-fusion NP pair. Nearest-column cosine is not the
accuracy metric. Merges are columns matching a kind+property sum at ≥ .9
while matching no single atom at that threshold; this diagnostic never merges
or deletes production columns.

The [budget audit](protocol-audit.json) records 240 held-out atom occurrences.
Factorial training has 144 unique texts; confounded training has 48 texts,
each presented three times under distinct document addresses. The native
recurrence requirement remains four distinct witnesses. Repeated text alone
therefore does not guarantee admission within this budget; shared semantic
residuals must recur across different frames. A null result cannot establish
that a longer exposure budget would also fail.

The only intervention is `require_selected_mint=False` at component commit.
It removes both the selected-individual guard and the selected-mint recurrence
gate for the ablation; recurrence, surprise residual, sparse inference and
all reference rules are unchanged. Default production keeps the reviewed rule.
The declared retirement gate is factorial identification ≥ .9 overall and for
each kind, better without the gate, with a confounded identification loss and
at least one merged pair. Otherwise the rule stays.

This is a comparison of admission with and without the selected-mint gate on
the same corrected source. It is not a trained-checkpoint comparison against
6.1, and it does not replace the remaining determiner, pronoun or positional
binding rules. A positive admission result would support that rule's lesson;
it would not by itself establish successful unified NP and identity learning.

The earlier [five-seed toy](../../2026-10-10-identity-ica-toy/README.md)
provides separate evidence for the adopted unmixing mechanism: on factorial
linear mixtures, residual minting with greedy selection raised identification
from .02 ± .01 to .95 ± .03. Those numbers are not native-model results. The
same toy found that identical-content individuals need determiner or later
distinguishing evidence; unmixing alone cannot separate them.

Balanced and biased identity branches each get 16 prediction-vocabulary,
32 pronoun, 80 determiner and eight same-kind presentations. They start from
the same warm-up state, independently of the recovery branches. Each evaluates
150 ordinary/control documents and 16 reversed-recency documents. The native
bank audit records the actual pre-compose candidate IDs alongside journal
candidate availability. Row accuracy, informative binding, cue conflicts and
uninformative controls remain distinct. Neither unresolved references nor noisy,
same-content determiner cases certify a rule's retirement.
The biased branch repeats 16 pronoun document IDs twice while the balanced
branch has 32 IDs: presentation budgets match, document novelty does not.
Admission-call counters include frozen evaluation calls; minted-column counts
are taken immediately after training and before evaluation.
Stage 5's accepted row-pair metric grades the first two mentions. The later
property mentions in same-kind lessons are present in the text and retained
observation provenance, but their individual links are not separately scored
by that metric. A correct row count alone cannot certify those later bindings.

## Outcome

The [first sweep](development/taxonomy-underflow-attempt/full/summary.json)
completed **5,769/5,769** cases: **5,480 passed, 286 skipped, two failed, one
xfailed**, in 981.87 seconds. It is retained even though a subsequent native
learning exception required a new source change and final-source sweep.
The retrieval repair's affected checks passed 132 tests with the unchanged
overlap xfail; the completed-row metric checks passed three tests.

The [post-retrieval-repair sweep](development/strict-checkpoint-attempt/full/summary.json) completed **5,776/5,776**:
**5,486 passed, 286 skipped, three failed, one xfailed**, in 972.36 seconds,
with 9.27 GiB peak aggregate worker memory. Its 771-file source manifest
matches that candidate. [Raw reports/logs](development/strict-checkpoint-attempt/full/logs.tar.gz),
[HTML report](development/strict-checkpoint-attempt/full/report.html),
[baseline comparison](development/strict-checkpoint-attempt/full/baseline-comparison.json).
The supplied-answer `five` failure persists. Two **new documentation failures**
caught paths moved while archiving the first attempt: the receipt's old full
links and the original README's relative links at its archive location. The
links are repaired, the original README is preserved verbatim as `.txt`, and
the separate [365-case link check](validation/doc-links-packaging.log) passes.
Those failures remain in the raw full sweep; its counts are not rewritten.

The [final sweep after the restoration-adapter correction](full/summary.json)
completed **5,776/5,776**: **5,488 passed, 286 skipped, one failed and one
xfailed**, in 1,010.50 seconds, with 9.11 GiB peak aggregate worker memory.
All 771 source files match. [Raw reports/logs](full/logs.tar.gz),
[HTML report](full/report.html), [baseline comparison](full/baseline-comparison.json).
The only failure is the retained supplied-answer `five` case. There is no new
failing node in this final sweep relative to accepted 6.1; the first attempt's
new binding-distribution failure and the later documentation failures remain
explicitly recorded in the comparison, as does the adverse overlap XPASS-to-XFAIL.

Against accepted 6.1 (`7c9aeac67`, 5,719 selected / 5,428 passed / 286 skipped /
four failed / one XPASS), the supplied-answer `five` failure persists.
**New in this candidate's failing-node list:**
`test_ordinary_initial_binding_distribution_includes_every_retained_candidate`.
Its measured range is .848586–1.179175 against the unchanged .9–1.1 bound.
The earlier receipt already recorded this initialization-sensitive finding;
that history is not a reason to hide its new status relative to the accepted
6.1 sweep. It passes the corrected-source sweep, which does not close this
initialization-sensitive finding. The other three accepted-baseline failing nodes pass here; their
historical findings remain open. The overlap XPASS-to-XFAIL remains adverse,
separate from the successful learned-chooser reachability diagnostic.

The [first warm-up](development/taxonomy-underflow-attempt/experiment/warmup/failure.json)
stopped on presentation 8 of the four-word lesson: a `sym` cue that is not a
positive concept handle entered the taxonomy order reader. Stages 1/2 passed
the recognition checks; the two-word CE stayed .3465735912 through all 64
presentations and its free readback was 0/4. Seven four-word presentations
completed. Stages 4 and all eight experiment branches were **not run**, not
zero-identification results. That attempt retained no checkpoint or exact bad
cue, so its traceback does not establish the invalid handle's value or origin.

The corrected source permits only positive integer concept handles into the
concept-order reader; other cues still unfold their actual vector payload.
Six invalid-address cases verify that boundary, and each also verifies a valid
concept's order remains available. The corrected run saves stage checkpoints
and audits the first invalid native cue for provenance. This changes retrieval
correctness, not the learning objective or rule-retirement criteria. The final
source sweep for that attempt is complete. Its warm-up subsequently completed,
but all branch starts failed strict checkpoint loading as described below.
No rule retired.

The corrected run passed four-word presentation 8 and captured
`('sym', -6431577289414244955)`. The [native cue](retrieval/unresolved-cue.json)
and [tensor payload](retrieval/unresolved-cue.pt) are preserved. A
[boundary replay](retrieval/cue-probe.json) of this exact cue through the old
reader reproduces the taxonomy `TypeError`; the corrected reader unfolds all
three occupied role payloads and sends only the positive handle to the order
reader. This is a boundary replay, not a full sentence replay. The repair
does not retag the negative address or establish why its producer supplied
`sym`; that provenance remains an open typed-reference finding.

The corrected run's two-word stage completed 64 presentations with final CE
.3465735912 on each row and derivation/free readback 0/4. Its initial small
CE decrease (.346491 minimum) did not survive to the final presentation.
The four-word stage also completed 64 presentations, with final CE .7945134640
on each row and derivation/free readback 0/4. Neither order-CE gate passed.

All four stage-4 reconstruction checks and all four supplied-answer checks
finished at 0/4. The shared warm-up completed in the retained attempt, but its
eight branch starts failed before presenting any training document: strict
parent loading rejected the 16 lazily created comparison-reader keys before
the child's existing load hook could materialize them. The completed warm-up,
failed starts and then-current source are preserved in
`development/strict-checkpoint-attempt` (5,501.95 seconds; source and corpus
matched). These are unrun identification/binding lessons, not zero scores.

The continuation's experiment adapter invokes that child's strict loader
before the parent's strict loader. It drops no saved keys, supplies no example
and leaves production loading unchanged. The [retained-checkpoint audit](validation/warmup-restore.json)
matches all 874 saved tensors, including the 16 comparison weights, with no
extra tensors. This checkpoint has zero noun columns. Seven adapter/reader
checks pass. A [restored-state audit](validation/learning-switch.json) confirms
online learning is enabled before branch training; the explicit freeze is
applied only for held-out evaluation. The initial regression invocation accidentally used the tiny
fixture's disabled loader; its failure is retained, and the corrected test
captures the native loader before constructing the fixture, as the branch
driver already does.

The [continuation harness](resume_branches.py) checks that the learner and
corpus are unchanged since warm-up; only the experiment restoration code and
its regression differ. It copies the same frozen checkpoint and records both
source manifests. There are no additional warm-up presentations or altered
learning budgets. The [exact source delta](validation/continuation-source.diff)
separates this adapter correction from the unchanged learner. The final-source
sweep is complete; the learning continuation is in progress.
