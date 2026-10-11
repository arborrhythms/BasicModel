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
Measurement seed 20261010; one run, no seed search or retry to obtain a pass.
Stages 1/2 present/re-present words and novel words; stage 3 gives 64
presentations each of two-word and four-word lessons; stages 4a–4d each get
eight sentence and eight supplied-answer presentations. The fixed diagnostic
budget continues after a failed stage, without claiming that stage passed.
There is no fresh disabled-attention training control in this experiment.

All eight branches restore the same warm-up checkpoint, RNG and optimizer.
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
meanings and zero atom vectors are misses. Nearest-column cosine is not the
accuracy metric. Merges are columns matching a kind+property sum at ≥ .9
while matching no single atom at that threshold; this diagnostic never merges
or deletes production columns.

The only intervention is `require_selected_mint=False` at component commit.
It removes both the selected-individual guard and the selected-mint recurrence
gate for the ablation; recurrence, surprise residual, sparse inference and
all reference rules are unchanged. Default production keeps the reviewed rule.
The declared retirement gate is factorial identification ≥ .9 overall and for
each kind, better without the gate, with a confounded identification loss and
at least one merged pair. Otherwise the rule stays.

Balanced and biased identity branches each get 16 prediction-vocabulary,
32 pronoun, 80 determiner and eight same-kind presentations. They start from
the same warm-up state, independently of the recovery branches. Each evaluates
150 ordinary/control documents and 16 reversed-recency documents. The native
bank audit records the actual pre-compose candidate IDs alongside journal
candidate availability. Row accuracy, informative binding, cue conflicts and
uninformative controls remain distinct. Neither unresolved references nor noisy,
same-content determiner cases certify a rule's retirement.

## Outcome

The [full sweep](full/summary.json) completed **5,769/5,769** cases on the
frozen source: **5,480 passed, 286 skipped, two failed, one xfailed**, in
981.87 seconds. [Raw reports and logs](full/logs.tar.gz),
[HTML report](full/report.html), [baseline comparison](full/baseline-comparison.json).
The affected checks passed 103 tests with the unchanged overlap xfail;
documentation links passed 365 checks. No full sweep was rerun to select a
better initialization.

Against accepted 6.1 (`7c9aeac67`, 5,719 selected / 5,428 passed / 286 skipped /
four failed / one XPASS), the supplied-answer `five` failure persists.
**New in this candidate's failing-node list:**
`test_ordinary_initial_binding_distribution_includes_every_retained_candidate`.
Its measured range is .848586–1.179175 against the unchanged .9–1.1 bound.
The earlier receipt already recorded this initialization-sensitive finding;
that history is not a reason to hide its new status relative to the accepted
6.1 sweep. The other three accepted-baseline failing nodes pass here; their
historical findings remain open. The overlap XPASS-to-XFAIL remains adverse,
separate from the successful learned-chooser reachability diagnostic.

The warm-up and matched learning branches are still running. No rule retired.
