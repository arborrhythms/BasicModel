# Item 6 part 2: R/S/T repair and matched rerun

The reviewed M/N/O/I preparation landed as `af634ed35`, with landing record
`f6d04afa5` and WikiOracle bump `7af3fbf`. It makes no learning claim.
This next pass implements [plan §7.4](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#74-hand-off-to-codex-alec-confirmed-2026-10-11).
The matched measurement is complete and accepted for landing in plan §8.4
(Alec, 2026-10-11), with no learning claim. R’s requested retention outcome
remains unmet. No rule is retired; R2 precedes the parts-of-speech pass.

## Changes

**R:** `reconstruction.content` compares the free reconstruction's meaning
bands with the clause's own leaves. It sums the positive and negative mass
of each coordinate separately across leaves, preserving multiplicity and
preventing opposing values from cancelling. Its baseline is those leaves'
absolute meaning, and its eligibility mask is the existing sentence
availability mask. Neither a perfect spelling nor a trace inverse with
undefined coordinates can hide this term. Targets are detached; the returned
content keeps its ordinary gradient. No operator is banned and no referent
mode changes. The coverage is invariant to leaf permutation; the order lesson
supplies orientation credit. This is a meaning reconstruction objective; its inverse reader
can still fail even when the closed row retains meaning. The held-out program
measurement below distinguishes those outcomes.

The [retained four-clause replay](validation/recovery-retained/result.json)
keeps the original checkpoint and attention scorer. All four closed meanings
remain zero, as expected without retraining; the added content term charges
1.0 relative error on every row ([raw costs](validation/recovery-retained/costs.jsonl)).
This verifies that the old reading now pays for its lost meaning. It does
not establish that a newly trained policy stops selecting it.

**R's requested retention outcome fails in the matched rerun.** The
[new factorial-off checkpoint replay](validation/recovery-summary.json)
has zero closed-row meaning in all 48 held-out clauses, and all 48 still
contain `lower` followed by `non`,`non`. Every row pays a relative content
charge of approximately 1.0; it is not hidden by the availability mask.
The cost repair is present, but the trained policy has not escaped the
discarding reading. [Replay archive](validation/recovery-archive-manifest.json).

A [four-clause paired-walk diagnostic](validation/keep-credit/result.json)
uses a discarded checkpoint copy, zero learning rate and frozen conceptual
admission; it verifies that no trainable parameter changed. Both sampled
readings have zero meaning and a content charge of 1.0 in every row. Total
keep costs are 1.85/1.85 in three rows and 1.85/2.05 in the fourth, so all
keep the greedy reading. The new content term gives no preference between
these sampled alternatives. This is a local diagnostic, not a claim that
every alternative during training tied or an extra learning branch.
[Diagnostic archive](validation/keep-credit-archive-manifest.json).

**S:** the candidate scorer bounds the shared context's L2 norm by one before
its existing MLP. The candidate features and all checkpoint parameters keep
their existing shapes. This removes the context-width contribution to
saturation without supplying an order or target position. On the archived
first read, the same saved weights now give preactivations −1.796 to 1.790,
zero tanh units at ±1, and four candidate logit gaps .00483–.00696. The raw
input reproduces exact saturation in the regression. The separate 64-step
native continuation uses the saved two-word checkpoint and a fresh optimizer;
its state is discarded before the matched experiment.

The [completed retained-checkpoint continuation](validation/attention-summary.json)
has zero saturated tanh units throughout all 64 presentations and both
untaught reads. The training CE goes from .34483–.34537 to .02591–.02610,
away from the previous log(2)/2 plateau (.34657359), but still above the
declared .01 order gate. Final untaught-read logit gaps are 2.95361–2.98685;
trace readback remains 4/4 and free readback remains 0/4 (the last word only).
Untaught reads do not emit lesson CE; the reported CE comes from the native
training presentations. This establishes the requested saturation repair
and stage movement, not free reconstruction or identity learning.
[Byte-verified raw archive](validation/attention-archive-manifest.json).

**T:** the observer separates the selected raw reference from the address
exposed to grading. A mint is resolved only to a uniquely matched, written
individual; absent or ambiguous individuals have named missing reasons.
No sentinel reaches the grading reference map. A mechanism fixture writes
an owned individual, then binds a later mention to its actual address and
grades the chain as one row. Another fixture verifies that a projected-away
individual cannot be assigned the final head's address.

The retained failure is more specific than an address plumbing failure:
the first replayed introductions write only a punctuation-headed row. Their
requested cat/bird individuals are absent from the writer tree. The observer
must report that absence, not invent their rows. The 690-reference replay
uses the retained final checkpoint and its original unnormalized scorer so
that S does not change the choices being audited; it is an observer
diagnostic, not a new learning run or a reconstruction of the old initial
evaluation state.

The [completed observer audit](validation/observer-summary.json) accounts for
all 690 annotated references, with **zero grading sentinels**. None resolves:
445 requested mints have no retained individual in the written clause, 243
mentions have no individual-reference request, and two leaves were not
admitted. All 690 original selected references match the previous run's raw
capture. This localizes the old zero scores to missing written individuals
and missing requests; it does not turn those missing choices into binding
errors with identified candidates. The passing explicit mint→bind mechanism
regression proves that a written chain can be graded, not that the corpus
taught one. [Raw replay manifest](validation/observer-archive-manifest.json).

## Measurement protocol

[Plan](experiment-plan.json), [source manifest](source.json),
[equality check against the previous experiment](same-experiment.json).
Every non-source plan field matches: seed 20261010; 64 presentations for each
two-/four-word order lesson; eight ordinary and eight supplied-answer
presentations for each stage-4 group; the same revision-3 corpus and all its
hashes; the same four recovery and four identity branches, branch optimizer
resets, admission ablation, and identification metric. The experiment and
full sweep run against the same frozen learner/test source.

The unchanged driver continues diagnostic budgets after an unmet warm-up
gate. Completion is not a claim that the prerequisites passed. The admission
rule stays unless its declared lesson passes without it; a vacuous gate
contrast cannot retire it.

The [completed shared warm-up](experiment/warmup-comparison.json) improves
two-word CE from .34657 to .03200–.03207 and four-word CE from .79451 to
.01653–.01671. Both remain above .01; every final free reconstruction score
remains 0/4. Supplied-answer scores remain 4/4 (identities), 4/4 (noun
phrases), 0/4 (noun–verb), and 1/4 (relations).

**New adverse result:** the final relation trace score falls from 4/4 to
2/4, both before and after its answer lesson. The two five-word comparisons
read `x is bigger y than` and `y is bigger x than`; the shorter relation
clauses remain exact. The improved taught CE does not establish correct
order on these later clauses.

## Eight-branch outcome

All eight branches exit zero after 13,941.33 seconds. The [full result](experiment/summary.json)
and [raw archive](experiment/archive-manifest.json) retain every branch; the
[independent integrity check](integrity.json) verifies the frozen source, all
12 corpus files, the unchanged non-source plan and the archived bytes. All
eight branches restore the same warm-up checkpoint exactly, with fresh
optimizers and no differing or extra tensors.

### Recovery and admission

Identification is the share of present atoms carried by an **active**
column with absolute cosine at least .9. It is not maximum cosine over the
dictionary. Each branch has 144 training and 48 held-out documents.

| Corpus | Gate | Held-out identification | Columns minted / total | Clean columns | Merged pairs |
|---|---|---:|---:|---:|---:|
| factorial | on | 0/240 | 0 / 58 | 0 | 0 |
| factorial | off | 0/240 | 0 / 58 | 0 | 0 |
| confounded | on | 0/240 | 0 / 58 | 0 | 0 |
| confounded | off | 0/240 | 0 / 58 | 0 | 0 |

The shared warm-up has 58 columns (41 previously). Each recovery branch
ends with 292 pending prototypes, 48 available held-out rows and zero rows
with an active encoding column. Every atom, including each noun kind, scores
zero. There is no factorial advantage over confounded training.

All 192 observed commits per recovery branch select mint; this includes
the 48 evaluations with admission frozen. Thus the training gate also never
changes eligibility. On/off checkpoints are byte-identical within each
corpus. The declared admission lesson fails every criterion; no rule is retired.

### Identity and individual rows

Each branch receives 136 training and 166 evaluation documents. The 146
binding probes include 32 neither-candidate controls; the other 114 probes
are informative. Both totals score zero, and all 20 row pairs are unresolved.

| Training | Gate | Binding correct / probes | Row pairs correct / pairs | Columns minted / total | Selected mint / commits |
|---|---|---:|---:|---:|---:|
| balanced | on | 0/146 | 0/20 | 57 / 115 | 870 / 906 |
| balanced | off | 0/146 | 0/20 | 58 / 116 | 870 / 906 |
| biased | on | 0/146 | 0/20 | 59 / 117 | 940 / 940 |
| biased | off | 0/146 | 0/20 | 59 / 117 | 940 / 940 |

**The balanced admission contrast is no longer vacuous:** gate-off mints one
more column and its final checkpoint differs from gate-on. Of 906 observed
commits, 36 have no selected individual mint. The counts combine training
and frozen evaluation; the one-column training difference establishes an
admission effect, without a binding or row-count gain. The biased pair
still selects mint on all 940 commits and has byte-identical on/off checkpoints.
No branch selects a non-mint individual reference in this counter. Compared
with the previous run, the balanced commit count falls from 940 to 906.

Each current branch resolves **150/690 annotated mentions** to written
individual addresses, versus zero previously. Native negative addresses
are valid; only -1 and 0 are sentinels. No sentinel reaches grading. The
other 540 mentions have named missing reasons:

| Training (both gates) | Mint not retained | No individual request | Leaf not admitted |
|---|---:|---:|---:|
| Balanced | 311 | 186 | 43 |
| Biased | 403 | 137 | 0 |

This improves mention-address coverage but does not establish identity
learning. On each branch, 108 of 342 candidate-mention slots have a valid
address, no probe has a complete candidate set, and none has a valid selected
probe reference. All 146 native banks contain entries; those entries do not
supply the missing annotated individuals. Balanced unadmitted leaves rise
from two in the retained old audit to 43 here, another adverse coverage finding.

The grouped scores below are identical across all four identity branches.
Every binding group has **zero correct and zero resolved** choices; every
row-pair group has zero correct and zero resolved pairs. Candidate coverage
is the mean fraction of candidate mentions whose addresses are available,
not binding accuracy. Joint candidate-count/recency/role cells remain in
the linked full result.

| Binding group | Probes | Correct | Resolved | Candidate coverage |
|---|---:|---:|---:|---:|
| all | 146 | 0 | 0 | 28.08% |
| candidates=1 | 4 | 0 | 0 | 0.00% |
| candidates=2 | 88 | 0 | 0 | 17.05% |
| candidates=3 | 54 | 0 | 0 | 48.15% |
| recency=0 | 58 | 0 | 0 | 25.29% |
| recency=1 | 70 | 0 | 0 | 25.24% |
| recency=2 | 18 | 0 | 0 | 48.15% |
| role=subject | 88 | 0 | 0 | 26.52% |
| role=object | 58 | 0 | 0 | 30.46% |
| first_mention=0 | 66 | 0 | 0 | 22.22% |
| first_mention=1 | 62 | 0 | 0 | 28.49% |
| first_mention=2 | 18 | 0 | 0 | 48.15% |
| family=pronoun | 32 | 0 | 0 | 18.75% |
| family=pronoun_one | 4 | 0 | 0 | 0.00% |
| family=pronoun_three | 54 | 0 | 0 | 48.15% |
| family=document_context | 8 | 0 | 0 | 0.00% |
| family=neither_candidate | 32 | 0 | 0 | 18.75% |
| family=reversed_recency | 16 | 0 | 0 | 18.75% |

| Row-pair group | Expected rows | Pairs | Correct | Resolved |
|---|---:|---:|---:|---:|
| determiner | 1 | 4 | 0 | 0 |
| determiner | 2 | 4 | 0 | 0 |
| determiner_conflict | 2 | 4 | 0 | 0 |
| same_kind | 1 | 4 | 0 | 0 |
| same_kind | 2 | 4 | 0 | 0 |

The neither-candidate control has no informative target; its unresolved
zero is not evidence of chance binding. The reversed-recency control also
has no resolved choices, so it does not demonstrate or rule out a learned
positional shortcut. Conditional accuracy is undefined in every group.
The same-kind row-pair metric covers the first two mentions; it does not
separately grade binding after the later distinguishing properties. The
corpus still reports the first-mention role/position confound after dropping
unverified fronting; refresh-mention crossing is retained.

## Validation and remaining work

The [85 affected checks pass](validation/affected-final.log), covering the
lost-meaning charge, opposing leaves, permutations, unavailable sentences,
saved scorer, actual mint→bind chain, and named missing reasons.

The [full sweep](full/summary.json) completes all 5,784 cases: 5,495 passed,
286 skipped, two failed, one xfailed. All 771 learner/test source files match.
Elapsed time is 1,004.31 seconds; peak aggregate memory is 9,393,927,560 bytes
(8.75 GiB). The [raw archive manifest](full/archive-manifest.json) verifies
each retained file's bytes. The failures are:

- The known forced-answer `five` failure,
  `test_forced_ordinary_answer_fills_committed_question_without_its_own_episode`.
- **New with this candidate:** the receipt README link check ran while two
  retained-replay files were being replaced. The files were restored; the
  [subsequent documentation check passes all 368 cases](validation/doc-links-restored.log).
  The original full-sweep counts and failure remain unchanged.

The [accepted-6.1 comparison](full/baseline-comparison.json) names the new
failure explicitly. Three other baseline failing nodes pass here, without
closing their historical findings. The cold overlap regression remains
XFAIL against 6.1's XPASS. Two development failures—an objective-name
contract omitting the added term, and a positional meaning charge penalizing
a complete permutation—are retained under `development/`; the final
affected checks pass without relaxing the operator certificate.
The [final documentation check](validation/doc-links-final.log) also passes
all 368 cases after the outcome tables and plan were completed.

The requested S movement and T accounting are demonstrated. R charges the
lost meaning but fails its retained-meaning/kept-program criterion. The
identity/admission lesson, both .01 order gates and free reconstruction
remain open, with the new relation-order and mention-coverage findings
carried in the plan and todo. The parts-of-speech pass in plan §6 follows
this review; no mechanism in that pass has been changed here.
