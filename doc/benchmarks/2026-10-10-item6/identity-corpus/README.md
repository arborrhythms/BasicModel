# Item 6 part 2: corpus and measurement before mechanism changes

This preparation was accepted in [plan §5](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#5-review-of-the-follow-up-landing-and-the-part-2-preparation-claude-2026-10-10)
and authorized for landing by Alec on October 10. It follows the reviewed
[H/I follow-up](../followup/README.md), based on `c1d1ea03c`.
The frozen metadata records its status at measurement time; follow-ups M, N, O
and I belong to the next receipt.
It adds revision 2 of the teaching text, separate grading labels and a native
measurement driver. The rejected passive-role prototype is retained below.
No production source, grammar, identity rule or admission policy changes.
No rule is certified for retirement by this receipt.

## Frozen inputs

- [Corpus and usage](../../../../data/identity_from_data/README.md),
  [generator](../../../../test/identity_corpus.py),
  [measurement](../../../../test/identity_measurement.py),
  [contract tests](../../../../test/test_identity_corpus.py).
- [Source manifest](source.json): 769 files, the H baseline plus three Python
  files and the corpus manifest. [Corpus hashes](corpus-source.json) cover all
  twelve data files, including the explanatory README and label sidecars.
- [Source archive](source.tar.gz), SHA-256
  `0137effa271e980e4996345522ddcf4ed90b02121353a234d130d0210ed28c1f`.
  [Metadata](source-metadata.json) records the base and review status.
  Each native report checks its source and corpus hashes again at completion.

The learner receives only sentence strings and opaque document addresses.
Candidate lists, roles, mention offsets and target identities are evaluator
data. The observer reads the selected order-one references and actual written
individual addresses; it never forces a parse or resolves an identity for
the learner. Negative native occurrence addresses are valid; `-1` and `0`
are sentinels. Unavailable or colliding candidates remain in the denominator.

## Lesson inventory

There are 740 documents across five streams:

| stream | documents | purpose |
|---|---:|---|
| train | 494 | singleton recurrence, factorial support, identity lessons |
| eval | 150 | held-out documents and kind pairings, including controls |
| confounded_train | 64 | kind/property confound, separate from main training |
| biased_train | 16 | follow-up always rewards the most recent candidate |
| reversed_eval | 16 | follow-up always rewards the other candidate |

The 494 training documents comprise 64 singleton witnesses, 192 factorial
sentences (64 at each maximum object support of 1, 2 and 3), 36 singleton
rehearsals, 16 kind/predicate lessons, 90 pronoun documents, 80 determiner
documents, eight same-kind documents and eight document-context cases.
Every kind/property/verb combination occurs at each participating noun
position in the factorial lesson. The stages count **objects**, not total
independent atoms; a sentence also contains properties and a verb. This does
not assert the toy's three-source recovery condition for a whole sentence.

The two-candidate pronoun lesson crosses target recency, introduction role
and first surface position jointly: eight cells, four documents each, in
both train and evaluation. Ordinary and object-topicalized introductions
(`a cat saw a dog .` / `a dog , a cat saw .`) separate grammatical role from
surface order; refresh mentions separate recency from introduction role.
The three-candidate lesson balances kind, target recency and position; its
two-subject/one-object marginal is reported rather than hidden.

Determiner cues are reliable in 90% of training first/repeat mentions. The
four held-out cue/content conflicts have `a black ... the white ...` with
two individuals in a world with exclusive colours. Same-content individuals
cannot be separated by content alone. Consequently the noisy determiner
items are not assigned a perfect-accuracy gate. Same-kind lessons add later
properties of the running and sleeping individuals; the stage-7 probe words
refer to different individuals in different documents.

The 150 evaluation documents provide 130 binding probes and 20 one-row/two-row
probes. Binding is reported by candidate count, recency, role, their joint
cells, and first-mention position, alongside resolved coverage, conditional
accuracy, candidate availability and collisions. Row counts use distinct
semantic addresses, never distinct word rows.

Candidate count is the count in the teaching document. The report's
`candidate_coverage` measures how many of those antecedents have recoverable
semantic addresses in the selected journal. It does not inspect the complete
live eight-space/cued-frame bank at the pronoun decision. That additional
inventory audit is needed to separate a missing native candidate from an
incorrect choice when a previous mention's address is known. This limitation
cannot turn an unresolved or colliding reference into a correct answer.

## Controls and checks

[Corpus audit](corpus-audit.json) includes counts, joint cells and explicitly
labelled grading controls. An oracle verifies the grader can give full credit;
it is not a learner and uses the held-out answers. Always selecting the most
recent candidate gives:

| grading sanity control | accuracy |
|---|---:|
| biased_train | 16/16 |
| reversed_eval | 0/16 |
| neither-candidate paired documents | 16/32 |

The recent-candidate control is given the gold candidate identities and
changes only the pronoun's selection. Its row-count fields therefore inherit
the oracle's identities; they do not measure a positional policy's ability to
mint or bind individuals.

The last control presents the same visible text with both hidden answers.
It tests chance-level **choice**; zero resolved references do not pass it.
These checks are not evidence that a trained native model learned either
content or recency. Claude's existing
[four-question toy](../../2026-10-10-identity-ica-toy/README.md) remains the
separate labelled toy experiment.

The [corpus checks](corpus-checks.log) pass all 13 tests. The earlier
[prototype affected checks](development/passive-role-prototype/affected.log)
passed 46 with the unchanged overlap xfail; that includes the earlier
11-check corpus version, not the corrected 13-check corpus certificate.
They cover regeneration and disjoint documents, factorial factors, joint
counterbalancing, cue reliability, identifiable conflicts, both controls,
missing/colliding candidates, negative occurrence addresses, semantic row
counts and document-dependent reference. The added check distinguishes actual
grammatical subject/object roles from passive agent/patient labels; another
checks the observer against the real clause writer. The full production receipt belongs
to H/I; this preparation adds no production change and makes no new full-sweep
claim. Documentation links pass all 364 checks (`doc-links.log`).

Before freezing this driver, a [cold four-document smoke check](development/native-smoke/results.json)
and a [four-training/four-evaluation check](development/native-train-smoke/results.json)
completed without exceptions. Both had zero resolved references on their four
evaluation documents. They checked the native invocation and reporting path;
neither is an accuracy gate, a successful chance control or a substitute for
the main corpus pass. Their development source manifests are retained in
their reports rather than represented as the final frozen source.

The [observer positive control](observer-check.json), run by
[this driver](check_observer.py), uses an explicitly forced journal and the
real writer. It exposed a measurement defect: the sentence and its owned NP
can have the same head and order, so the first adapter treated their two
addresses as ambiguous and left a minted NP unresolved. The observer now
follows the writer's actual owned-head reference, and the regression checks
that it reports the NP's address rather than the enclosing sentence's.
The [failed control facts](development/observer-before/observer-check-facts.json),
[earlier source](development/observer-before/source.tar.gz) and initial
revision-2 [cold](development/observer-before/identity-v2-cold/results.json) /
[biased](development/observer-before/identity-v2-biased/results.json) runs are
preserved. They precede the observer repair and are not the final row-count
measurement. The repaired positive control is measurement plumbing, not a
learned-identity result; neither native run forces a journal or a reference.

## Native measurement protocol

The driver uses the real tiny canonical model, CPU, one torch thread,
16-coordinate events, 4,096 concept rows, the configured 262,144 LTM rows,
a word bucket of 64 and document batches of four. It retains the production
identity rules, disables capture for this eager baseline, and uses the
production no-intra-predictor setting. The revision-2 baseline is **cold**:
all 150 evaluation documents with no optimizer updates. Native reading and
admission still run, so this is a stream starting from
cold parameters, not a model whose dictionary is frozen between documents.
The separate biased
preflight requests one pass through its 16 training documents at learning
rate .003, followed by the 16 reversed items. It is not matched to a trained
counterbalanced branch. Seed 20261010 fixes each measurement; it is never
searched or used to make an assertion pass. Neither run is a convergence
claim or a compiled-retention test.

A native failure stops the run and is preserved. Unreached evaluation cases
remain in the report with explicit errors. Their zeros are conservative
denominators, **not measured binding accuracy**. No failing batch is skipped,
rerolled or replaced with a forced reading.

## Revision-2 native outcomes

[Combined accounting](native-outcomes.json) separates completed cases from
batch failures and unreached cases. The final
[cold report](identity-cold/results.json) and
[biased preflight](identity-biased/results.json) both match all 769 source
hashes and all twelve corpus hashes. Every archived source/corpus member was
also checked against those manifests.

| run | training completed | evaluation completed | failed batch cases / unreached | resolved binding probes among completed cases | resolved row pairs among completed cases |
|---|---:|---:|---:|---:|---:|
| cold stream | 0 | 132/150 | 4 / 14 | 0/122 | 0/10 |
| biased preflight → reversed evaluation | 16/16 | 16/16 | 0 / 0 | 0/16 | — |

The cold run takes 359.03 seconds and stops with
`ValueError: compose trace binary operation underflows`. The failed batch
contains evaluation positions 132–135 (zero based); the report conservatively
marks that whole batch failed, without claiming that all four examples would
fail individually. The next fourteen cases are explicitly unreached. This
is a production exception before the observer extracts references. Its
precise trace has not been independently isolated on this revised wording;
the earlier passive-input trace below is a separate concrete reproduction of
the same exception class.

The completed cold cases expose four order-one reference requests and three
resolved individual mention addresses. No completed antecedent/probe binding
or first/second-mention pair has all the needed identities. The biased
preflight takes 263.17 seconds, has no exception, and exposes no order-one
reference requests for its labelled evaluation mentions. Both finish with
zero noun dictionary columns. That is an observed diagnostic, not an
atom-recovery score or a claim that staged training cannot learn columns.

The raw reports retain all requested cases in their denominators and show
zero unconditional accuracy; conditional accuracy is undefined because no
complete binding or row pair resolves. **There is no native chance-control,
learned-recency, row-count or rule-retirement success here.** The neither-
candidate and reversed-recency successes above belong only to the labelled
grading sanity controls.

Next measurement work is to repair the native trace/admission failure,
complete the full evaluation, audit the actual candidate bank, and run the
counterbalanced and biased lessons from the same trained warm-up state with
equal presentation budgets. Factorial-versus-confounded recovery, full staged
learning, trained A/B stored recovery, the order lesson, MM grammar and compiled
retention remain open. No identity rule is changed or retired in this preparation.

## Rejected passive-role prototype and its diagnostics

The first corpus used passives to counterbalance role and order, but its
subject/object labels continued to name the active agent/patient. A passive
patient is the grammatical subject, so those data did **not** satisfy the
requested grammatical-role/order balance. The main training diagnostic was
[stopped after 420 completed documents](development/passive-role-prototype/identity-main/stop.json),
with no evaluation reached. It was stopped for the corpus defect, not for a
native failure or an adverse accuracy result. Its
[log](development/passive-role-prototype/identity-main/run.log),
[frozen source/corpus archive](development/passive-role-prototype/source.tar.gz)
and [source metadata](development/passive-role-prototype/source-metadata.json)
are retained. Revision 2 replaces passives with object topicalization and
adds the text-level grammatical-role check; no production mechanism changes.
The interrupted prototype is not the corrected corpus's learning baseline.

The prototype's [cold biased run](development/passive-role-prototype/identity-biased/results.json) fails in its first batch,
before any of its 16 training documents complete. All 16 reversed-evaluation
documents are unreached. The exception is
`ValueError: compose trace binary operation underflows` in the production
`_derivation_program`, called by the original `_sentence_observation` before
the observer extracts references.

The [diagnostic replay](development/passive-role-prototype/biased-failure-replay/results.json) reproduces the
same seed, source, raw text and failure; it is an investigation, not another
accuracy attempt. [Driver](development/passive-role-prototype/capture_native_failure.py),
[trace facts](development/passive-role-prototype/biased-failure-replay/failure-context.json) and
[tensors](development/passive-role-prototype/biased-failure-replay/failure-context.pt)
are retained; rerun the driver from the prototype's archived source and corpus.
At sentence 0
in the explore trial, batch row 2 reads source positions
`[4,0,1,2,3,5,6,7]`. Position 4 (`by`) is absent from the grammar-leaf mask,
but source position 0 records binary rule 17. The derivation reconstruction
filters out position 4, then applies that binary operation with only one
operand. This records the inconsistent mask/trace; it does not establish why
the earlier admission excluded the position.

The source identifies the boundary to investigate: sentence scoring intersects
the trial's field admission with `source_leaf_mask`, saved before the trial's
attention handoff. `SentenceField` instead constructs its candidates from the
current trial's accepted reading. A newly admitted exploratory leaf can
therefore appear in the compose trace and disappear from the reconstruction
mask. This is a newly exposed preflight finding on unchanged production
source, not evidence of a regression introduced by H or of failed identity
learning. No mask repair or grammar bypass is included in this preparation.

That cold run lacks the main stream's common stage-1 warm-up. It is a native
preflight, **not a matched learned-recency comparison**. A valid learned
control must start both branches from the same trained warm-up state and use
equal presentation budgets, train
the counterbalanced and biased pronoun streams separately, and evaluate
ordinary and reversed recency without binding-label supervision. That
comparison remains pending; the corpus and scorer for it now exist.
