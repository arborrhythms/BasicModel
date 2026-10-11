# Item 6 part 2: first native experiment and review follow-ups M/N/O/I

The preparation landed as `06f6e2b88` after [plan §5](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#5-review-of-the-follow-up-landing-and-the-part-2-preparation-claude-2026-10-10), with WikiOracle bump `56544f1`.
This pass was accepted as preparation in [plan §7](../../../plans/2026-10-10-item-6-stored-idea-generativity.md#7-review-of-the-53-pass-mnoi-and-the-first-native-experiment-claude-2026-10-10)
and authorized for landing on October 11. This is the single receipt
for the correctness repairs, revised corpus and first matched experiment.
The reviewed preparation is committed as `af634ed35`.
Results below distinguish completed measurements from the retained failed
attempts. No learning claim is made and no rule is retired. The R/S/T repair
pass and its unchanged eight-branch measurement follow this preparation.

**Completed measurement:** all eight branches exit zero, with unchanged
learner source and corpus ([full experiment receipt](experiment/summary.json),
[integrity check](experiment/source-check.json)). Recovery scores 0/240 in
each branch. Identity scores 0/146 bindings and 0/20 row pairs in each branch;
all those references remain unresolved despite populated native banks.
Balanced identity training mints 61 columns and biased training 62, but
these counts do not establish correct identification. The gate-on/off
checkpoints are byte-identical within each corpus: every observed commit
selects mint, so this experiment supplies no admission eligibility contrast.

The current configuration has a nonzero meaning complement. The earlier
tiny-fixture runs had zero meaning dimensions and could not test unmixing
or the admission gate. Their zero scores and interrupted identity branches
are retained as invalid measurements below. In the corrected run, the
recovery examples instead lose their meaning during native composition.
This is a measured pipeline failure, not an informative isolated ICA lesson.

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
The [frozen learner and driver](source.tar.gz) and [revision 3 corpus](corpus.tar.gz)
have separate archives. All 12 corpus files match the experiment plan's
[hashes](corpus-sha256.json); grading labels remain outside model inputs.
The current native configuration has 104 form-event coordinates and a
128-coordinate meaning complement (64 pole pairs), 4,096 concept rows,
256 part rows and an 8,192-row LTM. It retains the production separation
between form and meaning. The older word-chunk parity fixture made the two
spaces equally wide, leaving no meaning coordinates at all. That was a
measurement setup error. Constructor and post-lexical warm-up checks now
require the meaning band and a nonzero native component signal.
The [preflight](validation/meaning-preflight/result.json) presents four ordinary
words four times, without an optimizer: all four read back, four basis rows
become nonzero, and three of four retained observations project nontrivially.
That separate preflight state is not reused as the learning run's warm-up.
The actual lexical warm-up's [checkpoint audit](validation/lexical-admission-audit.json)
finds 24 nonzero basis rows and eight nonzero projections among 24 retained
observations, with projection norms up to .258199 against the .2 mint threshold.
There are zero noun columns and zero pending prototypes at this point. This
establishes an available meaning signal, not learned factor recovery. The
[checkpoint](validation/lexical-checkpoint/lexical.ckpt.tar.gz) and its
[configuration](validation/lexical-checkpoint/identity_model.xml) are retained.
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
native presentation. This confounding applies to the branch's examples: both
branches inherit the same earlier word and sentence lessons, including
isolated lexical exposures. It is a recovery comparison conditioned on that
warm-up, not a claim that the confounded learner has never encountered separate
factors. Identification counts a present atom only if a nonzero
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

The [current final-source sweep](full/summary.json), including the corrected
meaning configuration, completed **5,776/5,776** cases: **5,488 passed,
286 skipped, one failed and one xfailed**, in 973.50 seconds with 7.92 GiB
peak aggregate worker memory. All 771 source files match. The sole failure
is the retained supplied-answer `five` case. [Raw reports/logs](full/logs.tar.gz),
[HTML report](full/report.html), [comparison with accepted 6.1](full/baseline-comparison.json).
There is no new failing node in this sweep. The earlier new binding-distribution
failure and packaging link failures remain recorded, and the overlap
XPASS-to-XFAIL remains adverse. No historical finding is closed by a passing
initialization. The learning experiment completes on the same frozen source
in 13,322.71 seconds. All eight branch processes exit zero; all 771 source
files and all 12 corpus files match their declared hashes.
The final [documentation check](validation/doc-links-final.log) passes all
365 cases; `git diff --check` also passes. Under the
[publish rule](../../../../todo.md), this complete candidate now waits for
Claude's code review before commit/push and the WikiOracle bump.

The [shared warm-up](validation/warmup-summary.json) completes every declared
presentation. Counts below are out of four, evaluated without reading or
answer teaching. The last-word answer baseline is a
[post-hoc descriptive check](validation/answer-baseline.json), not a trained
control. Neither order lesson passes its .01 CE threshold.

| Stage | Initial → final trace readback | Final free readback | Final answer | Last-word answer baseline |
| --- | --- | --- | --- | --- |
| Two words | 0 → 4 | 0 | — | — |
| Four words | 4 → 4 | 0 | — | — |
| Identities | 3 → 4 | 0 | 4 | 4 |
| Noun phrases | 2 → 4 | 0 | 4 | 4 |
| Noun–verb | 4 → 4 | 0 | 0 | 0 |
| Relations | 2 → 4 | 0 | 1 | 0 |

The [completed warm-up audit](validation/warmup-admission-audit.json) finds
41 noun columns, 334 pending prototypes, 24 nonzero basis rows and eight
nonzero projections among 66 retained observations. Its checkpoint is
`c479e513ee6690a887e4a2e3eb5e5c1ad899c7eeeb0de507919d844b380bc01d`.
All eight branches start with exact matches on all 2,138 saved tensors,
no extra tensors and the same nonzero meaning signal. The recovery and
identity branches all complete their declared training and evaluation budgets.

The [recovery results](validation/recovery-summary.json) are:

| Corpus | Selected-mint gate | Held-out identification | New columns | Total columns | Merged pairs |
| --- | --- | --- | --- | --- | --- |
| Factorial | On | 0/240 | 0 | 41 | 0 |
| Factorial | Off | 0/240 | 0 | 41 | 0 |
| Confounded | On | 0/240 | 0 | 41 | 0 |
| Confounded | Off | 0/240 | 0 | 41 | 0 |

Each branch completes 144 training documents and 48 held-out documents,
exits zero, and retains 334 pending prototypes. Every held-out row is
available but activates zero dictionary columns. All 192 observed commits
per branch report a selected mint (the counters include frozen evaluation).
Thus the on/off intervention changes no admission eligibility in this run.
The on/off checkpoint files are byte-identical within each corpus. The
numerical retirement gate fails; these results do not show that the gate is
necessary or that removing it improves recovery.

The [checkpoint audit](validation/recovery-admission-audit.json) and
[per-row signal probe](probe_recovery_signal.py)
([result](validation/recovery-signal.json)) locate a failure before sparse
identification: both corpora retain all 66 warm-up rows and add 192 observation
rows, all with nonzero full vectors but **zero meaning-band content** and zero
component projections. The meaning space still has 64 pole pairs and 24
nonzero basis rows; the same eight earlier observations supply its nonzero
population signal. Unlike the earlier zero-width fixture error, this is a
native pipeline failure to retain informative meaning in these examples.
The measured end-to-end misses are valid, but this run supplies neither an
informative factorial ICA recovery test nor a gate constraint to ablate.
No rule is retired.

The [identity summary](validation/identity-summary.json), reproduced by the
[read-only summarizer](summarize_identity.py), checks every native group
summary against its raw rows. Each branch presents 136 training documents
and evaluates the same 166 documents without optimizer steps, with component
admission frozen. All four branches have identical accuracy/coverage summaries:

| Identity training | Selected-mint gate | New columns | Total columns | Binding | Row pairs |
| --- | --- | --- | --- | --- | --- |
| Balanced | On | 61 | 102 | 0/146 | 0/20 |
| Balanced | Off | 61 | 102 | 0/146 | 0/20 |
| Recency-biased | On | 62 | 103 | 0/146 | 0/20 |
| Recency-biased | Off | 62 | 103 | 0/146 | 0/20 |

All 940 observed commit calls per branch select mint, including frozen
evaluation calls. Thus the gate changes no eligibility here either. No
admission drops are recorded. The on/off final checkpoints are byte-identical
within each training corpus; the extra column in biased training is not
evidence of better identification. Active-column atom identification and
merged-pair diagnostics are measured in the recovery branches, not inferred
from these identity column counts.

| Binding group, same in all four branches | Correct / cases | Resolved |
| --- | --- | --- |
| One candidate | 0/4 | 0 |
| Two candidates, including controls and document context | 0/88 | 0 |
| Three candidates | 0/54 | 0 |
| Most recent / next / third candidate | 0/58; 0/70; 0/18 | 0 |
| Subject / object introduction role | 0/88; 0/58 | 0 |
| First / second / third introduced candidate | 0/66; 0/62; 0/18 | 0 |

The 146 binding cases contain 114 informative cases and 32 neither-candidate
controls. Ordinary two-candidate pronouns score 0/32, document context 0/8,
the neither-candidate control 0/32, and reversed recency 0/16. There is no
resolved choice in any group. The neither-candidate result therefore does
not demonstrate chance-level uncertainty, and the reversed-recency result
cannot distinguish a learned recency shortcut from content-based choice.
The full candidate-count × recency × role cells are retained in the summary.

The 20 row-pair cases comprise ordinary determiners (0/4 expected one row;
0/4 expected two), determiner conflicts (0/4 expected two), and same-kind
individuals (0/4 expected one; 0/4 expected two). Every pair is unresolved,
so these results do not establish an incorrect resolved row count or correct
later property binding. The accepted metric grades the first two mentions.

The native candidate bank is nonempty in all 146 probed cases, in every
branch. However, the observer resolves **zero of 690 annotated mention
references**: balanced runs contain 445 mint sentinels and 245 missing
references; biased runs contain 372 and 318. Consequently none of the 342
graded candidate occurrences can be linked by ID to a bank entry. Zero
reported collisions is not evidence of distinct individuals when no candidate
IDs resolve. This audit locates an unresolved identity/provenance seam; it
does not establish that the populated banks lack semantically suitable frames.
The reason those selected mints do not become graded individual references
remains open. No pronoun, determiner, positional or admission rule is retired.

A [read-only replay of four held-out rows](validation/recovery-trace-audit/result.json)
shows that their adjective, noun and verb inputs each carry nonzero meaning
(norm .2), which the completed clause loses. The
[derivation replay](probe_recovery_trace.py)
([captured programs](validation/recovery-program-audit/result.json)) localizes
this on all four rows: each binary `lower` keeps its newer operand, while
unary `non` clears the expressed meaning after each content word. The last
binary step keeps the period; repeated `non` steps follow at closing. Thus
the observed semantic loss is already present before the period. `non` is
declared lossy and has no inverse; its selected use here cannot be undone
from the resulting value alone. Why the trained compose policy keeps these
lossy readings despite reconstruction and prediction costs remains open.
These replays use the frozen factorial-off checkpoint, no optimizer, targets
or reading guidance, and discard their state afterward. The recorded operation
norms are interpreted only on binary/unary entries; push entries can carry
journal placeholders ([schema note](validation/recovery-program-audit/schema-note.json)).
Actual word-input norms come from the separate pushed-word table and tensors.

In the current run, the two-word lesson completes all 64 presentations. Its
untaught final stored-derivation readback is **4/4**, up from 0/4 initially on
these same four training phrases. Read order was already correct initially.
Free generation still emits only the second word of each phrase (**0/4** exact),
and order CE remains .3465735912, so the order-learning gate fails. This is
training-phrase reconstruction evidence, not held-out identification or a
comparison against a trained 6.1 checkpoint. The four-word result is reported
separately below.
The [two-word checkpoint audit](validation/two-word-admission-audit.json)
finds ten noun columns and 72 pending prototypes, with the meaning signal
still present. These counts establish admission during warm-up, not correct
factor recovery. Its [checkpoint](validation/two-word-checkpoint/two-word.ckpt.tar.gz)
and [configuration](validation/two-word-checkpoint/identity_model.xml) are retained.
The [learning-weight audit](validation/two-word-policy-audit.json) compares that
checkpoint with the completed lexical stage: the attention scorer, all five
pair-chooser weights and the generate policy changed. Flat order CE therefore
does not mean that all these parameters were frozen. It does not establish
that their updates taught the required choices.
The [read-only scorer probe](probe_attention_readout.py), run once on that
saved checkpoint without an optimizer or reading guidance, finds a concrete
obstruction ([result](validation/attention-readout-audit/result.json),
[first-read tensors](validation/attention-readout-audit/first-read.pt)). All
64 tanh units on every row and candidate are exactly ±1. Linear preactivations
span approximately −140 to +140; candidate differences before tanh are nonzero
but vanish afterward. The two candidate logits are identical on every row.
Thus the hidden derivative is zero at these inputs and the order lesson cannot
rank them. Its CE is the uniform-choice value, log(2)/2. This diagnoses the
saved two-word checkpoint, not the onset or complete cause of saturation.
A [frozen-feature calculation](validation/attention-readout-audit/component-audit.json)
reproduces the saved preactivation exactly. On that same input, the earlier
lexical weights give a total range of −.507 to .761 and nonzero hidden
candidate differences, with no tanh outputs at ±1. The two-word weights give
a shared-context contribution of −139.765 to 140.115, compared with a candidate
contribution of −.284 to .279. This localizes the observed loss of distinction
to the trained scorer; it does not identify the responsible update or input band.
No normalization, initialization or learning mechanism is changed during the
declared experiment. The probe state is discarded and never enters a branch.

The architecture has explicit teaching paths for pair decomposition and the
generate walk. Both losses reach the training step after trial comparison;
the hard candidate search itself is detached. The frozen-artifact probe above
demonstrates one teachable pair choice. This supports reconstruction's
reachability on represented, distinguishable candidates, not a convergence or
generalization guarantee. The pair chooser uses five linear features and the
generate policy is linear in its state: neither can distinguish states whose
available inputs coincide. Exact recovery of an erased wording distinction
requires retained order, context or provenance. Derivation readback has access
to the stored trace, whereas the free walk has to choose its own tree.
The current pair teacher also skips compound operands outside its word
shortlist: its symbolic stack records a composed parent as `-1`, and an
absent pair identity contributes no CE. The generate-policy teacher still
teaches mapped parent operations. Thus the lexical-pair result does not
establish that all constituent choices receive sufficient teaching credit;
free recovery beyond two words remains an open learning claim.

The four-word stage also completes 64 presentations. Its
[initial and final evaluations](validation/four-word-checkpoint/results.json)
both read all four stored derivations exactly without reading guidance.
Thus this stage does not demonstrate improvement from its own training:
trace readback already worked after the two-word stage. Free generation is
0/4 in both evaluations, emitting only each phrase's final word; order CE
remains .7945134640. The [checkpoint](validation/four-word-checkpoint/four-word.ckpt.tar.gz)
and [configuration](validation/four-word-checkpoint/identity_model.xml) are retained.
Its [admission audit](validation/four-word-admission-audit.json) finds 21 noun
columns and 203 pending prototypes. As at the previous stages, eight retained
observations have nonzero meaning projections; column counts alone do not
establish correct atom identification.

The first stage-4 sentence lesson completes its eight presentations with 4/4
trace readback and 0/4 free readback. After eight supplied-answer presentations,
its [unguided answer evaluation](validation/identity-answer-checkpoint/results.json)
returns `animal`, `animal`, `three`, `four`: **4/4** on the four training
questions. This is a training-question result, without a held-out or query-control
claim. The [checkpoint](validation/identity-answer-checkpoint/identities-answer.ckpt.tar.gz)
and [configuration](validation/identity-answer-checkpoint/identity_model.xml) are retained.
The noun-phrase answer stage also scores 4/4, but the
[noun–verb answer stage](validation/noun-verb-answer-checkpoint/results.json)
scores **0/4**, returning `sleeps`, `runs`, `sleeps`, `runs` instead of the
requested subjects. In these first three answer stages, every reported answer
equals the free decoder's last-word output. A last-word baseline therefore
also scores 4/4, 4/4 and 0/4 on these stages. The earlier correct answers do
not establish question-sensitive retrieval. The negative noun–verb
[checkpoint](validation/noun-verb-answer-checkpoint/noun-verb-answer.ckpt.tar.gz)
is retained alongside the earlier positive checkpoint. Relations score 1/4,
returning `mat`, `dog`, `dog`, `x` against `cat`, `dog`, `x`, `y`. Two of those
answers differ from the free decoder's last-word output; one is correct.
This does not establish reliable relation answering or a held-out benefit.

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

The [sweep after the restoration-adapter correction](development/zero-meaning-band-attempt/full/summary.json)
completed **5,776/5,776**: **5,488 passed, 286 skipped, one failed and one
xfailed**, in 1,010.50 seconds, with 9.11 GiB peak aggregate worker memory.
All 771 source files match that candidate.
[Raw reports/logs](development/zero-meaning-band-attempt/full/logs.tar.gz),
[HTML report](development/zero-meaning-band-attempt/full/report.html),
[baseline comparison](development/zero-meaning-band-attempt/full/baseline-comparison.json).
The only failure is the retained supplied-answer `five` case. There is no new
failing node in that completed sweep relative to accepted 6.1; the first attempt's
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
separates this adapter correction from the unchanged learner.

The continuation completed all four recovery branches: each had 0/240
identification, zero columns minted, zero merges and zero pending prototypes.
Those are **invalid admission measurements**, not failed learning gates.
The [frozen-checkpoint audit](validation/admission-audit.json) explains why:
the word-chunk fixture had `meaning_start=104` and `meaning_pairs=0`. All
55 final basis rows were zero; all 434 nonzero retained observations projected
to zero. Thus the off branch had no residual that could start a prototype.
The factorial on/off checkpoint files were identical. Online learning was
enabled, so this is not an evaluation-freeze effect.

The remaining identity runs were stopped on that finding: balanced branches
had presented 20 training documents each, biased branches 12 each, and none
had evaluated. Their progress, source checks, complete recovery outputs and
invalidity marker remain in `development/zero-meaning-band-attempt`.
The checkpoint audit is a read-only inspection with no presentation or update.
The corrected model passes the nonzero-signal preflight; its fresh staged run
and final-source sweep are complete above. Presentation budgets, data, seed,
learning rate and rule-retirement requirements are unchanged. No rule retired.
