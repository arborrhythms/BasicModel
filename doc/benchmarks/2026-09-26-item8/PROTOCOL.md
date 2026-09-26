# Item 8: structural preference and learned utility

Declared before implementation or new learning measurements, starting at
`074093ec528ae38f99ee2b495174ae4da0f94d1b`. No commit or push before Claude's
review. Item 8 remains open until its empirical gates pass.

## Correctness and diagnostic scope

Run the three inherited MM grammar / XOR_grammar assertions unchanged first.
An ambient initialization passing does not resolve the recorded failing one.
Retain all outcomes, including pre-training failures. Investigate W=6 without
changing the XOR corpus, six-word capacity, epoch counts or quality assertions.
For the MM diagnostic declare seeds 0, 1, 2, Adam .01, all 900 updates, and
report initial/final/minimum MSE and gradients for every seed. These are
reproducible development measurements, never a passing-seed search or a
FineWeb-qualified utility evaluation.

Structural preference breaks **exact learned-score ties**, without an epsilon
bonus, target lookup or extra chooser. Better-scoring opaque candidates stay
eligible. Exploration and soft credit retain the ordinary MLP distribution.
Classify by the implementation's declared contract, not spelling, numerical
payload, symbol address or whether a rule happened to win. Unknown extensions
are opaque. The existing named structural operators remain structural;
trainable parameters alone do not make an operator opaque. Keep copy/stop
controls separate from executed grammar operations in coverage counts.

Retain the archived arbitrary-symbol poison probes, adapting retired APIs to
their current owners and preserving oracle isolation, changed input, reverse
relation and optimizer-update assertions. Cover renamed vocabulary and joint
native-reference renaming. Exact arithmetic and generated labels belong only
to corpus preparation/scoring, never learner execution, answer seeds or policy
features. Numerical concept coordinates are learned representations, not a
calculator. Correctness checks have no checkpoint-maturity skip.

## Predeclared causal quality study (pending qualifying training)

Every independently trained arm must carry the existing checkpoint exposure
record for at least **1,000,000 completed FineWeb training sentences**. Reuse
`LearningEvaluation.fineweb_readiness`; no lower default, epoch estimate or
same-checkpoint ablation substitutes for this prerequisite. Missing evidence
skips the study, never passes it. A qualified failure remains a failure.

Use seeds **0, 1, 2** for every arm. From the same seed's initialization,
train independently on the identical chronological corpus and supplied-answer
curriculum for **10,000 successful optimizer updates after qualification**,
with the same optimizer, batch size, parameter capacity and answer/reconstruction
weights. Freeze source, configs, checkpoint hashes, corpus/split manifests,
ordered example IDs, initialization and update receipts before scoring. If a
condition cannot be executed, record it as unavailable; do not replace it with
another condition after seeing results.

Arms: (1) learned ordinary thought with subgoals; (2) direct-answer, one root
execution followed by conclusion; (3) equal-compute no-subgoal, with the same
actual charged operation/context/attention opportunities as (1), spent on
root-level work rather than nested descent; (4) reconstruction-only. Equal
maximum budgets or dummy work do not qualify as equal compute. Use budget
caps 32 and 64 and record both actual `QueryWorkBudget` counts and elapsed
time, including unanswered/cutoff cases. Also retain a budget-zero diagnostic.

Prepare 256 held-out tasks before training: 64 direct facts/redundant questions,
64 unseen relation chains at depths 2–4, 64 missing/negative/conflicting-premise
cases, and 64 unfamiliar successor/addition combinations at depths 2–4.
Partition underlying problems/chains before wording; keep commuted equivalents
together. For every task retain an unseen wording and a bijectively renamed
vocabulary version with consistently renamed premises and answers. Random
symbol allocation is independent of numeric value and task answer. Premises,
root requests and target labels remain separate. Corpus generation and its
hashes must be frozen before any training; it has not yet been implemented.

Score exact typed answer meaning, full-vocabulary generated answers, paired
truth Brier error where applicable, answer coverage and work for **all** tasks,
plus each stratum/control. Unknown/malformed answers count as incorrect;
report selective error separately without dropping them. Gate per seed and
per control: higher exact answer accuracy at matched actual work, **or** lower
actual work with accuracy no worse (absolute tolerance 1e-6); no reduced answer
coverage. Require the paired 95% bootstrap interval for the improvement to
exclude zero (10,000 resamples, reporting seed 808), retain all seeds and report
their spread. Recompute with relevant intermediate results removed/corrupted,
and with question-history versus answer-history masked, preserving legal
continuation state and the same work opportunities. No per-seed retuning.

Reconstruction/discrimination must not regress against reconstruction-only:
mean held-out byte cost <= 1.01 times its matched control and fixed four XOR /
68 FineWeb categorical discrimination CP no lower by more than 1e-6 in any
seed. These thresholds are study gates, not replacements for existing tests.

## Routing coverage and throughput

Capture committed forward program rules on the **same ordered held-out
sentences**, retaining per-sentence IDs, corpus hash, declared catalog and
structural coverage stage. Count both opaque operations / all operations and
sentences with any opaque operation / all nonempty sentences. Exclude padding,
candidate enumeration and copied slots; reject missing programs or unknown
rule IDs. Report no-op sentences separately. Compare independently trained
nested catalogs: base structural inventory, added relation operators, then
added compositional arithmetic operators; keep the opaque candidate and MLP
choice path in every condition. Declare the exact catalogs/hashes before
training. Coverage must actually grow and sentence opaque share must decline
strictly from first to last in every seed, with no quality/control regression.
A catalog containing no opaque candidate yields a descriptive zero, **not**
evidence of declining opaque use. This coverage experiment remains pending
additional catalogs and qualifying checkpoints.

Repeat the unchanged reviewed 9b seed-42 MM_ladder measurement: seven updates,
two warmups, five timed updates, CPU eager, batch 2. Compare before/during/after
reconstruction to .1005906649 / .0948241442 / .0928765051, report warmed
sentences/second and actual counts, and require exact packed/single parity at
the recorded .6839025617 byte cost. These short-run results are diagnostics.
For qualified study arms measure 20 warmup + 100 timed training updates and
report actual completed sentences/second, operation counts and memory; no
new throughput claim follows from the short CPU fixture.

## Validation order and resources

Failing probes → fix → affected checks and retained diagnostic gates → one
source-matched full default sweep → stop for Claude's review. Every model/test
worker is capped at 8 GiB; concurrent reservations never exceed 24 GiB. Freeze
runtime/test/config sources during each run, archive all failures and source
maps, and identify opt-in quality cases separately from default-suite passes.
Learned utility is unproven until the full qualified comparisons above pass.
