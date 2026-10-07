# Operators final-b repair — October 7, 2026

Uncommitted repair of the measured combined source, following plan §§39–42.
The [first receipt](../2026-10-07-operators-final/README.md) is preserved intact.
Claude review is required before any commit; item 6.5 is pending.

The [measurement](measurements/summary.json) is complete on the frozen source:

| Gate | Result |
| --- | --- |
| XOR class at zero | 10/10; MSE 0 to 1.34×10⁻⁸ |
| XOR reconstruction | 10/10; all four word multisets in every run |
| Sum at ¼ | 10/10; MSE 0.249999985 to 0.25 |
| MM_xor, live gate | 10/10; best MSE 0.17037 to 0.19803 |

**R and E are identically zero on all 64,000 grammar training trial rows**,
including epoch 1 and the first nonzero-meaning snapshot. Evaluation is zero
too. No seed, training retry or replacement was used. No miss required a
bisection. The [receipt summary](receipt-summary.json) verifies the exact
source, measurement helpers, original receipt integrity and store bounds.

The [final frozen source](measured-source/source.zip) and
[delta](measured-source/changes.patch) start from the measured combined source. The first repair's
[source delta](delivered-source/changes.patch) is retained as history; §42
supersedes its two-orientation candidate expansion. Inverse shortlisting,
pair residuals, decomposition eligibility and readback now identify words
from form alone, with one entry per word. After selection the inverse reads
each meaning lane independently against the frozen context code. Min/max
can lose operand evidence; their inverse recovers compatible magnitudes,
without consulting the input derivation. The mean uses a nonnegative
two-code solve.

Four-cornered logic requires two independent evidence lanes. Interpretation
now makes `[form × presence | code × c⁺ | code × c⁻]`, and both the pushed
slab and the reference slab retain the pair. Neither a difference nor a sign
intervenes before storage. Closing no longer compares the two magnitudes to
choose Boolean polarity. Its required-evidence read takes the minimum of
nonzero contributions in each pole: A-true and B-false stores **both**.
Extent-code conjunction still takes coordinatewise minima including zeros,
as specified in part B. The two zeros remain distinguished as in plan §42.
The [corner certificate](tetralemma-certificate.json) covers all leaf corners
and connective corner pairs. Its tests also traverse the actual handoff,
interpretation, pushed/reference slabs and closing. The
[AST census](tetralemma-path-audit.json) lists 24 path sites and every direct
access to the retained evidence fields, with zero violations; mutation tests
reject injected sign, cross-pole arithmetic/comparison and lane reductions.

The snapshot boundary now precedes lexical staging. Previously the eager
staging path built composition inputs before `forward` refreshed the inverse's
meaning/centroid bank. Each codebook is now snapshotted once per forward and
held until committed writes finish. The [frozen-prefix certificate](repair-certificates.json)
recovers 4/4 multisets with exactly zero pair residual and R on both readings,
including the first reading with nonzero meanings and centroids, with B/C on.
The focused training test also checks both trials across bootstrap.

The requested [original Sum-08 replay](diagnosis/sum08.json) reproduces every
final prediction exactly, MSE **0.4929095506668091**. [Every epoch](diagnosis/per-epoch.json)
retains root form/meaning norms, presented-reader pre-activations, outputs and
gradient norms. The affine reader can represent and reach the mean: epoch 380
outputs are 0.49921–0.49922. It subsequently becomes unstable. Root norms stay
about 3.82–4.06 (form) and 1.06–1.17 (meaning), while record feature norms rise
from 15.7–16.8 to 89.3–90.6 as priming accumulates. Reader-weight gradient norm
rises from 0.097 at epoch 380 to 57.689 at epoch 395. There is no readout
saturation; both the readout and output denormalization are affine.

`readerBlockNormalization=true` divides form and meaning by separate, common
snapshot scales (maximum bank block norm, clamped below by one). The echoic
bank has its own common block scales. This is affine in every trial root;
it never normalizes a root by its own norm, alters composition, or writes
normalized meaning to LTM. `primingMaxBoost=2.0` bounds detached retrieval
boosts around neutral 1.0 while retaining the diffusion energy ledger.

The [protocol](measurement-protocol.json) requires a green full sweep of
this exact source, then ten sum, ten shared XOR class/reconstruction, and ten
MM_xor trainings, without seeds, retries or replacements. MM is a live gate;
its relative `.when` band through `PartSpace._embed_radix` is intentional.
The full sweep completed all 5,354 cases in 426 seconds with three failures.
[The failures](sweep-failures.json) and the [original sweep](full-sweep/result.json)
are retained. No standing training was started on that failed source.

The legacy-width fixture exposed a missing width guard: block handling
must apply only when the vector width matches the declared meaning layout,
just as `compose_blocks` already requires. The other two failures were copied
READMEs inside extracted executable snapshots, which intentionally lack their
original documentation trees. The documentation census now excludes those
executable copies while checking receipt prose and live project documentation.
The original receipt has not been edited.

The [corrected source](after-sweep-source/source.zip) and its
[two-file delta](after-sweep-source/changes.patch) are saved separately from
the swept source. All **350 focused checks pass**, including the original
legacy fixture, all documentation links, both training trials across bootstrap,
and the frozen-prefix certificate. This is **not a green full sweep**.
The user subsequently authorized the additional sweep in conjunction with
the §42 repair. The first sweep and all diagnostic evidence remain intact.
Thirty standing trainings and their cost/store reports follow a green sweep
on the final source. No training has been retried or replaced. No commit or push.

The [final-source sweep](green-sweep-summary.json) is green: **5,500 cases**,
5,214 passed, 285 skipped and one non-strict XPASS, in 411 seconds. All source
hashes match the frozen measured source. The thirty standing trainings then
completed in 1,254 seconds, with the ten sum controls read before XOR began.

The in-progress [tetralemma sweep](tetralemma-sweep/result.json) stopped at
4,072/5,500 cases when its source guard detected the `non` footprint correction:
its already-correct meaning write now declares `meaning` beside `poles`.
The [focused check](footprint-focused.log) passed 183 cases. This incomplete
sweep is retained and does not validate the final source; the complete
[final-source sweep](green-sweep/result.json) is recorded separately.

The [form audit](form-audit.json), [gate semantic certificate](semantic-gates.json),
[corpus semantic diagnostic](semantic-basicmodel.json),
[centroid audit](centroid-audit.json) and [consumer census](consumer-census.json)
were run on this repair. All four gate words move under the centroid; their
mean pairwise cosine changes from 0.40229 to 0.44757. On 67,391 corpus words,
44.69% move, cosine changes from 0.46656 to 0.47201, identity is recovered,
and containment violations fall from 3,114 to zero. The semantic diagnostic
covers all 90,676 admitted sentences: false-membership rates are 0.03154%
for conjunction, 3.69037% for disjunction and 1.75383% for the against extent
of negation. Exact membership remains a postings query.

[Integrity verification](prior-receipt-integrity.json) confirms that all 2,119
files of the original combined receipt are unchanged.

Final greedy operators are **36 conjunction and 4 disjunction** over the forty
XOR sentences (nine all-conjunction runs and one all-disjunction run). Both
read accurately. The [cost audit](cost-and-priming-audit.json) and
[policy audit](aggregate-audit.json) retain every original per-trial record
and aggregate the sampled narrowing departures. For XOR:

| Narrowing action | Departures | Credit against departure | Credit for departure | Tie | Sum ΔA |
| --- | ---: | ---: | ---: | ---: | ---: |
| and | 1,165 | 9 | 2 | 1,154 | 0.15469 |
| or | 1,382 | 0 | 0 | 1,382 | 0 |
| not | 1,099 | 753 | 289 | 57 | 214.03124 |
| gloss | 3,417 | 0 | 0 | 3,417 | 0 |
| descend | 943 | 231 | 62 | 650 | 75.01460 |

ΔR and ΔE are zero for every action; all nonzero credit is the answer term.
During the last twenty epochs, `not` receives credit against 40 departures,
for 15, and ties on 8. Thus the answer generally penalizes these departures;
the records also retain the favorable cases. Reconstruction ties keep the
greedy root under 2e's reader rule. The tenth run's
[ownership audit](measurements/xor-10/ownership/ownership.json) reports zero
conflicts over 800 backwards, with its gradients and inactive owners retained.

Priming is bounded at **[1.0, 2.0]** throughout the twenty grammar trainings,
with neutral 1.0. Final per-word weights are identical across these runs:

| Word | hello world | hello there | loving world | loving there |
| --- | ---: | ---: | ---: | ---: |
| hello | 2.0 | 2.0 | 1.36000 | 1.36000 |
| world | 2.0 | 1.36000 | 2.0 | 1.36000 |
| there | 1.36000 | 2.0 | 1.36000 | 2.0 |
| loving | 1.36000 | 1.36000 | 2.0 | 2.0 |

Before normalization, membership diffusion contributes 1.389704 to each
present word and 0.860292 to each other word in the same batch row. The full
precision values are in the cost audit. The numeric MM path has no sentence
membership nodes.

Every grammar training finishes with **8/1,024 rows: four DEF and exactly
four sentence rows**, each sentence re-witnessed 400 times. Evaluation
re-witnesses those same addresses once more. Each MM_xor run uses 4/1,024
rows, all DEF: its existing numeric raw-forward path has no sentence closing.
Per-run addresses, content keys, timestamps and witness counts are retained
in the [store report](receipt-summary.json).

All requested work is held uncommitted for Claude's review. Item 6.5 has not started.
