# Occupation flow and the NP continuation

From main `7a03ae333`, under the
[parts-of-speech specification](../../specs/2026-10-18-parts-of-speech.md)
§2b and §5. **The mechanism certificates pass; description failed its
prerequisite gate, and learning stopped. No completed NP stage, NP–VP fixture
or identity run is claimed.**

Symmetric operations carry every operand's words; not/non preserve their
operand; headed operations carry the declared head operand's words;
clause-forming operations carry none. Adverb declares its modified phrase,
argin1, as head. Kept-program counts and tensor chooser scores follow the
same flow. Each carried occurrence contributes its word's probability
without phrase renormalization or a first-word selector. Both black and cat
in `lower(a, intersection(black, cat))` occupy lower's second argument and
follow that operand into lift; nothing travels through lift. Formal output
slots remain distinct. No stage-3, clock, chain or surface kernel change.

Alec authorized description → adjective → definite adjective, each with
24 presentations / 12 paired updates: two passes, B4, seed **20261010**,
LR **.003**. Each lesson must pass its held-out gates and expected program
before the next restores its pre-evaluation artifact. The authorized
cumulative maximum was 108 presentations / 56 updates, including the
discarded nested branch and authorized reuse of four adjective held-out
cases (not exercised).
The final whole-quarter gate was conditional on all prerequisites passing.

| Executed lesson | Training / paired updates | Held-out brackets | Clean unlowered descriptions | Expected program | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| Bare description | 24 / 12 | 4/4 | **0/4** | **0/4** `intersection` | **Fail; stop** |

All four held-out programs were `lower`, each creating one unexpected
order-1 occurrence. There were no duplicates or collisions. All four
questions were answered correctly despite the structural failure. The
adjective lessons, their authorized held-out reuse and the final 14-item
gate did not run. **Actual cumulative work is 60 presentations / 32 paired
updates**, including the discarded branch. There was no extension or rerun.

Training kept three intersections, two definites and 19 lowerings. The
six-batch remainder of the eight-batch trajectory reports 43 fold targets /
48 lesson calls; reachable-state wait CE **.107229**, fold CE **1.755699**;
first-pair accuracy **3/24**, including four reversed-order readings that
never visited the (0, 1) pair. These reversals explain noun counts on lower's
first argument and adjective counts on its second. Accuracy measures the
kept action, not the chooser's highest score. All 24 kept training programs
and four held-out programs had two leaves and one binary step. Every compose step pays `WHAT_STEP_COST` in the keep
comparison. Two kept intersections had zero closed meaning; the third had norm .11547,
versus .2–.42857 for lower. The competing greedy/explore keep costs were not
saved, so these observations do not establish why a trial won.

[Tables](tables.md) give every word's label/counts before and after, the
noun prior, collapse, first-pair logits and occupation terms, and trajectory.
[Measurements](measurements.json) retain full precision, kept programs,
protocol, source hashes and audits. The [complete collapse table](collapse.json)
contains all 4,186 argument pairs before and after. Intersection's output
counts equal the pooled input counts by construction; its cosine with either
input need not be one. The noun-prior table has no individual-column ties
and is not read by the chooser.

| Role | Artifact | SHA-256 prefix |
| --- | --- | --- |
| Restored passed definite; last passed prerequisite | `output/np-definite-restored-20261018/trained.ckpt` | `b52392d9fc05` |
| Produced failed description; not a continuation fixture | `output/np-description-flow-20261018/trained.ckpt` | `33b2d3e87b9b` |
| Stage 3, unchanged and not rerun | `output/stage4e-student-order-20261017/order.ckpt` | `e19d84e8e708` |

The failed nested artifact was never restored. The definite artifact's word
identities, counts, labels, noun-prior and occupant cosines restored exactly.
The produced artifact passed a fresh-model, zero-presentation restoration
with every table exact. The definite restore had no drift. For the produced artifact, the native
loader changed 83 elements of `conceptualSpaces.0.layers.3.W` by at most
.11547; the existing measurement adapter restored the saved tensors.
Native loading alone was therefore not exact for this failed artifact. Optimizer
moments are not in these experiment artifacts: each lesson starts a fresh
optimizer, while model RNG resumes from the predecessor before its held-out
readings. The separate data-order RNG matched the declared permutations.
Held-out labels and answers were grader-only; occupation/component learning
was frozen for evaluation. No unidentified word leaves occurred.

The prior landing's lexicon index mismatch is recorded as a **tie event**,
per Claude's review and Alec's instruction; its random inputs were not
retained. The repaired test compares a common direct double-distance oracle
exactly and index sets within complete equal-distance groups. A tie crossing
rank k may select different subsets at the same distance. Returned FP32
distances retain their numerical check against the oracle; they are not
claimed bit-identical. Deterministic tied rows exercise the boundary case.
No seed or kernel change was introduced.

Validation before learning: **40 passed, eight skipped** in the focused
group, all six default mechanism certificates included; the explicit native
full-graph forward/backward contract also passed. The subsequent tie-boundary
assertion repair passed its focused check. An earlier certificate run hit
a self-extending event-list loop in the new test; it was repaired before
any continuation presentation. An earlier lexicon assertion exposed the
FP32 evaluation-order difference, leading to the common-oracle check.

Claude approved the source and protocol before learning; the spec wording,
symmetric-flow disclosure and tie-boundary fixes requested in that review
were applied before the run. Claude's post-run review found no source, test
or protocol blockers and approved a mechanism/preparation landing conditional
on the sweep. The single final-source default sweep at
`output/occupation-flow-default-20261018/` completed **5,751 cases: 5,416
passed, 333 skipped, one XFAIL, one carried c failure**, in 1,280.3 s. All six
occupation certificates, five formal-argument checks and the lexicon repair
pass. Carried c closes the question but misses the expected `five` referent;
no new failure was introduced. Slow contracts remain in the weekly tier;
the latest weekly record is overdue at 16.8 days. All 788 learning-source
hashes match the sweep's validated source and final working source. Claude's
final sweep addendum approved this landing with no blockers. The enclosing commit is the archive.
