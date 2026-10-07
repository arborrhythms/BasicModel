# Final operators update — rejected candidate, October 7, 2026

Status: **rejected under plan §39**. Reconstruction and sum at ¼ each reached
9/10, R was not identically zero, and the measured source had no green full
sweep. The [final-b repair](../2026-10-07-operators-final-b/README.md) supersedes
this candidate and was accepted by Alec under plan §45 on October 7.
This status label changes no source, measurements or diagnostic evidence below.

Baseline: `cce3a4f7b6e72fab4a9a27fd70d3fa7602553814` (round 3a), with the
preserved 4a/4a-0 work integrated. This was the uncommitted combined handoff in
plan §§30–38. Its original results remain historical evidence in the accepted
operators-update landing.

## Measurement

All thirty standing trainings are complete. The required miss bisections are
reported separately below; these counts are final and are not replaced.

| Standing gate | Result | Expectation |
|---|---:|---:|
| XOR class, four correct and MSE < .05 | **10/10 at zero** | 10/10 |
| XOR reconstruction, all four sentence word multisets recovered | **9/10** | 10/10 |
| Shared class and reconstruction | **9/10** | 10/10 |
| Sum control, additive | **10/10** | 10/10 |
| Sum control, final MSE within .02 of ¼ | **9/10** | 10/10 |
| Raw MM, best MSE < .20 | **10/10** | 10/10 |

“At zero” is the standing MSE < .05 band, not literal floating-point zero.
Class MSE ranges from 2.0481e-12 to .03307267. MM best MSE ranges from
.14995827 to .19926545. The full [per-run table](measurement-results.md),
[saved summary](measurements/summary.json), and
[source/storage validation](results-validation.json) retain every result.

**Sum-08** stays additive (checkerboard contrast 2.98e-8) but ends at
MSE **.4929095507**. Its predictions are `.00593820, .00717717, .00710717,
.00834617`, a large common-offset miss. **XOR-09** passes class at
MSE **.0016352634** but reconstructs only `hello`, `there`, `world`, `loving`
instead of the two words in each input. Its reconstruction count is 0/4.
Neither run is retried. B, C and D are each switched off individually for
both original entries in the [verified six-run bisection plan](verified-bisections/plan.json).

The first attempted bisection harness patched configuration construction, but
the live singleton was already constructed and used load/overlay instead.
It therefore left every requested switch on. Those six runs and the three
off-labelled bootstrap prefixes are **invalid for attribution** and are
retained unchanged. The [diagnostic error record](diagnostic-override-error.json)
identifies them. The verified controls apply the XML public setter after
load/overlay and assert effective values on every constructed index. This
repair changes diagnostic instrumentation only; the thirty standing results
and measured source remain unchanged.

The [protocol](measurement-protocol.json) declares one complete sweep followed
by thirty fresh, unseeded trainings: ten sum controls, ten shared XOR
class/reconstruction trainings, then ten raw MM controls. There is no selected
seed, retry or replacement. Separate bisections replay the recorded OS-initialized
entry state of the relevant run; they never replace a standing result. The
original [round-3a source](before/source.json), [sweep source](sweep-source/source.json),
and [measured delivered source](delivered-source/source.json) are retained, along with
their ZIPs, diffs, test ports and measurement-helper hashes.

**Sweep limitation:** the single full sweep completed all 5,341 selected cases
on the initial combined source: **5,035 passed, 285 skipped, 20 failed, one
XPASS**. It was not rerun or replaced. The failures exposed stale fixtures and
four implementation seams: dynamic adjacency checkpoint loading, a missing
default on minimal rule fixtures, recursion on deep committed journals, and
stale relation flags on durable references. Repairs and their exact source
delta are in [focused-repairs.patch](focused-repairs.patch). The affected
files and focused contracts then completed **268 cases: 250 passed, 18
skipped, zero failed**, before the thirty trainings. See the
[original failures](sweep-failures.json), [full sweep](full-sweep/result.json),
[focused repairs](focused-repairs-01/result.json), and
[repair verification](repair-verification.json). This is not a claim of a
green full sweep on the repaired source. The raw worker report includes three
extra passing subtest records for `test_flags_match_expected`; the counts
above count each selected case once.

## Delivered contracts

| Part | Implementation and switch | Focused evidence |
|---|---|---|
| A | Mandatory stable 64-bit address from source document, sentence index and ordered identified words; relative `.when`; address upsert; full store raises; reference/compaction/checkpoint migration | [address tests](../../../test/test_operators_round4a0.py), [reader inventory](when-readers.md) |
| B | `meaningWidth`: 128 in grammar gates, 896 in BasicModel; zero disables the values while preserving reserved geometry for attribution. Fixed content-key codes, detached membership mean on the for pole, independent bilattice, negative-activation pole exchange, kept-row expectation update | [meaning tests](../../../test/test_operators_round4a.py), [structural block tests](../../../test/test_operators_final.py) |
| C | `symbolCentroid=true`: adjacent occurrences supply the ceiling and weight; decreasing-part-count projection caps contained words; binding projects `c`, indexing reads `[c == 1]`; room pass retired | [centroid tests](../../../test/test_operators_final.py), [static audit](centroid-audit.json) |
| D | `membershipPriming=true`: concept and sentence-membership edges share a degree-normalized outflow; detached energy reaches the existing retrieval prior | [degree-budget test](../../../test/test_operators_final.py), saved per-run `priming` records |
| E | Declared determiner modes for mint/bind/kind; parameter-free lower, generic and lift; addressed relation names and equality/DEF provenance; operator properties replace semantic name dispatch; predicate identity is a rule property | [catalogue tests](../../../test/test_operators_final.py), the updated lexical/reference/relation/thought tests, [consumer census](consumer-census.json) |

Input source keys come from dataset documents/rows, serving conversation and
turn, or explicit ingestion sources. Direct callers without separate source
metadata treat the immutable input as its own document. Sentence positions
are one-based, following the October 3 amendment. The band is
`0.5 × (Q(i) + Q(i+1))` on the existing spatial coarse/fine frequency ladder,
whose period exceeds the document bound. The live `.where` onset encoding is
unchanged. `when_time` goes only to row timestamps; recency still reads them.
The [inventory fixture](../../../test/fixtures/when-readers-round4a0.json)
covers 75 named readers and 32 dynamic field readers.

Thought rows use their source document/turn and produced-row ordinal. DEF
rows use `(DEF, word, object)`. Legacy rows without recoverable source text
are assigned deterministic migration identities; migration preserves their
references and timestamps rather than inventing original document metadata.
Opaque direct writes retain an explicit native-content fallback. Identified
input sentences use the ordered identified-word content key.

The meaning layout is `[for_0 … for_K−1 | against_0 … against_K−1]`.
Grammar `K=64,s=3`; BasicModel `K=448,s=6`. The private generator consumes no
global RNG. Composed meanings remain in `o`; occurrence means use identity
codes and exclude DEF rows. Undeclared meaning writes pass the designated
operand through. Parameterized form maps receive a zero complement and cannot
mix meaning into form. The switches retain dimensions and parameter shapes
when disabled, so the bisections do not change initialization by resizing.

For centroid weighting, `W_P` counts distinct **native part atoms**, including
length atoms and collision mints; `W_U` counts distinct adjacent occurrences
identified by address and pair position. This implements the handoff's
“part atoms” wording; the adjacent-ceiling toy counted pair atoms only.
Re-reading replaces a row's adjacency witnesses rather than increasing its
weight. No relevance filter was added to attention.

## Static certificates

The [form audit](form-audit.json) finds zero final identity collisions,
indexed reconstruction errors and containment violations on the configured
vocabularies and separate nonvacuous witnesses. The fixed binding projection
is unchanged. BasicModel's static bank contains 67,391 words and 69,566 native
atom/word rows, above the unchanged 32,768 production part-store capacity;
this audit does not claim full-corpus model admission or training. The LTM
capacity is separately 262,144, sufficient for the local corpus occurrences.

The [semantic gate certificate](semantic-gates.json) is exact for all ordered
word pairs, including self pairs, and all four sentence rows in both grammar
configurations. All four word meanings differ. Both negation cases are
included. The focused small-width fixture deliberately produces false
membership and checks the histogram certificate against literal enumeration.

On BasicModel's 90,676 local corpus sentences and 67,391 words, the
[semantic diagnostic](semantic-basicmodel.json) exhausts all
411,809,304,981,556 ordered-pair/row checks per case by exact coverage counts:

| Meaning expression/pole | False membership / all checks |
|---|---:|
| `a ∧ b`, for | 0.0315436470% |
| `a ∨ b`, for | 3.6903728972% |
| `not a`, against | 1.7538262834% |
| `a ∧ not b`, against | 1.7538262834% |
| Either negation case, for | 0 |

There are no false negatives. These rates divide by **all** checks, not only
negative examples. They are diagnostics of the finite code; postings answer
membership exactly. No word-pair or sentence sample was used.

| Centroid audit | Words moved | Mean pairwise cosine, before → after | Containment violations, before → after |
|---|---:|---:|---:|
| Each grammar vocabulary | 4/4 (100%) | 0.4022911373 → 0.4475716628 | 0 → 0 (no native containment pairs) |
| BasicModel corpus | 30,116/67,391 (44.6885%) | 0.4665571763 → 0.4720063244 | 3,114 → 0 of 18,309 comparable pairs |

Every identity is recovered by `[c == 1]`; no coordinate falls below its
native lower bound. The BasicModel adjacency pass uses 88,380 word-bearing
occurrences (empty-word rows do not contribute adjacent pairs), yielding
1,399,716 distinct adjacent witnesses. The containment fixture is nonvacuous
even though the four gate words have no native inclusions.

The [consumer census](consumer-census.json) records 72 declared rules and 46
consumer sites, with zero runtime semantic comparisons on `op_name`,
`rule_name` or `method_name`. It retains nonsemantic inventory/diagnostic
comparisons separately. This is the stated AST inventory, not a claim that
every variable named `name` in the repository denotes an operator.

## MM attribution

The [paired 200-epoch comparison](mm-bisection/result.json) does **not** confirm
the expected RNG-only path: construction matches, but predictions differ from
the first forward and no epoch is identical. The
[zero-training first-forward comparison](mm-when-bisection/result.json)
holds Python, NumPy and Torch states, prepared inputs and the IR mask equal.
Restoring only the old time band restores the landing's prediction exactly.

The [reader-site bisection](mm-when-readers/result.json) isolates
`PartSpace._embed_radix`: its relative band travels through `_embed_ladder`
into the raw MM numerical forward. Restoring the old encoding only at that
site restores the landing result. Doing so only in `InputSpace._lex_and_embed`,
`PerceptField.read_percepts`, or DEF writes does not. Every variant retains
the same RNG and IR mask. The mandatory A change is kept; this discrepancy is
reported for review, not hidden as random drift.

## Verified miss bisections

All six controls use exactly the original failed run's saved unseeded entry
state; the saved state files are byte-identical. Each `effective-switch.json`
asserts the setting on every constructed index. See the
[bisection report](bisection-report.json) for original and variant costs,
first differing epochs, final operators, priming, storage and raw record paths.

| Original entry | Switch disabled | Final MSE | Relevant result |
|---|---|---:|---|
| sum-08 | B, meanings | .2500000894 | ¼ pass; additive |
| sum-08 | C, centroid | .2499999702 | ¼ pass; additive |
| sum-08 | D, membership priming | .2504028976 | ¼ pass; additive |
| xor-09 | B, meanings | 4.0489e-11 | Class and reconstruction pass |
| xor-09 | C, centroid | .0000113094 | Class and reconstruction pass |
| xor-09 | D, membership priming | .0003440356 | Class passes; reconstruction still fails |

Every individual switch changes sum-08's fit enough to pass, so this entry
does not single out one part as its unique cause. For xor-09, B and C each
move the failed reconstruction gate; D does not. The D-off readbacks remain
the same four single words as the original. These comparisons identify effects
on these captured entries, not general success rates for disabled variants.

The sum cost trajectory first differs at epoch 3 for B, and epoch 2 for C
and D. C-off alone removes sum-08's bootstrap R; B-off and D-off retain the
epoch-2 R. On xor-09, nonzero ΔR counts are **73 on, 0 B-off, 8 C-off,
73 D-off**. C-off has no bootstrap R, although it later has eight nonzero
narrowing differences. B-off retains the common epoch-2 R but removes the
later ΔR. These observed departures from R ≡ 0 remain review findings.

## Trial costs, final operators and priming

Every grammar run retains `run-audit.json`: both trial costs and R/E/A
components, keep decisions, signed advantages, departure probabilities and
proposal scale, reader exposure, and final operators. The
[aggregate](aggregate-audit.json) includes all 32,000 row comparisons, with
examples and the final twenty epochs per action. The
[absolute-cost and priming audit](cost-and-priming-audit.json) checks absolute
costs in addition to differences; its input hashes identify the saved records.

**R is not identically zero.** All sum runs have `R=.01803368889` at epoch 2
on both trials and zero otherwise, so their ΔR remains zero. XOR has **1,221
nonzero ΔR** row comparisons: 1,146 on `not`, 75 on `descend`. The maximum
absolute R is the same `.01803368889`. XOR-09 retains nonzero absolute R
through epoch 400. E and ΔE are zero throughout. The
[three-epoch bootstrap diagnostic](bootstrap-cost-bisection/result.json)
exactly reproduces sum-01's all-on prefix with its original entry state.
Its off-labelled cases are invalid because of the override error above and
cannot establish any part's effect.

Across XOR, 75 comparisons keep explore on lower R; 14,779 tie on R and
keep greedy. There are 9,214 nonzero total advantages. Narrowing credit is:

| XOR narrowing action | Departures | Nonzero credit | Rewards explore / greedy | ΣΔR | ΣΔA |
|---|---:|---:|---:|---:|---:|
| `not` | 1,153 | 1,146 | 165 / 981 | 20.66660747 | 238.69739786 |
| `descend` | 353 | 75 | 35 / 40 | −1.35252667 | 2.51572140 |
| `gloss` | 3,934 | 0 | 0 / 0 | 0 | 0 |
| `and` | 1,236 | 0 | 0 / 0 | 0 | 0 |
| `or` | 1,331 | 0 | 0 / 0 | 0 | 0 |

Positive ΔC rewards greedy, negative ΔC rewards explore. In the final twenty
epochs, all **47** sampled `not` departures receive nonzero credit against
the departure. The answer term contributes substantially, but R also
contributes; this is not the stipulated answer-only/R-zero result. All
16,000 sum comparisons tie, including its narrowing actions.

Final greedy XOR composition contains **37 conjunctions and three
disjunctions** across forty inputs. XOR-02 is mixed: disjunction for `hello
world`, `loving world`, `loving there`; conjunction for `hello there`.
It passes both gates. The other nine runs finish with conjunction on all
four inputs. Mixed operators are permitted by the handoff and are preserved.

The same final priming snapshot occurs in all twenty grammar runs. In batch
order `hello world`, `hello there`, `loving world`, `loving there`:

| Word | Membership energy received in the four streams |
|---|---|
| hello | 1.389703512, 1.389703512, .860292196, .860292196 |
| world | 1.389703512, .860292196, 1.389703512, .860292196 |
| there | .860292196, 1.389703512, .860292196, 1.389703512 |
| loving | .860292196, .860292196, 1.389703512, 1.389703512 |

The corresponding resulting word weights are **8.352931023** at the larger
entry and **3.647051573** at the smaller one. These are the last recorded
diffusion and weight snapshots, not an average over epochs. The raw numeric
MM path has no sentence-membership nodes and no grammar narrowing trials.
All recorded final word-read decisions are code winners without a priming
tie-break; that does not establish that priming had no earlier policy effect.

The detailed ownership observer is on the existing XOR-10 training, not an
extra training. It reports zero ownership conflicts, one reader update per
epoch, maximum chooser-gradient error 2.2352e-8 and finite-difference error
9.0498e-9. The closing image remains zero. All 256,000 identified rung-zero
reads across the twenty grammar trainings recover their identities exactly.

## Store census

Every sum/XOR training ends at **8/1,024 rows: four sentence rows and four
DEF rows**. Each sentence is witnessed exactly 400 times before evaluation;
evaluation re-witnesses the same addresses. Content keys and addresses are
the same across all twenty runs. The
[validation](results-validation.json) records exact addresses, timestamps,
witness counts and final usage for every run. All storage checks pass.

The unchanged raw numerical MM path has no completed sentence closing. It
ends at **4/1,024 rows, all DEF**, in each of its ten runs. Thus the receipt
does not claim four sentence rows for raw MM; the four-row sentence invariant
is exercised by the grammar gates and the focused 400-presentation fixture.

## Review boundary

The earlier [4a](../2026-10-07-operators-round4a/) and
[4a-0](../2026-10-07-operators-round4a0/) material is preserved as development
and historical evidence. Their measurements are not substituted into this
receipt. Executable source remains the version measured after the focused
repairs. After all measurements, one trailing space was removed from a test
assertion; its AST is identical. The
[formatting delta](postmeasurement-formatting.json) and
[review source](review-source/source.json) account for that byte difference.
Later receipt postprocessors and diagnostic scripts carry their own digests
and do not modify the measured implementation. The
[final integrity check](final-integrity.json) verifies the review tree,
frozen helpers, retained sweep and repairs, storage, bisections, receipt links
and clean whitespace. The next action is Claude's review.
