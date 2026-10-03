# Item 6.9 — reconstruction owns the codes (§20)

The closing measurements are finished; stop for review. **The round is not accepted and the remaining ports are incomplete.** One working tree at published HEAD `d679df2b5a2665d72a99ca4b6dfd47c1ba048e99`; no commits, no HEAD run. The ownership round remains unaccepted in [its original receipt](../2026-10-02-item6-9-ownership/README.md). This is the single receipt for the §20 continuation.

The completed primary measurements are **class 0/10**, **reconstruction 2/10**,
and **sum control 10/10**. The named table has **31 passing / 3 failing retained
cases**: the two predeclared first grammar gates and MM_xor's deferred
convergence proof (MSE .236159 at its unchanged .20 bar). Both XOR_exact checks
and the single MM_20M exact round trip pass. The table contains 34 cases after
the earlier suite trim, as in the ownership receipt; the item-7 landing had 49.
The required attribution is triggered by both primary gate counts. MM_grammar's
ten runs completed with median ending training MSE **2.20011×10⁻⁷**; one ended
at **.2500589**, the known plateau. Attribution, moved-case attempts and the
source-matched full sweep are complete. The full sweep has **8 failures**;
the moved cases have **5 failures and 3 stopped attempts**. These are retained
for review, without a passing rerun substituted for a failure.
All answers, read-backs and contrasts are in [the measurement tables](measurements.md).

The required attribution completed in **65.32 minutes**, with all 40 processes
exiting normally. Each arm uses ten fresh, unpaired unseeded models:

| Writers | Class passes | Reconstruction passes |
|---|---:|---:|
| Reconstruction | 0/10 | 1/10 |
| Reconstruction + expectation | 0/10 | 3/10 |
| Reconstruction + answer | 1/10 | 1/10 |
| All three | 2/10 | 2/10 |

Without the answer writer, its reader is untrained and its cost is omitted
from the trial comparison; those class outcomes are diagnostic. With all
three writers, two fresh runs learned XOR, but these do not replace the
predeclared primary class outcome. The small, unpaired arms do not isolate
the cause of the differences. Incomplete reconstruction also occurs with
the other two writers disabled.
The class and reconstruction passes occurred in different runs: **none of
the 40 attribution models met both existing bars**. This is a derived
observation, not an additional acceptance criterion.

## Implementation and probes

Reconstruction owns trainable concept dictionaries in every binding. The five surviving configurations from the eleven canonical rotation configurations now specify rate 0; the other six were deleted in the preceding accepted cleanup (see [dispositions](configuration-dispositions.json)). Similarity pressure stays 0. The concept VQ EMA refresh is disabled; rotation, the feature-derived code refresh, promotion/phrase code replacement and post-step normalization have been removed. Native admission only assigns identity and relations. Lookup preserves the parameter's magnitude. No new distributional objective, code constraint or answer reach was introduced.

The leaf remains code × signed activation. Reconstruction registers `reconstruction.bytes` (witnessed inverse) and `reconstruction.free_bytes` (no operand reference or witness offsets; both operands searched over the primed bank). Each is byte cross-entropy divided by `log(256)`, with its existing active-word/sentence reduction, and each carries `reconstructionScale`. Free read-back scores a candidate as `a*cosine(leaf,code)*priming`, with `a=dot(leaf,code)/dot(code,code)`. Zero codes score zero. Candidate code values remain live, including competitors; retrieval addresses and priming weights are detached. The gate reads this same free inverse and scoring. The answer still detaches the shared understanding. XOR_exact's evidence coefficients are answer-owned, their only reader.

The archived pre-change source is [before.zip](before.zip), with hashes in [before.json](before.json) and the environment in [environment-freeze.txt](environment-freeze.txt). Every probe directory contains its command, source hashes, guarded process result and complete output. Saved failures precede repairs: `ema-before`, `free-before`, `ownership-law-before`, `norm-before`; the unported sweep/moved cases in [prior-failing-cases.json](prior-failing-cases.json); `feature-writer-before`, `admission-code-before`, `phrase-code-before`; and the successive port probes under [probes](probes).

For review, [source-diff.patch](source-diff.patch) isolates this round's code,
configuration and test changes against that archive, including the two new
regression files and retired rotation test. Its [file index](source-diff-index.json)
has 58 entries. The archived source and [frozen candidate](source-final.zip)
remain the complete references; documentation is recorded separately and is
not part of this patch.

The live-training EMA probe (`training-ema-before`) observed **zero quantizer calls** in three XOR batches even before disabling EMA. Its unchanged cluster counts do not establish EMA as the cause of XOR's prior failure. The explicit quantizer regression now checks that neither W nor cluster counts changes. Feature refresh, promotion and phrase admission did rewrite codes without an optimizer step in the saved probes.

## Ports and owner repair

[The port index](ports.md) links every changed test file's old and new bodies; [ports.json](ports.json) lists all old test IDs, replacements and reasons. Each linked pair contains the **complete old and new files**, including every test body and fixture. The three expectation ports require detached sources/targets and nonzero predictor gradients. Rotation and unit-projection assertions retire under §20; their gradient-parameter replacements do not retain those old laws. The retired wordStore summary and detached grammar-student tests are removed with their paths, as decided earlier. No kept bar is relaxed.

The sparse-field failures had a live owner mismatch: admission wrote the terminal concept owner while field reading, passback, reverse and priming used the empty first stage. These paths now use `_concept_owner()`. Native DEF descriptions supply PS/WS priming references without the redundant word/META bridge. Repeated-read comparisons fix their admitted vocabulary; otherwise the second read promotes new percepts and does not start at the same state. The kept attention smoke fixtures retain their feature switches at smaller test geometry. The grammar reconstruction harness observes the free inverse at sentence commit before its journal is discarded, with target/decode row alignment retained; it never substitutes source leaves for an unavailable inverse.

The focused checks are verification of these repairs, not extra gate receipts. One broad diagnostic was interrupted when a moved two-epoch smoke fixture unexpectedly loaded the full FineWeb sample; that fixture now uses its original XOR subject and two capped training batches. Another multi-file diagnostic hit the unchanged 8 GiB process guard because it retained several production models; individual affected checks were then isolated. The final runner recycles bounded workers. An unrelated `TestLoadEmbeddingsEnwiki` check reported its missing embedding dataset; it was not repaired in this round.

## Closing protocol

[closing_campaign.py](closing_campaign.py) freezes the candidate source, runs the native production-batch stage-1 observation, then the named table and 10 unseeded class, reconstruction, sum-control and MM_grammar runs. The first predeclared class/reconstruction run supplies its named table row. There is one MM_20M exact round trip and no HEAD model. If class is below 8/10 or reconstruction is at most 3/10, the receipt-local R/RE/RA/REA attribution runs ten times per arm. Moved slow cases follow, then one source-matched full sweep on ten workers. Already-started failures and guard stops are not retried; continuations dispatch only never-started cases.

Ordinary workers retain 8 GiB; the previously authorized native-only production test retains 24 GiB at batch 28. The full sweep keeps the reference ten-worker schedule and the machine's ordinary aggregate reservation. No training seed, bar, production threshold or guard was changed. The audit lists declared owners and observed backward writers; zero conflicts are required. It also saves initial/final dictionary pairwise cosines and XOR-root singular values, and final VQ cluster sizes. Native full-reserve pair moments are exact without materializing a 65,536-square matrix; full pairwise matrices cover every observed primed row.

Comparison: §14 class 8/10 and reconstruction 3/10; prior reference sweep 5,219 cases / 122 minutes. MM_xor convergence remains red by §17 decision: its historical success used cross-word percept lookup, removed by meronomy. Its test, bar and markers are unchanged; word-level XOR is deferred to item 6.8. MM_grammar's known .25 stop remains a declared exception. All closing outcomes are retained, including failures.

## Audit findings and limits

The XOR audit reuses the first class run, and the native audit reuses the
production stage-1 run. Their recorded backwards have **zero ownership
conflicts** (18 active / 55 inactive parameter records for XOR, 45 / 108 for
native), and both trials start at the same parameter-version hash. The
selection audit reports zero violations of the reconstruction-precedence
rule. Inactive parameters and absent objectives are identified rather than
counted as evidence of gradient reach; expectation is absent in the native
observed trials.

The active XOR dictionary's code norms end between 1 and 2.35108, from an
initial norm of 1. Its VQ cluster counts stay `[1, 1, 1, 1, 1, 1]`; the native
indexed allocation has no cluster-count buffer. Dictionary pairwise cosines
and initial/final root spectra are retained in the audit. Neither fixture
supplies an activated outside-word competitor: **0 of 6,408** observed XOR
word occurrences and **0 of 216** native occurrences have one. The measured
outrank count is therefore zero but does not test competition with activated
words. These audits establish observed writer separation, not successful
cooperation or improved learning.

## Moved-case results

The moved-case campaign attempted all **165 newly dispatched cases** in
**23.57 minutes**: **157 passed, 5 failed, 3 stopped**. Seven further selected
cases reuse their passing named-table results, including the single exact
MM_20M round trip. [The result](extra-cases/complete.json) retains both
segments; only cases that had not started were dispatched in the continuation.

The kept attention coverage passes: six global-attention checks, seven
global-consume checks, eight reading-attention checks and the query-reasoning
training case. The real MNIST subset also trains its short batch with ergodic
off and on (1.49 and .93 seconds respectively), and the LFS-pointer loader
message check passes in the full sweep. These remain smoke/mechanism checks,
not additional convergence receipts.

The failures remain unwaived:

- Serial output memorization reached best training accuracy **.96875** at
  its unchanged **1.0** bar after 40 epochs. Its prior XML setup failure is
  repaired, so the earlier blocked result supplies no learning baseline.
- The three-epoch MM_ladder free-derivation round trip again reports **0.0**
  exact reconstruction against **1.0**, as in the ownership receipt.
- Two compiled/eager reconstruction tests still compare a witnessed-only
  helper with the new witnessed-plus-free cost and free ideas. Their helper
  ports remain incomplete; the current results compare different contracts.
- The two-phase activation test still reads the first-stage capacity (15)
  while the repaired terminal owner supplies 8 rows. Its observer port
  remains incomplete.

The worker running `test_radix_decode_gates_pad_slots` crossed the unchanged
**8 GiB** ceiling at **8.06 GiB**, after the free-derivation failure in the
same worker. This is not an isolated model-memory measurement. The runner
then aborted the active expectation and Inductor compilation cases; neither
reached its own memory or timeout ceiling, and neither produced an assertion
outcome. Those stopped cases were not retried. The [cause comparisons](failure-notes.json)
distinguish these resource stops, remaining ports and learning failures.

The full sweep collected **4,980 cases**, against the requested reference of
5,219 cases / 122 minutes. The previous ownership sweep had 4,976: this round
adds eight behavioral cases and six receipt-link checks, and retires ten
collected cases (nine functions, one parameterized for CPU/MPS). Renames net
to zero. [Every added and removed case ID](case-count-comparison.json) is saved.

## Full sweep and remaining work

The single source-matched sweep finished in **22.23 minutes** on the reference
**ten-worker schedule**: **4,686 passing cases, 8 failures, 285 skipped and one
non-strict XPASS**, with no process stops, unattempted cases or compile-cache
retries. The peak aggregate process-tree RSS was **9.58 GiB**, below its 28 GiB
reservation; each worker retained the 8 GiB ceiling. The requested comparison
is 5,219 cases / 122 minutes; the immediately preceding ownership sweep was
4,976 cases / 19.37 minutes. These are whole-suite measurements on different
candidates and case sets, not paired timings of the code change.

[The case summary](full-sweep/case-summary.json) counts each selected node once.
The runner's [raw receipt](full-sweep/receipt.json) reports 4,689 passes because
the three `unittest.subTest` configurations inside `TestOrthogonalFlags` each
emit a passing report in addition to their enclosing case. That is three extra
reports, not three repeated model tests. The weekly coverage warning remains:
the latest weekly slow run contains failures; this round does not replace it.

All **23 prior full-sweep failures pass**. Of the prior **60 failed or stopped
moved cases**, **51 pass, 2 fail, 5 are retired by the decided contracts, and
2 stop as aborted peers**. A retirement is not counted as a passing port.
[The triage](triage.md) lists every original ID and its current outcome.

The eight full-sweep failures are seven remaining fixture/observer ports and
one compiler failure:

- The definition-attribution stub lacks the terminal `_concept_owner` accessor.
- The gradient-factorization test still requires the retired buffer dictionary.
- Four separator cases wrap the byte-cost function without its new `priming`
  keyword; they fail before reaching the separator assertions.
- The live-reference observer still compares the free inverse's ideas with
  the exact witnessed leaves. Its preceding capture/storage/reference checks
  pass, but its subsequent gradient assertion is not reached.
- The fifth fresh-model stage-cache case hits Dynamo's unchanged eight-capture
  limit in `_reconstruct_sentences`. The last guard mismatch is the new `free`
  branch. The saved trace does not contain the full specialization history.

These ports and the three moved-case fixture ports must be finished; the
compiler capture failure, two learning failures and three stopped attempts
also remain unresolved. No compatibility fallback, weaker assertion or higher
limit was added to hide them. Source was frozen before the closing campaign;
this receipt records the failures rather than changing source after the sweep.
The gate results are below §14's comparison counts, and zero audited ownership
conflicts do not establish improved learning. Item 6.9 remains open for review.

Final verification: all **664 frozen source files** still match, HEAD is
unchanged, and the package freeze is identical. All **46 complete old/new
test-file pairs** match their source archives (645 old test bodies indexed).
`git diff --check` is clean and all **263 documentation-link checks** pass
after the receipt, GradientFlow, FutureWork and todo updates. The verification
record is `final-verification.json`; no training measurement was repeated for
this documentation check. Nothing is committed.

## Timing diagnostics

The production test passed at batch 28 in **1,714.71 seconds (28.58 minutes)**,
with a process-tree RSS peak of **23.63 GiB**, under the unchanged native-only
24 GiB ceiling. It does not fit the ordinary 8 GiB worker. The previous
ownership measurement took 893.76 seconds and peaked at 21.40 GiB; these are
separate unseeded measurements, not paired initializations. Both trials were
costed at the same parameter-version hash. The ownership audit reports zero
conflicts. Full term magnitudes, gradients, selection and geometry are in
[the audit report](audits.md), with the exact values in [audit-summary.json](audit-summary.json).
The saved pairwise cosine matrices are also available as [readable tables](geometry-tables.md)
and [JSON](geometry-pairs.json), without rerunning either model.

At about sixteen minutes, before the first trial report, a one-second macOS
`sample` observed the main thread under Dynamo's compiler callback and the
autograd engine, with Python operator dispatch/functionalization on the stack.
This is evidence of compiler/backward graph construction at that point, not a
completed numerical training step. The sample's physical footprint was 13.4 G;
the bounded runner separately records its process-tree RSS peak. The previous
ownership receipt first reported its greedy trial at 545.8 seconds. The
[complete stack sample](native-stack-sample.txt) is retained. Sampling was
read-only and did not change the model, RNG, source, optimizer or guard; its
one-second interval is included in this run's wall time.

The first MM_grammar repeat was also sampled once, about seven minutes after
launch and before its first 50-epoch progress record. All 699 main-thread
samples were below Dynamo's compiler callback, with a 2.0 G physical footprint.
This establishes compilation activity at that instant, not a full timing
breakdown or evidence of a new recompile cause. The [MM stack sample](mm-01-stack-sample.txt)
is retained; no source, model state, RNG, compile setting or guard was changed.

At about 15 minutes into the moved-case workers, one-second samples of
`test_unpacked_forward_observes_each_published_sentence_once[True]` and
`test_real_aligned_loop_matches_prior_compiled_semantics_across_chunks`
both found the main thread under Dynamo's compiler callback. The Inductor
case also shows tensor functionalization. Both samples reported a 3.4 G
physical footprint; the bounded runner's RSS measurements are separate.
The [expectation sample](moved-expectation-stack-sample.txt) and
[Inductor sample](moved-inductor-stack-sample.txt) identify compiler work at
that instant, not a full runtime breakdown or a measured recompile count.
These were read-only observations with their one-second intervals included
in the recorded runtime; no model, RNG, backend, source or guard changed.
