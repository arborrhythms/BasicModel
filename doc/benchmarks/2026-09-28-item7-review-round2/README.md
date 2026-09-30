# Item 7, review round 2 corrections

Status: stopped for Claude's review, 2026-09-29; not accepted or committed.
This receipt
implements [the ordered §14 work list](../../specs/2026-09-16-two-truths-ideas-and-relations.md#14-hand-off-to-codex-claude-2026-09-28-what-to-change-after-review-round-2).
The prior [30-failure receipt](../2026-09-28-item7-review/README.md) remains
available. No commit, push, submodule bump, or nanochat edit is part of this pass.

**Still red:** all 5,026 selected cases are accounted for: **4,678 passed,
22 failed, 322 skipped, one expected failure, and three memory stops**.
All **134 item 7 cases passed**. Sixteen previously passing tests now fail,
and three others that previously passed hit the unchanged memory guard.
These regressions are retained below and in the
[failure ledger](full-sweep/failure-ledger.json).

## Changes and probes

* Sentence entry now ensures grad anchors, including raw forward. Each trial
  passes its restored detached banks through `_carries_with_grad` before its
  first word. The new carry probe and the unchanged raw-forward/interleaving
  compiler tests all pass (4 cases; `probes/grad-after-eager-probe`).
* The grammar lesson's weighted generate objective is reported in the
  sentence gradient report's `output` column where it joins the sentence cost.
  The failing missing-column probe and the existing diagnostics pass after
  the change (`probes/output-before`, `probes/output-after`).
* Pair-driver assertions inspect the selected end state, count one commit,
  and require an empty operation record after closing. Truth-ingestion tests
  count user-origin rows and inspect independent scalar trust and neither
  evidence. Reduction runs through its deadline, including a deliberately
  unary-preferring chooser. Symbolic tests address SymbolSpace. Eleven
  retired API cases retain their exact bodies, hashes, and rationale in
  [the retirement record](retired-tests.json). The continuous-symbol assertion
  is ported and kept intact.
* Reading temporarily forces interpret's word-admission face for every
  observed native word, including mixing configurations. Canonical aligned
  reading already interprets its words. Native parts and property references
  supply the word evidence; no referent is invented for an unlearned word.
  Word definitions remain available at symbolic recursion budget zero.
  Reset does not readmit the same words as anonymous slots. SymbolSpace
  publishes the paired concept activations; its address bands no longer
  discard the opaque content. A completed higher-order field is preserved.
  The code and todo mark the forcing temporary. `XOR_grammar.xml` is unchanged.
* Byte scoring compacts valid candidates before the dot products and softmax,
  keeping their order and the static bank capacity. The captured failing
  inputs now give bit-identical values and gradients. The native packing
  assertion and tolerances are unchanged. The regression fixture comes from
  the previously recorded failure, and performs no random initialization.
  The slow native packing file also needed an observation port: reconstruction
  now runs per sentence, and the helper had overwritten a shorter row's live
  end state with the following sentence's empty slot. The capture is indexed
  by sentence. The three failures before this port remain in `explicit/group-06`.

Each correction has a retained failing probe before it. Intermediate failures
and the deliberately interrupted first compiler-heavy carry probe remain in
`probes/`; they are not counted as completed validation. The final affected-file
run completed 432 cases: **420 passed, 12 skipped, no failures**
(`probes/affected`).

## Measurements and gates

The reconstruction protocol is the reviewed four validation batches, seven
training batches (two warmup, five measured), and four validation batches,
with two sentences per batch. Seeds 0, 1, and 2 are declared in advance for
both HEAD `1678ee1fb79c474ffaee5725fd035716de5f1913` and the candidate. Each
runs in a fresh process with an 8 GiB guard and the existing 1,200-second
measurement limit. This is a short synthetic-corpus comparison, not the
million-sentence learning campaign. All six processes completed successfully:

| Tree | Seed | Before training | Training | After training |
|---|---:|---:|---:|---:|
| HEAD | 0 | 0.101097988 | 0.106763878 | 0.108514529 |
| HEAD | 1 | 0.122254942 | 0.121087149 | 0.118846087 |
| HEAD | 2 | 0.120851737 | 0.117216413 | 0.114435252 |
| Candidate | 0 | 0.118997995 | 0.127261607 | 0.120295336 |
| Candidate | 1 | 0.109202215 | 0.128573728 | 0.148749888 |
| Candidate | 2 | 0.116417574 | 0.113682452 | 0.116928777 |

The candidate's after-training mean is **0.128658000**, outside HEAD's range
**0.108514529–0.118846087**. Candidate seeds 0 and 1 exceed that range;
seed 2 lies within it. The training mean is also outside HEAD's spread;
the before-training values lie within it. **No re-baseline.** The reviewed
baseline remains unchanged. The [complete comparison](reconstruction-comparison.json)
retains every phase, process result, and atom fingerprint. Concurrent-run
timings are recorded, not claimed as a controlled performance comparison.

The first corrected word-concept XOR gate receipt (`probes/word-xor-reset-after`)
completed both 400-epoch runs: class-0 accuracy **0.0** against **0.5**, and
input reconstruction **0/4** against **50%**. Both are red. A preceding attempt
hit duplicate reset admission before training completed; that failure remains
in `probes/word-xor-gates`. The final source-matched run repeats both results.

HEAD's unchanged two-epoch graph-release test **passed**, peaking at
**5.95 GiB** under the unchanged **8 GiB** guard (`head-memory/run`).
The candidate's final explicit run hit that guard at **8.67 GiB**
(`explicit-final/group-02`); the initial run's **8.19 GiB** and the previous
receipt's **8.47 GiB** failures remain visible. The guard is still 8 GiB;
the sampled peak can overshoot it before the process is stopped
([comparison](memory-comparison.json)). The depth-three campaign remains
historically red at `[1, 1, 1, 1]`; absence of the million-sentence checkpoint
is a skip, not a passing learning result. MM and both XOR gates remain explicit.

The unchanged numerical parity protocol now reports **exact equality** for
all four sentences' roots, references, recovered ideas and byte costs.
Packed and single mean costs are both **0.38411849636759143**, with identical
initial parameters and dictionary and no truncation
([comparison](parity/comparison.json)). The declared measurement seed remains
42; the native test uses three ambient initializations, without a seed choice.

## Two newly named failure causes

[The cause record](renamed-causes.json) distinguishes the current failures
from their old exceptions. The detached-reverse probe now fails before its
student check because sentence commit publishes detached history and the
returned root has no `grad_fn`. The normal-policy training probe indexes stale
one-row SymbolSpace word-reference metadata after provisioning into a larger
batch; the noncanonical path did not replace that metadata. Neither assertion
is waived or changed.

## Protection and provenance

[The audit](protected-audit.json) verifies the closed taxonomy, row schema,
XOR XML, protected MM/XOR/depth/parity assertions, and deferred NonLayer and
ConjunctionLayer implementations. No `test_item7_*` test sets a seed. All
81 local todo links resolve, with no numbered `README.md` damage.
The final documentation-link check passed all **119 cases**
(`probes/documentation-final`).
The [document-preservation record](claude-document-preservation.json) confirms
that seven of Claude's documents are byte-identical to the first round-2
probe's snapshot. `FutureWork.md` gained a concurrent September 29 learning
progression and corpus discussion during the sweep; that addition was read
and preserved. Codex edited none of these eight documents in this pass.
Todo edits in this pass are confined to item 7.

Changes were developed in [an isolated copy](working-copy.json) while the main
source was frozen for native compiler checks, then
[reconciled](reconciled-working-copy.json) without copying Claude's documents.
The full sweep uses the exact source and fixture manifests from the final
gates. The [source bridge](measurement-source-bridge.json) records the sole
test-only capture-helper difference from the preceding measurements: model
code, configurations and imported fixtures are identical. Its
[exact patch](parity-capture.patch), preimage, and assertion audit are retained.
The sweep uses 3 workers, 8 GiB each, 24 GiB aggregate,
1,800 seconds per worker, and 10,800 seconds overall. Slow native cases receive
fresh workers under those same caps. No learning threshold moves.

The original pool stopped at 8.28 GiB in the category-codebook case. Its
immutable `full-sweep/run` receipt remains red. The
[continuation driver](continue_sweep.py) resumes only unfinished coverage,
retains every completed outcome, and never retries a resource-limited case.
It uses the remainder of the original 10,800-second cumulative allowance,
with the same worker and memory caps and the same source. Interrupted peers'
partial reports are retained separately; their completed assertions are not
rerun. The other category-codebook case and the 64-word trace case also hit
the unchanged guard. These are additional resource failures, separate from
the explicit two-epoch graph-release failure above.

## Final source-matched receipt

All six reconstruction measurements and numerical parity are complete.
The [final explicit receipt](explicit-final/summary.json) has **8 passed,
2 failed, 1 skipped**, plus the graph-release case stopped by its memory guard.
The one-graph forward, both packed reverse checks, MM learning, all three
ambient native parity repetitions, and excluded-candidate value/gradient
check pass. The two XOR gates fail with the measurements above. The missing
mature checkpoint skips depth three; its historical campaign stays red.

The [full coverage receipt](full-sweep/summary.json) accounts for all
**5,026 cases exactly once**: 5,023 reached a pytest outcome and three stopped
at the memory guard. There are no missing cases or duplicate completed
outcomes. The original pool and three continuations used **9,287.4 seconds**
of the unchanged 10,800-second cumulative allowance. Completed failures and
resource failures were never retried; interrupted unfinished cases resumed.
The complete raw reports remain in `full-sweep/run` and
`full-sweep/continuation-01` through `continuation-03`.

Compared with the incoming receipt, **13 failures now pass**, eleven tests of
retired APIs were removed with the record above, and all nine new selectors
pass. **Six previous failures remain**, including both newly named §13 M
causes, confirmed in this run. **Sixteen previously passing selectors fail**:

| Group | Cases | Observed failure |
|---|---:|---|
| Grounded XOR | 6 | No assigned provisional case rows; expected four. |
| Concept-output curriculum | 3 | Eleven lesson case rows; expected four. |
| Shared word router | 2 | Word interpretation passes a missing target to `add_concept_edge`, which calls `int(None)`. |
| Attended-field word reuse | 1 | More than one concept returned for a word. |
| Concept-membership boundary | 1 | Inventory axis changes from eight rows to nine. |
| Optimizer checkpoint restore | 1 | Optimizer parameter-group layout differs. |
| Compile-cache recovery harness | 2 | Nested run reports incomplete coverage (one of two / zero of two). |

The two cache-harness reports were produced in peers later aborted by memory
stops. They remain failures, with that execution context recorded; they were
not independently rerun. The word-router exception is directly in the new
word-admission path. The other observations above are measured failures,
not claims that their underlying causes have been isolated.

Three additional previously passing cases hit the guard: the two category
codebook cases at **8.28 / 8.26 GiB**, and the 64-word STM trace case at
**8.34 GiB**. The test files containing these 25 failing/resource-limited cases
are unchanged by this pass (the ledger records the hash comparison per case).
Their assertions and the source were left intact after the sweep for review.

The validated source is
699 files, digest `0d42ea689e6f5021024e84c69be1e5b5dbb78515870665146413041147d15e46`,
matching the final gates exactly. [The round-2 file list](round2-changed-files.json)
distinguishes this pass from the incoming candidate. Nothing is committed.
