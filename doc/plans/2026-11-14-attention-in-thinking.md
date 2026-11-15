# Attention inside the thinking loop: the performance branch (Claude, 2026-11-14)

This is the plan for the branch `performance`, cut from GitHub main
`3926005f12`. It is built on the laptop while rounds A and B of item 1.8
are built on another machine, and it is merged after they land. It is the
first part of item 1.8's round C.

## 1. Alec's words

On the numeric XOR file, 2026-11-14: "If this is the replacement for
grammatical xor, then we should have "time to think" about the input,
defined as the field of awareness (.where on the input ) presented at one
time. For a sentence, that thinking looks like a serial thought loop over
words. For an arbitrary input, such as an image or xor, it looks like
multiple attentional opportunities in a loop. Going forward, would it be
possible to bring the loop into the thinking process? For some stimuli,
the examination might conclude without any looping at all."

On what it leads to: "I am optimistic that it would be a gateway to
speed-reading and multiple-sentence reads, so in the future we can provide
a page of text and it can decide where to spend its attention on the
corpus; it may not be a linear read of words."

On doing it now: "can we skip the performance measurement, and just have
the performance thread execute attention() as a part of the thinking
loop? If that can be done now before 1.8 and then merged, please write
the plan for that and push it to GitHub on the "performance" branch."

## 2. What exists at `3926005f12`

- *The examination loop runs whole, before any word is read.*
  `ModelAttention.stage_input` stages a closure, `model._attention_read`,
  which calls `Attention.narrow_words`. That function opens one bracket
  over the sentence (`BracketTable.open`) and takes `attentionBudget`
  steps (32). At each step `chooser.attend` picks a bracket and one of
  six acts: divide, descend, gloss, or a field act (intersection, union,
  not). It is one batched tensor loop with no loop over rows. After the
  last step it resolves the field acts' evidence over their scopes and
  returns a `NarrowedWords`.
- *The loop over words comes after.* In
  `Models._run_sentence_word_bricks` the read is taken once for a
  sentence (`greedy_attention`), and again for a path that departs in
  attention (`exploit`, `departure`). `ModelAttention.handoff` publishes
  it as `model._attention_words`, and then `word_step` runs for each word
  in order, reading that word's poles from the finished reading
  (`read_poles`).
- *A file with no vector operator* runs `Examination.run`, which takes
  the same read once and applies its field acts (`Examination.operated`).
- *Measured on the processor, one lesson sentence of two paths:* the
  whole read takes 0.05 of 2.11 seconds; a step of `word_step` about
  0.05 seconds; the generate walks 0.62.

So the reader decides everything it will look at before it has understood
any of it, and every step of the budget is taken whether or not anything
is left to decide.

## 3. What this branch builds

Three steps, in order. Each keeps a sentence read in order exactly as it
is today; what changes is where the steps are taken. Nothing here changes
what the model learns.

### 3.1 The walk as steps

`narrow_words` is split into three functions over one state (*default*
names): `open_narrowing` (what precedes the loop), `narrow_step` (one
pass of the loop's body) and `close_narrowing` (the resolution of the
field acts and the `NarrowedWords`). `narrow_words` remains as their
composition, so every caller is unchanged.

*Certificate P1.* On recorded calls from the lesson configuration
(`data/MM_corpus_lessons.xml`) and from both stopping gates' fixtures,
among them a replayed walk with a departure, the composition returns
every field of `NarrowedWords` exactly equal to the present function's.

### 3.2 The walk ends when nothing is left to decide

A step in which no bracket of any row is live changes nothing. The walk
stops there (the stop is read once for a step) and the outputs that have
one entry for each step are filled out to the budget, so the result is
the same.

*Certificate P2.* The same equality as P1 with the early end in force;
and, for the first sixteen lesson sentences, the number of steps taken
beside the budget. A sentence of known words should need far fewer than
32.

### 3.3 The loop over words takes the steps

In `_run_sentence_word_bricks` the walk is opened where the read is taken
today, and its steps are taken from inside the loop over words: before a
word is composed, the walk is advanced until that word's reading can no
longer change, and `word_step` reads it from the walk's state. A path
that departs in attention replays the kept walk up to its departure, as
now. `Examination.run` is not changed; it keeps calling `narrow_words`.

When a word's reading "can no longer change" is the thing to establish.
A word accepted as its own bracket takes no further act, but a field act
taken earlier over a scope that holds it is resolved from every word of
that scope, some of which may still be read (the resolution at the end of
`narrow_words`). Find the rule that is exact, and say what it is.

*Certificate P3, the same updates.* From one fixed seed, on the first
sixteen training sentences of the lesson configuration and on both
stopping gates: every chooser action, every registered cost term and
every parameter after each batch is exactly equal on this branch and on
`3926005f12`.

*Stop rule.* If no rule makes some case exactly equal, do not
approximate it: stop, and write the case for Alec with how often it
occurs in the lesson and what each choice would change.

*Reported, not gated.* Seconds for those sixteen sentences before and
after, and the count of walk steps taken before the first word is
composed.

## 4. What this branch does not build

These change what the model does, and they wait for the merge with item
1.8's rounds A and B, as the rest of round C:

- attention's choice reading what has been understood so far (the state
  of the words already composed, or a thought's goal);
- words read in another order than their order in the sentence, a
  stretch taken as one unit, or a word passed over;
- a presentation of several sentences as one field of awareness;
- an attention act as an operation of the thought episode
  (`Models._run_selected_thought_once`);
- `unfocus`, and the renaming of the acts to `focus` and `mint`, which
  are round A's.

## 5. Files, and the merge

- *Edited here:* `bin/Attention.py`, `bin/ModelAttention.py`,
  `bin/SentenceAttention.py`, `bin/AttentionTraversal.py`, and in
  `bin/Models.py` only `_run_sentence_word_bricks` and what it needs.
  New tests in new files.
- *Not edited here,* because rounds A and B rewrite them: the grammar
  files, `bin/Language.py`, `bin/Layers.py`, `bin/Memory.py`,
  `bin/Expectation.py`, `bin/ConceptSpread.py`, `bin/Intuition.py`,
  `bin/FieldProjection.py`, `bin/Examination.py`,
  `bin/NumericPercepts.py`, `bin/Interpret.py`, `bin/FamiliarTier.py`,
  `bin/MereologicalCodes.py`, and existing tests' assertions.
- *The merge.* When rounds A and B are on GitHub main, main is merged
  into `performance` here, the conflicts in `_run_sentence_word_bricks`
  are resolved, and P1 to P3 are run again against the merged main. The
  branch is then reviewed and merged into main.

## 6. Codex text (for Alec to paste on the laptop)

~~~
The performance round on this laptop has a new first task. In
~/github/basicmodel-perf: git fetch origin, switch to the branch
performance (it holds one commit on 3926005f12, the plan
doc/plans/2026-11-14-attention-in-thinking.md), and bring over any
commits of perf-1 that do not touch the files the plan edits. Build
sections 3.1 to 3.3 of the plan, in order. Its section 5 says which
files you may edit and which you may not; the earlier instruction to
leave the sentence loop alone is lifted for those files only. The MPS
measurement asked for earlier is withdrawn. The rest of the earlier
list (the sweep's overhead, the leaf kernels, corpus loading) follows
this task.

Rules: every certificate compares this branch with 3926005f12 from the
same seed, exactly; no seed is chosen so that an assertion passes; no
fixture is resized; no existing assertion is changed; nothing is kept
behind a switch. Find each cause yourself. Stop only for the case
section 3.3 names, or for a design question, written for Alec with your
recommendation. Commit on performance as you go; push performance when
Alec says so, and never push main.

Return as section 7 of the plan: each step with its certificate, failing
before and passing after; the rule you found for 3.3; the steps taken
beside the budget; the seconds before and after; every edit outside
section 5's list, with its reason; and the commit.
~~~

## 7. Laptop return (2026-11-14)

### Source, boundaries and evidence

The checkout is `/Users/arogers/github/basicmodel-perf`, on `performance`.
The reference is exactly `3926005f12cabfd8233fc444963eae96c994d7ba`.
The plan arrived as `2d412e80fd`. The existing `perf-1` commit does not
touch any of §5's attention edit set; it was carried unchanged as
`28828fc59e` (original `86192877d5`). No push has been made.

That carried commit accounts for these edits outside §5: `Makefile`,
`test/bounded_tests.py`, `test/conftest.py`, and the added test in
`test/test_bounded_test_runner.py` reduce and measure harness overhead;
`bin/Layers.py` and `bin/Spaces.py`
contain the already-certified retained-leaf repairs; `bin/prepare_data.py`
and `bin/prepare_corpus.py` remove redundant preparation passes, with added
cases in `test/test_prepare_data.py`. No existing assertion was changed.
`doc/plans/2026-11-14-performance-round.md`, `test/perf_observer.py`,
`test/performance_leaves.py`, `test/test_performance_leaves.py`, and
`test/tools/performance_round.py` retain that round's report and evidence
tools. The attention task makes no further edits to those files. This
plan's §7 is edited as requested. New attention tests and its observer
are new files under `test`, as permitted. The withdrawn MPS measurement
is not rerun.

Local evidence is in ignored `output/attention-thinking/`. The new
`test/attention_thinking_certificate.py` observes the actual source
checkout: the first sixteen training sentences, native XML capacities,
normal losses and optimizer. It records the bytes of every named
parameter and optimizer tensor after each batch using SHA-256, every
registered `Error.add`/`Error.error` contribution, all attention fields,
and the other operation chooser results. Python, NumPy and Torch streams
are recorded and replayed without choosing a successful initialization;
the same recorded hash setting is used in both processes. The original
attention source is loaded from the specified Git object as a test
oracle, never as a production alternative. All comparisons are exact
(`rtol=0, atol=0` for direct tensor comparisons).

An observer initially failed while viewing a singleton strided tensor
as bytes. Its flat-copy serialization was corrected and the run restarted
from the **same already recorded initial stream**, not a new seed. That
interrupted observation remains in `base-lesson-observer-error/`.
Its recorded `initial-rng.pt` SHA-256 is
`b4f4d373fb74a4c57d8c6f8d16338a4c319bc1edac8a668527f8ea98dadd4745`.

### 3.1 — the walk as steps; P1

`open_narrowing`, `narrow_step`, and `close_narrowing` now share a
`NarrowingState`; `narrow_words` is their composition. The original
arithmetic, ordering, field resolution and returned fields are retained.

Before the extraction, the five new interface/equality cases failed on
the absent step API (`p1-before.log`). Afterwards all five pass, including
departure replay and the empty carrier (`p1-after.log`). The existing
fullgraph assertion was run unchanged with `RUN_SLOW=1`: one pass in
9.19 seconds (`p1-compile.log`).

The real lesson certificate records **35 calls, including three attention
departures**, over sixteen batches. Every `NarrowedWords` field, every
observed chooser result, all registered costs, every parameter after each
batch and the random streams match the original exactly
(`base-lesson/`, `p1-lesson/`). Both unchanged stopping gates pass and
match exactly (`base-gates/`, `p1-gates/`), including every numeric-XOR
optimizer update and the shared-class fixture's final parameters. These
two fixtures make **zero** `narrow_words` calls; their walk-field
comparison is therefore empty, not an additional word-walk witness.

| Observed work, CPU / one thread | Base seconds | After 3.1 seconds |
| --- | ---: | ---: |
| Sixteen lesson `runBatch` calls | 73.046459 | 72.701923 |
| Lesson driver, including observation/reference checks | 74.367262 | 75.586065 |
| Both stopping gates, including observation | 1.042574 | 1.003382 |

The per-batch timings exclude the extra reference-oracle comparison;
they still include the passive action/cost observation. These are one
paired run, not a claimed speedup. The base consumes all 32 rounds on
every one of the 35 recorded walks. The next step investigates its
repeated stop actions after the legal menu is exhausted.

The step extraction and P1 certificate are committed as `1799c4a260`.

### 3.2 — an inert tail ends the walk; P2

The stopping condition is that **no row has any legal act**, not merely
that no table slot remains marked `live`. The recorded lesson words
include contradictory/unsupported residual brackets: they can remain
live after all available field acts have been used. Their unchanged
inputs, unchanged remaining budget and empty menu make every subsequent
step the same stop. The first recorded walk, for example, spends twelve
actions and then returns action 192 twenty times. This is the cause of
the original full-budget work even on known forms.

The final implementation evaluates the first inert round, then repeats
its exact stop/-1, probability, count and scope entries to the budget.
It does not repeat the chooser or the field body. On a departure, the
original selector still draws one uniform per inert round; the omitted
rounds' draws are retained in order. Otherwise later choices and updates
would change even though the immediate walk result looked equal.

The two new exhausted-walk cases fail before the change (32 chooser
calls) and pass afterwards, including all returned fields and the RNG
state. Seven eager cases pass in 3.94 seconds. The full recorded lesson
and both stopping gates again compare exactly with the base, including
every batch's parameters, costs, chooser results and random streams
(`p2-lesson/`, `p2-gates/`).

The original fullgraph assertion also remains unchanged. A host break
first failed its capture, so captured callers use tensor conditionals
which skip the same exhausted body. This exposed two local PyTorch
2.15 AOT issues: mixed Boolean/integer outputs were passed as floating
tangents, and an unused stop-score parameter acquired a zero gradient
instead of `None`. The conditional carries discrete **bits**, without
converting integer addresses numerically, and detaches the already
unused stop-score input before the conditional. No framework file is
patched. The existing fullgraph assertion and a new exact all-field /
policy-gradient case with addresses above `2**33` pass (two cases in
47.51 seconds, `p2-compile-final.log`). This is compiler lowering, not a
user-selectable legacy mode; eager and captured execution both stop
chooser work at exhaustion.

| Lesson sentence | Each walk's evaluated rounds / budget |
| --- | --- |
| This is Felix. | 13/32, 13/32 |
| What is this? | 14/32, 14/32, 14/32 (departure) |
| That is Ciao. | 13/32, 13/32 |
| What is that? | 11/32, 11/32, 11/32 (departure) |
| Here is Felix. | 13/32, 13/32 |
| Who is here? | 11/32, 11/32 |
| There is Ciao. | 13/32, 13/32 |
| Who is there? | 11/32, 11/32 |
| That is Felix. | 13/32, 13/32 |
| What is that? | 11/32, 11/32, 11/32 (departure) |
| This is Ciao. | 13/32, 13/32 |
| What is this? | 11/32, 11/32 |
| Here is Ciao. | 13/32, 13/32 |
| Who is here? | 11/32, 11/32 |
| There is Felix. | 13/32, 13/32 |
| Who is there? | 13/32, 13/32 |

The first walk in each row is the existing input-staging read; the
second is the sentence-local read. The optional third is its departure.
The count includes the first inert round establishing the repeated tail.
Before, every entry was 32/32. The first sixteen lesson `runBatch` calls
take **73.046459 seconds before → 71.470917 seconds after** P2, with the
same observation contract and capacities as P1. This is one paired run.

The exhausted-tail change and P2 certificate are committed as `73eef9644a`.

### 3.3 — the word owner advances the walk; P3

The exact stability rule is a dependency test. Start with the requested
word positions. Traverse the already recorded field scopes **backwards**:
if a scope intersects the needed positions, include every word in that
scope. Then include equal, nonnegative identities that may supply those
words' shared descent witness. The reading is stable when none of those
positions is covered by a valid, unfinished bracket with remaining
budget, or when the whole legal menu or the total budget is exhausted.
The rule is conservative: an unsupported but still-live bracket can make
it wait longer than necessary. It never guesses a future chooser action.
Later brackets only split existing extents; they cannot reach outside
the unfinished source brackets. Shared descent is applied before field
resolution, which is why identity dependencies are expanded after the
backward field traversal.

Three new cases fail on the base's absent step/stability API and pass
afterwards (`p3-before.log`, `p3-rule.log`): an independent accepted word
can settle before another word; an earlier field operation must wait
for a later word's descent; and a later occurrence of the same identity
can change a word already accepted. Partial handoffs and the eventual
complete result are compared directly with the base at zero tolerance.

`ModelAttention` stages an opening closure beside the existing read.
`_run_sentence_word_bricks` opens the sentence walk at the former read
site, and advances it from the cached-word loop. Attention departures
replay the original trace and retain nondeparting rows exactly. The
original input-staging read stays in place because expectation consumers
need it before sentence composition. `Examination.run` is unchanged.

There is an additional dependency at the first word: sentence restoration
publishes the **whole** pole slab through `_pushed_word_slab` and
`commit_word_reference_slab`; `SentenceField` also builds its candidate
evidence from all words. Therefore all those sources must settle before
restoration and the first composition. In the sixteen measured sentences,
this requires completing each sentence-local walk on the first word-loop
entry. There are **19 word-driven walks** (16 greedy and three departures),
**233 evaluated rounds before the first composed word**, and **zero
additional attention rounds between the 49 word compositions**. The
first-word counts are the second and optional third entries of the P2
table: 11–14, against the original budget of 32 each. Input staging takes
another 197 rounds, giving **430 evaluated rounds versus 1,120** across
all 35 walks. The result does not yet make attention depend on what a
previous word has taught the model; that remains §4's later work.

The placement certificate fails on `3926005f12` at its first batch with
`sentence attention was completed outside the word loop`
(`p3-placement-before.log`). On this branch it observes the actual
`compose` loop index and the call from `start_compose`, and passes on all
sixteen batches (`p3-final-lesson/`). Every batch's attention outputs,
all chooser results, registered cost contributions, optimizer state,
parameters and random streams equal the base exactly. Both unchanged
stopping gates also pass and compare exactly (`p3-final-gates/`).

The new step and handoff cases plus the existing fullgraph assertion
pass (12 cases, 48.89 seconds, `final-step-certificates.log`). A further
empty-carrier case verifies that a supplied `spent` is ignored exactly
as in the original function (two parameterizations pass in
`empty-final.log`). The captured comparison now checks `requires_grad`
as well as tensor values and exact policy gradients; discrete outputs
retain their original detached contract.

The final observer also snapshots each accumulated cost record after
registration, including its row mask, owner/side, weight, numerators and
baselines, rather than checking only the supplied arguments. A fresh
paired replay from the **same original recorded stream** passes for all
901 registrations, all sixteen parameter/optimizer snapshots and all
35 walks (`base-complete-lesson/`, `p3-complete-lesson/`). The two stopping
gates again pass exactly (`base-complete-gates/`, `p3-complete-gates/`);
those fixtures register no `Error.add`/`Error.error` terms and take no
attention walks, so these portions of their comparisons are empty.

| Observed work, CPU / one thread | Base seconds | After 3.3 seconds |
| --- | ---: | ---: |
| Sixteen lesson `runBatch` calls, initial final replay | 73.046459 | 71.163715 |
| Sixteen calls, final accumulated-cost observation | 71.852380 | 70.905987 |
| Final lesson driver, including observation/reference checks | 73.167360 | 73.767570 |
| Both stopping gates, final observation | 1.000958 | 1.001150 |

The final run is about **0.23 sentences/second**, or 4.43 seconds per
sentence. The initial final replay is retained as separate evidence;
timing rows with different observation contracts are not mixed. These
small timing differences do not establish a throughput improvement.

The attention/traversal/native-field/whole-path regression selection
finishes with **87 passes and one failure** in 93.48 seconds
(`regressions.log`). The failure is the unchanged
`test_where_attention_handoff.py::test_mereology_raise_enables_conceptual_refinement`.
It compares two consecutive raw forwards after freezing admission, but
`BasicModel.forward` still calls `_advance_when_time` on each numeric
presentation. The clock is part of the input evidence. A same-stream
replay on the base and branch reproduces the failure with identical
forward outputs and parameters in all three calls; the two compared
outputs differ by at most `8.940696716308594e-08` on **both** versions.
Only the clock buffers change between those calls. A diagnostic holding
the last call's clock at the previous time makes the assertion pass
exactly on both versions (`mereology-{base,after}[-clock].{json,pt,log}`).
That diagnostic does not change the test or production clock. Unused,
uninitialized priming scratch buffers are not equality witnesses. No
attention regression was found, and the unchanged assertion is not
reported as passing.

An AST scope audit confirms that `_run_sentence_word_bricks` is the only
changed `Models.py` method. The existing bodies in both extended carried
test files remain identical to the base. No fixture or configured
capacity was resized for these checks, and no sentence loop was compiled.

The word-loop change and P3 evidence are committed as `b5b92126bc`.

### Carried performance work, checked after P3

The carried changes are unchanged from `28828fc59e`. The following CPU
refresh runs **sequentially**, with no other benchmark running, against
the original source and saved inputs. MPS is not rerun. The historical
complete default sweep and the original native-corpus measurements
remain in [the performance report](2026-11-14-performance-round.md);
no new complete sweep is claimed for the attention implementation.

The matched default worker replays all 256 original selectors, their
recorded initial Python/NumPy/Torch streams and hash setting, the six
one-thread limits, and the same 8 GiB cap. Both versions produce 255
passes and one skip. Initial **and final** random streams and each test's
outcome match exactly (`harness-audit.json`).

| Matched worker | Base seconds | Final branch seconds |
| --- | ---: | ---: |
| Process wall | 80.497796 | 55.567156 |
| Test bodies | 23.601552 | 22.798686 |
| Outside test bodies | 56.896244 | 32.768470 |
| Setup | 23.987195 | 0.118955 |
| Teardown | 24.859500 | 24.940574 |
| GC (overlaps phases) | 49.517987 | 25.666983 |

Outside-body time falls by 42.4%; peak process-tree memory is 2.723 → 2.699 GiB. The removed duplicate GC scan accounts for the setup reduction. The training fixtures remain unchanged. The original complete-sweep measurements separately identify model/configuration/training setup; their results are not extrapolated from this worker.

All **31 CPU leaf variants** again compare eager values and gradients
with the `3926005f12` fixed-input reference at **zero tolerance**. Each
produces one full graph across positive batches `3, 1, 2, 4, 8, 17`,
zero recompilations, and no checked `nonzero`, `item`, or
`_local_scalar_dense` operation in forward or backward. The final
`TORCH_LOGS=graph_breaks,recompiles` log contains zero such events.
Compiled-versus-eager floating arithmetic retains the earlier explicit
`rtol=2e-5, atol=2e-6` contract; integer indices are exact. This is not a
claim of bitwise floating equality between Inductor and eager. Source
before/after equality is exact. The original nonempty folds captured
but executed synchronizing `nonzero`; the retained repair removes it.
Zero-sized batches are outside the positive-batch claim, as documented
in the carried report. No production compilation default is changed.

Warm forward medians are **microseconds** below; startup is **seconds**
for capture and the forward/gradient comparisons. Both retain the shared
compiler disk cache; startup differences are not compiler speedup claims.
The complete fixed shapes, staging and profiler contract are unchanged
from the carried report.

| CPU leaf | Eager µs, base → final | Compiled µs, base → final | Startup s, base → final |
| --- | ---: | ---: | ---: |
| `householder` | 121.13 → 120.25 | 41.27 → 40.67 | 1.107 → 1.713 |
| `householder_action` | 97.07 → 95.67 | 35.30 → 35.23 | 0.123 → 0.139 |
| `householder_action_transpose` | 96.42 → 95.55 | 35.50 → 35.30 | 0.112 → 0.129 |
| `lane_neutral_forward` | 104.17 → 103.51 | 47.70 → 43.33 | 0.537 → 0.593 |
| `lane_neutral_inverse` | 104.53 → 104.15 | 48.71 → 48.43 | 0.351 → 0.362 |
| `lane_learned_forward` | 104.56 → 103.04 | 49.06 → 48.14 | 0.329 → 0.328 |
| `lane_learned_inverse` | 104.15 → 105.24 | 48.48 → 48.10 | 0.332 → 0.332 |
| `cube_chart_True` | 20.96 → 21.22 | 23.89 → 23.65 | 0.114 → 0.123 |
| `cube_chart_False` | 17.97 → 17.80 | 23.74 → 23.12 | 0.043 → 0.052 |
| `cube_unchart` | 16.07 → 15.97 | 22.86 → 22.53 | 0.042 → 0.057 |
| `meet_ratios` | 25.05 → 24.19 | 28.68 → 28.04 | 0.052 → 0.068 |
| `fold_False_False_False` | 88.40 → 72.10 | 53.64 → 35.04 | 2.487 → 0.157 |
| `fold_False_False_True` | 87.31 → 81.25 | 52.82 → 36.16 | 2.225 → 0.333 |
| `fold_own_False_False` | 105.93 → 98.69 | 64.28 → 43.01 | 2.677 → 0.403 |
| `fold_False_True_False` | 113.82 → 97.02 | 54.73 → 35.85 | 2.322 → 0.141 |
| `fold_False_True_True` | 112.27 → 106.24 | 54.40 → 37.49 | 2.148 → 0.341 |
| `fold_own_False_True` | 129.68 → 122.83 | 64.63 → 43.37 | 2.582 → 0.433 |
| `fold_True_False_False` | 110.33 → 94.39 | 55.41 → 36.27 | 0.911 → 0.116 |
| `fold_True_False_True` | 109.74 → 103.60 | 53.87 → 37.39 | 0.967 → 0.315 |
| `fold_own_True_False` | 127.94 → 121.30 | 64.90 → 43.84 | 1.157 → 0.388 |
| `fold_True_True_False` | 89.82 → 74.20 | 53.15 → 35.31 | 0.765 → 0.111 |
| `fold_True_True_True` | 89.38 → 83.48 | 53.24 → 36.73 | 0.807 → 0.292 |
| `fold_own_True_True` | 107.42 → 101.37 | 66.14 → 43.60 | 0.989 → 0.363 |
| `fold_empty` | 13.95 → 6.01 | 15.41 → 14.29 | 0.083 → 0.040 |
| `fold_unsorted` | 81.69 → 76.79 | 49.31 → 36.90 | 1.889 → 0.295 |
| `where_encode_2` | 7.96 → 7.87 | 14.80 → 14.62 | 0.140 → 0.142 |
| `where_decode_2` | 16.00 → 16.60 | 22.10 → 22.39 | 0.041 → 0.057 |
| `where_decode_index_2` | 18.43 → 19.35 | 14.14 → 13.94 | 0.034 → 0.043 |
| `where_encode_4` | 13.70 → 13.63 | 15.14 → 15.22 | 0.032 → 0.040 |
| `where_decode_4` | 29.67 → 30.19 | 22.41 → 22.23 | 0.047 → 0.063 |
| `where_decode_index_4` | 25.39 → 25.49 | 14.12 → 13.87 | 0.037 → 0.046 |

Corpus refresh uses the same pinned 94,511,680-byte source and 1,000
complete documents: **39,289 sentences**, with identical shard bytes,
row ordering, loaded splits and source addresses. The measurements use
five repetitions (three for assembly) and local transport for the plain
file installation. They are source/disk measurements, not network speed.

| Corpus operation | Base seconds | Final branch seconds |
| --- | ---: | ---: |
| Verify prepared data | 0.098110 | 0.101618 |
| Authenticate pinned source | 0.082722 | 0.041230 |
| Install plain local file | 0.141533 | 0.097508 |
| Assemble complete documents | 0.977438 | 0.936066 |
| Read/validate rows | 0.089705 | 0.089486 |
| Load splits and addresses | 0.154886 | 0.157139 |

Evidence: `harness-{before,after}/`, their guard receipts and logs,
`leaves-final/` and `corpus-final/`. The leaf and corpus baseline evidence
is the original `output/perf-1/before/{leaves-cpu,corpus}-controlled/`.
Unchanged paths serve as timing controls; small variations in them are
not attributed to the optimization.

The original native **math-chain** measurement is also replayed on the
final implementation, from its original recorded stream and hash setting.
All **60 sentences in nine batches** complete. Every recorded loss,
breakdown and answer, all final model tensors and optimizer state, and
the final Python/NumPy/Torch streams match `3926005f12` exactly
(`math-final/`, compared with `output/perf-1/before/math-controlled/`).
This retains the original single grade-00 presentation, batch eight and
native tails; it does not claim the full multi-epoch curriculum.

| Native math-chain measurement | Base seconds | Final branch seconds |
| --- | ---: | ---: |
| Model construction | 0.294618 | 0.293486 |
| All 60 completed `runBatch` sentences | 172.696435 | 172.333245 |
| Per completed sentence | 2.878274 | 2.872221 |
| Presentation wall | 173.157195 | 172.794142 |

That is about **0.35 sentences/second**, effectively unchanged. Its
`TORCH_LOGS=graph_breaks,recompiles` log has zero events because this
sentence owner is eager; this is not a full-model compile claim.

### Local commits and final state

| Work | Commit on `performance` |
| --- | --- |
| Carried performance round | `28828fc59e` |
| P1: walk extraction | `1799c4a260` |
| P2: exact inert-tail ending | `73eef9644a` |
| P3: word-loop owner and exact stability rule | `b5b92126bc` |

The final refresh changes documentation only. `certified-source.json`
records the implementation commit and source digests used by the final
certificates. The temporary base worktree was removed after its evidence
was saved. All work remains local on `performance`; nothing was pushed,
and `main` was not changed. The additional learned-state dependence and
nonlinear reading remain the explicitly deferred work in §4.

## 9. Sole sentence walk: equality investigation (2026-11-14)

**Stopped at the requested equality gate.** No production implementation
of the sole-walk change was made. The sixteen separate lesson walks are
equal, but moving expectation changes its tensors on twelve sentences.
The packed fixture also has different admission, poles and expenditure.
These are the cases Alec asked to receive before a choice is made.

### Source and certificate

This investigation starts at `a0ddd14125` on `performance`. The earlier
§7 work was pushed on Alec's subsequent instruction; its historical
"nothing was pushed" statements describe the earlier report time. This
follow-up is local and has not been pushed. `main` is unchanged.

The new observer is `test/attention_single_walk_certificate.py`, commit
`6efaf8283f`. It is a new test file permitted by §5. The only other edit
is this requested §9; no production file, existing assertion, fixture
capacity, or runtime option is changed. There is no alternative production
implementation behind a switch.

All runs replay §7's original Python, NumPy and Torch streams and hash
setting, without selecting another seed. The initial stream file's SHA-256
remains `b4f4d373fb74a4c57d8c6f8d16338a4c319bc1edac8a668527f8ea98dadd4745`.
The environment remains CPU/eager, one thread, on this M2 laptop. No MPS
measurement was made.

The observer checks each actual walk against the original attention
function at `3926005f12cabfd8233fc444963eae96c994d7ba`. The sixteen lesson
batches and both unchanged stopping gates also match the saved base
certificates exactly: chooser results, registered cost contributions,
parameters, optimizer state where present, and random streams. The packed
fixture was additionally run in a fresh detached checkout of that exact
base; its complete batch record matches this branch exactly. Both gates
pass, but make **zero attention walks and zero staged word expectations**;
they do not witness the proposed ownership change.

Expectation probes call the actual `handoff` and `stage_expectation` on
attribute overlays and a private copy of the expectation owner's bank
buffers. They consume no random draws. They capture all four word-input
tensors, all ten `ExpectedWords` fields, masks, content, handoff poles,
and the four shared-symbol-bank buffers. The isolated staging probe first
has to reproduce the real staging result exactly. Comparisons use the
current sentence's slots; candidate and symbol banks are compared whole.
Complete walk tensors are retained as well as the difference counts.

Evidence is in ignored `output/attention-single-walk/`:
`lesson-complete/`, `gates-complete/`, `base-packed/`, and `packed-prefix/`.
Each has its batch records and timing result; the branch runs also have
walk comparisons and expectation tensor snapshots. `receipt.json` records
source, observer digest, environment and totals. The initial census and
the earlier packed comparison remain alongside these runs.

### Sixteen separate lesson sentences

For **16/16**, staging and the actual greedy sentence walk have exactly
equal `accepted`, `descended`, `poles` and `spent`. Substituting the sentence
reading while holding the expectation call at the same point changes
none of its outputs or banks. The separate oracle calls at staging time
give the same result.

Moving that call to `start_compose` is different. On **12/16** sentences,
`probabilities`, `prediction`, `excess`, `omission`, `loss` and `activation`
change. The unchanged cases, numbered from one, are 1, 5, 9 and 13.
All other captured fields, including codes, banks, targets, content,
gain, top-one result and rank, remain equal on these sixteen inputs.
Maximum absolute differences are **0.000041604042** in probabilities,
**0.000040560961** in prediction, and **0.000007286668** in staged loss.
These small differences fail exact equality.

The cause is the clock. `_run_sentence_word_bricks` advances it before
composition; `Layers._prediction_inputs` converts temporal coordinates
relative to the current model time. For example, sentence 2 stages at
tick 3 and reaches `start_compose` at tick 8. A diagnostic that supplies
only the earlier clock restores **every captured field exactly on all
sixteen**. That diagnostic is not installed in production.

This is a counterexample for the requested staging tensors, not proof
that this particular clock shift changes an optimizer update: the current
sentence closing code recomputes the word loss from
`_word_expectation_input`. No changed production training path was run.

### One row containing the complete first lesson document

The additional fixture packs the existing document's four sentences into
one row: "This is Felix.", "What is this?", "That is Ciao.", and
"What is that?". It uses the original XML word capacity and allowance
(32), all the original training-form admission, and a normal inference
`runBatch`. No document is shortened and no capacity is adjusted.

The staging walk spends **27**, accepts **10** words, and descends on
four punctuation occurrences. The actual sentence walks do this:

| Sentence | Meter entering walk | Walk's final spent | Evaluated rounds | Accepted, staging → sentence | Local field operations |
| --- | ---: | ---: | ---: | ---: | ---: |
| This is Felix. | 0 | 12 | 13 | 2 → 2 | 6 |
| What is this? | 13 | 25 | 14 | 3 → 3 | 5 |
| That is Ciao. | 26 | 32 | 8 | 2 → 2 | 0 |
| What is that? | 32 | 32 | 1 | 3 → 0 | 0 |

`spent` is cumulative; it includes the entering allowance consumption,
not just this walk's work. The selected sentence paths also spend query
work between walks. The sentence loop already shares that meter across
the row: it does **not** grant each sentence a fresh allowance of 32.

On the corresponding sentence slots, `accepted`, `descended` and `poles`
match for the first **3/4** sentences. On the fourth, the accepted bits
for `is`, `that`, and `?` all disappear, the `?` descent bit disappears,
and its positive pole changes from 1 to 0. Thus **3/10** staged accepted
words are absent from the sentence walks. Cumulative `spent` differs
from the staging total on **4/4** sentence comparisons.

There is also a scope/budget difference before any sentence changes
model state. The row walk takes **zero field operations**. Attention's
`minimum` calculation reserves progress for all pending words in its
scope, suppressing optional field operations. Diagnostic sentence-only
walks opened at that same staging point with the same initial meter
instead spend **12, 12, 12, 13**, taking **6, 5, 6, 6** field operations.
Their accepted and descended bits still match on all four sentence
slices, but the fourth changes both poles on `that` and `?`: `(1, 0)`
becomes `(0, 1)`. Its local `not` field operation covers those two words.
Resetting the allowance per sentence would therefore also change the
reading; it is not an equality repair. Those four walks alone would
spend 49 units rather than the row walk's 27.

Expectations differ on **3/4** packed sentences even when both readings
are evaluated at the same time. The word candidate bank remains the
same ten identities from all live input words; acceptance controls
targets and arrival, not which identities enter that bank. Calling the
existing routine with only a sentence's accepted mask drops the earlier
sentences from its causal prefix. `WordExpectation.word_distribution`
updates its running context and activation from that prefix. At the
staging point, restoring only the earlier accepted words and their
poles makes expectations for sentences 2 and 3 exactly equal again.
Sentence 4 still differs because of its local pole changes.

At actual `start_compose`, the fourth sentence's targets change from
`[1, 9, 6]` to `[-1, -1, -1]`; those three content/gain/loss entries become
zero. Merely restoring the old clock does not repair the packed cases.
Earlier sentences have also changed history and the native symbol bank:
bank keys differ at sentences 2–4, with identities, addresses and orders
also differing at sentence 4. Freezing the old clock, word codes and
initial history still leaves the prefix/admission differences. These
controls separate the clock, prefix, scope and live-state dependencies.

One correction to the task's description matters for a later edit:
`stage_input` already skips charging the staging walk when
`_sentence_ends` is true. Sentence scoring charges private trial meters;
the selected path transfers its charges to the real meter. That existing
ownership is retained in this investigation.

### Walks, evaluated rounds and seconds

The sixteen sentences currently execute **35 walks / 430 evaluated
rounds**: 16 staging walks / 197 rounds, 16 greedy sentence walks / 197
rounds, and 3 attention-departure replays / 36 rounds. Each staging and
greedy walk's rounds, in lesson order, are
`13, 14, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 11, 13, 13`.
The three departure walks occur on sentences 2, 4 and 10. They are actual
extra walks, not omitted from a "one per sentence" count.

| Fixture | `3926005f12` walks / rounds | Current branch walks / rounds | After sole-walk change |
| --- | ---: | ---: | --- |
| Sixteen lesson sentences | 35 / 1120 | 35 / 430 | Not built: equality gate failed |
| Packed four-sentence row | 5 / 160 | 5 / 64 | Not built: equality gate failed |
| Both stopping gates | 0 / 0 | 0 / 0 | Not built |

The difference from the base in evaluated rounds is the already-certified
§7 inert-tail removal. This follow-up changes neither count. Subtracting
the sixteen staging walks would arithmetically leave **19 / 233**, if
all other decisions survived. That is a projection, not a certificate or
an implemented count; three departure replays would still remain.

| Measurement, seconds | Exact base | §7 branch before this task | Current unchanged production | After sole-walk change |
| --- | ---: | ---: | ---: | --- |
| Sixteen lesson `runBatch` calls | 71.852380 | 70.905987 | 70.732232 | Not measured |
| Both gate drivers, including observation | 1.000958 | 1.001150 | 1.009343 | Not measured |
| Packed document `runBatch` | 5.838789 | — | 5.781457 | Not measured |

The first two reference columns use §7's complete-registry observation;
the packed base is the fresh run described above. `runBatch` timings
subtract timed oracle/probe work but retain ordinary observation overhead.
The current sixteen-sentence driver including all diagnostics takes
79.918264 seconds; the packed driver takes 7.665632 seconds. These are
single observations, not a speedup claim. Current lesson throughput is
approximately **0.23 sentences/second**. There is no sole-walk after-time
because implementing it would choose between the unequal results.

### Decision for Alec

| Choice | What it preserves and changes |
| --- | --- |
| Preserve exact base behavior | Keep the row-wide admission, field decisions, staging-time forecast context and cross-sentence prefix. Retaining the current row walk preserves those results; replacing it requires a design that derives that same row-wide result. Simply substituting the existing sentence walks fails the certificates above. |
| Make each sentence's existing walk authoritative | Retain the shared row/query allowance; stage from that sentence's actual admission. The packed final sentence loses its three targets, and later-time forecasts change on 12/16 separate lesson sentences. Carry the causal prefix explicitly if earlier accepted words are to remain visible. This is a semantic change requiring a revised equality contract. |
| Give every sentence a fresh allowance | The cold comparisons recover all ten accepted words, but spend 49 bracket units and change the final sentence's poles and expectation. This changes the row budget as well as the reading. |

**Recommendation:** keep the shared row/query allowance and preserve the
causal prefix; make the sentence walk authoritative only if Alec explicitly
accepts the admission/forecast changes above. Do not reset the budget to
make the packed case pass. If equality with `3926005f12` remains mandatory,
retain the current production behavior pending a row-semantics design.
No choice in this table has been implemented.

The diagnostic commit is `6efaf8283f`; the following documentation-only
commit records this return. Both remain local on `performance`.

## 10. Accelerator sentence block: sentence ownership and host work (2026-11-14)

### Decision and certificate boundary

The initial checkpoint (`7a1645e4e6`) stopped on the §9 differences. Alec
then authorized making the semantic change: "in general Id expect this
to change things", and asked to make it before recovering performance;
better quality benchmarks will need a more highly trained model.
The implementation below therefore makes the sentence walk authoritative
with the existing shared row/query allowance and the earlier kept-word
prefix. It does not give each sentence a fresh allowance.

The old exact result at `3926005f12` remains a reported comparison, not a
claim of equality for this authorized change. Later host optimizations
must be exactly equal to the sentence-owned implementation. Every replay
uses §7's original ambient Python/NumPy/Torch stream and hash seed; no
seed is selected again, no fixture is resized, no existing assertion is
changed, and no production switch is introduced. Measurements are CPU
measurements; the withdrawn MPS measurement is not reinstated.

### Item 1: the sentence owns admission and expectation

`ModelAttention.stage_input` retains spans, keys, native poles, prior and
opening closure, but takes no walk when the sentence loop runs. Disabled
examination/vector configurations expose no sentence opening. Non-sentence
configurations retain their original one staged walk and charge.

At `start_compose`, the sentence walk finishes, publishes its leaf mask,
and charges a private path meter. The first path stages the word forecast
once; all explore paths share that forecast and its graph. Earlier kept
accepted words and poles remain the causal prefix, and the candidate bank
retains the native row's vocabulary. The selected path transfers its work
to the shared meter once. Before the first sentence there are no admitted
leaves from other sentences; this replaces the implicit mask formerly
supplied by staging. No other sentence-loop behavior is intentionally
changed.

The native sixteen-sentence replay now takes **19 walks / 233 evaluated
rounds**, down from **35 / 430**: sixteen primary sentence walks and three
actual attention-departure replays remain. There are exactly sixteen word
forecasts, one per sentence. All paths consume their sentence's identical
forecast object without modifying its loss. The packed four-sentence row
takes **4 walks / 36 rounds** instead of **5 / 64**, with four forecasts.
Both gates still take zero attention walks.

Every individual attention invocation, including departures, is exactly
equal to the `3926005f12` kernel for its actual inputs and RNG state. The
full lesson is intentionally unequal: parameter and optimizer hashes
change in all sixteen batches; registered costs change in fifteen batches;
non-attention chooser traces change in twelve. The RNG stream remains
exactly equal at every batch boundary. The packed inference replay changes
attention, costs and chooser traces; its parameters and RNG remain equal.
Both original stopping gates pass and all 51 gate records remain exactly
equal to `3926005f12`.

The new ownership tests also cover no walk during preparation, retained
non-sentence behavior, disabled examination, and the kept-prefix bank and
pole handoff. The relevant unchanged attention, compose and expectation
suite passes: **41 passed, 1 skipped**. The certificate additionally checks
one forecast per sentence, identity across paths and bounded meters. On
the unchanged `7a1645e4e6` source, the new ownership tests fail in the
three expected places (staging walk, disabled opening, prefix argument);
the retained non-sentence behavior already passes. All four pass after.

Evidence: `test/attention_block_certificate.py`,
`test/test_sentence_owned_attention.py`, and ignored
`output/attention-block/owned-v3-{lesson,packed,gates}`. The initial failed
integration replay is retained in `owned-v2-lesson.log`: it exposed the
missing initial leaf mask, which is fixed rather than bypassed.

| Same native driver, seconds | Before ownership (§9) | Sentence owned |
| --- | ---: | ---: |
| Sixteen lesson `runBatch` calls | 70.732232 | 69.342439 |
| Packed document `runBatch` | 5.781457 | 5.768452 |

These are single observations with the same oracle/registry observation
method, not a statistically established speedup. Quality after further
training is not established by these sixteen early training sentences.
A full accelerator sentence graph is not yet certified; the following
inventory identifies the remaining host boundaries.

### Items 2–3: inventory and independent exact optimizations

The unchanged-source census is complete: [inventory by phase and function](2026-11-14-attention-host-inventory.md), with a [complete source-site ledger](2026-11-14-attention-host-inventory.json). It records 655 phase/host-API sites and 1,836 phase/loop sites across 678 functions, including comprehensions and zero-iteration entries. Every entry includes its source operation, count, purpose, kind, proposed removal class and edit boundary. Bounded numerical loops can be device loops or statically unrolled; a Python loop by itself is not proof of a graph break.

The first sentence's detailed observer and the sixteen-sentence count-only observer reproduce original parameter, optimizer and RNG hashes exactly. The sentence-owned count-only observer reproduces all sixteen `94f2b03d9e` records exactly. No counter/AST/profiler time is used as a normal sentence timing. Host API calls include implicit `int`/`float` conversions; `cpu` on this CPU run may be a no-op. Public tensor calls, dispatcher calls and host-read calls are separate, overlapping counting levels, never summed.

| Count over sixteen sentences, staging through kept close/update | Before ownership | Sentence owned, before host optimizations | Final |
| --- | ---: | ---: | ---: |
| Public tensor API calls | 38,265,704 | 37,509,006 | 37,381,989 |
| Tensor dispatcher operations | 42,981,464 | 42,158,992 | 41,872,048 |
| Host-read API calls | 2,950,321 | 2,945,330 | 2,945,330 |
| Scalar dispatcher reads | 2,333,771 | 2,329,329 | 2,329,329 |
| Dynamic `nonzero` operations | 31,870 | 31,870 | 31,870 |

The first sentence alone has 1,145,468 public tensor calls, 1,424,926 dispatcher operations and 164,454 host API reads. Thought accounts for 102,437 of those reads, preparation/orchestration 28,523, close 22,373, generation 10,964, a word's step 120, the two walks 28 and restoration 9. `ConceptSpread.rule_order` alone accounts for 92,994 thought reads; it remains excluded by §5. The other dominant read owners are `MereologicalCodes` and `Memory`, both reserved for rounds A/B.

A separate cProfile-only pass, without the API/loop instrumentation, ranks the permitted candidates. On the original first sentence `_freeze_metadata` runs 138,954 times (0.180492 seconds inclusive), and `SentenceCompose.pack` 33,321 times (0.252057 seconds inclusive). These measured counts replace the supplied estimates. After sentence ownership, pack drops to 31,255 calls / 0.236066 seconds; metadata remains 138,954 / 0.180138. `ThoughtFeatures.semantic_metadata.field` takes 0.058314 seconds. These are profiled function costs, not additive whole-sentence timings.

### Local host changes and their certificates

Three local changes follow the ownership change. Each native lesson,
packed-row and gate replay is compared with `94f2b03d9e`, with exact
chooser traces, registered costs, parameters, optimizer state and RNG.
All comparisons pass. The fixed-input tests additionally load the actual
original functions from Git `3926005f12`; these are equality witnesses,
not assertions intended to fail on the correct original implementation.

| Change | Cause and repair | Fixed-input certificate | First-sentence inclusive seconds, immediately before → after |
| --- | --- | --- | ---: |
| `Meaning._freeze_metadata` | Native immutable scalars and containers repeatedly enter the generic `Mapping` ABC checks. Fast paths for exact built-in types avoid that dispatch; subclasses still follow the original precedence. | 20 cases: values and types, mutable ownership, rejected values and exact error messages, a scalar subclass that is also a mapping. | 0.180138 → 0.158280 |
| `ThoughtFeatures.semantic_metadata.field` | Every UTF-8 byte constructs a tensor and writes individual bit positions. Build the same byte-bit representation in Python, then construct one tensor per field. JSON, aliases, truncation and overflow remain identical. | 20 cases over float32/64/16/bfloat16, Unicode, typed aliases, excluded bindings, nested values and overflow; output bytes, dtype, device and grad status match. | 0.057899 → 0.022468 |
| `SentenceCompose.pack` | Protected storage repeatedly reads `_base`, checks `Parameter`, and converts an already immutable shape to a tuple. Reuse the storage decision and shape, retaining weak lifetime, copies and mutation checks. | Four cases: listed/unlisted parameters and buffer views survive mutation; unprotected scratch retains the same rejection. | 0.236047 → 0.231161 |

The metadata change was taken first because its generic dispatch was a
large, simple removable cost. The inventory then exposed the per-byte
tensor construction; it produced the largest measured local saving.
The saved-value bookkeeping change was last. This is the actual order;
the realized savings were not known in advance. Function costs overlap
and must not be added to estimate a whole-run gain. Final profiled costs
are 0.156729, 0.022373 and 0.231161 seconds respectively. Metadata still
has 138,954 calls, field construction 616, and pack 31,255 on this sentence.

Text preparation is left as prepared by item 1. Its `stage_input` self
time is about 0.000171 seconds; the 0.094907 seconds inclusive includes
its callees. A vocabulary cache would introduce invalidation/ownership
work with no demonstrated substantial saving here. Python metadata,
JSON, alias tables and saved-tensor ownership are still host work; these
repairs reduce their cost rather than claiming they became a graph.

Final review found an ordinary-composition regression in the initial
ownership implementation: when no attention walk was staged, replacing
a `None` grammar mask with all-false would close ordinary leaves. The
empty mask is now installed only when a sentence walk actually exists.
The new fifth ownership case fails on `94f2b03d9e` and passes after
`40e016be91`; all five ownership cases pass in 4.18 seconds. This added
control case disables attention without changing the native corpus or
capacities; none of the measured lesson/gate fixtures is altered.

The broader unchanged attention/expectation/compose/thought/closing-gradient
suite plus new tests passed **128 tests, with one existing slow compile
case skipped**, in 47.77 seconds before that final guard correction.
After the correction, the five ownership tests and all native lesson,
packed and gate certificates were rerun. All pass. The final sixteen
counter-observed parameter/optimizer/RNG records also exactly match the
ownership reference. The gate comparison still exactly matches all 51
original `3926005f12` records.

### Work and seconds before and after

| Sixteen native lessons | Walks | Evaluated rounds | Primary sentence rounds | Forecasts |
| --- | ---: | ---: | ---: | ---: |
| `3926005f12`, before §7's inert-tail ending | 35 | 1,120 | 512 | 16 staged |
| Accepted §7 / unchanged §9 | 35 | 430 | 197 | 16 staged |
| Sentence owned and final | 19 | 233 | 197 | 16 sentence owned |

The final primary round counts are
`13,14,13,11,13,11,13,11,13,11,13,11,13,11,13,13`.
Sentences 2, 4 and 10 also take real attention departures (36 additional
rounds). Thus "one walk" means one primary walk per sentence, with no
duplicated staging walk; it does not discard exploration. On the packed
row, total spent at the four path costs is `12,25,32,32`, including
non-bracket reads. Accepted words by sentence are `2,3,2,0`; forecast
targets retain the kept prefix and grow `2,5,7,7`. This is the authorized
shared-budget choice, not a fresh budget for the final sentence.

| Serial CPU observation, seconds | Sixteen lesson `runBatch` calls | Packed document `runBatch` | Both gates, driver |
| --- | ---: | ---: | ---: |
| Before ownership, §9 complete registry | 70.732232 | 5.781457 | 1.009343 |
| Sentence ownership (`94f2b03d9e`) | 69.342439 | 5.768452 | 1.009122 |
| Metadata fast paths (`6471da2415`) | 69.950676 | 5.593162 | 1.004184 |
| Byte feature construction (`87dd7c10c2`) | 69.459291 | 5.650377 | 1.004086 |
| Saved-value bookkeeping (`e0103c3980`) | 70.454726 | 5.569169 | 0.997807 |
| Final ordinary-composition guard (`40e016be91`) | 70.150731 | 5.588761 | 1.010076 |

These are single observations, with no concurrent benchmark. Lesson and
packed timings exclude the observer's reference-kernel checks; gate
timings use the whole driver, matching the prior report. The ownership
observer adds its forecast/meter audits. The count and profile runs are
separate and are not used as runtime measurements. The final result is
about **0.23 sentences/second, or 4.38 seconds/sentence**. Whole-run time
is effectively unchanged; the local savings do not establish an overall
speedup or improved learned quality.

The previous final count pass recorded 37,381,997 public API calls;
the last pass records 37,381,989. The eight-call difference comprises
eight fewer property accesses, four fewer `detach` calls and four more
`clone` calls. Dispatcher, scalar-read, host-read and `nonzero` counts
are identical, as are all model/optimizer/RNG records. These are observed
API totals, not a claim that Python lifetime bookkeeping has an invariant
call count. No saving is attributed to that tiny difference. The host
optimizations themselves leave 2,945,330 host API reads; most of the
large remaining sources belong to the reserved A/B files.

### Fullgraph result and design question for Alec

**The whole sentence is not fullgraph-clean.** A new probe attempts the
native first lesson's `_run_sentence_word_bricks` with
`torch.compile(backend='eager', fullgraph=True, dynamic=True)` and
`TORCH_LOGS=graph_breaks,recompiles`. Both the unchanged pre-round source
and final source fail capture at `if not bool(present.any())` in that
method (final line 21011), with a data-dependent `Eq(u0, 1)` guard error.
The eager backend tests capture only; this is no accelerator execution
or MPS measurement. No graph is produced, so there is no successful
whole-sentence recompilation or accelerator-speed certificate to report.

Removing that first Python choice alone is insufficient. The inventory
also finds Python meaning roles/references/bindings, registry/table
lookups, variable menus and thought step records. Some numerical choices
can become tensor control flow with equal results. Representing these
Python objects inside a device graph requires an agreed data boundary;
most dominant owners are excluded from this round. The inventory's
removal classes are conservative proposals, not certificates for rewrites
that have not been built.

**Question for Alec:** after A/B lands, should the sentence boundary use
tensor records for meanings, references, bindings, registry addresses,
menus and thought steps, with decoding/publication at the kept close?
**Recommendation:** yes. Establish that representation and its capacity
and overflow rules against the merged owners, then move choices and
loops onto the device while preserving ordering, RNG and gradients.
Use the model's native capacities; do not shrink fixtures or assume a
new arbitrary bound. Until that design is agreed, the safe result here
is sentence-owned attention plus the certified local host reductions.
No excluded owner is edited to bypass the boundary. A trained-model
quality benchmark should accompany the next performance comparison;
the current early-training replay certifies mechanics, not mature quality.

### Evidence, edit boundary and local commits

The compact [certificate receipt](2026-11-14-attention-block-certificate.json)
records source digests, exact comparisons, counts, timings and hashes of
the raw evidence. Raw artifacts remain in ignored `output/attention-block/`.
The stream remains §7's original `initial-rng.pt`, SHA-256
`b4f4d373fb74a4c57d8c6f8d16338a4c319bc1edac8a668527f8ea98dadd4745`.
Its recorded `PYTHONHASHSEED=1521266424` is replayed, not newly selected.
The environment is CPU/eager, one thread for Torch/OMP/MKL/OpenBLAS/
VECLIB/NUMEXPR/Inductor, Python 3.12.13 and Torch 2.15.0 on the M2 laptop.

Reproduce with `.venv/bin/python`, the saved
`output/attention-thinking/environment.json`, and `PYTHONPATH=bin:test`:

```sh
python test/attention_block_certificate.py lesson \
  --replay output/attention-thinking/base-lesson \
  --compare output/attention-block/owned-v3-lesson \
  --report-base output/attention-thinking/base-complete-lesson \
  --output output/attention-block/recheck-lesson
python test/sentence_host_inventory.py --source "$PWD" \
  --replay output/attention-thinking/base-lesson \
  --output output/attention-block/recheck-counts
python test/attention_block_compile_probe.py \
  --replay output/attention-thinking/base-lesson \
  --output output/attention-block/recheck-fullgraph
```

Use the certificate's `packed` and `gates` modes with the corresponding
ownership references. `--detail` gives the first-sentence inventory;
`--profile` gives the separate cProfile-only pass. Outputs must be new
directories. The original corpus/fixtures and saved stream are required;
there is no generated substitute fixture.

Production edits inside §5 are `ModelAttention.py` and, in `Models.py`,
the sentence owner plus `_sentence_path_cost`, which must consume its
single shared forecast. All are for item 1. The additional production
edits outside §5 are exactly `Meaning.py`, `ThoughtFeatures.py` and
`SentenceCompose.py`, the independent item-3 repairs explained above;
none is in A/B's supplied modified-file list. No other production file
is edited in this round.

New files outside the production edit list are the ownership and host
fixed-input tests, `attention_block_certificate.py`,
`sentence_host_inventory.py`, `attention_host_report.py`, and
`attention_block_compile_probe.py`; each supplies the requested replay,
inventory or capture evidence. This §10, the two host-inventory appendices
and the compact certificate receipt are the requested report artifacts.
No existing test assertion or fixture is changed.

| Work | Local commit on `performance` |
| --- | --- |
| Sentence ownership and forecast sharing | `94f2b03d9e` |
| Unchanged-source inventory and reporting tools | `27fde08df9` |
| Metadata dispatch reduction | `6471da2415` |
| Byte feature construction | `87dd7c10c2` |
| Saved-value bookkeeping | `e0103c3980` |
| Ordinary-composition mask guard and regression | `40e016be91` |

The final report/probe/receipt commit follows these implementation commits.
This round is committed locally on `performance`; no push is made and
`main` is unchanged. The remaining full-sentence graph work is the design
question above, not a completed compile claim.
