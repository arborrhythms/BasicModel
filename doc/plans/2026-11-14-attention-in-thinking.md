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
