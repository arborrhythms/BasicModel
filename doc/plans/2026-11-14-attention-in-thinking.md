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
