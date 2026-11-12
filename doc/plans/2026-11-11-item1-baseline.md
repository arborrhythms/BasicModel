# Item 1 baseline: where test and lesson time goes (Claude, 2026-11-11)

Read-only measurements taken at Alec's request to begin the compiler work,
and the order they suggest. Nothing here changes code. Items 1.5 and 1 are
Claude's to build ([item 0.5 spec](2026-11-09-item0-5-test-development.md),
§15, D5: "item 1's baseline and plan, read-only, when the machine is quiet;
then item 1.5", then item 1).

## 1. The request (Alec, 2026-11-11)

Asked where item 1.5 should be built, Alec: "What is 1.5? Maybe it makes
sense to do fullgraph and compile optimizations, which will help testing
speed. That can be done in a separate branch."

*What item 1.5 is.* The item Alec created on 2026-10-30
([plan](2026-10-30-item1-5-lookups-and-folds.md)): remove the Python work
per row, per edge and per word from the hot path before the compiler work
compiles it. Five steps: the derived concept forms kept as a tensor by one
scatter per order; memory lookups without host loops; the percept pass once
per sentence; the second trial forked at its departure; every rewritten
brick tracing under `fullgraph`. Steps 1 and 2 were built that day on the
sandbox branch `item1-5-lookups-and-folds` (`9151b6f10b`), with certificates
and a microbenchmark (the derive about 50 times faster, the membership read
23 times, LTM1's adds 45 times). They were never ported to main: Alec's order
of 2026-11-03 put the port after items 2.0 and 2. Steps 3 and 4 are not built.

## 2. Measurements

Taken while Codex's round-7 lesson panel was running, so absolute times are
inflated and only proportions and counts are claimed. The lesson figures of
§2.2 are from an export of the landed main `3c9399199d` (`git archive`,
outside the tree), `bin/corpus_lessons.py --eager --curriculum lesson-13`
on `data/MM_corpus_lessons.xml`, CPU, eight sentences after two warm-up
sentences.

### 2.1 The default sweep

From the receipt of the last complete default sweep
(`output/tests/20261110-045158-db639e`, 7,972 cases, three workers at the
28 GiB ceiling, 1,352 s of wall time):

| | |
|---|---:|
| worker launches | 142 |
| summed worker time | 3,886 s |
| summed test bodies | 1,496 s |
| outside the test bodies (start-up, collection, fixture setup) | 2,390 s, 61% |
| overhead per launch, median | 10.3 s |
| cases under 0.1 s | 6,598 |
| share of test-body time in the top 10, 100 and 500 cases | 20%, 52%, 82% |

Compilation is not what the test bodies pay for. The same file run twice on
the export, once under the sweep's policy (`MODEL_COMPILE` unset, Inductor
on CPU) and once under `MODEL_COMPILE=eager`: `test/test_output_walk.py`
113 s and 125 s; `test/test_familiar_tier.py` 18.8 s and 19.0 s.

### 2.2 A lesson sentence

Codex's round-7 panel (its receipts, round-7 tree, two arms at a time):
6 to 8 s of training and 3 to 5 s of diagnostics per sentence, so an arm of
160 sentences takes about 35 minutes and a panel of four more than two
hours of process time. The lesson runs are uncompiled: `--eager` sets
`util.TheCompileBackend` to `none` and replaces `torch.while_loop` by a
Python loop.

Where the training time goes, on the landed main, by plain wall-clock
timers around each function (no profiler):

| Part of `runBatch` | Share | Calls per sentence |
|---|---:|---:|
| `BasicModel.generate`, every walk | 70% | 22 |
| of which `Generative.inverse_menu` | 53% | 34 |
| of which `LanguageSpace.reverse_binary_step` | 43% | 615 |
| its callers, which overlap: `_sentence_reader_error` | 54% | 8 |
| `_sentence_answer_error` (it calls the reader error) | 27% | 2 |
| `SentenceField.reconstruct` | 13% | 2 |
| `_backward_training_loss` | 5% | 1 |
| `BasicModel.reverse`, the order-zero decoder | 3% | 24 |

A profile of the same eight sentences counts 16.1 million Python function
calls per sentence, and per sentence about 183,000 `torch.where`, 99,000
`sum`, 76,000 `all` and 38,000 `square` calls on small tensors. The walk
is `WalkTrials.generation_pair` (a greedy walk, then an explore walk that
repeats the greedy actions up to its departure; 11 pairs per sentence),
each step of which builds one menu by one `reverse_binary_step`
per binary generate rule, in a Python loop over the rules
(`inverse_menu`), followed by a recomposition of every candidate
(`LanguageSpace.decoder_eligibility`). `FormCentroids.fit` is called about
2,100 times per sentence underneath. Composition (`stage_cs_lang`, the word
steps) and perception are a few percent each.

### 2.3 What moved since 2026-10-30

The cost reading that made item 1.5 named the derived codebook read, the
membership read, the LTM indices and the per-word percept pass. On
lesson-13 today those are a few percent. The dominant cost is the generate
walk, which round 6 of item 2.0 made the reader of every reconstruction and
every supplied-answer score, and which runs as Python because the lessons
run uncompiled. Item 1.5's steps still scale with occupancy (occupied LTM
rows and lexical concepts), which lesson-13 does not exercise.

### 2.4 Is it a search of the lexicon? (Alec's question, 2026-11-11)

Alec: "Presumably because generate is looking for matches against the idea
in the entire lexicon?" Only at the entrance of a walk, and not at this
size. From the same profile:

- *Once per walk.* `Generative.walk_bank` calls `lexical_candidates`: one
  scan of every lexical concept that has a surface, in chunks of 256,
  keeping the `reconstructionBasisLimit` best matches of the root (16 by
  default). On lesson-13 that scan is 1% of a sentence's time, and
  `walk_bank` as a whole 2%, 21 times per sentence.
- *Every step after that works on the walk's bank*: those 16 rows and the
  primed structural candidates. For each binary generate rule,
  `LanguageSpace._bounded_binary_reconstruction` scores the parent against
  the bank and keeps the 16 best per side (`shortlist`, run twice, 7% of the
  reverse step), then forms every ordered pair, up to 256, recovers its
  lanes, fits a centroid mixture (`FormCentroids.fit`, 55% of the reverse
  step) and recomposes it (`compose_blocks`, 17%) to rank the pairs.
- *Where the time is.* A lesson sentence builds 34 to 39 menus over its 22
  walks (the sentences are short, fewer than two steps a walk). Each menu
  runs 18 binary reverse steps, 10 unary steps and one eligibility pass
  that recomposes the candidates. The cost is that count, times small
  tensors handled in Python, not the size of anything searched.
- *A second match against a bank sits inside the fit.*
  `FormCentroids._selected` finds the dictionary whole of each pair's two
  children by a distance to every whole form of the walk's bank: 512
  children per rule per step, about a sixth of the profiled time. *Claude's
  readings, to certify before either is used:* those children are the 16
  shortlisted rows of each side, expanded over the pairs, so 32 reads would
  serve where 512 are made, and their rows are already known from the
  shortlist; and the shortlist depends on the rule only through its
  `content_operand` and its meaning layout, and on neither side, so where
  those agree it is recomputed identically for every rule.
- *At a large vocabulary the presumption becomes true.* The entrance scan
  is linear in the lexicon, and it includes a Python pass over every lexical
  identity (the row set of `lexical_candidates`) and one `lookup_rows` per
  chunk, repeated for each of a sentence's walks. At lesson-13's vocabulary
  it is negligible; at item 0's it is the first place the walk would slow
  down. It has the shape of item 1.5's lookups: one product of the cue
  against a kept content matrix.

### 2.5 Why 22 walks, and how much of them is repeated (Alec's question, 2026-11-11)

Alec: "why are there 22 walks for one sentence? I thought we did two tree
traversals, an explore and exploit, which diverge at one location (so share
a large trunk that should be optimizable)."

The two traversals are the two compose trials. A generate walk is the
read-out that costs a result, and each read-out is a pair
(`WalkTrials.generation_pair`): a greedy walk, then an explore walk that
repeats the greedy actions up to one departure. From the profile's caller
graph, per sentence:

| What is read out | Caller | Pairs | Walks |
|---|---|---:|---:|
| The reconstruction cost of each compose trial | `_sentence_path_cost`, `SentenceField.reconstruct_generation` | 2 | 4 |
| The supplied answer's cost for each compose trial, by the presented reader and by its comparison twin | `_sentence_answer_error`, `_sentence_reader_error`, `AnswerComparison.forward` | 4 | 8 |
| The same for each of the closing thought's two trials | `WalkTrials.thought_pair`, `ThoughtAnswer.scorer` | 4 | 8 |
| The capture of the kept understanding at the end of the batch | `_capture_understanding`, `reverseOutput` | 1 | 2 |
| | | 11 | 22 |

The comparison twin (`AnswerComparison`) reads the same understanding as
the presented reader through independent copies of the output-owned
weights, so its walk may choose differently, but it starts from the same
idea with the same bank.

*How much is repeated.* Every walk's starting idea and every menu's inputs
and outputs were hashed over the same eight sentences
(`inverse_menu`'s parent, live mask, basis, validity, priming, limit,
constituents, terminal mask and whole forms; the output is every tensor of
the returned menu):

| Over eight sentences | |
|---|---:|
| walks | 176 |
| distinct starting ideas | 38 (3 to 7 per sentence) |
| menus built | 291 |
| distinct inputs | 117 |
| distinct pairs of input and output | 128 |
| inputs whose output differed on a later call | 11 |

So 56% of the menus built are exact repeats of a menu already built in the
same sentence: the same idea, bank and weights, and the same result. The
11 inputs whose output changed are calls on either side of a parameter
update inside the sentence, so a reused menu must be keyed to the
parameters it was built with. Nothing reuses a menu today.

### 2.6 Could the consumers share one pair? (Alec's question, 2026-11-11)

Alec, on "several consumers each run their own pair of walks": "Maybe they
could all share the same pair."

Every pair of eight lesson sentences was recorded on the branch (its
starting idea, greedy and explore actions, realized outputs, departure and
optimizer step): 88 pairs, 176 walks.

| | Pairs, of 88 |
|---|---:|
| no legal departure: the explore walk is the greedy walk again, equal in every case | 48 |
| the greedy walk repeats an earlier one of the sentence exactly (same reader, weights, idea, actions, output) | 41 |
| the whole pair repeats an earlier one exactly | 31 |

The whole-pair repeats are the thought's read-outs repeating a compose
trial's (12 by the presented reader, 8 by the twin) and the two compose
trials reading an unchanged understanding (4 reconstruction, 4 twin, 3
presented). The second kind is the design's own:
`WalkTrials.paired_departure` and `paired_action_draw` give corresponding
walks of the two outer trials the same quantiles ("Sharing its quantile
prevents unchanged understandings from winning an outer comparison just
because the inner decoder happened to explore a better path"). Sharing only
what is already equal would leave 81 of the 176 walks, 10 a sentence.

What keeps the consumers from sharing one pair:

- *The comparison twin* never starts from the idea of the reader beside it
  (0 of 32). `_materialize_answer_idea` conditions the answer with
  output-owned weights, and the twin has its own copies. Its walk is its
  own by design.
- *Reconstruction and the presented reader* start from the same idea in 12
  of 16 compose trials, but they cost different targets (the input, the
  supplied answer) and draw their explorations from different sequences
  (`decoder`, `output`).
- *A thought's read-out* equals a compose trial's read-out by the same
  reader, in idea and greedy walk, 30 times of 32. In 10 of those only the
  exploration differs: the thought's pairs come later in the sequence of
  draws and take fresh ones.

What is left of a sentence after the first step (§6), by plain timers at
one thread on the branch: `generate` 56%; the menus still built 37% (29 a
sentence); `propose_actions` 11%; decoding a walk 5%; `walk_bank` 2%;
`backward` 6%.

## 3. Reading

1. Alec's direction holds: the speed of testing is in the compile and
   `fullgraph` work, and specifically in the generate walk. The walk is
   written for `torch.while_loop` and the lessons replace that by a Python
   loop; neither the default sweep nor the lesson panel is paying for
   compilation today, they are paying for uncompiled Python.
2. Four levers on the walk, in the order of what they need. *Exact reuse
   comes first (§2.5):* a menu built once per distinct idea, bank and
   parameter version within a sentence, which changes no result and removes
   more than half of the menus. Then:
   - *One menu per step.* Build a walk step's menu in one batched pass over
     the generate rules instead of one `reverse_binary_step` per rule. The
     comment in `inverse_menu` records why it is a loop: passing the whole
     catalogue made a compiled menu trace one dispatch per rule per entry.
   - *Fewer walks.* Eleven pairs per sentence walk end slots that are often
     the same idea (the reconstruction, the reader error of each reading,
     the answer error). What may be shared between them, and what a pair's
     explore walk can take from the greedy walk before its departure, needs
     a certificate each.
   - *The walk compiled.* The body traced under `fullgraph`, so that
     lesson and training runs can keep the compiled path and the tests of
     the walk need not run it as Python.
3. The sweep's other 61% is launches and fixtures, which is harness work
   and item 0.5's.

## 4. Proposed order (for Alec)

1. *Item 1, first round: the generate walk* (the three levers of §3), with
   the gates item 1.5's plan set for a rewrite: equal menus, choices, costs
   and parameter updates against the current path on the standing fixtures
   and one lesson batch; each rewritten body tracing under `fullgraph`; one
   before and after timing on lesson-13, on a quiet machine.
2. *Item 1.5's port*: the sandbox's steps 1 and 2 onto main, then steps 3
   and 4, measured at an occupancy where they matter.
3. *The rest of item 1*: compiled against eager on the math corpus, the
   four runtime failure groups, the three `torch._assert_async` sites, the
   remaining eager hot spots.

*Where.* Alec, 2026-11-11: "That can be done in a separate branch." A
branch in a second worktree keeps Codex's checkout and its source hash
untouched. *Claude's reading:* this replaces the detached worktree with no
branch of the item 0.5 spec's D1. Round 7 of item 2.0 is uncommitted in the main checkout and
changes `Models.py`, `Language.py` and `Spaces.py`, so the branch is cut
from the landed main and moved onto round 7's commit when it lands.

## 5. Observed in passing

`test/test_output_walk.py::test_runbatch_answer_availability_never_reassigns_the_decoder`
fails on the landed main (the export, alone and in its file, under both
compile policies) and in the round-7 tree: with no supervised output, five
of the six generate-policy parameters change (`assert not
any(got['changed'])`, line 424). It passed in the two complete sweeps of
2026-11-10 (02:46 and 04:51) and was not in the focused run that validated
rounds 5 and 6 (2.0 spec §20.26). Not diagnosed here. Claude's reading: the
round-6 correction (2.0 spec §20.19) put the walk's credit back on
reconstruction, so the generate policy is now trained without a supplied
answer and the assertion predates that; round 7's complete sweep will meet
it.

## 6. First step, built on the branch: menus kept between the walks of a batch (Claude, 2026-11-11)

Alec, 2026-11-11: "You can do the generate walk first". The branch is
`item1-generate-walk`, cut from the landed main `3c9399199d`, in a second
worktree at `WikiOracle/basicmodel-item1`. Nothing is committed on it. With
the second step (§7) its source hashes to
`88dc4d653af4a134c5e64614e9c20ed66420e19141814e3319c4301bb9f0f9f5` by the
method of the 2.0 spec's §20.33 (1,043 files); the two steps add 199 lines
and remove 9 in `bin/Generative.py`, `bin/Models.py` and `bin/WalkTrials.py`,
with `test/test_item1_generate_walk.py` (the regression ran on
`8418b5e7c43a`; only that test file changed after it).

*What was already there.* `Generative.InverseMenuRead` is a cache of menus
for one eager CPU walk ("Parameters and dictionary snapshots stay fixed
inside a walk"), and `_output_generate_walk` builds every menu under
`torch.no_grad` ("The menu is an observation. Only the selected inverse is
an operand"), recomputing the chosen action live in `selected_inverse`. So a
menu carries no gradient and reusing one cannot move a pullback. The
repetition of §2.5 is between walks, which a cache that dies with its walk
never sees.

*What a menu depends on.* With equal inputs and no optimizer step between
two calls, 14 of 130 inputs still gave two different menus. In every case
the only field that differed was `occupation`, across all three readers.
`GenerateOccupation.features` reads `PartsOfSpeech._value_arguments`, which
`prepare_values` restages at every `walk_bank` and at the commit; nothing
else in a menu reads it.

*What is built* (`bin/Generative.py`, `bin/Models.py`):

- `Generative.MenuStore` keeps detached menus for one batch. A menu is
  named by the content of everything its builder reads (the parent, the
  live mask and every option of the walk: basis, validity, priming, limit,
  case bank, constituents, families, terminal mask, whole bank, proposed
  actions) and by a token of the parameters as they are bound when the
  walk starts: the model's `_parameter_epoch`, the identity and version of
  every parameter of the language and of the generate operators, and of
  the operators' buffers. A functional call (the comparison twin) or an
  update therefore names a different menu. An input the store cannot name
  is never reused, and a copied model starts with an empty store.
- `InverseMenuRead.read` looks in its own walk first, as before, then in
  the store. On a reuse it restores the operators' published activations,
  as the walk's own cache does, and reads `occupation` again from the
  staged directions.
- `runBatch` opens a store for its batch and removes it when the batch
  ends, and `_output_generate_walk` gives each walk's reader that store under
  the condition that already governs the walk cache (not compiling, CPU,
  compile backend `none`). A walk outside a batch has no store.
- `reconstruct_grammar_sentence`, the reading asked for by name, sets the
  store aside for its walk. *Corrected after the first regression run:* the
  store first lived on the model between batches, and
  `test_grammar_reconstruction_reads_the_concluded_state_before_trace_disposal`
  failed on the branch: its reading took menus that the same forward pass
  had built, so its spy on `reverse_binary_step` saw no call ("the reading
  must traverse the grammar inverse"). The values were equal; the gate's
  assertion is kept as it is, and a named reading always traverses.

*Certificates* (`test/test_item1_generate_walk.py`, six of its eight cases): a menu's name
follows content and never the tensor object, and every single change of an
input changes it; an update in place, a functional call and the parameter
epoch each change the token; a later walk takes the menu without a rebuild,
with activations restored and occupation read again, and a walk under
gradient, after an update or after the batch builds its own; verify mode
(`MenuStore.verify`) rebuilds on every reuse and fails on any difference;
one supervised batch on a real model reuses menus with every reuse equal to
its rebuild, and leaves no store behind; and a reading asked for by name
sees no store.

*Verify mode on the lessons.* Twelve lesson-13 sentences after two
warm-up sentences, every reuse rebuilt and compared in every field but
`occupation` and in the published activations: 927 menu reads, 366 built,
561 reused, 561 equal (an earlier run of the same kind: 621, 250, 371, 371).
On the final source, with the second step: 520 reads, 235 built, 285 reused,
285 equal.

*Timing* (lesson-13, eager, CPU, the landed-main export against the
branch, while Codex's panel was on the machine):

| Run | Main | Branch |
|---|---:|---:|
| The same ten seeded sentences, one thread, back to back | 57.4 s | 44.2 s |
| the same on the final source, with the second step | 54.1 s | 39.7 s |
| Twelve unseeded sentences, default threads, first pair | 12.8 s a sentence | 8.3 s |
| the same, second pair | 10.5 s | 9.3 s |

The seeded pairs are the comparison to read: the same draws, 23% less
time for this step (221 of 387 menu reads reused) and 27% less with the
second. Unseeded runs draw different walks (729
and 914 menu reads in the branch's two runs) and the second pair shared the
machine with the seeded runs. Codex's lesson launcher runs at one thread
(`OMP_NUM_THREADS=1`), which is the seeded pair's condition. §2.2's
shares were measured at default threads, where a sentence takes about twice
as long; at one thread the menus are about 40% of a sentence, not 53%.

*Same updates.* Ten seeded lesson sentences at one thread with
`PYTHONHASHSEED=0`, on the export and on the branch: the digest of all
9,856,003 parameters and of the sentence's cost components is identical
after every batch, for this step alone and again on the final source with
the second step. The export repeats itself exactly across runs, so the
instrument is deterministic. The seed serves this comparison only.

*Regression.* Fifteen walk-related test files, 175 cases. On the export of
the landed main: 165 passed, 8 skipped, 2 failed, both on main itself and
both outside the focused run that validated rounds 5 and 6 (§5, and
`test_concept_output.py::test_native_understanding_retains_field_and_native_percept_evidence`,
which expects `order_zero_field` among the reconstruction carriers; it
passed in the sweep of 2026-11-10 02:46 and failed in the one of 04:51).
On the branch as first built: the same two and the gate case above, which
the corrected scope repairs. On the final source: plain, 173 passed, 8
skipped and the same 2 failed as on main, nothing else; with both
certificates switched on for every test, no reuse differed from its rebuild
and no unexplored pair from its greedy walk, and the same 2 failed. (Two of
this item's own cases count builds and walks in the plain mode and failed
under the forced certificates; they now set their own mode, and the file
passes 8 of 8 both ways.)

*Not done.* No sweep: the machine is Codex's until round 7's sweep is
through, and sweeps take turns. Compiled walks and walks on a device do not
use the store (the walk cache they would hang it on is eager CPU only).

*What the measurements point to next,* each with its own certificate: the
menus that must still be built, a third of a sentence (inside one, the
fit's 512 dictionary reads where 32 are distinct, and the shortlist
recomputed for every rule and side, §2.4); `propose_actions`, a tenth of a
sentence, once per walk, built without gradient from the same banks, so the
same store can keep it with the activations it publishes; then the walk's
body under `fullgraph`.

## 7. Second step, built on the branch: no second walk without a departure (Claude, 2026-11-11)

From §2.6: in 48 of 88 pairs the explore walk had no legal departure and
was the greedy walk again. In `WalkTrials.generation_pair`, when no stream
of the greedy walk has a legal departure, the explore walk and its cost are
no longer run, and the pair takes the greedy walk's result for both. The
two quantiles are still drawn, in the same order, so the corresponding walk
of the other outer trial and every later draw read what they read before.

*What an observer can see.* A walk that is not run writes no line to the
lesson harness's `lexical-admission.jsonl` (one line a walk). Nothing the
model keeps changes: the seeded comparison of §6 is identical on this
source.

*Certificates.* `WalkTrials.verify_unexplored` walks the unexplored pair
anyway and requires equal actions, pending mask, residual, certificate,
credit features, credit tops and cost (`same_walk`). Two cases of
`test/test_item1_generate_walk.py`: a pair without a departure walks once,
costs once and still takes both draws, and verify mode fails on a walk that
differs. On the lessons in verify mode, twelve sentences: 54 unexplored
pairs, all equal to their greedy walks.

*Sharing a pair between consumers: what is exact, and what is Alec's.*

- *Exact and left to build:* a greedy walk that repeats an earlier one (41
  of 88). Its menus are already taken from the store; what would still be
  saved is the walk's own bookkeeping, its decoding and its cost, and it
  needs one walk's live graph to serve two costs. Small beside the menus
  that are still built.
- *A decision (not built):* a thought's read-out of an understanding the
  thought did not change could take the compose trial's read-out, draws
  included. 20 of 32 are equal today and 10 more differ only by a fresh
  draw. This extends the rule `paired_departure` states, and it changes
  which quantiles those ten walks read.
- *A decision (not built):* reconstruction and the answer reader could
  share one greedy walk where their idea is the same (12 of 16), each
  still costing it against its own target. Their explorations are separate
  sequences today.
- *Not proposed:* the twin reading the presented reader's idea. It would
  change what the comparison cost measures.

## 8. One pair for each stage: what stops it (Alec, 2026-11-11)

Alec, on §7's list: "I'm pretty sure that we only need one grammatical pair
across all compose, thinking, and generation. You give reasons we can't do
that now, so let's explore What stops everyone sharing one pair".

On the twin: "I see compose (to produce a state), thinking (to produce a
new state), and generate (to produce an output) as being able to have a
single pair each. So if we are doing output and reconstruction, and we have
different walks for each stage, we are doing compose-generate (for
reconstruction) and compose-thinking-generate (for output), with the
initial compose shared between both, for a total of 4 stages (thus four
pairs), none of which are covering the same ground." On reconstruction and
the answer reader: "Why not 16/16, and how can we reconcile these?" On the
thought's read-outs: "Again, I don't see why these aren't fully
overlapping."

### 8.1 What was measured

The lessons again, eager, one thread, on the branch source `88dc4d65...`
(its updates are main's, §6): sentences 3 to 10 (ideas stage) and 65 to 76
(relations stage), 20 sentences, 220 pairs, no seed. For each pair: the
roles its read-out is given, the end slots of the same sentence record,
what the question conditioner and the attention read add, the draw it
takes, its walks and its costs. The scripts are in Claude's session
scratchpad, not under source control.

§2.6 counted "the same idea" at `BasicModel.generate`'s argument: after the
question conditioner, before the attention read. The counts below separate
those steps. They come from another unseeded run and differ from §2.6's in
detail (there 12 of 16, here 35 of 40).

### 8.2 Today's eleven pairs against the four stages

| today | pairs | what it unfolds | in the four stages |
|---|---|---|---|
| compose trial, `'decoder'` pair | 2 | the trial's stack, `record.end_slots` | the kept trial's is stage 2's pair; the other is a price |
| compose trial, presented `'output'` pair | 2 | the trial's meaning | a price |
| compose trial, twin `'output'` pair | 2 | the same meaning, the twin's weights | a price, the one that enters $A$ |
| thought trial, presented pair | 2 | the kept composition's meaning again, or the thought's | a price |
| thought trial, twin pair | 2 | the same | a price, the one that keeps and credits the thought |
| batch end, `'output'` pair | 1 | `resolveAnswer`'s idea | stage 4's pair |

Nine of the eleven are prices: read-outs whose only job is to cost a trial
of an earlier stage. Each is itself a whole pair with its own departure.

### 8.3 The twin

- *It reads the same state.* Its roles are the presented reader's in 40 of
  40 compose-trial reads. The one difference before the walk is
  `question_conditioners`. The twin's copy shifts the root role by 0.002 to
  0.05 (38 of 40 reads). The presented reader's adds exactly zero in all
  40, and in every thought and batch-end read. In a separate check of six
  sentences the presented weight stayed exactly zero and no gradient reached
  it, while the twin's copy took a nonzero gradient and moved every
  sentence. "Never the same idea" is one weight matrix that trains in the
  twin and not in the reader it copies (§8.7, first).
- *Why it exists* (operators update §21 to §23, 2026-10-06). On the XOR
  gate a reader trained on kept roots could not rank the other composition,
  and trained on both roots it lost its fit (class 2 of 10); two readers
  gave 9 of 10. That reader was an affine head, and in `XOR_grammar.xml` it
  still is: `answerSynthesis` is false there and no walk is involved. In
  the lesson configuration `answerSynthesis` is true, the reader is the
  generate walk, and the twin doubles every price.
- *As a judge in the lessons.* Its greedy cost is within 1.2% (ideas) and
  2.6% (relations) of the presented reader's at the median. In the 9
  sentences whose two compositions differ, the two readers' greedy
  read-outs rank the compositions the same way in 8. Their kept read-outs,
  which carry each reader's own explore draw, agree in 5. (The twin's kept
  read-out is the $A$ of a question; `_answer_trial_cost` sets $A$ to zero
  on a statement, whose read-outs then only train the readers. 3 of the 9
  are questions.) In the 11 sentences whose compositions end in the same
  state the twin prices both equally (the shared draws).

### 8.4 Reconstruction and the answer reader

- Reconstruction unfolds the stack compose left:
  `SentenceField.sentence_generated` on `record.end_slots`, "untransformed"
  (here always one slot, the root). The answer reader unfolds the
  sentence's meaning: `BasicModel._sentence_reader_error` stacks the roles
  of `observation['meanings']`, an open clause replaced by its fused point.
- The two are the same tensor in 35 of 40 trials (16 of 16 ideas, 19 of 24
  relations). They differ where the close fuses a clause into a point other
  than the root (both trials of "Ciao is a cat.") or closes on two or three
  roles (one trial of "A cat moves.", both of "Is Ciao a cat?").
- The output side then applies two learned steps of its own before it
  walks: the question conditioner (zero for the presented reader, §8.3) and
  `_generation_read`, the answer attention's read, which moves the idea by
  7e-5 to 2e-3. After them no presented walk starts from the reconstruction
  walk's input (0 of 40). With equal roles the greedy actions still agree
  in 27 of 35 and the walked words in 19.
- In the four stages these are different stages on purpose: stage 2 reads
  what compose left, stage 4 reads what thinking concluded. What the code
  has between them is the meaning substitution, the conditioner and the
  attention read: a change of state inside the read-out, which no pair
  explores. *Reconciling them* means naming stage 4's operand. If it is the
  thought's concluded state, and those steps belong to thinking or go, then
  wherever thinking changes nothing the two stages walk one state, and one
  greedy walk can be costed against both targets.

### 8.5 The thought's read-outs

- `ThoughtAnswer.scorer` prices a thought trial with the same two readers.
  It substitutes the thought's meaning (`bound_view`) only when the thought
  concluded an accepted meaning of one or three roles; otherwise it reads
  the kept composition's state again. Of 40 thought trials 5 concluded an
  accepted meaning and 1 was substituted. So 39 of 40 presented read-outs
  start from a compose trial's roles: the price of the thought is the price
  of the composition, read again.
- They are not the same walk, for two reasons. `_sentence_generation_draws`
  is set to `None` before `_commit_sentence` runs the closing thought, so
  each of these pairs draws its own departure (0 of 40 from the shared
  table; of 32 with a legal departure, 31 explored differently from the
  compose trial's pair). And the attention read's source has moved on, so
  the walked idea differs in its last digits (0 of 39 identical; the greedy
  actions agree in 39 of 39, the greedy cost in 33).
- *The consequence.* In 19 of 20 sentences both thought trials read the
  same state. The twin's greedy price for the two was equal in 19 of 19.
  Its kept price, which is what `thought_pair` compares, differed in 5, and
  in 4 the explore thought was kept and credited on that difference. The
  difference came from the read-out's own departure, not from the thought.
  `paired_departure` states the rule this breaks: "Sharing its quantile
  prevents unchanged understandings from winning an outer comparison just
  because the inner decoder happened to explore a better path."

### 8.6 What is finally said starts from a third idea

The batch-end pair, which produces the presented output, starts from
`resolveAnswer`'s idea. In 18 of 20 sentences that is none of the ideas a
trial's read-out priced (in 2 it is the last thought read-out's). The
prices that kept and credited the composition and the thought are prices of
a read-out that is not the output.

### 8.7 Two findings for Codex, whatever is decided

1. The presented reader's question conditioner does not train in the
   lessons: its weight is exactly zero after 76 sentences and no gradient
   reaches it, while the twin's copy of the same weight trains every
   sentence. Its docstring calls it "the differentiable consumer of the
   question". The cause was not looked for.
2. The thought pair's keep and credit follow the luck of an inner read-out
   (§8.5, last point). The first arrangement of §8.8 removes the case;
   otherwise the shared draws have to cover the thought's read-outs.

### 8.8 The arrangements, in walks a sentence

Today: eleven pairs, 22 walks; 19.75 measured on the branch, where a pair
without a legal departure walks once (§7).

*One pair for each stage (the four).* Compose, the reconstruction read-out,
the thought and the output read-out each explore once. A trial of an
earlier stage is priced by one greedy read-out, not by a pair:

- each composition: one reconstruction walk and one answer walk (4; 2 where
  both compositions end in the same state, 11 of 20 here);
- the kept composition's reconstruction explore walk (1);
- each thought trial whose state changed: one answer walk (0 to 2; 1 of 40
  trials here). An unchanged state takes the price already read;
- the output pair (2).

Between 5 and 9 walks, 5 or 7 for most of these sentences. A second reader
on every price adds one walk for each priced state.

*One departure a sentence.* The rule of operators update §14.3 ("One
departure per sentence, over both walks", then narrowing and compose;
`SentenceCredit.departure` draws the walk, then the round) extended to the
thought and the two read-outs: two complete derivations that share
everything before the departure. 2 to 4 walks. Each chooser is credited on
fewer sentences, with a larger importance weight.

Both change which walks are explored and what the choosers are credited on;
neither is an exact saving. By walk count alone (generate is 56% of a
sentence, §2.6) the first takes roughly a third off a sentence and the
second somewhat more: an estimate, not a measurement.

*A third choice inside the first.* A composition's answer price reads the
composition's meaning directly today, before any thought. Read as
compose-thinking-generate, each composition would be thought about before
it is priced, which means thinking on a composition that is not yet
committed.

### 8.9 Decided (Alec, 2026-11-11, question tool)

1. *The arrangement:* "One pair per stage (Recommended)".
2. *The twin where the reader is the generate walk:* "Retire it for walk
   readers (Recommended)". With `answerSynthesis` true the presented reader
   prices the trials; `XOR_grammar.xml` keeps its twin (affine head, no
   walk).
3. *Whether a composition is thought about before it is priced:* "we want
   both reconstruction and output to influence a single understanding; so
   if I understand correctly, we need to price it through thought (this is
   also why we remove the projection of the output gradient onto the
   reconstruction gradient, i.e. to prevent gradient fighting)."

*Claude's reading of the third.* He understands it correctly. Today the
$A$ that prices a composition is the cost of a read-out of the
composition's own meaning, and what is finally said starts from another
idea in 18 of 20 sentences (§8.6): the output objective does not yet judge
an understanding by its output. Priced through the thought, $A$ is the cost
of the output that understanding leads to, so $R$ and $A$ are two prices of
one understanding, and the compose pair keeps the one both prefer. It is
the discrete side of the 2.0 spec's §1 and §2: there the two gradient
chains share the trunk up to the understanding and output's opposing
component is removed; here the two costs choose the understanding.

*A correction.* The option's text said pricing through the thought costs
"about two more walks". It does not. The walk that prices a composition
becomes the output's own greedy walk, so the output pair needs no greedy
walk of its own: 4 to 7 walks a sentence against §8.8's 5 to 9. What it
adds is one more thought where the two compositions differ, and thinking
before the commit.

*A sentence under the three decisions.*

1. The compose pair as now: the greedy composition and one departure
   (`sentence_pair`, `SentenceCredit.departure`).
2. Each composition is priced once. $R$: one greedy reconstruction walk of
   its stack (`SentenceField.sentence_generated`). $A$: its thought run
   greedily, `resolveAnswer`'s idea, and one greedy output walk costed
   against the supplied answer. A composition that ends in the other's
   state takes the other's price.
3. The kept composition is committed, and each later stage explores once.
   The greedy half of each pair is the walk or thought already made for the
   price:
   - the reconstruction read-out: one explore walk, kept and credited on
     $R$;
   - the thought: one explore thought, priced by one greedy output walk of
     its answer (none where the answer's idea is unchanged), kept and
     credited on $A$;
   - the output read-out: one explore walk, kept and credited on $A$. The
     kept walk is what is presented.
4. One reader where `answerSynthesis` is true.

Walks: 2 or 4 for the compositions, then 1, 0 or 1, and 1: 4 to 7, with 2
or 3 thoughts. Today: 22 walks and 2 thoughts.

*Open for the specification* (nothing is built, and no Codex text exists):

- *The thought before the commit.* `_commit_sentence` writes the sentence's
  rows and then runs `ThoughtClosing.close`. Pricing a composition through
  its thought needs that order made safe for a trial that is then rejected.
  Round 7 is changing this code (the thought writer, the episodic entries,
  the clocks), so this part is specified against round 7's tree.
- *The question conditioner and the attention read* (§8.4, §8.7). With one
  reader, the conditioner that never trains is the only one left. Whether
  it is repaired or goes, the thought then carrying the question, is a
  choice for the specification, as is the attention read's place.
- *What the reader trains on.* Today the presented reader takes the kept
  trial's read at the sentence step, the kept thought's read and the
  batch-end output.
- *The reconstruction read-out's credit.* Each compose trial's inner pair
  credits the generate policy today (two samples a sentence); under the
  decision the kept composition's pair does (one).
- *Gates.* `XOR_grammar`'s certificate (its twin stays, but its decoder
  pair changes), the lesson acquisition panel, the sweep.
- *Who builds it, and when.* Alec, 2026-11-11: "Codex, after round 7". Claude
  writes the specification and keeps to the exact item 1 steps on the
  branch.

### 8.10 The twin goes everywhere (Alec, 2026-11-11)

Alec, after §8.9: "I am concerned that "XOR_grammar's twin" is a workaround
that was added because something in the architecture failed to learn
appropriately during testing."

The record bears him out.

- *What failed* (operators update §15 to §23, 2026-10-06). With the answer
  read by one affine reader trained on the kept composition alone, the
  answer cost did not move the compose chooser: a composition the reader
  had not trained on read badly, and the kept one was "the chooser's
  prior". Trained on both compositions the reader lost its fit (class 2 of
  10), the other composition's roots being unreadable by an affine map. The
  twin gave the comparison its own copy of the answer path.
- *The cause was withdrawn the next day* (operators update §27,
  2026-10-07). With identity by construction "Both projected conjunction
  and disjunction roots are affinely separable on the gate sentences"; the
  floor "was an artifact of the dense max-joined codes"; the expectation
  that the answer prefers conjunction "is withdrawn; correct behaviour is
  indifference". Both of the twin's reasons had gone from the gate it was
  built for. It stayed.
- *Today* (checked on landed main): in `XOR_grammar.xml` `word_brackets` is
  true, so `comparison_reader` is an `AnswerComparison` holding 22 tensors
  (the answer attention, `answer_record_reader`, the intuition `pi` and
  `sigma` of the conceptual spaces, the output space's read-out), and its
  costs leave the presented reader's within three batches.
  `separate_chooser` is false there since item 2.0, so the answer's
  gradient reaches the understanding, which it did not on 2026-10-06 (2.0
  spec §2).
- *The gate is not healthy with it.* The grammar XOR certificate converges
  in about two thirds of initializations (item 2.3 spec §33.1, finding 14;
  cause not diagnosed). Whether the twin's rule takes part is not known:
  the presented reader trains on the kept composition alone while the
  chooser is meant to be indifferent.

*Claude's error.* §8.9's second decision followed a recommendation to keep
the twin in `XOR_grammar`, made to leave that gate untouched and without
re-reading §27.

*Decided (Alec, 2026-11-11, question tool):* "Retire it everywhere
(Recommended)". This replaces the second decision of §8.9: one reader in
every configuration, and `AnswerComparison` goes. The round measures the
grammar XOR certificate without it over enough unseeded runs to read, and
what then fails to learn is diagnosed, not covered.

*The specification* of the round (Claude, 2026-11-11) is
[one pair for each stage, one reader](2026-11-11-one-pair-per-stage.md),
with its Codex text in §8 there, to be given after round 7 has landed. A
toy run first is in `doc/benchmarks/2026-11-11-one-reader-toy/`.

### 8.11 Whole paths replace a keep for each stage (Alec, 2026-11-11)

§8.9's first decision is replaced. Alec asked whether "we decide which is
kept at the end of every stage", and whether to "consider only which path
to keep across all stages". Claude answered that a whole-path keep would be
right only "if the stages' choices interacted, or if an explore could not
reuse its trunk", and that neither held. Alec: "The compose exploration
might be lost at the initial stage because the keeping acts like a local
minimum on the error surface, whereas if we have one departure per sentence
our minimum is evaluated only at final stage over all stages. It will
probably be slower, but more thorough. If we wish to increase the credit,
we might introduce a second explore path."

He was right, and §8.8's pair for each stage rested on the same error as
Claude's answer. The stages are a chain: the reconstruction read-out's
words become the
order-zero field (`OutputOccurrences.generated`, `SentenceField.publish`),
and the thought reads that field (`_selected_thought_memory`) and writes
into it (`Intuition.write_knowing`). Reconstruction and output do not split
after compose. A keep of the reconstruction walk on $R$ alone, as the first
specification had it, is a keep at a local minimum. §8.8's walk counts for
a pair at each stage were also too low: with each explore's whole tail run
again it is about 8 walks and 4 thought episodes a sentence.

*Decided:* "Yes, default one explore". One exploit path and explore paths
of one departure each, anywhere in the chain, compared once at the end on
$R + A$; `architecture.explorePaths`, default 1. The
[specification](2026-11-11-one-pair-per-stage.md) is rewritten for it
(§1.1 there has this record; the file keeps its first name). §8.9's other
decisions and §8.10 stand.

### 8.12 Silence counts, and output follows from the thought (Alec, 2026-11-11)

Alec: "Perhaps output in cases where there is no supervised answer should
supervise silence (i.e. penalize any answer). That would seem to teach both
answering questions and not talking when no question was asked." Decided:
"Yes, count silence". Silence was already a target for the reader's
training on a statement (`_answer_trial_cost`: "A counts asked rows; the
registry retains silence for output training"); it now counts in the keep
and the credit as well, so every row of data that supervises output has an
$A$.

On Claude's caution that silence must come from an empty answer and not
from a suppressed walk, Alec: "all that needs to happen is that thinking or
the intuitive step reduce the understanding to zero (i.e. after
comprehension). Output is never a direct consequence of understanding the
question; it is a consequence of the answer produced by thinking about the
question." `_resolve_answer` already forms the answer so for a row whose
episode it is given (the concluded answer when the episode is accepted,
zero otherwise). What contradicts it today is the default for a row with no
episode, the field's own meaning as the answer (`identity`), and the
trial reads of `_sentence_reader_error`, which unfold the sentence's
meaning (§8.4). Both go at a sentence close
([specification](2026-11-11-one-pair-per-stage.md) §3.1).

On text that supervises no output, Alec: "The only difficulty with this
supervised training regime is that most textual data is not dialog, so we
cannot use predictive processing to do model building for all of it. But
perhaps degraded input reconstruction can serve the same purpose?"
Claude's reading: a degraded sentence is a question the text answers
itself; the hole is an open slot for the closing thought and the restored
part is the answer's target. Not specified; it goes with item 0.5's
corpora. How the existing input mask (`create_ir_mask`) reaches the
sentence loop was not traced.

### 8.13 What is degraded today, and where the model's numbers sit (Claude, 2026-11-11)

Alec asked where the world model that does the reconstruction would mostly
live (percepts, order-zero concepts, higher-order concepts), and whether
lossy reconstruction is parameterized enough to be used as a teacher.
Traced for the answer, on the landed main:

- *Nothing is hidden on the sentence path.* `create_ir_mask` is the
  "Whole-slab IR mask (BERT-style hide-a-token)": Bernoulli masking of
  percept slots at `maskRate` (default .15), with the loss at the masked
  slots, and it runs only on the path without the per-word loop. On the
  grammar path `gaussian_window_word` or `word_span_window` replaces it,
  and it "is NOT target-hiding"; the reconstruction there reads "the
  COMPLETE UNMASKED" input. `maskRate` on that path is only the width of
  the gaussian context. This corrects §8.12's "not traced".
- *The lesson model's 9.86 million trained parameters:*
  `candidate_attention` 6.47 million (65.6%); the percept forms
  (`perceptualSpace._owned_bases.what`) 1.44 million (14.6%);
  `symbolSpace.subspace.layers` 1.10 million (11.1%); the three
  `concept_predictor`s .15 million each; `percept_expectation` .14 million;
  the generate policy 11 thousand.
- *Its largest buffers* (written by rules, not by gradient):
  `symbolSpace.subspace.layers` 13.7 million elements;
  `conceptualSpaces.0.layers` 5.1 million; the priming bridges about 4.9
  million over the three conceptual spaces; the percept store 1.4 million.
  What each of these tables holds was not opened.

### 8.14 An adaptive loss over the input, by omission (Alec, 2026-11-11)

Alec: "My guess is that we will want to create an adaptive loss function
over the input to teach the model before item 0, so minimally a maskRate
for both forms and words. It ought to be able to drop both percepts and
concepts. Ensuring a high, long-term reconstruction rate (e.g. 90%) would
allow learning while at the same time creating an raising bar for
prediction." And: "The kind should be omitted using an out-of-band signal,
not corrupted, because the best model reproduces corruption on seeing
corruption. Do we have an all-zeros vector for that purpose still? we used
to have such a mask."

*The all-zeros mask is still in the code.* `create_ir_mask` writes it into
the WHAT slice of a masked percept slot and keeps the slot's WHERE and
WHEN, for the meronomy percept space, which has no NULL row ("MASK is the
all-zeros vector"). The embedding percept space uses a learnable
`__NULL_PERCEPT__` row instead (`null_percept_idx`), which is a learned
code and not out of band. Two limits: it runs only on the path without the
per-word loop (§8.13), and an all-zero WHAT is also what a padding slot
carries, so a masked slot differs from padding only by keeping its place.

*Claude's reading.* The out-of-band signal is a slot that has a place and
no content. One order up it is a located word whose concept evidence is
zero on both lanes: "not known", which the two lanes keep apart from
"against". The closing question's open slot is the same thing at the level
of meaning. Nothing drops a word on the sentence path today.

*Decided (Alec, 2026-11-11):* the reconstruction rate held is "Of
everything" in the input, not of what was hidden. For a target rate $r$
(his example .9) and a restoration accuracy $a$ on hidden items, the share
hidden is then $m = (1-r)/(1-a)$: a tenth while nothing is predicted, a
fifth at $a = .5$, a half at $a = .8$. It limits itself, because $a$ falls
as more is hidden.

*Not specified.* Two rates, one for forms and one for words, each adapted
to that rule over the long term; what counts as restored at each level;
and where the restored part is the target (§8.12). It goes before item 0,
with the degraded-input regime.

### 8.15 Omission specified, inside the whole-path round (Alec, 2026-11-11)

Alec: "ask your questions: I want to spec this before the gym". His
answers (question tool): what was hidden is restored by "Reconstruction
only (Recommended)"; a dropped word loses "Its whole form (Recommended)";
the omission is "On by default, but you can override to get existing tests
to pass"; and it is built "Inside the whole-path round". This replaces
§8.12's reading that the restored part would be the answer's target: the
output stays silent on text that asks nothing, and reconstruction's target
is the input before the omission. The rules are §3.11 of the
[specification](2026-11-11-one-pair-per-stage.md), with four certificates
and a third lesson arm; the general rate is
$m = \operatorname{clip}((v-r)/(v-a), 0, m_{\max})$, which hides nothing
until the visible reconstruction rate $v$ passes the target.

Then: "One further thought. Since degraded input trains reconstruction
only through system1, we need a way for the monologue corpus to train
system 2. thoughts?" The record answers it twice, with the next sentence
(2026-09-15; 2.0 spec §8: "System 2 is judged by expectation against the
next input's order 0, and reconstruction by the current input"). Traced:
it is not built for the thought. On text that supervises no output a
thought's trials cost nothing, and expectation's error credits no
departure. It is an open item in the specification's §6 until Alec says
which round builds it.

### 8.16 No next-sentence judge: reconstruction through compose and generate (Alec, 2026-11-11)

§8.15's open item is closed. Alec: "No, I do not think predictive
processing is a good solution; reconstruction is better... So to train
system2 on that, we might do reconstruction as augmented by compose() and
generate() without doing any thinking()? So call generate() on the LTM0
idea to alter the order-0 conceptual contents before doing reconstruction?
Does that make sense? otherwise we are not using our higher-order
understanding in order to reconstruct, which does not feel psychologically
valid."

It is what the order-zero reconstruction already is, since his correction
of 2026-11-10 (loop spec §9) and in the code: `SentenceField.reconstruct`
runs the generate walk on the composed end state (`sentence_generated`),
the walk's words become the order-zero field (`OutputOccurrences.generated`),
and `reverse` renders it against the input (`reconstruction_order_zero`).
So under omission System 2's compose and generate take part in the
restoring and are trained by it; only the thought is outside. The question
tool's option of §8.15, "Expectation and intuition (System 1) do the
restoring", understated this and was Claude's wording.

Specification §1 decision 9 and §3.11: the order-zero reconstruction is
costed against the input before the omission at every place; the input
reconstruction (`reconstruction_reverse`), which inverts the observed field
without the grammar, is costed only where content was observed (Claude's
choice, for Alec's yes). The next-sentence judge is recorded there as
considered and not to be built.

Alec then confirmed the arrangement in his own words and answered the
choice: "whole path is an integration of the reconstruction and output
paths, both of which share the input path but do not share the same
reverse (or at least call the same reverse mechanism from two different
conceptual content spaces)"; and "we want to compare the order-zero
reconstruction as augmented by higher-order-understanding but without
thinking". He also corrected a table of Claude's: generate is taught by
the output path as well as by reconstruction. Claude's statement that the
batch end "falls back to the sentence's own meaning when no thought was
passed in" described today's `_resolve_answer` ("Each completed field
supplies its full semantic operands as the answer seed"), which the round
removes; it is not the design.

### 8.17 The XOR test tests the design (Alec, 2026-11-11)

Reviewing the specification before its hand-off, Claude found that decision
7 would leave `XOR_grammar.xml` nothing to learn from: its closing thought
concludes nothing in 48 of 48 sentences and its affine head reads the
composed state. Claude proposed to keep a numeric head outside the rule.
Alec: "We have converged on a model design at this point: so the purpose of
the xor test is to test that design on the xor problem, not to make a new
model architecture".

So no read-out is kept or added for the gate. The grammar XOR gate keeps
its four sentences, classes and compose grammar, and is posed on the
design: the class supplied as an answer word, concluded by the thought and
said by the output walk ([specification](2026-11-11-one-pair-per-stage.md)
§1 decision 10 and §3.12). The numeric certificate
`test_xor_class_accuracy` becomes four exact answers. It is likely red at
first, since the lessons answer 0 of 320 on this design; the return then
says which stage does not learn. The other XOR proofs and numeric heads
elsewhere are not decided (specification §6).

### 8.18 A continuous output is a modality: the XOR head reads the concepts with no grammar (Alec, 2026-11-11)

Superseding the last paragraph of §8.17. Asked whether a red XOR test stops
the round, Alec: "Yes, it stops the round. A continuous output (and
continuous input) can be treated as alternate modalities. Using a linear
output head on top of order 0 concepts instead of verbal reconstruction can
be a model option. So with that caveat, you can keep xor close to its
current form."

Claude read "order 0" literally and measured it on round 7's working tree
before specifying (one unseeded training that converged; thirteen
checkpoints). The order-zero field that generate unfolds from the composed
understanding holds exactly the two words' concepts at amplitude 1, so
every linear read of it is a sum over the two words, and XOR is not such a
sum; the fixture's thought concludes nothing in 48 of 48 closes (it
declares a `<compose>` section and no thought entries), so the output
path's order-zero concepts are empty; the composed root does carry the
combination ($|x_1 - x_2 - x_3 + x_4|$ is 0 at the first batch and 1.37 to
1.98 afterwards, norm 8.2). Put to Alec: "Sorry; when I said order-0, I
meant no grammar (higher-order concepts are in fact required for xor)."

So the continuous output is a linear head on the concepts, higher orders
included, with no generate grammar, as a model option;
`XOR_grammar.xml` keeps its numeric classes, `answerSynthesis` false and
the head `_forward_head` has now, on each path's composed sentence state;
the twin goes there as everywhere. The gate stops the round: ten unseeded
trainings, five or more passing (the count is Claude's; the certificate
passes about two initializations in three today, item 2.3 spec §33.1
finding 14). What a continuous head reads where a thought does conclude,
and continuous input, are not decided
([specification](2026-11-11-one-pair-per-stage.md) §1 decision 10, §3.12
and §6).

### 8.19 The episodic entry holds concepts, not percepts (Alec, 2026-11-11)

Alec: "So does reconstruction really need to go through compose() and
generate()? Only if the ideas stored in LTM are the post-compose ideas.
That way it can talk about its memories. But if the memories are just
zero-order concepts, it can do direct reconstruction. This is how animal
cognition works, but presumably multiple concepts are stored per time step.
So perhaps animals should store concepts instead of percepts, or are the
former sufficient? Is there psychological evidence?"

*From the code* (round 7's working tree). The two reconstructions are
separate costs: `reconstruction_reverse` from the order-zero open field
with no grammar, and `reconstruction_order_zero` through compose and
generate. A sentence stores two things: the composed idea in its LTM0 row
and, beside it, up to four basic percepts of the open attention pass. The
grammar route trains generate to invert compose, so that a stored idea can
be said; a recalled order-zero memory can also be talked about, by being
understood again. And `EpisodicMemory.reinstate` already turns the kept
percepts into an order-zero concept field before anything reads them.

*From the literature* (Claude, from memory; not fetched again for this
entry).

- The store is an index over high-level codes, not raw percepts:
  hippocampal indexing (Teyler and DiScenna 1986; Teyler and Rudy 2007);
  its input is object-level, perirhinal cortex coding objects as
  conjunctions of features (Murray and Bussey 1999; Bussey and Saksida
  2002); reactivating the index cells recalls the memory in mice (Liu et
  al. 2012).
- Several items are kept for a moment, bound with place and time: scrub
  jays (Clayton and Dickinson 1998); rats remember many items in context
  and replay their order (Panoz-Brown et al. 2016, 2018); about four
  objects are held at once (Luck and Vogel 1997; Cowan 2001).
- Percept detail is reinstated through the concept and fades before the
  gist: remembering reactivates sensory cortex (Wheeler, Petersen and
  Buckner 2000); surface form is lost while meaning stays (Sachs 1967);
  detailed visual memory rests on conceptual distinctiveness (Konkle,
  Brady, Alvarez and Oliva 2010).
- Percepts alone are not nothing: pigeons and baboons memorise thousands of
  pictures (Vaughan and Greene 1984; Fagot and Cook 2006). That gives
  recognition of the same view; inference over stored relations needs the
  hippocampus in rats (Dusek and Eichenbaum 1997), and prior schemas speed
  new learning (Tse et al. 2007).

So: concepts as the key, percepts reached through them. Alec: "This is
actually current work: it will clarify the architecture to store concepts
instead of percepts. That will allow us to do animal cognition (system 1
reconstruction) without calling compose and generate." And: "So that would
be an immediate change."

Specified as [specification](2026-11-11-one-pair-per-stage.md) §1 decision
11 and §3.13, certificates 13 and 14, and item 16 of its Codex text, given to
Codex as a mid-round update (its §8.1; round 7 had landed as `a39cbfdef8`
and the build had begun): `EpisodicMemory.stage` keeps up to
`episodicConcepts` (4) order-zero concepts of the open pass with their
poles and brackets; `reinstate` publishes them as a field directly;
`BasicModel.reverse` of that field is the recalled moment's reconstruction
with no compose and no generate. Three consequences are recorded there: a
percept with no concept leaves no memory; the entry is the reading of its
moment; the exact witness is not kept. A configuration that runs with no
compose and no generate at all is not built in the round
([FutureWork](../FutureWork.md#numeric-input-as-a-modality-xor-as-a-system-1-model-and-one-model-class-alec-2026-11-11-to-specify)).

*Amended the same day (Alec).* Codex built the entry with each concept's
pole pair and bracket. Alec: "We should be storing order-0 concepts, not
"brackets": what even are brackets?"; "I think .when and .where were useful
when LTM stored percepts, but if we are not storing percepts, we have no
need of those fields."; "a single signed value can be kept, as an
approximation of the pair; dharmakirti says that contradiction is not a
valid mental object, so presumably contradiction is a working-memory only."
The entry is now references in reading order, each with one signed value;
a review lays its readings out from their order
([specification](2026-11-11-one-pair-per-stage.md) §3.13 as amended and
§8.2). The signed value is for these stored references only (Claude's
reading); elsewhere the pair of two-truths §1.1 stands.
