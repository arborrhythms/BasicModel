# Item 6.9: XOR_grammar, the baseline of grammatical learning

> **Status:** plan, written by Claude on 2026-09-29 from Alec's request of
> that day and from probes run that day on a copy of the working tree (item
> 7's candidate, round 3, copied at 14:59). Nothing in the repository was
> edited for the probes. Alec answered the six questions of §6 on
> 2026-09-29 and 2026-09-30. **Ready for Codex** after item 7 is accepted
> (the hand-off is §8).
> **Updated 2026-10-01:** stopped after step 5 with 0/10 at the bar; measured
> (§3.13); decided by Alec: the answer's error trains the whole path for a
> sentence with a supplied answer (§4 step 5a, §6 question 7). Codex continues
> with steps 5a to 9 (hand-off §9).
> **Reframed 2026-10-04** (Alec: the XOR test "is kind of meaningless for
> grammar (on an untrained model)"): the XOR table is the **composition
> mechanism gate**, not a grammar gate; grammatical learning is measured by
> the wording lessons now and by item 9's evaluation on item 0's checkpoint
> ([6.8 plan §12.1](2026-09-27-item-6-8-one-attention.md#121-the-composition-gate-not-a-grammar-gate-alec-2026-10-04)).
> **Closed 2026-10-03** (Alec: "do you accept" — yes), on the baseline of
> §25.3 and §26: one decoder for reconstruction and output; one writer per
> weight; product, mean and `not`; the free read-back with the antipode
> term; the gates red in the closing run with their causes recorded, the
> §22 measurement as the prior; the no-regression rule for later items;
> the catalog (§20.3). Rounds §10 to §26 are the history. Alec's two
> closing comments are in the [6.8 plan §9](2026-09-27-item-6-8-one-attention.md#9-carried-from-item-69-alec-2026-10-03).
> The item is taken after item 7 is accepted and before 6.8. The
> rest of the [operator catalogue](../specs/2026-09-29-operator-catalogue.md)
> follows it, so that the operators change under the protection of this
> baseline.

## 1. What is asked

Alec, 2026-09-29: "Since we have done this much of the spec, let's make
item 6.9 an effort to fix xor-grammar. In particular, you mentioned 'That
the pressure is enough has not been shown. XOR_grammar has conjunction,
disjunction and not available and answers one half to everything. It is
already the gate of this update, so it is the test.' Perhaps we need
intersection/union? But it seems we should refine enough of the grammatical
operators to get that baseline working, so that we can prevent any
grammatical regression."

Earlier the same day XOR was called "a basic proof of nonlinear learning"
that is not to regress, and XOR_grammar's two gates were made the gate of
the operators update
([two truths §15.4](../specs/2026-09-16-two-truths-ideas-and-relations.md#154-questions-for-alec-and-his-answers-2026-09-29)).

**Exit.** XOR_grammar's two gates pass without a seed: the class gate at
the bar decided on 2026-09-29, all four answers right with an error below
.05, and the reconstruction gate with every sentence recovered as its own
words, a transposition being the only error admitted (§6 question 4); the
negative control of §4 step 8 fails, as it must; and every other XOR
proof, the slow MM_20M_xor exact round trip named among them, and
MM_grammar's ten-run table are no worse than at the start of the item.

## 2. The fixture

`data/XOR_grammar.xml` reads four sentences: "hello world" is 0, "hello
there" 1, "loving world" 1 and "loving there" 0. The reading is serial,
with the mixing binding. The grammar has three rules, `not`, `conjunction`
and `disjunction`. There are six concept rows of ten numbers. The answer is
one number, a linear map with a bias, trained for 400 epochs. The fixture's
comment names "not, intersection, union" as the primitives of XOR's
disjunctive form; its rules were changed to `conjunction` and `disjunction`
on 2026-05-02 (`6121f171`).

Its two gates are in `test/test_explicit_dimensions.py`, both slow. **Class
accuracy** asks for at least half of each class right. **Reconstruction**
asks for at least two of the four sentences recovered word for word.

## 3. What was measured

### 3.1 The class gate reads a placeholder

`TheReport.mnistReport` returns a one-element zero for every configuration
whose data type is `embedding` (`bin/visualize.py:277`, since 2026-05-13,
`169d119a`): "Skip for IR-only text models". XOR_grammar's data type is
`embedding`. So `model.rCorrect` is `[0.0]` whatever the model answers. The
gate fails on the first class before it reads an answer, and there is no
second class. It cannot pass. Its expected-failure mark was removed on
2026-09-21 (`d4dc385f`), and it has failed in every receipt since. It is
HEAD's too: `visualize.py` is unchanged in the working tree.

The model does print its four answers, in the report's lines with `label=`
and `predicted=`. The probes of this plan read those.

### 3.2 Read correctly, the bar is met by chance

"At least half of each class" is two answers right, one in each class. An
answer that is a coin flip meets it nine times in sixteen. The test's own
history says so: the bar was set at ".5 (random baseline)" so that it would
not "fluctuate with seed luck". It would not detect a regression of the
grammar.

### 3.3 The answer does not read what the grammar composed

The answer is read from the conceptual slab, one concept for each position
of the sentence, flattened, through the linear map
(`Models._forward_head(body_sub)`). The code says so: "The single S is
PRODUCED and verified here ... but NOT yet consumed by the loss". A linear
reading of a slab of words adds one contribution for each word. XOR is not
such a sum, so the best this answer can do is one half everywhere, which
is what every receipt has recorded.

### 3.4 What the answer's error trains

Over the gate's own 400 epochs, the answer's error reached three
parameters: the linear map, its bias, and the input vocabulary. It reached
none of the grammar's, the chooser's three anchors, and none of the
conceptual spaces'. The sentence cost of this configuration has no
reconstruction term, since the tied traversal is off here. What trains the
chooser is the expectation cost, in the 800 backward passes of the sentence
trials.

### 3.5 What the grammar is given for a word

In the serial loop of this configuration the leaf pushed for a word is the
word's own conceptual event. It carries no concept identity (row −1), and
the replacement of a word by its object never happens (object row −1, in
every push). Of its ten numbers, eight are the position bands and are the
same in every sentence; the word is carried by the first two. Each sentence
gives the grammar three leaves, not two: the space between the words is
pushed as a unit.

So the four roots can differ in two numbers at most. After the gate's
training they differed in the first word only (singular values of the
centred roots .135 and .002), and the best linear reading of them, with a
bias, answers [.5, .5, 1, 0], error .125.

### 3.6 The operators are not the obstacle

In XOR_grammar with its rules changed (probe copies of the fixture, two
runs each): `intersection` and `union` in place of `conjunction` and
`disjunction`; all five together; `lift` and `lower`. Every run answers
one half.

In isolation: the repository's operation-selection layer, chooser and
operators, built as XOR_grammar builds them; the four words given as codes;
the answer a linear reading of the root with a bias; the answer's error the
only objective; 400 epochs; ten unseeded runs of each.

| the words arrive as | operators | learned (error < .05, all four right) |
|---|---|---|
| two numbers in [0, 1], learned | not, conjunction, disjunction | 1 of 10 |
| the same, fixed | the same | 0 of 10 |
| the same, learned | not, intersection, union | 5 of 10 |
| the same | conjunction and disjunction reading zero as no evidence (two truths §1.1) | 3 of 10 |
| ten numbers in [0, 1], fixed | not, conjunction, disjunction | 2 of 10; 10 of 10 at 1,000 epochs |
| ten signed numbers, fixed | not, conjunction, disjunction | 10 of 10 |
| the same | not, intersection, union | 10 of 10 |
| the same | not, lift, lower | 10 of 10 |
| the same | not, sum, product | 9 of 10 |
| the same | sum alone | 0 of 10 |
| the same | conjunction alone | 8 of 10 |
| the same, the root cut from the answer as decided (§4 step 2) | not, conjunction, disjunction | 8 of 10 |

Once the words arrive as full codes, one binary operation that is not a
sum makes four roots that a linear reading can tell apart, whichever of
these operators it is. A sum cannot, being one contribution for each word.
Two numbers a word are too few, whatever the operators.

### 3.7 The two changes together, in XOR_grammar

Two probe patches, in XOR_grammar itself: the answer reads the root (§3.3),
and the leaf of a word is its object's code from the concept dictionary
(§3.5).

| | runs | error | all four on the right side | half of each class, as the gate is written |
|---|---|---|---|---|
| 400 epochs, the fixture's | 4 | .17 to .21 | 1 | 4 |
| 1,000 epochs | 3 | .04, .14, .25 | 2 | 3 |
| nothing changed, 1,000 epochs | 2 | .25, .25 | 0 | 2 |

The last row is §3.2 at work: answers of one half everywhere, .49 to .51,
meet the bar as it is written.

After training, the four roots are in general position (singular values of
the centred roots .70, .38, .12) and a linear reading of them with a bias
fits XOR exactly (error 0.0). What remains is the reading's convergence:
the grammar's choices, trained by expectation, and the concept codes,
rotated by distribution, move while the answer's map learns to read them.
At the bar of §6 question 1, error below .05 with all four right, it was
met in none of four runs at 400 epochs and in one of three at 1,000.

### 3.8 The reconstruction gate does not pass through the grammar

During training and during the gate's reconstruction, the grammar's
reverse was not called once: not the reverse program, not its bounded
search, not the reverses of `conjunction`, `disjunction` or `not`. What the
gate reads is PartSpace's rendering of the model's reverse of the per-word
state, each word stamped at an offset decoded from its `.where`; hence
answers such as "  world", with leading blanks. The reconstruction gate
tests perception's reverse, not the grammar's.

### 3.9 The grammar's reverse, with no witness, blends the two orders

Measured on the bounded search itself (four random codes of ten numbers,
ten trials of the four sentences): for each of `conjunction`,
`disjunction`, `intersection`, `union`, `lift`, `lower`, `product` and
`sum`, the pair of least residual was the right pair of words in 40 of 40
cases, and the search returned it in 0 of 40. Each of these operators is
symmetric, so a pair and its reverse have the same residual. The search
returns a weighted mean over pairs, which gives both operands the same
half-and-half blend of the two words.

### 3.10 Two smaller things

* In the closing rounds the chooser applies `not` again and again, up to
  sixteen times: `not` is always admissible and it is preferred to
  stopping. As `not` is its own inverse, what is left depends on the parity
  of the number of rounds.
* `not` exchanges the first two numbers of whatever it is given. For a pair
  of poles that is the decided `not`. For a concept's full code it exchanges
  two arbitrary coordinates.

### 3.11 Whether the bar can be reached at all

Alec, of the bar: "Yes, but let's make sure it's theoretically possible."

The answer is a linear map, with a bias, of the understanding. It can give
the four targets exactly if and only if the pattern 0, 1, 1, 0 lies in the
span of the four understandings and a constant. If the understanding is a
sum of one contribution from each word, it never does. The answer would
then be *f*(first word) + *g*(second word) + *b*, so "hello world" and
"loving there" together would equal "hello there" and "loving world"
together: 0 on one side, and 2 on the other. The same holds of an
understanding in three slots with a word in each, read by a linear map; XOR's
sentences are ideas, and end in one slot.

Checked by exact algebra, with no training: least squares in double
precision, a dependency among the four understandings counted as such.

| the grammar composes | operation | its choices | exactly reachable |
|---|---|---|---|
| object concepts: each word its object's code, ten signed numbers | minimum, maximum, the coded `intersection` or `union` | the same for every sentence | for 300 of 300 random codes; with the space as a third leaf, as in the fixture, for 290 to 300 |
| the same | sum | the same | for none |
| symbols: each word the presence of its object's symbol | any of these, with `not` anywhere | the same for every sentence | for none |
| the same | maximum (`disjunction`), or minimum reading zero as no evidence | the negation of the second word chosen by the first word | in 2 of the 4 ways of choosing |
| the same | minimum (`conjunction`) | the same | never: two different words have no presence in common |

So the bar can be reached when the grammar composes object concepts, as
Alec decided (§6 question 3), with any of the four operations and the same
choices for every sentence. The answer's map does the rest, and §3.6 and
§3.7 measured it being learned. The bar cannot be reached by symbols
composed the same way for every sentence. Symbols reach it only if the
grammar's choices differ with the words, which the chooser would have to
learn from the answer; and the answer's error does not reach the chooser
(§3.4). That the bar can be reached does not show that it is reached within
the fixture's 400 epochs, which is §4 step 5.

Whether it is learned, in isolation (the repository's operation-selection
layer and chooser; the answer a linear reading of the root; 400 epochs;
ten unseeded runs each; "cut" is the cut of 2026-09-20, the answer's error
stopping at the understanding):

| the slots hold | operations | the answer's error | learned |
|---|---|---|---|
| object codes | `conjunction`, `disjunction`, `not`, as coded | cut | 8 of 10; 8 of 10 with a silent third leaf, as in the fixture |
| symbols (presences) | the same, `not` exchanging each concept's poles | cut | 2 of 10; 2 of 10 with the third leaf |
| symbols | the same | reaching the chooser | 6 of 10 |
| symbols | `conjunction` and `disjunction` reading zero as no evidence | cut | 6 of 10; 6 of 10 with the third leaf |
| symbols | the same | reaching the chooser | 10 of 10 |

With symbols the answer depends on choices that differ with the words. The
chooser makes such choices even untrained, since its scores read the
words, but it learns the right ones only when the answer's error reaches
it. Conjunction as coded makes it worse: the conjunction of two different
words' presences is empty, and an empty result gives the chooser nothing to
learn from.

### 3.12 What the answer is shown in training

Alec, 2026-09-29: "It may be that the conceptual understanding necessary
for producing the right xor answer is produced infrequently, and that said
frequency is not sufficient to train the understanding-to-correct-output
path."

Measured with the two patches of §3.7, over the gate's own 400 epochs, at
every call of the answer:

* **The understanding is never degenerate.** In every epoch the four
  understandings could each be read right by some affine map (the third
  singular value of the centred understandings never fell below .01; its
  median was .15).
* **But it is not the same understanding from one epoch to the next.** It
  moved by a median of .12 per sentence between consecutive epochs. One
  reading fitted to all 400 epochs leaves an error of .21, and to the
  first quarter .22; fitted to the last quarter alone it is exact. The
  answer's map is chasing a target that settles only near the end.
* **What it is shown is usually not what it is tested on.** Each sentence
  is read twice in training, the greedy derivation and an explore
  derivation that differs from it in one action chosen at random, and the
  cheaper of the two is kept and read by the answer. Evaluation reads the
  greedy derivation only. In training the explore derivation was kept in
  85% of rows with the patches, and in 67% in the fixture as it is.
* **The comparison is not between the derivations.** The second trial is
  costed after the optimizer has already stepped on the first
  (`SentenceCompose.sentence_pair`). In a control where the second trial
  repeats the greedy derivation exactly, it still "won" in 93% of rows
  with the patches and 88% without. So the understanding kept is chosen
  mostly by that step, and not by which derivation is better, and a
  sentence's understanding changes from epoch to epoch with the random
  action of the explore trial.
* Freezing the chooser, or the concept codes, did not stop the movement
  (one run each). Keeping the greedy trial always settled the understanding
  at the end, but in that run the chooser, still trained by expectation,
  had drifted to a derivation that lost the second word.

So Alec's reading holds in this form: the understanding the answer must
learn to read, the one evaluation uses, is shown to it in a minority of
training steps, and what it is shown instead is different every epoch. The
bias of the comparison is item 7.5's, and it is in every serial
configuration, not only this one. Claude accepted item 7.5 in review
without seeing it.

### 3.13 After step 5: what holds XOR back (measured 2026-09-30 and 10-01)

Steps 1 to 5 landed and the bar was met in 0 of 10 runs
([receipt](../benchmarks/2026-09-30-item6-9/README.md)). Step 5 works: the
explore trial is now kept in about 4% of rows, and the understanding
settles by the end of training.

* **The understanding can be read, but only barely.** Claude's probe
  (`deriv69.py`, kept in the
  [October 1 receipt](../benchmarks/2026-10-01-item6-9/README.md)) rebuilt
  each derivation and tried all 128 full derivations of first word, space
  and second word with min, max and `not` over the trained leaves. The best
  derivation needs answer weights of 2.7 to 4.2 at the first epoch and 4.7
  to 13.4 at the end; the chooser's final derivations need 21 to 1,789; the
  trained answer map reaches about 5. Random signed codes would give 1.8:
  the codes drift away from what XOR needs.
* **Exploration is not what is missing.** The explore trials try 149 to 275
  different derivations per run and often produce a readable result, but
  the comparison of the two trials does not see the answer, so nothing
  keeps them.
* **What the answer's error must reach.** Variants measured on the repaired
  candidate, ten runs each
  ([October 1 receipt](../benchmarks/2026-10-01-item6-9/README.md)): the
  answer's error, detached, only choosing between the trials (a): 0 of 10;
  the same with four explore trials (b): 0 of 10; the answer's error in
  each trial's cost with its gradient, reaching the codes and the chooser
  (c): 9 of 10.
* **Not the configuration.** Trained as the gate trains, five runs each:
  MM_grammar as it is, 0 of 5; XOR_grammar with `nDim` 14, with
  `subsymbolicOrder` 3, and with all of MM_grammar's space settings, 0 of 5
  each. MM_grammar's ten-run table succeeds because its harness runs the
  raw forward with its own optimizer: the sentence trials never run, so the
  codes and the chooser keep their random start, where XOR reads easily.
  That table therefore cannot see a regression in grammatical learning.
  The fixture is stale in one respect: at `nDim` 10 an event carries two
  numbers of content, not the four its comment states. `nDim` 14 makes
  the understanding easier to read (weights 1.75 to 17) but does not pass
  alone.

## 4. The plan

In order. Steps 1 to 5 are what the measurements require; 6 to 9 make the
gates a guard.

1. **The gate reads the answers.** It is computed from the four answers the
   model gives, not from the report's placeholder (§3.1), at the decided
   bar: all four right, with an error below .05 (§6 question 1).
2. **The answer begins with the understanding.** "Answers begin with the
   understanding left in the 1 or 3 slot representation" (Alec,
   2026-09-29), and not with one concept for each word position (§3.3). It
   is also the design decided before: output matching "from a GIVEN
   concluded idea", with the "state cut at the concluded idea"
   (2026-09-20,
   [plan §8.4](2026-09-15-next-sentence-as-the-production-objective.md#84-gradient-boundaries-and-learning-evidence)).
   The cut stands: the answer's error trains the map that reads the idea,
   and not the grammar through the idea. Measured, the cut does not stop
   XOR being learned (§3.6, last row). *Amended 2026-10-01 (step 5a):* for
   a sentence with a supplied answer the answer's error also trains the
   grammar and the codes, since with the cut the fixture met the bar in no
   run (§3.13).
3. **A word reaches the grammar as its object's code, in every binding.**
   "The grammar operations are conducted over the object concepts, not the
   word concepts" (Alec, 2026-09-29). The leaf the loop pushes for a read
   word is its object's code, times its
   activation, as the answer-path plan's invariants 3 and 4 have it
   ([2026-09-14, §2](2026-09-14-answer-path-ownership-and-training.md#2-standing-invariants-re-read-after-every-compaction))
   and as two truths §17.3 has it for the published symbol: "The symbol
   published for a read word is its object's". Item 7 made the published
   symbol right. The leaf the grammar composes is still the word's own
   event wherever the loop has no object row, as in XOR_grammar (§3.5).
   *Read, not measured:* the loop's object rows are published only by the
   serial aligned reading.
4. **`not` on a code is the reflection through the origin.** Decided on
   2026-09-28: `not` maps `d → −d`, "just another point, not a region"
   ([accessible mind §2.6.2](../specs/2026-09-20-accessible-mind-subsystems.md#262-negation-exists-only-for-concepts-not-percepts-or-symbols)).
   On a pair of poles that is the exchange of the poles, as now. On a code
   in a slot it is the code's negation, and no longer the exchange of two
   coordinates (§3.10).
5. **The two trials are compared before either is trained on** *(decided,
   Alec, 2026-09-30: §6 question 6)*. Both derivations of a sentence are
   costed under the same parameters, and only then does the optimizer step
   on each; the cheaper is kept. It is done the cheapest way that keeps the
   comparison equal, and its cost in time and memory is measured against
   today's pair. Keeping a snapshot where the two derivations branch, so
   that the second need not recompute the first's prefix, is future work
   ([FutureWork](../FutureWork.md#a-snapshot-where-a-sentences-two-trials-branch-future-work-alec-2026-09-30)). The control of §3.12 becomes a test: when the second trial
   repeats the first derivation, it wins in no row. This corrects item
   7.5's sentence pair in every serial configuration, so MM_grammar's
   ten-run table and item 7.5's receipts are measured again. Then the
   answer's map must converge: steps 2 and 3 make the understandings
   separable (§3.7), and §3.12 says why the map has not settled. The
   fixture's 400 epochs and the bar are not changed to pass. If ten runs
   of ten do not meet the bar after steps 1 to 5, Codex stops and reports,
   and Alec decides. *(It stopped: 0/10, 2026-09-30; §3.13.)*
5a. **The answer's error trains the whole path** *(decided, Alec,
   2026-10-01: §6 question 7)*. For a sentence that comes with a supplied
   answer, each trial's cost includes the answer's error with its graph
   intact, so it reaches the map that reads the idea, the chooser, and the
   object codes the grammar composed, through that trial's own perception
   pullback. Both trials are still costed under the same parameters before
   either step (step 5), and the batch-end answer update stays. A sentence
   without an answer keeps the cut. This amends, for sentences with a
   supplied answer, the cut of 2026-09-20 and "codes by distribution, maps
   by the objectives" of 2026-09-21
   ([plan §8.4](2026-09-15-next-sentence-as-the-production-objective.md#84-gradient-boundaries-and-learning-evidence)).
   Measured as variant (c) on 2026-10-01: 9 of 10 runs met the bar.
6. **The reconstruction gate reads back through the grammar, and admits
   transpositions** *(decided, Alec, 2026-09-29; §6 question 4)*. Each
   sentence is read back from its understanding through the grammar's
   reverse. The grammar's operations are symmetric, so the order of the
   words is not in the understanding, and a transposition is the only
   error admitted: every sentence comes back as its own words, in either
   order; any other error fails the gate. Where the reverse searches among
   the known words, it returns the pair of least residual, which §3.9 found
   right in 40 of 40 cases, and not the blend it returns today; the soft
   weights are kept for the gradient.
7. **No operation undoes the one before it at the same place.** A unary
   whose reverse is the operation just applied at the same slot is not a
   candidate, so the closing does not spend its rounds exchanging poles
   back and forth (§3.10). *Claude's recommendation;* it changes no
   decided rule, since two such operations together are the identity.
8. **A negative control.** The class gate is run once more with the
   grammar's binary operators replaced by `sum` alone, which is additive;
   there the answer must stay at one half. If it does not, the gate is not
   measuring the grammar. In isolation `sum` alone learned 0 of 10 (§3.6).
9. **Receipt.** Ten unseeded runs of each XOR_grammar gate, with the four
   answers and the four reconstructions of every run; the negative control;
   the full XOR table, as for every receipt, with the slow proofs named,
   the MM_20M_xor exact round trip among them; and MM_grammar's ten-run
   table, since step 2 changes what its answer reads too.

## 5. Not in this item

* **The operator set.** Which operations the fixture's rules name is §6
  question 5; any of the four can reach the bar (§3.11).
* **Credit to the chooser from the answer.** The answer's error does not
  reach the chooser, by the cut of 2026-09-20, and the gate does not need
  it (§3.6, last row). Whether a sentence with a supplied answer should
  count the answer's error in the comparison of its two trials, which is
  credit by choice and not by gradient, is noted in
  [FutureWork](../FutureWork.md#credit-to-the-chooser-from-a-supplied-answer-noted-2026-09-29).
* **The word whole.** The space pushed as a leaf between two words is
  6.8's
  ([plan §3a](2026-09-27-item-6-8-one-attention.md#3a-the-word-whole-alec-2026-09-29-taken-up-with-this-item)).
* **Names.** "The operator rename can also be future work" (Alec,
  2026-09-29).
* **The clock inside the understanding** *(measured 2026-09-30)*. The
  clock's `.when` stamp is muxed into every idea, the chooser and the
  answer read it, and a trained MM_xor's answers drift with it: maximum
  error .061 at the end of training, .068 ten batches later, .240 a
  thousand later. Its removal is item 5.5
  ([spec §1.1](../specs/2026-09-30-occurrence-tense-aspect.md#11-the-clock-is-inside-every-idea-measured)).
  Until then this item's gates read the answers right after training, as
  now, where the clock has moved least; a gate run long after training is
  not evidence either way.

## 6. Questions for Alec, and his answers (2026-09-29)

1. *The bar.* **Decided:** "Yes, but let's make sure it's theoretically
   possible." All four answers right, with an error below .05. It is
   reachable when the grammar composes object concepts, and not when it
   composes symbols with the same choices for every sentence (§3.11).
2. *What the answer reads.* **Decided:** "Answers begin with the
   understanding left in the 1 or 3 slot representation." For XOR's
   sentences, which are ideas, that is the one slot.
3. *What the grammar composes.* **Decided:** "The grammar operations are
   conducted over the object concepts, not the word concepts."
4. *Reconstruction.* **Decided:** "no, the grammar is symmetric, so make
   reconstruction error a lower bar, but ensure that errors are
   transpositions. The output xor values should remain accurate to
   precision." Asked whether that means all four sentences: "All four".
   Every sentence comes back as its own words, in either order (§4 step
   6), and the class gate keeps the bar of question 1.
5. *Which composition.* **Decided:** "conjunction/disjunction, then, since
   meaning is not significant." The fixture's rules stay as they are.
   Asked whether `conjunction` and `disjunction` act on the object concepts'
   codes in the slots, as the code does now: "Yes". The bar is reachable so,
   and was learned 8 of 10 in isolation with the cut (§3.11). (Had they
   acted on the objects' symbols, the answer would have depended on choices
   that differ with the words, which the chooser learns only when the
   answer's error reaches it: §3.11.)

6. *The comparison of the two trials (§3.12).* **Decided (2026-09-30):**
   "Good justification for this: equal comparison. But let's opt for making
   this efficient; maybe even a snapshot at where they branch, if that
   helps significantly (the snapshot overhead may be high, in which case
   don't). Let's add this optimization as future work." Both derivations
   are costed under the same parameters (§4 step 5); the snapshot is in
   FutureWork.

7. *What the answer's error may train (§3.13).* **Decided (2026-10-01):**
   "Let's just implement the answer training the whole path, as you said,
   and have codex continue with all of 6.9." For a sentence with a
   supplied answer, the answer's error trains the reading map, the chooser
   and the object codes (§4 step 5a); the narrower variants (codes only,
   chooser only) are not measured first.

## 7. The probes

In the session's scratchpad (`xor69/`), not in the repository:
`run_gate.py` (the gate as the test runs it, reading the four answers);
`inspect_after_run.py` (the derivation and the roots after training);
`probe_grads.py` (which parameters each objective reaches);
`probe_pushes.py` (what the loop pushes, and which reverses run);
`probe_head_reads_idea.py` and `probe_leaf_code.py` (the patches of §3.7);
`unit_grammar.py` and `unit_reverse.py` (§3.6 and §3.9); `possible.py`
(§3.11); `probe_track.py`, `probe_exploit_only.py` and
`probe_freeze_chooser.py` (§3.12). That scratchpad was cleared on
2026-10-01; the probe of §3.13, `deriv69.py`, is kept in the
[October 1 receipt](../benchmarks/2026-10-01-item6-9/deriv69.py), with
Codex's correction of its stack window.

## 8. Hand-off to Codex (after item 7 is accepted)

Read §3 to §6. The rules of the item 7 hand-offs stand: each repair has a
failing probe saved before it; no seed, threshold, guard or configuration
is changed except as §6 decides; no protected assertion is weakened; every
port keeps its old and new body; nothing is committed. XOR first: the XOR
table, every proof named and the slow ones included, on HEAD and on the
candidate before anything changes and after each step.

Then §4, in its order: the gate reads the answers at the decided bar (1);
the answer begins with the understanding (2); a word reaches the grammar as
its object's code, in every binding (3); `not` on a code is its negation
(4); the two trials of a sentence are costed under the same parameters,
then the answer's map must converge, or the item stops and reports (5); the
reconstruction gate reads back through the grammar and admits
transpositions only, all four sentences (6); no operation undoes the one
before it at the same place (7); the `sum`-only negative control (8); the
receipt (9). Stop for review.

## 9. Hand-off to Codex, continued (2026-10-01)

Steps 1 to 5 and the six repairs of the October 1 receipt stand. The
receipt rule of October 1 replaces §8's XOR runs: each repeated
measurement once, on the candidate only, with no HEAD run; XOR_grammar's
ten unseeded runs stay. Then:

* **Step 5a.** Land variant (c) as the candidate's behaviour: for a
  sentence with a supplied answer, the answer's error, with its graph, in
  each trial's cost; both trials costed under the same parameters before
  either step; the batch-end answer update kept; sentences without an
  answer keep the cut. Each test that asserts the cut for supervised
  sentences is ported, old and new bodies kept, and a test is added that a
  sentence without an answer still gives the codes and the chooser no
  answer gradient. Ten unseeded runs of the class gate at the bar; Codex
  continues whatever the count and reports it.
* **Steps 6 to 9** as §4 has them. The `sum`-only control must still stay
  at one half with step 5a in place: a sum is one contribution per word,
  whatever the codes learn.
* **Receipt and review.** The XOR table, MM_grammar's ten runs, one
  source-matched full sweep; then stop for review. Part 4 of the October 1
  hand-off (trimming the suite) follows 6.9.

## 10. Review of the continuation (Claude, 2026-10-01)

Of the [continuation receipt](../benchmarks/2026-10-01-item6-9-continuation/README.md).
Steps 5a, 6 and 7 are implemented as decided: a supplied answer's error is
read from each trial's own end state with the cut lifted (the numeric head,
or generation in `answerSynthesis` configurations); the read-back takes one
least-residual pair over the known object vocabulary with the soft
gradient kept; each slot carries its last unary and an immediate inverse
there is masked. The class gate met the bar in 8 of 10 runs after step 5a
and in 10 of 10 at the end.

**The `sum`-only control failed because of the separator leaf.** The space
between the two words is pushed as a grammar leaf that keeps its perceived
event, since it has no object. Probed on a copy of the candidate, with the
`sum`-only grammar: the word leaves are identical across sentences, every
derivation is `sum(sum(L0, L1), L2)`, and the non-additive part of the
understanding equals the separator leaf's exactly. Its two content numbers
change with the pair of words around it, and since step 5a the answer's
gradient reaches them through perception. The runs' answer contrasts track
it (a separator contrast of .25 gave an answer contrast of −.82). With the
separator's leaf emptied before it is pushed:

| | runs | stay at one half | all four right | meet the bar |
|---|---|---|---|---|
| `sum`-only control | 10 | 10 | 0 | 0 |
| XOR_grammar | 10 | 0 | 10 | 9 |

So the grammar composes XOR from the word codes; the separator was the one
leak. It also explains the reconstruction gate's 0 of 10: every read-back
has three words, the separator's leaf decoded as a word.

**Decision applied (from Alec, 2026-09-29, §6 question 3):** "The grammar
operations are conducted over the object concepts, not the word concepts."
A separator has no object concept, so it is not a grammar leaf: in the
mixing binding a white-space unit is not pushed onto the grammar's stack
(identified as white space, not by a missing object row, so that a word not
yet admitted is unaffected). It stays in perception and in the byte
reconstruction. Only white space occurs in XOR_grammar; punctuation is left
as it is, for 6.8 and 5.5. This takes from 6.8 only the part of §5's "word whole" note that 6.9
cannot pass without.

**The full sweep's 21 failures.** Sixteen construct the private sentence
state without step 7's two new fields (a fixture port). One expects the
retired blended inverse (a port to the hard least-residual pair, which §4
step 6 decided). Four stop at the walk's policy cost recorded during a
sentence backward: the trial's generation call now records that cost. The
policy keeps its one owner, its credit at the batch end, so a trial's answer
neither records nor trains it, and those four tests pass unchanged.

**Stage 1 of the objective-conflicts spec.** Codex asked how one run per
configuration can compare training with and without the answer term: it
cannot; each configuration gets two runs, one with step 5a and one with the
cut, and the other stage-1 measurements come from the step-5a run.

**Hand-off.** In order, each repair with its failing probe saved first:
(1) separators off the grammar's stack, then ten runs each of the `sum`-only
control (all at one half), the class gate and the reconstruction gate;
(2) the policy cost stays out of the trials; (3) the seventeen ports, with
old and new bodies; (4) stage 1, two runs per configuration; (5) the XOR
table, MM_grammar's ten runs and one source-matched full sweep. Stop for
review. Then part 4 of the October 1 hand-off, then item 6.85.


## 11. Review of the repairs (Claude, 2026-10-01, evening)

Of the [review-repairs receipt](../benchmarks/2026-10-01-item6-9-review/README.md).
The separator repair, the policy ownership, the packed-sentence scope fix
and the seventeen ports are right, and the full sweep has no assertion
failure (one test timed out at the 30-minute worker limit:
`test_item9b_schedule.py::test_native_interleave_supplies_context_then_reads_the_same_sentences[True]`,
the slowest test of the suite, which part 4 takes up).

| | runs | result |
|---|---|---|
| `sum`-only control | 10 | all at one half (largest distance .0057, contrast at most 6e−8) |
| class gate, answers of both campaigns | 20 | all four right in 20; the bar met in 16 |
| reconstruction gate | 10 | 1 passes; every read-back now has two words |

**Not accepted yet: the two gates do not pass reliably.** The class gate
meets the bar in 16 of 20 runs, so it would fail about one sweep in five.
The reconstruction gate fails because nothing trains the grammar to read
back: XOR_grammar does not reconstruct in the loop, so its trials carry no
reconstruction term, and min and max keep only whichever word dominates a
coordinate. The read-backs show it: "hello hello", "world world", one word
lost. The answer shapes the codes for XOR, not for read-back.

So the reconstruction gate needs the grammar's tied reconstruction trained
in XOR_grammar's trials, which puts reconstruction and the answer in one
trial cost: the case item 6.85 exists for. Stage 1 of the
[objective-conflicts spec](../specs/2026-10-01-objective-conflicts.md#9-stage-1-results-and-a-question-on-equal-magnitudes-2026-10-01)
measured XOR_grammar and found a question for Alec there. The questions
this raises are put to Alec; 6.9 stays open until both gates pass under the
cost function 6.85 settles.

**Alec's answers (2026-10-01, evening;
[objective-conflicts spec §10](../specs/2026-10-01-objective-conflicts.md#10-alecs-answers-to-9-and-to-the-69-review-2026-10-01-evening)):**
every configuration reconstructs its input through the understanding, so
XOR_grammar gains the read-back training its reconstruction gate needs;
terms become relative errors of a similar norm in the `Error` registry; no
separate worktrees. 6.9 closes after item 6.85, under its cost function.

## 12. Remaining steps (2026-10-01)

There is no separate item for the cost function (Alec: "No separate item,
it would take too long. Add it to the current work, or even better, the
next work"). In the one working tree, in order:

1. **Item 7's operation records.** They are sized for the whole packed
   row and rewritten in full every operation round, so the production
   benchmark does not fit at its own batch (stage-1 receipt). They are
   sized for one open sentence, and written detached only if no
   objective's reach changes.
2. **The production stage-1 measurement**, as a `RUN_SLOW` test at the
   benchmark's own batch.
3. **The suite trim**, items 1 to 10 of the October 1 hand-off; the venv
   rebuild and the old reading modes wait until this item closes.
4. **The cost function**
   ([objective-conflicts spec §8 and §10](../specs/2026-10-01-objective-conflicts.md#10-alecs-answers-to-9-and-to-the-69-review-2026-10-01-evening)):
   the mixing binding stages its own reconstruction bank, and the XOR
   measurements with and without the trial answer are taken; every
   configuration then reconstructs through the understanding; every term
   becomes a relative error in the `Error` registry, with weight 1 unless
   a configuration states a priority; reconstruction's precedence holds in
   the gradient and in the choice between trials; every term is described
   in the docs.
5. **Close:** ten runs of each gate and of the `sum`-only control, the XOR
   table, MM_grammar's ten runs and one source-matched full sweep.

## 13. Code review of plan §12 parts 1 to 4a (Claude, 2026-10-01, night)

Of the [stage-1 and trim receipt](../benchmarks/2026-10-01-stage1-and-suite-trim/README.md),
against the state reviewed in §11. Three reviews: the runtime changes, the
deletions and inlining, the test moves. No blocker. Parts 1 to 3 are
accepted once the fixes below are made; part 4 continues after them.

**Measured on the way (production benchmark, single runs, stage 1):** with
the answer training the whole path, the end-of-training reconstruction cost
is .81 against .18 with the trial answer cut, and the answer error .067
against .181. On the chooser, at the first batch, the two gradients oppose
(cosines −.53 and −.45) and the answer's is about 2,400 times larger. The
conflict of the objective-conflicts spec is real in production.

**Should fix:**

1. *A stray closing overwrites the open sentence's closing frames.*
   `Models.py` (stage_cs_lang): `closing_active = intermediate_end` is not
   gated by the row's open sentence, and the sentence-local layout
   (`_sentence_journal_layout`) gives every closing group the same window.
   In a packed row, a word that ends a *different* sentence during the open
   sentence's pass fires a closing whose operation records land in the open
   sentence's closing frames (before the repair they went to a separate
   group). Live when that closing applies an operation (a nonzero compose
   temperature, or an exhausted closing budget). Fix: gate it with
   `row_gate`, check that no other per-word operation touches a row that
   does not hold the open sentence (a committed row's state can be mutated
   after commit today), and add a packed two-row test whose stray closing
   applies an operation.
2. *Non-ASCII words are double-encoded in the mixing reconstruction bank.*
   `_stage_mixing_reconstruction_bank` re-encodes `word_texts` as UTF-8,
   but the stem builds them by decoding bytes as latin-1, so "café" becomes
   "cafÃ©" and the target stops matching the input. Take the target from the
   raw byte span (or encode latin-1), make the candidate surfaces
   (`word_surface_for_row`) agree, and add a non-ASCII test.
3. *One folded guard checks a different file.* `test_retired_names.py`
   checks `data/complete.grammar`; the guard it replaced checked
   `test/fixtures/transitional_pos.grammar`. Check both.
4. *The new slow marker moves nineteen older gates to the accelerator under
   `make test_all`*, among them both XOR_grammar gates and the MM_20M
   exact and blind round trips, which are calibrated on CPU. Dispatch the
   device on explicit marks only (the XOR proofs stay on CPU), or use a
   separate weekly-selection marker; update `doc/Testing.md`.
5. *One UTF-8 "port" does not test what it claims:*
   `test_meronomy_utf8.py::test_admitted_word_and_utf8_bytes_replay_together`
   asserts nothing the admission step affects. Record that behaviour as
   retired, or assert the admitted word's id.
6. *Code orphaned by the trim* (no legacy paths): `IdeaSubSpace`
   (`Language.py`, used only by its own test), the retired spelling protocol
   in `Layers.py` (`emit`, `bind_marker`, `canonical_marker`,
   `bound_markers`), and `space_carrier.py`'s
   `mark_codebook_parameters_changed` / `mark_codebook_structure_changed`
   (no callers). Delete them with their tests, or show a live use.

**Minor:** restore inductor for the compiled K=2 test, whose subject is
compilation (the saving was small: 1,390 to 1,060 seconds); a test of
`test_meronomy_ladder.py` changes the shared trained fixture without
restoring it (use `monkeypatch`); correct the coverage map (marker replay;
a test id that does not exist) and the receipt's "inactive gaps retain
separate slots" (they share frames 0 to 2, harmlessly); the record width
now varies per batch (check for recompilation); the mixing bank is staged
even when nothing reads it; nothing runs `python bin/Legacy.py` or the
`bin/etc` inline tests (put them in the weekly run; the `make SigmaPi`,
`SymPercept` and `SPNN` demos now need `--demo`); stale comments in
`test_where_bracket.py`, `doc/Mereology.md`, `Legacy.py` ("bin/py"),
`Layers.py`, `Spaces.py`, `data/model.xsd` and two tests. Pre-existing:
`python bin/Layers.py` fails at `TruthLayer.test()`.

**Checked and correct:** the sentence-local layout and its readers, the
live gradient through the records, the mixing bank's separation from word
identities and its staging order, the eager reconstruction loop; every
deletion has no remaining use and no current checkpoint needs a removed
migration; the inline tests run and match; of 4,500 test definitions none
is lost (4,079 unchanged, 246 moved with identical bodies, 175 deleted with
a reason), no assertion loosened, no seed added; 5,058 cases collect; the
weekly selection is 302 cases. Not yet seen: a complete weekly slow run.

## 14. Next work (2026-10-01, night)

From the closing receipt of §12 (class gate 9 of 10, reconstruction gate 0
of 10, `sum`-only control without interaction, eleven sweep failures):

* **The old reading modes are migrated now** (trim item 12, pulled into
  6.9). XOR_grammar and 48 other configurations were exempt from
  reconstruction through the understanding because they read text with an
  old mode, nearly all by inheriting `<synthesis>lexicon</synthesis>` from
  `data/model.xml`; in them only perceptual reconstruction runs, so the
  grammar learns from expectation and a supplied answer alone, and the
  reconstruction gate has nothing to learn from. The default moves to
  meronomy, with the configurations that name an old mode, and the scope
  rule then applies to them by construction (Claude's scoping error: the
  exemption was meant for a few configurations, not most).
* **The journal width comes from a configuration bound**, not the batch's
  longest sentence, so that sentence lengths do not cause another graph
  capture.
* **The `sum`-only control's criterion, made exact:** it passes when no run
  shows an XOR interaction (checkerboard contrast at most 1e-4 in every run)
  and no run meets the class bar. Its answers need not equal one half to
  float precision; a finite run ends near it.
* **The eleven failures:** the detached reverse "student" on a retained
  reading and the legacy event reporting with tied reconstruction off are
  paths the decisions retire, deleted with their tests (keeping a migration
  only if a current checkpoint needs it); the three no-candidate cases are
  ported to the decided rule (no reconstruction term, counted); a term of
  weight zero must not require its source object; the compose-deadline
  zero top vector needs its root cause; the extra graph capture follows the
  journal width.

**Candidate receipt, 2026-10-02 (stopped for review):** the
[§14 receipt](../benchmarks/2026-10-01-stage1-and-suite-trim/review14/README.md)
retains the pre-repair probes, complete old/new ports, all configuration
first-batch attempts and the first weekly run's cause triage. The ten class
runs pass 8/10; the ten reconstruction runs pass 3/10; the ten sum controls
pass. The table is 30/34 and MM_grammar's median ending MSE is 4.36e-11.
The source-matched full sweep attempted all 5,060 cases in 214.66 minutes:
4,672 passed, 314 skipped, 71 reported failures and three stopped without
a result, against 5,219 cases / 122 minutes previously. The receipt discloses
ten case IDs inadvertently repeated by its continuation; first outcomes
are retained and wall time includes the interruption. The corrected
continuation adds no further duplicates. Nothing is committed; acceptance
remains for review. The measured source is preserved in the receipt's
`closing/source-final.zip`; its `before.zip` predates §14.

**Alec's clarification during the §14 receipt:** lack of run-to-run consistency
is acceptable; demonstration of learning was 6.9's primary goal. The observed
8/10 class passes demonstrate that learning. The 3/10 reconstruction result
remains a concern, with gradient conflict a hypothesis for review. Keep the
measured bars and assertions unchanged and finish the closing receipt.

Alec then deferred the gradient proposal to
[FutureWork](../FutureWork.md#separate-gradient-ownership), preferring a
structural design in which objectives train separate weights. This is future
design work, not a further change to the current 6.9 candidate.

## 15. One writer for each weight (Alec, 2026-10-02)

Alec, on Claude's proposal in reply to the
[FutureWork section](../FutureWork.md#separate-gradient-ownership): "Your
proposal is approved; no need to dig too deep into it, let's run it and see
if it works (I think it will)." On expectation: "Expectation changes
expectation only is fine: please do what is most efficient to make that
happen; it may be that training should be deferred one step, but detaching
the gradient is fine too." On the answer: "since understanding is sufficient
to reconstruct the input, it's bijection should be as a good a basis for
output derivation as the input itself, and it will be operating on the
understanding. Let's make sure that the output prediction has everything in
context that the input reconstruction has. (E.g. the priming on symbols,
etc)."

This replaces step 5a's whole-path answer (§4) and the supplied-answer
projection of the objective-conflicts spec. No two objectives write the same
weight, so there is nothing left to project. The objectives meet only in the
choice between a sentence's two trials, which is a choice, not a sum of
gradients.

### 15.1 Owners

| Weights | Trained by |
|---|---|
| Perception: PartSpace and WholeSpace percept codes, the joint perceptual embedding | Reconstruction, through each trial's perception pullback. The existing perceptual `embedding` term stays. |
| Object codes where they are parameters (XOR_grammar, MM_grammar). Where `conceptualContextLearningRate` is set, the distributional rotation is unchanged. | Reconstruction, through the leaves |
| Compose operators, their code-to-operation projections and tied inverses. Generate uses the same numerical hosts. | Reconstruction, through the recorded derivation's tied inverse |
| The compose chooser | Reconstruction, through its straight-through surrogate. Compose grammar lessons, where enabled, train it too. |
| The expectation predictors: within-sentence, between-sentence, ARMA, contrastive | Expectation only (§15.2) |
| The answer's readers: the numeric head; in `answerSynthesis` configurations, the generate chooser and conditioner | The supplied answer and the output policy. The answer's error stops at the understanding, as in the cut of 2026-09-20. |

Rules:

- Each objective's backward is restricted to the parameters it owns, for
  example with `backward(inputs=...)`, or by detaching at the boundary.
  Every optimizer state then belongs to a single objective. This also
  removes the case in FutureWork where Adam's per-coordinate scaling turns
  orthogonal gradients into a collision.
- Generation passes through the operators without updating them. Generate
  grammar lessons train the generate chooser only; compose lessons train the
  compose chooser only. No lesson moves an operator.
- Penalties stay with the weights they regularize.
- **Trial selection.** The explore trial is kept only if its reconstruction
  is no higher and its reconstruction-plus-answer total is strictly lower.
  Expectation is not in the comparison. A tie keeps the greedy trial.
- **The negative image stays.** The compose chooser reads expectation's
  prediction as detached evidence. That is a forward read, not a gradient.
- `expectationPolicyWeight` stays 0. If enabled, it would let expectation
  train the thought controller, which breaks the rule above.

### 15.2 Expectation trains only its predictors

Every expectation term detaches both its targets and its sources.

- **Within-sentence term.** `expectation.intra` today compares its
  prediction with the word's live idea (`idea_bd` in `Models.py`, passed to
  `_accumulate_intra_loss` in `Spaces.py`). Its error therefore falls when
  words become alike: the collapse channel. XOR_grammar inherits
  `intraLossWeight` 0.1 from `model.xml`. Its STM context is detached as
  well.
- **Between-sentence terms.** These already detach the arriving meaning.
  They also detach their preceding context, which is still live within the
  optimizer step.

Detaching is enough; training need not be deferred by a step. The
predictors keep their weights and their steps.

### 15.3 The answer reads what reconstruction reads

Per trial, one understanding record is built and read by both the tied
reconstruction and the answer. Built once and read twice, it cannot drift
between them. It holds:

- the concluded root and end slots;
- the recorded derivation: rule ids, arities and operand positions;
- the operation journal's witness offsets (what each lossy operation
  dropped);
- the per-word retained references;
- the primed symbols (below).

**The primed symbols replace the sentence's concept lookup bank.** Today
that bank holds only the sentence's own words, and it is the
reconstruction's only source of candidates. Alec, 2026-10-02: "we really
want the connections bank of symbols. That is what the initial
open-attention (parallel) read will prime, and the significant point is
that we will also be reading the words that those words activate. Quite a
bit of thought went into the algorithm that does this priming, so please
make sure to use it in place of the 'concept bank' (although if you want to
select a top-k from it, that is probably a necessary concession to
efficiency)."

The algorithm is the SEEN priming of 2026-07-12 (`Space.prime_seen`). At
each write:

- the surface decays toward neutral;
- a `primingSpread` fraction of each row's standing energy flows to its
  neighbors over the concept store's (concept, constituent) edges;
- the rows just seen are bumped.

Repeated priming carries energy further into the connected symbols.

The bank is the top of that surface, per row: the rows the sentence itself
primed, plus the `reconstructionBasisLimit` most primed others. Primed rows
with a word surface are the reconstruction's byte candidates. Every bank row
enters the answer's context with its priming weight.

Three changes make the algorithm run where 6.9 reads:

1. **Diffusion also runs in serial reading.** `_priming_edges` returns
   nothing unless the pass is parallel with a positive symbolic order
   (`_sparse_active`). So XOR_grammar, MM_ladder and BasicModel prime only
   the words they see, never the words those words activate. The store's
   definitions exist in either mode. Until item 6.8's open bracket exists,
   the sentence's SEEN write before its serial words
   (`_prime_sentence_symbols`) stands in for the open read, and it diffuses
   over the store's edges.
2. **No edge cap.** The edge list is built by a host loop over every
   nonzero, and diffusion is skipped, with a warning, above 4,096 edges,
   which a production store passes early. Build the list on device from the
   store's sparse indices. Keep only edges whose source has standing energy:
   a source at neutral sends no flow, so the result is exact.
3. **One surface per batch row.** The surface is a single `[V]` vector
   shared by the whole batch, so the rows of a batch prime each other. Make
   it `[B, V]`, each row primed by its own sentences.

The bank is taken after the sentence's SEEN write, and both trials read
that same snapshot. The taxonomy's hop priming in `Language.py`
(`Taxonomy.propagate`, read only by the dark inverse recommender) is a
different mechanism and is not used here.

The answer reads the same record, detached, because its error stops at the
understanding:

- The numeric head takes the record as further fixed-width input.
  Recommended for the bank: read it through the existing `GlobalAttention`
  scorer and its consume gate, which starts at zero. The bank is the
  addressable space, the priming weight is the boost feature, and the end
  slots are the query. This is the learned reader that global attention's
  consumer already provides (6.8 plan §7); here the answer owns it. Codex
  chooses the fixed-width encoding of the other fields and lists it in the
  receipt.
- Generate takes the record as named context.

The receipt lists the record's fields, and a test checks that both
consumers read the same object.

Notes for the gates:

- The witness offsets of `min` and `max` are already nonlinear in the
  words, so the answer can read XOR from the record as well as from the
  root. The `sum`-only control still guards against an additive shortcut,
  because there the whole record is additive in the words.
- Reconstruction now chooses among the activated words as well as the
  sentence's own, so it is a harder test than §14's. The receipt reports
  how often an activated word outranked the sentence's own word.

### 15.4 Retired

The following go, with their tests:

- step 5a's in-trial answer read without the cut;
- the supplied-answer projection against reconstruction and its helpers;
- expectation's gradient into ideas, codes and context.

Precedence for reconstruction remains only in trial selection (§15.1). The
per-operator diagnostic becomes an ownership audit: during a training batch,
each optimizer parameter receives gradient from exactly one objective.

If the gates fail, the §14 candidate (its `before.zip` and the receipt) is
the comparison. Restore from there rather than keeping a switch.

### 15.5 Measurements

The §14 closing set, once, on the final source:

- the class gate, 10 runs;
- the reconstruction gate, 10 runs;
- the `sum`-only control, 10 runs;
- the named XOR table;
- MM_grammar, 10 runs;
- one source-matched full sweep.

In addition:

- the ownership audit on XOR_grammar and on
  `BasicModel_answers_tied_benchmark`;
- the production stage-1 slow test, once, under its 24 GiB ceiling.

The comparison is §14: class 8/10, reconstruction 3/10. If class falls
below 8/10, or reconstruction does not rise above 3/10, run the attribution,
10 runs each:

- reconstruction alone;
- reconstruction with expectation;
- reconstruction with the answer;
- all three.

Seeds, bars and assertions are unchanged.

### 15.6 Test and configuration removal

Alec: "You can write the redundant test removal into the final 6.9
feedback: I don't mind if it goes in untested, since it's test deletion and
not code changes." Apply §2.1 and §2.2 of the
[configuration review](2026-10-02-configuration-review.md):

- **Delete 21 configurations,** with their tests and tool references.
  The tests that go:
  - `test_svo_end_to_end.py`, `test_symbolic_iteration.py`,
    `test_mm_boolean.py` and `test_modality_configs.py`;
  - the two nanochat cases that load the historical files;
  - the deleted names in `test_training_diagnostic_contracts.py`.
- **Point the training defaults at `data/BasicModel.xml`:** `make train`'s
  `MODEL`, `bin/train.py --model`, `bin/eval_nanochat_grammar.py`'s
  default, and the Training/Installation docs. Then delete
  `MM_20M_fineweb.xml` and move its seven fast cases to `BasicModel.xml`.
- **Merge `MM_ltm_consolidation_stateful_fixture.xml`** into
  `MM_ltm_consolidation_fixture.xml`: its six cases set
  `<stateless>false</stateless>`.

Check only collection, the documentation links and the moved cases.

Also decided (Alec, 2026-10-02), from review §2.3 and §2.4:

- **MNIST and ergodic stay.** Keep `mnist.xml`, `ergodic.xml` and
  `ergodic-only.xml` ("Keep Ergodic, it's better in principle than random
  weight init"; the wider question is in FutureWork). MNIST fails only
  because `data/mnist_train.csv` is a Git LFS pointer: install git-lfs and
  pull the file. The loader should name that cause when it meets a pointer.
  Add one short MNIST test on a subset, run with `ergodic` both on and off.
- **Delete `simple.xml` and `tomatoes.xml`.** With them go `make simple`,
  `make tomatoes`, the `tomatoes` dataset branch and loader, and the
  `XML1`/`XML2` defaults. The README quick start then uses `mnist.xml` and
  `XOR_exact.xml`.
- **Retire three opt-in features,** with their tests:
  - the D3 idea decoder: the `ideaDecode` seeding hook and flag, and
    `MM_decode.xml`;
  - the word store: `wordStore` and `matrix/MM_20M_grammar_wordstore.xml`;
  - the overlapped where tiling: `overlapWhereTiling`, `WhereTilingLayer`
    and its staging, `bin/eval_where_tiling.py`, `test_where_tiling.py` and
    `MM_overlap_tiling.xml`. FutureWork records the tiling so that it can be
    revived from `d679df2b`.

  `radialStmReduce` stays.
- **Move surviving tests onto kept configurations,** then delete the eight
  files they come from: `MM_phrase_decode`, `MM_meronomy_smoke`, and the six
  files of review §2.4 (b). The serial features go to `MM_ladder`, and the
  parallel field and the symbol tower to `XOR_exact` or `MM_20M_xor`.
  `make bench_local` then defaults to `MM_ladder`.
- **Shrink `MM_add_verb`** below the 8 GiB guard, and drop its
  `ideaDecode`.
- **Keep reading attention and global attention until item 6.8.** That
  includes `MM_reading`, `MM_global`, `MM_qa` and
  `matrix/MM_20M_grammar_reading`; the 6.8 plan's §7 lists what each
  carries over. Repair `MM_qa`'s TruthSet input, and fix the mixing-leaf
  staging defect from the weekly triage (a row with no `word_texts`).

These items change code. Do them before the §15.5 sweep, so that the one
sweep covers them. In addition, run the moved cases, the MNIST test, and the
weekly-tier cases of the kept attention configurations once.

### 15.7 Documents

- **GradientFlow** is rewritten to this contract: owners, detached
  expectation, the shared record, the selection rule.
- **FutureWork's "Separate gradient ownership"** is marked as taken up here.
- **todo 6.9** records the decision.

## 16. Review of the §14 receipt (Claude, 2026-10-02)

The receipt is complete and its source hashes match. The results:

| Measurement | Result |
|---|---|
| Class gate | 8/10 |
| Reconstruction gate | 3/10 |
| Sum control | 10/10 |
| MM_grammar, direct-forward runs | 8 of 10 near zero (median 4.4e-11) |
| Named XOR table | 30/34 |
| Full sweep | 5,060 cases: 71 failed, 3 workers stopped |

Codex triaged every failure (`review14/closing/failure-triage.json`), and its
follow-ups are right. This section adds what blocks acceptance and sets the
order of work.

### 16.1 MM_xor's XOR proof regressed with the reading migration (blocking)

Two cases of the standing XOR table failed in the §14 receipt:

- `test_mm_xor.py::test_convergence` (MM_xor): XOR loss .242 after 200
  epochs, against a bar of .20;
- `test_mm_grammar_learns_xor_signal` (MM_grammar): .25 after 900 epochs.

Claude measured both, unseeded.

| Source | MM_xor convergence | MM_grammar proof |
|---|---|---|
| §13 snapshot (`review14/before.zip`) | 5 of 5 | 1 of 2. The failure stopped at exactly .25, the symmetric fixed point. |
| §14 candidate | 1 of 5, and 1 of 3 on `closing/source-final.zip` | 2 of 2, but each run needed about 250 s where §13's pass needed 3 s |
| §13 code, with only MM_xor's two reading settings migrated to meronomy | 2 of 5 | — |
| §14 final source with `reconstructionScale` 0 | 1 of 5 | — |

**MM_xor has regressed, and the cause is the migrated reading.** MM_xor
moved from `radix`/`byte` to `meronomy`. The cause is neither the new
reconstruction term nor the rest of §14's code.

**MM_grammar's failure in the table is not a regression.** Its proof
sometimes ends at .25 on §13's source too. On the candidate it passes, but
it uses most of its 900-epoch budget to do so. That slower convergence is
consistent with MM_xor's regression. MM_grammar inherits `model.xml`'s move
from `lexicon` to meronomy.

Before §15, measure what meronomy reading gives MM_xor's answer, as §3.5 to
§3.7 and §3.13 did for XOR_grammar:

- how many content numbers each word's leaf carries;
- the singular values of the four centred understandings;
- the best affine fit;
- the derivations chosen.

Fix the cause within meronomy; the old modes stay retired. MM_xor must pass
as it did in §13. If the cause is not found quickly, report before starting
§15. MM_grammar's occasional stop at .25 is the known symmetric fixed point
(item 11). It is recorded as existing flakiness, not as a bar for this
round.

### 16.2 Fixtures of the retired reading modes

These failures come from the same migration:

- 41 retained fixtures that use the retired lexicon interface:
  `PartSpace._embed`, `RadixLayer.doc_spans`, `_token_stream` and `getW`,
  and `Spaces.Embedding`;
- 6 fixtures that select `analysis=word`;
- 4 tests of the retired mode dispatch;
- the retired-name guard;
- the two promotion fixtures;
- the resume fixture;
- one incomplete configuration port.

Codex's follow-ups stand. Port each behavior that survives in meronomy:
attested-whole preference, complete tiling, division, promotion thresholds,
resume persistence. Delete the tests whose subject is the retired interface.
Add no compatibility methods.

### 16.3 Fixed input-address capacity

Seven failures raise `occurrence exceeds its configured where-space capacity`
in `_embed_radix` while stamping the word offset grid. They stop:

- MM_math, in five cases;
- `idempotent`;
- XOR_spaces, which §15.6 deletes.

This is the same group as the §14 first-batch exceptions (MM_add, MM_math,
stream_smoke). Find the mismatch of units or capacity that the migration
introduced, without raising the configured limits. `idempotent.xml` joins
the repair list.

### 16.4 A compile failure in the compiled reconstruction

`test_word_admission[XOR_grammar.xml]` fails inside `_reconstruct_sentences`:
Inductor's while-loop lowering receives a Python int among the loop's
operands (`'int' object has no attribute 'meta'`). The case passed in §13.
BasicModel compiles this reconstruction in production, so this is a code
bug, not a fixture. Every operand of the loop must be a tensor, or the int
must be closed over.

### 16.5 Four behaviors whose causes are unresolved

Find each cause before changing any test.

- **Reference side.** The inverse sees `reference_side[1]` true during trial
  reconstruction (`test_grammar_reconstruction_gate`). Is this a staged
  object operand, or a regression in reference ownership?
- **Repeated reading.** Two readings of the same input reconstruct
  differently (`test_output_synthesis`), which suggests state leaking between
  readings. This matters for §15.3, whose record must be the same object for
  both consumers.
- **Category codebook.** It assigns no centroid after fifteen forwards
  (`test_category_em_smoke`). BasicModel enables `categoryCodebook`.
- **A strict expected failure passed** (`test_topk_recovered_words_overlap_input`).
  The measured overlap varies from run to run (.8 and .5 are both
  recorded). Make the marker non-strict and record that history; do not count
  it as a pass.

### 16.6 Smaller points

- **Three workers stopped:**
  - `test_reasoning_cde_model` went over 8 GiB;
  - two heavy cases were aborted or interrupted.

  Section 3 of the configuration review moves such cases to the weekly tier,
  or ports them.
- **The weekly run's "recent 6.9" failures** are ported to the new contracts
  in the same round:
  - probes that call the sentence-local journal writer without its index
    map;
  - assertions on the former whole-row journal lifetime;
  - proximal registration at each trial step;
  - the backward probe that sees no loss;
  - the absent parameter snapshot.

  The leaf-staging defect is in §15.6.
- **The sweep's time does not compare.** The sweep ran two workers where the
  reference ran up to ten, so its 214.7 minutes cannot be set against the
  reference's 122. Use the reference schedule unless memory requires fewer
  workers.

### 16.7 Order of work

1. §16.1.
2. §15 (ownership, expectation, the record, the retirements), together with
   §15.6's cleanup and §16.2 to §16.6.
3. §15.5's measurements, once.

The XOR table must be back at the accepted baseline: only the predeclared
first gate runs may fail.

## 17. Review of the §16.1 stop (Claude, 2026-10-02)

Codex stopped at §16.1, as §16.1 told it to, and changed no production
source ([receipt](../benchmarks/2026-10-02-item6-9-ownership/README.md)).
Its diagnosis is careful.

- **MM_xor is parallel.** `serial` is false and `symbolicOrder` is 0, so its
  grammar never runs. Three butterfly `ConceptualCombine` stages feed a
  linear numeric head.
- **Under meronomy each word is one six-number percept.** A separator
  percept sits between the two words, and the rest of the eight slots is
  padding.
- **The first word drops out.** By the end of the failed run, the first word
  of every sentence (hello, loving) reads all six content numbers as zero,
  and the answers sit at .50.
- **Only the last of the three bindings receives the answer's gradient.**
  Each binding reads the original percepts, and the WholeSpace carrier
  between bindings returns a neutral field. The declared recurrence
  therefore does not occur. §13's source does the same.

**What §13's pass rested on (Claude, measured).** Trained as
`test_convergence` trains it, §13's MM_xor promoted four chunks that cross
the word boundary: `hello wo`, `hello th`, `loving w` and `loving t`. That
gives one percept per sentence. It passed in 20 and 22 epochs. The old radix
reading did no whitespace split and promoted recurring spans of the whole
line. Meronomy bounds promotion to words.

So in §13, XOR over these four inputs was a lookup of four distinct codes,
which any affine reader fits. That test never asked the parallel path to
compose two words. The migration removed the shortcut, and the test now
asks exactly that. The parallel path cannot do it. It has no live
recurrence, and the head reads word slots linearly. This is §3.3's finding
again: a linear reading of words is a sum of contributions, which cannot be
XOR, and its best is one half everywhere.

So there is no loss of nonlinear learning: §13's green was never evidence
of it. The other proofs in the §14 table still show nonlinear learning:

- XOR_exact (the parallel field);
- XOR_grammar's class gate (8 of 10);
- grounded XOR (6 of 6);
- the concept-output curriculum.

**For Alec: how the gate treats this test.**

- **(A) Fix the parallel path now.** Make each binding consume the one
  before it, and give the path a cross-word nonlinear operation before the
  head (the field's `and`/`or`/`not`). Item 6.8 rebuilds this mode as the
  open read.
- **(B) Recommended.** Keep `test_convergence` and its bar unchanged. Record
  it as red, with the cause above, and make word-level XOR on MM_xor an
  acceptance test of item 6.8's open read and field operations. Continue 6.9
  with §15 now. The 6.9 acceptance rule then exempts this one test, by name
  and with this reason.
- **(C) Make MM_xor serial,** so that its grammar composes the words. That
  duplicates XOR_grammar.

**Decided (Alec, 2026-10-02): option B.** "Ok, that would explain learning
xor the wrong way… Yes, option B, we'll get it working in the next stage."

- `test_convergence` keeps its test and its bar, and stays red.
- The receipt and todo 6.9 record it, with the cause above.
- 6.9's acceptance exempts it by name.
- Word-level XOR on MM_xor is an acceptance test of item 6.8 (6.8 plan §8).
- 6.9 continues with §15, together with §15.6 and §16.2 to §16.6, then
  §15.5's measurements once.

## 18. Review of the ownership round (Claude, 2026-10-02, evening)

The [receipt](../benchmarks/2026-10-02-item6-9-ownership/README.md) carries
out §15 to §17 and records the results faithfully. Not accepted.

**XOR table: 31 of 34.**

- **`XOR_exact`'s output proof is a new failure (blocking).** Its answer
  reads named concept evidence directly, so the evidence coefficients are its
  only reader. The partition gave them to reconstruction. Reconstruction is
  already exact there (cost .0000), so nothing moves them, and all four
  answers stay at 0.
- **The other two failures are the predeclared first gate runs.**
- MM_xor (§17) and MM_grammar passed their single runs.

**The gates.**

| Gate | This round | §14 |
|---|---|---|
| Class | 1/10 | 8/10 |
| Reconstruction | 1/10 | 3/10 |
| Sum control | 10/10 | 10/10 |

**The attribution.** Ten fresh, unpaired runs per arm:

| Arm | Class | Reconstruction |
|---|---|---|
| Reconstruction alone | 0 | 2 |
| With expectation | 0 | 4 |
| With the answer | 1 | 1 |
| All three | 0 | 3 |

The audit found no weight with two writers.

Two conclusions follow:

1. **With its error stopped at the understanding, the answer does not
   learn XOR in any arm.** This repeats §3.13.
2. **Reconstruction's low gate is not a collision.** Reconstruction alone
   passes 2 of 10.

**Why the bijection argument was not tested.** Alec's argument needs its
premise: "since understanding is sufficient to reconstruct the input". That
premise does not hold for the understanding alone.

- **Reconstruction trains on the whole record**, including the journal's
  witness offsets, which hold what each `min`/`max` dropped. With them the
  inverse is exact whatever the root holds. That is why reconstruction's
  trained cost is low (relative .107), yet nothing requires the root to
  carry the words.
- **The gate reads back without witnesses.** It reads from the root and the
  known vocabulary, and it returns duplicates ("world world", "there there")
  and other sentences' words. This is §3.9's blending, at about 2 in 10 with
  reconstruction alone.
- **So the cut was tested on an understanding that is not a bijection of
  the input.**

**For Alec, the next step:**

- **(A) Make the premise true, then test the argument.** Train
  reconstruction as the gate reads back: from the root, over the primed
  candidates, without witnesses. This is "a mind has to see what is there"
  taken literally: the understanding must hold what is there. Keep §15's
  ownership, and give `XOR_exact`'s evidence coefficients to its answer,
  since they are its reader. If class stays below 8 of 10, fall back to (B)
  in the same round.
- **(B) Restore the answer's reach into the understanding** for sentences
  with a supplied answer, as measured in §14 (8 or 9 of 10). Keep
  expectation detached, the shared record and the selection rule. This
  closes 6.9's demonstration. The question of witnesses stays open.

**Also in the round:**

- **The primed bank found no competitors.** In 6,400 own-word
  occurrences no activated word competed. Either the diffusion reaches no
  other words in XOR_grammar, or the extra rows carry no surface. Under (A)
  the bank becomes the read-back's candidate set, so its contents matter.
- **The full sweep took 19.4 minutes with ten workers** (§14: 214 minutes
  with two; reference 122). The heavy-case work paid off.
- **23 sweep failures remain, mostly unfinished ports:**
  - old probe signatures;
  - imports of the merged stateful fixture;
  - three expectation tests that still assert the retired contract (that
    expectation trains its source);
  - the row-local priming fixtures;
  - two old `getW` tests;
  - a branch-diagnostic accessor;
  - a shape change in the thought catalog.

  Of the 170 moved slow cases, 48 failed and 12 stopped. They need the same
  pass.
- **Codex found real bugs and fixed them:**
  - the public inverse overwrote the scored trial;
  - the category codebook read byte ids;
  - the byte-address capacity was taken from a word-slot count;
  - a Python int was lifted into the compiled loop;
  - lazy predictors were missing from the optimizer.

## 19. Verification before option A (Claude, 2026-10-02, night)

Alec leans to (A): "we have to get reconstruction before we get prediction
or output." He asked three things to be verified from the receipts and the
code, without new runs.

### 19.1 Was the reading path trained by reconstruction alone?

By gradient, yes. The ownership audit (`xor-ownership/ownership.json`,
1,200 backwards, 0 conflicts) shows reconstruction's gradient reaching
exactly five tensors:

- the percept dictionary (256 rows × 2);
- the stage-1 concept dictionary (6 rows × 10): the object codes;
- the chooser's three anchors (stop, reduce ×2, apply: 40 numbers).

Nothing else on the reading path has parameters. `not`, `conjunction` and
`disjunction` are `−d`, `min` and `max`. Expectation reached only its
stage-1 within-sentence predictor; the between-sentence predictor never
fired (one sentence per row). The answer reached its head and the record
reader.

**But the object codes have a second writer, outside autograd.** The
concept codebook is a VQ with EMA state (`codebook-ownership.json` lists
`vq.cluster_size`, `vq.embed_avg`), and `Codebook.forward` re-points `W` at
the VQ's codebook after every training quantize (`Spaces.py`
`replace_W(self.vq.codebook)`, gated only by `vq.ema_update`, which is on for
ConceptualSpace; only the perceptual codebook is switched to gradient-only
mode, and only the SBOW similarity codebook is built without EMA). So each
training forward moves every code row toward the mean of the word events
assigned to it. The audit counts backwards, so it cannot see this. The code's
own comment records that this refresh once pinned XOR_exact's output at the
constant floor for 600 epochs ("THE #13 mechanism"), which is why
WholeSpace's codebook has it gated off.

That the quantize runs in XOR_grammar's training forward is inferred, not
measured: the audit shows reconstruction's gradient reaching
`conceptualSpaces.1.layers.3.W`, which is that codebook's straight-through
path. The measurement that would settle it is `vq.cluster_size` after
training, or the drift of the six rows.

### 19.2 What changed reconstruction?

Nothing broke it; it never passed. XOR_grammar's reconstruction gate:

| Receipt | Pass |
|---|---|
| September 30, old lexicon mode (the gate read perception's reverse, §3.8) | 0/10 |
| §13 | 0/10 |
| §14, reading through the understanding | 3/10 |
| Ownership round | 1/10; attribution arms 2, 4, 1, 3 of 10 |

Three of ten and one of ten are the same rate. The bar is now all four
sentences as word multisets (`test_explicit_dimensions.py`: "the decided
contract is now all four").

Alec's guess, that enforcing word-level concepts broke it, is right for
MM_xor (§17: the radix store promoted whole phrases) and does not apply
here: XOR_grammar's old lexicon mode already gave one leaf per word (§3.5).

What the gate measures has never been trained. Training's inverse receives
each word's retained reference and the journal's witness offsets
(`_reconstruct_sentences`; the `candidate_basis is None` branch). The gate
zeroes the reference (`ref = torch.zeros_like(ref)`, no reference side) and
searches the candidate vocabulary for both operands. With witnesses the
inverse is exact whatever the root holds, so the trained cost is low
(relative .107) while the free read-back fails. Reconstruction's gradient on
the codes and the chooser is therefore small, and it does not ask the root
to determine the words.

### 19.3 Does the representation have what XOR needs?

Alec's conditions, checked:

- **Words are unitized.** `Interpret.forward` (tensor face) builds the leaf
  as `object_atoms × activation`, concatenated with the word event beyond
  the atom's width. The atoms are the full event width (10), so the leaf is
  the word's 10-number code times its signed activation; no position band
  remains. One leaf per word, plus the separator's.
- **Maximal entropy over symbols, at the start.** The six code rows are
  random signed rows of ten numbers. The EMA writer of §19.1 pulls them
  toward the word events, which carry two content numbers (the percept
  dictionary is 256 × 2) and position bands. If it runs, the codes lose
  their spread and come to differ in two numbers: the regime where §3.6
  learned XOR in 1 of 10 and 0 of 10. §3.13 measured exactly such drift
  (needed answer weights 3 at the start, 13 at the end; random codes need
  1.8).
- **Distinct when combined.** Coordinate-wise `min`/`max` over random
  10-number codes gives a distinct root for each pair of words, and the
  four roots are exactly readable by an affine map for 300 of 300 random
  codes (§3.11). So one nonlinear layer, the grammar's `min`/`max`, and an
  affine reader are enough; no further MLP is needed. Two conditions carry
  it: the codes keep their spread, and the chooser uses the same derivation
  for every sentence and keeps both words (§3.12 saw it drift to a
  derivation that lost the second word). Nothing in training enforces
  either today.
- **Invertibility.** `min`/`max` are lossy, but over a known vocabulary the
  pair is identified by search. The free read-back does this search; the
  training inverse does not need to, because it has the witnesses.

### 19.4 Option A, amended

1. **One writer for the codes.** Gate the concept codebook's EMA refresh off
   (`ema_update` false for ConceptualSpace, as WholeSpace's already is).
   The codes then move only by reconstruction's gradient. First measure
   that the refresh runs: `vq.cluster_size` after training, and the drift
   of the six rows from their start.
2. **Train reconstruction as the gate reads back.** The trained inverse
   receives no reference and no witness offsets, and searches the primed
   bank for both operands. The witnessed inverse remains a diagnostic. The
   journal still records the witnesses for the answer's record. A
   derivation that loses a word, or codes that merge, then cost
   reconstruction directly.
3. **XOR_exact.** Its evidence coefficients are its only reader; the answer
   owns them.
4. **Code geometry in the audit.** Pairwise cosines of the six rows and the
   singular values of the four roots, at the start and the end of training,
   so that 1 and 2 are seen to hold.

The rest of §15 stands: expectation detached, the record shared, the
selection rule, the cleanup. If class is still below 8 of 10 with 1 to 4 in
place, fall back to §18 (B) in the same round.

## 20. Option A, decided (Alec, 2026-10-02, night)

Alec: "we have to get reconstruction before we get prediction or output";
reconstruction keeps its witnesses ("I think we can do both at the same
time"); the rotation of codes "is mostly done so that the magnitude of the
input vectors can be read as a certainty match under inner product. We can
drop that also if it might get in the way, but let's create a catalog of
what we want to reintroduce after getting things working."

### 20.1 The law for this round

**The codes have one writer: reconstruction's gradient.** Everywhere,
including BasicModel.

- The concept dictionary is a parameter, trained by reconstruction, with no
  norm constraint.
- The VQ EMA refresh of the concept codebook is off (§19.1). It is
  k-means over perceptual events and carries no co-occurrence; it was a
  second writer the audit could not see.
- The contextual rotation is off: `conceptualContextLearningRate` 0 in the
  eleven canonical configurations, and `conceptualSimilarityScale` stays 0.
  Their codes were buffers moved only by the rotation; they become
  parameters moved only by reconstruction. The contract test
  (`test_canonical_dictionary_has_one_distributional_owner`) asserts the new
  law.
- The leaf stays `code × signed activation`. Without unit codes the
  activation is not a pure certainty; the catalog (§20.3) returns that.
- The read-back's candidate scoring must not assume unit codes.

**Reconstruction has two terms, both its own.** The witnessed inverse, as
today, and the free read-back: no reference, no witness offsets, both
operands found by search over the primed bank, as the gate reads. The second
term is what makes the root carry the words and keeps the codes
identifiable. Both enter the Error registry as relative errors under
`reconstructionScale`.

**`XOR_exact`'s evidence coefficients belong to the answer**: they are its
only reader (§18).

**The audit measures code geometry**: pairwise cosines of the dictionary
rows and the singular values of the four XOR roots, at the start and the
end of training, and `vq.cluster_size` after training to show the EMA
refresh is gone.

The rest of §15 stands: expectation detached and out of the comparison, the
shared record, the selection rule, one writer for every other weight.

### 20.2 The tension to watch

On XOR's four sentences, *hello* and *loving* have the same contexts, so any
distributional pressure pulls them together, and XOR needs them apart. This
round has no distributional pressure, so the gates measure reconstruction
and the reader alone. When distribution returns (§20.3), the XOR table is
the test of the balance between the two.

### 20.3 Catalog: set aside now, to return once reconstruction and XOR hold

| Set aside | Why it existed | Why it is set aside | Returns when |
|---|---|---|---|
| **Unit-sphere codes** (retired in 6.8 §14–§15) | A unit code was intended to make inner products read certainty. | A symbol's form is the coordinate-wise join of its positively evidenced parts at full presence in perception's [0,1] cube. Wholes bound that form; perception owns its prototypes and evidence. The paired concept's order-zero meaning is the detached context mean. Neither face has a unit-sphere constraint. | Not planned to return. §15 also removes the class reader's unit-norm transform: XOR_grammar and the sum control read the raw root through the same output-owned affine head. |
| **Magnitude = certainty** (returned in cube form, 6.8 §15) | Separate a symbol's identity from how certainly it is present. | The code is at full presence; evidence selects parts without scaling their codes. Certainty belongs to the leaf's signed activation, not to differences in the magnitudes of form codes. | Returned as the projection coefficient `(leaf·c)/(c·c)` onto the full-presence code `c`; the leaf is activation times that code. No unit-sphere constraint or antipode loss. |
| **Distributional pressure on the codes** (co-activation: the CBOW-with-negatives rotation; the SBOW gradient term) | A word2vec-like space: co-occurring concepts near, others apart. "They learn a distance metric from the gradient and their co-activation." | The pull-apart half already comes from the free read-back, whose negatives are the primed competitors (§20.1). Only the pull-together half is set aside: a second objective on the codes before the first one works, and on XOR's corpus it merges the words XOR must separate (§20.2). | The gates hold and the primed bank is live. Then, as Alec asks (2026-10-02): a slight SBOW term whose window is the spreading-activation field, co-primed symbols as positives weighted by their activation and unprimed rows as negatives, a gradient term in the Error registry with a weight well below reconstruction's, so that the audit sees it and the XOR table tests the balance. The existing SBOW spine (`conceptual_sbow_loss`) is parallel-only and windows the sentence's slab; it would take the field as its window and run on the serial path. The rotation retires (no legacy). The alternative with no second term: compose a concept's code from its constituents over the store's edges, so co-activated concepts share code structure through the fold. |
| **The primed bank as context and candidates** | The connections bank of symbols (§15.3): the open read primes the words that the words activate. | In the ownership round it held no activated word in 6,400 occurrences: the fixtures' concept stores have few edges. | The store's edges exist in the fixtures, measured by the audit's competitor count being nonzero. Then it is the read-back's candidate set and the distribution's context. |
| **Decoder exploration** (§26.3; implemented, measurement pending) | The generate walk infers operations from the root (§25.1). | Greedy argmax alone did not leave STOP when the missing-word penalty supplied no untaken-transition credit. | Implemented: greedy and a sampled one-departure explore walk are both costed before learning; only strictly lower reconstruction keeps explore. The generate straight-through path remains. The decoder STOP-versus-undo margin and its gradients are audited. The 6.8 §16.3 compose chooser separately uses the detached-cost `p(a_dep)·(C_explore−C_greedy)` surrogate with a hard pair inverse. Mechanism checks pass; the once-only gate campaign is held because the full-sweep supervisor rejects an existing non-strict XPASS. See the [receipt](../benchmarks/2026-10-03-operators-attention/README.md). |
| **The VQ EMA refresh** | k-means over quantized events. | A hidden writer; clusters word forms, not meanings; the "#13 mechanism" in the code's own comment. | Not planned to return. Recorded so that its removal is deliberate. |
| **The answer's reach into the understanding** (step 5a) | 8 or 9 of 10 on the class gate in §14. | It makes the understanding answer-shaped; (B) in §18. | Fallback only, if (A) fails after §20.1 with distribution and the sphere restored. |

**Naming.** The code calls this quantity the priming surface, boosts, heat and relevance; the writes are SEEN and DESIRE priming. Alec asked for a better name (2026-10-02). Recommended: *spreading activation* for the mechanism (Collins & Loftus 1975; Anderson's ACT-R, where base-level activation decays and spreading activation propagates over the network, which is this algorithm exactly), *activation* for the per-symbol value, and *priming* only for the writes. "Field" is already the pooled perceptual reading (accessible mind §2.0), so the state over symbols is better called the symbol activations than a field. The rename is code work across many identifiers; it is a separate proposal, not part of this round.

Cross-references already catalogued elsewhere: reading and global
attention (6.8 plan §7); word-level XOR for MM_xor (6.8 plan §8);
multi-resolution `.where` tiling and ergodic exploration everywhere
(FutureWork).

### 20.4 Order of work and measurements

1. §20.1, with the remaining ports: 23 sweep failures and the 48 failed and
   12 stopped moved cases of the ownership round.
2. §15.5's set once: the gates ×10 each, the sum control ×10, the named XOR
   table, MM_grammar ×10, one source-matched full sweep, the ownership audit
   with code geometry on XOR_grammar and `BasicModel_answers_tied_benchmark`,
   the production stage-1 slow test, the moved and MNIST cases.

The XOR table must be at the accepted baseline except MM_xor's
`test_convergence` (§17) and MM_grammar's known stop at .25 (§16.1).
Comparison: §14's class 8/10 and reconstruction 3/10. The attribution runs
only if class is below 8/10 or reconstruction is not above 3/10.

**Reading the results (Alec, 2026-10-03).** "Affine readers will get .25 or
.0 depending on starting point; for them, that is an XOR success. MM_grammar
is expected to have no error." So the class runs are read as bimodal: a run
ending at 0 has learned XOR; a run ending at .25 sits at the symmetric fixed
point of item 11; a run ending between the two is the reader chasing a
moving derivation (§3.12), which is a failure of the understanding's
stability, not of the reader. MM_grammar's ten runs are expected at zero.

### 20.5 Success criteria for an affine reader on XOR

Alec, 2026-10-03: "I need you to understand the success criteria for affine
transformations on xor. There is a lot of literature about that." The
criteria used in every review from here on:

**The exact floor.** On the four corners the XOR target (0,1,1,0) is
orthogonal to the affine span: its centered form (−½,½,½,−½) has zero inner
product with the centered inputs (−½,−½,½,½) and (−½,½,−½,½). The best
affine predictor is therefore the constant ½, with squared error exactly ¼
(Minsky and Papert 1969, in least-squares form). The problem is convex, so
descent reaches ¼ from every start. An affine reader over an *additive*
representation never does better; the `sum` control is this theorem as an
experiment (observed .250 to .264, checkerboard contrast at most 6e-8).

**The bimodal outcome.** With one nonlinear layer under the reader the
global minima are at 0 (Rumelhart, Hinton and Williams 1986). There is also
the symmetric stationary configuration, hidden units interchangeable and
output ½ everywhere, error ¼. Blum (1989) and Lisboa and Perantonis (1991)
took it for a local minimum of the 2-2-1 network; Hamey (1998) and
Sprinkhuizen-Kuyper and Boers (1998) showed the finite stationary points are
saddles, with minima only at infinity; Fukumizu and Amari (2000) generalized
them to singular regions where symmetry makes the gradient vanish. Descent
sits on the ¼ plateau for a long time, or forever from the exact point. Item
11 found the same here: the zero-initialized symmetric fixed point with π
gradient identically zero, escaped by any unseeded start. So a run ends at 0
or at ¼ by its basin; the fraction at 0 and the epochs to reach it are the
statistics; the ¼ runs are a rate to report.

**The criteria.**

| Final answer error | Reading |
|---|---|
| 0, within tolerance (the gate's .05 bar) | XOR learned: the composition made the four roots separable and the reader found it |
| ¼ | The plateau. Expected at some rate under unseeded starts. A defect only if every run lands there, which means the start is the exact symmetric point |
| Between 0 and ¼ | Not a stationary point of an affine reader over a fixed representation: either the budget ended mid-descent (the trajectory shows it) or the representation moved under the reader |
| MM_grammar's direct-forward harness | Trains the nonlinear path end to end: every run at 0; ¼ is the plateau; any other value is a defect |

**This round's ten class runs, so read:** none at 0; four on the plateau
(.264, .262, .260, .249); six between (.060, .098, .146, .213, .215, .235).
The six are the moving-target reading of the audit, so the fault is the
understanding's stability under the reader, not the reader.

## 21. Review of the §20 round (Claude, 2026-10-03)

[Receipt](../benchmarks/2026-10-02-item6-9-reconstruction/README.md). Not
accepted. Codex says so itself, and records the failures without reruns.

### 21.1 What held

- **The law held.** Zero ownership conflicts over 1,200 backwards; the codes
  have one writer; VQ cluster counts stay at 1, so the EMA refresh is gone.
- **Reconstruction trains.** Its weighted cost falls from .137 to .0015 over
  the run; in the audited run the free read-back is exact at the end.
- **The codes stay in general position.** Norms 1 to 2.35, pairwise cosines
  spread, three nonzero singular values for the four roots, at the start
  and at the end. So the understanding is readable in principle.
- **`XOR_exact` passes again.** Both checks, and the MM_20M exact round
  trip.
- **MM_grammar: 9 of 10 at zero, 1 on the plateau** (.2500589). By §20.5
  that is the expected shape.
- **The sweep: 8 failures in 22 minutes on ten workers**, all 23 of the
  previous round's failures fixed. The remaining 8 are seven unfinished
  ports and one Dynamo capture limit in the new free branch of
  `_reconstruct_sentences`.

### 21.2 The answers, read by §20.5

Every run's final answer error, in bands: at 0 (below the .05 bar), at ¼
(within .02), between, or above ¼.

| Runs | At 0 | At ¼ | Between | Above ¼ |
|---|---:|---:|---:|---:|
| Class gate | 0 | 5 | 5 | 0 |
| Reconstruction gate | 0 | 4 | 4 | 2 |
| Sum control | 0 | 10 | 0 | 0 |
| Attribution, R (reader untrained) | 0 | 10 | 0 | 0 |
| Attribution, R+E (reader untrained) | 0 | 10 | 0 | 0 |
| Attribution, R+A | 1 | 4 | 4 | 1 |
| Attribution, all three | 2 | 1 | 7 | 0 |

Three readings:

1. **The theorem holds where it should.** The sum control and the two arms
   with an untrained reader sit at ¼ to three decimals: the affine floor,
   twenty times out of twenty.
2. **The basin at 0 exists.** Three runs reached it (.032, .043, .021), so
   the composition does make the four roots separable and the reader can
   find it.
3. **Most runs with a trained reader end where no stationary point is.**
   Of the 40 runs that trained the reader, 20 end between 0 and ¼ and 3
   above ¼, worse than the constant predictor. These are not slow descents:
   the audit's trajectory shows the greedy trial's per-row answer cost
   swinging by 10× within the last 50 epochs, and explore trials kept in
   18 to 52 percent of rows. The representation moves under the reader.

So the round's finding is not about the reader, and not about the law. It
is that reconstruction, which now owns the understanding, gives the
derivation no reason to settle, and the comparison between trials lets the
answer's noise change it:

- Reconstruction's cost is near zero whichever derivation is used, so its
  gradient to the chooser is tiny (3e-7 in the exploit trial), and Adam
  turns a tiny noisy gradient into full-size steps: the policy random-walks.
  (Inferred from the gradient norms and the optimizer; not measured
  directly.)
- The trial comparison is reconstruction-plus-answer. With reconstruction
  near zero in both trials, the answer decides, and the answer varies with
  the derivation by chance, so explore wins often (519 of 1,600 rows).
- The reader is trained on both trials' roots every epoch, so half its
  updates come from a derivation it will never see at evaluation.

### 21.3 Reconstruction's own failures

The reconstruction gate passed 2 of 10. In the failing runs the read-back
duplicates one word: "world world / there there", "hello hello / loving
loving". One word never enters the root. The free read-back term should
make that costly, but at `reconstructionScale` .1 its pull on the chooser is
small (median relative .52 in training, weighted .05), while the answer's
weighted cost (.25) dominates the comparison between trials. The witnessed
term, which is exact whatever the derivation, carries most of
reconstruction's weight and none of its pressure.

### 21.4 Next, within A

The understanding must be made stable by reconstruction alone, before the
reader is asked to read it:

1. **Selection by reconstruction only.** Keep explore only if its
   reconstruction is strictly lower; a tie keeps greedy. The answer leaves
   the comparison. This is "reconstruction before output" applied to the
   choice as well as the gradient.
2. **The reader trains on the kept trial only.**
3. **The free read-back is the reconstruction term that matters.** Give it
   reconstruction's weight; the witnessed term stays as the exact inverse
   for the record and the generate walk, at a lower weight or as a
   diagnostic. Alec to confirm, since §20.1 kept both at the same weight.
4. **Measure the derivation's stability directly.** Record each sentence's
   rule sequence per epoch in the audit, and report the fraction of epochs
   on the modal derivation and the number of distinct derivations per
   sentence. This is the quantity §3.12 and this round infer from the
   answer's trajectory.
5. The eight sweep failures and the moved-case ports.

Expected under §20.5: class runs at 0 or ¼ only, with the fraction at 0 the
success rate; MM_grammar at zero but for the plateau; the derivation
stability near 1.

### 21.5 Item 3 in detail (for Alec, 2026-10-03)

**The two reconstruction terms.** Each undoes the recorded derivation from
the root, one operation at a time, and scores each recovered leaf by the
bytes of the word it names.

| Term | What the inverse is given | What its cost can say | Measured |
|---|---|---|---|
| Witnessed, `reconstruction.bytes` | One operand's retained reference and the journal's witness offsets: what `min`/`max` discarded | Nothing about the derivation or the codes: with the references and offsets the inversion is exact algebra. A consistency check of the tied inverses | relative .011 at evaluation, .019 median in training |
| Free, `reconstruction.free_bytes` | The root and the rule sequence only; both operands found by search over the primed bank, scored as activation × cosine × priming | Whether both words entered the root, whether the codes are tellable apart, whether the composition kept the pair. It is what the gate measures | relative .52 median in training; 0 at the end of the audited run |

Both carry `reconstructionScale` (.1). Since the witnessed term is near
zero, the free term already dominates reconstruction's gradient; what the
witnessed term adds is its own noise gradient on the same parameters, and a
"reconstruction" total that looks solved while the gate fails.

**3a.** Reconstruction trains on the free read-back alone. The witnessed
inverse stays as the record's exact inverse for generation and as a
diagnostic, untrained.

**The deeper problem: indifference under Adam.** Once the free read-back is
solved, reconstruction is indifferent among every derivation that keeps both
words and among many placements of the codes. Its gradient to the chooser
falls to about 3e-7. Adam divides a gradient by its own running magnitude,
so a consistently tiny, noisy gradient still produces a step of about the
learning rate in a random direction: over 1,200 backwards, a random walk of
order .3 per coordinate on 10-number anchors, enough to flip the argmax. The
codes are in the same position; their norms grew from 1 to 2.35 with nothing
asking them to. The derivation and the roots therefore drift under the
reader. In §14 the answer's gradient gave these weights a direction; in
§3.11's isolation they were fixed; in this round they had neither. Inferred
from the gradient norms and the optimizer's rule, not measured; the
measurement is each weight's displacement per step against its gradient,
which Codex's FutureWork note already proposed.

**3b.** A weight whose objective is at its floor must not move.
Reconstruction's owner uses an optimizer whose step scales with the gradient
(plain gradient descent, with momentum if wanted), not Adam; the reader keeps
Adam, since its objective is at the floor only once XOR is learned. Record
displacement against gradient for the chooser anchors and the codes.
Alternative for this round: the measurement alone, with the fix decided on
its result.

Items 1 and 2 of §21.4 stand: they remove the other two sources of
movement, the answer's noise in the comparison and the reader's training on
the explore trial.

### 21.6 Witnesses, operators, and the field (Alec, 2026-10-03)

Alec: "the journal's witness offsets (what min or max discarded) should not
be preserved; the idea is for conceptual space to preserve what is required
for decoding back into symbols. Min and max might be less good than product
and mean, which is looking more like AND and OR (or some other t-norm that is
more easily invertible). The other option would be to preserve a field of
AND and OR on the concept codebook, so that what is stored in conceptual
space is not a vector but a set of activations on known concepts. That would
be a fairly big change, however, and I have some hope that higher-order
concepts will already store this information implicitly."

**Witnesses (decided).** The journal keeps no witness offsets, and the
understanding record has none (amending §15.3 and §20.1). Reconstruction is
the free read-back alone: from the root and the rule sequence, over the
primed bank. The understanding carries what decoding needs. The journal
reduces to the rule sequence and operand positions.

**Operators (analysis; decision open).**

| Operation | Invertible given one operand | Gradient | XOR readable by an affine map |
|---|---|---|---|
| `min` / `max` | No: where the known operand won, the other is only bounded (hence the witnesses) | Only to the operand that won each coordinate | Yes, generically (§3.11) |
| Product | Yes: divide out, except at zero | To both operands | Yes: products of random codes are in general position |
| Mean | Yes: subtract | To both | **No**: (h+w)+(l+t) = (h+t)+(l+w); the four roots are affinely dependent and the floor is ¼. The `sum` control is this case |
| Probabilistic sum, a + b − ab | Yes | To both | Yes, bilinear |

Product is the better AND for decoding and for learning; mean is not a usable
OR for XOR, its dual the probabilistic sum is. Cautions: product is not
idempotent (a·a = a²), against the operator catalogue's idempotent
intersection; the t-norm reading needs codes in [0, 1], while on signed codes
product per coordinate is XNOR. XOR_grammar is the gate of the operators
update, so this is that update's first decision. Recommended: baseline with
the current operators first (the dropped-word failures are the derivation's,
and `min`/`max` roots are pairwise distinct), then product and probabilistic
sum under the gate.

**The field (noted).** The closing's row already holds references to its
words' concept rows (two truths), so decoding a sentence from its row is
exact without inversion: the higher-order concept stores its parts
implicitly, as Alec hopes. The gate tests the vector, which is what the
answer reads; both stand. A set of activations over known concepts is where
item 6.8's open read goes (presences over concepts, with the field's AND/OR
over them), so it arrives with 6.8 rather than as a separate change.

## 22. Decisions of 2026-10-03: witnesses gone, product and mean, the optimizer

Alec: "A) operators now; I think that if 'the t-norm reading needs codes in
[0,1]' we should be using the symbols of the concepts, which are already on
0..1. B) reconstruction can change now, unless accepting the current item
can happen without it." And on idempotence: "not a must have ... I would
like to have it, though. Is it possible that adjectives can be construed as
an operation which operate on the same noun twice, and that is why they are
idempotent?"

### 22.1 Idempotence

Hawaiian adjectives are not idempotent at the surface: reduplication is
productive and intensifies or pluralizes (*nui* → *nunui*, *liʻi* →
*liʻiliʻi*, *wiki* → *wikiwiki*; Elbert and Pukui, *Hawaiian Grammar*).
English "big big", Italian "piano piano" and Indonesian plural reduplication
are the same pattern. Repetition is a degree or number operator, not a
second AND.

Alec's construal reconciles the two: an adjective restricts the noun's
extension, and the second application acts on the restricted extension and
removes nothing. A projection is idempotent. Reduplication is a different
operator applied to the first, the catalogue's adverb-like multiplicative
one. So intersection stays idempotent as semantics.

Among t-norms only `min` is idempotent; product gives a·a = a². In
probability, product assumes independence, and the same predicate twice is
maximally dependent (P(A∧A) = P(A)). Rule: conjunction of the same reference
is that reference; conjunction of distinct references is the product.

### 22.2 The operators (A, decided)

A leaf is `unit code × activation`, the activation in [0, 1] the certainty.
The elementwise product of two leaves multiplies the certainties (a product
t-norm on [0, 1]) and binds the two identities into a vector distinct for
each pair, dissimilar to both parts, invertible by division or by search:
binding in Vector Symbolic Architectures (Plate 1995, 2003; Gayler 2003,
MAP: multiply, add, permute; Kanerva 2009), where bundling is the mean and
clean-up memory is search over known items. The product of two unit codes
has norm about 1/√D, so the identity is renormalized and the certainties
carry the magnitude.

- `conjunction(x, y) = ‖x‖·‖y‖ · unit(x ∘ y)`; the same reference → `x`.
- `disjunction(x, y) = (x + y) / 2`.
- `not(x) = −x`, unchanged.

Mean is additive: a derivation of disjunctions alone is the `sum` control
and reads XOR at ¼. XOR is computed where the grammar chooses conjunction;
§3.6 learned it 9 of 10 with `not`, sum and product. Gradient flows to both
operands of a product, where `min`/`max` trained only the winner of each
coordinate. All three faces of both operators change (compose, generate,
reverse), in every configuration on `complete.grammar`, BasicModel included;
the stage-1 slow test measures that. Operand order is still not encoded (the
gate accepts transpositions); MAP's permutation is the catalogued answer if
it is ever needed.

### 22.3 Witnesses and the optimizer (B, decided)

- No witness offsets in the journal or the record (§21.6). Reconstruction is
  the free read-back only.
- Reconstruction's owner trains with an optimizer whose step scales with the
  gradient (gradient descent with momentum); the reader keeps Adam
  (§21.5, 3b). The audit records each weight's displacement per step against
  its gradient for the chooser anchors and the codes.
- Selection by reconstruction only, strictly lower, ties to greedy; the
  reader trains on the kept trial only (§21.4, 1 and 2).
- Derivation stability is measured: per sentence, the fraction of epochs on
  its modal derivation and the number of distinct derivations.

Expected under §20.5: class runs at 0 or ¼ only; MM_grammar at zero but for
the plateau; stability near 1.

**Amendment (Alec, 2026-10-03):** "we can offer multiple operators and let
the UG decide, so let's keep the min and max as self-titled operators for
inclusion in a later grammar." So `min` and `max` stay in the operator
catalogue under their own names, with their three faces and a unit test for
each, offered to any grammar file and used by none of the current ones
(`complete.grammar`, XOR_grammar's and MM_grammar's rules move to the product
and the mean). A catalogued operator with tests is not a legacy path: the
grammar chooses among the operators offered, and a later grammar may offer
these.

**One training, both bars (Alec, 2026-10-03: "Are there separate 'class
runs' and 'reconstruction runs'? That would be weird.").** The two gates are
two test functions that each train their own model, so every receipt has run
twenty trainings and counted one bar on each ten. Every run yields both
quantities. From this round on the two tests share one training (a
module-scoped fixture) and assert their own bars on the same model; the
campaign runs ten trainings and reports class, reconstruction, and both in
the same run, which is the statistic that matters (this round: 0 of 20
jointly; the attribution: 0 of 40). The bars are unchanged; the first run
supplies both XOR-table rows.

## 23. Review of the §22 round (Claude, 2026-10-03)

Receipts: [free read-back, operators, stable ownership](../benchmarks/2026-10-03-item6-9-free-readback/README.md)
(the measurements) and [one training, both bars](../benchmarks/2026-10-03-item6-9-shared-gates/README.md)
(the consolidation; nine partial shared runs: class 8/9, reconstruction 3/9,
both 2/9; not re-measured, by Alec). Not accepted; close.

### 23.1 Results by §20.5

| | §22 round | §20 | §14 |
|---|---|---|---|
| Class | 9/10: nine at 0, one between (.058) | 0/10 | 8/10 |
| Reconstruction | 5/10 | 2/10 | 3/10 |
| Sum control | 10/10, all at ¼ | 10/10 | 10/10 |
| MM_grammar | 9 at 0, 1 at ¼ | 9 + 1 | 8 + 2 |
| Named XOR table | 33/34 (MM_xor by §17) | 31/34 | 30/34 |

Sweep 5,000 cases, 8 failures, 19.5 minutes on ten workers; moved cases
159/165; all attention and MNIST cases pass; XOR_exact and the MM_20M round
trip pass. The class pattern is the predicted one for the first time.

### 23.2 What the result is

In the audited run the largest code gradient was 1.6e-10 and the largest
chooser-anchor gradient 1.2e-12; every code and anchor displacement was
exactly zero over 800 updates. The nine runs at 0 are therefore the affine
reader over a frozen random composition, product in place of `min`: §3.11
reproduced in the real fixture. A stable nonlinear composition and an affine
reader learn XOR. Reconstruction trained nothing: its cost was .986 relative
with no gradient.

### 23.3 Two defects in the free read-back

1. **Dead gradient.** The byte scorer (`Models.py`, `_BYTE_ASSIGNMENT_TAU`
   = .1) takes a sharp softmax over the candidates with the null column's
   logit at 0, so the null mass underflows; builds the target byte's
   probability from the assignment; and clamps it at 1e-6 before the log.
   When the wrong candidate wins, the correct byte's probability lies below
   the clamp and its derivative is zero. In the MM_ladder free derivation,
   548 of 884 target probabilities were below the clamp (0 of 64 recovered).
   Fix: compute the log probability in log space (log-sum-exp over the
   emitting candidates minus the normalizer), no clamp. Numerics, not
   tuning.
2. **Unaries on operands are not unwound.** For
   not(conjunction(not(l), t)) the search looks for a pair of raw codes;
   not(l) is not one, so it returns (loving, loving). Reproduced by Codex
   from the saved roots to 3e-8; it accounts for the repeated-word
   read-backs. The inverse must follow the recorded derivation, including a
   `not` applied to an operand. Product roots are then exactly recomposable
   from the bank.

### 23.4 Production regressions and ports

From the operator change: the free-search branch intercepts the sum inverse
that answer generation uses (two cases); the truth store's falsity penalty
calls the grammar's disjunction for a set union and gets a mean (two cases;
the union is the catalogued `max`/union, not the grammar's disjunction); a
compiled byte score drifts 4.6e-5 from eager. Two optimizer-state observers
expect Adam state on reconstruction's weights; four moved fixtures; the
one-batch chooser test sees no movement (consistent with 23.2). MM_ladder's
serial supervised case ends at .969 against 1.0 after passing in the focused
check: one run, unresolved.

### 23.5 Open for Alec

Codex kept temporary numerical frames in the journal for item 7's clause
closing ("the operation actually performed, including non-replayed relation
semantics"), cleared at the closing, read by neither reconstruction nor the
answer. Not witnesses for decoding, but the journal is not rules and
positions only. Recommended: accept, with the closing's need written down.

### 23.6 Next round

The two read-back fixes (23.3), the production regressions and ports (23.4),
then the measurements once with one training and both bars. With the
gradient live, reconstruction moves the codes for the first time; under
momentum SGD the movement should stop as the invertible composition
converges. That is the real test of §20 to §22.

### 23.7 Decided (Alec, 2026-10-03)

- **The journal:** "If there is no use of the operations that are performed,
  please do not write them down." A frame is written only where the clause
  closing reads it; the receipt names what the closing reads and why. Nothing
  unread is written.
- **The corrections of §23.3 and §23.4 are made now, with minimal testing:**
  the focused tests for each fix, the eight sweep failures and six moved
  failures being fixed, one shared XOR_grammar training with its audit, and
  collection and documentation links. No ten-run gates, attribution, full
  sweep or native run.
- **The extensive round moves to the next check-in.** After 6.9 the sequence
  is the grammatical operators update (the rest of the catalogue; AND, OR
  and NOT are already changed here), then 6.8. (The conference freeze that
  stood first was dropped by Alec on 2026-10-03.) The
  operators update is XOR-gated, so the full campaign fits there.

## 24. Review of the corrections round (Claude, 2026-10-03)

[Receipt](../benchmarks/2026-10-03-item6-9-corrections/README.md). The §23
corrections landed: the dead gradient is gone (1,682 wrong read-backs, all
with nonzero gradients), the unary-aware inverse recovers all four saved §22
roots, generation and the truth penalty are repaired, the eight sweep
failures and four of six moved cases pass, the journal writes only what the
closing reads. Minimal testing, as decided; not an acceptance round.

### 24.1 The one training, by §20.5

Class at ¼: answers .4999 to .5000, MSE .2500, a clean plateau. Read-back
2/4 ("world world", "there there"). Derivation stability 95 to 100 percent
per sentence: the moving target of §21 is gone. Final relative
reconstruction error .06.

### 24.2 What the live gradient did

The dictionary moved by 1.57 in norm over 800 updates. The cosine of `world`
and `there` went from .36 to 1.0; the centered root singular values from
(1.19, .80, .54) to (2.03, .0002, .00007). Reconstruction trained the codes
and merged the pair XOR needs apart; the understanding collapsed onto a line.

### 24.3 Why reconstruction allowed it

The read-back score is activation × cosine × priming (§20.1, Claude's form),
and the primed bank is snapshotted after the sentence's own seen-write. So
the own words carry the highest priming, `there` barely competes in "hello
world" and `world` barely competes in "hello there", and the read-back never
has to tell them apart by code. Nothing in reconstruction then resists their
merging; reconstruction stays at .06 while it happens. The prior did the
read-back's work and removed the pressure that was to keep the codes
identifiable.

### 24.4 The fix (proposed)

Priming chooses the candidate shortlist, as Alec intended (the words those
words activate), and does not enter the score. The read-back scores by
activation × cosine over the shortlist; the cross-entropy over candidates
then pushes competing codes apart. Alternative: snapshot the priming before
the sentence's own write; it also removes the shortcut, but in XOR_grammar
nothing is then primed. Recommended: the first. Minimal testing again: the
focused test, one shared training with the audit; the quantities to watch
are the `world`/`there` cosine and the root spectrum.

Still red, undiagnosed this round: MM_ladder's serial supervised case (.953
against 1.0) and its free round trip (0/64 after three epochs).

### 24.5 What decoding needs, and where each part comes from (Alec and Claude, 2026-10-03)

Alec: "knowing which words are involved does not give us the word order, and
it does not give us which operations were used to combine the words: both of
those things must be known to do the decoding." Today:

| Needed | Comes from |
|---|---|
| Which operations | The journal: the recorded rule sequence and operand positions. The inverse follows them; it does not infer them from the root. |
| Word order | Nowhere. Product and mean are commutative and associative, so the root carries no order; the gate accepts transpositions for this reason. Order would come from a non-commutative binding (MAP's permutation, catalogued). |
| Which words | The root, through the codes; or, when the read-back is weighted by the sentence's own priming, the prior. |

When the prior supplies the identities, the root has no decoding work left,
reconstruction is satisfied whatever the codes are, and the codes merge
(§24.2). The answer reads identities from the root and cannot shape the codes
itself, so reconstruction is the only pressure keeping them distinct. Hence
the snapshot before the sentence's own write (§24.4): the shortlist stays the
activations, weighted; the identities come through the codes.

Encoding order in the composition is a separate step; it would let decoding
recover order and the gate stop accepting transpositions, but it would not
keep codes apart while the bag answers "which words".

## 25. Decoding from conceptual space, and closing 6.9 (Alec, 2026-10-03)

Alec: "we are still using a generate grammar, and it's within the context of
that grammar that a word choice must be made. Choosing that word from
recently active words is the right idea; reconstruction based on echoic
memory should not be a problem, although ultimately it is true that we want
the primary source of the words to be their representation in conceptual
space. ... we need to infer operations from the root: that's the whole
reason that generate (and think) exist as collections of operators. I think
we should code this up, where reconstruct() and output() are decoding from
the understanding left in conceptual space, and move to later items where
our ability to deliver on these items will improve (I'm not sure we have
everything we need in 6.9)."

### 25.1 One decoder

- `reconstruct()` and `output()` are the generate grammar walking back from
  the understanding in conceptual space (root and end slots). At each step
  the generate chooser infers the operation to undo from the root; the
  journal's rule sequence is not given to decoding. A binary undo produces
  its two children by search over the shortlist; a unary undo applies the
  operator's generate face.
- **Echoic memory.** The shortlist is the primed symbols, the sentence's own
  words included, weighted by their activation. The pre-write snapshot of
  §24.4 is withdrawn.
- Each recovered leaf is scored against the input bytes with the log-space
  byte loss (§23); the gate's read-back is the same walk.
- **Ownership, one writer.** Reconstruction teaches the decoder: the generate
  chooser, the operators' generate faces where parameterized, the codes.
  Output steers it: the question conditioner in `answerSynthesis`
  configurations; in XOR_grammar the numeric head over the root, the end
  slots and the echoic shortlist. The answer's record carries no derivation
  fields.
- The journal keeps only what the closing reads (§23.7).

### 25.2 Expectation on record

With echoic memory supplying the words, nothing yet requires the codes to
stay distinct (§24.3), so the XOR class gate is expected to stay at ¼ until
identities come from conceptual space. That is the "not everything we need
in 6.9". It joins the catalog (§20.3) with the order marker (§24.5).

### 25.3 Closing 6.9

After this round's single measurement, 6.9 closes as the baseline. The
receipt and todo record what passes and what is red and why: MM_xor by §17;
XOR_grammar's class and reconstruction gates as measured. The standing XOR
rule for later items is no regression against that record, with the catalog
naming what each later item is expected to turn green: the operators update
(the rest of the catalogue; identities from conceptual space), 6.8 (the open
read; word-level XOR for MM_xor), the surface markers for order. Next in the
todo: the operators update, 6.8 (no conference freeze; Alec, 2026-10-03).

### 25.4 What merges the codes, and the antipode (Alec, 2026-10-03)

Alec: "What forces them to come together? We had worked out a plan
previously that moved non-selected codes toward an antipode to ensure even
spherical distribution, and that should still work."

**The merging force (inferred).** The decoder's search is soft: the
recovered leaf is a residual-weighted blend over the candidate pairs. In
"hello world" the pairs (hello, world) and (hello, there) have similar
residuals, so the recovered second leaf lies between `world` and `there`,
and the byte loss pulls `world` toward it, that is, toward `there`; "hello
there" does the reverse. Each step brings them closer and the blend more
even. The echoic prior keeps the own word winning the byte softmax, so the
loss stays low. Verifiable from the audit's saved code-gradient coordinates
without training: `world`'s gradient should point toward `there`'s code.

**Why the read-back's softmax does not repel.** A competitor's push is
scaled by its assignment weight, which the prior makes tiny for exactly the
competitor that matters.

**The antipode term, inside reconstruction.** At each decoded leaf the
selected word's code is aligned with the recovered leaf, and every other
shortlist code, plus the usual hashed random rows, is pushed toward the
leaf's antipode (cosine toward −1), in the existing SBOW negative form
(`embed.conceptual_sbow_loss_codes`'s negative half; the rotation's
`negative_pull`), registered as a relative error (baseline log 2) under
`reconstructionScale`, owned by reconstruction. One writer. This is the
pull-apart half of the catalogued distributional pressure; the pull-together
half stays catalogued (§20.3), since on XOR it would merge *hello* and
*loving*.

Expectation revised from §25.2: with the antipode term, codes should stay in
general position, and the class gate may leave the plateau.

### 25.5 Activation and location (Alec's question, 2026-10-03)

Alec: "what is the relation between spreading activation between concepts
and the distance between codes in conceptual space? I see activation being a
short-term version of the location, which suggests that it may be used
long-term to train location (which would allow us to decrease connection
strength between unrelated ideas and thus avoid collapse)."

- Spreading activation is a diffusion over the store's edges with decay,
  a_t = γ·D·a_{t−1} + bumps. The pattern one concept induces over all V rows
  is its row in (I − γD)⁻¹: the successor representation (Dayan 1993), the
  expected long-run activation it causes. Distance between such signatures
  is diffusion distance (Coifman and Lafon 2006). Activation is therefore a
  position in the space of concepts; short-term because it is a state.
- Location is that position made long-term and low-dimensional: codes whose
  inner products reproduce the signatures' similarities embed the diffusion
  operator (Laplacian eigenmaps, diffusion maps; node2vec/DeepWalk as the
  learned form, which factorizes a diffusion matrix, Qiu et al. 2018, as
  word2vec factorizes PMI, Levy and Goldberg 2014). Stachenfeld, Botvinick
  and Gershman (2017): the hippocampal predictive map learned from transient
  activation. Fast versus slow weights (Hinton and Plaut 1987) is the same
  division.
- Two halves. Co-activated concepts drawn together (positive); never
  co-activated concepts pushed toward the antipode (negative). Collapse is
  attraction without repulsion. The negative half is Alec's "decrease
  connection strength between unrelated ideas", in the codes; it would have
  prevented the `world`/`there` merge, since those never co-activate (§25.4
  installs it inside reconstruction). The positive half needs a self term on
  XOR's corpus, where *hello* and *loving* share all contexts; the SR has it
  on its diagonal, and reconstruction supplies it here.
- With location trained from activation, the shortlist (by activation) and
  the nearest codes (by location) agree; in the collapsed run they disagreed.

The catalog's "spatial representation of the priming field" row (§20.3) is
this: location trained from activation, negative half now, positive half
later with a self term.

## 26. Review of the closing round (Claude, 2026-10-03)

[Receipt](../benchmarks/2026-10-03-item6-9-closing/README.md). The §25
implementation is as decided: one generate walk decodes for reconstruction
and output; the generate chooser infers the operation from the root and never
sees the journal's rule sequence; the shortlist is the echoic priming; the
antipode term is inside reconstruction; ownership has zero conflicts with the
generate policy's parameters written by reconstruction alone; the answer's
record carries no derivation fields. The saved-array check confirms §25.4's
merging mechanism in its angular form (descent toward `there` on `world`'s
tangent plane in 774 of 800 steps). 29 focused cases pass; collection 5,019;
documentation links 272. Nothing committed.

### 26.1 The one training, by §20.5

| | Result |
|---|---|
| Answers | .275, .637, .694, .397: all four on the right side; contrast −.66 |
| Class | MSE .115, *between*; bar red |
| Reconstruction | 0 of 4; the decoder emitted one word per sentence |
| Derivation stability | .885, .83, .44, .44; 6 to 7 distinct sequences for the last two |
| Codes | `world`/`there` cosine .15 → .62; `hello`/`loving` −.05 → .52; mean squared off-diagonal cosine .07 → .37; roots' centered singular values (.74, .59, .40) → (.52, .18, .15) |

XOR is being learned (the sides and the contrast), but on a moving
representation, so the run ends between the two stationary points.

### 26.2 Two causes, one root

1. **The decoder has no exploration.** All 3,200 decodes chose STOP at the
   first step and emitted a single leaf. The chooser's weights moved, but a
   hard argmax trained through a straight-through surrogate cannot discover
   that undoing the binary operation would produce the missing word: the
   missing-word penalty does not depend numerically on the transition that
   was not taken, so no gradient points at it. Compose solved the same
   problem in item 7.5 with an exploit and an explore trial; the generate
   walk has only exploit. Operation inference from the root therefore stays
   unproven, which is the "not everything we need in 6.9".
2. **The codes converge, despite the antipode.** With one emitted leaf, that
   leaf is the root itself, so the alignment half of the antipode term pulls
   the selected word's code toward the sentence's root; roots that share a
   word are alike, so `world` and `there` drift together through `hello`.
   This follows from 1: once the decoder undoes to the words, the emitted
   leaves are the words and the alignment is to themselves.

### 26.3 For the catalog

**Decoder exploration**: the generate walk gets an explore path, one
departure from the greedy walk, kept when its reconstruction is strictly
lower, as compose has from item 7.5. First work of the operators update,
before identities from conceptual space can be asked of the codes.

### 26.4 The closing stands

By §25.3, 6.9 closes on this baseline. The record is complete: this run's
two red gates with their causes; the §22 measurement (class 9/10,
reconstruction 5/10, sum control 10/10, table 33/34) as the prior; MM_xor red
by §17; the no-regression rule for later items; the catalog. Next: the
operators update, then 6.8; no conference freeze (Alec, 2026-10-03).
