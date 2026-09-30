# Item 6.9: XOR_grammar, the baseline of grammatical learning

> **Status:** plan, written by Claude on 2026-09-29 from Alec's request of
> that day and from probes run that day on a copy of the working tree (item
> 7's candidate, round 3, copied at 14:59). Nothing in the repository was
> edited for the probes. Alec answered the six questions of §6 on
> 2026-09-29 and 2026-09-30. **Ready for Codex** after item 7 is accepted
> (the hand-off is §8).
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
   XOR being learned (§3.6, last row).
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
   and Alec decides.
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

## 7. The probes

In the session's scratchpad (`xor69/`), not in the repository:
`run_gate.py` (the gate as the test runs it, reading the four answers);
`inspect_after_run.py` (the derivation and the roots after training);
`probe_grads.py` (which parameters each objective reaches);
`probe_pushes.py` (what the loop pushes, and which reverses run);
`probe_head_reads_idea.py` and `probe_leaf_code.py` (the patches of §3.7);
`unit_grammar.py` and `unit_reverse.py` (§3.6 and §3.9); `possible.py`
(§3.11); `probe_track.py`, `probe_exploit_only.py` and
`probe_freeze_chooser.py` (§3.12).

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

