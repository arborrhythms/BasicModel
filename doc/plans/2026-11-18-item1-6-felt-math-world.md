# Item 1.6: the number modality, the math corpus, and the felt world that combines them

*Claude, 2026-11-18. A design for Alec. No text for Codex: §9's questions
are open, and the item is built after item 1.7 lands.*

Alec, 2026-11-18, placing the item: "Then 1.6 becomes the number modality,
the math corpus (it needs more development), and the work you called out as
"unnumbered: felt world for the math corpus" which combines both."

Alec, 2026-11-18, starting it:

> Please begin 1.6 "The number modality, the math corpus, and the felt
> world for the math corpus, which combines them."
>
> In particular, a detailed spec for how we can teach a synthetic math
> corpus in which human language describes numerical (continuous)
> percepts.
>
> That involves a math corpus, continuous perception (partially
> implemented, I think), and how multimodality changes the picture of
> word-concept processing that we are currently doing.

And of the test it is for (2026-11-18; in full in the item 1.7
specification, `2026-11-14-sigma-pi-layers.md` §19.26): "we should make a
model test that matches words to objects, where objects are
feeling-intensities within a 3-D space. So a sentence like "one was on top
of two", even though it doesnt make a lot of sense, will still describe a
world that is phenomenally accessible to the model, an in which the
feeling of one is actually above the feeling of two."

**Its place.** After item 1.7, before item 1.5. At its submission the slow
tier (`RUN_SLOW`) is run once, overnight (1.8 §72). The architecture
changes no more after this item.

**How to read this document.** What Alec has decided is given with its
date. Everything else is marked *Claude's reading* or *default* and is his
to change. The source was read on 2026-11-18 at `1ccc8cce3f`, which is
item 1.8's tree; item 1.7 will change part of what §2 describes, and §2.5
says which part.

**Three words used throughout.**

- A *scene* is what is felt with one sentence: some felt values, each at a
  place.
- A *level* is the kind a felt value is perceived as. Alec, 2026-11-13:
  "all input intensities are tuned into level-sets of percepts"; "percept
  activations have a kind there but no magnitude".
- A *look* is one act of attention: a region of place handed to
  perception (Alec, 2026-11-16: "attend() looks").

## 1. What the item is, in one page

A sentence of the math corpus is read together with a scene. The numeral
"one" is an opaque word, as now. What is new is that the model also feels
an intensity of one, at a place, and can learn that the word names that
feeling. Nothing computes on the intensity (Alec, 2026-10-07 and
2026-11-13).

The item has three parts.

1. *Continuous perception* (§3). A felt value at a place on three axes,
   read by the one reader that already serves the value beside a sentence,
   XOR's two numbers and an image's pixels.
2. *The corpus* (§5). Generated sentences with the scene each one
   describes, in grades that follow the order of learning Alec gave on
   2026-11-18: words, objects, then the tie between them.
3. *What changes in word and concept processing* (§4), and the tests that
   show it (§7).

## 2. What exists

### 2.1 A felt value beside a sentence

`bin/ScalarPercepts.py` ("A continuous native modality beside text").

- *Given with the sentence.* `InputSpace.prepInput` takes a row
  `dict(text='one', scalar=1.)`. It refuses any other key.
- *Felt.* `NumericPercepts.feeling`: a number $n \ge 0$ is felt as
  $f(n) = g/(1+g)$, $g = \ln(1+n)$. Zero is felt as nothing. The numbers
  of the corpus:

  | $n$ | 0 | 1 | 2 | 3 | 5 | 10 | 19 | 20 |
  |---|---|---|---|---|---|---|---|---|
  | $f(n)$ | 0 | .409 | .524 | .581 | .642 | .706 | .7497 | .7528 |

  The scale is logarithmic: one and two are .114 apart, nineteen and
  twenty .003.
- *Perceived as a level.* `BasicPercepts.perceive_value` admits each exact
  value met as a percept that is its own part and its own whole, in the
  table `value_codes` (as many seats as `basicPerceptCapacity`).
- *At one place.* `WhereRegistry` gives the range `scalar` one address.
  There is no place axis.
- *Joined to the sentence.* `ScalarPercepts.join`, called by
  `SentenceField.open_field` at the sentence's first look, adds the felt
  value to the sentence's field as one more column beside the words'.
- *Tied to words.* At a kept close `ScalarPercepts.commit` pairs every
  word of the sentence with the percept present.
  `PerceptCollections.observe` keeps one candidate in MTM for each word,
  and on the third pairing (`concept_mint_count`) makes a concept of order
  zero that collects the percepts, and writes it as the word's object
  (`InterpretLayer.define`).
- *A concept over levels is an interval.*
  `BasicPercepts.collection_code` takes the least part and the greatest
  whole of the levels collected. A value between them belongs
  (`read_collection`); `Ops.partOf` on two levels is less-than.
- *Asked and answered.* "what is this ?" with a felt value is answered
  with its word (`test/scalar_question_curriculum.py`,
  `data/scalar_lessons.json`); a felt value with no words closes and is
  reconstructed
  (`test_item18_scalar.py::test_scalar_only_presentation_closes_and_reconstructs_its_modalities`).

### 2.2 Numbers at places

`bin/NumericPercepts.py`, when the configuration's `dataType` is `numeric`.

- A row of numbers of any shape is flattened. Each number is a level
  percept at its place, in the range `numeric` of `WhereRegistry`
  (as many addresses as the product of the input shape). A place on
  several axes is one row-major address, decoded by the declared shape
  (`WhereEncoding.decode_index`).
- One concept of order zero for each place collects the levels met there.
- Here the place is part of what the percept is:
  `BasicPercepts.reserve(parts, wholes, place=...)` makes one basic
  percept for each pair of level and place. The same level at two places
  is two percepts, which is what keeps XOR's two inputs apart.
- It is its own stem. A configuration reads numbers or text, not both.

### 2.3 Modalities and places, as decided

Alec, 2026-11-13: each modality has its own perceptual space with its own
number of place axes; modalities meet at concepts of order zero; the
modality and the order are read from the range the place lies in
(`doc/Spaces.md`, "Modalities: One Perceptual Space Each"). The scalar
beside the words has no axis, text one, an image two.

### 2.4 The math corpus

`bin/math_chain_corpus.py`; frozen under `data/math/` with a manifest.

- Numerals zero to twenty as opaque words. Nine grades: counting
  ("zero plus one is one."), identity ("x is two. what is x ?"), alias,
  plus zero, plus one, plus k, distractors, depth two, depth three.
- Addition is reached only by chaining the successor in thought. Four
  pairs are held out, four lie beyond the trained range.
- No sentence has a scene. Nothing numeric reaches the learner.
- Loaded by `Data.loadMathChain` (`bin/data.py`); its frozen grades are
  read by `bin/training_corpus.py` and `bin/stage_models.py`.

### 2.5 What item 1.7 changes under this

From `2026-11-14-sigma-pi-layers.md` §19.21 and §19.25; not yet built.

- *Rows on recurrence* (rule 20). A basic percept becomes a row only when
  MTM nominates it after `concept_mint_count` occurrences. Nothing is a
  row on first sight.
- *One concept for each row* (rule 21), and what
  `PerceptCollections.collect` writes today becomes a column of the Sigma
  of order zero (rule 8b).
- *Words, objects, then their meta concepts* (rule 31; Alec, 2026-11-18).
  A word generalization and an object generalization are each learned
  from their own occurrences. The meta concept that joins them is learned
  afterwards from their consistent association. An occurrence keeps its
  place and time; a generalization holds neither.
- *The vertical spread and the way back to percepts* (rules 23 to 25 and
  30). A word that is found raises its object, and the way back gives the
  object's percepts.
- *Certificate 25, `TestCrossModalTie`*, poses rule 31 on the scalar
  beside the words: the word alone, the object alone, the two together,
  each raising the other, no tie without consistency, two words tied to
  one object.

*Claude's reading of the division.* Item 1.7 builds the tie between a
word and a felt value with no place. Item 1.6 gives the feeling places,
puts several felt things in one scene, and generates the corpus. Where
this document says "the tie", it means the one item 1.7 builds.

## 3. Continuous perception

### 3.1 A scene beside a sentence

A row becomes `dict(text=..., scene=[(x, y, z, n), ...])`: each entry is a
number `n` at the place `(x, y, z)` (*default form*). `scalar=v` remains
and means a scene of one value where the modality has one place. A row
with a scene and no text is allowed, as a felt value alone is today.

*Example.* "one is above two." with the scene `[(1, 2, 3, 1.), (1, 2, 2,
2.)]`: an intensity of one at height 3 and an intensity of two at height
2, in the same column.

### 3.2 Places on three axes

- *A range of its own.* `WhereRegistry` gets one input range for the felt
  modality, with as many addresses as the product of its declared shape.
  *Default:* the key `feltShape`, 4 by 4 by 4, 64 places. With shape 1 it
  is today's `scalar` range, which it replaces.
- *One address for each place,* row-major, decoded by the shape
  (`WhereEncoding.decode_index`), as for an image. The place band of a
  vector is not widened.
- *Why 4 on each axis* (*default*). Two would let "above" be learned as
  "at the top"; a relation of two places has to hold at several heights.
  Sixty-four places are still few enough for a bounded test.

### 3.3 One reader

Alec, 2026-11-12: "We should share as much code between the two
modalities as possible."

- The felt scene is read by the functions of `NumericPercepts.read` (felt,
  level, place), and joined to the sentence's field by what
  `ScalarPercepts.join` does today for its one column: one column for each
  felt thing in the scene, each with its place and the sentence's time.
- The value beside the words is then the case of one place, and
  `ScalarPercepts` keeps no reading of its own. XOR and the image file
  read through the same functions with no sentence beside them.
- Nothing new is given a store or a recognition loop of its own (Alec,
  2026-11-12, of the scalar's former bounds: "they should still be bounded
  by perceptual parts and wholes").

### 3.4 What the percept is: the level, with its place on the occurrence

*Claude's reading; the point on which this item most depends on how item
1.7 lands.*

For the felt world, the numeral's object has to be the same thing at
every place. So the basic percept is the level, and the place belongs to
each occurrence of it, as the letter "a" is one percept wherever it stands
in a sentence. This is rule 31's "an occurrence keeps its place and time;
a generalization does not hold them".

It differs from §2.2, where place is part of the percept. XOR needs that
today, because its two inputs differ in nothing but place. Under item 1.7
a percept's vector carries its place (rule 13) and the Sigma of order zero
is dense over that vector (rule 8), so XOR's two inputs may be told apart
by the vector without two rows. *Not checked: item 1.7 is not built. It is
re-read at item 1.7's return, and is question 5 of §9 if it does not fall
out.*

### 3.5 Levels, when the intensities are not exact

Today every exact value met is its own level. That is sound while "one" is
always felt as exactly 1.0. If a felt one varies, say between .9 and 1.1,
two things fail:

- every reading is a new level, so `value_codes` fills
  (`basicPerceptCapacity` seats);
- under rule 20 nothing recurs, so no level is ever nominated as a row.

What already works for a varying value: a word's concept is an interval
over its levels (§2.1), and a new value inside the interval belongs
without a row of its own (`BasicPercepts.read_points`,
`read_collection`). With the felt ones between .9 and 1.1 and the felt
twos between 1.9 and 2.1, the two intervals are .391 to .426 and .516 to
.531, and do not meet. At nineteen and twenty they are .7494 to .7500 and
.7525 to .7530, still apart, by .0025.

What is not designed: what counts as the same level when values vary, so
that something recurs. Two ways, neither chosen:

- *A declared grid of levels,* as grey levels are for an image. A reading
  is the level whose bounds it lies between. On the felt scale a uniform
  grid needs at least 332 levels to keep nineteen and twenty apart; a grid
  uniform in the number itself needs far fewer but must declare a greatest
  number.
- *Bounds from what is met.* A reading that lies inside an existing
  level's part and whole is that level again; one that lies outside every
  level is a new one. It needs a rule for when a wide level divides.

*Default, question 2 of §9:* the corpus's intensities are exact in every
grade that teaches, and one later grade varies them and is a measurement,
with the levels left as they are (so it measures the intervals and
nothing else). The grid or the bounds are decided after that measurement.

### 3.6 Zero

Zero is felt as nothing, so the word "zero" has no feeling to be tied to.
Under item 1.7's rule 29 an absence that was expected is what a Not entry
at the basic level records (Alec, 2026-11-17: "Omission teaches not").
*Default:* "zero" appears in scenes as an empty place and is reported,
not required; whether the word comes to be tied to an absence is a
measurement of this item.

### 3.7 Attention over the scene

Attention reads one axis of place today. `doc/FutureWork.md` lists a
second axis as remaining: a look is a box, its parts and wholes are boxes.

- *Not needed for a numeral's tie.* The whole scene is read at the
  sentence's first look, with the whole sentence. With two felt things
  present, "one" is paired with both; over many scenes it meets the
  intensity of one every time and each other intensity now and then, and
  the tie is learned from that consistency (certificate 25, point 5).
- *Needed if a word of place is to act between two looks* (§6).
- *Default:* no look narrower than the whole scene in the first round.

### 3.8 Sizes

Declared in the configuration and never resized in a fixture (Alec,
2026-10-24). *Default* for the bounded tests: 64 places; levels for zero
to twenty; at most three felt things in a scene. No operation runs over
all pairs of places.

## 4. How a second modality changes word and concept processing

This is the third part Alec asked for. The left column is the source as
item 1.8 leaves it, with item 1.7's rules where they apply. The right
column is what the felt world makes possible. Each line names where it
was decided or designed; nothing in the right column is a new mechanism
except where §6 and §9 say so.

| | With text alone | With the felt world |
|---|---|---|
| *Where a word's object comes from* | It is made for the word, out of what the word recurs with (`PerceptCollections.observe`, found by `defining_word`), or it is the blank object `interpret` makes (rule 20). Alec, 2026-11-18: "a bootstrap because we cannot know the object expect through word for a unimodal model". | It is learned on its own first, from the feeling's own recurrence, with no word present (rule 31). An existing object is looked for before one is made (Alec, 2026-11-18: "We should probably check to see if an object-concept already exists"). |
| *The tie* | Written at once at trust one, flagged (Alec, 2026-11-13: "lets just use one, and flag it"). | Learned from consistent association. Its certainty can be the agreement of the two modalities (`doc/FutureWork.md`, "The certainty of a word's mapping to its object, from a second modality"; Alec, 2026-11-11, naming this corpus as the place). |
| *Which word of the sentence* | Every word is paired with what is present; "is" and "this" are parted from "water" afterwards by falling agreement. | The same, and now in both directions: several felt things are present with several words, and only the consistent pairs are kept. |
| *Two words for one thing* | Each gets an object of its own, with the same content. | Both are tied to the one object, and through it to each other (Alec's "Alec" and "he"). |
| *Which individual a word or pronoun means* | An open reference whose choice has nothing to learn from (Alec, 2026-11-15: "we have no real basis for assigning an identity yet"). | The value seen with each sentence tells which individual is meant (`doc/FutureWork.md`, "Identity taught by a second modality"). |
| *Why a sentence is believed* | Authority and reasoning. | Witnessing as well: the sentence can be held against the scene, and a sentence that is false of its scene has evidence against it. |
| *The way back from a word* | To its spelling. | To the feeling: word, meta concept, object, and the object's percepts at a place (rule 30; Alec, 2026-11-17, "a kind of imagination rooted in the word"). |
| *The dictionary regress* | Words are defined by words. | Alec, 2026-11-18: "We will soon be breaking the dictionary regress by experience: namely, continuous values as scalars or perhaps images." |

**In the order of one sentence** (functions as they are named today).

1. `InputSpace.prepInput` stages the text and the scene.
2. At the first look, `SentenceField.open_field` reads the words and, by
   §3.3, the scene, into one field: the words' columns and the felt
   things' columns, each with its place. Their places lie in different
   ranges, so each column's modality is read from its place.
3. `intuit` runs the vertical spread (rule 25). A word that is found
   raises its meta concept and, through it, its object. An object that is
   felt raises its word.
4. The grammar composes the sentence. Operators act on concepts, never on
   a felt value.
5. At a kept close, what was present together is noted. Today that is
   `ScalarPercepts.commit`, which collects under each word; under rule 31
   it is the co-presence of a word's concept and an object's concept,
   from which the meta concept is learned. *Which form it has is item
   1.7's; this item only gives it more than one object at a time.*
6. On the way back, a word whose scene was withheld gives its object's
   level at a place.

**What does not change.** Numerals stay opaque words. `plus` stays a verb
and addition a chain of successors in thought. No operator reads a felt
value: of the values, only part, least and greatest exist (Alec,
2026-11-11: "Maybe part, max, and min, so that we don't have to invent
overflow rules"), and "arithmetic do not operate on the codes"
(2026-11-13). `test_arithmetic_isolation.py` stands.

**A hypothesis, not a claim.** Alec, 2026-11-18: "feelings may correspond
to a felt continuous sense that corresponds to a continuous variable in
nature, rather than a sign (cf. Pierce) or symbolic system where that
does not hold." Nothing in this item or its tests depends on it.

## 5. The corpus

### 5.1 The generator and its files

*Defaults throughout.*

- `bin/felt_world_corpus.py`, beside `bin/math_chain_corpus.py`, whose
  numerals, held-out pairs and document record (`ChainDocument`) it uses.
  A document's sentence gains a scene.
- Frozen under `data/felt/` as `grade-NN.jsonl` with a `manifest.json`
  (seed, command, hashes, counts by split), as `data/math/` is. A test
  never generates.
- `Data` loads it as it loads the math chain, and hands each row to
  `prepInput` with its scene.
- A scene is generated from the sentence's meaning, never from its
  words: the generator holds the arrangement and writes both.
- Intensity of the numeral $n$ is $n$. Places are drawn afresh for each
  document, so no numeral has a place of its own.

### 5.2 The grades

The order is Alec's of 2026-11-18: "we learn word generalizations, and we
learn object generalizations, and THEN we learn word-object
meta-concepts". Each grade uses only what the grades before it taught.

| Grade | What is presented | Example | What it teaches |
|---|---|---|---|
| F0, feelings | A scene with no words: one intensity at one place | scene: 3 at (0, 2, 1) | The object alone, at many places |
| F1, words | The numerals with no scene: the math corpus's counting grade as it is | "two plus one is three." | The word alone |
| F2, naming | One word with one intensity | "three." with 3 at (2, 0, 3) | The tie |
| F3, among others | Two or three intensities, and a sentence that names some of them | "this is three." with 3 and 7 present | The tie by consistency, with other things present |
| F4, asked | A question of what is felt | "what is this ?" with 5 present; answer "five" | The object raising its word |
| F5, arrangements | Two intensities at two places, and a sentence of their relation | "one is above two." | A word of place (§6) |
| F6, asked of an arrangement | A question of a relation | "what is above two ?"; answer "one" | The relation, read from the scene |
| F7, kinds | Alec's "Maybe girls are even and boys are odd" | "two is a girl. she is above one." | A kind over numerals; a pronoun (§9, question 4) |
| F8, change | One individual whose intensity changes from sentence to sentence | Alec, 2026-11-15: "one summed with two the other day, and then he summed with three" | Identity across sentences, told by what is seen |
| F9, varied | Grade F2 and F4 with intensities that vary about their number | "three." with 3.07 present | A measurement (§3.5) |

Notes on single grades.

- *F0 and F1* are certificate 25's points 2 and 1 at corpus size.
- *F3.* Every word is still paired with everything present. The grade is
  what shows that the consistent pair is kept and the others are not.
- *F5* has six relations, two for each axis. *Default words:* "above" and
  "below"; "left of" and "right of"; "in front of" and "behind". Alec's
  "on top of" is taught as a second wording of "above" for two places one
  step apart. The past tense of his example ("was") is left out of this
  item (*default*): the scene is present with its sentence.
- *F8* is to be specified with Alec after the first rounds. One thing is
  worth saying now: if an individual keeps its place while its intensity
  changes, the place is what shows that it is the same individual.
- *The nine grades that exist* (§2.4) are unchanged and keep no scenes
  (*default*). Whether a chain of additions is also shown, each name's
  value felt at each sentence, is part of F8.

### 5.3 What is held out

- *Places.* For each numeral, some places at which it is never felt in
  training. Recognizing it there shows that the object holds no place.
- *Pairs.* The math corpus's four held-out pairs are never arranged
  together in training. A relation recognized between them shows that
  the word of place holds no particular numerals.
- *Truth.* Training sentences are true of their scenes. Test rows include
  sentences false of their scenes, to measure the witness of §4.
- *Never an answer in the scene.* At a held-out question the scene shows
  what the question's premises show and nothing that only the answer
  would.

### 5.4 Never direct

The scene carries intensities and places, never a numeral's word, a
digit or a readout. The manifest says so, as `data/math/manifest.json`
does (`never_direct`). The successor is still stated in words; what the
felt world adds is that each numeral's object can be ordered against the
others by `Ops.partOf`, which is perception and not arithmetic.

## 6. A word of place

*Not designed. This section says what was found and where the design
already points; the choice is question 3 of §9.*

**A finding.** The same scene is described by "one is above two" and by
"two is below one". So nothing that is felt is present with "above" and
absent with "below": being present together cannot teach either word.
What tells them apart is which thing the sentence names first, and that
is order, which a field does not hold. As decided in direction for one
attention (Alec, 2026-09-27): the operations of a field are the
order-independent ones, and everything order-dependent, the preposition
among them, is the grammar's and acts between looks.

**Where the design already has it.** `doc/Language.md`, "Position":
"Tense, aspect and the preposition are one operation: unfold the head
into its cases, select among them by the object, fold what is left ... a
noun phrase's cases are things laid out in space ... The object decides
the test: a place compares location ("sat on the mat")."

- The operators exist: `preposition` (`PrepositionLayer` of
  `bin/Language.py`, a phrase headed by its marker whose content is its
  object; `data/complete.grammar`: "refines the NP or VP via a .where
  relation") and `compound` (`bin/CaseSelection.py`), which selects among
  the head's cases by a test the object supplies.
- In that design "above two" is a selection: of the things laid out in
  the scene, those that are above two. "one is above two" then says that
  one is among them, which the scene confirms or denies.

**What is missing for it.**

1. The scene's felt things as the cases of a noun phrase. Today the
   cases come from stored definitions (`CaseSelection`: "The owner stages
   existing sigma definitions once per reading").
2. A test that compares places. `CaseSelection` tests content; no
   reading of `.where` was found in it (*by reading; to be confirmed by
   Codex's inventory*).
3. What the word "above" contributes: which comparison of two places it
   is. This is what would be learned, and its form is not designed. A
   comparison along one axis is an order, and the only order the design
   has on felt things is part-of on values.
4. If the two things are taken one at a time, a look that is a box on
   three axes (§3.7).

**What must not be done.** A reader that decodes the address and compares
heights in code would put the word's meaning in the source. A concept's
position must not be written from a percept's place (Alec, 2026-11-15 and
2026-11-16).

## 7. Tests

Certificates of this item, numbered F1 to F12 (*defaults*). Each names
what it prints, as the testpoints do. Bounded tests use small declared
sizes; the corpus-size measurements are run once at submission.

| | Certificate | Passes when |
|---|---|---|
| F1 | One reader | The value beside the words, XOR, the image file and a felt scene are read by the same functions (a trace names them); the scalar tests pass with `scalar=` given as a scene of one place |
| F2 | Places | The felt range has the product of `feltShape` addresses, disjoint from text; each felt thing's place decodes to its three coordinates; no vector is widened |
| F3 | The object alone | After grade F0, each intensity met often enough has its row and concept; no word is tied to it; the row holds no place |
| F4 | The tie, with others present | After F2 and F3, each numeral's object is its own intensity's concept; a word paired as often with varying intensities is tied to none |
| F5 | Each raises the other | The word alone leaves its object expected, and the way back gives the intensity (within the level's bounds) at a place; the intensity alone answers "what is this ?" with its word |
| F6 | Place is not held | A numeral is recognized at a place where it was never felt in training |
| F7 | Two words, one object | A second word for an intensity is tied to the same object and no second object is made |
| F8 | The witness | A sentence false of its scene leaves evidence against it; *reported until its mechanism is decided* |
| F9 | A word of place | By the answer to question 3 |
| F10 | Never direct | `test_arithmetic_isolation.py` holds with scenes; no operator of the grammar reads a felt value; no scene at a held-out question carries its answer |
| F11 | Sizes | No table is resized; a full table raises its capacity error |
| F12 | What stands | The math-chain tests with no scenes, both stopping gates and the XOR proofs are unchanged |

**Measurements, reported and not required:** "zero" (§3.6); grade F9
(§3.5); the tie's certainty against the agreement of the two modalities;
the largest expected value of any concept never presented with a word
(certificate 25, point 7).

**Tiers** (1.8 §72). Each build round runs its own certificates, the
tests of the files it changes and both stopping gates. The item has one
complete default sweep, at the round that may be pushed. `RUN_SLOW` is
run once, overnight, at submission.

## 8. Not in this item

*Defaults; question 1 of §9.*

From `doc/FutureWork.md`, "Numeric input as a modality": learning on
images, grey levels, one model class, and the whole-slab body of the
other numeric configurations. Attention's second axis, unless §6 needs
it. Sum and product on felt values (Alec, 2026-11-11 and 2026-11-13).
Two modalities other than text and feeling.

## 9. Questions for Alec

1. *What the item takes of the number modality.* *Default:* only what the
   felt world needs: felt values at places on three axes beside a
   sentence, and the one reader (§3). The rest of §8 stays in FutureWork.
2. *Exact or varying intensities* (§3.5). *Default:* exact in the grades
   that teach, varied in one later grade as a measurement.
3. *How a word of place gets its meaning* (§6). *Default:* numerals
   first; then the position selection that `doc/Language.md` designs,
   with the scene's things as the cases, specified in a second round when
   the first has returned.
4. *What carries even and odd* (grade F7). An intensity is not felt as
   even or odd. *Default:* it is said, not felt: "two is a girl." makes a
   kind over numerals, and a pronoun is then resolved against the scene.
   The other way: each felt thing has a second felt quality that carries
   it.
5. *The percept is the level and the place is on the occurrence* (§3.4).
   *Default,* to re-read at item 1.7's return.
6. *The grid:* 4 by 4 by 4 (§3.2).
7. *"zero"* as an absence, measured (§3.6).
8. *The words of grade F5,* and the past tense left out (§5.2).
9. *Whether the nine existing grades get scenes* (§5.2, F8).
10. *Where this item's Codex texts and returns live.* *Default:* this
    file.

## 10. The order of building

*After the answers; a sketch, not a text.*

1. *Round one: the modality and the tie.* §3.1 to §3.3, grades F0 to F4,
   certificates F1 to F7 and F10 to F12.
2. *Round two: arrangements.* §6 as decided, grades F5 and F6,
   certificate F9, and F8 if its mechanism is decided.
3. *Round three: kinds, pronouns and change.* Grades F7 and F8, and the
   measurement of F9.
4. *Submission.* The item's one default sweep, then the slow tier
   overnight.
