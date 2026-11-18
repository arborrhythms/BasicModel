# Item 1.6: the number modality, the math corpus, and the felt world that combines them

*Claude, 2026-11-18. A design for Alec. No text for Codex: questions are
open (§11.6), and the item is built after item 1.7 lands. Alec answered
the first four questions the same day (§11) and the questions those
opened (§12). §12 stands wherever it differs from §11, and §11 wherever
it differs from §3 to §10.*

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

*Decided (Alec, 2026-11-18): "Varied from the start".* So what a level is
has to be decided before a round is built: §11.2, with Claude's
recommendation of a declared grid.

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
| F5, arrangements | Alec, 2026-11-18: "A small image in which the numbers are vertically arranged as described." | "one is above two." with an image in which one is above two | A word of place (§6, §11.3) |
| F6, asked of an arrangement | A question of a relation | "what is above two ?"; answer "one" | The relation, read from the scene |
| F7, types and operations | Alec, 2026-11-18: "words like even and odd, and other simple math concepts like basic operations". The gendered words are dropped | "two is even." | A type over numerals (§11.4) |
| F8, change | One individual whose intensity changes from sentence to sentence | Alec, 2026-11-15: "one summed with two the other day, and then he summed with three" | Identity across sentences, told by what is seen |

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

*Alec, 2026-11-18: "Add 2d intensity images also."* So images whose
pixels are intensities are in the item (§11.1). Still out, from
`doc/FutureWork.md`, "Numeric input as a modality": one model class, and
the whole-slab body of the other numeric configurations. Attention's second axis, unless §6 needs
it. Sum and product on felt values (Alec, 2026-11-11 and 2026-11-13).
Two modalities other than text and feeling.

## 9. Questions for Alec

*Questions 1 to 4 are answered: §11. Questions 5 to 10 stand as defaults,
with §11.6's.*

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

## 11. Alec's answers (2026-11-18), and what they change

Asked the first four questions of §9, each with Claude's default. His
answers are given whole.

### 11.1 What the item takes of the number modality

> Add 2d intensity images also.

**What it changes.** The item takes the felt world and images whose
pixels are intensities, on two axes of place. Of the list of §8 that
brings in two entries of `doc/FutureWork.md`: *grey levels* (an intensity
image is a grey image) and *learning on images*. One model class and the
whole-slab body were not named and stay out (*Claude's reading*).

**What exists for it.** `data/MM_image.xml` reads 28 by 28 images of two
levels through `NumericPercepts`, with no attention.
`bin/image_corpus.py` freezes them under `data/images/`, packed one bit
to a pixel, so its format cannot hold an intensity.

### 11.2 Exact or varying intensities

> Varied from the start

**What it changes.** §3.5's default is withdrawn, and grade F9 with it:
every grade's intensities vary about their number. Two things have to be
decided before a round is built, since under item 1.7's rule 20 a percept
becomes a row only when it recurs, and an exact value that varies never
recurs.

1. *What a level is.* The same question as where a grey image's levels
   come from. *Claude's recommendation: a declared grid of levels,* a
   key of the modality, as the 256 byte values are the declared alphabet
   of text.
   - A reading is bounded by the grid level at or below it, its part,
     and the one at or above it, its whole. That is Alec's "categorized
     against perceptual parts and wholes, whose codes are the values"
     (2026-11-12), with the bounds given beforehand in place of each
     exact value met.
   - Readings between the same two levels are the same basic percept, so
     it recurs and MTM can nominate it.
   - A word's concept stays the interval over the levels it collected
     (§2.1), which is what generalizes over the variation.
   - The other way, bounds grown from what is met, needs a rule for when
     a wide level divides, which would be a new mechanism.
2. *The scale and the amount.* On today's felt scale (§2.1) nineteen and
   twenty are .003 apart, so a uniform grid would need at least 332
   levels. *Default:* the corpus's intensity for the numeral $n$ is
   $n/20$, taken as it is with no logarithm, on a grid of 256 levels;
   neighbouring numerals are then about 13 levels apart. A reading
   varies by up to a quarter of the gap between neighbours, about 3
   levels, so two numerals' readings never meet. *Whether `feeling`'s
   logarithm is kept for numbers given beside a sentence, and whether
   neighbouring numerals may overlap, are his.*

### 11.3 A word of place

Asked how a word such as "above" should get its meaning:

> A small image in which the numbers are vertically arranged as
> described.

**What it says.** What is felt with "one is above two" is a small image
in which the intensity of one stands above the intensity of two. The
scene of §3.1 is therefore an image, on two axes.

**What it leaves open** (*Claude's reading*). It gives what is perceived.
It does not say what in the model reads the arrangement, and §6's finding
stands: the same image goes with "one is above two" and "two is below
one". Two things in the design bear on it, neither built for an image:

- *An arrangement as an ordered run.* In text a basic percept's parts
  are an ordered run: `BasicPercepts.carrying` keeps (a, b) apart from
  (b, a). One intensity above another is an ordered run along the
  image's vertical axis. If the image's parts are read that way, the
  arrangement is itself a percept with an object of its own, and "one
  above two" and "two above one" are two percepts.
- *The sentence against the arrangement.* The two sentences that go
  with one image differ in their word of place and in the order of their
  numerals, and the position selection of `doc/Language.md` (§6) is what
  reads that.

*Default:* round one builds the image scenes and the numerals' ties;
the reading of an arrangement is specified with Alec after it returns.

**The third axis.** His first words were "feeling-intensities within a
3-D space" and §3.2 gave the feeling three axes. An image has two. *Open,
question 1 of §11.6: whether the scene of this item is a small image and
nothing else.* *Default:* yes; §3.2's `feltShape` is then two axes,
8 by 8, still 64 places, and §3.3's one reader is the image's.

### 11.4 Even and odd

> I was just trying to give objects for a gendered language. Let's skip
> that idea for learning that, and just have the corpus learn words like
> even and odd, and other simple math concepts like basic operations. If
> it learns types well, it should be able to generalize.

**What it changes.** The gendered words and the pronouns resolved by
them are dropped. Grade F7 becomes the words "even" and "odd", other
simple concepts of mathematics, and the basic operations. This is the
development of the math corpus that item 1.6's title asks for.

**A list to trim** (*Claude's default; each line is his to strike*).

| Concept | Example | Felt or said |
|---|---|---|
| less, greater, equal | "one is less than two." | Felt: one level is part of the other (`Ops.partOf`) |
| next, before | "two is after one." | Said, as the counting grade says it; felt as the next greater intensity |
| even, odd | "two is even." | Said. An intensity is not felt as even or odd |
| plus | "one plus two is three." | Said, and reached by the chain of successors, as now |
| minus | "three minus one is two." | Said; the inverse of the successor |
| times, divided by | "two times three is six." | Said; *default:* not in this item |

**"If it learns types well, it should be able to generalize."**
*Claude's reading:* the numerals are one type, "even" and "odd" are
types within it, and what is taught of some members is asked of the
others. So the held-out rows of F7 are numerals whose type was stated
and of which an operation or a comparison was never shown, and numerals
of which it was shown and whose type is asked.

**No operator on values, still.** "less than" read from two felt levels
is part-of, which exists. Nothing adds or multiplies a felt value (Alec,
2026-11-11 and 2026-11-13).

### 11.5 What stands and what goes in §3 to §10

- §3.1, §3.3 to §3.8 stand, with the scene an image (§11.3).
- §3.5's default goes; §11.2 replaces it.
- §4 stands whole.
- §5.2: F5 and F7 as amended in the table; F9 goes; F8 (one individual
  whose intensity changes) is not built unless Alec asks for it, since
  the pronoun it was written with is dropped (*Claude's reading*).
- §6 stands as the record of what a word of place needs.
- §7: F7 (two words, one object) is kept on two wordings of one numeral,
  not on a pronoun. F9 is by §11.3.
- §10: round one is the image scenes with varied intensities on the
  declared grid, grades F0 to F4; round two, arrangements; round three,
  grade F7.

### 11.6 Open after these answers

1. *Is the scene of this item a small image and nothing else,* with no
   third axis (§11.3)? *Default:* yes, 8 by 8.
2. *A declared grid of levels* (§11.2, point 1). *Recommended.*
3. *The scale and the amount of variation* (§11.2, point 2).
4. *The list of concepts for grade F7* (§11.4).
5. *What a numeral looks like in the image:* one pixel of its intensity,
   or a small patch. *Default:* one pixel, and the rest of the image at
   zero, which is no presence.
6. Questions 5 to 10 of §9.

## 12. Alec's second answers (2026-11-18): three axes, a level as a float, and a corpus that teaches language

Asked whether the scene is a small image and nothing else, and given
§11.6's list. His answers are given whole.

### 12.1 Three axes are kept

> Keep 3D, I guess, since it is isomorphic to English language. So we
> might as well render images as 3-D data with only one z plane (at
> z=0), but make the .where able to address the entire space.

**What it changes.** §11.3's default, an image and nothing else, is
withdrawn. §3.2 stands: the felt modality has three axes of place, and
its range of `.where` has an address for every place of the declared
shape. An image is the plane at z = 0 of that space; the other planes are
empty until something is put there.

*Default:* `feltShape` 8 by 8 by 8, 512 addresses, of which an image
uses 64. Only the count of addresses grows; no vector is widened (§3.2).

*Claude's reading of "isomorphic to English language":* English speaks of
places on three axes (above and below, left and right, in front and
behind), so what its words describe should have the three.

### 12.2 A level is a float from 0 to 1

> a level is a 0-1 float always, input preprocessing will map the data
> there.

**What it changes.** No grid of levels is declared; §11.2's
recommendation is not taken. A level is a number from 0 to 1, and
whatever maps the data into that range is preprocessing, outside the
model. `NumericPercepts.feeling` is one such map, for numbers from zero
to infinity. The corpus's own map is a choice of the generator
(*default:* the numeral $n$ at $n/20$, so zero is nothing felt and twenty
is 1; a reading varies by up to a quarter of the gap between
neighbours).

### 12.3 The difference is enough; the bracketing adapts

Of the scale and the amount of variation:

> that is enough of a difference for perceptual bracketing, which is
> adaptive anyway.

*Claude's reading:* "perceptual bracketing" is a felt value bounded by a
part below it and a whole above it (Alec, 2026-11-12: "categorized
against perceptual parts and wholes, whose codes are the values"), and
those bounds move with what is met.

**What the source has of it** (read 2026-11-18; he is describing what
should exist, so this says where it does and where it does not).

- *It adapts at the concept.* A word's concept over felt values is the
  interval from the least part to the greatest whole of what it has
  collected (`PerceptCollections.collect`,
  `BasicPercepts.collection_code`), and it widens with each new member. A
  reading inside the interval belongs with no row of its own
  (`BasicPercepts.read_points`, `read_collection`).
- *It does not adapt at the percept.* `BasicPercepts.perceive_value`
  matches a reading only to a value exactly equal to it, and otherwise
  gives it a new seat in `value_codes`, where it is its own part and its
  own whole: a point, not a bracket.
- *What that costs when readings vary.* Every reading is a new percept.
  The table fills (`basicPerceptCapacity` seats), and under item 1.7's
  rule 20 no percept recurs, so none becomes a row.
- *Where the design has it.* `doc/FutureWork.md`, "Images (future)":
  "exact-key matching gives way to a log-scale tolerance for graded
  percepts", and "The codebook snap keeps its residual, so quantization
  gives categorical perception without erasing continuity". Not built.

**The gap, for Alec** (question 1 of §12.7). For the bracketing to adapt
at the percept, a reading that falls within an existing value percept's
bounds has to be that percept again, and the percept's bounds have to be
able to differ from one another. Nothing says yet when a reading joins a
percept and when it starts one. It is not designed here.

### 12.4 What the corpus is for

> We can build an extensive vocabulary here from a mathematics education
> book. The perception of number is not necessary beyond establishing
> conceptual foundations. Perhaps the biggest value is in teaching
> english language from a synthetic corpus: we can talk about the size
> of 1 (a continguous shape constituted by 1's), how it walked toward 2
> or ran toward 3. then we can ask which was faster, the walk or the
> run? So this basic langauge learning and synthetic corpus will allow
> us proof of concept over basically all language learning. It may also
> serve as a basis or prep step for language learning over a large text
> corpus, but we should only bother with that step after we have that
> proof of concept. So it is the job of the AI to construct simple
> sentence and small image pairs that are valid and teach these things.

And of what a numeral looks like: "lets start with 1 pixel, but the
mereology should generalize fairly well."

**What it changes.**

1. *The purpose.* The corpus is a proof of concept of learning language
   from sentences paired with small images. Feeling a number is needed
   only to found the concepts. Its use as preparation for a large text
   corpus comes after the proof and is not part of this item.
2. *What is in the world.* A thing is a contiguous shape made of one
   level: "the size of 1 (a continguous shape constituted by 1's)". It
   has a size. It moves: "walked toward 2", "ran toward 3". Two events
   can be compared: "which was faster, the walk or the run?"
3. *The vocabulary of mathematics* is wide and is taken from how
   mathematics is taught, in words. Most of it is said, not felt
   (§11.4).
4. *Every pair is valid.* *Claude's reading of "the job of the AI":*
   Claude specifies the world, the vocabulary and the lessons; the
   generator, which Codex builds, holds the state of the world and writes
   the sentence and the image from it, so that no pair can disagree
   (§5.1). If he means that the pairs are written out one by one, that is
   question 3 of §12.7.

**What this reaches that the math corpus could not.** `doc/Training.md`
lists as not reached by the generated math suite: "space, motion,
prepositions, and most verb classes", and "aspect: activities,
progressive, perfect, durations". A world of shapes that move reaches
both.

### 12.5 The world and what can be said of it: a first sketch

*Claude's draft, for Alec to correct. It is a list of what the lessons
would need; nothing here is a mechanism.*

**The world.** A space of 8 by 8 by 8 places, and steps of time. At each
step some things are in it. A thing is a set of touching places at one
level; in the first grades it is one place. A sentence is read with the
image of one step. A change is shown as successive images, some with no
sentence, as a felt value alone is presented today (§2.1).

**What a numeral names** (*default, question 4 of §12.7*). A scene holds
at most one thing of each level, and the numeral is that thing's name:
"1" is the shape made of ones. So "it walked toward 2" speaks of two
individuals, and nothing has to count things. Counting ("two shapes")
would give the numerals a second sense and is left out of the first
grades.

| What is perceived | Words | What it rests on | State |
|---|---|---|---|
| A thing's level | the numerals | the tie of a word to a felt object (item 1.7, certificate 25) | item 1.7 |
| The order of two levels | less, greater, equal | part-of on levels (`Ops.partOf`) | exists |
| A thing's extent | size, big, small, bigger | neighbouring places of one level knit into one percept; `doc/FutureWork.md`: "a connected object is told from a scattered sum" | not designed |
| Where two things stand | above, below, left of, right of, in front of, behind, near, far | the reading of an arrangement (§6, §11.3) | not designed for a scene |
| A thing's place changing | moved, walked, ran, stopped; toward, away from | what holds a succession across sentences (item 1.7's open point 7); a verb phrase's phases, before, change and after (`doc/Language.md`, "Position") | open in item 1.7 |
| How much it changes in a step | fast, slow, faster | two events compared | not designed |
| Nothing felt | even, odd, plus, minus, and the rest of the taught vocabulary | said in sentences (§11.4) | exists, as the math corpus |

**The grades, redrawn** (replacing §5.2's F5 to F8).

| Grade | Teaches | Example |
|---|---|---|
| F0 to F4 | as §5.2: feelings, words, naming, naming among others, asked | "what is this ?" |
| F5, order | less, greater, equal | "one is less than two." |
| F6, arrangements | words of place | "one is above two." |
| F7, shapes | size | "one is bigger than two." with a shape of four ones and a shape of two twos |
| F8, motion | verbs of motion | "one walked toward two." after the images of it |
| F9, events compared | comparison of two events | "which was faster, the walk or the run ?" |
| F10, said mathematics | types and operations | "two is even." |

Note that "bigger" (F7, of extent) and "greater" (F5, of level) are kept
as two words on purpose: a small shape of a high level and a large shape
of a low level tell them apart.

### 12.6 What has to be designed before each round

| Round | Builds | Waits on |
|---|---|---|
| One | Places on three axes with images at z = 0; scenes beside sentences; floats that vary; one pixel; grades F0 to F5 | question 1 of §12.7 (when a reading is the same percept again) |
| Two | Arrangements, F6 | what reads an arrangement (§6, §11.3) |
| Three | Shapes, F7 | what knits touching places into one thing |
| Four | Motion and events compared, F8 and F9 | item 1.7's open point 7; the comparison of two events |
| Any | Said mathematics, F10 | the vocabulary (question 2 of §12.7) |

Each of the four waits is a design conversation with Alec, taken in this
order (*default*). Round one can be specified as soon as question 1 is
answered and item 1.7 has returned.

### 12.7 Open after these answers

1. *When a reading is the same percept again* (§12.3).
2. *Which book or which years of teaching* the vocabulary of mathematics
   is taken from. *Default:* Claude drafts the list from what the first
   years of school teach (counting, comparing, order, shapes and their
   places, adding and taking away), in its own sentences, with no book's
   text copied.
3. *Whether the generator writes the pairs* from Claude's lessons
   (§12.4, point 4). *Default:* yes.
4. *A numeral names the one thing of its level* in a scene (§12.5).
5. Questions 5 to 10 of §9, of which 6 (the grid) is now 8 by 8 by 8.

