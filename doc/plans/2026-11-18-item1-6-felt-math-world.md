# Item 1.6: the number modality, the math corpus, and the felt world that combines them

*Claude, 2026-11-18. A design for Alec. One text for Codex, §14.8: the
corpus's generator and its frozen examples, which touch no source of the
model. The model's part has no text: it is built after item 1.7 lands. Alec answered
the first four questions the same day (§11), the questions those opened
(§12), and then set what the examples are (§13). A later section stands
wherever it differs from an earlier one.*

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

## 13. Alec's third answers (2026-11-18): percepts are discriminants; examples by grammatical rule

### 13.1 When a varying reading is the same percept

Asked "When a felt value varies, should a reading near an existing
percept count as that percept again?":

> percepts are discriminants, so we will discriminate if there are
> different (numeric) parts and wholes between the variations of the
> felt value. Becuase even though the feeling is continuous, we perceive
> it using percepts, which categorize that continuous intensity
> mereologically between percept codes in the percept towers, just as we
> do today.

**The rule.** A reading is categorized between the codes that the part
and whole towers hold. Two readings are told apart when a part's or a
whole's code lies between them, and are the same percept when their
parts and wholes are the same. No tolerance is added and no grid is
declared. This answers question 1 of §12.7.

**In the source.** `BasicPercepts.reserve(parts, wholes)` gives the same
row to the same parts and wholes, and `BasicPercepts.read_points` reads
any reading against the codes that exist. The one thing to know is what
the towers hold: `perceive_value` puts every value met into them, as its
own part and its own whole. So today two readings that differ at all are
two percepts.

**What it means for the corpus** (*default*). The generator writes its
intensities at the resolution image data has, 256 steps from 0 to 1. A
numeral's readings are then a handful of values (seven, for a variation
of three steps either way), each met many times. The towers hold some
sixty codes for nine numerals, every percept recurs, and nothing at the
percept is built for this item. The word's concept is the interval over
its numeral's percepts, as now (§2.1). *Data of much finer resolution
would fill the table of values; that is not this item's data.*

**One thing this leaves to item 1.7** (*Claude's note*). Rule 31 has the
object learned on its own before any word. With seven percepts for one
numeral, something has to gather the seven into one object while no word
is present. Today the gathering is done for the word
(`PerceptCollections.observe`). It is re-read at item 1.7's return.

### 13.2 What the corpus is for, again

> The sketch is good. As long as the english language used to describe
> the scene is correct, it may not matter if it is weird; the structure
> of the english language will port to new language, walking can pick up
> entailments about legs, and the mind will forget the original
> approximation where it conceived of different objects as different
> numbers. So we can learn all english operators within this corpus (or
> at least a large set).

> So we need a set of training examples, maybe 100 per, categorized
> according to which english grammatical rules are used within that
> sentence. we can limit the cast of characters (nouns) to single
> digits, and talk about the evens and the odds. And we should expect
> generalization and inference to work over that space before we begin
> paying for computer time on a remote machine.

**What it decides.**

1. §12.5's sketch stands.
2. *Correct English, however odd.* A sentence may say something strange
   ("three ran toward four"). It may not be wrong English, and it may
   not be false of its images in training.
3. *The aim is the operators of English,* all of them or a large set,
   learned in this corpus.
4. *The examples:* about a hundred for each category, the categories
   being the grammatical rules a sentence uses.
5. *The cast:* the single digits, with "the evens" and "the odds" as
   their two kinds.
6. *The bar before remote machines are paid for:* generalization and
   inference work over this space, locally.

### 13.3 What exists for examples by rule

- *Games by operator.* `bin/game_corpus.py` freezes 31 game variants
  under `data/games/` from recipes (`data/games/recipes/*.json`), 5,042
  rows. A row has sources and a target ("a bear." and "in the beach."
  give "a bear in the beach."). The recipe's note: "The intended operator
  is an evaluation label, never a learner constraint".
- *Ten lessons* in the order the operations are learned
  (`data/lessons/parts-of-speech.json`; `doc/Training.md`, "Concept
  hierarchy learning and curriculum order"): words, interpret, clause,
  ask, verb, modifiers, determiners, adverb, prepositions, relations.
- *Gates are behaviors* (Alec, 2026-10-18): "we do not demand which
  operators must be chosen, only which behaviors must be exhibited."

So a category here is a construction of English. Each row carries the
names of the operators its construction is expected to use, as a label
for reading results. No gate reads the label: a row is passed by its
answer and its reconstruction.

### 13.4 The world's cast and words

*Defaults; question 1 and 2 of §13.8.*

- *The cast:* one to nine, each the one thing of its level in a scene
  (§12.5). Zero is nothing felt (§3.6) and is no character. The evens
  are two, four, six and eight; the odds are one, three, five, seven and
  nine.
- *Kinds give the determiners their work.* With one thing of each level,
  "a three" and "the three" are the same thing. "an even", "the evens",
  "every odd", "no even" differ, and nothing has to be counted.
- *What is seen of a thing:* its level, its place, its size once shapes
  are built, and how its place changes from image to image.
- *Pronouns.* The gendered ones are dropped (§11.4). "it" and "they" are
  kept (*default*), since a later mention of something named is one of
  the operators.

### 13.5 The categories

*Claude's draft. About a hundred examples each, so some 2,600 in all,
beside the games' 5,042 rows and the math grades' 660. The last column
names what the category waits on (§12.6); "nothing" means it can be
generated and taught as soon as round one's scenes exist.*

| # | Construction | Example, with what its images show | Asked | Operators, as a label | Waits on |
|---|---|---|---|---|---|
| 1 | A word alone; a feeling alone | "three." ; an image of three with no words | "what is this ?" | `interpret`, `sign` | nothing |
| 2 | Naming | "this is three." | "what is this ?" | `interpret`, `part` | nothing |
| 3 | Presence and absence | "three is here." "four is not here." | "is four here ?" | `change`, `not` | nothing |
| 4 | Kinds | "two is even." "two is an even." "the odds are one, three, five, seven and nine." | "is three even ?" "what is two ?" | `part`, `whole`, `lower` | nothing |
| 5 | And, or | "two and four are here." "this is two or three." | "is two here ?" | `union`, `intersection` | nothing |
| 6 | Order of levels | "two is less than five." "five is greater than two." | "which is greater, two or five ?" | `part`, `modifier` | nothing |
| 7 | A clause with one thing | "three moved." "three stopped." (images before and after) | "what moved ?" | `change` | succession |
| 8 | A clause with two things | "three touched four." "three followed four." | "what touched four ?" "what did three touch ?" | `verb`, `change` | arrangement; succession for "followed" |
| 9 | The same said the other way | "four was touched by three." | both roles | `swap`, `surface` | as 8 |
| 10 | Adjective | "the big three moved." "three is big." | "what is big ?" | `intersection`, `modifier` | shapes |
| 11 | Comparative of size | "three is bigger than four." | "which is bigger ?" | `modifier`, `part` | shapes |
| 12 | Determiners over kinds | "an even moved." "the evens moved." "every odd stopped." "no even moved." "some odds moved." | "did every even move ?" | `lower`, `modifier` | succession |
| 13 | Adverb | "three moved slowly." "three moved up." | "which moved slowly ?" | `modifier` | succession |
| 14 | Place of two things | "three is above four." "three is near four." "three is on top of four." | "what is above four ?" "where is three ?" | `preposition`, `compound` | arrangement |
| 15 | Place among three | "three is between two and four." | "what is between two and four ?" | `preposition`, `union` | arrangement |
| 16 | Motion with a direction | "three walked toward four." "three ran away from four." | "what did three walk toward ?" | `verb`, `preposition`, `change` | arrangement; succession |
| 17 | Tense | "three was above four." "three is above four." "three will move." | "where was three ?" | `compound`, `marker` | succession |
| 18 | Aspect | "three is moving." "three has moved." | "has three moved ?" | `compound`, `marker` | succession |
| 19 | Order of events | "three moved, then four moved." "before four moved, three stopped." | "which moved first ?" | `compound`, `union` | succession |
| 20 | Events compared | "three walked toward two. it ran toward four." | "which was faster, the walk or the run ?" | `modifier`, `bind` | succession; two events compared |
| 21 | A later mention | "three moved. it stopped." "the evens moved. they stopped." | "what stopped ?" | `bind` | succession |
| 22 | A clause inside a phrase | "the even that is above three moved." | "what moved ?" | `bind`, `intersection`, `lower` | arrangement; succession |
| 23 | Two clauses joined | "three moved and four stopped." "three moved but four did not." | "did four move ?" | `union`, `not` | succession |
| 24 | If | "if three is above four, four is below three." | a question that needs the rule | `implies`, `infer` | arrangement |
| 25 | What follows from a kind | "every even is above three." (and, before it, "four is even.") | "is four above three ?" | `implies`, `infer`, `isPart` | arrangement |
| 26 | Said mathematics | "two plus one is three." "three minus one is two." | "what is two plus one ?" | `add`, `subtract`, `part` | nothing: the math corpus as it is |

*Left out of the first set* (*default*): what might or could have
happened ("three might move", "if three had moved"). Its images would
have to show what did not happen. It is the second difference
(`doc/Philosophy.md`) and is specified when the first set is learned.

*Read against the ten lessons:* categories 1 to 3 are lessons 0 and 1;
7 is the clause; the questions of every category are lesson 3; 8 and 9
are the verb; 5, 10 and 11 the modifiers; 12 the determiners; 13 the
adverb; 14 to 20 the prepositions and what `compound` selects; 4, 24 and
25 the relations. The order of teaching is that of the lessons.

### 13.6 A row, and how it is known to be valid

*Defaults.*

- *A row is a document:* the math corpus's `ChainDocument` (sentences,
  question, answer) with, for each sentence, the image it is read with,
  and beside it the category and the operator labels. An image with no
  sentence is a sentence with empty text.
- *The generator holds the world* and writes every sentence and every
  image from it (§5.1, §12.4). A sentence is made from a record of what
  it means, and that record is checked against the world: true in every
  training row, and marked true or false in the test rows that measure
  the witness (§5.3).
- *The English is checked.* Each template is read by a person before it
  is frozen. A test checks what can be checked by rule on every row:
  agreement ("two is", "the evens are"), the article ("an even", "an
  odd", "a three"), the past forms ("ran", "was"), and that no word
  outside the declared vocabulary appears.
- *Frozen* under `data/felt/`, one file for each category, with a
  manifest: seed, command, hashes, the count of each split, and the
  vocabulary.

### 13.7 Generalization and inference, before any remote machine

*Claude's reading of what is measured; the bar is his to set, as the
lessons' criteria are (`doc/Training.md`: "declared by Alec (null until
he sets it)").*

- *Generalization.* Each category holds some of the cast out of it: a
  construction taught on some digits and pairs is asked of others never
  met in it. For example "what is above four ?" where three was never
  above four in training. The answer and the reconstruction are the
  measure.
- *Inference.* Questions whose answer is neither said nor shown and
  follows from what was. Four kinds: the other way round ("three is
  above four", asked "what is below three ?"); through a kind (category
  25); through a chain ("three is above four. four is above five.",
  asked "is three above five ?"); through time ("which moved first ?").
- *Where it runs.* On the small configuration, locally, with targeted
  tests in each round and one sweep at the item's push (1.8 §72). The
  slow tier's one overnight run at the submission of this item is the
  first run of the whole set.

### 13.8 Open after these answers

1. *The cast is one to nine,* with zero as nothing felt (§13.4).
2. *"it" and "they" are kept* (§13.4).
3. *The bar* for generalization and inference (§13.7).
4. *The list of categories* (§13.5): any to strike, add or divide.
5. *What is left out of the first set* (§13.5).
6. Of §12.7: where the vocabulary of said mathematics is taken from, and
   whether the generator writes the pairs. Questions 5 to 10 of §9.

**A part that can start early** (*Claude's note*). The generator, the
frozen rows and the test of their validity touch no source of the model.
Their text for Codex can be written before item 1.7 lands, for the
categories marked "nothing" and, as data only, for all of them.

## 14. The corpus's generator and its frozen examples (Claude, 2026-11-18; amended 2026-11-23; the text for the implementer is §14.8)

Alec, 2026-11-18, asked whether to write the text for the generator and
its frozen examples now: "Yes, write it". §14.1 to §14.7 are what the
text builds; every choice in them is *Claude's default* unless a section
above gives it as Alec's.

*Amended 2026-11-23 (Claude), before anything of the corpus was built, by
the sections added to this document after §14 was written (hand-over,
H.4, step 1). The frames as first written are in this file at
`ad8fafea92`.*

| What changed | Where | From |
|---|---|---|
| An image is pixel values at places; a digit's word names a band of values, and a number has no percept | §14.1, §14.4 | "What is felt and what a number is" (Alec, 2026-11-22) |
| No image (`null`) is told from an image with nothing in it (`[]`) | §14.1, §14.3, §14.6 | suggestion 2; Alec, 2026-11-21 |
| One step and eight places are reserved for each value | §14.1, §14.5, §14.6 | suggestion 3 |
| Frames 16 and 20 describe a scene; "walked" and "ran" are gone | §14.2, §14.4 | Alec, 2026-11-20 |
| Frame 20 has both answers; frame 24 has both answers and a case in which nothing follows | §14.4, §14.5 | suggestion 4 |
| Contrast sets on "above" and "below" | §14.3, §14.5, §14.6 | suggestion 1, the first milestone |
| Category 27, counting | §14.3 to §14.6 | Alec, 2026-11-22 |
| The text is for an Opus implementer and returns to Claude | §14.8 | H.1 |

### 14.1 The world

- *Space.* 8 by 8 by 8 places, written (x, y, z): x is the column, 0 at
  the left; y is the row, 0 at the top; z is the depth. An image is the
  plane z = 0 (Alec, §12.1), and in this first set every thing is in it.
- *A value.* What an image holds is pixel values (Alec, 2026-11-22: "we
  are referring to image data"). There are nine values, 1 to 9. The
  value $d$ is a band of intensities: a whole number of steps from
  $25d - 3$ to $25d + 3$; the felt intensity is the steps over 255
  (Alec, §12.2: "a level is a 0-1 float always"). A row stores the
  steps. Every other place of the plane is 0, the background, where
  nothing is felt. A digit's word names a value: "three is above four."
  says that the pixel value 3, at its place, is above the pixel value 4,
  at its place. A number itself is a symbol and has no percept.
- *A thing* is a level set: a set of places, each touching another by a
  side, whose intensities all lie in one value's band. A scene has at
  most one thing of each value. Things do not overlap. A thing is one
  place unless its category is about size.
- *Levels.* The steps are drawn for each place of each image, from the
  six steps of the band that are not the value's reserved step (§14.5).
- *Time.* A document is a list of entries. An entry is a text, which may
  be empty, and one image. Successive entries are successive moments.
  "Now" for a sentence is the image it is read with. Item 1.4 will
  observe the change between successive images (H.3); they are kept as
  designed here.
- *No image, and an image with nothing in it.* An entry's scene is a
  list or `null`. `[]` is an image that was seen and holds nothing: it
  is evidence against anything's being here. `null` is no image: the
  sentence is heard with nothing seen, which is evidence of nothing
  either way. Alec, 2026-11-21: "The missing scene does not affect
  reconstruction cost." Where §14.4 and §14.5 say "no image", the scene
  is `null`.

### 14.2 What can be true of the world

The generator evaluates these on the images. A recipe ties a word to
one of them; the generator's code holds no word (as
`bin/game_corpus.py`: "it holds no word of any language").

| Predicate | True when |
|---|---|
| `present(a)` | a has a place in the image |
| `member(a, k)` | the recipe's table of kinds lists a under k |
| `level_before(a, b)` | every step of a is below every step of b |
| `row_before(a, b)` | every place of a is in a row above every place of b, and they share a column |
| `column_before(a, b)` | every place of a is in a column left of every place of b, and they share a row |
| `adjacent(a, b)` | a place of a and a place of b touch by a side |
| `between(a, b, c)` | a, b and c lie on one row or one column, a strictly between the other two |
| `distance(a, b)` | the greater of the row and column gaps between their nearest places; "near" is at most 2, "far" at least 5 |
| `size(a)` | the number of a's places; "big" is 4 or more, "small" is 1 |
| `count(i)` | the number of things in image i |
| `displaced(a, i, j)` | a's places differ between images i and j |
| `event(a, i, j, n, dir)` | from image i to image j, a moves n places at every step, all in the direction dir (up, down, left, right); n is 1 or 2 |
| `approaches(a, b, i, j)` | b is not displaced, and a's distance to b falls at every step |
| `recedes(a, b, i, j)` | b is not displaced, and a's distance to b rises at every step |
| `at_rest(a, i)` | a is not displaced over images i - 1, i and i + 1 |
| `precedes(e, f)` | event e's last image is before event f's first |
| `faster(e, f)` | event e's n is greater than event f's |

A step of 1 is what "slowly" says, a step of 2 what "quickly" says. A
graded word is used only where its bound holds: no training scene has a
distance of 3 or 4 under "near" or "far", nor a size of 2 or 3 under
"big" or "small".

*From a side.* An approach of a to b throughout which
`column_before(a, b)` holds is from the left; with
`column_before(b, a)`, from the right; with `row_before(a, b)`, from
above; with `row_before(b, a)`, from below.

### 14.3 Files and a row

- `bin/felt_world_corpus.py`: the world, the predicates, the expansion
  of recipes, `verify`, `freeze`, and `load(category, split)`, which
  returns documents and needs no model (as `bin/image_corpus.py`'s
  `load(name, split)` does).
- `data/felt/recipes/`: `cast.json` (the nine values with the forms of
  their words, the two kinds with theirs and their members, the counting
  words, the vocabulary) and one recipe for each category.
- `data/felt/category-NN-<name>.jsonl` and `data/felt/manifest.json`
  (seed, command, each file's hash and counts by split, each recipe's
  hash, the vocabulary, each value's reserved step and places, and a
  `never_direct` line as `data/math/manifest.json` has).
- `test/test_felt_corpus.py`.

A row, one JSON object to a line:

| Field | Holds |
|---|---|
| `key`, `category`, `name`, `variant`, `split` | which row it is |
| `operators` | the labels of §13.5, for reading results only |
| `entries` | a list of `{"text": ..., "scene": ...}`; a scene is `[[x, y, z, steps], ...]`, or `null` where there is no image |
| `meaning` | for each statement and the question: its entry, its predicate, its arguments, and whether it is true |
| `question`, `answer` | the entry that asks, and what is to be said: one word; in category 27 the counting words in order, a space between; the empty text where nothing follows (§14.5, category 24) |
| `inference` | null, or one of `converse`, `kind`, `chain`, `time`, `rule`, `open` |
| `set` | null, or the key that the rows of one contrast set share (§14.5) |
| `probe` | null, or what a held-out row measures: `commuted` or `unseen` (§14.5, categories 4 and 26) |

`load` gives a document with `sentences`, `question` and `answer` named
as `math_chain_corpus.ChainDocument` names them (its `question` is the
index of the sentence that asks), with `scenes` beside them. It is a
record of its own: `ChainDocument` is frozen and has fields this corpus
has no use for.

### 14.4 The frames

`{a}`, `{b}`, `{c}` are the words of values ("three"); `{k}` is a kind
as an adjective ("even"), `{K}` as a noun ("even"), `{Ks}` its plural
("evens"), `{aK}` with its article ("an even"). Lower case; a statement
ends with a full stop joined to its last word; a question ends with a
space and a question mark, as the math corpus writes its questions; a
comma is joined to the word before it, as the games write theirs. (The
model's reader makes each mark a word of its own either way,
`util.parse`.) An answer is one word, except in category 27. Where a
frame is of the past, the images of what it tells come before it as
entries with no text. Where these frames differ from §13.5's examples,
these stand.

| # | Statements | Questions and their answers |
|---|---|---|
| 1 | `{a}.` with no image; an image of a with no text | none |
| 2 | `this is {a}.` `that is {a}.` | `what is this ?` {a} |
| 3 | `{a} is here.` `{a} is not here.` | `is {a} here ?` yes, no. `what is here ?` {a} |
| 4 | `{a} is {k}.` `{a} is not {k}.` `{a} is {aK}.` `the evens are two, four, six and eight.` `the odds are one, three, five, seven and nine.` And of the kinds themselves, {l} the other kind: `no {K} is {l}.` `the {Ks} are not {l}.` | `is {a} {k} ?` yes, no. `what is {a} ?` {k} |
| 5 | `{a} and {b} are here.` `{a} or {b} is here.` `this is {a} or {b}.` `{a} is {k} and {b} is {k}.`, each with its own kind | `is {a} here ?` yes, no |
| 6 | `{a} is less than {b}.` `{b} is greater than {a}.` `{b} is not less than {a}.` | `which is greater, {a} or {b} ?` {b}. `is {a} less than {b} ?` yes, no |
| 7 | `{a} moved.` `{a} did not move.` `{a} stopped.` | `what moved ?` {a}. `did {a} move ?` yes, no |
| 8 | `{a} touches {b}.` `{a} reached {b}.` `{a} left {b}.` | `what touches {b} ?` {a}. `what did {a} reach ?` {b} |
| 9 | `{b} is touched by {a}.` `{b} was reached by {a}.` | `what is {b} touched by ?` {a}. `what was reached by {a} ?` {b} |
| 10 | `{a} is big.` `{a} is small.` `the big {K} is {a}.` `the small {K} moved.` | `what is big ?` {a}. `which {K} is big ?` {a} |
| 11 | `{a} is bigger than {b}.` `{b} is smaller than {a}.` | `which is bigger, {a} or {b} ?` {a} |
| 12 | `{aK} moved.` `the {Ks} moved.` `every {K} moved.` `all the {Ks} moved.` `both {Ks} moved.` `some {Ks} moved.` `no {K} moved.` | `did every {K} move ?` yes, no. `did {aK} move ?` yes, no |
| 13 | `{a} moved slowly.` `{a} moved quickly.` `{a} moved up.` `{a} moved down.` `{a} moved to the left.` `{a} moved to the right.` | `which moved slowly, {a} or {b} ?` {a}. `did {a} move up ?` yes, no |
| 14 | `{a} is above {b}.` `{b} is below {a}.` `{a} is to the left of {b}.` `{b} is to the right of {a}.` `{a} is on top of {b}.` `{b} is under {a}.` `{a} is next to {b}.` `{a} is near {b}.` `{a} is far from {b}.` | `what is above {b} ?` {a}. `what is {a} above ?` {b}. `is {a} above {b} ?` yes, no; and the same three for each relation, by the same pattern |
| 15 | `{a} is between {b} and {c}.` | `what is between {b} and {c} ?` {a} |
| 16 | `{a} is moving toward {b}.` `{a} moved toward {b} slowly.` `{a} moved toward {b} quickly.` `{a} moved away from {b}.` `{a} is moving toward {b} from the left.`, and the same `from the right.`, `from above.` and `from below.` | `what did {a} move toward ?` {b}. `what moved toward {b} quickly ?` {a} |
| 17 | `{a} was above {b}.` `{a} is above {b}.` `{a} moved.` `{a} will move.` | `what was above {b} ?` {a}. `is {a} above {b} ?` yes, no |
| 18 | `{a} is moving.` `{a} was moving.` `{a} has moved.` `{a} has stopped.` | `is {a} moving ?` yes, no. `has {a} moved ?` yes, no |
| 19 | `{a} moved, then {b} moved.` `{a} moved before {b} moved.` `{b} moved after {a} moved.` `before {b} moved, {a} stopped.` | `which moved first, {a} or {b} ?` {a} |
| 20 | `{a} moved toward {b}.` then `it moved toward {c}.`, with one move of step 1 and one of step 2 in either order; and, for the words, `{a} moved toward {b} slowly.` then `it moved toward {c} quickly.`, and the same with the quick move first | `which move was faster, the first or the second ?` second, first. `which move was slower, the first or the second ?` first, second |
| 21 | `{a} moved.` then `it stopped.` `{a} is here.` then `it is {k}.` `the {Ks} moved.` then `they stopped.` | `what stopped ?` {a}. `what is {k} ?` {a} |
| 22 | `the {K} that is above {b} moved.` `the {K} that moved is {a}.` `the {K} that is next to {b} is {a}.` | `what moved ?` {a}. `which {K} is above {b} ?` {a} |
| 23 | `{a} moved and {b} stopped.` `{a} moved but {b} did not.` `{a} moved and {b} moved.` | `did {b} move ?` yes, no |
| 24 | `if {a} is above {b}, {b} is below {a}.` `if {a} moves, {b} moves.` `if {a} moves, {b} stays still.` | none in train. In the inference split, after a rule of motion and `{a} moved.`, all with no image: `did {b} move ?` yes, no. With `{a} moved.` left out: the same question, and nothing follows |
| 25 | `every {K} is above {b}.` `every {K} is below {b}.` `no {K} is next to {b}.` | `is {a} above {b} ?` yes, no: in train, of a thing in the image; in the inference split, said only, of a digit whose kind category 4 states |
| 26 | `{a} plus one is {b}.` `{b} minus one is {a}.` `{a} plus {b} is {c}.` `{c} minus {b} is {a}.`, sums to nine, with no image | `what is {a} plus {b} ?` {c}. `what is {c} minus {b} ?` {a} |
| 27 | an image of one to nine things, with no text | no question is written. The answer is the counting words, one for each thing, in order: `one two three` for three things |

What a frame's words require of its images, where the word alone does
not say it.

- *`or`:* exactly one of the two is present.
- *`{a} is not here.`, and a `no` to `is {a} here ?`:* an image is seen,
  so the scene is a list, which may be empty; it is never `null`.
- *The determiners of 12:* at least two of the kind are present, and
  something of the other kind moves where the sentence denies. `{aK}
  moved`: exactly one of them moved. `the {Ks}`, `every`, `all the`: all
  of them. `both`: exactly two are present. `some`: two or more, and not
  all. `no`: none.
- *`stopped`:* an event of a, and then a at rest. *`did not move`:*
  something else moved.
- *`touches`:* adjacent now. *`reached`:* a approaches b and is adjacent
  at the end and not before. *`left`:* adjacent, and then a recedes.
- *`on top of`, `under`:* above or below, and adjacent. *`next to`:*
  adjacent in one row.
- *`is moving toward`:* its image is inside an approach. *`moved
  toward`, `moved away from`:* the approach or the recession is complete
  and a is at rest now. *From a side:* §14.2.
- *`what moved toward {b} quickly ?`:* something else moved toward b
  slowly in the same document.
- *`was`:* true at an earlier image and not now. *`will move`:* the
  entries after it show the event. *`is moving`:* its image is inside
  an event. *`has moved`, `has stopped`:* an event of a, and a at rest
  now. *`was moving`:* inside an event at an earlier image, at rest now.
- *The frames of 20:* the two events are one thing's, one of step 1 and
  one of step 2, each toward a thing that does not move. Which comes
  first is drawn, with the question and the order of telling fixed. Half
  the documents of every split tell the two moves with no word of their
  speed, so that the answer follows from what was seen; the other half
  name the speeds, for the words "slowly" and "quickly" (amended after
  the first build, §15.8).
- *A relative clause of 22:* at least two of the kind are present and
  exactly one satisfies the clause.
- *`if {a} moves, {b} moves`:* its document's images show at least
  three steps; a is displaced in at least two of them, and b is
  displaced in every step in which a is.
- *`if {a} moves, {b} stays still`:* at least three steps; a is
  displaced in at least two of them; b is displaced in none of those and
  in at least one other.
- *Category 27:* the things are of any values, at any places, and
  nothing else is in the image. The counting words are the cast's, "one"
  to "nine": the words that name the values, here in a second sense
  (the question of it is in "Completing the item", below).

Things the sentence does not name may be in an image, up to two of
them, except where they would change whether a statement is true, and
except in category 27.

### 14.5 The splits

For each category, as far as its frames allow:

| Split | Documents | What it is |
|---|---|---|
| `train` | 100 | true of its images |
| `validation`, `test` | 20 each | the constructions and casts of train, with other images |
| `generalization` | 20 | a cast held out of this category: never in its train, validation or test |
| `witness` | 20 | one statement that is false of its image, marked so; only in categories with images |
| `inference` | 20 | said only, with no image: the answer is not among the statements and follows from them |
| `contrast` | 20 sets of 12 | category 14 only: the first milestone's sets, below |
| `haystack` | 20 | the constructions and casts of train in a crowded image: a needle in a haystack |

- *Held out of a category with two things:* the four pairs of
  `math_chain_corpus.HELD_OUT_PAIRS`, (2, 7), (4, 8), (7, 3) and (9, 6),
  in either order. *With one thing:* each pair of frame and digit for
  which the frame's index and the digit sum to a multiple of 4. *Of
  category 26:* the sums 2 and 7, 3 and 5, 6 and 2, 1 and 4. *Of
  category 27:* each set of values whose sum is a multiple of 4. Every
  digit, every frame and every count is in train.
- *Reserved for each value* (suggestion 3): one step of its band,
  $25d + 1$ where $d$ is odd and $25d - 1$ where it is even, and eight
  places of the plane, those with $x + y + d$ a multiple of 8. In no
  image of any category does a thing of the value $d$ have its reserved
  step at any place, or lie on any of its reserved places, with one
  exception: category 2's generalization split, which is this and not
  the pairs above. Each of its documents is one image of one thing at
  its reserved step on one of its reserved places, asked `what is
  this ?`. Category 1's images with no text are the object alone, before
  its word: by this rule they show the six other steps at the other
  fifty-six places.
- *Kinds are never held out.* Category 4's train states the kind of
  all nine. Its generalization split asks of a digit in a frame it was
  not stated in. *What is held out of a kind is its denial, for two
  digits:* of six and of seven no split but `inference` says that it is
  not of the other kind, in a statement or in an answer of no, in any
  category. Their own kind is stated as for the others.
- *Both tests of a sum* (Alec, §15.10: "Both tests are good"). Of
  category 26, the four sums above are held out in the order written
  and taught the other way round: "five plus three is eight." is in
  train at least twice and "three plus five" nowhere but in
  `generalization`. Four more are held out both ways, with their
  differences: 2 and 3, 2 and 5, 3 and 4, 4 and 5, so that "three plus
  four", "four plus three", "seven minus three" and "seven minus four"
  are nowhere but in `generalization`. Its twenty rows are ten of each,
  marked `commuted` and `unseen` in `probe`; a row's statement is no
  fact of the sum it asks. No sum with one is among the eight, so every
  "plus one" is taught.
- *A needle in a haystack* (Alec, §15.10: "Why not needle in an haystack
  also?"). The two things added that no frame needs are a habit of the
  other splits and not a rule. A `haystack` document is one of train's
  constructions with every value its statements and its question leave
  free put into the image, wherever that changes no truth and no
  answer, and standing still: six things at the least in each image.
  Not made in categories 1, 2, 26 and 27, and not where a frame's
  meaning forbids the others (say which).
- *Contrast sets* (suggestion 1, "Three is above four"). A set takes two
  values a and b, a column, and two rows of it. Its two scenes hold a
  above b at those places, and b above a at the same places. Each scene
  is asked `is {a} above {b} ?`, `is {b} above {a} ?` and `is {b} below
  {a} ?`, each a document of one entry, the question with its image.
  The same six are made again with both things moved by one offset.
  Steps are drawn afresh for every document. The twelve share a `set`
  key. Ten sets are of pairs that train holds and ten of the held-out
  pairs.
- *Inference, by category:* 6, 11 and 14, the other way round and a
  chain of two (the other way round: one premise in one wording of a
  pair, asked in the other, as "one is less than six." asked "which is
  greater, one or six ?", or "three is above four." asked "is four below
  three ?"; the pairs are less and greater, bigger and smaller, above and
  below, to the left of and to the right of; amended after the first
  build, §15.8); 19, a chain through time; 24, the rule applied, eight
  that follow as yes and eight as no, and four with the premise left
  out, marked `open`, whose answer is the empty text; 25, through the
  kind; 4, two steps through the kinds (Alec, §15.10): after `{a} is
  {k}.`, asked `is {a} {l} ?` of the other kind, which is no; half of
  them with `no {K} is {l}.` or `the {Ks} are not {l}.` said in the row
  and half with it left to what train taught; six and seven are in at
  least half of each, marked `unseen` in `probe`. And, for its answers
  of yes, after the list of a kind's members, asked `is {a} {k} ?` of
  one of them. The others have none.
- Answers of yes and no are each between 40 and 60 in a hundred, in
  every split of a category that has them; so are `first` and `second`
  in category 20.

### 14.6 What the test checks

`test/test_felt_corpus.py` reads the frozen files, never generates, and
imports no module of the model.

1. Every file's hash and counts are the manifest's.
2. Every scene that is not `null`: places in range, z = 0, steps within
   3 of a value's 25d, one thing for each value, every thing's places
   touching, no place used twice. No thing has its value's reserved step
   or lies on one of its reserved places outside category 2's
   generalization split, and every thing in that split has both.
3. `verify`: every row is one its recipe produces, and every `meaning`
   record, evaluated again on the row's images, has the truth recorded.
   Outside `witness` every statement is true. Every answer is the
   world's.
4. Every word is in the declared vocabulary and no text has a digit
   character. The manifest records the greatest length of a text, in
   words and in bytes.
5. The English that a rule can check: `is`, `was` and `has` with one
   subject and `are` with several; `an` before "even" and "odd" and `a`
   elsewhere; no doubled word; one terminal mark.
6. The held-out casts are in no train, validation, test, witness or
   haystack row of their category; every digit, every frame and every
   count is in train. Outside `inference` no row of any category denies
   six or seven the other kind. Of category 26, a sum held out both ways
   and its differences are in no row outside `generalization`; a sum
   held out one way is in train the other way at least twice; each
   `generalization` row's statement is no fact of the sum it asks.
7. A scene is `null` in every entry of an `inference` row, in frame 1's
   word alone and in category 26, and nowhere else. An `inference`
   row's question does not ask a fact one of its statements states in
   the statement's own wording: where the fact is a statement's, the
   wording is the other of its pair. An
   `open` row's answer is empty, and its record says that neither yes
   nor no follows.
8. The balance of yes and no, and of `first` and `second`.
9. Every contrast set has its twelve rows. Its two scenes differ only by
   the exchange of the two things, and its second six only by one
   offset. Exchanging the things turns every answer, and `is {b} below
   {a} ?` has the answer of `is {a} above {b} ?`.
10. In category 27 every answer is the first n counting words, for the
    n things of its image.
11. `load` returns the documents with no model built.
12. Every image of a `haystack` row holds six things or more, and the
    row's statements and answer hold as its recipe's do (check 3).

### 14.7 What it does not touch

No source of the model, no existing file under `data/`, and not
`data/training-corpus.json`. Nothing reads the new files yet: a loader
for them and the scene beside a sentence (§3) are the model's part of
this item.

### 14.8 Text for the implementer

Written 2026-11-18 for Codex; amended 2026-11-23 for an Opus implementer
that Claude starts and reviews (H.1). It shares no file with item 1.5.

~~~
Item 1.6, the corpus only. Read §12 to §14 of
doc/plans/2026-11-18-item1-6-felt-math-world.md, and the sections added
after §14; §14 is what you build, and where it differs from an earlier
section, §14 stands. It changes no source of the model and no existing
file under data/. If you find that it must, stop and say which file and
why.

1. bin/felt_world_corpus.py, in the manner of bin/game_corpus.py
   (recipes expanded by a seeded random.Random; a manifest of hashes),
   with load(category, split) as bin/image_corpus.py has it: the world
   of §14.1, the predicates of §14.2, the expansion of recipes into
   documents, verify, freeze and load. It holds no word of any
   language: a word reaches a sentence only from a recipe. A document
   is made by drawing a world, and for the frames of motion a short
   script of events, until every statement's meaning is true of the
   images; verify evaluates the same predicates on the frozen row.
2. data/felt/recipes/: cast.json and one recipe for each of the 27
   categories (§13.5's 26, and counting), with the frames of §14.4
   exactly as written there and the conditions under its table. Add no
   frame and change no wording. If a frame cannot be made valid, or its
   English looks wrong to you, leave it out and list it in the return;
   Claude and Alec read the frames before the files are final.
3. Freeze with
   .venv/bin/python bin/felt_world_corpus.py freeze --seed 20261118 --recipes data/felt/recipes --out data/felt
   into the files of §14.3, with the splits, the held-out casts, the
   reserved steps and places and the contrast sets of §14.5. Where a
   category cannot reach a split's count with distinct documents,
   freeze what it gives and report the count; do not repeat a document
   and do not loosen a condition. The seed is the corpus's identity; it
   is not chosen so that a check passes.
4. test/test_felt_corpus.py, the eleven checks of §14.6. Run it with
   BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/test_felt_corpus.py
   and run test/test_game_corpus.py and test/test_math_corpus.py, which
   must pass unchanged. No sweep, nothing under RUN_SLOW, no skip, no
   expected failure, and no stopping gate: no source of the model
   changes.
5. Touch nothing of §14.7. Leave the docs; Claude writes
   doc/Training.md and doc/Testing.md when you return.
6. Return as §15, at the end of the same document: the files and their
   counts by category and split; every frame left out and why; every
   category that fell short of a count; the layout of a recipe as you
   made it; twelve documents printed whole, one each from categories 3,
   6, 12, 14, 16, 18, 20, 22, 24, 25 and 27 and the first four rows of
   one contrast set, each with its images drawn as small grids of
   digits; and anything in §14 that was unclear or that you had to
   decide. You do not stop on a question of design: propose, build, and
   mark what you decided, and Claude takes it to Alec. A defect of your
   own test's setup you repair without stopping.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed. Receipts stay local and are not linked.
~~~


## Added 2026-11-19: reading without comprehension, before reading with it

Alec (sigma-pi plan, §19.88): "We should try to teach with reconstruction
only, and try just reading without comprehension as a task prior to
reading with comprehension. Add this detail to 1.6".

A first task, before any sentence is parsed for its meaning or any question
is answered: the model reads a sentence and puts it back. No reading is kept
for comprehension. The reconstruction's error is the only teacher, and what
it has to teach is where each percept is, so that the sentence is put back
in order (sigma-pi plan, rule 49: the error is over what a percept is and
where it is).

What is known from item 1.7's trial of the looks (its round 2): taught by
the reconstruction alone, held-out orders were read at 4 of 6; taught beside
the reading lesson, at 6 of 6. So the task is tried knowing that the
reconstruction alone has not yet been enough.

To specify with the rest of this item: the corpus of the task (the same
sentences as the comprehension lessons, or shorter ones first); when a model
is said to read (every sentence of a held-out set put back in order); and
whether the reading lesson of item 1.7 stays as a second teacher or goes.

## Added 2026-11-20: a song over the output, and counting as its test

Alec (sigma-pi plan, §19.99): "Let's add a test for sequence learning in the
sense of "singing a song over the output". For 1.6, counting would be a good
test of that that might be simpler than moving focus reliably over words."

A test of a behaviour chain: a sequence that has been taught is produced
over the output from its first item alone, each item from the ones before
it, with no input between. Here the sequence is counting: given `one`, the
model says `two three four` and on to the end of what it was taught.

What it tests that nothing else in this item does: that an order of acts is
held and replayed (from STM, from LTM, and in the end as concepts), which is
what a learned way of reading a sentence also is. It asks less than reading
does, since nothing has to be found in the input.

To specify with the rest of this item: how far the count goes (the cast is
the single digits, §13.4); what teaches it (the sentences of the corpus that
count, or a lesson of its own); that a second sequence sharing an item with
the first, counting down or counting by twos, is continued as its own; and
what is printed (the testpoint `TestSingASong`, sigma-pi plan, certificate
59).

Also from §19.99, for the order of this item's lessons: attention is taught
by successive approximation, "we can't teach focus directly, but we can
reward behaviors that require a successively narrow focus" (sigma-pi plan,
rule 70). The grades are the sentence, then its words in order (reading
without comprehension, above), then a word's parts.

## Added 2026-11-20: the item's weight, and scene description in place of anthropomorphic verbs

Alec, 2026-11-20: "1.6 is mostly about multi-modal and good testing. As an
aside, it should not use anthropomorhic language like "one runs to two from
the left", but scene description language of the 3D space in which 1 → 2
over some background of 0's. That would look more like "one is moving toward
two from the left"."

**What the item is for.** A second modality beside text, and tests good
enough to trust what the two together show. The architecture is item 1.7's
and is not changed here.

**The corpus describes a scene.** A sentence says what is in the space and
how it changes: a digit at a place over a background of zeros, moving toward
or away from another, from a side, slowly or quickly. It does not say what a
digit *does* as an animal would. So:

| Frame of §14.4 | As written | Restated |
| --- | --- | --- |
| 16 | `{a} walked toward {b}.` `{a} ran toward {b}.` `{a} walked away from {b}.` `{a} ran away from {b}.` | `{a} is moving toward {b}.` `{a} moved toward {b} slowly.` `{a} moved toward {b} quickly.` `{a} moved away from {b}.`, and with the side: `{a} is moving toward {b} from the left.` |
| 16, questions | `what did {a} walk toward ?` `what ran toward {b} ?` | `what did {a} move toward ?` `what moved toward {b} quickly ?` |
| 20 | `{a} walked toward {b}.` then `it ran toward {c}.`; `which was faster, the walk or the run ?` | `{a} moved toward {b} slowly.` then `it moved toward {c} quickly.`; `which move was faster, the first or the second ?` |

and §12.5's "walked toward 2" and "ran toward 3" likewise. `touches`,
`reached` and `left` (frames 8 and 9) are kept: they say where two things
are and were. The cast, the images and the events are unchanged; only the
words of the frames are. *To settle when the generator is next run:* whether
its frozen examples hold frames 16 and 20 as first written, and if so they
are generated again. *Settled 2026-11-23:* nothing had been frozen; §14.4 now
holds the restated frames, with the side given for each of the four sides.

## Suggestions for the item, from `suggestions.txt` (GitHub main, 6eebcc29ff, 2026-11-21): relevant, and taken in

~~~
Item 1.6 suggestions — Grounding grammar in the felt world

First milestone — “Three is above four”

This one sentence exercises object recognition, word-to-object binding, and
argument order on a scene small enough to inspect on the M2. Keep all 26
constructions in the corpus, and use this one to see where learning succeeds
or fails before moving to more expensive training.

1. Give the chooser two interpretations with different consequences

   Place three and four in the same column. Ask both questions of both scenes.

                           Three above four     Four above three
   Is three above four?           yes                  no
   Is four above three?            no                 yes

   “Is four below three?” should give the same answers as the first row.
   Translate both objects together, vary their intensities within their
   bands, and repeat on digit pairs held out of this construction. Count a
   set as correct when all its answers are correct. This makes it possible
   to distinguish reliance on wording, argument order, or absolute position.

   Where the explore and exploit paths bind the two roles in opposite ways,
   compare their completed reconstruction-plus-answer costs on both scenes.
   With the wording fixed, does exchanging the objects reverse which binding
   has lower cost? If the answers change correctly but the path costs tie,
   the answer alone does not show that the chooser received a learning signal
   from that comparison.

2. Distinguish an unavailable observation from an observed empty board

   An empty board is visual evidence against “three is here.” Hearing a
   premise without seeing a board supplies no visual evidence either way.
   Represent the first as scene: [] and the second as scene: null. Carry
   scene availability through the reader and reconstruction loss. This
   matters for text-only inference and for the model's for/against lanes.

3. Establish object identity before introducing its word

   Each digit has seven possible intensity values. Reserve an interior
   value and several locations for each digit. During object-only learning,
   present the other values without words. Then examine whether the held-out
   value at a new location activates the same object. Introduce the name
   afterwards. The generator knows the intensity bands. The learner sees
   only the observations. This separates a perceptual failure from a later
   failure to bind a word or compose a sentence.

4. Make event comparison and inference depend on the example

   Category 20. “Which was faster, the walk or the run?” always has the
   answer “run.” Keep that wording for vocabulary. To measure comparison,
   describe the movements neutrally and ask “Which was faster, the first
   movement or the second?” Exchange the speeds while keeping the question
   and event order fixed. The answer should change.

   Category 24. The written inference frame always answers “yes,” while
   the split calls for both “yes” and “no.” For example, teach either
   “If three moves, four moves” or “If three moves, four stays still.”
   In both cases supply “Three moved” and ask whether four moved during
   that step. Each answer follows from its premise and rule. An omitted
   premise would leave the answer unknown.

Local decision before remote compute

Record the number of teaching examples and elapsed training time to reach
a declared accuracy on complete held-out spatial sets. Report naming at new
positions, role reversal on observed scenes, and inference from text alone
separately. Each result points to a different part of the model to improve.

References

Item 1.6 felt-world design, especially §§6, 13.1, and 14.1–14.6
https://github.com/arborrhythms/BasicModel/blob/b3d2e3ae899ac6eef9a184398d7f8c37b31fac17/doc/plans/2026-11-18-item1-6-felt-math-world.md

Gardner et al., “Evaluating Models’ Local Decision Boundaries via Contrast Sets”
https://aclanthology.org/2020.findings-emnlp.117/
~~~

*Claude's evaluation.* All of it bears on this item, and none of it on item 1.7.

- *The first milestone, "Three is above four".* Taken as the item's first gate before any costlier training: one sentence that needs object recognition, the tie of a word to its object, and the order of the arguments, on a scene small enough to inspect.
- *1, two interpretations with different consequences.* Taken: the four-way set (both scenes, both questions, and "below" for the same answers), both objects moved together, intensities varied within their bands, digit pairs held out, a set counted correct only when every answer in it is. Its second half is a measure and not a test of answers: where the explore and the exploit paths bind the two roles in opposite ways, whether exchanging the objects reverses which binding costs less. It belongs with the chooser's moves of item 1.5 (item 1.5 plan, "Codex's four moves for the chooser", moves 1 and 2).
- *2, a scene that is not available against a board that is seen empty.* Taken, and it corrects §14.1, which writes "a sentence with nothing felt" as an empty list, the same as a board seen with nothing on it. They are two of the four corners (the tetralemma's two lanes): an empty board is evidence against, a missing scene is no evidence. `scene: []` for the first and `scene: null` for the second, carried through the reader and the reconstruction cost. It is needed by the inference split of §14.5, whose premises are heard with nothing felt.
- *3, the object before its word.* Taken; it agrees with the order of learning already decided (words, objects, then meta concepts; sigma-pi plan §19.25) and adds what that order lacked, a held-out intensity and held-out places for each digit, so that a failure of perception can be told from a failure to tie a word.
- *4, categories 20 and 24.* Taken. Category 20's question always has one answer; the restated frame of this plan ("which move was faster, the first or the second ?") already describes the movements neutrally, and the suggestion adds the part that matters: exchange the speeds with the question and the order of events fixed. Category 24's inference frame always answers "yes"; it gets a rule for each answer ("if three moves, four moves" and "if three moves, four stays still"), and a case with the premise left out, whose answer is not known.
- *The measure before any remote machine.* Taken as the item's record: examples and elapsed time to a declared accuracy on whole held-out spatial sets, with naming at new places, the roles exchanged on a seen scene, and inference from text alone reported apart.

*Decided (Alec, 2026-11-21):* "The missing scene does not affect reconstruction cost." So `scene: null` charges nothing on the scene; an empty board is reconstructed and charged like any board. *Decided (Alec, 2026-11-21), on the unit of the count:* "Per answer, unless I hear better rationale". The gate is the accuracy per answer; the count of whole sets right is reported beside it, with which cells fail, and is not a gate.

## An object counting task (Alec, 2026-11-22)

Alec, while item 1.7's looks were being taught to read in order: "make sure to introduce an object counting task as 1,2,3,4 counting word on the output." The input holds objects; the output is the counting words, one for each object, in order. It uses the mechanism of sigma-pi §19.126 to §19.128 (a concept lit for the task; the memory of the last look; the output's cost alone), first tried as a toy in item 1.7 (round 6d, tenth pass, step 9). In this item it is a category of the corpus.

## A common perceptual space (Alec, 2026-11-22, a possibility)

"By the way, .what and .where might be muxed together at the basic percept just as different modalities (such as words and numbers) mux together. That would cause all percepts to live in a common (3d) perceptual space, even if some were of reduced rank (e.g., the location of smell might be considered as a point at the nose)." Decided the same day for the `.where`: "NB: we are treating numbers as felt things. I would like to see .where as a 3D space for all models, it simplifies things." So the numbers of this item are felt things, and they and the words have their places in one space of three dimensions (sigma-pi §19.132; built in item 1.7, round 6d, twelfth pass, f). Whether `.what` is muxed as a modality is stays a possibility.

*Corrected 2026-11-23 (Claude, on reading the source at `ad8fafea92`):* the space of three dimensions is decided and is not built. The twelfth pass's return says "Item f (three axes) is not started" (sigma-pi §23.6.18); `WhereRegistry.axes` is one axis, and `doc/FutureWork.md` has "The grid in three dimensions (decided, not built)". See "Completing the item", C.2 and question 1.

## What is felt and what a number is (Alec, 2026-11-22)

"Text will be 1D: zeros for dim 2 and 3.
The perception of intensity (feelings) can be 1,2, or 3D.
Images will generally be intensity, so zeros for dim 3. The can in principle be 1D
Volumes will be no different than 3D images.

Numbers are higher-order, so symbolic (they are pure forms, and do not have a perceptual representation). So in the tests where we had been talking about "one is above two", it should be seen as a shorthand for "the pixel value 1 (at x1,y1,z1) is above the pixel value 2 (at x2,y2,z2)": we are referring to image data. We can continue to refer to shapes in feeling space as level sets to teach basic geometry."

For this item:

- The felt world is image data: intensities at places of the one space of three dimensions. Text is on the first axis; an image on two, or one; a volume on three.
- A number has no percept. It is a symbol of a higher order. The corpus's sentences about numbers ("one is above two") are sentences about pixel values at places, and are written and tested as such.
- This corrects the section above, where the numbers were called felt things: the intensities are felt, and the numbers are symbolic.
- Shapes are level sets of the intensity, and the geometry lessons are taught on them.


## Hand-over: completing the item after item 1.7 (Claude, 2026-11-23)

Alec, 2026-11-23: "After 1.7 gets pushed, I would ask you to complete 1.6
while codex works independently on 1.5. You and I had a conversation about
that thread already called "1.6", which I would ask you to complete on a
different machine. The Codex text is §14.8". And: "Please use Opus for
worker threads, as usual."

This section is for the session that completes the item. It was written in
the session that reviewed item 1.7, and says what that session knows and
this document does not.

### H.1 Who builds what

- Claude completes item 1.6: the design that is still owed, the texts, the
  review. Code and tests are written by Opus implementers that Claude
  starts and reviews, not by Claude and not by Codex.
- Codex works on item 1.5 (training speed) on `main` at the same time. So
  item 1.6 is built on a branch of its own, in a worktree of its own, and
  takes `main` in by merging. Neither moves the other's branch.
- §14.8 is addressed to Codex. It is now a text for an Opus implementer;
  its last paragraph and its "Return as §15" stand, with Claude as the one
  it returns to.

### H.2 Where item 1.7 stands at its close

Read in `2026-11-14-sigma-pi-layers.md`: §19.142 to §19.149 (the last
decisions), §23.7 (Codex's measurements and the sweep's failures), §23.8
and §23.9 (Claude's reviews, with Alec's answers).

What it can say, each measured from input to output: a word, a word with
its mark and a sentence of two words are said back; sequences of familiar
words are read in source order after the reading lesson; the input is put
back by one pass at the opening.

What it cannot yet say: that the model learns where to look without the
lesson; that a novel word is read in its place; that it speaks on a
condition; any age above 0.00.

What changed under this document since §2.5 was written:

- *The reconstruction is one pass at the sentence's opening, with attention
  wide open* (sigma-pi §19.149). Thought, intuition and action are optional
  after it. "Reading without comprehension" (the section added 2026-11-19)
  is now this pass, and is what the item starts from.
- *Reading in order is taught by a paced lesson*
  (`bin/pretrain_reading.py`, `data/lessons/reading-order.json`, the key
  `architecture.data.readingOrderLesson`; sigma-pi §23.10). Each training
  page comes with the word due at each look, as text; the look's expected
  spelling is compared with it after the walk. No occurrence is named.
  Outside the lesson the same cost points at the look's own choice, which
  sharpens what the look already prefers; whether that holds or spoils an
  order was not measured when this was written. Measure it before a lesson
  of this item leans on it.
- *Two collections are the same only at full cover both ways*
  (`conceptMatchPart` = 1.0, sigma-pi §23.7). The tie of a word to its
  object must not rest on a near match.
- *The first measure of the tie* (sigma-pi §19.148): "The first grounding
  measure presents the same two objects in both spatial orders and asks
  about both operand orders."
- *The age* is a parameter of the model that the assessment writes
  (`Assessment.assess_age`, `bin/AgeChart.py`, `doc/ModelAge.md`). A test of
  a saved model may name the row it needs (`requires_mental_age`), and an
  unmet age fails.
- *Every claim is measured from input to output.* Nothing is put into the
  middle of the model to make a thing work or to measure it.

### H.3 What is not this item

Item 1.4 (`2026-11-23-item1-4-verbs-in-the-field.md`) comes after items 1.6
and 1.5: a verb's extension, the last sentence's basic percepts as a second
field, and episodic memory as percepts. None of it is built here. It asks
one thing of this item: that a document keeps its successive images
(§14.1, "Time") and its frames of motion as designed, since those are what
it will observe a change in.

### H.4 The order of work

1. *Bring §14 up to the sections after it, then have the corpus built.* The
   corpus is not built: there is no `bin/felt_world_corpus.py`, no
   `data/felt/`, no `test/test_felt_corpus.py`. Before §14.8 goes to an
   implementer, §14.4's frames and §14.8's list are amended by what came
   later in this document: scene description in place of the verbs of an
   animal (2026-11-20, frames 16 and 20); the object counting category
   (2026-11-22); that intensities are felt and a number is a symbol with no
   percept, so "one is above two" is a sentence about pixel values at places
   (2026-11-22); and what was taken from `suggestions.txt` (2026-11-21).
   The corpus touches no source of the model, so it can be built while the
   rest is designed.
2. *Read the model as item 1.7 left it, and write what §10's first round is
   now.* §2 and §3 were written before the looks, the common space of three
   dimensions and the opening reconstruction were built. Say what exists of
   the scene beside a sentence, of places on three axes and of the one
   reader, before specifying what is missing.
3. *The questions still open with Alec* are in §11.6, §12.7 and §13.8 and
   in the sections added afterwards ("To specify with the rest of this
   item"). Gather them, drop those the later sections answered, and put the
   rest to him once, before any text for the model's part.
4. *Then the rounds of §10*, each with its targeted tests, and the item's
   one default sweep at its end.

### H.5 Rules that held through item 1.7

- Commit locally; never push unless Alec says so in that conversation. The
  pointer of the outer repository is his.
- A text for an implementer is written in this document and names its
  sections; it is written only after the questions it rests on are
  answered. An implementer does not stop on a question: it proposes,
  builds, and marks what it decided.
- No seed is pinned to make an assertion pass. No skip and no expected
  failure is added. Nothing is put under `RUN_SLOW`. The two stopping gates
  are not changed: `test/test_mm_xor.py::TestMMXorConvergence::test_convergence`
  and `test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`.
- No second path and nothing kept for an older form. Something removed is
  moved to a dated folder under `/tmp`, not deleted.
- Every default is in `data/model.xml` and `data/model.xsd`; none in code.
  `data/BasicModel.xml` gains no key. No language is named in code: a word
  reaches the model from a data file.
- No rule that forbids a walk. A tie is not an identification. Evidence is
  summed by degree, not decided by a winner.
- Rounds run targeted tests; the full sweep is run once for the item.
- The standing docs (`doc/*.md`, not `doc/plans`) say what the code does
  now: no dates, no item or round numbers, no history. Plans go to
  `doc/FutureWork.md`, history to `doc/DevelopmentHistory.md`, decisions to
  this document.
- Write `.where`, `.what` and `.when` with the dot. Use Alec's words where
  he gave them, and plain terms otherwise.

## Completing the item: the first report (Claude, 2026-11-23)

Alec, 2026-11-23, starting it: "Please sync it and begin item 1.6, which
you are to complete while Codex works independently on item 1.5." And of
the order: the gates on this machine before anything is changed; then H.4,
step 1; then steps 2 and 3 while the corpus is built, with the questions
put to him "once, before any text for the model's part".

### C.1 The sync, the branch, and the gates on this machine

- `main` was fast-forwarded from `2109a42a5d` to `ad8fafea92`, 642 commits,
  with nothing local. The branch is `item1-6`, in the worktree
  `.worktrees/item1-6`, cut from `ad8fafea92`. Nothing is pushed and the
  outer repository's pointer is not moved.
- On this machine (an Apple M2, 24 GB, eight cores; not the one item 1.7
  was measured on), on the CPU, with `RUN_TRAINED` and `RUN_SLOW` unset, in
  one run of the bounded runner before any file was changed: the two
  stopping gates and the seven tests of `test/test_item17_echo.py`.
  **9 of 9 passed**, in 318 s (the worker 291 s, at most 1.6 GiB).
- *Git in this worktree.* The checkout is a submodule, and its
  `core.worktree` is inherited by a linked worktree and points nowhere
  there, so plain `git` sees every file as deleted. Claude gives every git
  command its work tree (`git --work-tree="$PWD"`) and adds files by name.
  The implementers do not run git: Claude commits what it has reviewed.
  This is the one place §14.8's last paragraph departs from H.1.

### C.2 What exists in the model as item 1.7 left it (H.4, step 2)

Read at `ad8fafea92` by three Opus readers that changed nothing; the rows
on the scene, the place space, the percept, the tie, the looks and the
reconstruction Claude then read again in the source.

| Of §2 and §3 | As built | Not built |
|---|---|---|
| *A scene beside a sentence* | A row is `text` and at most one `scalar`; any other key is refused (`InputSpace.prepInput`). The value is joined to the sentence's field as one column, once, at the sentence's opening and only for the first sentence of a presentation (`SentenceField.open_field`, `ScalarPercepts.join`). At a kept close in training every word of the sentence is paired with it (`ScalarPercepts.commit`). | A `scene` key; more than one felt value with a sentence; no image told from an empty one; any loader of such rows (`bin/data.py` loads no frozen corpus; a felt value reaches the model only from tests and `Assessment.read_sentence`). |
| *Places on three axes* | One axis (`WhereRegistry.axes`). Its ranges, in order: `text`, `numeric` (the product of the input shape, in a numeric model only), `scalar` (one address), `parts`, `wholes`, `conceptual`. A place is one integer, written as four wires on two rungs (`WhereEncoding`); the table of wires is fixed and is made at its first reader. The way back matches a reading's wires to the grid in the `text` range alone. | The space of three dimensions: decided (sigma-pi §19.132) and not started (§23.6.18, where its size is given as 13 files of `bin/` and 76 of `test/`). Any key for a felt modality's shape. |
| *One reader* | Two. The value beside a sentence has a reading and a way back of its own (`ScalarPercepts`); the numeric stem has `NumericPercepts.read`, and a configuration still reads numbers or text, not both. They share `feeling`, `BasicPercepts.perceive_value` and `value_code`, and the store of `PerceptCollections`. | One reader. Intensity images: `bin/image_corpus.py` packs one bit to a pixel. |
| *What the percept is* (§3.4) | Each exact value met has a seat and is made a row at first sight (`perceive_value` calls `promote`): rule 20 holds for words and not for values. A value's code is its bins up to the value, set out by a draw keyed on its place (`value_coordinates`), so one intensity at two places has two codes that share nothing by construction, and part-of orders values within one place only. The value beside a sentence has the one place of its range. | The level as one percept wherever it is, with its place on the occurrence. |
| *The tie* (§2.5) | The object is made for the word, from what the word recurs with (`PerceptCollections.observe`: gathered under `defining_word`, made at `concept_mint_count` = 3). An object that already reads the percept wholly is looked for first. The certainty of a tie is kept (`WordTies`) and is the trust of the word's definition. A word's concept over several exact values is their interval, and a value inside it belongs. `conceptMatchPart` is not read on this path. | Rule 31: no object is learned from a feeling alone, and no meta concept afterwards. `TestCrossModalTie` prints its stages and asserts one thing, that a second word is tied to the first word's object; its record was "each raises the other 0 and 0" (sigma-pi, line 22399). |
| *The looks* | A look takes a word bracket of the sentence's text. The request's four `.where` wires are compared with places of the `text` range. The joined value is read at the opening only. | A look that takes a felt thing. |
| *The opening reconstruction* (§19.149) | Text alone: the joined column has no wires, the paste keeps the `text` range, and the cost compares bytes. The value is read back by `reconstructModalities` and charged nothing. An absent value and a value of zero are treated alike. | The scene put back and charged; `null` charging nothing. |
| *Asked and answered* | A felt value may be taken by the thought's bind, and a bare value is said by the word of the object that holds it. A supplied answer is scored after the response is fixed. The two tests of it from input to output (`test_item18_recognition.py`, `test_90_*`) are slow and expected to fail. | A passing measure of "what is this ?" from input to output. |
| *A sequence over the output* | A toy outside the model, and one slow test in the model that asserts only the length of its record (`test_item17_own_reading.py`). | `TestSingASong`; counting. |

### C.3 Where the hand-over and the source differ

The hand-over was written at `8503e9cee0`; ten commits followed it on
`main` before the push. By the rule of this session the source stands.

1. *The space of three dimensions is not built.* H.4, step 2, and the
   section "A common perceptual space" say it is. C.2 has what is.
2. *The paced lesson is gone* (`e1760b934b`). H.2 says a training page
   comes "with the word due at each look, as text". Now training and
   assessment give the page alone: before a look the request forecasts the
   activity of basic percepts at their places, and the pyramid compares
   that with what the field holds where the model looked
   (`doc/Testing.md`, `doc/GradientFlow.md`). H.2's caution, that outside
   the lesson the cost points at the look's own choice, was written of the
   cost that was replaced. Codex's one paired comparison: with the term,
   122 of 122 held-out sequences read in order; without it, 24 of 122.
3. *Reading in order is lost during grammar training* (`suggestions.txt`,
   "Item 1.7 after Claude's §23.10 review"). In the landing sweep the
   verbal XOR certificate gave 0 of 4 at the grammar's input and through
   its inverse, every pair reversed, after a reading prerequisite of 58 of
   58; the order was lost within the first 20 epochs of grammar training.
   It passed 4 of 4 in the focused run. It is open, to be assessed again
   at item 1.4. So H.2's "sequences of familiar words are read in source
   order after the reading lesson" holds for the lesson and not yet
   through a grammar's training.
4. *The landing sweep:* 7,459 cases, 7,086 passed, 336 skipped, 35 failed;
   both gates passed.
5. *Item 1.7 did not build rule 31.* §2.5 divided the work so: item 1.7
   builds the tie of a word to a felt value with no place, and this item
   gives it places. The tie that exists is the one item 1.8 left, with
   the look for an existing object added.

### C.4 What §10's first round is now

*A sketch, not a text. Each round rests on the questions of C.5 named
beside it.*

§10 began at "the modality and the tie" and took the tie from item 1.7.
The first gate is now one sentence, "three is above four" (the
suggestions, taken in), and what it needs was not there to take. In order:

| Round | Builds | Measured from input to output | Rests on |
|---|---|---|---|
| The corpus | §14 | The frozen files are valid (`test/test_felt_corpus.py`) | nothing; under way |
| One: the scene is perceived | Places for a scene; the `scene` of a row, with no image told from an empty one; a loader; one reader, each pixel that is not background a percept at its place in the sentence's field; the opening pass puts the image back and is charged for it | An image alone is put back, at varied steps and at places not met; a sentence with its image puts both back; a sentence with no image is charged nothing for one | questions 1 and 2 |
| Two: the object and its word | The object, alone and then named; "what is this ?"; several things present | Categories 1 and 2: a value named at its reserved step and places; naming with others present | question 3 |
| Three: the first milestone | What reads an arrangement | Category 14's contrast sets, per answer, on pairs held out | questions 4 and 5 |
| After | Kinds and what is said; counting; shapes; succession and motion; events compared; the witness | their categories | each a conversation of its own, as §12.6 |

Round one is "reading without comprehension" for an image: the one pass
at the opening, with the image as its input. Certificates F1, F2, F11 and
F12 of §7 are its own; F3 to F7 are round two's; F9 is round three's.

### C.5 The questions still open with Alec, in one list

Gathered from §9, §11.6, §12.7, §13.8, the sections added after §14, and
what C.2 and C.3 found. Each has Claude's default. Questions 1 to 3 are
needed before round one's and round two's texts; 4 and 5 before round
three's; 6 to 8 are in the corpus as defaults and are changed by editing a
recipe and freezing again.

1. *Who builds the space of three dimensions, and when.* It is decided
   and not built, and round one needs places on two axes at least.
   (a) This item builds it first, as a round of its own on its branch;
   Codex's item 1.5 is changing `main` meanwhile, and by the estimate 13
   files of `bin/` and 76 of `test/` name the width. (b) Codex builds it
   on `main` and this branch takes it in. (c) Round one gives the felt
   modality a range of 512 addresses on today's one axis, row by row
   (§3.2 as first written), and the three axes come after. *Default:*
   (a). With (c) nothing in a place's wires says that two places share a
   column, which is what "above" is about. With it, his own question:
   four wires to an axis or three.
2. *What the percept of a pixel is* (§3.4; question 5 of §9, never
   answered). As built a value's code is drawn by its place, so the 3 at
   one place and the 3 at another share nothing. *Default:* the level is
   the percept, the same at every place, and its place is on the
   occurrence as its wires, as a letter is one percept wherever it stands.
   XOR's two inputs are then told apart by their wires. This is near his
   possibility of 2026-11-22, `.what` and `.where` muxed at the basic
   percept.
3. *The object before its word.* His order is words, objects, then their
   meta concepts (rule 31), and suggestion 3 was taken. As built the
   object is made for the word, and nothing gathers the steps of one value
   into an object while no word is present. (a) Round two keeps what is
   built: category 1's images are taught by the reconstruction alone, the
   object is made when the word comes, and what is measured is the name at
   the reserved step and places, which the word's interval gives. (b)
   Round two builds the object first. That needs a rule for which
   readings are one object with no word to say so. *Claude's proposal for
   (b):* levels met with no gap between them wider than the variation are
   one object, as touching places of one level are one thing. *Default:*
   (b) if he accepts the rule, else (a) with rule 31 left to a later
   round.
4. *What reads an arrangement* (§6, §11.3; open since 2026-11-18). The
   first milestone waits on it. The two candidates on record: the
   arrangement as an ordered run that is a percept of its own; the
   position selection of `doc/Language.md` (`preposition`, `compound`),
   with the scene's things as its cases and a test that compares two
   places. *Default:* Claude writes a proposal in this document after
   round one returns, and he chooses then.
5. *Reading in order through grammar training* (C.3, point 3). The
   milestone turns on the order of two words. (a) Build on it as it is
   and measure, in round three, whether the order of the looks is kept
   while the felt lessons are taught, reporting a loss as item 1.7's open
   point. (b) Wait for its repair. And whether the reading lesson is a
   prerequisite of this item's models, as Codex proposed for every text
   grammar model (sigma-pi §23.9, "his to accept"). *Default:* (a), with
   the lesson as a prerequisite.
6. *Counting* (category 27). (i) The counting words are the words that
   name the values, so "three" is said of a pixel value in one category
   and is the third word of a count in another. *Default:* accepted, as
   one word has two uses in English. The other way: the things counted
   are all of one value, or the count is asked in words ("how many are
   here ?"). (ii) Nothing in the input names the task: an image with no
   text, and the count asked for on the output alone, as he decided for
   reading and counting over words (sigma-pi §19.129). Category 1's images
   with no text ask for nothing. *Default:* as written. (iii) The song (a
   count continued from its first word; a count with a number left out) is
   not in the corpus and has no test in this item until it is talked
   through, as `doc/ModelAge.md` says of it.
7. *What is said when nothing follows* (category 24, the premise left
   out). *Default:* nothing is said: the answer is the empty text. The
   other way: a word for it.
8. *The bar* (§13.7; question 3 of §13.8; the suggestions' "declared
   accuracy"). The accuracy per answer on held-out contrast sets, on the
   generalization splits and on the inference splits that must be reached
   locally before a remote machine is paid for. *Default:* reported with
   no bar until he declares one, which he may do after the first
   measurements.

*Taken as decided unless he says otherwise* (each was a default in §9,
§11.6, §12.7 or §13.8, and a later section built on it): the cast is one
to nine, and zero is the background, where nothing is felt, with no word
in this corpus; "it" and "they" are kept; all the categories are kept,
and what might or could have happened is left out of the first set; said
mathematics is category 26 as written, in Claude's sentences, from no
book; the nine grades of the math corpus get no images; the space is 8 by
8 by 8 with an image at z = 0, and a thing is one pixel except where its
category is about size; this item's texts and returns live in this file;
the slow tier is run once, at submission.

*Not asked now, since each is a conversation before its own round*
(§12.6): what knits touching places into one thing; what holds a
succession across sentences, and the verb as it is built today (item 1.4
observes the change afterwards); two events compared; what a sentence
false of its image leaves against it (F8).

### C.6 Alec's first answers (2026-11-23)

Shown C.1 to C.5 and asked whether the defaults stand:

> Ok. You are running in arborbook at ~/GitHub or WikiOracle?
> Please build the 3D grid if not built.

And, while the grid was being read for its text:

> You will probably flush out some integration issues with multimodal, but
> hopefully they are percepts that can integrate after a separate Pi and
> not, so that concepts can tie them together (across modality).
>
> I am not certain if integrating the spatial position of words will
> confuse the spatial position of intensity, but it might. It might make
> sense to "subtitle" images by ensuring that the two domains do not
> overlap (even though we are putting them both in a common 3D space).

**What they decide.**

1. *Question 1 is answered:* this item builds the space of three
   dimensions, first. It is the round of §16.
2. *Where the session runs:* on ArborBook, in the WikiOracle checkout
   (`WikiOracle/basicmodel`, the worktree `.worktrees/item1-6`).
   `~/GitHub` there holds no checkout.
3. *The modalities meet at concepts.* Each modality's percepts pass a Pi
   and a Not of their own, and the concepts tie them together across
   modality. *Claude's reading:* the Pi and the Not are the basic level's,
   one pair for a modality; nothing joins two modalities below the Sigma of
   order zero. It agrees with 2026-11-13 ("modalities meet at concepts of
   order zero", §2.3). It bears on round one's reader and on question 2,
   and on nothing of the grid.
4. *Subtitles.* Words and intensities are in one space of three
   dimensions and take regions of it that do not overlap, as a subtitle
   lies under its picture. For the grid: each range of places is a box in
   the space and no two boxes share a place (§16). Where the box of an
   image lies beside the line of its sentence is round one's to say.

*Not decided by "Ok" as Claude reads it:* questions 2 to 8 keep their
defaults as defaults. None is needed for the grid. Question 2 is needed
before round one's text and is put to him again with it.

*Moved, the same day.* Alec: "Please do not work on iCloud Drive. Check out
1.7 in ~/github and work there." The work is now in
`~/GitHub/wikioracle/basicmodel`, cloned from GitHub at `ad8fafea92`, with
the branch `item1-6` fetched from the old checkout (nothing pushed) and its
worktree at `.worktrees/item1-6`. The Python environment is a copy of the
one the gates were measured with (Python 3.12.13, torch 2.15.0), since
`requirements.txt` pins no version.

## 16. The grid in three dimensions: the round before round one (Claude, 2026-11-23; the text for the implementer is §16.6)

Alec, 2026-11-23: "Please build the 3D grid if not built." It is not built
(C.2). What is to be built was decided in item 1.7 and specified there as
step f of the twelfth pass (sigma-pi §19.132 and §19.133, and the twelfth
pass's text, f); this section is that step, written for the source as it
is now.

### 16.1 What is decided

All Alec's, 2026-11-22, in sigma-pi §19.132 and §19.133:

- "I would like to see .where as a 3D space for all models, it simplifies
  things. I am not convinced that we need 4 elements per dim as opposed to
  3, but if they are only bf16 that's not too bad." So three axes in every
  configuration, and an axis keeps four wires until he chooses.
- "Text will be 1D: zeros for dim 2 and 3. The perception of intensity
  (feelings) can be 1,2, or 3D. Images will generally be intensity, so
  zeros for dim 3."
- "so we need an NN.param called cerebellum which is the 3D space filled
  with sinusoids that is copied (muxed) into the basic percepts, and
  demuxed to create the location for the associated vector in the percept
  codebook. The spatial parameter should be fixed". And of its name:
  "lets call it the grid."

And of 2026-11-23 (C.6): the domains of two modalities do not overlap in
the common space.

### 16.2 What exists

From a reading of every file that names the width (an Opus reader, at
`ad8fafea92`).

- *The width.* `WHERE_WIRES = WHEN_WIRES = 4` (`bin/BasicPercepts.py`); the
  band of each kind of space is `(4, 4)` in `bin/architecture.py`
  (`_CANONICAL_SHAPE`), and `(0, 0)` for the conceptual and output spaces. A
  space's declared `nDim` is its event's width and embraces the band: its
  `.what` is `nDim` less the band (`Models._obj_size`,
  `EventEncoding`). A form's event is 64 + 64 + 8 = 136 wide
  (`WordIdentity.resolve_identity_layout`). The numeric configurations
  give the input's width as `nDim - 8`: 9 for XOR's one number at each of
  two places, 36 for an image of 28 by 28. The schema has `nWhere` for a
  space and no configuration sets it.
- *A place* is one whole number, its address on one line of named ranges
  (`WhereRegistry.slices`). Its four wires are a ladder of two rungs over
  that number (`WhereEncoding`), whose periods come from the registry's
  whole capacity.
- *The grid* is `WhereRegistry.code`: one row of wires for each address,
  made at its first reader, a parameter in no optimizer. One test makes a
  registry of 512,808,192 addresses and never reads it.
- *The match* on the way back is already written axis by axis
  (`FieldReverse.where_match`: "One axis today").
- *Where the 4 is written out* and not taken from the constant: STM's field
  of a chunk's place, the checks of a place's shape in MTM and in a
  conceptual field, the band's place in an LTM point (`when_start - 4`, at
  thirteen sites), `CoordinateCheckpoint.py`, `checkpoint_migrations.py`,
  and the tests.
- *Not places, though they use the ladder:* the budget's counter
  (`bin/Language.py`) and the extent of a `.what` (`Models._what_where_encoder`).
- *Whole numbers that are keys and not wires:* a percept's `place`
  (`BasicPercepts.reserve`, `perceive_value`, `value_coordinates`), the
  ranges' bounds, `where_offset`, the brackets of a field
  (`[start, end)`), `ModelState.last_where`.

### 16.3 The design

*Claude's, for the build, except where 16.1 gives it as Alec's.*

1. *A place is three whole numbers,* (x, y, z), none below zero.
2. *Its wires are twelve:* the four wires of today's ladder for x, then
   the four for y, then the four for z. One ladder serves the three axes,
   with the periods the registry's capacity gives today, so that a step
   along any axis moves a place's wires by the same amount. `WhereEncoding`
   stays what it is, the code of one number, and is the code of an axis.
3. *An address stays.* Every place of every range keeps its whole-number
   address, and everything that uses an address as a key is unchanged. The
   registry gains the two maps between an address and its place.
4. *A range is a box.* It has a shape on the three axes and an origin. A
   range declared as today, by a count, is a line on the first axis that
   begins at x = its first address, with y = z = 0: so for every range
   that exists, x is the address, and its first four wires are the wires
   it has today. The registry refuses a layout in which two boxes share a
   place (C.6, "subtitle").
5. *Text* is such a line. *The numeric input* is a box of its own shape:
   an input of n places of one number each lies on the first axis; an
   input of h rows of w numbers is an image, with x its column and y its
   row, and its addresses stay row by row as they are. So XOR's two places
   are where they were, and an image's pixel has a place on two axes.
6. *The grid* is one parameter named `grid`, of one row of twelve wires
   for each address, made at its first reader as today, fixed, in no
   optimizer, taking a gradient. The name `code` goes.
7. *The match* is `where_match` as written, over the three axes.
8. *The width.* The number of axes is a key, `architecture.whereAxes`, 3
   in `data/model.xml` and in the schema, in no other configuration. The
   wires to an axis stay four, in the code, where they are today. The
   width of the band is their product and is computed in one place.
   **Every space's `.what` keeps its width in every configuration:** each
   declared width that embraces the band grows by 8, and nothing else of a
   configuration changes.
9. *From wires to a place:* each axis's four wires are decoded to its
   number, and the three numbers are the place; its address is the
   registry's. Nothing decodes the first axis alone and calls it the
   place.
10. *What stays on a line in this round.* The brackets of a field, which
    are addresses. The pyramid of the reconstruction and of a look's
    forecast, which is over text's positions (an image is put back in
    round one). The two ladders that are not places, which keep four
    wires.
11. *Saved models.* A model saved before this round has the old width and
    is not read by the new source. No migration is written for it.

### 16.4 What it is not

No range for a felt modality, no `scene`, no reader, nothing of round one.
No change to how a look chooses, to what a cost charges, or to the
pyramid's axes. No change to the time's wires. Nothing learned in the
grid. Not the measure of four wires against three as a change: it is
measured and reported (16.5, G9), and an axis keeps four.

### 16.5 Certificates

In `test/test_item16_grid.py`, each printing what it measured.

| | Certificate | Passes when |
|---|---|---|
| G1 | The width | `WHERE_WIRES` is four times `whereAxes` and is 12. For every configuration under `data/`, every space's `.what` width, its number of vectors and its number of places are what they were at `ad8fafea92`; the table of both is in the return |
| G2 | The first axis is today's | For every address of a model's registry, the first four wires of `grid` are the wires that address had at `ad8fafea92`, and the other eight are the code of zero on y and on z |
| G3 | There and back | For every address of every range, a box among them, the place of the address gives the address back; the twelve wires decode to the place, at the capacities the configurations use and at the greatest a test declares |
| G4 | Boxes | An image of 28 by 28 has row r, column c at (c, r, 0) and at the address it has today; two boxes that share a place are refused |
| G5 | The match on two axes | On a box, each place's wires match its own address best, and its four neighbours by a side are equally far: a step in x costs what a step in y costs |
| G6 | Text reads as before | The seven tests of `test_item17_echo.py`, `test_reading_pretraining.py` and the tests of the reconstruction that passed in the baseline pass |
| G7 | The gates | Both stopping gates pass, and their configurations differ from `ad8fafea92` in declared widths alone |
| G8 | The great registry | The registry of 512,808,192 addresses is still made without the grid being built |
| G9 | Four wires or three (a measure) | The least distance between the wires of two neighbouring places, with four wires to an axis and with three in the forms that fit (say which), at the model's capacities. Reported; nothing is built on it |

*G1 and G2 as they are held (Claude, 2026-11-24, after the return; §18.4).*
G1: every space's `.what` width is what it was, except the conceptual
space's: its whole width is a concept, a concept embraces the percept's
band, and it grew by 8 with the band. G2: holds for every address of a
range that is a line; in an image's box the addresses run row by row and
the first four wires are the column's code.

**Measured beside them:** the time of the sixteen sentences of the
standard lesson and of the seven echo tests, before and after, on this
machine; the greatest memory of a worker.

### 16.6 Text for the implementer

~~~
Item 1.6, the grid in three dimensions. Read §16 of
doc/plans/2026-11-18-item1-6-felt-math-world.md whole, then C.2, C.6 and
H.5 of the same document, then sigma-pi §19.132 and §19.133
(doc/plans/2026-11-14-sigma-pi-layers.md, from line 9950). §16.3 is what
you build and §16.4 what you do not. This changes the source of the
model in some twenty files and its tests in some ninety, so keep each
change as small as it can be: main is being changed by another
implementer at the same time and this branch has to take it in.

Keep a file, tmp/item1-6/grid_progress.md, and bring it up to date after
each step below: what is done, what is left, what you decided. If your
context runs short, stop at the end of a step with that file current.

0. Before you change anything: write the script that prints, for every
   configuration under data/, each space's declared width, its band, its
   `.what` width, its number of vectors and its number of places, and the
   registry's ranges. Run it on the frozen copy of the unchanged source
   (its path is in your prompt) and keep the output: it is G1's "before",
   and G2's wires are taken from the same copy.
1. The registry (bin/WhereRegistry.py) and the code of an axis
   (WhereEncoding in bin/Spaces.py): points 1 to 6 and 9 of §16.3. The
   parameter is `grid`; nothing is left named `code`.
2. The width: point 8. One key, `architecture.whereAxes`, in
   data/model.xml and data/model.xsd, and in no other file of data/. The
   band's width is computed in one place and every literal that is the
   band's width, or a place's width, or the band's place in a vector,
   reads it. §16.2 lists the kinds; the reader's whole inventory is in
   your prompt. A literal 4 or 8 that is not a place's width stays.
3. The configurations: each declared width that embraces the band grows
   by 8, so that step 0's script prints the same `.what` widths, vectors
   and places as before for every configuration. Change nothing else in
   them.
4. Every producer of a place's wires takes them from the grid, and every
   decoder goes through the registry: points 7 and 9. The two ladders
   that are not places keep their four wires; say where they are.
5. The numeric input as a box: point 5.
6. test/test_item16_grid.py, the certificates G1 to G9 of §16.5.
7. Port the tests that name the width or the grid. A test's assertion is
   kept and given the new width; none is weakened, skipped or marked as
   expected to fail, and no seed is chosen. A test that asserted
   something this round makes false (an address decoded row by row from
   four wires, say) asserts what is true now; list each such test.
8. Run, with the bounded runner (.venv/bin/python test/test_report.py
   <files>), RUN_TRAINED and RUN_SLOW unset: both stopping gates,
   test/test_item17_echo.py, test/test_reading_pretraining.py,
   test/test_item16_grid.py, every test file you changed, and every test
   file that names WHERE_WIRES, nWhere, WhereEncoding, WhereRegistry,
   where_registry or the grid. Compare with the baseline sweep of the
   unchanged source (its result.json is named in your prompt): no test
   that passes there may fail here. For one that does, find the cause
   and repair the source, not the test. Do not run the whole sweep;
   Claude runs it at review.
9. Touch nothing of §16.4, no file of the corpus (bin/felt_world_corpus.py,
   data/felt, test/test_felt_corpus.py) and no standing doc: Claude
   writes doc/Spaces.md, doc/Glossary.md, doc/Params.md,
   doc/FutureWork.md and the others when you return. Nothing removed is
   deleted: a file that goes is moved to a dated folder under /tmp and
   named in the return.
10. Return as §17, at the end of the same document: every file changed
    and, in a line each, why; the table of G1, before and after; each
    certificate with what it printed; every width that changed; every
    test ported, and apart from those every test whose assertion changed;
    the comparison with the baseline (tests that passed there and fail
    here, each with its cause; tests that failed there and pass here);
    the two times and the memory; the saved models and committed fixtures
    that the old width was written into, and what became of each;
    whatever in §16.3 was wrong or could not be built as written, and
    what you built in its place. You do not stop on a question of design:
    propose, build, mark it. Stop only if a gate cannot be made to pass
    with its assertion as it is.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~


## 15. The corpus as built: the implementer's return (Claude on Opus, 2026-11-23)

The text of §14.8, built. Everything below that says "verified" was run in
the worktree `~/GitHub/wikioracle/basicmodel/.worktrees/item1-6`; the rest is
what the files hold. No source of the model and no existing file under
`data/` was changed; no git command was run.

### 15.1 The files and their counts

| File | Lines | What it is |
|---|---|---|
| `bin/felt_world_corpus.py` | 2066 | the world, the predicates, the expansion of recipes, `verify`, `freeze`, `load` |
| `data/felt/recipes/cast.json` | 229 | the nine values and their words, the two kinds and their forms, the counting words, the words of yes and no, the English a rule can check, the vocabulary (87 words) |
| `data/felt/recipes/category-NN-<name>.json` | 4904 in all | one recipe for each of the 27 categories |
| `data/felt/category-NN-<name>.jsonl` | 5120 in all | one row to a line |
| `data/felt/manifest.json` | 1000 | seed, command, each file's hash and counts, each recipe's hash, the cast's hash, the vocabulary, the greatest text, each value's reserved step and places, the held-out pairs, `never_direct`, and `short` (empty) |
| `test/test_felt_corpus.py` | 459 | the eleven checks of §14.6, in twelve tests |

The command, run as §14.8 gives it:
`.venv/bin/python bin/felt_world_corpus.py freeze --seed 20261118 --recipes data/felt/recipes --out data/felt`.
A freeze takes 13 to 21 seconds of wall time on this machine. Two freezes
from the same recipes write the same 28 files byte for byte (verified, with
the second written to a scratch directory and each file compared with
`cmp`). The test runs, verified:

| Command | Result |
|---|---|
| `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/test_felt_corpus.py` | 12 passed |
| `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/test_game_corpus.py` | 7 passed |
| `BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest test/test_math_corpus.py` | 5 passed |

Rows by category and split (contrast counts rows; its 240 rows are 20 sets
of 12):

| # | File | train | validation | test | generalization | witness | inference | contrast |
|---|---|---|---|---|---|---|---|---|
| 1 | `category-01-word-alone` | 100 | 20 | 20 | 20 | | | |
| 2 | `category-02-naming` | 100 | 20 | 20 | 20 | 20 | | |
| 3 | `category-03-presence` | 100 | 20 | 20 | 20 | 20 | | |
| 4 | `category-04-kinds` | 100 | 20 | 20 | 20 | | | |
| 5 | `category-05-and-or` | 100 | 20 | 20 | 20 | 20 | | |
| 6 | `category-06-order-of-levels` | 100 | 20 | 20 | 20 | 20 | 20 | |
| 7 | `category-07-one-thing-clause` | 100 | 20 | 20 | 20 | 20 | | |
| 8 | `category-08-two-thing-clause` | 100 | 20 | 20 | 20 | 20 | | |
| 9 | `category-09-passive` | 100 | 20 | 20 | 20 | 20 | | |
| 10 | `category-10-adjective` | 100 | 20 | 20 | 20 | 20 | | |
| 11 | `category-11-comparative-size` | 100 | 20 | 20 | 20 | 20 | 20 | |
| 12 | `category-12-determiners` | 100 | 20 | 20 | | 20 | | |
| 13 | `category-13-adverb` | 100 | 20 | 20 | 20 | 20 | | |
| 14 | `category-14-place-of-two` | 100 | 20 | 20 | 20 | 20 | 20 | 240 |
| 15 | `category-15-place-among-three` | 100 | 20 | 20 | 20 | 20 | | |
| 16 | `category-16-motion-direction` | 100 | 20 | 20 | 20 | 20 | | |
| 17 | `category-17-tense` | 100 | 20 | 20 | 20 | 20 | | |
| 18 | `category-18-aspect` | 100 | 20 | 20 | 20 | 20 | | |
| 19 | `category-19-order-of-events` | 100 | 20 | 20 | 20 | 20 | 20 | |
| 20 | `category-20-events-compared` | 100 | 20 | 20 | 20 | 20 | | |
| 21 | `category-21-later-mention` | 100 | 20 | 20 | 20 | 20 | | |
| 22 | `category-22-relative-clause` | 100 | 20 | 20 | 20 | 20 | | |
| 23 | `category-23-two-clauses` | 100 | 20 | 20 | 20 | 20 | | |
| 24 | `category-24-if` | 100 | 20 | 20 | 20 | 20 | 20 | |
| 25 | `category-25-kind-follows` | 100 | 20 | 20 | 20 | 20 | 20 | |
| 26 | `category-26-said-mathematics` | 100 | 20 | 20 | 20 | | | |
| 27 | `category-27-counting` | 100 | 20 | 20 | 20 | | | |
| | in all, 5120 | 2700 | 540 | 540 | 520 | 460 | 120 | 240 |

### 15.2 Frames left out

None. Every frame of §14.4 is in its recipe with its wording as written.
Where the recipe differs in form only: frame 5's `{a} is {k} and {b} is
{k}.`, "each with its own kind", is written `{a} is {k} and {b} is {l}.`,
with `{l}` the second kind's adjective; frame 14's questions are written
out from the pattern its row gives, three for each of the nine relations
(27 question frames, `q01` to `q27`); frame 24's `{a} moved.` and `did {b}
move ?`, which §14.4 names only for the inference split, are frames of
recipe 24 marked `inference_only`; frame 27 is a frame whose text is
empty. No frame's English was judged wrong.

### 15.3 Counts short, and splits not made

No category fell short of a count (`short` in the manifest is empty). The
splits a category does not have are absent by §14.5's rules or by a
decision listed in §15.6:

| Split | Absent from | Why |
|---|---|---|
| witness | 1, 4, 26, 27 | 26 and 27 and frame 1's word have no statement of an image; 4's statements are of kinds, which no image makes false (decision 7) |
| generalization | 12 | no frame of 12 names a digit, and kinds are never held out (decision 9) |
| inference | all but 6, 11, 14, 19, 24, 25 | §14.5 |
| contrast | all but 14 | §14.5 |

### 15.4 The layout of a recipe

A recipe is one JSON object:

| Key | Holds |
|---|---|
| `category`, `name`, `construction`, `operators` | which category; the operators are §13.5's labels, copied into every row |
| `held_out` | the rule of §14.5 the category uses: `pairs`, `frame_digit`, `sums` (with `sum_pairs`), `count_sets`, `reserved` (with `reserved_question`), or `none` |
| `forms` | each placeholder of a text and the slot and form it is: `{a}` is slot `a`'s word, `{K}` slot `k`'s noun, `{Ks}` its plural, `{aK}` its form with the article, `{l}` slot `l`'s adjective |
| `slots` | each named slot's type, `digit` or `kind`; any other slot a script names (`x`, `y`, `z`, `w`, a hidden `c`) is a digit |
| `splits` | the count wanted for each split |
| `statements` | the statement frames `s01`, `s02`, ...: a frame is `text`, `predicate`, `args` and an optional `script`; a frame of several sentences told in order (20, 21) is `sentences`, a list of such, with one `script` |
| `questions` | the question frames `q01`, ...: as a statement, with `answer`: `{"type": "yes_no"}`, `{"type": "wh", "slot": s}` (the one thing for which the predicate holds, which must be slot `s`'s), or `{"type": "value"}` (the predicate gives the answer: a digit, a kind with `form`, an index into `words`, or with `roles` a choice between two slots, drawn which) |
| `inference`, `contrast` | how those splits are made (rule, frames, counts) |
| optional | `sizes` (sizes a thing may have), `world: "none"` (26), `distinct: false` (26: a digit may repeat), `compose` (`statement` for 1 and 24's train, `count` for 27) |

`args` names slots, and a literal is `{"=": value}` (a speed, or a direction
as a vector such as `[0, -1]` for up, so that no word of direction is in the
code). A `script` says what the images must hold beyond what the predicate
checks: `hidden` slots (things the frame needs and does not name),
`kinds` (a slot's digit is, or is not, of a kind slot), `size` (`big` or
`small`), `absent`, `one_of`, `only` (no further things), `layout`
(predicates that guide where things are put), `events` (each `who`, and as
needed `rel` of `toward`, `reach`, `away` or `leave` with `to`, `speed`,
`dir`, `dir_not`, `steps`, `start` or `end` as a predicate the mover starts
or ends in, `same_steps`, and `part`, the sentence of a group it serves) and
`timing` (`sequential` or `concurrent`). A question's script may be split
by its wanted answer, `{"yes": ..., "no": ...}`. A question with `bind`
takes those slots from the statement (frame 20's `a`).

The predicate names are the identifiers of §14.2 and of the conditions
under §14.4's table (`row_before`, `reached`, `kind_moved_one`, ...); with
the split names, the inference labels and the relation names of scripts
they are the only names in the code, and none of them reaches a text. Every
word of a text comes from `cast.json` or a recipe.

### 15.5 Twelve documents

Printed whole from the frozen files. A grid is the image of an entry, row
`y` = 0 at the top; a place shows the value whose band its steps lie in,
and `.` is the background; the steps themselves are not shown. One each
from categories 3, 6, 12, 14, 16, 18, 20, 22, 24, 25 and 27 (the shortest
train document among each file's first forty that asks a question, where
the category asks one), then the first four rows of the contrast set
`set-14-00`. The records are the rows' `meaning`, shortened to entry,
frame, predicate, arguments, truth and answer value.

~~~
felt-03-train-0000  variant s02+q02  inference None  answer 'nine'
   0 'four is not here.'
   1 'what is here ?'  (asks)
     0         1       
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     .....9..  .....9..
     ........  ........
     ........  ........
     ........  ........
     {"entry": 0, "frame": "s02", "predicate": "absent", "args": [4], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "present", "args": ["?"], "truth": true, "value": 9}
~~~

~~~
felt-06-train-0000  variant s01+q02  inference None  answer 'no'
   0 'two is less than four.'
   1 'is nine less than four ?'  (asks)
     0         1       
     ........  ........
     ........  ........
     ..5..4..  ..5..4..
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     .29....6  .29....6
     {"entry": 0, "frame": "s01", "predicate": "level_before", "args": [2, 4], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "level_before", "args": [9, 4], "truth": false, "value": false}
~~~

~~~
felt-12-train-0001  variant s07+q02  inference None  answer 'no'
   0 ''
   1 ''
   2 'no odd moved.'
   3 ''
   4 'did an odd move ?'  (asks)
     0         1         2         3         4       
     .......3  .......3  ..8....3  ..8....3  ..8....3
     ........  ........  ........  ........  ........
     .1...5.9  .18..5.9  .1...5.9  .1...5.9  .1...5.9
     ........  ........  ........  ........  ........
     ..8.....  ........  ........  ........  ........
     ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........
     {"entry": 2, "frame": "s07", "predicate": "kind_moved_none", "args": ["odd"], "truth": true}
     {"entry": 4, "frame": "q02", "predicate": "kind_moved_one", "args": ["odd"], "truth": false, "value": false}
~~~

~~~
felt-14-train-0000  variant s09+q17  inference None  answer 'two'
   0 'two is far from one.'
   1 'what is five under ?'  (asks)
     0         1       
     .......2  .......2
     .......5  .......5
     ..1.....  ..1.....
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     {"entry": 0, "frame": "s09", "predicate": "far", "args": [2, 1], "truth": true}
     {"entry": 1, "frame": "q17", "predicate": "on_top", "args": ["?", 5], "truth": true, "value": 2}
~~~

~~~
felt-16-train-0020  variant s01+q01  inference None  answer 'seven'
   0 ''
   1 'nine is moving toward seven.'
   2 ''
   3 'what did nine move toward ?'  (asks)
     0         1         2         3       
     ........  ........  ........  ........
     ........  ........  ........  ........
     ........  ........  ........  ........
     .......3  .......3  .......3  .......3
     ........  ........  ........  ........
     ........  ........  ........  ........
     ..7....9  ..7..9..  ..79....  ..79....
     ........  ........  ........  ........
     {"entry": 1, "frame": "s01", "predicate": "moving_toward", "args": [9, 7], "truth": true}
     {"entry": 3, "frame": "q01", "predicate": "moved_toward", "args": [9, "?"], "truth": true, "value": 7}
~~~

~~~
felt-18-train-0002  variant s01+q01  inference None  answer 'no'
   0 ''
   1 'one is moving.'
   2 'is five moving ?'  (asks)
   3 ''
     0         1         2         3       
     ........  ........  ........  ........
     ........  ........  ........  ........
     ........  ........  ........  ........
     ........  ........  ........  ........
     ........  ........  ........  ........
     ........  ........  ........  ........
     ..51....  ..5.1...  ..5..1..  ..5..1..
     ........  ........  ........  ........
     {"entry": 1, "frame": "s01", "predicate": "moving", "args": [1], "truth": true}
     {"entry": 2, "frame": "q01", "predicate": "moving", "args": [5], "truth": false, "value": false}
~~~

~~~
felt-20-train-0006  variant s02+q01  inference None  answer 'first'
   0 ''
   1 ''
   2 ''
   3 ''
   4 ''
   5 ''
   6 ''
   7 'eight moved toward two quickly.'
   8 'it moved toward seven slowly.'
   9 'which move was faster, the first or the second ?'  (asks)
     0         1         2         3         4         5         6         7       
     ......7.  ......7.  ......7.  ......7.  ......7.  ......7.  ......7.  ......7.
     ......8.  ........  ........  ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........  ........  ........  ........
     ........  ......8.  ........  ........  ........  ......8.  ......8.  ......8.
     ........  ........  ........  ........  ......8.  ........  ........  ........
     ........  ........  ......8.  ......8.  ........  ........  ........  ........
     ......2.  ......2.  ......2.  ......2.  ......2.  ......2.  ......2.  ......2.
     ........  ........  ........  ........  ........  ........  ........  ........
     8         9       
     ......7.  ......7.
     ........  ........
     ........  ........
     ......8.  ......8.
     ........  ........
     ........  ........
     ......2.  ......2.
     ........  ........
     {"entry": 7, "frame": "s02", "predicate": "moved_toward_speed", "args": [8, 2, 2], "truth": true}
     {"entry": 8, "frame": "s02", "predicate": "moved_toward_speed", "args": [8, 7, 1], "truth": true}
     {"entry": 9, "frame": "q01", "predicate": "faster_move", "args": [8], "truth": true, "value": 0}
~~~

~~~
felt-22-train-0007  variant s03+q02  inference None  answer 'three'
   0 'the even that is next to three is two.'
   1 'which odd is above six ?'  (asks)
     0         1       
     ........  ........
     ..9...32  ..9...32
     ........  ........
     ...1....  ...1....
     ........  ........
     ........  ........
     .......5  .......5
     ......6.  ......6.
     {"entry": 0, "frame": "s03", "predicate": "the_k_next_is", "args": ["even", 3, 2], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "the_k_above", "args": ["?", "odd", 6], "truth": true, "value": 3}
~~~

~~~
felt-24-train-0002  variant s01  inference None  answer None
   0 'if five is above eight, eight is below five.'
     0       
     ........
     .....5..
     .....8..
     ........
     ........
     ........
     ........
     ........
     {"entry": 0, "frame": "s01", "predicate": "if_above_below", "args": [5, 8], "truth": true}
~~~

~~~
felt-25-train-0000  variant s02+q01  inference None  answer 'no'
   0 'every odd is below four.'
   1 'is five above two ?'  (asks)
     0         1       
     .4.....2  .4.....2
     .1......  .1......
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     ........  ........
     .5......  .5......
     {"entry": 0, "frame": "s02", "predicate": "every_below", "args": ["odd", 4], "truth": true}
     {"entry": 1, "frame": "q01", "predicate": "row_before", "args": [5, 2], "truth": false, "value": false}
~~~

~~~
felt-27-train-0000  variant s01  inference None  answer 'one two three four five six'
   0 ''  (asks)
     0       
     .....6..
     ........
     ........
     ........
     .7....4.
     .....53.
     ........
     ..9.....
     {"entry": 0, "frame": "s01", "predicate": "count", "args": [], "truth": true, "value": 6}
~~~

~~~
felt-14-contrast-0000  variant q03  inference None  answer 'yes'
   0 'is nine above one ?'  (asks)
     0       
     ........
     ...9....
     ........
     ........
     ........
     ...1....
     ........
     ........
     {"entry": 0, "frame": "q03", "predicate": "row_before", "args": [9, 1], "truth": true, "value": true}
~~~

~~~
felt-14-contrast-0001  variant q03  inference None  answer 'no'
   0 'is one above nine ?'  (asks)
     0       
     ........
     ...9....
     ........
     ........
     ........
     ...1....
     ........
     ........
     {"entry": 0, "frame": "q03", "predicate": "row_before", "args": [1, 9], "truth": false, "value": false}
~~~

~~~
felt-14-contrast-0002  variant q06  inference None  answer 'yes'
   0 'is one below nine ?'  (asks)
     0       
     ........
     ...9....
     ........
     ........
     ........
     ...1....
     ........
     ........
     {"entry": 0, "frame": "q06", "predicate": "row_before", "args": [9, 1], "truth": true, "value": true}
~~~

~~~
felt-14-contrast-0003  variant q03  inference None  answer 'no'
   0 'is nine above one ?'  (asks)
     0       
     ........
     ...1....
     ........
     ........
     ........
     ...9....
     ........
     ........
     {"entry": 0, "frame": "q03", "predicate": "row_before", "args": [9, 1], "truth": false, "value": false}
~~~

### 15.6 Decided by the implementer

1. *A document.* One statement frame (or group of sentences) and one
   question frame, each with a slot filling of its own. The question's
   digits are taken from the statement's things three times in four, so
   that it speaks of what the statement speaks of, and otherwise drawn
   freely; its answer is the world's. For the questions that choose between
   two slots (6, 11, 13, 19) which of the two the images favour is drawn,
   so that the answer is not always the first named. Category 1 and the
   train split of 24 have statements only; 27 has the image only.
2. *Where a text is put.* A document without motion has one entry for each
   text, every entry the same arrangement with its steps drawn afresh. In a
   document of motion each text is put at an image where its meaning holds,
   statements in the order told, the question after them where any later
   image allows it and otherwise anywhere. 243 of 5120 rows ask before a
   statement, in categories 7, 8, 9, 12, 17, 18, 21, 22 and 23, where the
   wanted answer holds only at an earlier image (a no to "did {b} move ?"
   before {b}'s event; a yes to "is {a} moving ?"). Up to three copies of the last image are available for texts at
   the end; the longest document has 19 entries.
3. *Scripts of motion.* An event is 2 or 3 steps of 1 or 2 places in one of
   four directions; the first starts after 0 to 2 still images; events in
   sequence are 1 or 2 still images apart, or start together where a frame
   says `concurrent` (24's "if {a} moves, {b} moves"). One still image
   follows the last event. A mover is one place. The thing an event goes
   toward, reaches, leaves or goes away from is still, placed on the
   mover's line, and the gap at the end of a reaching (and at the start of
   a leaving) is one place; frame 20's two events are separated by still
   images.
4. *`meaning` records.* One for each text (and for frame 1's image without
   text and 27's image): `entry`, `frame`, `part` (in a group), `slots`
   (those the frame's text and arguments use), `predicate`, `args` (with
   `"?"` for a wh question's answer), `truth`, `cast` (the digits the frame
   names: its text's digits, or its arguments' where it has no text, with a
   wh answer), and for the question `value` (the answer before it is a
   word) and, in the inference split, `follows`.
5. *Three values.* A predicate is true, false, or undefined where its
   presupposition fails (a thing absent, a graded word between its bounds,
   two answers to a wh question, an event under way when "moved" is
   asked). No text is kept where its meaning is undefined.
6. *Readings of the conditions.* At the last image `at_rest` looks only
   back. "moved": an event of the thing complete by now and none under
   way. "is moving": displaced on both sides of now. "has moved" and "has
   stopped" are one predicate (§14.4 gives them one condition). "will
   move": still now, not displaced before, displaced after. "this is
   {a}": a is the only thing in the image. A no to "did {a} move up ?":
   {a} moved, in another direction. "did {aK} move ?": yes when exactly one
   of the kind moved and no when none did; two or more is not used, so the
   "exactly one" of §14.4 and the English "at least one" agree on every
   row. "every {K} is above {b}" (25) and "the big {K}" (10) presuppose two
   or more of the kind in the image, as 12 and 22 do; in 25 {b} is not of
   the kind. A relative clause of 22 does not count its own {b}. A wh
   question of an image is not answered by a thing its own arguments name
   (a thing is "near" itself); arithmetic is exempt (two plus two).
7. *The witness split.* One sentence of the statement group is made false;
   the events and layout that serve it are left out, half the time
   replaced by an event of no relation, and the world is drawn until that
   sentence is false and the rest true. The mark is `truth: false` in its
   record. Not made in 4 (a kind is not false of an image), nor from frame
   24's `if {a} is above {b}, {b} is below {a}.`, which no image makes
   false.
8. *Balance.* Each document's wanted answer is the yes or no (or first or
   second) given least so far in its split, and the images are drawn to
   it; no condition is loosened. Over the corpus 888 answers are yes and
   889 no. In the inference split a converse yes/no question can only
   answer no (a converse yes restates its premise and is excluded by
   check 7), so the yeses come from chains; converse and chain are drawn
   at random for each document.
9. *Held-out casts.* A frame's index is its place in the recipe's list,
   statements then questions, from 1, and a group of sentences is one
   frame. The one-thing rule applies to each frame of a document and each
   digit it names; the two-thing rule to any held-out pair among one
   frame's digits; the sums of 26 are the ordered addends `(a, b)` as
   §14.5 writes them; 12 has no generalization split. Category 4's
   generalization is the one-thing rule, which "asks of a digit in a frame
   it was not stated in".
10. *Reserved places* are avoided by every place of a big thing, by a mover
    at every image and by every still thing; a contrast set's two rows and
    its shifted rows avoid the reserved places of both values. A contrast
    set's two rows are any two of the column, not only neighbours; the
    offset is up to 3 places on each axis.
11. *Things not named.* The limit of two is read as a limit on extra
    things: at most two are added that no frame needs. The things a frame
    needs and does not name by a digit (the members of a kind in 10, 12,
    21, 22 and 25; "something else moved" in 7; the slow mover beside the
    quick one in 16) are not counted. So some documents show more than two
    unnamed things: 3 in 7, 28 in 10, 117 in 12, 19 in 16, 2 in 21, 49 in
    22, 28 in 25 (and all of 27, as §14.4 allows).
12. *Sizes.* A thing is one place except in 10 (1, 4, 5 or 6 places) and
    11 (1 to 6 places). The bounds of "big" and "small" are kept by giving
    10 no size of 2 or 3.
13. *Inference.* In 6, 11 and 14 the premises of a document are of one
    order (in 14 only above, below, to the left of and to the right of,
    the relations that chain); 6's premises are true of the numbers. 19's
    rows are chains through time. 24 has 8 rows that follow as yes, 8 as
    no and 4 `open`. 25's question asks of a digit of the kind, the kind
    never stated in the row. A premise's record has `truth: true`; the
    question's `follows` is what a closure of the premises gives.
14. *Category 27's operators* are an empty list: §13.5 has no row for it.
15. *Interfaces.* `verify(rows, recipe, cast)` takes the cast besides the
    recipe. `load(category, split)` returns `FeltDocument` records with
    `key`, `category`, `split`, `sentences`, `scenes`, `question`,
    `answer`, `variant`, `inference` and `contrast_set` (the row's `set`).
16. *Words the test needs* (which forms are singular, which take "an", the
    conjunction, the clause words and verb forms that bound a subject, the
    marks) are in `cast.json` under `english`. The greatest text is 10
    words and 48 bytes.

### 15.7 What in §14 was unclear

- *Frame 20, "which comes first is drawn, with the question and the order
  of telling fixed".* With the two frames as written, the sentences name
  the speeds, so the answer follows from the words as well as from the
  images. Built as two frames, drawn at random, with first and second
  balanced; a document that shows the two events and asks with no
  statement would measure the comparison from what was seen alone, and is
  not a frame of §14.4.
- *"Something of the other kind moves where the sentence denies"* (12) is
  applied to `no {K} moved.` only.
- *"Up to two" things not named* against the things a frame needs (15.6,
  decision 11).
- *A witness for a kind* (4): no image can make "two is odd." false of
  it.
- *`at_rest(a, i)`* at the last image, which has no image after it
  (15.6, decision 6).
- *Check 7 and the converse:* see 15.6, decision 8.
- *The sums of 26*: read as ordered addends; if "in either order" was
  meant, (7, 2), (5, 3), (2, 6) and (4, 1) are in train now.

### 15.8 Review of the first build (Claude, 2026-11-23)

*What was run again, by Claude.* The three test files in one run: 24
passed. A second freeze into a scratch folder: every file the same, byte
for byte. Every frame of the 27 recipes read against §14.4: as written. A
checker of Claude's own, sharing no code with the generator, over the 5,120
rows: every pixel in a band and in range; no reserved step and no reserved
place outside category 2's generalization split, and nothing else inside
it; a scene `null` in the inference split, in frame 1's word alone and in
category 26, and nowhere else; category 27's answers the first n counting
words; in category 14, the truth of every "above" and "below" statement and
the answer to every such yes or no question, read from the image (294
cases); each of the 20 contrast sets twelve rows on four arrangements, six
yes and six no, ten sets on held-out pairs; yes and no within 40 to 60 in a
hundred in every split. Nothing failed. The generator names no word of the
vocabulary except as a key of a recipe or as an identifier of a predicate.

*Accepted as built:* the decisions of §15.6, among them that a question may
come before a statement where its answer holds only at an earlier image
(243 rows); that things a frame needs and does not name are not counted
among the two extra; that category 4 has no witness and category 12 no
generalization; that the sums of 26 are held out as ordered addends, so
that its generalization split asks a sum taught the other way round.

*Two things to change, both faults of §14 as Claude wrote it and not of
the build.*

1. *Frame 20 says the speeds.* "eight moved toward two quickly. it moved
   toward seven slowly." answers "which move was faster" by its words.
   Suggestion 4 asked for the moves described neutrally. §14.4 and its
   condition are amended: half the documents tell the two moves with no
   word of their speed.
2. *The converse.* Check 7 compared meanings, so a question that asks a
   premise's fact back in other words was refused, and a converse could
   only be answered no; and it let through "six is greater than one."
   asked "which is greater, six or one ?", which is said and not inferred.
   §14.5 and check 7 are amended: the other way round is a premise in one
   wording of a pair, asked in the other.

Both go back to the implementer; the corpus is frozen again after them,
with the same seed.

### 15.9 The corrections of 15.8, as built (Claude on Opus, 2026-11-23)

Both corrections are built and the corpus is frozen again with the same
command and seed. Verified by running: two freezes write the same 28 files
byte for byte (about 14 seconds each); `test/test_felt_corpus.py` 12 passed,
`test/test_game_corpus.py` 7 passed, `test/test_math_corpus.py` 5 passed.

*What changed.* `bin/felt_world_corpus.py` (recipe-declared halves; the
converse by wording; even yes and no in the inference split),
`test/test_felt_corpus.py` (checks 7 and 8), the recipes of categories 6,
11, 14 and 20, and their four row files and the manifest. No other recipe
changed, and no row outside these four categories changed: each category
already draws from a stream of its own (the seed times 100 plus the
category). Against the first freeze the changed rows are: category 6, its
20 inference rows; 11, 19 of its 20 inference rows; 14, its 20 inference
rows and its 240 contrast rows (the contrast sets are drawn after the
inference split from the same category's stream, so they are drawn again;
they pass check 9 as before); 20, 179 of its 180 rows.

*Counts.* Unchanged by split:

| # | train | validation | test | generalization | witness | inference | contrast |
|---|---|---|---|---|---|---|---|
| 6 | 100 | 20 | 20 | 20 | 20 | 20 | |
| 11 | 100 | 20 | 20 | 20 | 20 | 20 | |
| 14 | 100 | 20 | 20 | 20 | 20 | 20 | 240 |
| 20 | 100 | 20 | 20 | 20 | 20 | | |

In every split of 20 half the documents use a neutral group and half a
worded one, and within each half `first` and `second` are equal (25 and 25
in train, 5 and 5 in each other split).

The inference split by kind and answer:

| # | converse yes | converse no | converse, a digit | chain yes | chain no | chain, a digit |
|---|---|---|---|---|---|---|
| 6 | 5 | 2 | 3 | 3 | 5 | 2 |
| 11 | | | 10 | | | 10 |
| 14 | 5 | 4 | 1 | 4 | 6 | |

So yes and no are 8 and 7 in 6, and 9 and 10 in 14. Category 11's converse
is "{b} is smaller than {a}." asked "which is bigger, ... ?" only; its 10
converse rows are 10 distinct documents, and there are 144 such documents to
draw from (72 ordered pairs, each asked in two orders), so it is not short.

*Decided by the implementer.*

1. *Recipe 20* gains `s03` (`{a} moved toward {b}.` then `it moved toward
   {c}.`, the step-1 move first) and `s04` (the same words, the step-2 move
   first), each sentence the unspeeded `moved_toward`; the worded groups stay
   `s01` and `s02`. The variant key names the group (`s03+q01`, `s04+q02`),
   so the two neutral groups, which have the same words, are told apart by
   it. A recipe key `halves` (`[["s01", "s02"], ["s03", "s04"]]`) gives each
   half half the documents of a split and balances the answers within each
   half; recipes without it draw exactly as before.
2. *The neutral witness.* A sentence of a neutral group can be made false
   without changing its words: "{a} moved toward {b}." is false when {a}'s
   first move does not approach {b} (its event is left out or replaced by
   one of no relation). Category 20's witness split therefore uses all four
   groups: 10 of its 20 rows are neutral (false sentence the first in 8 of
   them, the second in 2).
3. *Wordings* are declared in the recipes as the key `wording` on the
   frames that have one: `less` and `greater` (6), `bigger` and `smaller`
   (11), `above`, `below`, `to the left of`, `to the right of` (14), on the
   statement frames and on every question frame of those relations; the
   pairs are the inference spec's `converse_pairs`. A converse row draws
   its premise among the wordings whose other wording some question has (in
   11, "smaller" only), and its question among the frames of the other
   wording.
4. *Converse and chain alternate* in 6, 11 and 14 (recipe key `alternate`),
   so each is 10 of the 20 inference rows.
5. *Even yes and no.* While yes and no are unequal in an inference split,
   the next row accepted is a yes/no question with the answer that is
   behind. Allowing a lead of one, as before, left category 6 at two yes
   and one no on the first try.
6. *Check 7* now compares facts: a choice question asks the order fact its
   answer makes ("which is greater, six or one ?" asks that one is less than
   six), a wh question the fact with its answer put in, a yes/no question
   its own. Where a statement states the asked fact, the two wordings must
   be the two of a declared pair; a converse row's question asks its
   premise's fact or that fact reversed, in the other wording; a chain row's
   question asks no premise's fact. Run on the first freeze's rows, this
   check rejects 6 of them (4 in category 6 and 2 in 11, as "six is
   greater than one." asked "which is greater, six or one ?"). Check 8
   checks the balance within each half of category 20, and that the halves
   are equal.

The four documents, printed as in 15.5: a neutral document of category 20
answered `first`, one answered `second`, and a converse row with a yes from
6 and from 14.

~~~
felt-20-train-0019  variant s03+q02  inference None  answer 'first'
   0 ''
   1 ''
   2 ''
   3 ''
   4 ''
   5 ''
   6 'one moved toward four.'
   7 'it moved toward five.'
   8 'which move was slower, the first or the second ?'  (asks)
     0         1         2         3         4         5         6         7       
     ........  ........  ........  ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........  ........  ........  ........
     .......1  ........  ........  ........  ........  ........  ........  ........
     ........  .......1  ........  ........  ........  ........  ........  ........
     ..5.....  ..5.....  ..5....1  ..5....1  ..5..1..  ..51....  ..51....  ..51....
     .......4  .......4  .......4  .......4  .......4  .......4  .......4  .......4
     ........  ........  ........  ........  ........  ........  ........  ........
     8       
     ........
     ........
     ........
     ........
     ........
     ..51....
     .......4
     ........
     {"entry": 6, "frame": "s03", "predicate": "moved_toward", "args": [1, 4], "truth": true}
     {"entry": 7, "frame": "s03", "predicate": "moved_toward", "args": [1, 5], "truth": true}
     {"entry": 8, "frame": "q02", "predicate": "slower_move", "args": [1], "truth": true, "value": 0}
~~~

~~~
felt-20-train-0005  variant s03+q01  inference None  answer 'second'
   0 ''
   1 ''
   2 ''
   3 ''
   4 ''
   5 ''
   6 ''
   7 ''
   8 'six moved toward seven.'
   9 'it moved toward one.'
  10 'which move was faster, the first or the second ?'  (asks)
     0         1         2         3         4         5         6         7       
     ........  ........  ........  ........  ........  ........  ........  ........
     ......6.  ........  ........  ........  ........  ........  ........  ........
     ........  ......6.  ........  ........  ........  ........  ........  ........
     .1......  .1......  .1....6.  .1....6.  .1....6.  .1..6...  .16.....  .16.....
     ........  ........  ........  ........  ........  ........  ........  ........
     .3....7.  .3....7.  .3....7.  .3....7.  .3....7.  .3....7.  .3....7.  .3....7.
     ........  ........  ........  ........  ........  ........  ........  ........
     ........  ........  ........  ........  ........  ........  ........  ........
     8         9         10      
     ........  ........  ........
     ........  ........  ........
     ........  ........  ........
     .16.....  .16.....  .16.....
     ........  ........  ........
     .3....7.  .3....7.  .3....7.
     ........  ........  ........
     ........  ........  ........
     {"entry": 8, "frame": "s03", "predicate": "moved_toward", "args": [6, 7], "truth": true}
     {"entry": 9, "frame": "s03", "predicate": "moved_toward", "args": [6, 1], "truth": true}
     {"entry": 10, "frame": "q01", "predicate": "faster_move", "args": [6], "truth": true, "value": 1}
~~~

~~~
felt-06-inference-0000  variant s02+q02  inference converse  answer 'yes'
   0 'five is greater than three.'  [no image]
   1 'is three less than five ?'  (asks)  [no image]
     {"entry": 0, "frame": "s02", "predicate": "level_before", "args": [3, 5], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "level_before", "args": [3, 5], "truth": true, "value": true, "follows": true}
~~~

~~~
felt-14-inference-0002  variant s02+q03  inference converse  answer 'yes'
   0 'one is below four.'  [no image]
   1 'is four above one ?'  (asks)  [no image]
     {"entry": 0, "frame": "s02", "predicate": "row_before", "args": [4, 1], "truth": true}
     {"entry": 1, "frame": "q03", "predicate": "row_before", "args": [4, 1], "truth": true, "value": true, "follows": true}
~~~


### 15.10 Alec on the review (2026-11-23)

Shown four of the decisions accepted in §15.8, each answer whole.

- Of a question that comes before a statement, where its answer holds
  only at an earlier image: "This will be useful for learning tense, and
  learning to query LTM".
- Of things a frame needs and does not name, which are not counted among
  the two extra: "What limit now?" *Answered (Claude):* two things added
  that no frame needs, and no limit on those a frame needs. As frozen, of
  the 4,960 documents outside counting, 4,573 show at most two things no
  sentence names, 387 show three to seven, and the fullest image holds
  eight things. The most are in category 12, where "all the odds moved."
  needs the odds to be there.
- Of no witness for kinds, since no image makes "two is odd." false: "Two
  is odd because two is even and evens are not odd. So reasoning can
  proceed in two illuminative steps. Does this make sense? But this is a
  proof in higher order types." *Claude's reading:* that "two is odd" is
  false is not seen; it is reasoned, from two's kind and from what holds
  between the kinds. The corpus states the first step ("two is even.") and
  never the second (nothing says "no even is odd."), so today "two is not
  odd." is taught digit by digit. Put to him: a statement of the kinds
  themselves in category 4, and rows in its inference split that ask "is
  two odd ?" after "two is even.".
- Of the sums of 26 held out as ordered addends: "I don't understand what
  you are saying." *Said again (Claude):* "two plus seven" is in no
  training document, and "seven plus two" may be. As frozen, three of the
  four held-out sums are in training the other way round. Put to him:
  hold out both orders, so that a held-out sum has to be worked out.

**His answers, the same day**, each whole:

1. "Good"
2. "Ok. Seems more like general guidance than a rule. Why not needle in
   an haystack also?" *Taken:* the `haystack` split of §14.5.
3. "We ought to be able to say that evens are not odds, and odds are not
   even, in conceptual space by negation over their order relation. The
   only problem is that they occur at the same order. Thoughts? We might
   have not-odd and not-even at a higher order, and then we can have the
   two entailments we wanted from positives to the negations."
4. "Both tests are good." *Taken:* §14.5, "Both tests of a sum".

*Claude's thoughts on 3, from the standing docs and not from the source*
(`doc/Glossary.md`, "The layers" and "Negation"; `doc/Logic.md`, "And, or
and not in the layers").

- *The problem is as he says.* A kind collects "from every lower order and
  from none at or above its own". Even and odd are kinds of one order, so
  neither can hold an entry on the other, and "odd" cannot be made of "not
  even" where they stand.
- *His proposal fits the layers as they are described, with nothing new.*
  A concept of not-odd one order above them is a column with two entries:
  a Not entry on odd ("Above it, a Not entry reads a symbol's not-value")
  and an entry on even. Then two lights even, and even lights not-odd:
  two steps upward, each a positive entry. That not-odd is lit tells
  against odd by the way back through its Not entry.
- *One concept serves both sentences.* "two is not odd." and "the evens are
  not odd." name the same not-odd: a kind collects from every lower order,
  so it may hold two itself and even.
- *It gives the two entailments and no third.* Even gives not-odd and odd
  gives not-even. Not-odd does not give even, which is right: it holds here
  only because two kinds divide nine digits between them.
- *What it gives that the corpus could not test before.* Taught digit by
  digit ("two is not odd."), the denial is a list. Held at the kinds, it
  reaches a digit whose denial was never said. So the corpus now keeps the
  denial of six and of seven out of every split but `inference`, and asks
  it there after "six is even.".
- *Not checked:* whether a sentence about two kinds raises such a concept
  in the source as it is, and how a question's no is read from it. It is
  reasoning across orders, which sigma-pi §19.148 puts after this item and
  item 1.5; the rows are reported and are not a gate of this item.

### 15.11 The third revision, as built (Claude on Opus, 2026-11-23)

The revision of §15.10 and the amended §14, built and frozen again with the
same command and seed. Verified by running: two freezes write the same 28
files byte for byte (about 18 seconds each); `test/test_felt_corpus.py` 14
passed, `test/test_game_corpus.py` 7 passed, `test/test_math_corpus.py` 5
passed.

*What changed.* `bin/felt_world_corpus.py`, `test/test_felt_corpus.py`,
`cast.json` (the key `denial_held_out`), every recipe of 3 to 25 (the
`haystack` split), the recipes of 4 and 26, all 27 row files and the
manifest. Every row gains `probe`, null unless stated. Against the second
freeze, with `probe` set aside: categories 4 and 26 are drawn again in
every split; every other category keeps every row it had and gains its 20
`haystack` rows. No category but 4 had to change to keep the denial of six
and seven out: only 4's frames deny a kind (`{a} is not {k}.` and a no to
`is {a} {k} ?`), and 5, 21 and 25, which name kinds, only affirm them. The
second freeze held 11 rows, all in 4, that the new check rejects.

*The haystack split.* 20 rows in each of the 23 categories 3 to 25, 460 in
all, every image holding 6 to 9 things; none in 1, 2, 26 and 27. Frames
that bear no haystack: `what is here ?` (3) and `this is {a} or {b}.` (5),
which want one thing in the image, so they are in no haystack row. In 14,
7 of its 27 question frames are in no haystack row only because 20 rows
cannot hold 27 frames.

*Category 4's inference split*, 20 rows with no image:

| `inference` | answer | the kinds said in the row | `probe` | rows |
|---|---|---|---|---|
| `exclusion` | no | yes | `unseen` (six or seven) | 3 |
| `exclusion` | no | yes | null | 3 |
| `exclusion` | no | no | `unseen` | 3 |
| `exclusion` | no | no | null | 2 |
| `kind` | yes | | null | 9 |

*Category 26's generalization split:* 10 rows `commuted` and 10 `unseen`.

*Decided by the implementer.*

1. *Labels.* `exclusion` for 4's rows that answer no ("six is even." then
   "is six odd ?"), `kind` for its rows that answer yes (a kind's list,
   then "is four even ?"); `kind` is also 25's label, both being
   inference through a kind.
2. *9 yes and 11 no in 4's inference split* (45 in a hundred). Only 9
   distinct yes documents exist (two lists, nine members), so 10 cannot
   be made without repeating one. The 11 no rows are 6 with the kinds
   sentence and 5 without, 3 of six or seven in each.
3. *The first step of a no row* is `{a} is {k}.` or `{a} is {aK}.`: with
   the first only, the rows of six and seven with no kinds sentence are
   two documents ("six is even. is six odd ?", "seven is odd. is seven
   even ?"), fewer than the three asked.
4. *The new frames* `no {K} is {l}.` and `the {Ks} are not {l}.` (`s06`,
   `s07`) have the predicate `kinds_exclusive`, true when the two kinds
   share no member. Their index pushes the questions' frame indices from
   6 and 7 to 8 and 9, so category 4's held-out pairs of frame and digit
   for its questions changed with them.
5. *What follows in 4's rows* is computed from the row's statements and
   the cast: after a digit's kind, a no to the other kind (a digit is of
   one kind, the kinds sharing no member), with or without the kinds
   sentence said; after a kind's list, a yes for a member. Check 7 counts
   a list as stating each member's kind (the wording `list` beside `is`).
6. *The denial held out.* `cast.json` names the two digits. A row outside
   `inference` may not hold a true `{a} is not {k}.` of them, nor a no to
   `is {a} {k} ?`; witness rows, whose false statements are not said as
   true, are not counted (4 has none).
7. *Both tests of a sum.* A fact of a sum is any `sum` record (a plus
   statement or question, or its difference); its addends decide the test.
   The other order of each sum held one way is in train at least twice:
   the recipe key `required` makes "seven plus two is nine.", "five plus
   three is eight.", "two plus six is eight." and "four plus one is five."
   twice each before train's other rows. A generalization row's question
   asks the held sum and its statement holds no held fact at all.
8. *The haystack.* Every free value is put in at first; when the
   images cannot be drawn so, the attempts put in fewer, never fewer than
   six things in all. So a row holds as many as a draw found and not
   always every free value: the most that a frame bears is not proven.
   The values added stand still. A frame wanting one thing (`only` in its
   script) is left out of the split.

The six documents, printed as in 15.5: two haystack rows (14 and 22), two
no rows of category 4 on six or seven (with the kinds sentence and
without), and one `commuted` and one `unseen` row of 26.

~~~
felt-14-haystack-0000  variant s09+q11  inference None  probe None  answer 'eight'
   0 'nine is far from two.'
   1 'what is two to the right of ?'  (asks)
     0         1       
     ........  ........
     9.......  9.......
     ........  ........
     1..65...  1..65...
     ........  ........
     ........  ........
     7...3...  7...3...
     ...82..4  ...82..4
     {"entry": 0, "frame": "s09", "predicate": "far", "args": [9, 2], "truth": true}
     {"entry": 1, "frame": "q11", "predicate": "column_before", "args": ["?", 2], "truth": true, "value": 8}
~~~

~~~
felt-22-haystack-0000  variant s03+q02  inference None  probe None  answer 'eight'
   0 'the even that is next to five is eight.'
   1 'which even is above two ?'  (asks)
     0         1       
     ......9.  ......9.
     ........  ........
     ...4....  ...4....
     7.......  7.......
     85...3..  85...3..
     .......6  .......6
     ........  ........
     2......1  2......1
     {"entry": 0, "frame": "s03", "predicate": "the_k_next_is", "args": ["even", 5, 8], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "the_k_above", "args": ["?", "even", 2], "truth": true, "value": 8}
~~~

~~~
felt-04-inference-0000  variant s03+s07+q01  inference exclusion  probe unseen  answer 'no'
   0 'six is an even.'  [no image]
   1 'the evens are not odd.'  [no image]
   2 'is six odd ?'  (asks)  [no image]
     {"entry": 0, "frame": "s03", "predicate": "member", "args": [6, "even"], "truth": true}
     {"entry": 1, "frame": "s07", "predicate": "kinds_exclusive", "args": ["even", "odd"], "truth": true}
     {"entry": 2, "frame": "q01", "predicate": "member", "args": [6, "odd"], "truth": false, "value": false, "follows": false}
~~~

~~~
felt-04-inference-0006  variant s03+q01  inference exclusion  probe unseen  answer 'no'
   0 'seven is an odd.'  [no image]
   1 'is seven even ?'  (asks)  [no image]
     {"entry": 0, "frame": "s03", "predicate": "member", "args": [7, "odd"], "truth": true}
     {"entry": 1, "frame": "q01", "predicate": "member", "args": [7, "even"], "truth": false, "value": false, "follows": false}
~~~

~~~
felt-26-generalization-0000  variant s03+q02  inference None  probe commuted  answer 'one'
   0 'eight plus one is nine.'  [no image]
   1 'what is five minus four ?'  (asks)  [no image]
     {"entry": 0, "frame": "s03", "predicate": "sum", "args": [8, 1, 9], "truth": true}
     {"entry": 1, "frame": "q02", "predicate": "sum", "args": ["?", 4, 5], "truth": true, "value": 1}
~~~

~~~
felt-26-generalization-0001  variant s01+q01  inference None  probe unseen  answer 'seven'
   0 'one plus one is two.'  [no image]
   1 'what is three plus four ?'  (asks)  [no image]
     {"entry": 0, "frame": "s01", "predicate": "succ", "args": [1, 2], "truth": true}
     {"entry": 1, "frame": "q01", "predicate": "sum", "args": [3, 4, "?"], "truth": true, "value": 7}
~~~


### 15.12 Review of the third revision (Claude, 2026-11-23)

Run again by Claude: the three test files, 26 passed; a second freeze, the
same byte for byte; and, with code of its own over the 5,600 rows: outside
the inference splits nothing denies six or seven the other kind, in a
statement or in an answer; each of the four sums held out one way is in no
row outside category 26's `generalization` and is stated the other way
round in train two or three times; each of the four held out both ways,
with its two differences, is in no row outside it; every "plus one" is in
train; all 460 `haystack` rows have six to nine things in every image; and
apart from the new `probe` field no row of a category other than 4 and 26
differs from the second freeze. Accepted as built, with the implementer's
decisions of §15.11: category 4's inference split is 9 of yes and 11 of
no, since only nine such rows of yes exist.

### 16.7 Alec: non-overlapping boxes, and no dimension for the modality (2026-11-23)

Alec, while the grid was being built: "so what are your thoughts about
words with no location? can we move them out of the .where range of the
intensity modality, or do we need a separate .where dimension that indexes
modalities?" And to Claude's answer: "Ok, non-overlapping boxes, no modality
dimension."

*Decided.* The words of a sentence and the intensities of an image take
boxes of the one space that share no place, each on its own stretch of the
first axis (§16.3, point 4, as it is being built). The `.where` gets no
axis that indexes the modality: the range a place lies in says its
modality, as `doc/Spaces.md` has it, and as part rows, whole rows and
conceptual slots are told from text today.

*Claude's reasons, as put to him.* A dimension for the modality would cost
four wires on every percept and concept of every model to say what the
range says; and it would set a word at x = 3 and a pixel at x = 3 one step
apart, differing on one axis alone, which is the confusion to avoid. The
place of a word and the place of a pixel never need comparing: an
occurrence keeps its place, a generalization holds none, and the two
modalities meet at concepts (C.6, point 3). The subtitle taken literally, a
row of text under the image sharing its columns, is not taken: shared
columns would invite word order to be tied to left and right in the image.

*What it leaves to round one.*

1. A modality's place match stays inside its own range. Text's
   reconstruction keeps the `text` range of the match
   (`FieldReverse.reverse`); an image's keeps its box.
2. A look's request for a place is centred and scaled for one modality
   (`RequestRead.spatial_match` takes its centre and scale over the text
   places today), so that the next word and the thing above are two
   requests.
3. A measure: the short rung of the place code repeats, so a place of text
   and a place of an image far apart on the first axis may differ on the
   long rung alone. The least distance between the wires of any place of
   text and any place of the image is measured, at the model's capacities.
   If it is small, the image's box is given another origin, which the
   registry allows. A plane of the third axis is not spent on it: volumes
   need that axis.

## 17. The grid in three dimensions: the return (implementer, 2026-11-23)

The work of §16.6, in its steps. Steps 0 to 6 are complete; step 7 is
complete for the tests that name the width or the grid and **not complete**
for the tests built on the small fixtures (17.6); step 8 was run as 17.7
says; nothing of §16.4 was touched. The running record, with every
decision, is `tmp/item1-6/grid_progress.md`.

### 17.1 What was built, and the decisions

- *The width* is computed once, in `bin/architecture.py`: `AXIS_WIRES = 4`,
  `WHERE_AXES` read from `architecture.whereAxes` (3, in `data/model.xml`
  and `data/model.xsd` only), `WHERE_WIRES = AXIS_WIRES * WHERE_AXES = 12`,
  `WHEN_WIRES = 4`; `_CANONICAL_SHAPE` reads them, so every banded space is
  `(12, 4)`. `BasicPercepts.WHERE_WIRES` is that constant. *Decision:* the
  key is read when `architecture` is imported, from the defaults that `util`
  loads, since no other configuration may set it.
- *The code of an axis.* `WhereEncoding` keeps its code of one number
  (`encode`, `decode`, `decode_index` read the first `axis_wires`) and takes
  `axes`; `encode_place`, `decode_place` and `point` (a number on the first
  axis) are new. *Decision:* its `nDim` stays the band's width (12 for the
  registry's), with `axis_wires = nDim // axes`, so that the slot layout
  (`index`, `resolve`) of every space is unchanged code. The legacy counter
  (`forward`, `stamp`) stamps `encode_place(point(p))`.
- *The registry* (`bin/WhereRegistry.py`): a range is declared by a count
  (a line on x from x = its first address) or by a shape `(w, h[, d])`, with
  an optional origin; the default origin is `(first address, 0, 0)`. Boxes
  that share a place are refused. New: `boxes`, `place`, `address`, `decode`
  (wires to address, through the place) and `grid`, the `nn.Parameter` of
  one row of twelve wires for each address, built at its first reader, fixed,
  in no optimizer; `code` is gone. `axes` is the three slices of four wires.
- *The numeric input as a box:* `inputShape [h, w]` with `w > 1` is the box
  `(w, h)`, x the column and y the row, its addresses row by row as before;
  `[n, 1]` is a line of n. XOR's two places are where they were.
- *Producers* take their wires from the grid: the text and numeric paths of
  `InputSpace` now call `registry.encode('text', ...)`; a stamp with no
  registry (standalone spaces, `Layers.compact`, the raw mux of `SubSpace`,
  the un-fold stamp in `Models`) is `encode_place(point(x))`, the grid's own
  row for a text address. *Decoders:* `SentenceAttention.arrive` uses
  `registry.decode`; `_decode_where_offset` requires the place's other axes
  to decode to zero; `FieldReverse` matches over `registry.grid` on the three
  axes; `SentenceField` takes the nearest grid row with the LTM band read by
  `WHERE_WIRES`.
- *The two ladders that are not places* keep four wires:
  `Language.budget_encoding` (`WhereEncoding(nWhere=4)`) and
  `Models._what_where_encoder` (`WhereEncoding(extent, 4, 0)`).
- *Literals that were the band or a place's width* now read the constant:
  STM's field 11, `STMChunks.locate`, MTM's `_where`, the LTM point's where
  band (`when_start - WHERE_WIRES`, at every site that wrote or read it), the
  shape checks of `Understanding`, `ClauseRow` and a program's symbol bands,
  `PrependedSymbols`' `symbol_where`, `ThoughtChunks`' empty where, the
  grammar operator's `percept_band` (`Layers`), `Models._stamp_channel`,
  `CoordinateCheckpoint` and `checkpoint_migrations`. *Decision:* an LTM
  store has a time band only when its percept width holds the whole band
  (`>= WHERE_WIRES + 4`, was `>= 8`).
- *Decision, the conceptual width.* A concept is the concatenation of a
  percept's `.what`, `.where` and `.when` (§19.133), and every configuration
  that gives the conceptual space a width gives it the percept's event (an
  invertible conceptual space needs its input's width; a word configuration
  derives it from the form event). So the conceptual width grew by 8 with the
  rest, and the conceptual `.what` (its band is `(0, 0)`) grew by 8: this
  departs from the letter of G1 ("every space's `.what` width ... as it
  was") and agrees with §19.132 ("a concept 144 wide for 136").
- *Decision, the place's match in the attended table.* `place_factor`
  and `where_share` (`BasicPercepts`) read the `.where` axis by axis, the
  place the product of the axes' shared sizes (new `place_factor_by_axis`,
  `place_share_by_axis`; sigma-pi §19.132, "one table for an axis, a place
  the three results together"). Summed over twelve wires, the eight wires
  of y and z, which every place of text holds alike, diluted the match of a
  place's row (a wrong place read about 0.87 for 0.6), and the reading
  lesson's sampled certificate fell to 121 and then 116 of 122 under the
  bounded runner; with the product it is 122 of 122 again (run
  `20261123-213534`), and a place of text reads exactly as before.
- *Decision, the byte buffer.* In `BasicModel.xml`, `BasicModel_corpus.xml`
  and `BasicModel_answers_tied_benchmark.xml` the input's `nDim` and the part
  space's `nInputDim`, 512, are the packed byte buffer and no event: they
  stay 512 (grown, the reshape 144 x 8192 / 520 does not divide).

### 17.2 Files changed

`bin/` (24): `architecture.py` (the width), `WhereRegistry.py` (boxes, the
grid, the maps, the numeric box), `Spaces.py` (`WhereEncoding`, the producers
of InputSpace/SubSpace/PartSpace, `_decode_where_offset`, standalone axes),
`BasicPercepts.py` (the constant), `FieldReverse.py` and `SentenceField.py`
(the grid), `SentenceAttention.py` (decode through the registry),
`Language.py` (the symbol carrier's axes), `Layers.py` (`compact`'s stamp,
`percept_band`), `Models.py` (LTM where band, STM band reshape, the un-fold
stamp, `_stamp_channel`), `Memory.py` (STM field, `locate`, MTM `_where`, the
time band's guard), `EntryRead.py`, `Comprehension.py`, `ClauseRow.py`,
`Understanding.py`, `ReferenceClose.py`, `Queries.py`, `PartClosing.py`,
`SentenceStrata.py`, `PrependedSymbols.py`, `ReadingMemory.py`,
`ThoughtChunks.py` (the band's place or width), `CoordinateCheckpoint.py` and
`checkpoint_migrations.py` (the band's widths; the moved ranges are lines).

`data/` (42): `model.xml` and `model.xsd` (`whereAxes`; the defaults 9 -> 17
and their comments), and 40 configurations (the widths of 17.3; the comment
of `MM_xor_loopback.xml`).

`test/`: `test_item16_grid.py` and `fixtures/item16_widths_before.json`
(new); the ports of 17.6.

### 17.3 The widths, and G1 before and after

Every positive `nDim`, `nInputDim` and `nOutputDim` of the input, part,
modal, conceptual and whole spaces, and the output's `nInputDim`, grew by 8:
314 values in 40 configurations (each in `tmp/item1-6/config_widths.log`);
the defaults of `model.xml` 9 -> 17. Unchanged: the output's `nDim` and
`nOutputDim` (the answer has no band), the 512 of 17.1, every count.

The table of step 0 (`tmp/item1-6/grid_widths.py`) was taken on the frozen
copy (`g1_before.json`, 42 of 42 built) and on this branch (`g1_after.json`,
42 of 42 built). For every configuration and every banded space the `.what`,
the vectors and the places are as before, the band 8 -> 16, and the
registry's ranges are the same addresses; the conceptual spaces' `.what`
grew by 8 (107 spaces, 17.1). Some rows (event / band / `.what`):

| Configuration | Space | Before | After | Vectors | Places |
|---|---|---|---|---|---|
| MM_xor.xml | InputSpace | 9 / 4+4 / 1 | 17 / 12+4 / 1 | 2 | 2 |
| MM_xor.xml | PartSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 8 | 8 |
| MM_xor.xml | ConceptualSpace[0] | 14 / 0+0 / 14 | 22 / 0+0 / 22 | 8 | 8 |
| MM_xor.xml | WholeSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 64 | 8 |
| MM_xor.xml | SymbolSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 0 | 0 |
| MM_xor.xml | OutputSpace | 1 / 0+0 / 1 | 1 / 0+0 / 1 | 1 | 1 |
| MM_image.xml | InputSpace | 36 / 4+4 / 28 | 44 / 12+4 / 28 | 28 | 28 |
| MM_image.xml | PartSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 8 | 8 |
| MM_image.xml | ConceptualSpace[0] | 14 / 0+0 / 14 | 22 / 0+0 / 22 | 8 | 8 |
| MM_image.xml | WholeSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 64 | 8 |
| MM_image.xml | SymbolSpace | 14 / 4+4 / 6 | 22 / 12+4 / 6 | 0 | 0 |
| MM_image.xml | OutputSpace | 1 / 0+0 / 1 | 1 / 0+0 / 1 | 10 | 10 |
| MM_corpus_lessons.xml | InputSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 256 | 256 |
| MM_corpus_lessons.xml | PartSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 4096 | 8 |
| MM_corpus_lessons.xml | ConceptualSpace[0] | 168 / 0+0 / 168 | 176 / 0+0 / 176 | 8 | 8 |
| MM_corpus_lessons.xml | WholeSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 9 | 8 |
| MM_corpus_lessons.xml | SymbolSpace | 168 / 4+4 / 160 | 176 / 12+4 / 160 | 0 | 0 |
| MM_corpus_lessons.xml | OutputSpace | 1 / 0+0 / 1 | 1 / 0+0 / 1 | 1 | 1 |
| BasicModel.xml | InputSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 8192 | 8192 |
| BasicModel.xml | PartSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 32768 | 8 |
| BasicModel.xml | ConceptualSpace[0] | 1032 / 0+0 / 1032 | 1040 / 0+0 / 1040 | 8 | 8 |
| BasicModel.xml | ConceptualSpace[3] | 1032 / 0+0 / 1032 | 1040 / 0+0 / 1040 | 8 | 8 |
| BasicModel.xml | WholeSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 9 | 8 |
| BasicModel.xml | SymbolSpace | 1032 / 4+4 / 1024 | 1040 / 12+4 / 1024 | 0 | 0 |
| BasicModel.xml | OutputSpace | 1 / 0+0 / 1 | 1 / 0+0 / 1 | 1 | 1 |
| MM_ladder.xml | InputSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 8192 | 8192 |
| MM_ladder.xml | PartSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 4096 | 8 |
| MM_ladder.xml | ConceptualSpace[0] | 136 / 0+0 / 136 | 144 / 0+0 / 144 | 8 | 8 |
| MM_ladder.xml | ConceptualSpace[3] | 136 / 0+0 / 136 | 144 / 0+0 / 144 | 8 | 8 |
| MM_ladder.xml | WholeSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 4096 | 8 |
| MM_ladder.xml | SymbolSpace | 136 / 4+4 / 128 | 144 / 12+4 / 128 | 0 | 0 |
| MM_ladder.xml | OutputSpace | 1 / 0+0 / 1 | 1 / 0+0 / 1 | 16 | 16 |

### 17.4 The certificates (`test/test_item16_grid.py`, run `20261123-204220`, all passed)

- G1: `whereAxes=3 AXIS_WIRES=4 WHERE_WIRES=12`; 43 configurations, 314
  declared widths grew by 8, every banded `.what`, vector count and place
  count as before; built MM_xor and MM_image as in the table.
- G2: 4179 addresses of MM_xor: the first four wires equal the one-axis
  ladder exactly, the other eight the code of zero. Checked once against
  the frozen copy's own `code` too (same values; the test compares with
  `WhereEncoding(nWhere=4, nWhen=4).set_capacity(capacity)`).
- G3: every address of MM_xor (4179) and of a registry with a 28 by 28 box
  (4961) round-trips; at 512,808,192 addresses the ends of every range and
  100,000 drawn addresses round-trip.
- G4: pixel (r, c) is at the box's origin (its first address, 0, 0) plus
  (c, r, 0), at its row-by-row address; two boxes that share a place are
  refused; MM_image's numeric range is the box (28, 28, 1), MM_xor's a line.
- G5: every place of a 28 by 28 box matches itself best; a step in x and a
  step in y are equally far, 0.012649 each.
- G6: text is a line on x with y = z = 0; the echo (7 of 7), the reading
  lesson (6 of 6, after 17.1's axis-by-axis match) pass.
- G7: MM_xor.xml differs in declared widths alone; the item 2.5 gate's
  fixture (`equality_fixture`) is 24 wide for 16, its event point's place at
  `-(WHERE_WIRES + 4)`.
- G8: a registry of 512,808,192 addresses is made, and places read, with no
  grid built.
- G9 (one axis; the three-wire forms that fit are three phases of one rung):
  at 4179 addresses, four wires on two rungs: neighbours 1.228e-2, any two
  1.228e-2; three on the long rung: 4.697e-4, 4.697e-4; three on the short
  rung: 1.503e-2, any two 0 (it aliases). At 65,536: four 1.227e-2; three
  long 2.936e-5; three short 1.503e-2. Nothing is built on it.

*G2 and G1 against §16.5:* G2 holds for every range that is a line; for an
image box only x's code is the column's, not the address's (17.8).

### 17.5 Times and memory

Under load (the baseline sweep, the corpus implementer and these runs
shared the machine), the seven echo tests took 281 s here (run
`20261123-204220`) against 354 s in the baseline sweep; the reading lesson's
six took 527 s against 149 s, and side by side, one certificate at a time,
228 s here against 147 s in the frozen copy. Greatest memory of a worker:
5.57 GiB here (that run), 4.12 GiB in the baseline sweep. These are not
clean measures; the sixteen sentences of the standard lesson were not timed
apart.

### 17.6 Tests ported, and where step 7 stops

Ported (assertion kept, given the new width): `test_coordinate_registry`
(the conceptual slots' bands `(8, 12)`, decoded by `registry.decode`),
`test_occurrence_coordinates` (symbol bands 12 wide, decoded through the
registry), `test_canonical_shape` ((12, 4)), `test_basicmodel` (widths
10 -> 18 and symbolDim 9 -> 17, (12, 4), a demuxed `.where` 12 wide),
`test_clause_closing` (three-axis stamps, decoded to (3, 0, 0)),
`test_item17_time`, `test_definition_rows`, `test_item17_looks` (the grid is
the place code of every address), `test_item17_places` (a printed slice),
`test_item17_where_trace`, `test_output_path_supervised` (136 -> 144, 264 ->
272), `test_item20_round2_memory` (the migration at the new layout),
`identity_experiment.py` (232 -> 240), `test_orders_learned` (17.4, G7),
and the fixtures `test_cs_sparse_weights` and `test_cs_symbol_table`
(`_D = 16`, the band's width, where it was 8 = 4 + 4), `definition_fixtures`
and `test_clause_acceptance` (a store has a band when it holds one). `registry.code`
-> `registry.grid` in seven tests. A literal 8 that was `_D` became 16 in 47
files (`torch.eye`, `ones`, `zeros`, `LTM0(8`, `STM0(8`).

*Asserted something this round makes false, and now asserts what is true:*
`test_item18_places` (784 addresses decoded row by row from four wires; now
each pixel's place is its box offset (c, r, 0) and its wires decode to its
address); `test_where_bracket` and `test_where_encoding_period` needed no
change (`WhereEncoding` is still the code of one number).

*Not ported:* 99 tests in 26 files fail here and pass in the baseline (run
`20261123-215254`; the list is `tmp/item1-6/remaining_regressions.txt`).
All but two are fixtures that build percepts or stores of the old band by
literal widths: `LTM0(12, percept_width=8)` and its kind (`test_item18_finish`,
`test_item18_extend`, `test_item17_query`, `test_modal_timelines`' memory
brick), 8-wide literals against the 16-wide fixture (`test_bind_thought`,
`test_stm_chunk_gates`, `test_accessible_mind`, `test_identity_threading`,
`test_step2b_storage`), four-coordinate ended-field bands
(`test_clause_storage`), centred-checkpoint sidecars
(`test_unsigned_coordinates`), a symbol space's 8-wide width
(`test_partition_resolve`), a printed 144/152 (`test_wholespace_property_migration`),
MeaningKernels of width 8 (`test_item20_round6_repair`, `test_item18_operators_a`).
Each wants its fixture widened by 8 with its slices.

### 17.7 The runs of step 8, and the baseline

The baseline sweep of the frozen copy (`20261123-190752-028c4d`): 6,299 of
7,462 cases, 69 failed. Here, with the bounded runner, `RUN_TRAINED` and
`RUN_SLOW` unset:

- `20261123-213534` (after the last change to `bin/`): both gates, the
  seven echo tests, the reading lesson, `TestXorGrammarReconstruction`:
  15 passed, 1 failed (`test_piecewise_overall_at_least_50_pct`, below).
- `20261123-204220`: gates, echo, reading lesson, `test_item16_grid.py`:
  26 passed, 1 failed (the reading certificate, before the axis-by-axis
  match).
- `20261123-210415`: the named list and every changed test file, 1,764 of
  1,764 cases: 1,592 passed, 102 failed, 70 skipped (before the axis-by-axis
  match and before six files were restored).
- `20261123-215254`: the 26 files with failures, after the last change:
  303 passed, 99 failed, 1 skipped.

*Fail here, pass in the baseline:* the 99 of 17.6, and
`test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`
("0/4 in order", every pair reversed: C.3's point 3). Side by side, one at a
time: here 1 of 2 passed, the frozen copy 2 of 2. Not resolved; it may be
the word order's known fragility, or the band.
*Flaky in both:* `test_item17_symbols.py::test_cert_c...` (1 of 3 runs of
the frozen copy fail, 1 of 2 here). *Fail in both:*
`test_item17_symbols.py::test_54bd...` (the same numbers), and
`test_performance_leaves.py::...[where_decode_2]`. *Fail there, pass here:*
`test_item55_mechanisms.py::test_clock_independence`.

### 17.8 What in §16.3 was wrong or could not be built as written

- G1's "every space's `.what` as it was" cannot hold for the conceptual
  space, whose whole width is a concept that embraces the percept's band
  (17.1); it grew by 8.
- G2's "for every address" holds for lines; an image's addresses are row by
  row while x is the column, so a pixel's first four wires are its column's
  code (point 5 and point 4 together).
- A summed shared size over twelve wires is not "the match ... over the
  three axes": the attended table's place factor is a product over axes
  (17.1), which §16.3 did not say; `where_match` was already axis by axis.
- `WhereEncoding.nDim` is the band's width with `axis_wires` the code of one
  number, not a four-wide encoder beside a separate band (17.1).
- "Nothing decodes the first axis alone": the legacy `Encoding.reverse` of
  the standalone counter still decodes the first axis of a stamp it made
  on the first axis; left as it is.
- Saved models: none is committed. The committed fixtures named in the
  inventory hold no band (the five load and their tests pass, except
  `test_forward_discard`, whose identity experiment needed the conceptual
  width 240, and passes). `CoordinateCheckpoint` and `checkpoint_migrations`
  read the new widths; no migration is written for the old one (§16.3.11).
- Nothing was removed; no file was moved to `/tmp`.

### 17.9 The ports finished, and the audit of the scripted rewrite (Claude on Opus, 2026-11-23)

Second implementer. Only test files changed; `bin/`, `data/`, the gates, `test_explicit_dimensions.py`,
`test_felt_corpus.py` and `test_item16_grid.py` are untouched. The running record is
`tmp/item1-6/port_progress.md`.

*The ports.* All 98 tests of `remaining_regressions.txt` (25 files, `test_explicit_dimensions.py` left
out) pass. `_D` below is the small fixtures' concept width (8 -> 16, §17.6); "band" is
`WHERE_WIRES + WHEN_WIRES`, read from `architecture`.

- `test_partition_resolve`: the symbol and concept widths default to the band; the content stays 0.
- `test_wholespace_property_migration`: 144/136 become `128 + band + 8` and `128 + band`.
- `test_modal_timelines`: the actions are built at `f.cs.nDim`; the cs-wide values `torch.full((8,))`
  take `f.cs.nDim`; the readouts `.2` and `.15` per wire become `1.6 / width` and `1.2 / width`, the
  same spectrum as before (a readout row sums the concept's wires; at `.2` over 16 wires the two
  `post=True` cases fail, and they pass with `_D = 8` or with the scaled readout). Its timeline pool
  (an `STM0` of width 8 with no band) keeps 8.
- `test_item18_operators_a`: no change; fixed by `test_modal_timelines`' fixture.
- `test_item20_round6_repair`, `test_reading_round`: `staged.codes.shape` (1, 4, 8) -> (1, 4, `f.cs.nDim`).
- `test_item18_finish`, `test_item18_extend`: `_W = 4 + band` (12 -> 20) for the `LTM0` stores and their
  rows; the centred bands of an old entry are reshaped as groups of four over the band.
- `test_item17_query`: every width of its world is the band (`_B`): STM `concept_dim`, `_cs` dim, the
  `LTM0` width and `percept_width`.
- `test_bind_thought`: atoms and site roles at `f.cs.nDim` (also fixes `test_reference_review10`).
- `test_stm_chunk_gates`: `w = f.cs.nDim` for concepts, mentions, chunk and entry widths, deposits and
  menus; the counts (capacity 8, prepend 8, `range(1, 8)`) stay.
- `test_accessible_mind`: `_meaning`'s width and the part operands are `_D`.
- `test_identity_threading`: two cs-wide `torch.full((8,))` take `f.cs.nDim`.
- `test_clause_storage`: an ended field's `.where` is `arange(WHERE_WIRES)`, its `.when` the next four.
- `test_step2b_storage`: the store, `SymbolExpectation`'s `n_dim`/`concept_dim` and its bank are `_D`.
- `test_unsigned_coordinates`: stores `LTM0(band + 4, percept_width=band)`; saved places are four wires
  per axis; `migrate(..., band)` and its slices.
- `test_definition_taxonomy`, `test_grammatical_query_vps`, `test_query_vp_boundaries`,
  `test_query_work_boundaries`, `test_taxonomy_public_helpers`: stores and descriptions at `cs.nDim`;
  the invalid width is `cs.nDim + 1` (was 9); the description `arange(1, n + 1) / n` (was `/ 8`).
- `test_item20_round7_repair`: `4 + band` (12 -> 20) for the `_cs` dim, `percept_width` and `LTM0`.
- `test_item25_repair`: `meaning()` takes a `width` (default 8); the recall world is one band; the
  sentence fixture's `percept_width` is the band.
- `test_thought_answer_adapters`: end state, STM `concept_dim` and grammar context at the world's width.
- `test_thought_infer`: the scripted literal was restored (audit, below).

*Assertions changed.* `test_unsigned_coordinates::test_checkpoint_sidecar_conversion_keeps_empty_bands_and_is_once_only`
compared every saved STM field with one expected band; the `.where` field is now 12 wide and the
`.when` field 4, so each field is compared with its own converted band. No other assertion changed
in meaning; the rest are given the new width.

*Suspected defects in `bin/`.* None found in the 25 files.

*The audit.* The diff against the frozen copy was read for every differing test file (65, these
ports included). A literal was kept when it is the band, a place, or a width that embraces it (the
fixtures' `_D` included); four were not and are restored, each test run after and passing:
`test_accessible_mind.py:95` (operands of a `ConceptualSpaceCapability(8)`), `test_item18_extend.py:503`
and `test_item18_operators_a.py:219` (standalone meanings with no store or concept),
`test_thought_infer.py:82` (a `ShortTermMemory(concept_dim=8)`; the 16 had made it fail). One literal
the script left beside a changed partner was ported: `test_query_executors.py:140`
`full((8,), nan)` -> `full((16,), nan)`, so that the NaN case is again a NaN case and not a width
mismatch. Changed literals by the first implementer, per file, and how many restored:

| File | Changed | Restored |
|---|---|---|
| test_accessible_mind | 15 | 1 |
| test_item18_extend | 1 | 1 |
| test_item18_operators_a | 1 | 1 |
| test_thought_infer | 3 | 1 |
| test_query_executors | 8 | 0 (+1 ported) |
| test_query_phase_permissions | 8 | 0 |
| test_stm_chunk_gates | 7 | 0 |
| test_output_path_supervised | 7 | 0 |
| test_basicmodel | 6 | 0 |
| test_item17_time | 6 | 0 |
| test_part_closing | 6 | 0 |
| test_taxonomy_entry | 6 | 0 |
| test_canonical_shape | 5 | 0 |
| identity_experiment | 5 | 0 |
| test_item17_where_trace | 5 | 0 |
| test_item18_finish | 5 | 0 |
| test_item20_round2_memory | 12 | 0 |
| test_item17_query | 4 | 0 |
| test_orders_learned | 4 | 0 |
| test_query_contract_boundaries | 4 | 0 |
| test_query_work_boundaries | 4 | 0 |
| test_taxonomy_policy | 4 | 0 |
| test_clause_closing | 3 | 0 |
| test_definition_taxonomy | 3 | 0 |
| test_query_work_budget | 3 | 0 |
| test_clause_acceptance, test_identity_threading, test_occurrence_coordinates, test_query_structural_references, test_query_vp_boundaries, test_step2b_storage, test_thought_answer_adapters | 2 each | 0 |
| definition_fixtures, test_bind_thought, test_coordinate_registry, test_cs_sparse_weights, test_cs_symbol_table, test_definition_rows, test_definition_writer, test_grammatical_query_vps, test_item17_places, test_memory_consolidation, test_negative_expectation, test_reference_chain_mention, test_reference_close, test_replacement_kernel_contracts, test_retire_mode, test_thought_review, test_unified_thought_controller | 1 each | 0 |
| test_item17_demuxed_way_back, _leaf_reconstruction, _looks, _reconstruction_said_back, _table, test_item18_places | 0 (names and decoders) | 0 |

Totals: 65 files read; 4 literals restored; 1 unchanged literal ported. No restored literal had to be
put back.

*The final run.* The bounded runner over the 25 files, `test_query_executors.py`, and every test file
that imports one of them (82 files, `tmp/item1-6/port_final_files.txt`), run `20261123-224043-9a2f52`:
`Result: test_failure, exit 1; 1025/1025 cases completed.` (988 passed, 33 failed, 4 skipped).
Every test of the 26 files touched that passes in the baseline passes here. Of the 33 failures, one
fails in the baseline too (`test_percept_collection_binding.py::...[2.0-True]`). The other 32 pass in
the baseline and fail here; they are in 13 files this round did not touch and that were not in
`remaining_regressions.txt` (26 of them are in the first implementer's `new_fail3.txt`):
`test_item18_reduction` 8 (`test_reference_close.fixture`'s `LTM0(12, percept_width=8)`),
`test_item6_2_thinking` 4 (line 386), `test_case_views` 3 (the modal fixture's `(1, 4, 8)` and its
phase questions), `test_item20_round5_memory` 3 (`LTM0(12)`, `operation_fixture`'s `LTM0(owner.nDim)`),
`test_live_query_occurrences` 3 (an operand not of the conceptual width), `test_selected_relation_meaning` 2
(`LTM0(8)`), `test_unresolved_fixed_operand` 2 (`eye(8)` into the 16-wide codebook), `test_verb_learning` 2,
`test_form_subspaces` 1 (`64 + 64 + 8` against 144), `test_item18_completion` 1 (`[2, 8]` rows),
`test_item18_intuit` 1 (a `[8, 2]` field), `test_operator_catalogue` 1, and
`test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader` (`assert calls`;
deterministic; not diagnosed, and the only one whose message names no width). The list is in
`tmp/item1-6/port_progress.md`. They are not ported here.

### 17.10 The further ports (Claude on Opus, 2026-11-23)

Second implementer, on the coordinator's extension to the 32 tests of 17.9. Only test files changed;
`bin/`, `data/`, the gates and the three excluded test files are untouched. The record is
`tmp/item1-6/port_progress.md`. "band" is `WHERE_WIRES + WHEN_WIRES`, read from `architecture`.

*The ports.* 31 of the 32 pass. Two shared fixtures were ported once, in their own files, and every
user was run: `test_reference_close.fixture` (39 users) and `test_operation_measurements.operation_fixture`.

- `test_reference_close`: `fixture(percept=True)` is the band and 4 wires past it (12 -> 20), its
  `percept_width` the band; the working store (`percept=False`) keeps its 12.
- `test_item18_reduction`: `C` = band, the first wire past the percept band; the content atoms 8..11
  are C..C+3, the exclusion pair (8, 9) is (C, C+1); `SentenceStrata` at the store's width with
  `percept_width=C`; the legacy centred bands span C in groups of four; `test_63`'s store is C + 4.
- `test_operation_measurements`: the fixture's `LTM0` holds at least the band (`max(owner.nDim, band)`).
- `test_item20_round5_memory`: a link's `.where` is `zeros(WHERE_WIRES)`; the 12-wide idea is kept in an
  `LTM0` of `max(12, band)` and read back at its own width.
- `test_verb_learning`: the sentence fixture's actions and values at `f.cs.nDim`; the standalone test keeps 8.
- `test_case_views`: staged codes at `f.cs.nDim`; phase contents are two lanes of `nDim/2` (the first two
  wires of each, the first wire of each); case values at wire 1 of each lane.
- `test_item6_2_thinking`, `test_live_query_occurrences`, `test_unresolved_fixed_operand`,
  `test_operator_catalogue`, `test_item18_intuit`, `test_selected_relation_meaning`: concept-wide
  values, stores and banks at the world's `cs.nDim`.
- `test_form_subspaces`: declared widths 9 and 137 -> `1 + band` and `129 + band`; the event `64 + 64 + band`.
- `test_item18_completion` (stimulation): `_cs(dim=8)`. Its rows are a percept event of 4 and two lanes
  of 2, with no band, so the concept keeps its own width.

Further regressions found in the users of the touched files, and ported: `test_selected_nested_meaning`
(5; stores at the world's width), `test_identity_rows` (2; `LTM0(band + 4, percept_width=band)`),
`test_item20_round1` (1; a place of four wires an axis, the empty `.where` `[-1] * WHERE_WIRES`),
`test_item18_tree_first::test_1_word_deposit_keeps_its_value_row_and_order` (the place's slice
`start - WHERE_WIRES`).

*Assertions changed.* `test_item20_round1::test_thought_deposits_advance_when_with_empty_where_on_one_model_clock`
asserts an empty `.where` of `WHERE_WIRES` wires (was four). No other assertion changed in meaning.

*`test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`.* `calls` counts
the calls of `owner.basic_percepts.read_collection` during `model.understand(['alpha'])`. When the test
passes, the call is reached from `Models._run_sentence_word_bricks` -> `score` -> `WholePaths.finish`
(`bin/WholePaths.py:306`) -> `WholePaths.output` (`:153`) -> `reverseOutput` -> `generate` ->
`OutputOccurrences.generated` (`bin/OutputOccurrences.py:99`) -> `FieldProjection.stimulate`
(`bin/FieldProjection.py:111`) -> `PerceptCollections.read` -> `read_collection`. `WholePaths.output`
returns at once when `model.inputSpace.data.has_supervised_outputs` is false (`bin/WholePaths.py:137`,
gated at `:267`). That data object is the process-wide `data.TheData`. The fixture `named_scalar_model`
does not set the flag; it is true only when an earlier test in the same process left it true. Run alone,
the test fails on the frozen copy of the unchanged source (scratch copy, plain pytest and the bounded
runner) and on this source, with `calls` empty in both. Run after
`test_item18_paths.py::test_numeric_understanding_owns_evidence_across_next_forward` (an MM_xor model,
supervised), as it was in the baseline sweep's worker 29, it passes in both (6 calls). So it is an
order dependence present before this change, not a regression of it, and not a width. It is not
changed here: making it independent means declaring supervised outputs in its fixture, a choice of
what the test covers, left to the reviewer. No defect in `bin/` is claimed; the shared, unreset
`has_supervised_outputs` is the mechanism.

*Not ported.* `test_item18_tree_first::test_2_first_document_matches_recorded_object_readings` replays a
committed recorded fixture (`test/fixtures/item18-comprehension-first-document.pt.gz`, hash-checked,
recorded with four-wire symbol bands) and fails on `symbol occurrence bands must align to program
leaves`. Porting it means re-recording or converting that fixture (§16.3.11: no migration); left for a
decision.

*Suspected defects in `bin/`.* None.

*The final run.* The bounded runner over the 13 files, the two fixture files, and every test file that
imports one of them (71 files, `tmp/item1-6/port_final_files2.txt`), run `20261123-231424-968fdf`:
`Result: test_failure, exit 1; 975/975 cases completed.` (962 passed, 2 failed, 11 skipped). Both
failures pass in the baseline: `test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`
(order dependence, above) and `test_item18_tree_first::test_2_first_document_matches_recorded_object_readings`
(committed fixture, above). Every other test of these files that passes in the baseline passes here.

### 17.11 The recorded first document, converted (Claude on Opus, 2026-11-23)

The recording `test/fixtures/item18-comprehension-first-document.pt.gz` (item 1.8, `source_snapshot`
`7768411a...`) was converted, not re-recorded, and replayed by
`test_item18_tree_first.py::test_2_first_document_matches_recorded_object_readings` with its
assertions and tolerances unchanged. The test was not changed: nothing in it reads a band's width.

*Where the bands were.* The recorded concept is 392 wires: form 352, `.where` [352, 356), `.when`
[356, 360), meaning [360, 392). Four-wire `.where` bands: `program.symbol_where` (4 x 4) and
`program.held_initial.where` (8 x 4). Vectors that embrace the band (392 wide): `leaves`,
`end_state`, `operation_values`, `identity_values`, `timeline_values`, `case_values`, `case_bank[1]`,
`held_initial.values`, every `bank.lookup_rows` read, and the expected `frames`, `roles` and `point`;
`action_records` is the 392 and three flags (395). Layout integers: `percept_width` 360, `when_start`
356, `percept_band` (352, 360, 392), the strata's `width` 392 and `percept_width` 360, and in each
pickled grammar operator `nInput`/`nOutput` 392, `percept_band` (352, 360, 392) and `meaning_layout`
(360, 392). Not band: `form_wholes` and `form_whole_presence` (352, form only), the conceptual
field's bracket addresses, the 16-wide lane kernels, the `.when` bands, every id and action.

*The capacity.* The recorded bands are in the snapshot's centred convention (each pair a unit
vector, sin and cos raw), so they were decoded through `unsigned_phase`. Over every power-of-two
long period from 2^8 to 2^31 (`tmp/item1-6/rec_capacity.py`), only the periods (2^15, 256), a
capacity in [16255, 32254], re-encode every recorded symbol and held band to itself within one
float32 ulp (5.96e-8) and decode them to the addresses 0, 5 and 8, the recording's word starts;
longer periods are off by 2e-7 or more and decode to meaningless addresses. Every band embedded in
a recorded vector re-encodes at these periods too, except the expected role 1 (below).

*The script.* `tmp/item1-6/convert_first_document.py` (not committed) keeps each band's four
recorded wires as x, and appends the code of zero on y and on z: `2 * encode(0) - 1` at capacity
32254, in the recording's centred convention (it equals the recording's own band for address 0);
an all-zero band widens with zeros (149 bands coded, 541 empty). Each vector that embraces the
band gains eight wires at [356, 364) (606 vectors, 49 action records). The layout integers move by
8: 368, 364, (352, 368, 400), 400 and 368; each operator's 392 -> 400, (352, 368, 400), (368, 400)
(13 operators). Nothing else changed. The converted file is
`tmp/item1-6/item18-comprehension-first-document.converted.pt.gz` (102,765 bytes, sha256
`9669b7ab...c90a3`), its manifest beside it with the `conversion` key.

*The result.* The test does not pass. Every action, object and native id, every operator frame
(operands and result) of every action of all seven paths, roles 0 and 2, the point, the relation and
every extension node match exactly (greatest difference 0). The first differing tensor is
`clause.meaning.roles` at sentence 1, path 0 (the clause closing after action 6, `part`), row 1,
the predicate's role: at index (1, 0) the recording has 0.070700 and the replay 0.070136; 399 of its
400 wires differ (387 outside the band's place, 12 within), greatest difference 0.989 at (1, 359).
The same row differs in the six relative clauses; the seventh path (no relation) passes whole. Row 1
is not read from the recording: it is `ClauseRow.predicate_point('part', ...)`
(`bin/ClauseRow.py:35`), `References.symbol_code` of the predicate's identity at the concept's width
with no band. At 400 wires it is a different code: its first 352 wires are the old ones times
0.99201 (the same code spread over eight more wires), and the wires at [352, 364) are code values,
not a place. So the difference is in a width-dependent identity code, not in comprehension's
numbers and not in a place's wires; the conversion's rule (a place's code of zero) does not apply
to it, and nothing was adjusted to make it pass. The original recording and manifest are back in
`test/fixtures/` unchanged (byte-identical to the frozen copy); the test still fails as in 17.10.

*What was done in the end* (the reviewer's decision: the predicate's identity code is a constant the
source generates for the concept's width, not a recorded result, so the conversion regenerates it).
The script first checks, before changing anything, every expected `roles` row of every record against
`symbol_code(predicate_identity(name), 392, n_where=0, n_when=0)` for `part` and `implies`. Six rows
match, row 1 of the six clauses whose relation is `part`, each with greatest difference 0.0 (exact),
and the name always agrees with the clause's relation; no clause with a relation lacks one. A scan of
every 392-wide float tensor of every record finds that constant nowhere else. Only those six rows were
replaced, by the same call at 400. The converted recording (102,964 bytes, sha256
`85fc988b69a739fa6e3b15d3c0057c32c74fe78218f3a05673b0b9d6fa715d1c`) and its manifest (new `sha256` and
`bytes`, a `conversion` key, `source_snapshot` unchanged) are installed in `test/fixtures/`; the test is
unchanged. `test/test_item18_tree_first.py` whole, bounded runner, run `20261123-233212-dfd78b`:
`Result: passed, exit 0; 15/15 cases completed.` So `test_2` passes with its tolerances as they were:
every action, object, operator frame, role, relation and extension node that item 1.8 recorded is
reproduced by today's comprehension with the grid in three dimensions, which is strong evidence that
the grid left comprehension's numbers alone. The originals are kept as
`tmp/item1-6/item18-comprehension-first-document.original.{pt.gz,json}`.

### 17.12 The third round of ports (Claude on Opus, 2026-11-24)

Third implementer, on the 48 tests of `tmp/item1-6/remaining_after_sweep.txt`. Only test files changed;
`bin/`, `data/`, the four excluded test files and the gate's function, helpers and fixtures are
untouched. The record is `tmp/item1-6/port3_progress.md`. "band" is `WHERE_WIRES + WHEN_WIRES` (16),
read from `architecture`.

*The ports* (file, tests of the list, what it needed):

- `test_forgetting` (14): `BAND` and `WIDTH = BAND + 4`; `LTM0(12, percept_width=8)` -> `LTM0(WIDTH, percept_width=BAND)`
  at its six sites; the hook test's content slice `slots[:8, 8:]` -> `[:8, BAND:]` (the gradient stays 8); the taxonomy
  test's `LTM0(8)`, `ones(3, 8)`, `ones(8)` -> `cs.nDim` (`_operator`'s space, 16).
- `test_episodic_concepts` (3), `test_item20_round7_episodes` (1), `test_item18_memory_trust` (1): stores of four wires
  of what and the band (12 -> 20), their rows at that width.
- `test_item18_parts_round_c` (1): `LTM0(owner.nWhat + 4)` and its padding of 4 -> `owner.nWhat + WHERE_WIRES` (+8).
- `test_operators_round4a0` (4): `LTM0(12, n, percept_width=8)` -> `LTM0(WIDTH, n, percept_width=BAND)` (seven sites),
  the fixture meaning `full((12,))` -> `(WIDTH,)`.
- `test_item18_round_a` (2): stores of `band + 4`; the exclusion pair at the first two wires past the band, `C = band`:
  points `[0, 8]`, `[1, 9]` -> `[0, C]`, `[1, C + 1]`, `coordinate_exclusions(8)` -> `(C)`, `((8, 9),)` -> `((C, C + 1),)`.
- `test_stm_chunk_repairs` (1): percept `4 + band`, its `.when` `[percept - WHEN_WIRES, percept)`, width `percept + 4`
  (16 -> 24); the `.when` slices `[8:12]` take that slice.
- `test_taxonomy_unknown` (2): `LTM0(8)`, `eye(8)` -> the band.
- `test_stm_chunks` (3): `PERCEPT = 4 + band` (12 -> 20); `percept_band` `(4, 12, 12)` -> `(4, PERCEPT, PERCEPT)`; the
  four-wire places are the same four wires on each axis (`repeat(WHERE_WIRES // AXIS_WIRES)`), for STM and MTM.
- `test_item4_native` (1): `SymbolExpectation(8, 8, 8, concept_dim=8)` and the host STM's `concept_dim` -> the
  fixture's `cs.nDim`. Its two `[omitted]` cases (failures in the baseline) now fail as there (line 53, one element).
- `test_item25_completion` (3): `_optional_reading`'s `symbol_where` `arange(4)` -> `arange(WHERE_WIRES)`. The gate
  does not use this helper.
- `test_thought_operation_catalog` (4): concept-wide values at `_D` (16, imported beside `_cs`); the word stream
  `ones(1, 8)` stays.
- `test_packed_reconstruction_parity` (shared `measure_layout`; the 3 of `test_one_backward_standing`): the root's
  stamp `[when_start - 4, + 8)` -> `[when_start - WHERE_WIRES, + band)`.
- `test_operators_round3a` (1), `test_config_scoping` (1), `test_dimensional_governance` (2),
  `test_word_loop_buckets_and_orders` (1): declared widths and geometry, below. `test_config_scoping`'s inline
  configuration declares `nDim` 9 -> `1 + band` (17) for the input, part, conceptual and whole spaces (the conceptual
  space's embraces the form's event, and its `LTM0` needs the band); the output space keeps 1.
  `test_dimensional_governance` replaces the configuration's text `1032` (was `1024`, as the data file now declares).

*Assertions whose value changed* (each a width that embraces the band, +8):

- `test_operators_round3a::test_numeric_mm_keeps_its_original_geometry`: `muxedSize == 64 + 64 + 4 + 4` ->
  `64 + 64 + WHERE_WIRES + WHEN_WIRES` (136 -> 144).
- `test_dimensional_governance::test_cs_ws_recurrent_input_mismatch_raises`: the message names `144` (was `136`).
- `test_dimensional_governance::test_cs_os_direct_handoff_mismatch_raises`: `8x144` and `7x144` (were `8x136`, `7x136`),
  as `128 + band`.
- `test_word_loop_buckets_and_orders::test_basicmodel_declares_one_dynamic_capacity_and_independent_inventories`:
  `BasicModel.xml`'s part and whole `nDim`/`nOutputDim` 104 -> `96 + band` (112); the conceptual
  `nInputDim`/`nDim`/`nOutputDim` 1000 -> `992 + band` (1008).
- `test_thought_operation_catalog::test_model_boundary_builds_the_common_thought_context_without_a_reasoner`:
  `conceptual_space.width == 8` -> `== _D` (16).

No other assertion changed; none was weakened, skipped or marked; no seed chosen.

*Suspected defects in `bin/`.* None. Every failure was a fixture's or an expectation's width.

*The gate.* `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`: passed
before any change (pytest, alone) and after (the file whole under pytest, 24 passed, and in the final run).

*The final run.* The bounded runner over the 18 files touched, every test file that imports one of them, and the two
test files that import `attention_thinking_certificate` (which imports the gate's file): 34 files,
`tmp/item1-6/port3_final_files.txt`, run `20261124-003258-e6f48d`:
`Result: test_failure, exit 1; 357/357 cases completed.` (345 passed, 4 failed, 7 skipped, 1 xpassed). All 48 of
the list pass. The four failures fail in the baseline too: `test_hash_order_producers::test_collision_queue_follows_admission_order`,
`test_item4_native::test_observed_cat_survives_native_answer_and_store[False]` and `[True]`, and
`test_operators_round4a0::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers`; the xpass
(`test_forgetting::test_unforced_grammar_turnover_reports_its_retained_occurrences`) is an xpass in the baseline.
No test that passes in the baseline fails here.

## 18. The grid in three dimensions: the review (Claude, 2026-11-24)

*Review of §17 as it stands at `20a31b10ea` (source, configurations and
tests) and `35e704840f` (the standing documents). Every number here was
measured on this machine with `RUN_TRAINED` and `RUN_SLOW` unset.*

**The grid is accepted.** No test that passes on the unchanged source fails
on the grid for a reason that is the grid's. Two things are left open and
are named in 18.2 and 18.6.

### 18.1 The sweep against the baseline

Both sweeps ran from a frozen plain copy of their source, so that nothing
could change under them: the unchanged source at `ad8fafea92` (run
`20261123-190752-028c4d`) and the grid at `20a31b10ea` (run
`20261124-003722-d69da8`).

| | Unchanged | Grid |
|---|---|---|
| Cases | 7,462 | 7,488 |
| Passed | 7,053 | 7,080 |
| Skipped | 338 | 338 |
| Failed | 69 | 68 |
| xpassed, xfailed | 1, 1 | 1, 1 |
| Seconds | 6,173 (the machine was shared) | 4,887 (alone) |
| Greatest memory at once | 9.54 GiB | 9.68 GiB |

The 26 cases added are the corpus's 14 (`test_felt_corpus.py`) and the
grid's 12 (`test_item16_grid.py`); all pass. No case of the baseline is
absent.

*Fail in both, 66.* Fifty need the repository's history, which a plain copy
has not (`test_attention_host_work.py`, 44; `test_item17_layers.py`, 6).
Three are `test_doc_links.py`'s (`doc/Params.md`, `doc/Philosophy.md`, the
sigma-pi plan), which do not resolve from a copy or a linked worktree.
Thirteen fail on the unchanged source for reasons of its own:
`test_compose_observer::test_observer_counts_detached_compose_calls`,
`test_concept_memberships::test_definition_boundary_preserves_gradient_owned_codes`,
`test_hash_order_producers::test_collision_queue_follows_admission_order`,
`test_item17_fragments::test_50d_...`, `test_item17_sparse_growth::test_58_...`,
`test_item17_symbols::test_54bd_...`, `test_item17_word_said_back::test_the_answer_depends_on_the_spoken_request`,
`test_item18_expect::test_81_...`, `test_item4_native::test_observed_cat_survives_native_answer_and_store`
(both cases), `test_operators_round4a0::test_when_reader_inventory_covers_current_attribute_and_dynamic_readers`,
`test_percept_collection_binding::test_bind_percept_collection_reads_membership_not_coordinate_overlap[2.0-True]`,
`test_performance_leaves::...[where_decode_2]`.

*Fail on the unchanged source, pass on the grid, 3.*
`test_item55_mechanisms::test_clock_independence` and
`test_reasoning::TestWriteMask::test_partition_isolation` pass on the grid
in both of its sweeps. The third,
`test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`,
is not mended: run alone it fails on both sources, twice of twice on each
(the four rows come back in another order), and it passed in the grid's
sweep by what ran before it in its worker. It is a fault of the unchanged
source that depends on the order of the tests.

*Pass on the unchanged source, fail on the grid, 2.* Neither is the
grid's, by what follows.

- `test_item17_stacks::test_seconds_against_an_mlp_layer[mps]`: the
  graphics processor reported a hang
  (`kIOGPUCommandBufferCallbackErrorHang`) while the test timed a layer. Run
  again, the file alone, twice on each source: it failed once of twice on
  the unchanged source and once of twice on the grid, with the same
  message. It is the machine's.
- `test_explicit_dimensions::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct`:
  18.2.

### 18.2 The order of two words after grammar training

The test trains verbal XOR from a fresh start and asks that the two words
of each of four sentences come back in the order they were read. It is
C.3's third point and item 1.7's open one. Counted over every fresh start
on this machine:

| | Passed | Failed, and how |
|---|---|---|
| Unchanged source | 6 of 8 | 2 of 4 in order; 3 of 4 in order |
| Grid | 5 of 9 | 0 of 4 in order, four times |

(Unchanged: the baseline sweep, five alternating repeats, two side by
side. Grid: two sweeps, five alternating repeats, two side by side.)

Nine runs and eight cannot tell five of nine from six of eight, so there is
no evidence that the grid loses the order more often. One thing is seen and
not explained: each of the grid's four failures is every pair reversed,
and the unchanged source's two failures here are partial. Every pair
reversed is also what Codex's landing sweep of item 1.7 recorded on the
unchanged source, so the grid did not bring it. Five more alternating
repeats on the two frozen copies were started after this was written; their
count is added below when they return.

It bears on question 5 of C.5 and on nothing of the grid's design: the
first milestone turns on the order of two words, and a model that keeps the
order in a little over half of its fresh starts cannot be measured on it
with one run.

### 18.3 Times and memory

One source after the other on the machine alone, the bounded runner, the
file whole:

| | Unchanged | Grid |
|---|---|---|
| The reading lesson, six tests (`test_reading_pretraining.py`) | 125 s | 129 s |
| of which the certificate (122 sampled readings) | 114.3 s | 117.0 s |
| Greatest memory of its worker | 3.54 GiB | 3.56 GiB |
| The seven echo tests (`test_item17_echo.py`) | 243 s | 225 s |
| Greatest memory of its worker | 0.91 GiB | 0.95 GiB |

Both files passed whole on both sources. The 228 s against 147 s of 17.5
was the shared machine and not the grid: the band's eight more wires cost
the reading lesson about three seconds in two minutes. The greatest memory
of one case in the sweeps is the same test on both
(`test_item20_round2_memory::test_one_whole_inventory_on_every_configuration[BasicModel.xml]`,
6.17 GiB and 6.20 GiB). The sixteen sentences of the standard lesson were
not timed apart from their tests, which §16.5 asked; the certificate above
trains on them and is the nearest measure.

### 18.4 Where the build departs from §16.3, and what is accepted

- *The conceptual space grew by 8.* G1 asked that every space's `.what`
  stay as it was. A concept embraces the percept's band, and the conceptual
  space's whole width is a concept, so it could not. Accepted; G1 is
  reworded in §16.5. It is "a concept 144 wide for 136".
- *The place match is a product over the axes.* §16.3 said "the match over
  the three axes" and did not say how. Summed over twelve wires, the two
  axes a line of text does not use diluted the match, and the reading
  certificate fell to 116 of 122. Axis by axis and multiplied
  (`BasicPercepts.place_factor_by_axis`), an axis on which two places agree
  gives one and the certificate is 122 of 122. Accepted, and it is the
  better reading of "one space": two places are near when they are near on
  every axis.
- *G2 holds for lines.* In a box the addresses run row by row and x is the
  column, so a pixel's first four wires are its column's code. Accepted; G2
  is reworded in §16.5.
- *No saved model of the older width loads.* None is committed, and no
  migration was written (§16.3, point 11). A model trained before the grid
  is trained again.
- *The recorded first document was converted, not recorded again* (17.11).
  Accepted on its proof: only rows equal to the old constant, with a
  greatest difference of 0.0, were made again, and the test's assertions
  are unchanged.
- *The scripted rewrite of 8 to 16 in 47 test files* was audited by the
  second implementer; four literals it had wrongly changed were restored
  and one it had missed was corrected (17.9).

No assertion was weakened, skipped or marked in any round; no seed was
chosen; both stopping gates are as they were, and their configurations
differ in declared widths alone.

### 18.5 Found on the way, and not this item's

For Codex and Alec, each a fault of the unchanged source:

1. `test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`
   passes or fails by the order of the tests: an earlier test leaves
   `TheData.has_supervised_outputs` set on the process-wide data (17.10).
2. `test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`
   fails alone and can pass after other tests (18.1).
3. `test_item17_stacks::test_seconds_against_an_mlp_layer[mps]` fails about
   one run in two on this machine with a hang of the graphics processor
   (18.1).
4. The three of `test_doc_links.py` fail from a linked worktree of the
   submodule and from a plain copy.
5. The order of two words, 18.2.

### 18.6 What the grid leaves to round one

§16.7's three, unchanged: a modality's place match stays inside its own
range; a look's request for a place is centred and scaled for one modality;
and the least distance between the wires of any place of text and any place
of an image is measured at the model's capacities, the image's box being
given another origin if it is small. Where the box of an image lies beside
the line of its sentence is round one's to say (C.6, point 4). The choice of
four wires to an axis or three is in `doc/FutureWork.md` with G9's measure.

`origin/main` stands at `ad8fafea92`, where this branch was cut; there is
nothing of item 1.5 to take in yet.

## 19. Alec on the percept of a pixel, the object and the name (2026-11-24)

Asked question 2 of C.5 (what the percept of a pixel is) and question 3
(the object before its word), after the review of the grid:

> The object should be made at the same time as the word, at this point,
> because we don't treat the word as a representation of something else: it
> is, prima facia, also an object. But in virtue of a test which we should
> call "naming" (or perhaps the name of the corresponding Piagetian stage),
> they come to be associated. We need to replace the logic that creates the
> meta concepts directly, and that learns the association of the meta as a
> PI that includes both concepts in virtue of them being presented at the
> same time.
>
> Meta concepts should be introduced by a model variable
> (createMetaConcepts) until we make sure the architecture supports it and
> we develop a test that works reliably to introduce names for things.
>
> The percept of a pixel is just a value, 0..1 like other percepts. Percepts
> grow in size on the two percept pyramids to include surrounding areas, and
> their part structure is the same as their surface structure: they are
> values sorted by < (improper parthood).

### 19.1 The percept of a pixel (question 2, answered)

**Decided.** The percept of a pixel is its value, from 0 to 1, and nothing
of its place. Its place is on the occurrence, in the twelve wires. The same
value at two places is one percept met twice.

**Against the source.** `doc/Spaces.md` ("A level at a place") and
`BasicPercepts` have it otherwise today: one basic percept for each pair of
place and level, and for each place a table of values on coordinates of its
own (`value_coordinates(width, place)` is a keyed permutation of the bins;
`perceive_value(value, place=...)` reserves the row by its place). The
reason given there is that XOR's two inputs stay distinct. With his answer
they are told apart by their wires and by the reading each has at its own
place in the field, and the table by place goes.

*Values sorted by <.* This is `value_code` as built, with the bins in
order: the code of a value is the bins at or below it, a lower value's code
lies inside a higher one's, and `partOf` on two of them is less-than or
equal, which is his improper parthood. "Their part structure is the same as
their surface structure" is what `doc/Mereology.md` says of a scalar: the
growth of the parts and the contraction of the wholes move one cut, so the
two halves of the code are one.

*What 0..1 is, for a pixel.* Claude takes it as the intensity over its
greatest, the step over 255. The numeric stem's `feeling` is a chart of the
numbers from zero to infinity onto 0..1 and is not needed for a number that
already lies there. With 64 bins a bin is four steps: the variation of one
value of the corpus (seven steps) falls in two or three neighbouring bins,
and two values (25 steps apart) are six bins apart.

*The two percept pyramids.* They are the part tower and the whole tower
(`PartSpace`, bottom-up; `WholeSpace`, top-down), with basic percepts on
them as for text; a percept higher up covers a greater area than the
percepts under it, as a form covers its characters. He confirmed it the
same day, while this section was being written:

> wholes are similar, on their tower they apply uniformly to a range of
> positions (< 0.5 is a property of the entire object, it is a pervasion in
> the buddhist sense).
>
> So we have two towers and basic percepts there, as elsewhere.

*A whole pervades its range.* On the whole tower a whole is a property, a
cut of the sorted line of values such as "below 0.5", and it is the whole
of a range of positions when it holds at every one of them. It agrees with
`doc/Spaces.md` ("properties that divide the input field into regions") and
with a value's wholes half, which is the bound the value lies under. The
tightest whole that pervades an area is then the cut at its greatest value,
which is what `BasicPercepts.collection_code` calls the covering whole.
*Not yet said:* the rule on the part tower. Two are in the source: the
greatest of the parts' codes, by which a form is built of its characters
(`doc/Mereology.md`), and the common part, the least, by which a collection
is read. Claude proposes one in the design of the round that builds areas,
and he chooses there. Round one does not rest on it, since it builds the
pixel alone.

**One reader.** There is to be no second path (H.5), so the answer holds
for every number the model feels: an image's pixels, the numeric stem's
places (`NumericPercepts`) and the value beside the words
(`ScalarPercepts`). Round one therefore changes how the numeric XOR proofs
and `data/MM_image.xml` are read, and measures them before and after.

### 19.2 The object and its word (question 3, answered otherwise than asked)

**Decided.**

1. *A word is an object.* It is not, at first, a representation of
   anything. The object and the word are made alike and at the same time,
   each from its own occurrences. Neither of C.5's two ways is taken: the
   object is not made for the word (as built, `PerceptCollections.observe`
   gathers what a word recurs with under `defining_word`), and it is not
   made before the word by a rule of its own (Claude's proposal of gaps).
2. *Naming.* The two come to be associated in virtue of a test, called
   "naming". It is a stage in the sense of `doc/Training.md`: what gates is
   the test, and nothing is gated inside the model. Piaget's name for what
   appears there is the semiotic (or symbolic) function, at the close of
   the sensorimotor period; Claude would keep "naming", which says what is
   measured.
3. *The meta concept is learned.* The logic that makes a meta concept
   directly is to be replaced by one that learns it: a Pi that includes
   both concepts, in virtue of their being presented at the same time.
   Directly made today: `interpret` makes a word's meta concept, with a
   blank object where the word has none, when it interprets a word whose
   meta concept does not yet exist (`Interpret.blank`, `Interpret.define`;
   `doc/Orders.md`, "A meta concept and the blank object"); and the tie of
   the felt path gathers an object under its `defining_word` and keeps its
   certainty in `WordTies`.
4. *`createMetaConcepts`.* A model variable introduces meta concepts until
   the architecture supports learning them and a test introduces names
   reliably.

**Claude's reading of 4, which he confirmed (19.4).** The key governs the direct
making: where `createMetaConcepts` is true, `interpret` makes meta concepts
as it does today; where it is false, none is made directly and a name is
learned or not at all. Its default in `data/model.xml` is true, so that
every model of item 1.7 and both stopping gates are as they are; the felt
world's configurations set it false. When naming is reliable the key and
the direct logic are removed together. This is a second path kept by his
word, the one exception to H.5's rule against one.

**What it agrees with.** Rule 31 as `doc/Training.md` has it: "the meta
concept of the two ... is learned from their consistent association, when
the word's concept and the object's have been present together on enough
occasions", and Helen Keller at the well-house. `doc/Architecture.md` names
three ways a symbol is made: implicitly by the Pi, by MTM's nomination "on
recognizing that its constituents always occur together", or by `interpret`
for a meta concept. His answer removes the third in favour of the second.

**What it leaves open, for round two's design.** With no word to gather
them, something must still make one object of the several steps of one
value (72 to 78 are all "three"). The candidates: the bins of the value's
code, which already hold neighbouring steps together; the contexts that
teach the Sigma of order zero, the word among them; a larger percept of the
pyramid. Claude proposes one in round two's design, after reading how the
Sigma of order zero is taught, and before any text.

### 19.3 The rounds, as they now stand

C.4's table, with these answers:

| Round | Builds | Measured from input to output | Waits on |
|---|---|---|---|
| One: the image is perceived | A value is a percept whatever its place, in one table of values, for every reader of numbers; an image's box beside its sentence's line; the `scene` of a row loaded, `null` told from `[]`; the opening pass puts the image back and is charged for it | An image alone is put back, at varied steps and at places not met; a sentence with its image puts both back; a sentence with no image is charged nothing for one; the numeric proofs and both gates hold | nothing |
| Two: objects, words and naming | The felt thing's concept made from its own occurrences, as a word's is; `createMetaConcepts`; the name learned as a Pi over both concepts from their being present together; the test "naming" | Categories 1 and 2: with the key false, a word raises its object and an object its word, at the reserved step and the reserved places; with others present | Claude's proposal for what gathers the steps of one value |
| Areas | Percepts that grow on the two pyramids | The categories about size | Claude's proposal for the rule on the part tower, and his choice |
| Three: the first milestone | What reads an arrangement | Category 14's contrast sets, per answer, on pairs held out | questions 4 and 5 of C.5 |

Questions 4 to 8 of C.5 keep their defaults as defaults.

### 19.4 A Pi for each mereological space, one Not, and the default (Alec, 2026-11-24)

Shown 19.1 to 19.3:

> And I think, because it is a separate mereological space, that we will
> want to have a different PI operation on that space. But the two
> modalities can be combined before negation.
>
> And yes, createMetaConcepts should default to true until the assessment
> tool and testing harness can be brought up to speed.

**Decided.**

1. *A Pi of its own.* The values are a mereological space of their own, a
   sorted line on which parthood is less-than-or-equal, where a form's
   parts are a set. The first Pi over values is therefore another
   operation than the first Pi over forms, and not a second instance of
   it.
2. *One Not.* The two modalities are combined before negation: each has
   its Pi, and one Not follows both. This stands over C.6, point 3, where
   Claude read "a separate Pi and not" as a Pi and a Not for each
   modality. What was kept from there: nothing joins the two modalities
   below the Not, and concepts tie them together.
3. *`createMetaConcepts` is true by default* in `data/model.xml`, until the
   assessment tool and the testing harness can carry the test of naming.
   The felt world's configurations set it false when round two builds the
   learned name.

*Against the source, for round one's design.* A different reading for
values is partly there: a collection of values is read by its two bounds,
each 0 or 1 (`BasicPercepts.read_points`, `read_collection`), "and not
through the Sigma" (`doc/Spaces.md`), while a form is read by the share of
its parts. Round one says which operation is the Pi over values and where
the two modalities' rows meet before the Not.

## 20. Round one: the image is perceived (Claude, 2026-11-24; the text for step A's implementer is §20.6)

*Design, after two Opus readers that changed nothing and Claude's reading
of what they named. It rests on 19.1 (the percept of a pixel is its value),
19.4 (a Pi for each mereological space, one Not), 16.7 (boxes that share no
place) and C.6 (the modalities meet at concepts).*

### 20.1 What exists

| | As built |
|---|---|
| *The first Pi over forms* | `BasicPercepts.read_bank`: over a sentence's bank of occurrences, each a `.what` on two lanes with its `.where` and `.when` wires. A row reads an occurrence for where the occurrence holds the row's parts and the row's wholes hold the occurrence, and against where it holds the parts and lies outside a whole (`held_at`), times the row's entries on the wires (`place_factor`). A conjunction is the product over its members (`SparsePiLayer3`, `bin/OrderZero.py`). |
| *A value in the first Pi* | Nothing: a value's row of the Pi is empty (`member_keys` and `against_keys` return none for it). |
| *The numeric stem* | `NumericPercepts.read` sets a value's two lanes by hand, 1 and 0 where its row is the reading's, and sends them through `order_zero_field` (the Not, then the Sigma) in a call of their own. It keeps a concept for each input place (`NumericPercepts.rows`). |
| *The value beside the words* | No Not and no Sigma: read by its two bounds, each 0 or 1 (`read_points`, `read_collection`), and joined to the sentence's field as one more column after the Sigma (`ScalarPercepts.join`). |
| *The one Not* | `ConceptualSpace.order_zero_not`, applied in `order_zero_field`, which `cs_read_percept_field` calls for text. Text's lanes and a number's never meet before it. |
| *A place told by its wires* | For text, yes: a place's row of the first Pi has its entries on the where wires and may keep a what (`reserve_place`), and is nominated when occurrences recur at it (`Places.meet`). For a number, no: the place is in the percept's key. |
| *A row of input* | `text` and at most `scalar`; any other key is refused (`InputSpace.prepInput`). A text model's registry has no box for an image; the numeric box is a numeric model's alone. Nothing carries a frozen corpus's document to `InputSpace`. |
| *The opening reconstruction* | `FieldReverse.reverse` matches places in the `text` range only and skips a value's row; `SentenceField.charge` compares bytes. Nothing is charged for a value. |

### 20.2 The design

1. **An image has a bank of occurrences of its own** (reworded by 20.7;
   as first written a pixel was an occurrence in its sentence's bank). A
   pixel that is not background is an occurrence in its image's bank: its
   `.what` is the code of its intensity, its `.where` the twelve wires of
   its place, its `.when` the moment's. This is "basic percepts there, as
   elsewhere". What an occurrence of a word has in its sentence's bank, an
   occurrence of a pixel then has in its image's without its being built a
   second time: a place nominated when occurrences recur at it, the first
   Pi's way back, a look that may take it.
2. **A value in one table.** The bins of a value's code are in order for
   every place; the row of a value is keyed by the value alone. The draw
   by place (`value_coordinates`), the place in the key (`reserve(...,
   place=)`) and `BasicPercepts.places` go.
3. **The Pi over values.** A value's row reads a value's occurrence by its
   two bounds on the sorted line, as a form's row reads by its parts and
   its wholes: *for* where the occurrence holds the row's value and the
   row's bound holds the occurrence (the same bin); *against* where the
   occurrence holds the row's value and lies above its bound (a greater
   value). An occurrence below the row's value does not hold it, and the
   row is absent there. It is the rule of `held_at` carried to a line, on
   which part and whole are one cut. *To be measured, not assumed:* that
   "against a lesser value" is what the Sigma wants; the other reading is
   for on the equal bin and nothing elsewhere.
4. **Two spaces.** The image's bank is read by the Pi over values and the
   sentence's by the Pi over forms, so a form's row is never read at a
   pixel nor a value's row at a word, though their codes lie on the same
   coordinates of the cube (reworded by 20.7).
5. **One Not and one Sigma.** Both kinds of rows stand in the one field of
   percepts that goes to `order_zero_field`. There is one call for a pass
   of reading, and the Not sees both.
6. **A place is told by its wires.** "The value 1 at the first place" is a
   place's row of the first Pi that keeps that what, nominated on
   recurrence as a place's row is for text. XOR's conjunction is over two
   such rows. The numeric stem's concept for each input place goes, since
   a generalization holds no place.
7. **One reader.** The numeric stem makes a bank of its input (each place
   of its box whose number is above zero is an occurrence) and reads it by
   the functions of points 1 to 6; the lanes set by hand go. *Named
   exception:* the value beside the words takes point 2 now, and keeps its
   own reading and its join until round two replaces the tie that reads
   them (19.2).
8. **The scene of a row.** A row may carry `scene`: `null` for no image,
   an empty list for an image with nothing in it, or pixels `[x, y, z,
   steps]` with steps from 1 to 255. The value of a pixel is its steps
   over 255. A text model that is to feel a scene declares its box, 8 by 8
   by 8, and the registry sets it on a stretch of the first axis of its
   own (16.7).
9. **Put back and charged.** At the opening the image is put back as the
   sentence is: from the concepts of order zero through the Sigma and the
   first Pi's kept percept to a value, read exactly from the table of
   values, at the place its wires match within the scene's box (16.7,
   point 1). The cost compares, place by place over the box, what was put
   back with what was felt. A row with no image is charged nothing for
   one; an empty image is charged for whatever is put back.

**Not built in this round.** A look that takes a felt thing (round three).
Objects, names and `createMetaConcepts` (round two). Percepts that grow on
the two towers. The value beside the words through the one reader (with
round two).

### 20.3 The steps

| Step | Builds | Measured |
|---|---|---|
| A: the scene reaches the model | Point 8 and the bridge from a frozen document to rows | 20.4, S1 to S4 |
| B: values in the bank | Points 1 to 7, on the numeric stem, which has certificates from input to output already | 20.4, V1 to V6 |
| C: the scene in the sentence's field | Points 1, 5 and 9 for a row of text and a scene | F1, F2, F11 and F12 of §7 |

Step A is needed whatever step B becomes and waits on nothing. **Step B is
put to Alec before its text is written:** it makes a number an occurrence
in the same bank as a word, which reworks the numeric stem and changes what
the numeric XOR certificate asserts of its rows (a conjunction over two
places' rows, where it now names a concept for each axis).

### 20.4 Certificates

Step A, in `test/test_item16_scene.py`, each printing what it measured:

| | Certificate | Passes when |
|---|---|---|
| S1 | The box | A model whose configuration declares the scene's shape has a range `scene` of 512 addresses, 8 by 8 by 8, that shares no place with another range. A configuration that does not declare it has no such range, and every range it has is where it was: the addresses and the wires of `text`, `numeric`, `scalar`, `parts`, `wholes` and `conceptual` are equal, before and after, for every configuration under `data/` |
| S2 | The row | `prepInput` takes rows of `text` and `scene`, tells `null` from an empty list in what it keeps, and keeps each pixel's place and value. It refuses, each with its own message: a key it does not know, a pixel outside the box, steps outside 1 to 255, two pixels at one place, and a scene on a model with no box. A row with an image and no words is active |
| S3 | The corpus | Every document of the 27 frozen files becomes rows that `prepInput` takes on a model with the box, and the rows give back the document's sentences and scenes exactly |
| S4 | Far enough apart (a measure) | The least distance between the wires of any place of `text` and any place of `scene`, beside the least distance between two neighbouring places, at the capacities of the configuration used. If the first is less than the second the box is given another origin and the measure is taken again. Reported |

Step B (for its text, when it is written): V1, the same value at two
places is one row and part-of on two values is less-than-or-equal; V2, a
form's row reads nothing at a value's occurrence and a value's nothing at a
form's, on one bank that holds both; V3, one call of `order_zero_field`
holds both kinds of rows; V4, the numeric XOR certificate from input to
output, its error, its answers and its exact reconstruction as they are;
V5, the image configurations' tests that pass in the baseline, with time
and memory before and after; V6, both gates, the echo tests and the
reading lesson.

### 20.5 What step A does not do

Nothing reads the scene yet: a row's scene is kept beside its text and the
model's field is what it was. No training loop, no loader in `bin/data.py`,
no change to the corpus or to the reconstruction. The value beside the
words is as it is.

### 20.6 Text for step A's implementer

~~~
Item 1.6, round one, step A: the scene reaches the model. Read §20 of
doc/plans/2026-11-18-item1-6-felt-math-world.md whole, then §19, §16.7,
§18.4 and H.5 of the same document, and doc/Spaces.md, "Modalities: One
Space of Places". §20.2's point 8 and §20.4's S1 to S4 are what you build;
§20.5 is what you do not. main is being changed by another implementer at
the same time, so keep each change as small as it can be.

Keep a file, tmp/item1-6/scene_progress.md, current after each step.

1. The key. One key in data/model.xml and data/model.xsd that declares
   the shape of a scene's box, with a default that declares none; no other
   file of data/ gains it except the one configuration of step 5, and
   data/BasicModel.xml gains nothing. No default lives in code. Name it
   beside `whereAxes` and say in the return what you named it.
2. The registry (bin/WhereRegistry.py, install_where_registry): where the
   key declares a shape, a range `scene` of that shape, on a stretch of
   the first axis of its own, after the ranges that exist, so that no
   address and no wire of an existing range moves in any configuration.
   Where it declares none, the registry is what it was.
3. The row (InputSpace.prepInput in bin/Spaces.py): `scene` beside `text`
   and `scalar`. Keep the scenes of a batch as `_scalar_values` keeps the
   values, cleared as it is cleared: for each row `None` where the row has
   no image, and otherwise its pixels, an empty image being empty and not
   `None`. A pixel is [x, y, z, steps]; its value is steps / 255. Refuse
   what S2 lists, each with a message that says what was wrong. A row
   with an image and no words gets its sentence's slot, as a row with a
   value and no words does (the two places that test the scalar for that:
   bin/Spaces.py near 9904 and bin/ModelAttention.py near 173).
4. The bridge: a function beside `load` in bin/felt_world_corpus.py that
   gives a FeltDocument's rows, one for each entry, each `text` and
   `scene`, in order, and its question and answer as they are. No model is
   involved in it and it changes nothing of what is frozen:
   test/test_felt_corpus.py passes untouched.
5. One configuration for the tests, a copy of the text model that
   test/test_item17_echo.py uses with the scene's shape declared (8, 8,
   8), under data/. Say which you copied.
6. test/test_item16_scene.py: S1 to S4 of §20.4. For S1's "before", print
   every configuration's ranges on the frozen copy of the source as it is
   now (its path is in your prompt) before you change anything, and keep
   the output.
7. Run, with the bounded runner (.venv/bin/python test/test_report.py
   <files>), RUN_TRAINED and RUN_SLOW unset: both stopping gates,
   test/test_item17_echo.py, test/test_item16_grid.py,
   test/test_item16_scene.py, test/test_felt_corpus.py,
   test/test_item18_scalar.py, and every test file that names prepInput
   with a dict row or install_where_registry. No test that passes in the
   sweep of this source (its outcomes are named in your prompt) may fail.
   Do not run the whole sweep.
8. No assertion is weakened, skipped or marked; no seed is chosen; the
   gates are not touched. No standing doc: Claude writes them at review.
   Nothing removed is deleted but moved to a dated folder under /tmp and
   named.
9. Return as §21, at the end of the same document: every file changed and
   why in a line each; the key's name and default; the box's origin in
   each configuration that declares it; each certificate with what it
   printed; S4's two distances and whether the origin moved; whatever in
   §20 was wrong or could not be built as written, and what you built in
   its place. You do not stop on a question of design: propose, build,
   mark it. Stop only if a gate cannot be made to pass with its assertion
   as it is.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

### 19.5 How a meta concept comes about naturally (Alec, 2026-11-24)

> If we think about what it will take for a meta concept to be created
> naturally, the co-presentation of both name and its object will cause both
> to develop basic percepts. Because they occur together, a single concept
> may form that unites them also, as MTM would assume such as a special case
> of making a single object. And I think that would be the ...
>
> object event, actually.
> The objects will have to be known at order one first, as separate and
> independent things. but at order one the two nouns can be taken together
> into a single first order concept (which is why symbols were necessary),
> and that uniting concept will be the meta concept that we had been
> creating.
>
> We can look for evidence of meta-concepts in the architecture by looking
> at unitizations across modality. If they occur at order 1 or above,
> perhaps they automatically become candidates for interpret() (and somehow
> they are substitutable because they are both symbols). So the activation
> of one spreads to the activation of the other, and the name begins
> carrying the properties of its object. At some point not only will it
> carry those properties through priming, but we will be calling interpret
> as an operation (which is basically asking of a word, "What does this word
> mean?", and thereby stimulating the meaning of the object).
>
> Does this make sense?

**Claude's answer, as given.** It does, and the making is already in the
source for concepts in general: "A conjunction of concepts is promoted as a
new symbol, a Pi row over its members, under a new concept, a Sigma over
that symbol" (`doc/Architecture.md`; `FamiliarTier.observe_field`, which
offers each witnessed concept's co-present concepts as a candidate, and
`promote_candidates`). The natural meta concept is that promotion when the
members are of two modalities.

*The stages it gives the test of naming.* (i) Each of the two is known as
a thing of its own. (ii) A unit over both exists one order above them.
(iii) One raises the other when it is met alone. (iv) `interpret`, asked of
the word, gives the object. Each can be measured apart.

*Points raised with him.*

1. *The count of orders.* By `doc/Orders.md` a word over its occurrences
   and an object over its own are concepts of order zero, and the concept
   that unites their symbols is of order one. His "known at order one
   first" is read as order zero; asked.
2. *Never in the first Pi.* The unit is a Pi over two symbols at order one.
   The first Pi makes no conjunction of a form's row and a value's row,
   since the two are separate mereological spaces with a Pi each (19.4).
3. *A Pi is silent with one member absent.* The unit's own value is nothing
   when the word is met alone, so "the activation of one spreads to the
   other" is the way from a member through its holder to the other member,
   which the source has in part (a conjunction's holder at the first Pi,
   `BasicPercepts.held`; expectation, whose test
   `test_item18_expect::test_81_...` fails on the unchanged source). To be
   read before round two's design.
4. *"Always" against the corpus.* The word is met with no image and the
   thing with no word, and a word is met with several things present. The
   rule that nominates must go by recurrence of their being together and
   not by their never being apart, and the consistent pair must win over
   the accidental ones. Round two measures it with others present.
5. *What marks a name.* Two felt things that are always together unite too,
   as a whole of two parts, and two words as a phrase. What marks a name
   is that its members lie in two modalities, which the ranges of their
   percepts' places tell. That is his measure: units across modality at
   order one or above, counted by the assessment tool.
6. *Still owed.* What makes one object of the several steps of one value,
   with no word to do it: the more needed now, since the object is to be
   known as a thing of its own before it is united with its word.

### 20.7 An image, not a sentence; intensities (Alec, 2026-11-24)

> Just reading what you are saying about pixels.
> "A pixel that is not background is an occurrence in its sentence's bank"
> For pixels, lets just call those images and not sentences.
> A better way of speaking is intensities, so that we can represent sound,
> taste, touch, and smell using intensities, also. But for now pixels
> within an image is a good metaphor.

**Decided.**

1. *A pixel belongs to an image.* An image holds its pixels as a sentence
   holds its words; a pixel is not an occurrence of a sentence.
2. *The modality is intensities.* Sound, taste, touch and smell are to be
   intensities too. Pixels within an image is the case built now and the
   way of speaking for now.

**What it changes in §20.2.** Point 1 becomes: *an image has a bank of
occurrences of its own.* Each of its pixels that is not background is an
occurrence there, with the code of its intensity as its `.what`, its
place's twelve wires as its `.where` and the moment as its `.when`. The
image's bank is read by the Pi over values and the sentence's by the Pi
over forms; the rows of both stand in the one field that goes to the Not
(point 5). Point 4 follows from it without a test of kind in one bank: a
form's row is never read at a pixel because the two are in different
banks. The numeric stem's input is an image: XOR is an image of two
places.

*Names.* The row's key stays `scene`, as the frozen corpus has it: a scene
is the image a row carries. In prose and in comments it is an image of
intensities.

### 19.6 Nouns are at order one (Alec, 2026-11-24)

Asked whether a word and its object are concepts of order zero and the
concept that unites them of order one:

> You asked if word and object are order zero: no, they nouns and thus are
> abstractions of the event, so that probably cannot occur until order 1.

**Claude's reading, by the ladder of `doc/Orders.md`, put to him.**

| Level | What it is | For "three" and its intensity |
|---|---|---|
| Basic percept | An occurrence, an event | The form at a place; an intensity at a place |
| Concept of order zero | A Sigma over basic percepts: what is the same over the events | The word over its occurrences; the intensity over its own |
| Symbol, the Pi of order one | A thing: a noun | The word as a thing; the object as a thing; and the two present together, his "object event" |
| Concept of order one | A Sigma over symbols | The meta concept |

So the nouns are symbols of the Pi of order one, as `doc/Orders.md` has it
("the word as a thing is a symbol over that concept"), and they are "taken
together into a single first order concept" by the Sigma of order one,
"which is why symbols were necessary". Claude's question of 19.5 had the
nouns one level too low.

*What it would settle of 19.5.* The Pi over both, present together, is the
event that nominates (his "a PI that includes both concepts in virtue of
them being presented at the same time", 19.2). The meta concept is the
concept over the nouns' symbols. If that Sigma reads each noun's symbol,
either noun raises it, which is his "substitutable because they are both
symbols" and "the activation of one spreads to the activation of the
other", and 19.5's third point (a Pi is silent with one member absent)
does not arise. The promotion as built puts the new concept over the
conjunction's symbol alone; whether it also reads the two nouns' own
symbols is asked of him.

### 18.7 The order of two words: five more repeats (Claude, 2026-11-24)

Five more alternating fresh starts on the two frozen copies
(`xor_repeat2.log` in the session's scratch folder), while step A's
implementer shared the machine:

| | Passed | Failed, and how |
|---|---|---|
| Unchanged source | 4 of 4 | none; a fifth run ended in an error of the test's set-up, not a failure, and is not counted |
| Grid | 3 of 5 | 2 of 4 in order, twice |

With 18.2's runs: the unchanged source 10 of 12, the grid 8 of 14. If the
two were alike, a difference this great or greater would come by chance
about one time in six (Fisher's exact test, one side, 0.155). So the grid
is not shown to lose the order more often, and it is not shown not to: 18.2's
"no evidence" is too strong, and the honest state is a lean against the
grid that these runs cannot settle. The grid's two new failures are partial,
so 18.2's remark that the grid fails by every pair reversed does not hold
as a rule.

*What it does not change.* The grid stays as built. The order of two words
after grammar training is item 1.7's open point on either source and bears
on question 5 of C.5. *What would settle it:* some twenty more starts on
each copy, on the machine alone (about seven hours), or finding what loses
the order.

## 21. Round one, step A: the return (Claude on Opus, 2026-11-24)

The work of §20.6, steps 1 to 9. Nothing reads the scene: a row's scene is
kept beside its text and the model's field is what it was (§20.5). No git
was run; nothing is committed. Nothing was removed, so no folder under
`/tmp` was made.

### 21.1 Files changed, and why

| File | Why |
|---|---|
| `data/model.xml` | The key `architecture.sceneShape`, beside `whereAxes`, default empty (no box). |
| `data/model.xsd` | `sceneShape`, `xs:string`, beside `whereAxes`. |
| `bin/Spaces.py` | `scene_shape(raw)`, the key read to a shape; `InputSpace.scene_shape` and `InputSpace._scenes`, both set in `__init__`; `InputSpace.scene_pixels`, a row's scene checked and kept; `prepInput` takes `scene` beside `text` and `scalar` and keeps the batch's scenes in `_scenes`, cleared where `_scalar_values` is cleared; `_finalize_sentence_word_layout` makes a row with an image and no words active, beside the scalar's line. |
| `bin/ModelAttention.py` | The early return for a row with no words and no value now also waits for a row with an image (the second of the two places of step 3). |
| `bin/WhereRegistry.py` | `install_where_registry` appends the range `scene` after `conceptual` where `inputSpace.scene_shape` is declared; otherwise the list of ranges is as it was. |
| `bin/felt_world_corpus.py` | `rows(document)`, beside `load`: a FeltDocument's rows, one for each entry, each `text` and `scene`, in order, with its question and answer as they are. |
| `data/scene/BasicModel_scene.xml` (new) | The configuration of step 5 (21.3). |
| `test/test_item16_scene.py` (new) | S1 to S4. |
| `test/fixtures/item16_scene_ranges_before.json` (new) | S1's before: every configuration's ranges on the frozen copy, each with its addresses, box and a sha256 of its wires. |
| `tmp/item1-6/scene_ranges.py`, `scene_ranges_{before,after}.{json,log}`, `scene_progress.md`, `scene_run{1,2}.log`, `scene_pytest1.log` | The script that printed S1's before and after, its output, the progress file, the run logs. Not source. |

`test/test_felt_corpus.py` is untouched. `data/BasicModel.xml` gains
nothing; no other file of `data/` has the key but the configuration of
step 5.

### 21.2 The key

- **Name:** `sceneShape`, in `<architecture>`, beside `whereAxes` in both
  `data/model.xml` and `data/model.xsd`.
- **Form:** up to `whereAxes` positive whole numbers separated by spaces
  (`8 8 8`); an axis not given is 1. Anything else is refused when the
  input space is built (`scene_shape`).
- **Default:** empty: the model feels no scene, has no range `scene`, and
  refuses a row that carries an image.

### 21.3 The configuration, and the box's origin

`data/scene/BasicModel_scene.xml` is a copy of `data/BasicModel.xml`, the
file `test/test_item17_echo.py`'s model is built from
(`_tiny_canonical_model` parses it and sets its widths in the test), with
two differences: `<sceneShape>8 8 8</sceneShape>`, and `weightsPath`
`BasicModel_scene.ckpt` in place of `BasicModel.ckpt`, so that the two
never write one checkpoint (the file's own comment asks that the XML and
its weights share a stem). It is a plain copy, not the echo fixture's
widths baked in: those are set by the fixture in code, and the test builds
the echo's model with the shape this file declares (21.4).

It is in a folder of its own, `data/scene/`, and not in `data/` itself:
G1 of `test/test_item16_grid.py` asserts that the configurations
`data/*.xml` are exactly the 42 of its frozen record, so a 43rd at the top
level fails it. `data/matrix/` is the precedent. *Marked: decided by the
implementer.*

| Configuration | `scene` addresses | Origin | Shape | Capacity | Long period |
|---|---|---|---|---|---|
| `scene/BasicModel_scene.xml` | 40978 to 41490 | (40978, 0, 0) | (8, 8, 8) | 41490 (40978 + 512) | 65536, as `BasicModel.xml`'s |
| the echo's model with its shape (the test's) | 4370 to 4882 | (4370, 0, 0) | (8, 8, 8) | 4882 (4370 + 512) | 8192, as without it |

The origin is the registry's default for a range given after the others,
`(its first address, 0, 0)`: the box's stretch of the first axis, 40978 to
40986, begins where every other range's ends. S4 did not ask for it to
move.

### 21.4 The certificates (`test/test_item16_scene.py`), as printed

The test's models are the text model of the echo test, built by
`test_item17_echo.echo_model` (so `_tiny_canonical_model`'s widths), once
as it is and once with `sceneShape` set to what
`data/scene/BasicModel_scene.xml` declares. `BasicModel.xml` at its own
widths takes about 6.6 GB to build and `BasicModel_corpus.xml` about 16 GB,
beyond the bounded runner's 8 GiB (my first run was stopped for memory, 21.6),
so the full-size configurations are measured by `tmp/item1-6/scene_ranges.py`
before and after, and the test builds three that fit.

S1, the box.

- `S1 sceneShape: default empty; declared by ['scene/BasicModel_scene.xml']
  alone, as '8 8 8'; it differs from BasicModel.xml in
  ['/architecture/sceneShape', '/architecture/weightsPath']`
- `S1 44 configurations before; 45 now, the one added scene/BasicModel_scene.xml`
  (the 42 of `data/*.xml` and the two of `data/matrix/`).
- Built afresh, each in a process of its own, against the frozen copy's
  record (`test/fixtures/item16_scene_ranges_before.json`): addresses,
  origin, shape and the sha256 of each range's rows of the grid, the
  capacity and the period, all equal:
  `S1 MM_text.xml: 6 ranges, capacity 4369, period 8192`;
  `S1 MM_xor.xml: 6 ranges, capacity 4179, period 8192`;
  `S1 MM_image.xml: 6 ranges, capacity 4961, period 8192` (a numeric
  box of 28 by 28).
- `S1 the echo's text model with sceneShape '8 8 8': scene at addresses
  4370..4882, origin (4370, 0, 0), shape (8, 8, 8); the first axis of the
  other ranges ends at 4370; capacity 4370 -> 4882, period 8192 -> 8192;
  text, numeric, scalar, parts, wholes, conceptual: addresses, boxes and
  wires as without it` (wires compared row by row, exactly; the seven boxes
  given to the registry again, which refuses two that share a place).
- *Outside the test, every configuration at full size*
  (`tmp/item1-6/scene_ranges_before.json` on the frozen copy,
  `tmp/item1-6/scene_ranges_after.json` on this tree, logs beside them):
  all 44 equal before and after, range by range with the wires' digests,
  capacity and period. `matrix/MM_20M_grammar_reading.xml` is built
  neither before nor after (`ValueError: marker requires declared form and
  meaning blocks`, a fault of the unchanged source). The 45th,
  `scene/BasicModel_scene.xml`: its six other ranges equal to
  `BasicModel.xml`'s in every field, the scene at addresses 40978 to 41490,
  origin (40978, 0, 0), capacity 41490, period 65536 as before.

S2, the row.

- `S2 kept: [((1, 2, 0, 0.7843137254901961), (7, 7, 7, 0.00392156862745098),
  (0, 0, 0, 1.0)), (), None, None, ((3, 4, 0, 0.2),)]`: a row with an
  image, an empty image (kept `()`, not `None`), `scene: null`, no key
  `scene`, and an image with no words. A batch of text alone keeps no
  scenes (`_scenes` is `None`); a row of text and a value keeps `[None]`.
- Refused, each with its message:
  `a row accepts text, scalar and scene only, not ['picture']`;
  `pixel [8, 0, 0, 10] lies outside the scene's box (8, 8, 8)`;
  `pixel [0, 0, -1, 10] lies outside the scene's box (8, 8, 8)`;
  `pixel [0, 0, 0, 0] has steps outside 1 to 255`;
  `pixel [0, 0, 0, 256] has steps outside 1 to 255`;
  `two pixels at the place (2, 3, 0)`;
  on the model with no box, `a scene was given to a model with no box for
  one (architecture.sceneShape)` (and `scene: null` is taken there).
- `S2 a row's sentence slot (an image, an empty image, nothing, a value):
  [True, True, False, True]`: with no word active in any row, the
  sentence's slot is open for the rows with an image, empty or not, as for
  the row with a value, and not for the row with nothing.

S3, the corpus.

- `S3 5600 documents of 27 files, 28333 rows, 27652 images (22 empty),
  102984 pixels: taken, and given back exactly`. Every split of every
  file; each document's rows taken by `prepInput` on the model with the
  box; the texts kept (`_last_sentences`) equal to its sentences, each
  scene kept `None` where the document's is, and otherwise its pixels with
  their steps (value times 255, rounded) equal to the document's.

S4, far enough apart (a measure).

- On the test's model: `S4 at capacity 4882 (period 8192), box origin
  (4370, 0, 0): least text-to-scene 0.195090, least between neighbours
  0.012277, ratio 15.9`.
- On `scene/BasicModel_scene.xml` at full size (the same measure, run
  once before the test was moved to the small model): `capacity 41490
  (period 65536), box origin (40978, 0, 0): least text-to-scene 0.923879,
  least between neighbours 0.012271, ratio 75.3`.
- Euclidean distance over the twelve wires; text-to-scene over every
  place of `text` and every place of `scene`; neighbours one step on one
  axis, along the line of text and along each axis inside the box. The
  first is the greater at both capacities, so **the origin did not move**;
  the test asserts it.

### 21.5 What in §20 was wrong or could not be built as written, and what was built

1. *"One configuration ... under data/"* (step 5) cannot be in `data/`
   itself while G1 of `test/test_item16_grid.py` holds the top-level set
   of configurations to its frozen record. Built in `data/scene/` (21.3).
2. *"no wire of an existing range moves in any configuration"* (step 2)
   holds of every configuration under `data/`, by S1. It is not a property
   of the registry as built: the ladder's long period is derived from the
   registry's whole capacity (`WhereEncoding.set_capacity`,
   `_ladder_periods`), which the box's 512 addresses join. A configuration
   whose capacity lies within 512 (and the ladder's 1/63 margin) below a
   power of two would have its period doubled by a box, and then every wire
   of every range moves (and `where_origin` with it). `BasicModel.xml` is
   far from it (40978 and 41490 both under the threshold for 65536, which
   is about 64511). Not changed: sizing the ladder by the greatest
   coordinate on any axis rather than by the address count is the remedy
   if it is wanted (equal for every existing configuration, since their
   last range is a line), and it is Claude's to choose. *Marked.*
3. *A row with an image and no words.* A dict row still requires `text`,
   as it did; a row with an image and no words has `text: ''`, which is
   what the frozen corpus writes. Not changed.
4. *S1's "every configuration under data/", in a test.* Taken
   recursively (the two of `data/matrix/` are in it). They cannot all be
   built under the bounded runner: `BasicModel.xml` and its copies take
   about 6.6 GB each and `BasicModel_corpus.xml` about 16 GB, against a cap
   of 8 GiB for the worker and its children. The test builds three that
   fit (text, a numeric line, a numeric box), checks the key in every
   file, and checks the box on the echo's model; all 44 at full size were
   compared by the script, outside the runner (21.4). *Marked.*
5. *The steps of a pixel.* Kept as the value, steps over 255, as step 3
   says; S3 gives the steps back by rounding the value times 255, which is
   exact for every step from 1 to 255.

### 21.6 The run of step 7

`env -u RUN_TRAINED -u RUN_SLOW ... test/test_report.py` with: both
stopping gates (`test/test_mm_xor.py::TestMMXorConvergence::test_convergence`,
from H.5, and `test/test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`),
`test_item17_echo.py`, `test_item16_grid.py`, `test_item16_scene.py`,
`test_felt_corpus.py`, `test_item18_scalar.py`; the files that give
`prepInput` a dict row (`test_item18_recognition.py`,
`test_percept_collection_binding.py`, `test_scalar_question_curriculum.py`
through `scalar_question_curriculum.py`); `test_item16_grid.py` is the one
that names `install_where_registry`; and, for the key,
`test_model_defaults.py`, `test_two_mode_dispatch.py` (it reads every
configuration under `data/`, recursively) and
`test_chooser_architecture.py::test_xml_schema_rejects_invalid_capacity`
(it validates `data/model.xml` against `data/model.xsd`).

- First run, `output/tests/20261124-034831-c7f9ef`: stopped for memory
  (8.14 GiB) at 22 of 126 cases, in S1's then form, which built every
  configuration at full size four at a time. Changed as 21.4 says.
- **Final run, `output/tests/20261124-041159-0812a4/result.json`:
  `Result: test_failure, exit 1; 86/86 cases completed.`** 377 s, peak
  3.4 GiB. 83 passed, 2 skipped, 1 failed. Both gates passed, the seven of
  the echo passed, G1 to G9 passed, S1 to S4 passed (24 cases).
  The failure is
  `test_percept_collection_binding.py::test_bind_percept_collection_reads_membership_not_coordinate_overlap[2.0-True]`,
  `failed` in the sweep of this source too; the two skipped
  (`test_item18_recognition.py::test_90_*`) are `skipped` there. No test
  that passes in the sweep fails.

### 21.7 Review of step A (Claude, 2026-11-24)

**Accepted.** Claude read the whole change (six files of source and
configuration, 86 lines; the new test, its record of the ranges before,
and the configuration) and ran, with the bounded runner, both stopping
gates, `test_item16_scene.py`, `test_item16_grid.py`, `test_felt_corpus.py`,
`test_item18_scalar.py`, `test_model_defaults.py`, `test_two_mode_dispatch.py`
and `test_doc_links.py` (run `20261124-041932-b7135e`): 128 cases, 125
passed, and the three that failed are `test_doc_links.py`'s that fail from
any worktree (18.5).

*On the four departures of 21.5.*

1. *The configuration under `data/scene/`.* Accepted: G1 holds the top
   level of `data/` to its record, and `data/matrix/` is the precedent.
   The file is `data/BasicModel.xml` whole with two lines changed, and S1
   asserts that it differs in those two alone, so when `main` changes
   `data/BasicModel.xml` the copy is brought up with it or S1 fails. That
   is wanted.
2. *S1 builds three configurations in the test and compares all 44 by a
   script outside the runner.* Accepted: the greatest configurations do
   not fit the runner's memory. The script and its two records are in
   `tmp/item1-6/` and are not source; the claim "every range equal in every
   configuration" rests on that one run of it.
3. *A box may double the ladder's long period.* True as he says: the
   period is taken from the count of addresses, and a box adds its count.
   It moves no configuration that exists. A model with a box is a model of
   its own, with weights of its own, so nothing is carried between a
   registry with the box and one without; the period is left as built.
   Sizing the ladder by the greatest coordinate on any axis is kept as a
   remedy if a configuration ever falls near a power of two.
4. *The origin did not move.* The nearest place of text is 15.9 times as
   far from the box as two neighbouring places are from each other on the
   test's model, and 75 times at full size. §16.7's third point is
   answered for the box as it lies.

*Standing documents written by Claude with this step:* `doc/Params.md`,
`doc/Spaces.md`, `doc/Training.md`, `doc/Testing.md`,
`doc/DevelopmentHistory.md`.

## 22. Alec's answers to the thirteen questions (2026-11-24)

Claude gave him every open question in one list, numbered 1 to 13 (the
paste lost the numbering; it is restored here). His words are quoted whole.

### 22.0 Before the answers: where a noun begins

> Please verify my logic, but a pi on events would still seem to be an
> event, as would a sigma on top of that. To remove the .when and .where
> elements, which is necessary for nouns, I think we need the first-order
> pi. This would be a good thing to document.

*Verified against the source and the documents.* A row of the first Pi is
read at an occurrence, which has a place and a time. A concept of order
zero is a Sigma over those rows; `doc/Architecture.md` says of it "A concept
is opaque: it keeps its `.where` and its `.when`", and its width embraces
the percept's band (§18.4). A symbol, a row of the Pi of order one, "has no
vector, place, form or trust" and "stores no `.where` or `.when`". So the
first thing with neither place nor time is the symbol of order one, as he
says. One sentence of `doc/Training.md` said the opposite of a concept of
order zero ("a generalization holds neither"); it is corrected.
*Documented* in `doc/Orders.md`, "Where a noun begins".

### 22.1 Needed for round one, step B

**1. Is the meta concept raised by either noun alone?**

> my guess is that the first order words and objects will live in different
> parts of conceptual space, so they cannot be united by a hypercone (pi).
> So it will be the sigma that makes a list of two seemingly unrelated
> items. That does not need raising, it is just concept formation but order
> one MTM.

*Decided.* The meta concept is a Sigma of order one over the two nouns'
symbols, a list of two. A Pi cannot unite them, since a Pi's row is a
region of sameness and the two lie in different parts of conceptual space.
It is formed by MTM at order one as any concept is, and nothing is raised
for it. This stands over 19.2, point 3 ("a PI that includes both"): what
is left of the Pi there is the being present together, which is what MTM
goes by. Either noun raises it, being a sufficient part.
*For round two's design:* as built, what recurs together is promoted as a
Pi row under a new concept; a Sigma over two symbols from their being
present together is to be designed.

**2. Should the numeric stem's input be read as an image too?**

> yes, one reader.

**3. Is a pixel's 0..1 the step over 255, linear?**

> we can do linear, the log chart was to avoid clipping,and sensation seems
> to be log. But we can treat that as a preprocessing step, and require all
> input in 0..1, where a linear map is best (also good for BF16)

*Decided.* Every reader of intensities requires its input in 0..1 and
takes it as it is. The logarithmic chart (`NumericPercepts.feeling`) is a
step before the reader, for numbers that are not in 0..1.

**4. May the value beside the words keep its own reading until round two?**

> implementer's decision

**5. Should the row's key be `scene` or `image`?**

> I do not understand this question.

*Withdrawn; nothing to decide.* It asked only what the field of an input
row that carries the image is called. The frozen corpus calls it `scene`
and step A built it so.

### 22.2 Needed before round two

**6. What makes one object of the several steps of one value?**

> the max of the parts are the lower bound, the min of the wholes (as
> properties) is the upper bound. So it is the same as for character
> percepts.

*Decided.* An object over values is bounded as a character's percept is:
its lower bound is the greatest of its parts, its upper bound the least of
its wholes, the wholes being properties. This is `doc/Mereology.md`'s rule
for a form's code ("the parts half of a percept's code is the greatest of
its parts' codes, and its wholes half the least of its wholes' codes"),
and it answers 19.1's "not yet said": the rule on the part tower is the
greatest of the parts. Neither the word nor a rule of gaps gathers the
steps: a property that holds of every step of "three" and of no step of
"four" does. *For round two's design:* what the properties of the line of
values are, as text has its table of wholes.

**7. Does the first Pi never join a form's row with a value's row?**

> I am proposing a separate pi for each space, becuase they are not
> mereologically related.

### 22.3 Needed before round three

**8. What reads an arrangement?** His answer is §23 whole. *Claude's note:*
the question was asked of "three is above four" as a relation of two
places in an image. He answers it as the question beneath it, how any
other sort is mapped onto the one sort the model has, parthood, by
relations ("we can do an ordinal sort using relations"), and then what
three and four are. So an arrangement is read by relations and not as a
percept of its own. Claude still owes the proposal for "above", after
round one.

**9. Reading in order through grammar training.**

> sorry about this. We can wait for 1.4 which I believe will help temporal
> verbs and sequencing in general.

*Decided.* What turns on the order of two words waits for item 1.4. Round
three's certificate on category 14 is not asked of this item before it;
the lean of 18.7 is left with it.

### 22.4 The corpus's defaults

**10. Do the counting words double as the value names?**

> "Three" is an abstract concept, it has any number of uses.

**11. Is the task named in the input?**

> we can do it this way, but a well-trained general model will not make
> that assumption. So if you want this test to run as a part of a battery
> on a large number of tests that already have sentence comprehension, we
> need a version that asks a question about the image in the text modality.
> Also, this points out the need for numerical output as well as input, so
> please add that to the spec. We will become artists. But we will probably
> have to keep our day jobs also, and make art of them to the best of our
> ability.

*Decided.* Counting keeps the form with no text and gains one that asks of
the image in words. And the model is to give intensities out as it takes
them in (§24).

**12. What is said when nothing follows?**

> if there is a lack of understanding, the best might be to ask a question.
> But silence is fine for now.

**13. The bar.**

> this question is not about accuracy, but the amount of training required
> to build a model required upon which further development can continue. My
> guess is we want a ten year old, or at least a child who has passed all
> piagetian stages, before we can give it all the books to read.

*Decided.* The bar is not an accuracy. It is a model on which development
can go on: one that has passed every stage, each stage a test
(`doc/Training.md`), before it is given the books. No remote machine is
paid for before that.

## 23. Mathematics: what three and four are (Alec, 2026-11-24, with Claude's notes)

> You are asking how to map various sorts to the sort that we already have
> using relations. So we can do an ordinal sort using relations, and then
> you seem to be asking "what is three or four such that we can sort them?".
> The question is probably best answered in terms of the extension; three
> is the set of all things that have three objects, and defined analogously
> for four. But now we need to map that to our partial sort, so three is the
> set of "greater than two things" and "less than four things". Four is also
> greater than two things, but it is less than five things. So there is a
> mathematics over greater than and a mathematics over less than that are
> parthood relations over the extension of the sets so defined. Please read
> a quick bit about mereology if this is not familiar, the work by Peter
> simons is a good source for how few mereological axioms such as parthood
> can be used to define all of mathematics (we use mereology for parts and
> Lesniewski's naming operation which is equivalent to our symbolic
> formation at higher orders). Please add this to the doc in a mathematics
> section. The short answer is that four is defined at a high order based
> on greater than three and less than five, and it is defined over the
> integers, which are learned via the successor operation, which I think
> has to come from order-raising rather than continuous mapping within a
> given space. I think I have good words about this in Cognitive Set
> Theory, I will put a copy in the doc directory.

**What it says, in the model's terms.**

1. *One sort.* The model has one order, parthood: `partOf` on two codes,
   which on two values is less-than-or-equal (19.1). Every other sort
   (above and below, before and after, more and fewer) is mapped onto it
   by relations.
2. *A number by its extension.* Three is what all collections of three
   things have in common: a kind over them, at an order above the things
   and above their collections.
3. *A number by two bounds.* On the partial sort, three is "more than two
   things" and "fewer than four things". More-than and fewer-than are
   parthood between the extensions: a collection of three is a part of a
   collection of four that holds it. A number is so bounded from below and
   from above, as a level of intensity is bounded by its part and its
   whole (22.2, question 6).
4. *The integers come by the successor,* and the successor by raising an
   order, not by a step along an axis of one space. So a number is not an
   intensity. That "three" names an intensity in the first categories of
   the corpus is one use of an abstract word (22.4, question 10); the
   number three is met in the counting categories.

**From the literature, as Claude has it** (to be checked against his
*Cognitive Set Theory* when the copy is in `doc/`). Leśniewski built three
systems one on another: Protothetic, a logic of propositions; Ontology, a
calculus of names with the one primitive "is" ("a is one of the b"), in
which a name may name nothing, one thing or many; and Mereology, in which
the one primitive is "part of" and a class is the concrete whole of its
members and never a further abstract thing, so that Russell's class of
classes that are not members of themselves cannot be formed. Simons
(*Parts*, 1987) sets these out beside the calculus of individuals, with
parthood as a partial order (each thing a part of itself, no two things
parts of each other, a part of a part a part) and the principles of
supplementation and of sums above it. Parthood among individuals alone
gives a Boolean algebra without its zero and no arithmetic; number comes in
Ontology, where "there are exactly n of the a" is defined of names, at
higher and higher semantic categories. That is his point here: mereology
gives the parts, and the naming that makes a new name of what names have in
common, at a category above them, is the model's formation of symbols at
higher orders. A number is defined there, "at a high order".

**What it bears on.** Round three (22.3, question 8). The counting
categories and the said mathematics of the corpus (categories 26 and 27).
`doc/ModelAge.md`'s song and the successor. Nothing of round one or two.

## 24. Added to the specification by his answers (Claude, 2026-11-24)

1. **Intensities out.** The model gives intensities out as it takes them
   in: an answer may be an image, pixels in the scene's box with values in
   0..1, as an answer may be words. "We will become artists." *Not designed
   yet.* Its nearest ground is round one, step C, which puts an image back
   at the opening; the way out is that way back asked for at the output.
   It is designed after step C returns, with its certificate: an image
   asked for in words is given, measured place by place.
2. **Counting asked in words.** Category 27 gains a form in which the image
   comes with a question in text, beside the form with no text. It is a
   revision of the corpus (a recipe and a freeze), made by an implementer
   after step B returns, so that two implementers do not change one tree
   at once.
3. **Input in 0..1.** Every reader of intensities requires 0..1. The
   logarithmic chart is a step before a reader (22.1, question 3).
4. **Silence when nothing follows** stays the answer. To ask a question
   where understanding is lacking is the better answer and is not built.
5. **The bar** is every stage passed, not an accuracy (22.4, question 13).
6. **What turns on the order of two words waits for item 1.4** (22.3,
   question 9).

### 20.8 Step B: values in an image's bank (the text for its implementer)

*Rests on 19.1, 19.4, 20.2 as reworded by 20.7, and 22.1 (one reader; input
in 0..1; the value beside the words is the implementer's to decide).*

Certificates, in `test/test_item16_values.py`, each printing what it
measured:

| | Certificate | Passes when |
|---|---|---|
| V1 | One table | The same value at two places is one row of the basic percepts. Its code is the bins at or below it, in order, whatever its place; `partOf` on two values' codes is less-than-or-equal, over sampled pairs. Nothing draws a value's coordinates by its place, and no percept's key holds a place |
| V2 | Input in 0..1 | A reader of intensities takes a value in 0..1 as it is and gives it back exactly; it refuses a value below 0, above 1 or not a number, with a message. The logarithmic chart is applied by no reader |
| V3 | The Pi over values | On an image's bank, a value's row reads for at an occurrence of its own bin, against at an occurrence of a greater value, and nothing at an occurrence of a lesser value or where nothing is. It is a function of the value space, named apart from the first Pi over forms |
| V4 | A place by its wires | After an image has recurred as often as nominates a form, each place that held a value has a place's row that keeps that value's what. The same value at two places: one value's row, two places' rows |
| V5 | XOR from input to output | `test/numeric_xor.py`'s certificate: its error, its answers and its exact reconstruction as they are; its conjunction is over two places' rows. The stopping gate `test_mm_xor.py::TestMMXorConvergence::test_convergence` passes untouched |
| V6 | One Not | A pass of reading sends its rows through `order_zero_field` once |
| V7 | Images | The tests of the image configurations (`MM_image.xml`, `mnist.xml`) that pass in the sweep pass; their time and greatest memory before and after |
| V8 | Text as it was | The other stopping gate, the seven echo tests, the reading lesson, `test_item16_grid.py`, `test_item16_scene.py` |

~~~
Item 1.6, round one, step B: values in an image's bank. Read §20 of
doc/plans/2026-11-18-item1-6-felt-math-world.md whole (20.7 rewords 20.2;
20.8 is yours), then §19.1, §19.4, §22.0 to §22.2 and H.5 of the same
document, then doc/Spaces.md ("Level Sets", "Modalities"),
doc/Mereology.md from "A scalar is the same mechanism", and the source:
bin/BasicPercepts.py (read_bank, held_at, reserve_place, place_factor,
kept_percept, value_code, perceive_value, read_points), bin/Places.py,
bin/NumericPercepts.py, bin/OrderZero.py, Spaces.py's order_zero_field and
cs_read_percept_field, test/numeric_xor.py and test/test_item17_places.py.
§20.2's points 1 to 7 are what you build, on the numeric stem; point 8 is
built and point 9 is not yours. main is being changed by another
implementer at the same time, so keep each change as small as it can be.

Keep a file, tmp/item1-6/values_progress.md, current after each step. If
your context runs short, stop at the end of a step with that file current.

0. Before you change anything, on the frozen copy of the source as it is
   now (its path is in your prompt): run test/numeric_xor.py's certificate
   and the tests of the image configurations, and keep what they print,
   their times and their greatest memory. They are V5's and V7's "before".
1. One table of values (point 2). The bins of a value's code are in order
   for every place; a value's row is keyed by the value alone.
   value_coordinates, the place in reserve's key and BasicPercepts.places
   go, with every caller's place argument. Nothing is kept for the older
   form, in the source or in a saved state's loading.
2. Input in 0..1 (22.1, question 3). A reader of intensities takes its
   input as it is and refuses what is outside 0..1. The logarithmic chart
   stays a function for a caller whose numbers are not in 0..1 and is
   applied by no reader. Find every numeric data set the configurations
   use and say what range its numbers have; change a configuration only
   where the linear value makes one of its declared values false, and
   list each.
3. An image's bank (point 1 as reworded). The numeric stem makes a bank of
   its input: each place of its box whose value is above zero is an
   occurrence, its .what the value's code on two lanes, its .where the
   twelve wires of its place from the registry's grid, its .when the
   moment's. Use the bank that a sentence has (PerceptBank) if it can hold
   it; if it cannot, say why and what you built.
4. The Pi over values (point 3): a function of BasicPercepts for the value
   space, by the two bounds, as V3 says, with the row's entries on the
   wires (place_factor) as for a form. Show in the return whether it
   agrees with held_at on a line. If a gate or V5 cannot pass with
   "against at a greater value", build the other reading (for on the equal
   bin, nothing elsewhere), and say which passes and what each measured.
5. One Not (point 5): the bank's rows go through order_zero_field once.
   The lanes set by hand in NumericPercepts.read go.
6. A place by its wires (point 6): a place's row is nominated for an image
   as it is for a sentence (Places.meet), and keeps the what. The concept
   for each input place (NumericPercepts.rows) goes. The way back
   (NumericPercepts.reverse) gives each place its exact value from the
   table of values.
7. The value beside the words (bin/ScalarPercepts.py) is yours to decide:
   it takes step 1 in any case; whether it keeps its reading and its join
   until the tie is replaced, or goes through the one reader now, say
   which and why. If it keeps them, it applies the logarithmic chart
   before it reads, so that its numbers are what they were.
8. test/test_item16_values.py: V1 to V8 of §20.8.
9. Port the tests this makes false. A test that asserted a value's percept
   by its place, or the numeric stem's concept for each place, asserts
   what is true now; list each with its old assertion and its new one.
   The reader's list is in your prompt. No other assertion is weakened,
   skipped or marked; no seed is chosen; neither stopping gate is touched.
10. Run, with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
    stopping gates, test/test_item17_echo.py,
    test/test_reading_pretraining.py, test/test_item16_grid.py,
    test/test_item16_scene.py, test/test_item16_values.py, every test file
    you changed, and every test file that names NumericPercepts,
    ScalarPercepts, perceive_value, value_code, numeric_xor or
    MM_image. No test that passes in the sweep of this source (its
    outcomes are named in your prompt) may fail; for one that does, find
    the cause and repair the source, not the test. Do not run the whole
    sweep.
11. No standing doc: Claude writes them at review. Nothing removed is
    deleted but moved to a dated folder under /tmp and named.
12. Return as a file, tmp/item1-6/section25.md, headed
    "## 25. Round one, step B: the return": every file changed and why in
    a line each; each certificate with what it printed; V5 and V7 before
    and after; every test ported, with its old and new assertion; every
    configuration changed; what you decided for the value beside the
    words; whatever in §20 was wrong or could not be built as written,
    and what you built in its place. You do not stop on a question of
    design: propose, build, mark it. Stop only if a gate cannot be made to
    pass with its assertion as it is, and then say exactly where it fails.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 25. Round one, step B: the return (Claude on Opus, 2026-11-24)

The work of §20.8, steps 0 to 12, on the numeric stem. No git was run;
nothing is committed. Progress was kept in `tmp/item1-6/values_progress.md`.
The files as they were before are kept in `/tmp/item1-6-2026-11-24/` (`bin/`
and `test/`), since code was removed from them; no file was deleted.

### 25.1 Files changed, and why

| File | Why |
|---|---|
| `bin/BasicPercepts.py` | Step 1: `value_coordinates`, the `places` buffer, the place in `reserve`'s key and every `place` argument (`value_code`, `_boundary_codes`, `bin_of`, `perceive_value`, `reserve`, `subspace_bounds`) go; the extra state no longer saves or reads `places`, and the load default for it goes. Step 3: `value_bank`, an image's bank as a `PerceptBank`. Step 4: `read_values`, the Pi over values (`held_at` on a line, by reading, the conjunction's Pi and `held` over the readings), and `place_bounds`, a row's where entries read by their bounds, 0 or 1 (25.6, 1). `reserve` takes a conjunction with no parts (one over places' rows). |
| `bin/NumericPercepts.py` | Step 2: `intensities`, the reader's check of 0..1; `feeling` and `number` stay as functions for callers, applied by no reader here. Steps 3, 5, 6: `read` makes the image's bank, meets its places (`Places.meet`), reads the promoted rows by `read_values`, sends them once through `order_zero_field`; the lanes set by hand and the concept for each input place (`rows`, made in `__init__`) go; the conjunction is nominated from the places' concepts read together at the bracket. `reverse` gives each place its exact value from the table of values. |
| `bin/ScalarPercepts.py` | Step 1 (and step 7): `code`, `bare` and `perceive` pass no place. Its reading and its join are kept (25.5). |
| `test/numeric_xor.py` | Step 9: the certificate's conjunction is over two places' rows (25.4). |
| `test/test_item18_places.py` | Step 9: `test_level_percepts_have_native_bounds_and_distinct_places` (25.4). |
| `test/test_item18_perceptual_scalar.py` | Step 9: `test_scalar_object_collects_value_bounded_basic_percepts` (25.4). |
| `test/test_item18_fields.py` | Step 9: `test_numeric_scale_order_and_level_code` (25.4). |
| `test/test_item17_code.py` | Step 9: `test_49_collections_of_values_are_told_apart`, `test_49_the_numeric_certificate_keeps_its_bar` (25.4). |
| `test/test_item16_values.py` (new) | Step 8: V1 to V7. |
| `test/test_item18_paths.py`, `test/test_understanding_reconstruct.py`, `test/test_output_synthesis.py`, `test/test_mm_xor.py` (one test, not the gate) | Ports of the review's round: inputs above 1, and inputs presented until their places are known (25.10). |
| `tmp/item1-6/values_measure.sh`, `values_fit.py`, `values_before/`, `values_after/`, `values_run_files.txt`, `values_run*.log`, `values_progress.md` | The measure of V5 and V7 before and after, the fit outside the test, the run's list and logs, the progress file. Not source. |

`data/` is unchanged: no configuration, no key. `data/BasicModel.xml`
gains nothing. Neither stopping gate's file is touched, nor `test/test_mm_xor.py`.

### 25.2 What was built (steps 1 to 7)

1. *One table of values.* A value's code is the bins at or below it, in
   order, whatever its place; its row is keyed `((ref,), (ref,))` by the
   value alone. The table of values (`value_codes`) is as it was.
2. *Input in 0..1.* `NumericPercepts.intensities` returns its input as it
   is and refuses below 0, above 1 or NaN with `a reader of intensities
   takes numbers from 0 to 1; chart other numbers onto 0..1
   (NumericPercepts.feeling) before they are read`. `feeling` is applied
   by no reader of the numeric stem; its one caller in `bin/` is
   `ScalarPercepts._felt` (step 7).
3. *An image's bank.* `BasicPercepts.value_bank` builds a `PerceptBank`
   (it holds it): one slot per place of the box; a place whose value is
   above zero is an occurrence (a real span, its token the place's
   reading), its `.what` the value's code on two lanes (nothing against),
   its `.where` the place's twelve wires from the registry's grid
   (`registry.band`), its `.when` the moment's wires; a place with nothing
   has an empty span and no activity. The readings (`where`) are each
   place, then the bracket over all. Its rows are named after `Places.meet`
   (`dataclasses.replace`).
4. *The Pi over values.* `BasicPercepts.read_values`: `held_at` on the
   bank's table (part: the occurrence holds the row's value; whole: the
   row's bound holds the occurrence), for = part x whole, against =
   part x (1 - whole), times the occurrence's activity and the row's
   entries on the wires; each reading the greatest over the occurrences it
   holds; a conjunction the first Pi (`self.pi`) over its members'
   readings; then `held`. "Against at a greater value" passed the gates and
   V5, so the other reading was not built. *Agreement with `held_at` on a
   line:* it is `held_at`; V3 asserts on every pair of the test that part
   is `bin(occurrence) >= bin(row)` and whole is `bin(occurrence) <=
   bin(row)`.
5. *One Not.* The promoted rows (values, places, conjunction) go once
   through `owner.order_zero_field`; the lanes set by hand and the separate
   `basic.pi` over conjunction rows in `NumericPercepts.read` go.
6. *A place by its wires.* `Places.meet(owner, bank)` on the image's bank
   counts each place's reading under its wires and nominates a place's row
   (`reserve_place`, with the what kept: least parts, greatest wholes of
   the codes met) and its concept, as for a sentence. The conjunction is
   nominated from the place concepts read for at the bracket
   (`observe_intersection`, as before over the axis concepts). The concept
   for each input place (`NumericPercepts.rows`, made in `__init__`) goes.
   `reverse` reads each place's value from the table by the value's row the
   field holds there (`field.percepts`), exactly; nothing where none.
7. See 25.5.
8. *The way back through the field (review, fix 1).* `reverse` gives a
   place its value from the table only where the field, at that place's
   reading, reads for (for lane above zero) a concept of order zero that
   reads, by an entry that is not a Not (`order_zero_members`, negated
   false), the value's row held there or a place's row whose cone holds
   that place (its wires within the cone, tolerance 1e-6). The place's
   row's kept what is not checked against the value. Nothing elsewhere.
   **On the first presentations no concept reads a value's row or its
   place's row** (a value's row brings none; a place's row is nominated
   on the `conceptMintCount`-th reading), **and the way back gives 0 at
   every place until a place is known**: a place gives its value back once
   its place's row is nominated (and so its concept reads it), as a form is
   a concept once it has recurred. V2 states it (25.10).

### 25.3 The certificates (`test/test_item16_values.py`), as printed

Run alone with `-s` (all 7 passed) and in the bounded run (25.7).

- **V1.** `{'percepts': [[0, 0], [0, -1], [-1, 0]], 'rows_of_the_value': [0]}`
  (0.5 at both places, at one, at the other: one row);
  `{'code': [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]}` (MM_xor is 3 bins a half);
  `{'pairs': 1369, 'values': 37, 'bins': [1, 3, ..., 63]}` (partOf equals
  bin(a) <= bin(b) on every ordered pair of k/37, 64 bins);
  `{'keys': ['((-4,), (-4,))']}`. Asserted also: no `value_coordinates`,
  no `places`, no parameter `place` on `value_code`, `perceive_value`,
  `reserve`, `bin_of`, `_boundary_codes`.
- **V2.** The three refusals, each with the message above; on MM_xor
  `read` refuses 2.0; 37 images of two places (k/37 and its reverse) put
  back exactly (`torch.equal`), first three `[[0.0270, 1.0], [0.0541,
  0.9730], [0.0811, 0.9459]]` both ways; `chart_named` False for
  `NumericPercepts.read`, `.reverse`, `BasicPercepts.read_values`,
  `.value_bank`, `.perceive_value`.
- **V3.** At width 3 (MM_xor), values 0.25, 0.5, 0.75, 1.0 (bins 0, 1, 2,
  2): `0.5 at 0.5 (1,0); 0.5 at 0.75 (0,1); 0.5 at 1.0 (0,1); 0.5 at 0.25
  (0,0); 0.5 at 0.0 (0,0); 0.25 at 0.5 (0,1); 0.75 at 1.0 (1,0)` (same
  bin), and so on for all 20 pairs, each as expected; `numeric_read_uses:
  ['read_values', 'value_bank']`, not `read_bank`.
- **V4.** On MM_xor after `[[1],[1]]` three times in training:
  `{'places': {0: 1, 1: 2}, 'value_row': 0, 'percepts': [[0, 0]]}`, each
  place's kept what equal to the value row's code, each with a concept;
  `place_rows_at_the_two_places: [[1.0, 0.0], [0.0, 1.0]]`. On MM_image
  after one shapes test image three times: `{'lit': 63, 'places': 63,
  'values': [0.0, 1.0]}`, the set of places' addresses (decoded from their
  wires) equal to the lit pixels, each keeping the code of 1.0.
- **V5.** `{'answers': [1.8189894035458565e-11, 1.0, 1.0,
  1.8189894035458565e-11], 'weights': [1.313035249710083,
  1.313035249710083, -2.626070499420166], 'against': [0.0, 0.0, 0.0],
  'promoted_row': 2, 'mse': 1.6543612251060553e-22}`; `{'conjunction': 3,
  'members': (1, 2), 'places': True}`. The stopping gate `test_convergence`
  passed untouched (25.7).
- **V6.** `{'calls': 1, 'rows': [{'values': 1, 'places': 2,
  'conjunctions': 1}]}`: one `model.forward` of the XOR batch, one call.
- **V7.** In the file: MM_image reads a shapes image and puts it back
  exactly (atol 0); `forward_seconds 0.0053, reverse_seconds 0.0002`. The
  tests of the image configurations are in 25.6's table and passed in the
  run. **V8** is the run (25.7): the item 2.5 gate, the seven echo tests,
  `test_reading_pretraining.py`, `test_item16_grid.py`,
  `test_item16_scene.py`, all passed.

### 25.4 V5 and V7, before and after

`tmp/item1-6/values_measure.sh`, each node in its own process
(`/usr/bin/time -l`), `values_before/` on the frozen copy, `values_after/`
on this tree. Wall seconds, greatest resident memory:

| Test | Before | After |
|---|---|---|
| `test_item18_places::test_66` (V5, numeric_xor) | passed, 10.24 s, 959 MB | passed, 10.28 s, 960 MB |
| `test_item18_places::test_65_67` (MM_image) | passed, 10.97 s, 1370 MB | passed, 10.69 s, 1329 MB |
| `test_item16_grid::test_g1[MM_image.xml]` | passed, 10.28 s, 981 MB | passed, 10.20 s, 980 MB |
| `test_item16_scene::test_s1[MM_image.xml]` | passed, 18.46 s, 955 MB | passed, 18.31 s, 951 MB |
| `test_mnist.py` | 1 passed, 2 skipped (no MNIST), 9.47 s, 587 MB | the same, 9.39 s, 587 MB |

`test_65_67`'s own print: before `forward_seconds 0.0687, reverse_seconds
0.0095, forward_peak_bytes 1002422272`; after `0.0130, 0.00025,
988168192`. V5's certificate printed the same answers, weights, against
and mse before and after, to the last digit. `mnist.xml`'s real-data tests
skip on this machine before and after (no `data/mnist_train.csv`).

### 25.5 The value beside the words (step 7): decided

**Kept its own reading and its join** until round two replaces the tie
that reads them. It takes step 1 (`code`, `bare`, `perceive` pass no
place; its percept is a row of the one table). It already applies the
logarithmic chart (`_felt` calls `feeling`) before it reads, so its
numbers are what they were. *Why:* §20.2's named exception says so, and
moving it through the one reader now would change the scalar tests'
fields and the tie round two replaces; the smallest change is the one
that leaves it be.

### 25.6 Tests ported (step 9), old assertion and new

| Test | Old | New |
|---|---|---|
| `test/numeric_xor.py` `certificate` (and so `test_item18_places::test_66`, `test_item17_code::test_49_the_numeric_certificate_keeps_its_bar`, V5) | readout rows `[*model.numeric_percepts.rows, conj]` | the conjunction's two members are places' rows (`in basic.place_rows`); readout rows `[concept of each member, conj]`; the signs asserted as before |
| `test_item18_places::test_level_percepts_have_native_bounds_and_distinct_places` | two distinct percepts for 1 at two places; `basic.places[p] == base + axis`; code `value_code(felt, w, place)`; codes differ | one percept for 1 at both places; its code `value_code(1, w)` in each half; its felt value 1 (now linear) |
| `test_item18_perceptual_scalar::test_scalar_object_collects_value_bounded_basic_percepts` | code equals `value_code(value, w, basic.places[p])` | code equals `value_code(value, w)` |
| `test_item18_fields::test_numeric_scale_order_and_level_code` | `perceive_value(v, place=0)`; `value_code(v, 4, 0)` | `perceive_value(v)`; `value_code(v, 4)` |
| `test_item17_code::test_49_collections_of_values_are_told_apart` | `place = 7` passed to `perceive_value`, `value_code`, `bin_of` | no place |
| `test_item17_code::test_49_the_numeric_certificate_keeps_its_bar` | features of `model.numeric_percepts.rows` differ | features of the two places' rows' concepts (the conjunction's members) differ |

Nothing else was weakened, skipped or marked; no seed; no gate touched.
`test_item17_symbols::test_54a` (NumericPercepts.read names no code
attribute) passes as it is.

**Configurations (step 2).** None changed. The numeric data sets: `xor`
(`MM_xor`, `MM_xor_fixture`, `XOR_exact`, the default of `model.xml`):
0 and 1. `images`/`shapes` (`MM_image`): 0 and 1 (binary, float32).
`mnist` (`mnist.xml`, `loadMNist`): pixels over 255 then standardized by
the training mean and std, about -0.42 to 2.8; refused by the reader
before (negative numbers, by `feeling`) and after; its data is absent here.
The image corpus `mnist` (threshold 128) is 0 and 1 and no configuration
uses it. No declared value of a configuration turns on the felt chart.

### 25.7 The run of step 10

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=... .venv/bin/python
test/test_report.py` with both stopping gates and the 25 files of
`tmp/item1-6/values_run_files.txt` (the echo, the reading lesson, grid,
scene, values, every changed test file, and every test file naming
NumericPercepts, ScalarPercepts, perceive_value, value_code, numeric_xor or
MM_image).

**`Result: test_failure, exit 1; 284/284 cases completed.`**
`output/tests/20261124-052225-ce3b4d/result.json`; 1824 s; greatest
aggregate memory 9.56 GB over its workers. 248 passed, 31 skipped, 5
failed. **Both stopping gates passed.** Of the five failures, four are
known: `test_item17_symbols::test_54bd` and
`test_percept_collection_binding::...[2.0-True]` fail in the sweep;
`test_explicit_dimensions::...test_piecewise_overall_at_least_50_pct` and
`test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`
are on the prompt's list of chance. The fifth,
`test_item17_symbols::test_cert_c_an_operator_class_reads_its_arguments_through_symbols`,
passes in the sweep: **it fails by chance on the frozen copy too** —
run alone four times on each tree: frozen copy failed 2 of 4, this tree
failed 2 of 4 (`AssertionError: [12x]` on a symbol's members, a text
lesson that never builds the numeric stem). Not repaired. Every other
selected test has the sweep's outcome (the new files' tests are absent
from the sweep and passed).

### 25.8 What in §20 was wrong or could not be built as written, and what was built

1. *"with the row's entries on the wires (place_factor) as for a form"*
   (step 4) does not tell XOR's two places apart. `place_factor` at a
   neighbouring place reads 0.9968 on MM_xor (0.9955 on MM_image): the
   ladder's wires move little in one step (G5's 0.0126). Built with it, the
   certificate's fit gave mse 1.67e-10 and answers
   `[-7.4e-08, 1.0000181, 0.9999816, 4.5e-08]`, missing the bar of 1e-6
   on the answers by 1.8e-5 (`tmp/item1-6/values_fit.py`). **Built:**
   `BasicPercepts.place_bounds`, a row's where entries read by their
   bounds, 0 or 1, as `read_points` reads a value's: one where the
   occurrence's wires lie within the entries on every wire (each lane at
   least its entry, tolerance 1e-6), nothing elsewhere; one everywhere for
   a row with no where entries (a value's row); its when entries read by
   `field_factor` as before. A place whose cone is one place holds that
   place alone. Then the certificate is as before to the last digit. It is
   used by the Pi over values only; text's `read_bank` keeps
   `place_factor`. It passes no gradient to the where entries. *Marked.*
2. *A conjunction over places' rows is read at a reading, not at an
   occurrence.* `read_bank` takes a conjunction's Pi at each attended
   percept; two places' rows are never both held at one pixel, so on the
   image's bank the Pi is taken over the members' readings, where the
   bracket holds both (as the numeric stem did before). *Marked.*
3. *A conjunction with no parts.* `reserve` refused a row with no parts;
   a conjunction over places' rows has none (places have no parts), so
   `reserve` takes one when it has members. *Marked.*
4. *The bracket in the bank.* A `PerceptBank` gives each occurrence one
   reading (`token`); the bracket, which holds every place, is read by
   `read_values` as the greatest over the occurrences its span holds, not
   by a token. *Marked.*
5. *No look.* The image's bank is read whole by `read_values`;
   `attended` (and `attendedPercepts`, 256, below MM_image's 784 places) is
   not applied to it: the numeric stem's read is not a look.
6. *Timing of the conjunction.* The places' rows are nominated on the
   second forward of the XOR batch (two streams hold each place), and the
   conjunction is promoted after the read in which its count is reached,
   so it is read from the next forward: in `numeric_xor.fit` it appears
   during the first evaluations of LBFGS (still in training mode), where
   before it was read on the third forward. The certificate holds.
7. *A saved state.* The extra state no longer has `places` and its
   loading no longer reads it; nothing is kept for the older form. A state
   of an older numeric model, whose rows were one per place and level,
   would load with those rows collapsed into one per value (the index
   shrinks, as it already does where text rows share a key). A check that
   refused such a state was built and removed: it would refuse text states
   whose rows share a key, which exist. *Marked.*
8. *The tolerance of `place_bounds`* is 1e-6, as `read_points` has. A
   place's row's entries are copies of the grid's float32 wires at the
   places met, so its own place is held exactly; another place is told
   apart if one of its twelve wires differs from the entry by more than
   1e-6. Neighbouring places differ by about 3e-3 a wire (MM_xor's and
   MM_image's ladders), so the wires need float32 or better; in bfloat16
   (about 4e-3 near 1) neighbours could be read as one place, and a place
   whose wires were rounded apart from its entries could miss itself.
9. *`when_encoding` (review, fix 2).* `install_where_registry` sets
   `model.when_encoding` on every model, and the numeric stem needs the
   registry, so `NumericPercepts.read` reads it directly and
   `value_bank` takes it as required (no branch for none).
10. *V7 and V8 in the file.* The tests of the image configurations and V8's
   tests are certified by the run (25.7), not re-run from inside
   `test_item16_values.py`; V7's measure of the image's read and way back
   is in the file.

### 25.9 Found on the way

- `test_item17_symbols::test_cert_c_...` is flaky on the unchanged source
  (25.7).
- `mnist.xml` cannot be read by the numeric stem before or after: its
  loader (`data.loadMNist`) divides the pixels by 255 and then
  standardizes them by the training mean and std, giving numbers outside
  0..1 (about -0.42 to 2.8), which the reader refuses.

### 25.10 The review's two fixes, and where the first one stops

**Fix 2** is made (25.8, 9). **Fix 1** is made as 25.2, 8 says, and V2
now asserts: on a fresh MM_image, the first presentation of an image of
the 37 values of the grid gives back 0 at every place
(`first_presentation_greatest 0.0`); after it has recurred
`conceptMintCount` times (763 places' rows) every place is given back
exactly; and with the field's evidence set to zero, nothing at every place
(`silent_greatest 0.0`). V7 now presents the image as often as nominates
a form before it puts it back. The XOR certificate and both gates pass
with the way back through the field: the certificate's places' rows exist
by the time it reconstructs, its answers, weights and mse are those of
25.3.

Two ports for 0..1 alone (step 2, not fix 1), each an input above 1:
`test_item18_paths::test_numeric_understanding_owns_evidence_across_next_forward`
(the second forward `values * 2.` -> `values * .5`) and
`test_understanding_reconstruct::test_second_forward_does_not_mutate_the_first_understanding`
(`inputs + 0.5` -> `inputs * 0.5`, which was 1.5).

**Run** (both gates and the files the review named, with every test file
that calls `reverseReconstruct` or `numeric_percepts.reverse` on a numeric
model: `tmp/item1-6/values_run2_files.txt`):
`Result: test_failure, exit 1; 169/169 cases completed.`,
`output/tests/20261124-062149-a53e9d/result.json`. 161 passed, 6 failed,
1 xfailed, 1 skipped. **Both stopping gates passed.** Known:
`test_explicit_dimensions::...test_piecewise_overall_at_least_50_pct`
(chance), `test_item17_symbols::test_54bd` (fails in the sweep). **Four
pass in the sweep and fail here:**

| Test | Where it fails | Cause |
|---|---|---|
| `test_item18_places::test_65_67_image_uses_the_same_reader_and_recovers_every_pixel` | `assert_close(rebuilt, sample)`: 63 of 784 places are 0 | one forward and one `understand` of an image: no concept reads it yet, so the way back gives nothing (fix 1) |
| `test_item18_paths::test_numeric_understanding_owns_evidence_across_next_forward` | `assert_close(restored, values)`: 4 of 8 places are 0 | the first `understand` is the first presentation (fix 1) |
| `test_output_synthesis::test_numeric_field_inverse_scores_identity_and_a_changed_target` | `float(cost) == 0.` is 0.25 | the first `understand`, put back as nothing, against its input (fix 1) |
| `test_mm_xor::TestMMXorConvergence::test_forward_keeps_continuous_symbols` | `symbols.std() > 1e-6` is 0.0 | a fresh model's first forward reads no concept at all (no concept for each input place, no place's row yet): step 6, not fix 1. **Missed by the first return:** its run named only the gate of `test_mm_xor.py`, not the file |

**The way out that was tried and not kept.** A value's row bringing a
concept of order zero when it is first perceived (as a nominated row does)
makes the first presentations read and put back, and the three of fix 1
pass, V1 to V7 too (code kept as
`/tmp/item1-6-2026-11-24/NumericPercepts.with_value_concepts.py`). But
`test_concept_output::test_native_cli_curriculum_learns_output_and_keeps_located_inverse[0..2]`
then fails at `assert 1e-6 < without_intersection` (3.09e-14): the
value's concept reads at the bracket "some place holds 1", an OR, and
XOR = 2 OR - x0 - x1 is linear in it and the two places', so XOR is
learned without the conjunction. With it the certificate still passed,
but the places' readout weights fell to 1.19e-8 (the conjunction's
-1.313): the readout used the value's concept and the conjunction. A
model whose order zero is small (MM_xor has 8 seats) also runs out of
seats on graded images (one concept per value). So it was removed.

**Decided (the coordinator): option 4.** No concept for a value's row.
The concept for each input place was removed by decision (§20.2, point 6),
so "an image is put back, or read as symbols, at its first presentation"
is one of the things this step makes false, and step 9 covers it: such a
test asserts what is true now. An image is put back once its places are
known, as a form is a concept once it has recurred. Option 1 (a value's
row brings its own concept) is round two's subject, objects made as words
are; what was found about it above (an OR, and XOR learned without the
conjunction) goes to Alec with round two. Option 2 would treat an image's
places unlike a sentence's; option 3 would make the way back empty of
meaning again.

**The six tests ported in this round.** Every assertion and tolerance is
as it was; each presentation count is read from the model
(`concept_mint_count`), never a literal.

| Test | Old | New |
|---|---|---|
| `test_item18_places::test_65_67_image_uses_the_same_reader_and_recovers_every_pixel` | one `forward` of the image, then `understand` and put back | the image is first presented `concept_mint_count` times (before the `read_field` counter is installed, so `len(calls) == 2` is as it was), then as before |
| `test_item18_paths::test_numeric_understanding_owns_evidence_across_next_forward` | `understand(values)`; `forward(values * 2.)`; put back the first | `values` presented `concept_mint_count` times first; the second forward `values * .5` (2.0 lies outside 0..1) |
| `test_output_synthesis::test_numeric_field_inverse_scores_identity_and_a_changed_target` | `understand(x)` of a fresh model, then the costs | `x` presented `concept_mint_count` times first, inside the test |
| `test_mm_xor::TestMMXorConvergence::test_forward_keeps_continuous_symbols` | one `forward` of a fresh model, symbols' std above 1e-6 | the batch presented `concept_mint_count` times first, inside the test's own body; nothing the gate uses (`_fresh_model`, `_fit_xor`, `test_convergence`) is touched |
| `test_understanding_reconstruct::test_second_forward_does_not_mutate_the_first_understanding` | second `understand(inputs + 0.5)` (1.5 outside 0..1) | `understand(inputs * 0.5)` |
| `test_item16_values` V2, V7 (new in this step) | V2: put back on the first presentation; V7: one forward | V2 asserts 0 on the first presentation, exact after recurrence, 0 with the evidence zeroed; V7 presents first |

No assertion was added to an old test; the first-presentation behaviour
is stated in V2 alone.

**The run after the ports.** Both stopping gates; `test_mm_xor.py` whole
and the files the review named; and every file of `test/` that builds a
numeric model (names MM_xor, MM_xor_fixture, XOR_exact, MM_image,
numeric_xor, reverseReconstruct or numeric_percepts) not run in the two
runs before: 42 more files (`tmp/item1-6/values_run4_files.txt`).
**`Result: test_failure, exit 1; 481/481 cases completed.`**,
`output/tests/20261124-064521-840734/result.json`. 434 passed, 44 skipped,
3 failed. **Both stopping gates passed.** The three:
`test_item17_symbols::test_54bd` (fails in the sweep);
`test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`
(on the list of chance); `test_item17_symbols::test_cert_c_...` (fails by
chance on the frozen copy too, 2 of 4, 25.7). Every other test has the
sweep's outcome.

### 25.11 Review of step B (Claude, 2026-11-24)

**Accepted, after two rounds of correction.** Claude read the whole change
and ran, with the bounded runner (run `20261124-065405-b90ced`),
`test_mm_xor.py` whole with its stopping gate, the other stopping gate,
`test_item16_values.py`, `test_item16_scene.py`, `test_item16_grid.py`,
`test_item18_places.py`, `test_item18_paths.py`, `test_output_synthesis.py`,
`test_understanding_reconstruct.py`, `test_concept_output.py`,
`test_item17_code.py`, `test_item18_fields.py`,
`test_item18_perceptual_scalar.py`, `test_item18_scalar.py` and
`test_item17_echo.py`: **131 of 131 passed**, both gates among them. In the
gate's file the one change is inside the body of
`test_forward_keeps_continuous_symbols`; the gate, `_fresh_model` and
`_fit_xor` are as they were.

*What Claude sent back.*

1. *The way back did not go through the field.* As first returned,
   `NumericPercepts.reverse` gave each place its value from what the field
   had stored at input, whatever the field's evidence, so "put back
   exactly" would have held of a field that read nothing. It now gives a
   value only where a concept reads for the value's row or a place's row
   that holds the place, and V2 shows a field with no evidence giving
   nothing.
2. *A defensive read* of the model's clock encoding, now read directly.

*What the correction uncovered, and what was decided.* With the way back
through the field, a fresh numeric model puts nothing back, and reads no
concept, for its first presentations: the concept for each input place
went by design (20.2, point 6), a value's row brings no concept, and a
place's row is nominated on the count that nominates a form. Four tests
that asserted otherwise now present their input that many times first,
with every assertion and tolerance as it was (25.10). Claude chose this
over a concept for a value's row, over nominating an image's places on
their first reading, and over a way back that needs no concept.

*For Alec, with round two.* The implementer tried a concept for a value's
row and withdrew it (25.10). With it, "1 somewhere in the image" is read
at the bracket, which is an OR of the two places, and XOR = 2 OR - x0 - x1
is then learned with no conjunction:
`test_concept_output::test_native_cli_curriculum_learns_output_and_keeps_located_inverse`
asserts an error above 1e-6 "without the intersection" and got 3e-14.
Round two gives a value a concept of its own, so that certificate's
premise ends there, and whether XOR is still to need its conjunction is
his to say.

*On the departures of 25.8.*

1. *The place of a value is read by bounds, 0 or 1* (`place_bounds`), not
   by the soft match text uses, which reads .997 at a neighbouring place.
   Accepted: it is the reading by bounds that a value has (rule 55),
   carried to the wires, and it is of the value space's Pi alone. It gives
   no gradient to a place's entries, and it needs the wires at single
   precision or better: at bfloat16 two neighbouring places may be one.
   That is to be measured before a model of intensities is run at that
   precision, which he named (22.1, question 3).
2. *The Pi over values is `held_at` on a line,* under a name of its own
   (`read_values`). He asked for "a different PI operation on that space";
   what is different is the reading of the place and that a form's row is
   never offered a pixel, and what is shared is the test of part and whole,
   which `doc/Mereology.md` calls "the exact same mechanism" for a scalar.
   Put to him.
3. *An image's bank is read whole,* with no cap of attended percepts.
   Accepted for the numeric stem, whose read is not a look. A look that
   takes a felt thing is round three's.
4. *An older numeric state loads with its rows merged.* None is committed;
   a numeric model trained before this step is trained again.
5. *The value beside the words keeps its reading, its join and the
   logarithmic chart* (his "implementer's decision"). It is the one second
   path left, until round two.

*Found on the way, not this item's:* `mnist.xml`'s loader standardises its
pixels to about -0.42 to 2.8, which no reader of intensities takes, before
or after; `test_item17_symbols::test_cert_c_...` fails by chance on the
unchanged source, 2 of 4.

*Standing documents written by Claude with this step:* `doc/Spaces.md`
("Level Sets", rewritten for the number modality), `doc/Architecture.md`,
`doc/Params.md`, `doc/Philosophy.md`, `doc/DevelopmentHistory.md`.

## 26. The corpus's fourth revision: counting asked in words (Claude, 2026-11-24; the text for the implementer is 26.2)

*Rests on 22.4, question 11: "we need a version that asks a question about
the image in the text modality."*

### 26.1 What changes in §14

Category 27 has two forms.

| Form | Entry | Answer |
|---|---|---|
| The count, as it is | an image of one to nine things, with no text; no question is written | the counting words, one for each thing, in order: `one two three` |
| The number, asked | the same kind of image with the text `how many are here ?` | the one counting word of the number of things: `three` |

- The first form's rows stay as they are frozen, key for key and byte for
  byte, if the way the rows are drawn allows it; if it does not, the
  return says which rows moved and why.
- The second form has as many rows again in each split as the first (100,
  20, 20 and 20), drawn afresh, under the same rule of what is held out
  (each set of values whose sum is a multiple of 4 is kept for the
  generalization split).
- `how` and `many` join the vocabulary if they are not in it; `are` and
  `here` are. The words live in the recipes, as every word does.
- Check 10 of §14 becomes: in category 27 the answer of a row with no text
  is the first n counting words for the n things of its image, and the
  answer of a row that asks `how many are here ?` is the nth counting word
  alone.
- The second form's answer is one word, so §14's "an answer is one word,
  except in category 27" now excepts the first form only.

### 26.2 Text for the implementer

~~~
Item 1.6, the corpus's fourth revision: counting asked in words. Read
§26.1 of doc/plans/2026-11-18-item1-6-felt-math-world.md, then §14 of the
same document (the frames' table, the checks, §14.8) and §15.11 and
§15.12 for how the last revision was made and reviewed.

1. data/felt/recipes/category-27-counting.json gains the second form of
   §26.1 and, in data/felt/recipes/cast.json, the words it needs. No word
   is written in bin/felt_world_corpus.py.
2. bin/felt_world_corpus.py: the predicate, `verify` and whatever composes
   category 27 take the second form. Check 10 as §26.1 words it.
3. Freeze again with the documented command (seed 20261118). Say in the
   return whether the first form's 160 rows are the same bytes as before,
   and whether any other category's file changed at all (none should);
   compare the sha256 of every file with the manifest as it was.
4. test/test_felt_corpus.py: the counting certificate holds both forms.
   test/test_item16_scene.py reads every document: if it asserts a count
   of documents, rows, images or pixels that this revision changes, give
   it the new count and list it. No other assertion changes.
5. Run with plain pytest, not the bounded runner (another change may be
   under way in doc/): BASICMODEL_DEVICE=cpu PYTHONPATH=bin:test
   .venv/bin/python -m pytest test/test_felt_corpus.py
   test/test_item16_scene.py test/test_game_corpus.py -q -p
   no:cacheprovider.
6. No standing doc and nothing under doc/: Claude writes them at review.
   Nothing removed is deleted but moved to a dated folder under /tmp.
7. Return as a file, tmp/item1-6/section27.md, headed "## 27. The corpus's
   fourth revision: the return": files changed and why; the counts of
   category 27 by split and form; three documents of the new form, whole;
   the hashes before and after; what in §26.1 could not be built as
   written and what you built.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 28. The rest of the item, designed (Claude, 2026-11-24)

*A design at the level of decisions and certificates. Each step's text is
written when its turn comes, from the source as it then is. Where a point
is Claude's proposal and not Alec's word it says so. It stands over 19.3's
table.*

### 28.1 What step B showed

An image cannot be put back with its values from the concepts that exist.
Only a place's row brings a concept, and its cone keeps the least and the
greatest value met at the place (`Places.meet`); in the corpus any value
lies at any place, so a place's concept comes to say "something here". The
numeric stem's way back knows the value from the input (`field.percepts`),
not from a concept. **So a value needs a concept of its own before an image
is put back**, and step C is divided by it.

### 28.2 The order

| Step | Builds | Measured from input to output |
|---|---|---|
| C1 | The scene's bank in its sentence's field at the opening: `InputSpace._scenes` to `value_bank` in the `scene` box, places met, `read_values`, its rows and the sentence's through `order_zero_field` in one call; its readings joined to the field as columns | A sentence reads as it did with and without an image (both gates, echo); one call holds both kinds of rows; an image's places have their concepts in the field; `null` adds no column, `[]` adds columns that read nothing |
| D1 | *Objects over values* (28.3) | Category 1's images alone: after the lessons there are nine objects, one for each value, and a reserved step (25d+1, 25d-1) falls in its value's object and in no other |
| C2 | The image put back and charged, through the objects: an object's concept, the Sigma, the first Pi's kept percept, the place by its wires within the box; the cost place by place over the box | F1, F2, F11, F12 of §7: an image alone is put back at varied steps and at places not met; a sentence and its image are both put back; `null` is charged nothing; `[]` is charged for what is put back |
| D2 | *Nouns and the name* (28.4), with `createMetaConcepts` | The count of units across modality at order one; each of naming's four stages (19.5) |
| E | *Areas* (28.5) | The categories about size and shape |
| F | *Intensities out* (§24, 1) | An image asked for in words is given, place by place |
| G | *Relations of place* (28.6); what turns on the order of two words waits for item 1.4 | Category 14's contrast sets, per answer, on pairs held out |
| H | The closing (28.7) | The item's one default sweep against the baseline |

### 28.3 D1: objects over values (Claude's proposal on 22.2, question 6)

1. *A value's row is nominated on recurrence and brings its concept,* as a
   form's does (rule 20); it is not promoted at first sight. This is his
   "made alike, each from its own occurrences".
2. *An object is a cone over values:* its lower bound the greatest part its
   occurrences all hold (the least value met), its upper bound the least
   whole that holds them all (the greatest), as `collection_code` and a
   place's cone already compute, and as he says of a character's percept.
3. *What gathers the occurrences,* with no word: **proposed**, an
   occurrence joins the cone it lies within or touches (the next bin), and
   starts one of its own otherwise; two cones that come to touch are one.
   In the corpus a value's steps span two or three bins and two values
   lie six bins apart, so nine cones form and stay apart; a reserved step
   lies inside its value's cone. The measure of D1 is whether this holds.
   *His to accept or replace.*
4. *XOR.* With a concept for the value, "1 somewhere" is read at the
   bracket and XOR is learned without the conjunction (25.11). The
   assertion of `test_concept_output` that it is not becomes: with
   promotion off there is no conjunction's row. *His to confirm.*

### 28.4 D2: nouns and the name

1. *Nouns.* The word's concept and the object's each have their symbol in
   the Pi of order one, where place and time are gone (22.0).
2. *The meta concept* is a Sigma of order one over the two symbols, made by
   MTM when the two are present together on the count that nominates
   (22.1, question 1). **Proposed:** it is nominated where today's
   promotion would make a Pi row of a conjunction (`FamiliarTier`), in the
   one case that its members' percepts lie in two ranges of the registry;
   within one modality the Pi row stands, a whole of parts. Of several
   things present with a word, the pair that recurs wins by its count.
3. *`createMetaConcepts`* in `data/model.xml`, true: `interpret` makes meta
   concepts and blank objects as today. False, in the felt world's
   configurations: neither is made directly, and `interpret` of a word
   gives the other member of a unit across modality that holds its symbol,
   or the word itself where there is none.
4. *Naming, the test,* in four stages (19.5): each known alone; the unit
   exists; one raises the other when met alone, by the Sigma's way down to
   its other member; `interpret` gives the object. Measured on categories
   1 and 2 at the reserved step and places, alone and with others present.
5. *The assessment tool* counts units across modality by order.
6. *The value beside the words* goes through the one reader here, and its
   tie (`PerceptCollections.observe`, `WordTies`) with it.

### 28.5 E: areas

A percept on the part tower covers a range of places and keeps the
greatest part all its pixels hold; on the whole tower a property pervades
its range (19.1). **Proposed:** ranges grow by touching, as 28.3's cones
do on the line of values: pixels of one object that touch are one thing,
and the thing's percept has the box of its places as its `.where`. This is
"what knits touching places into one thing" (C.5), and the categories
about size and shape measure it.

### 28.6 G: relations of place (Claude's proposal on 22.3, question 8)

An axis of place is a sorted line, as the line of values is. "Above" is
less-than on y between two things' places, read by bounds as a value is
read; "left of", "between", "next to" likewise on their axes. So an
arrangement is read by the one sort the model has (§23), by a relation of
two places, and is no percept of its own. The words for it are learned as
any words are. Its certificate is asked after item 1.4.

### 28.7 H: the closing

`mnist.xml`'s loader gives intensities in 0..1. The 0-or-1 match of a place
is measured at bfloat16. The standing documents say what the source does.
`main` is merged in. The default sweep is run once against the baseline,
and the slow tier once at submission.

### 28.8 What waits on Alec

28.3, points 3 and 4. Whether the Pi over values may share `held_at`
(25.11). Nothing else: the steps go in the order of 28.2.

## 27. The corpus's fourth revision: the return

The work of §26.2. Frozen again with the documented command and seed
(`freeze --seed 20261118 --recipes data/felt/recipes --out data/felt`); a
second freeze into `/tmp/item1-6-2026-11-24/refreeze` writes the same 28
files byte for byte. Run as step 5 says (plain pytest, `-p no:cacheprovider`):
`test/test_felt_corpus.py`, `test/test_item16_scene.py`,
`test/test_game_corpus.py`: **38 passed**.

### 27.1 Files changed, and why

| File | Change |
|---|---|
| `data/felt/recipes/category-27-counting.json` | Gains `questions`: one frame `q01`, text `how many are here ?`, predicate `count`, answer type `number`. `splits` doubled to 200, 40, 40, 40: the file's counts, which each form shares equally. |
| `data/felt/recipes/cast.json` | `how` and `many` join `vocabulary`; `many` joins `english.plural_nouns` (27.5, decision 3). |
| `bin/felt_world_corpus.py` | `answer_word`: the type `number` is the nth counting word, `counting[n - 1]`. `Generator.counting` now draws each form (the recipe's statements, then its questions) for every split, an equal share each, every split of one form before any of the next; the body that drew the rows is unchanged, moved to `count_form(frame, split, want)`. `self.counted` holds the rows between the per-split calls. No word is written in the Python. `p_count` and `verify` needed no change: `verify` already evaluates any frame's `count` record on the image and compares the answer with `answer_word` of the frame's answer type. |
| `data/felt/category-27-counting.jsonl`, `data/felt/manifest.json` | Frozen again. |
| `test/test_felt_corpus.py` | Check 10 as §26.1 words it: a row with no text answers the first n counting words; a row with the recipe's question answers the nth word alone; no other text; each split holds as many rows of each form. No other assertion changed. |

`test/test_item16_scene.py` asserts no count this revision changes (its only
count is 27 files); it is not changed. Its S3 now prints 5760 documents,
28493 rows, 27812 images (22 empty), 103799 pixels.

### 27.2 Category 27 by split and form

| Split | `s01` the count (no text) | `q01` `how many are here ?` | All |
|---|---|---|---|
| train | 100 | 100 | 200 |
| validation | 20 | 20 | 40 |
| test | 20 | 20 | 40 |
| generalization | 20 | 20 | 40 |

The second form's images are drawn afresh: none of its 160 images is an
image of the first form, and none repeats. Held out as before: every
generalization row's values sum to a multiple of 4, no other row's do, and
train holds every count 1 to 9 (check 6 passes). The second form's answers
over all splits: three 23, five 22, nine 22, six 21, seven 18, one 16, four
14, two 13, eight 11.

### 27.3 The first form, and the other categories

- **The first form's 160 rows are the same bytes as before**, key for key and
  in the same order: filtered to `variant == "s01"`, the new file's lines equal
  the old file's lines exactly. Each split's first-form rows come first, keys
  `-0000` to `-0099` (train) and `-0000` to `-0019` (the others); the second
  form follows at `-0100` to `-0199` and `-0020` to `-0039`.
- **No other category's file changed**: the sha256 of each of the other 26
  row files equals its hash in the manifest as it was, and their manifest
  entries are unchanged in every field. In the manifest, only the entry of
  category 27, `cast_sha256` and `vocabulary` differ; `longest_text` stays
  10 words, 48 bytes.

### 27.4 Hashes, before and after

| File | Before | After |
|---|---|---|
| `category-27-counting.jsonl` | `577e5e9fe26747925724032d880136c09f06346faff315142ef048973ce8fd6c` | `043c62867b2f770a46623d934aaf7ed19aab5a56af7416455c23e5928e9147f3` |
| `manifest.json` | `3a3b27ac1fd99e4af4995646680d48b68d2914a674cefeb2c5820493b525287a` | `13bea151138c8b09faab454bddbe521447a8147ca812d67d5c5747bf21f6eb6a` |
| `recipes/cast.json` | `4048cd498b2dbd43fafd7f1b466c001ce593174e33d6a6f8bc18fe15883cdd65` | `4467b7789dbf54aec38182d490ff17fc949b960017c10b3961901dfea1b326d8` |
| `recipes/category-27-counting.json` | `f6791ebfd0f0db5418be4b4a20ef9ff15d856d4ac059112501511937b1627167` | `14a2c1912f0260d96822d913bf7aaabc3417f29f1f9d4eb2f0a80ae76902ea9b` |
| the other 26 `category-*.jsonl` | as in the manifest before | unchanged |

The files before are kept in `/tmp/item1-6-2026-11-24/corpus_before/` (the
manifest and category 27) and `/tmp/item1-6-2026-11-24/felt_full_before/`
(all of `data/felt/`).

### 27.5 Three documents of the new form, whole

~~~
{"answer":"three","category":27,"entries":[{"scene":[[0,5,0,225],[1,0,0,123],[3,7,0,101]],"text":"how many are here ?"}],"inference":null,"key":"felt-27-train-0100","meaning":[{"args":[],"cast":[],"entry":0,"frame":"q01","predicate":"count","slots":{},"truth":true,"value":3}],"name":"counting","operators":[],"probe":null,"question":0,"set":null,"split":"train","variant":"q01"}
.5......
........
........
........
........
9.......
........
...4....
~~~

~~~
{"answer":"seven","category":27,"entries":[{"scene":[[1,0,0,102],[2,4,0,222],[4,3,0,151],[4,4,0,51],[5,2,0,175],[5,4,0,27],[6,4,0,201]],"text":"how many are here ?"}],"inference":null,"key":"felt-27-validation-0020","meaning":[{"args":[],"cast":[],"entry":0,"frame":"q01","predicate":"count","slots":{},"truth":true,"value":7}],"name":"counting","operators":[],"probe":null,"question":0,"set":null,"split":"validation","variant":"q01"}
.4......
........
.....7..
....6...
..9.218.
........
........
........
~~~

~~~
{"answer":"four","category":27,"entries":[{"scene":[[1,1,0,123],[3,0,0,147],[3,5,0,50],[7,0,0,175]],"text":"how many are here ?"}],"inference":null,"key":"felt-27-generalization-0020","meaning":[{"args":[],"cast":[],"entry":0,"frame":"q01","predicate":"count","slots":{},"truth":true,"value":4}],"name":"counting","operators":[],"probe":null,"question":0,"set":null,"split":"generalization","variant":"q01"}
...6...7
.5......
........
........
........
...2....
........
........
~~~

(The grids are the plane, row y down, column x across, each thing shown by
its value; the third's values sum to 20, a multiple of 4, as its split
requires.)

### 27.6 What could not be built as written, and what was decided

Everything in §26.1 was built as written. Decided by the implementer:

1. *How the first form stays byte for byte.* Category 27 draws from one
   stream (the recipe's seed, `20261118 * 100 + 27`, as before). Drawing the
   second form after the first within each split would have moved every
   later split's first-form draws, so every split of the first form is drawn
   before any row of the second; the second form is drawn afresh from the same
   stream after them. No other seed is used.
2. *The second form is a question frame* (`questions`, `q01`), with a new
   answer type `number`, the nth counting word read from `cast.json`'s
   `counting`. The first form stays the statement `s01` with its answer type
   `count`. The meaning record of a `q01` row is the `count` predicate's, as
   for `s01`, with `value` n and `truth` true.
3. *`many` as a plural subject.* Check 5 requires `are` to have several as its
   subject, and reads what is plural from `english.plural_nouns` in
   `cast.json`. The subject of `are` in `how many are here ?` is `how many`,
   so `many` (a plural pronoun here) is listed there; without it check 5
   rejects every row of the new form. This is a word of the cast's English,
   not a change to the test.
4. *The recipe's `splits`* are the file's counts (200, 40, 40, 40), divided
   equally among the forms, so that the manifest's `short` keeps comparing
   what a split holds with what the recipe asks.

Nothing removed: no file was deleted.

### 27.9 Review of the fourth revision (Claude, 2026-11-24)

**Accepted.** Claude read the change and ran `test_felt_corpus.py` and
`test_item16_scene.py` (31 passed). Category 27 holds 320 rows, 160 of each
form; the first form's rows are the bytes they were and no other
category's file changed. One thing decided by the implementer: `many` is
among the cast's plural nouns, so that `how many are here ?` passes the
check that `are` has a plural subject. Accepted: the word is a recipe's
datum and no assertion changed.


## 29. Alec on 28.8's three points (2026-11-24), and what they change

### 29.1 What makes an object of values (28.3, point 3): not adjacency

> Lets take an example of the binomial coefficients. [1,3,3,1] has as parts
> [1,2,1] in two places, and [1,1] in three places. that is the part
> structure that exists over the part/particle tower; the whole/property
> tower could have the property [4] (it could also have the property "odd
> numbers", but we keep it simple for now).
> Those parts and wholes might combine into a basic percept: the formation
> for that is just as it is for letter percepts.

**Decided.** Claude's rule of touching bins is withdrawn. An object of
intensities is a basic percept knit of parts and wholes, formed as a
letter's percept is and nominated on recurrence.

*Claude's reading of the example.* A pattern is a part of an image at a
place when the image holds it there, value by value (each value of the
part at or below the image's): [1,2,1] lies within [1,3,3,1] at its first
place and at its second, and [1,1] at three places. That is improper
parthood on values (19.1) carried over places, and the part tower holds
such parts at growing widths. A whole is a property that holds of every
place of its range (19.1): [4] is read as a bound no value exceeds, and
"odd numbers" as a kind every value is of.

*What it changes.* 28.3 (D1) and 28.5 (E) are one step, and the towers
come first: parts found by containment at places, at growing widths;
properties that pervade a range; a basic percept from both, as
`doc/Mereology.md` has it for a form (the greatest of its parts' codes
below, the least of its wholes' above). The steps of one value are then
one object because they hold the same parts and lie under the same
property, and for no other reason.

### 29.2 XOR (28.3, point 4): the certificate is kept, by a setting

> We need to ensure that the two values 1 and 0 are presented seperately,
> and not chunked into a single percept or concept. We should be able to
> limit the .where specification,: so just as we force word classes to stop
> at spaces (we look for a whole that has a run of letters), we would need
> to limit our concepts to a .where width of one (presumably a model setting
> that is not very useful for non-xor cases, but lets do it as a proof).

**Decided.** The assertion that XOR is not learned without its conjunction
stays. A model setting limits the `.where` width of what a concept of a
value is read over; at one, a value is read at single places and never
over the bracket, so "1 somewhere" is no concept, and only the conjunction
over two places' rows spans the two. Its default in `data/model.xml` sets
no limit; the XOR configurations set it to one. It is the limit a run of
letters sets for a word, on the where.

### 29.3 The Pi over values (25.11): its own, and from positive evidence

> This seems to mess with innards... If this is done for reconstruction, I
> would think that we can reconstitute from positive evidence using a "most
> likely composition" logic that I will leave you to derive.

**Decided.** The Pi over values does not call the first Pi over forms'
test (`held_at`); it is an operation of its own. And the way back does not
need what tells against.

**Derived (Claude).**

1. *The Pi over values is containment.* A part reads for at an occurrence
   that holds it: a row of value c at an occurrence of value v where c is
   at or below v. Every lesser value so reads for a greater one, which is
   improper parthood, and nothing tells against. A property reads for
   where it holds: an upper bound u at v where v is at or below u. A basic
   percept that knits a part and a whole reads for where both do, which is
   an interval.
2. *The most likely composition.* Zero is no presence, and a code is
   "built up from all zeros by its parts". So of all the images that hold
   every part read for at its place, the most likely is the least: at each
   place the greatest part read for there, and nothing where none is. A
   property read for over a range bounds it from above; where the greatest
   part exceeds the least bound the evidence disagrees, the lesser is put
   back and the difference is charged. With evidence in degrees, a part
   counts by its degree.
3. *What it says of his example.* From [1,2,1] at two places and [1,1] at
   three the composition is [1,2,2,1]: the least image that holds them.
   The threes come back only if parts that reach three are read; what is
   left over is what was not analysed, and the cost shows it.

*What it changes.* Step B's `read_values` (for on the same bin, against at
a greater value, through `held_at`) and `place_bounds` are replaced by 1,
and `NumericPercepts.reverse` by 2, in the step that builds the towers.
The 0-or-1 match of a place goes with it if containment on the wires
tells two places apart; that is measured there.

### 29.4 The order, revised

| Step | Builds |
|---|---|
| T | The two towers over intensities (29.1), the Pi over values as containment (29.3, 1), the `.where` width setting (29.2), on the numeric stem; the XOR gate and certificate as they are |
| C1 | The scene's bank in its sentence's field, one Not (28.2) |
| C2 | The image put back by the most likely composition and charged (29.3, 2) |
| D2, F, G, H | as 28.2 |

### 29.5 His corrections to 29.1 to 29.3 (Alec, 2026-11-24)

> 1. literally I am talking about the less-than operation, which is
> numerical parthood, and applied elementwise in this case. Please dont
> call it containment.
> 2. Said more correctly, [1,0] is never a part; all percepts are atomic
> over digits. this is what we need in order to present the XOR problem
> properly, without cheating by aggregation.
> 3. "A part reads for wherever the occurrence holds it" , please use your
> analogical reasoning to relate this to our character percepts: A is a
> part of AB. if we store the parts in order, and we have the powerset of
> parts, we can reconstitute without error by placing the parts according
> to their overlap. If taht gets slow, build a cache.
>
> Please continue to build all of 1.6

**They stand over 29.1 to 29.3 where they differ.**

1. *The word.* The operation is less-than-or-equal, numerical parthood,
   applied element by element. "Containment" is not used of it.
2. *XOR.* The setting of 29.2 is that every percept is atomic over digits:
   no part spans two places, so [1,0] is never a part and nothing
   aggregates the two inputs but the conjunction.
3. *The way back.* As A is a part of AB: the parts are kept in their order,
   and with the powerset of the parts a thing is reconstituted without
   error by placing its parts according to their overlap. 29.3's "greatest
   part read for at each place" is that placing on one place, where the
   parts that overlap are ordered by less-than and the greatest is the
   thing itself, which is among its own parts. A cache is built if it is
   slow.

## 30. Step T: numerical parthood on the numeric stem (Claude, 2026-11-24; the text for the implementer is 30.2)

### 30.1 What it builds, and its certificates

1. *The Pi over values is numerical parthood, element by element.* A row
   reads for at an occurrence where what the row keeps is less than or
   equal to what the occurrence holds, on every coordinate: on the
   `.what`, the row's value's code against the occurrence's (so a row of a
   lesser or equal value reads for); on the `.where`, the row's entries
   against the occurrence's wires (as `place_bounds` does today). Nothing
   tells against. It is one operation of `BasicPercepts` for the value
   space and calls nothing of the first Pi over forms (`held_at`). What a
   conjunction's holders add (`held`) stays.
2. *A value's row is nominated on recurrence and brings its concept,* as a
   form's does, and is not promoted at first sight.
3. *Percepts atomic over digits.* A key in `data/model.xml` gives the
   greatest `.where` width of a percept of intensities, with a default
   that sets no limit; the XOR configurations set it to one. At one, a
   value's row and its concept are read at single places and never over a
   reading that spans two, and no part spans two places.
4. *The way back from positive evidence.* `NumericPercepts.reverse` places
   the parts read for according to their overlap: at each place, the
   greatest value among the rows that a concept of the field reads for
   there. It does not read `field.percepts`.

In `test/test_item16_values.py`, replacing V3 and adding:

| | Certificate | Passes when |
|---|---|---|
| V3 | Numerical parthood | A value's row reads for at an occurrence of an equal or greater value and nothing at a lesser one or where nothing is; its against lane is nothing everywhere. `read_values` names no `held_at` |
| V9 | Recurrence | A value met once has no concept; met as often as nominates a form, it has one |
| V10 | Atomic over digits | On an XOR configuration no reading wider than one place reads a value's row or its concept, and no row but the conjunction spans two places |
| V11 | Reconstituted from what is read for | After its values and places are known an image comes back exactly, at graded values; with the field's evidence at nothing, nothing comes back; `reverse` names no `field.percepts` |
| V5 | XOR, as it is | `test/numeric_xor.py`'s certificate, the stopping gate `test_mm_xor.py::TestMMXorConvergence::test_convergence`, and `test_concept_output.py` whole, its "without the intersection" among them, pass with their assertions as they are |

### 30.2 Text for the implementer

~~~
Item 1.6, step T: numerical parthood on the numeric stem. Read §30.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md (what you build), then
§29 whole with 29.5, §25.8, §25.10 and §25.11 (what step B built and why),
and H.5. Then the source: bin/BasicPercepts.py (read_values, place_bounds,
value_bank, perceive_value, promote, held), bin/NumericPercepts.py,
bin/Places.py, and how a form's row is nominated on recurrence and brings
its concept (owner.nominates, bring_concept; bin/FamiliarTier.py).

Keep tmp/item1-6/towers_progress.md current after each step.

1. Point 1 of §30.1: read_values by numerical parthood on the what and on
   the where, one operation, no held_at. Say whether place_bounds is then
   part of it or goes.
2. Point 2: a value's row is nominated on the count that nominates a form
   and brings its concept then.
3. Point 3: the key, named by you beside whereAxes in data/model.xml and
   data/model.xsd, default no limit; set to one in the XOR configurations
   (MM_xor.xml, MM_xor_fixture.xml, XOR_exact.xml, and model.xml's own
   default data set if it is XOR, saying which you set). No default in
   code; data/BasicModel.xml gains nothing.
4. Point 4: reverse, without field.percepts. If it is slow on MM_image,
   build the cache and say what it holds.
5. test/test_item16_values.py: V3 replaced, V9 to V11 added, V5 as it is.
6. Port a test only where one of the four points makes its assertion
   false, and list each with old and new. A test that needed its input
   presented until its places were known may need it presented until its
   values are known too; read the count from the model. No other
   assertion is weakened, skipped or marked; no seed is chosen. Neither
   stopping gate, nor anything a gate uses, is touched.
7. Run with the bounded runner (.venv/bin/python test/test_report.py),
   RUN_TRAINED and RUN_SLOW unset: test/test_mm_xor.py, the other
   stopping gate, test/test_item16_values.py, test/test_item16_scene.py,
   test/test_item16_grid.py, test/test_concept_output.py,
   test/test_item18_places.py, test/test_item18_paths.py,
   test/test_output_synthesis.py, test/test_understanding_reconstruct.py,
   test/test_item17_code.py, test/test_item18_fields.py,
   test/test_item18_perceptual_scalar.py, test/test_item18_scalar.py,
   test/test_item17_echo.py, and the files of
   tmp/item1-6/values_run4_files.txt. No test that passes in the sweep
   (its outcomes are named in your prompt) may fail, apart from those
   known to fail by chance.
8. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
9. Return as tmp/item1-6/section31.md, headed "## 31. Step T: the return":
   files changed and why; each certificate with what it printed; the key's
   name; every test ported, old and new; the XOR certificate's answers,
   weights and error before and after; what in §30.1 could not be built
   as written and what you built. You do not stop on a question of design:
   propose, build, mark it. Stop only if a gate, the XOR certificate or
   "without the intersection" cannot pass with its assertion as it is, and
   then say exactly where it fails and what you tried.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 31. Step T: the return (Claude on Opus, 2026-11-24)

The work of §30.2 on the numeric stem. No git was run; nothing is committed.
Progress was kept in `tmp/item1-6/towers_progress.md`. Every file changed was
copied first to `/tmp/item1-6-2026-11-24/towers_before/` (`bin/`, `data/`,
`test/`); the one function removed (`BasicPercepts.place_bounds`) is kept
there, in `towers_before/bin/BasicPercepts.py`. V5 as printed before any edit
is in `towers_before/v5_before.log`.

### 31.1 Files changed, and why

| File | Why |
|---|---|
| `bin/BasicPercepts.py` | Point 1: `parthood(bank, values)` (new) replaces `place_bounds`: one less-than-or-equal, element by element, on the `.what` (the row's code against the occurrence's, and the row's value against the occurrence's intensity, 31.4, 1) and on the `.where` (the row's entries against the occurrence's wires on both lanes), times the `.when` read by `field_factor` as before. `read_values(bank, values, *, where_width)` reads by it: for = parthood x activity, against = 0; it names no `held_at`. Point 3: where `where_width` is above zero, a value's row (`is_value`) reads nothing at a reading wider than that many places. Point 2: `perceive_value` reserves a value's row and no longer promotes it. |
| `bin/NumericPercepts.py` | Point 2: in `read`, each value's row is counted under `('value', row)` by `owner.nominates` once per stream that holds it, and on nomination `Places.bring` promotes it and brings its concept. Point 3: `__init__` reads `architecture.intensityWhereWidth` (no default in code) and `read` passes it to `read_values`. Point 4: `reverse` rewritten (31.4, 3). The readout: `pooled(field)` (new), used by `forward` before `read_field` (31.5). |
| `bin/Places.py` | `admit`'s second half becomes `bring(owner, row, kind)`: promote a nominated row and bring its concept of order zero (symbol type `kind`), `None` where order zero has no room. Places call it with `'place'`, values with `'value'`. |
| `bin/ScalarPercepts.py` | `perceive` promotes the value it perceives itself, so the value beside the words is promoted at first sight as before (25.5: its reading is kept until round two). |
| `data/model.xml`, `data/model.xsd` | The key `architecture.intensityWhereWidth`, beside `whereAxes`; default `0` (no limit); `xs:nonNegativeInteger`. |
| `data/MM_xor.xml`, `data/MM_xor_fixture.xml`, `data/XOR_exact.xml` | `intensityWhereWidth` 1. |
| `test/test_item16_values.py` | V3 replaced; V9, V10, V11 added; V4 ported (31.3). |

`data/BasicModel.xml` gains nothing. Neither stopping gate's file is touched.
`model.xml`'s own default data set is `xor` (numeric), but its key is left at
0: `model.xml` is every configuration's default, and 1 there would make
`MM_image` (which declares no key of its own) atomic over digits too, against
"a default that sets no limit". The three XOR configurations named set 1.
`MM_global`, `MM_reading`, `MM_qa`, `MM_20M_xor` name the xor data set but
are `dataType embedding` (text); they build no numeric stem and are not set.

### 31.2 The certificates, as printed

All ten of `test/test_item16_values.py` pass, alone and in the run (31.6).

- **V3 (replaced), numerical parthood.** On MM_xor (width 3), values 0.25,
  0.3, 0.5, 0.75, 1.0, five images: `bins_of_25_and_30: [0, 0]`; every pair
  `(1.0, 0.0)` where the occurrence is at or above the row's value and `(0.0,
  0.0)` elsewhere and where nothing is, e.g. `'0.3 at 0.25': (0.0, 0.0)`,
  `'0.25 at 0.3': (1.0, 0.0)`, `'0.5 at 0.75': (1.0, 0.0)`, `'0.75 at 0.5':
  (0.0, 0.0)`, `'1.0 at 1.0': (1.0, 0.0)`; the against lane is zero
  everywhere; `read_values` and `parthood` name no `held_at`;
  `numeric_read_uses: ['read_values', 'value_bank']`.
- **V9, recurrence.** MM_xor, `[[.5],[0.]]` in training, one stream:
  `mint_count 3`, `seen: [{time 1, promoted False, concept None}, {time 2,
  promoted False, concept None}, {time 3, promoted True, concept 0}]`.
- **V10, atomic over digits.** `intensityWhereWidth: {model.xml: 0, MM_xor: 1,
  MM_xor_fixture: 1, XOR_exact: 1}`; on MM_xor after the XOR batch has
  recurred: `value_row_at_places [[0,0],[0,1],[1,0],[1,1]]`,
  `value_row_at_bracket` all 0; the value's concept `at_places
  [[0,0],[0,.7616],[.7616,0],[.7616,.7616]]`, `at_bracket [0,0,0,0]`,
  `as_the_readout_reads_it [0,0,0,0]`; `spans: {0: value 1, 1: place 1, 2:
  place 1, 3: conjunction 2}` (every row is a value's, a place's holding one
  place of the box, or the conjunction).
- **V11, reconstituted from what is read for.** MM_image (width 3, so the
  grid's 37 values fall in `bins_of_the_grid [0, 1, 2]`), the image of every
  k/37, after it has recurred, put back with `percepts=None`: `given [0.7568,
  0.7838, 0.8108, ...]`, `back` the same, `exact True` (`torch.equal`); with
  the field's evidence at zero, `silent_greatest 0.0`; `reverse_names_percepts
  False`.
- **V5, XOR as it is.** See 31.4. `{'conjunction': 3, 'members': (1, 2),
  'places': True}`. The stopping gate `test_convergence` and
  `test_concept_output.py` whole (its "without the intersection" three times:
  `without_intersection 0.25`) passed in the run.
- V1, V2, V4, V6, V7 as before; V7 `forward_seconds 0.0097, reverse_seconds
  0.0003` (no cache was needed: `reverse` is one tensor operation per value's
  row with a concept); V6 `{'values': 1, 'places': 2, 'conjunctions': 1}`.

### 31.3 Tests ported

| Test | Old | New |
|---|---|---|
| `test_item16_values::test_v4_a_place_by_its_wires` | `basic.place_bounds(bank.table, [p0, p1])[:, 0] == [[1,0],[0,1]]` | `basic.parthood(bank with rows (p0, p1), values)[:, 0] == [[1,0],[0,1]]`: `place_bounds` is now part of `parthood` (point 1); the assertion is as it was |
| `test_item16_values::test_v3_the_pi_over_values` | for at its own bin, against at a greater value; `held_at` part and whole on every pair | replaced by §30.1's V3 (31.2) |

No other test was ported, weakened, skipped or marked; no seed; no count
literal (V9, V10 read `concept_mint_count`). The gates and what they use are
untouched.

### 31.4 XOR, before and after; what could not be built as written

**The certificate** (`numeric_xor.certificate`, MM_xor), before (this tree
before any edit) and after:

| | Before | After |
|---|---|---|
| answers | `[1.8189894035458565e-11, 1.0, 1.0, 1.8189894035458565e-11]` | the same |
| weights (places, places, conjunction) | `[1.313035249710083, 1.313035249710083, -2.626070499420166]` | the same |
| against | `[0.0, 0.0, 0.0]` | the same |
| mse | `1.6543612251060553e-22` | the same |
| promoted_row (the conjunction's concept) | 2 | 3 (the value's concept now takes row 0) |

1. *The `.what` by its code alone does not give "a row of a lesser or equal
   value reads for".* A value's code is its bin; on MM_xor and MM_image a
   half is 3 bins wide, so 0.3's row read for at 0.25 (both bin 0), and the
   way back put 12/37 at every place holding a lesser value of its bin
   (V2 failed: `back 0.3243` where `0.0270` was given). **Built:** on the
   `.what` the row's value (from the table of values; 0 for a row that is no
   value's) is one more coordinate of the less-than-or-equal, against the
   occurrence's intensity, which `read_values` now takes beside the bank.
   The code is still compared too. *Marked.*
2. *Point 3 alone does not keep "1 somewhere" out.* With the value's row and
   concept read at single places only, "without the intersection" still
   failed (3.09e-14) and the certificate's places' weights fell to 1.19e-8:
   see 31.5. **Built:** `NumericPercepts.pooled`. *Marked.*
3. *`reverse`*: at each place, the greatest value among the values' rows
   that a concept of order zero of the field reads (by an entry not a Not)
   where that concept's for lane is above zero at the place's reading. A
   place's row is no longer consulted (it keeps no exact value). On the
   first presentations nothing comes back, as before (V2), now because no
   value's row has its concept yet.
4. *A value's row that cannot have a concept* (order zero full) is left
   unpromoted and unread (`bring` returns `None`), as a place's row is never
   reserved then.
5. `ScalarPercepts` keeps promotion at first sight (31.1): moving the value
   beside the words to recurrence is round two's.

### 31.5 What pools a concept's evidence over readings

`OutputSpace.read_field` (`bin/Spaces.py`, about line 20930) begins
`evidence = field.evidence.amax(2)`: every concept's readout is the greatest
over all the field's readings. On the numeric field that is the value's
concept read over every place at once, an OR of the places: exactly the
aggregation point 3 forbids, done after the field. That is why the version in
`NumericPercepts.with_value_concepts.py` broke "without the intersection",
and why it broke again here with point 3 in place (3.09e-14). Nothing else
between the field and the readout pools over readings: `FieldProjection.key`
keeps a centroid per reading, and `close` reads per reading.

Two ways were tried. *Reading the readout at the bracket only* (the field cut
to its last reading) made "without the intersection" 0.25 but changed the
certificate's form: at the bracket the one Not already silences the places
where the conjunction holds, so XOR was fit by the places alone with the
conjunction's weight 3.6e-7 (positive), and the certificate's sign assertion
failed. Withdrawn. *Kept:* `NumericPercepts.pooled(field)`, used by
`forward` before `read_field`: where `intensityWhereWidth` is set and
narrower than the input, the evidence of each concept that reads a value's
row (by an entry not a Not) is kept at the bracket alone, where it is not
read, so the greatest over readings is nothing; every other concept is pooled
as before, and with no limit the field is passed as it is. `read_field`
itself is unchanged (text models read it too). *Limit, marked:* a concept
above order zero made over a value's concept would still be pooled over the
places; none is made on the XOR configurations (only the places' concepts
are observed together).

### 31.6 The run

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools
.venv/bin/python test/test_report.py` with the other stopping gate and the
57 files of `tmp/item1-6/towers_run_files.txt` (§30.2 step 7's list and
`values_run4_files.txt`; `test_mm_xor.py` whole with its gate):

**`Result: test_failure, exit 1; 520/520 cases completed.`**
`output/tests/20261124-080706-85b80a/result.json`; 704 s. 472 passed, 44
skipped, 4 failed. **Both stopping gates passed**, and
`test_concept_output.py` whole. The four:

- `test_item17_symbols::test_54bd_...` fails in the sweep.
- `test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`
  is on the list of chance.
- `test_item16_grid::test_g1_built_widths_are_as_before[MM_xor.xml]` and
  `[MM_image.xml]` pass in the sweep and alone here, and fail by test order:
  in the run they followed `test_item16_values.py` in one worker, and after
  any test that builds MM_image through `recon_bench` (V7 as it was, or V11)
  the registry's text range is 784 where the fixture has 4096. **The same
  two fail in the same order on a copy of the tree with the files as they
  were before this step** (`pytest test_item16_values.py::test_v7_...
  test_item16_grid.py::test_g1_built_widths_are_as_before`: 2 failed, 1
  passed on both trees). Not this step's; not repaired.

Every other test of the run has the sweep's outcome; the new tests are absent
from the sweep and passed.

### 31.7 For the reviewer

- The key's name: `architecture.intensityWhereWidth` (0: no limit).
- `place_bounds` goes: it is the `.where` half of `parthood`, whose zero
  entries hold everywhere under less-than-or-equal, so its "empty reads one"
  case needs no branch.
- 31.4, 1 and 31.5 are the two departures from §30.1 as written; both are
  needed for V11 and for "without the intersection" respectively.

### 31.9 Review of step T (Claude, 2026-11-24)

**Accepted.** Claude read the change and ran `test_mm_xor.py` whole, the
other stopping gate, `test_item16_values.py`, `test_concept_output.py`,
`test_item18_places.py`, `test_item16_scene.py` and `test_item18_scalar.py`
(run `20261124-082150-8f8fbe`): 53 of 53 passed.

*On its two departures.*

1. *The value itself is compared beside its code.* A code tells values
   apart only to its bin, three bins wide on the XOR and image
   configurations, so by the code alone 0.3 was a part of 0.25. Accepted:
   less-than-or-equal on the values is the operation he named, and the
   code is its coarser image.
2. *The readout pools a concept over its readings*
   (`OutputSpace.read_field`, `field.evidence.amax(2)`), which read a
   value's concept over both places at once, an OR, whatever the width of
   a percept. Under the limit the numeric stem gives the readout a value's
   concept at no place (`NumericPercepts.pooled`). Accepted as the proof he
   asked for, and marked: it is a rule of the numeric stem beside the
   readout and not in it, and a concept above order zero over a value's
   would still be pooled. The readout's pooling is the thing to look at if
   the limit is to mean more than this proof.

*Found on the way:* `test_item16_grid::test_g1_built_widths_are_as_before`
for `MM_xor.xml` and `MM_image.xml` fails after any test that builds
`MM_image.xml`, on the tree before this step too: it depends on the order
of the tests.


## 32. Step C: the image read with its sentence, put back and charged (Claude, 2026-11-24; the text for the implementer is 32.2)

### 32.1 What it builds, and its certificates

On a text model that declares the scene's box (step A), with the reader of
step T.

1. *The image's bank at the opening.* At a sentence's opening, where the
   value beside the words is joined today (`SentenceField.open_field`,
   first sentence of a presentation), a row that carries a scene has an
   image's bank made of its pixels (`BasicPercepts.value_bank`) at the
   addresses of the `scene` range; its values are perceived and its places
   met, as the numeric stem does.
2. *One Not.* The image's rows, read by numerical parthood
   (`read_values`), and the sentence's rows, read by the first Pi over
   forms, stand in one table of lanes: a form's row reads nothing at an
   image's reading and a value's or a place's row of the image nothing at
   a sentence's. The table goes through `order_zero_field` in one call.
3. *Joined to the field.* The image's readings are columns of the
   sentence's opened field after the sentence's, each with its `.where` in
   the box and its wires. Which readings an image has (every place of the
   box, or the places a pixel holds, and the bracket) is the implementer's
   to choose and say. A row with no image (`null`) joins none; an empty
   image joins readings that read nothing.
4. *Put back.* At the opening's reconstruction the image is put back from
   the field alone: at each place of the box, the greatest over the
   value's rows of the row's value times the degree to which a concept of
   the field reads for it at that place's reading; nothing where none
   reads. The sentence is put back as it is today.
5. *Charged.* The opening's errors gain a term for the image, the squared
   difference place by place over the box between what was put back and
   the scene (a place with no pixel is 0), weighted and booked as the
   sentence's reconstruction is. A row with no image is charged nothing
   for one; an empty image is charged for whatever is put back.
6. *Not built:* a look that takes a felt thing; objects, names and
   `createMetaConcepts`; parts wider than a place.

In `test/test_item16_scene_field.py`, each printing what it measured:

| | Certificate | Passes when |
|---|---|---|
| C1 | Text as it was | A row of text with `scene: null`, and the same text given as a string, have the same opened field, the same put-back sentence and the same opening errors, exactly. Both stopping gates and the seven echo tests pass |
| C2 | One Not | At an opening with a scene, `order_zero_field` is called once and its rows hold forms' rows and values' rows |
| C3 | In the field | After an image has recurred as often as nominates, its readings in the opened field read for its values' and its places' concepts, at addresses of the `scene` range; `null` adds no reading; `[]` adds none that reads |
| C4 | Put back | Images of category 1 alone, each presented until its values and places are known: every pixel comes back at its place with its value exactly, at each of the varied steps, and nothing comes back elsewhere in the box. A sentence with its image (category 2) puts both back |
| C5 | Charged | The opening's errors hold the image's term where a row has an image and none where it has not; the term is nothing when the image comes back exactly, and above nothing for an empty image when something is put back; a gradient of it reaches the Sigma of order zero |

### 32.2 Text for the implementer

~~~
Item 1.6, step C: the image read with its sentence, put back and charged.
Read §32.1 of doc/plans/2026-11-18-item1-6-felt-math-world.md (what you
build), then §20 (20.1's table of what exists, 20.7), §21 (step A as
built), §30 and §31 (step T as built), §7 (its F2, F11 and F12 are to hold as
they stand: places, no table resized, what stands), §29.5,
and H.5. Then the source: bin/SentenceField.py (open_field,
reconstruction, charge, reconstruct), bin/ScalarPercepts.py (join, as the
example of a second modality joined at the opening), bin/FamiliarTier.py
(open_sentence), Spaces.py's cs_read_bank, cs_read_percept_field and
order_zero_field, bin/FieldReverse.py, bin/NumericPercepts.py and
BasicPercepts.value_bank, read_values, parthood; where the opening's
errors are made and merged in bin/Models.py (search
_sentence_open_fields and _opening_errors).

Keep tmp/item1-6/field_progress.md current after each step.

1. Points 1 to 3 of §32.1. The scenes of a batch are InputSpace._scenes.
   Where the image's lanes join the sentence's before the Not is yours to
   find; say where. Nothing of the sentence's reading may change when no
   row of the batch has an image.
2. Points 4 and 5. Say where the term is booked and by what weight.
3. test/test_item16_scene_field.py: C1 to C5, on the text model that
   test/test_item16_scene.py builds with the box (the echo test's model
   with sceneShape), fed by felt_world_corpus.rows.
4. Port a test only where a point of §32.1 makes its assertion false, and
   list each with old and new. No other assertion is weakened, skipped or
   marked; no seed is chosen. Neither stopping gate, nor anything a gate
   uses, is touched.
5. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_item17_echo.py,
   test/test_reading_pretraining.py, test/test_item16_grid.py,
   test/test_item16_scene.py, test/test_item16_values.py,
   test/test_item16_scene_field.py, test/test_item18_scalar.py,
   test/test_item18_recognition.py, test/test_percept_collection_binding.py,
   and every test file that names open_field, ScalarPercepts.join,
   _opening_errors or reconstruction_order_zero. No test that passes in
   the sweep (its outcomes are named in your prompt) may fail, apart from
   those known to fail by chance. Report the time of the seven echo tests
   and of the reading lesson before and after.
6. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
7. Return as tmp/item1-6/section33.md, headed "## 33. Step C: the
   return": files changed and why; where the two modalities' lanes meet;
   which readings an image has; each certificate with what it printed;
   every test ported; the two times; what in §32.1 could not be built as
   written and what you built. You do not stop on a question of design:
   propose, build, mark it. Stop only if a gate cannot pass with its
   assertion as it is, and then say exactly where it fails.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 33. Step C: the return (Claude on Opus, 2026-11-24) (Claude on Opus, 2026-11-24)

The work of §32.2 on the text model with the scene's box. No git was run;
nothing is committed. Progress was kept in `tmp/item1-6/field_progress.md`.
Every file changed was copied first to `/tmp/item1-6-2026-11-24/field_before/`
(`bin/FamiliarTier.py`, `bin/Models.py`, `bin/BasicPercepts.py`,
`test/test_serial_concepts.py`). Nothing was removed.

### 33.1 Files changed, and why

| File | Why |
|---|---|
| `bin/ImagePercepts.py` (new) | Points 1 to 5. `image_bank` (the batch's images as one `value_bank`), `perceive` (values perceived and nominated, places met, as the numeric stem does), `image_rows` (the rows of the image's space), `joined` (the one table of lanes), `read` (the opening's reading), `put_back`, `felt`, `charge`. |
| `bin/FamiliarTier.py` | `open_sentence` reads its bank through `ImagePercepts.read` in place of `owner.cs_read_bank(bank)`; `read` is `cs_read_bank(bank)` itself where no row of the batch has an image. |
| `bin/Models.py` | At the opening's reconstruction (beside `SentenceField.charge`, where `_opening_errors` is made), `ImagePercepts.charge` books the image's term in `_opening_errors`. Three lines. |
| `bin/BasicPercepts.py` | `property_presence` passes over a value's row. Found on the way (33.7, 1): on a text model shown an image before any word, the value's row is row 0, the row a word slot with no reference reads, and its whole (a value's reference, below zero) indexed the primitive properties out of range. One line. |
| `test/test_item16_scene_field.py` (new) | C1 to C5. |
| `test/test_serial_concepts.py` | Ported (33.5): its stand-in `inputSpace` gains `_scenes=None`, the attribute every `InputSpace` has. |

`data/` is untouched: no key was added (the image's reading uses
`architecture.intensityWhereWidth`, step T's, whose default sets no limit).
Neither stopping gate's file nor anything it uses is touched.

### 33.2 Where the two modalities' lanes meet

In `FamiliarTier.open_sentence`, between the sentence's one lookup
(`cs_lookup`, which meets the sentence's places) and the Not. Where some row
of the batch has an image and the sentence is the first (`sid == 0`, as the
value beside the words is joined), `ImagePercepts.read`:

1. builds the image's bank (`image_bank`) and perceives it (`perceive`):
   each value perceived (`perceive_value`), its row counted under
   `('value', row)` and, on nomination, promoted with its concept
   (`Places.bring`); the places met (`Places.meet`). This comes before
   either Pi reads, because a row nominated here writes its where entries
   and the Sigma's entries, and a reading saved for backward must not see
   them change (the first build, which perceived after the sentence's
   `read_bank`, failed "a sentence intermediate was mutated before
   backward");
2. reads the sentence's bank by the first Pi over forms
   (`BasicPercepts.read_bank`) and the image's by numerical parthood
   (`read_values`, over `image_rows`: the values' rows, the places' rows
   whose cone decodes to an address of `scene`, and conjunctions over these
   alone);
3. joins them (`joined`) into one `PerceptField`: its keys the sentence's
   then the image's rows not already among them; its readings the
   sentence's, then the image's; a form's row (any row not the image's) is
   nothing at an image's reading, and an image's row is set to nothing at
   the sentence's readings; events, spans, brackets, times, the when band,
   the boundaries, each event's reading and occurrence joined the same way;
   and each reading's `.where` wires (`Places.readings` of each bank);
4. sends it to `cs_read_percept_field`, whose loop calls `order_zero_field`
   once for each stream with all of it, and publishes the wires.

The opened field (`SentenceField.open_field`) is then made as today; the
value beside the words (`ScalarPercepts.join`) is joined after the image's
readings.

### 33.3 Which readings an image has

The places its pixels hold, each one place of the box (in the order of
their addresses), and the bracket over the whole box. Not every place of
the box: 513 more readings on every opening would carry through every
order for no pixel. Within a batch the image's columns are as many as the
greatest count of pixels, plus the bracket; a row's columns beyond its own
pixels are empty brackets (`[lo, lo]`). A row with no image (`null`) has
every column empty, the bracket too: it joins no reading. An empty image
(`[]`) has the bracket alone, which reads nothing. Every reading of an
image is at the present moment (`model.present()`).

### 33.4 The certificates, as printed

From the final run (33.6). The model is the echo test's text model with
`sceneShape` `8 8 8` (`test_item17_echo.echo_model` with
`architecture_overrides`), the eager reading of the echo test; rows from
`felt_world_corpus.rows`. It is built for one stream (its STM's `nStreams`
is the training `batchSize`, 1), so a batch of several rows cannot be
presented; where a certificate speaks of a batch of rows, the bank or the
charge is read on one (C3, C5).

- **C1, text as it was.** Two models built from the same state of the
  generator, each step taken by one and then by the other from the same
  state (no seed is set): `C1 'this is eight.' as a string and as a row
  with scene null: field (15, 1, 8, 2) same True, wires same True; put back
  'this is eight.' and 'this is eight.'; errors
  ['reconstruction_order_zero.pyramid'] same True` (`torch.equal` on the
  field's ids, evidence, `.where`, `.when` and wires, and on every term of
  the opening's errors; no image term on either). Both gates and the seven
  echo tests: 33.6.
- **C2, one Not.** `C2 {'text': 'this is eight.', 'scene': [[3, 2, 0,
  200]]}: order_zero_field called 1 time(s) at the opening, lanes (12, 1,
  10, 2); its rows hold 4 forms' rows, 1 values' rows (values [0.7843]), 1
  places' rows of the box` (the sentence's 8 readings and the image's 2).
- **C3, in the field.** `C3 scene range 4370..4882; 1 values' and 1
  places' concepts; each row alone: {'image': {'readings': [(4389, 4390),
  (4370, 4882)], 'value': [(4389, 4390)], 'place': [(4389, 4390)], 'reads':
  True}, 'null': {'readings': [], 'value': [], 'place': [], 'reads':
  False}, 'empty': {'readings': [(4370, 4882)], 'value': [], 'place': [],
  'reads': False}}; one batch of the three, its readings by row: [[(4389,
  4390), (4370, 4882)], [], [(4370, 4882)]]`. The pixel (3, 2, 0) is
  address 4389; the value's and the place's concepts read for there; the
  null row (with the same words) has no reading in the box; the empty
  image's bracket reads nothing.
- **C4, put back.** The first images of category 1 (train) in order, as
  many as order zero has room for (eight), each presented as often as
  nominates and once more, then each read again: `C4 category 1, 8 images
  (7 steps, 8 places), put back: [([[7, 4, 0, 200]], [(39, 0.7843)], True),
  ([[5, 2, 0, 127]], [(21, 0.498)], True), ([[2, 3, 0, 122]], [(26,
  0.4784)], True), ([[2, 1, 0, 200]], [(10, 0.7843)], True), ([[3, 0, 0,
  73]], [(3, 0.2863)], True), ([[7, 6, 0, 28]], [(55, 0.1098)], True), ([[5,
  6, 0, 24]], [(53, 0.0941)], True), ([[3, 6, 0, 197]], [(51, 0.7725)],
  True)]` (each: the image, the places of the box put back with their
  values, and `torch.equal` of what was put back over all 512 places with
  what was felt). `C4 category 2, {'text': 'this is eight.', 'scene': [[3,
  2, 0, 200]]}: the sentence put back 'this is eight.' (alone 'this is
  eight.'); the image put back exactly True`.
- **C5, charged.** `C5 {'text': '', 'scene': [[3, 2, 0, 200]]}: the
  image's term at first sight 1.0000, once known 0.0000; with a row with no
  image, by row: numerators [0.0, 0.0], entries [512.0, 0.0]; a batch with
  no image has the term False; the image taken as empty, term and
  |gradient| at the Sigma of order zero, last (0.0012014610692858696,
  0.0010091656586155295)`. The pair is charged on the field the image's row
  opened, given to two rows, one with the image and one with none: the
  second has no entry. The empty image is charged inside a training
  opening on that opening's field, and its gradient taken at
  `order_zero.value`.

### 33.5 Tests ported

| Test | Old | New |
|---|---|---|
| `test_serial_concepts.py`: its `fixture` (used by `test_open_pass_reads_once_before_composition_and_never_mints`, `test_open_pass_ignores_later_attention_narrowing_and_respects_inactive_streams`, `test_prime_boundary_joins_lexical_and_open_field_rows_in_one_step`) | a stand-in `inputSpace=SimpleNamespace(_packed_sentence_ids=..., _word_active_mask=...)` | the same with `_scenes=None`: `open_sentence` now asks the input space whether a row has an image (point 1). No assertion changed |

No assertion was weakened, skipped or marked; no seed chosen.

### 33.6 The runs, and the two times

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools
.venv/bin/python test/test_report.py` with the 42 selectors of
`tmp/item1-6/field_run_files.txt`: the gate of `test_item25_completion.py`,
`test_mm_xor.py` whole (with the other gate), §32.2 step 5's list, the 31
other test files that name `open_field`, `_opening_errors` or
`reconstruction_order_zero` (none names `ScalarPercepts.join`), and
`test_item17_looks.py` (it wraps `open_sentence`).

- Run 1 (`output/tests/20261124-084034-d15637`): 5 failed. Three were
  `test_serial_concepts.py`'s, ported (33.5).
- **Final run, `output/tests/20261124-090136-5f5c1b/result.json`:
  `Result: test_failure, exit 1; 355/355 cases completed.`** 1256 s. 349
  passed, 4 skipped, 2 failed. **Both stopping gates passed**, and
  `test_mm_xor.py` whole, the seven echo tests, `test_reading_pretraining.py`,
  G1 to G9 of the grid, S1 to S4, the values' V1 to V11, C1 to C5. The two:
  `test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`,
  on the list of chance; and
  `test_percept_collection_binding::test_bind_percept_collection_reads_membership_not_coordinate_overlap[2.0-True]`,
  `failed` in the sweep. The four skipped are skipped in the sweep. Every
  other test has the sweep's outcome.

**The two times**, each from a run of the bounded runner with
`test/test_item17_echo.py` and `test/test_reading_pretraining.py` alone
(the per-test durations of its workers):

| | Before (unchanged tree, `20261124-082518-8d80b7`) | After (`20261124-092249-42a745`) | In the final run (`20261124-090136-5f5c1b`) |
|---|---|---|---|
| The seven echo tests | 284.1 s (4.4, 41.2, 87.4, 12.7, 13.2, 83.4, 41.9) | 312.2 s (4.5, 49.5, 86.7, 13.1, 13.4, 54.0, 91.1) | 267.4 s |
| The reading lesson (6 tests) | 120.2 s (the checkpoint test 112.9) | 160.0 s (the checkpoint test 150.4) | 170.8 s |

Both are dominated by training that runs until a criterion is met: an
echo cell trains round by round until every word is echoed in two rounds
running (from a start drawn afresh), and the reading lesson trains epoch by
epoch, its rows in a fresh order, until it qualifies
(`pretrain_reading.train_reader`). Their tests move against each other
from run to run (the echo's sixth 83 then 54, its seventh 42 then 91), and a
third run of the checkpoint test alone took 403 s
(`20261124-093140-96fd05`). On a batch with no image, which is every batch
of these tests, the change adds two tests of `InputSpace._scenes`
(`ImagePercepts.imaged`) to each opening and one `is_value` to each row of
`property_presence`; no difference above the spread of the runs is
attributable to it, and none is claimed to be shown absent by these
numbers.

### 33.7 What in §32.1 could not be built as written, and what was built

1. *"The degree to which a concept of the field reads for it"* (point 4)
   cannot put a pixel back exactly: a concept of order zero is the Sigma's
   `tanh` of its entries, so a value's concept fully read is 0.7616 and
   the value times it is not the value. **Built:** the degree is whether
   the concept reads for (its for lane above zero), one or nothing, with
   the derivative of the for lane passed through (`(for > 0) + (for -
   for.detach())`, the zero added first so that the value comes back to the
   last bit). This is `NumericPercepts.reverse`'s rule with a gradient.
   *Marked.*
2. *The weight of the term* (point 5). Booked in `_opening_errors` as
   `reconstruction_order_zero.image`, by `Error.squared` (the squared
   difference at each of the 512 places against the felt image's own
   square, the codebase's uninformed baseline; an empty image, whose
   baseline is nothing, is a penalty per place), weight
   `model.loss.reconstruction_scale`, category `reconstruction`: the
   sentence's term (`reconstruction_order_zero.pyramid`) has the same
   weight and category, and `SentenceField.reconstruct` sums every
   `reconstruction_order_zero.*` term into the sentence's cost. Masked by
   the active rows that have an image. *Marked.*
3. *"order_zero_field is called once"* (C2): `cs_read_percept_field` calls
   it once for each stream, so once for a batch of one, which the test
   model is (33.4). The certificate counts calls inside the opening's
   reading.
4. *A batch of rows with and without images* cannot be presented to the
   echo test's model (one stream). The masking is built for it (empty
   columns for a `null` row, the charge's mask) and read at the bank and
   the charge (C3, C5) rather than through `runBatch`. *Marked.*
5. *C4's "each presented until its values and places are known"*: on the
   echo model order zero has room for about eight images of one pixel
   (each needs a value's and a place's concept, and category 1's 93 train
   images have 37 steps and 52 places). The test takes them in the corpus's
   order as long as they fit half of order zero's room, reading the room
   from the model; eight, seven distinct steps.
6. *Found on the way:* `BasicPercepts.property_presence` read a value's
   row's whole as a primitive property (33.1). On any model whose row 0 is
   a value's row it indexed outside the properties (here, a text model
   shown an image before any word). It now passes a value's row over. On
   a model whose row 0 is a form, as every existing test's, nothing
   changes.
7. *Not built*, as §32.1 point 6 says: a look that takes a felt thing;
   objects, names and `createMetaConcepts`; parts wider than a place. The
   image's readings are not given to the looks. `read_values` is given
   `architecture.intensityWhereWidth` (no limit on the text model), so a
   value's row may read at the bracket; in C3 its concept did not read for
   there, which was not looked into further.

### 33.9 Review of step C (Claude, 2026-11-24)

**Accepted.** Claude read the changes to the existing files (nine lines)
and the new reader's entry, and ran both stopping gates,
`test_item16_scene_field.py`, `test_item16_scene.py`,
`test_item16_values.py`, `test_item17_echo.py`,
`test_reading_pretraining.py` and `test_item18_scalar.py` (run
`20261124-093948-c9cd66`): 52 of 52 passed.

- *Text as it was.* Where no row of a batch has an image, `ImagePercepts.read`
  returns `cs_read_bank(bank)` and `charge` returns at once.
- *Times.* In Claude's run the seven echo tests took 301 s and the reading
  lesson 79 s; the implementer's were 312 s and 160 s after, 284 s and
  120 s before; §18.3's were 225 s and 129 s. Both train to a criterion and
  swing by more than any of these differences, so no slowing is shown and
  none is ruled out. The closing sweep times them again.
- *A concept reads or does not.* A concept fully read gives the tanh of
  its entry, .76, so a value times that degree never comes back exactly;
  the image is put back from whether a concept reads for, with the
  gradient of its for lane passed through. Accepted.
- *Room at order zero.* The test's small model holds about eight
  one-pixel images, since each value and each place brings a concept. The
  felt world's configuration needs room for 64 places and some 80 values
  at order zero before round D2; its sizes are set then.
- *One stream.* A batch that mixes rows with and without images is not
  run through the test's model, which takes one stream; the mask is
  checked on the charge directly.


## 34. Step D2: nouns, the name, `createMetaConcepts` (Claude, 2026-11-24; the text for the implementer is 34.2)

### 34.1 What it builds, and its certificates

*Rests on 19.2, 19.4 (the key, true by default), 19.5 (the four stages),
22.0 (nouns are symbols of order one), 22.1 question 1 (the meta concept is
a Sigma of order one, made by MTM as any concept), 28.4.*

*Its bound.* It names an exact intensity: a word with a pixel of one
value. What makes one object of the several steps of one value (29.1) is
not built here and is the open point of the step after it; so the corpus's
naming at a reserved step is reported and not asserted.

1. *The key.* `createMetaConcepts` in `data/model.xml` and
   `data/model.xsd`, true. True: `interpret` makes a word's meta concept,
   and a blank object where the word has none, as it does today. False:
   it makes neither. `data/scene/BasicModel_scene.xml` sets it false.
2. *Nouns.* The concept of a word and the concept of a value, each known
   from its own occurrences, each have a symbol of order one.
3. *The meta concept is learned.* Where a concept whose percepts lie in
   the `text` range and one whose percepts lie in the `scene` range are
   present together in an opened field, on the count that nominates, MTM
   nominates a concept of order one that is a Sigma over the two nouns'
   symbols: a list of two, not a Pi's row. Each pair present together is
   counted for itself, so of several things present with a word the pair
   that recurs is the one made. Within one modality today's promotion
   stands.
4. *One raises the other.* Met alone, either noun raises the unit, being
   a sufficient part of a Sigma, and through the unit the other member is
   raised in the field, by the way down that the source has between a
   concept of a higher order and its members. If the source has none that
   serves, the implementer says so, builds the least that does, and marks
   it.
5. *`interpret`,* where the key is false: of a word, the other member of a
   unit across modality that holds the word's symbol; the word itself
   where there is none.
6. *The count.* The assessment tool reports the units across modality, by
   order.

In `test/test_item16_naming.py`, on the text model with the scene's box
and the key false, each printing what it measured:

| | Certificate | Passes when |
|---|---|---|
| N1 | The key | With the key true a text model makes meta concepts and blank objects as before: both stopping gates and the tests of `interpret` that pass in the sweep pass. With it false no meta concept and no blank object is made by `interpret` |
| N2 | Each known alone (stage i) | After a word alone and a pixel alone have each recurred, each has its concept and its symbol of order one, and there is no unit across modality |
| N3 | The unit (stage ii) | After the word and the pixel have been present together as often as nominates, there is one concept of order one whose members are the two nouns' symbols, read as a Sigma; the count of units across modality is one |
| N4 | The pair that recurs | A word present with its own pixel every time and with another pixel some of the times is united with its own |
| N5 | One raises the other (stage iii) | After N3, the word alone raises the unit and the object's concept in the field above what it read before the unit was made, and the pixel alone raises the unit and the word's concept likewise |
| N6 | `interpret` (stage iv) | After N3, `interpret` of the word gives the object; of a word with no unit, the word |
| N7 | The corpus (a measure) | Categories 1 and 2 taught in order: the count of units by order, and for each value's word how many of its steps it is united with. Reported; nothing is asserted on it |

### 34.2 Text for the implementer

~~~
Item 1.6, step D2: nouns, the name, createMetaConcepts. Read §34.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md (what you build), then
§19.2, §19.4, §19.5, §19.6, §22.0, §22.1 (question 1), §28.4, §33 (step C
as built) and H.5. Then the standing documents on what you will touch:
doc/Orders.md ("What makes what", "Where a noun begins", "A meta concept
and the blank object"), doc/Architecture.md ("Above order zero", "What
co-occurs is a necessary part"), doc/Language.md on interpret. Then the
source: bin/Interpret.py (interpret, blank, define, denotations),
bin/FamiliarTier.py (observe_field, observe_candidate,
promote_candidates, maybe_raise_order's caller), how a symbol of order
one and a Sigma of order one are made and read (bin/Symbols.py and the
conceptual space in bin/Spaces.py), bin/ImagePercepts.py, and
bin/Assessment.py.

Keep tmp/item1-6/naming_progress.md current after each step.

1. Point 1: the key, beside whereAxes; false in
   data/scene/BasicModel_scene.xml and nowhere else; no default in code;
   data/BasicModel.xml gains nothing.
2. Points 2 and 3. Say exactly where the unit is nominated, what counts
   the pair, and how a Sigma of order one over two symbols is stored and
   read. Text models with the key true are not to change.
3. Point 4, then point 5, then point 6.
4. test/test_item16_naming.py: N1 to N7. Its rows are made in the test (a
   word and a pixel of one value at a place, so that the value is exact);
   N7 uses felt_world_corpus.rows on categories 1 and 2. Give the test's
   model the room it needs at order zero and order one by its
   configuration in the test, and say what you set.
5. Port a test only where a point of §34.1 makes its assertion false, and
   list each with old and new. No other assertion is weakened, skipped or
   marked; no seed is chosen. Neither stopping gate, nor anything a gate
   uses, is touched.
6. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_item17_echo.py,
   test/test_item16_scene.py, test/test_item16_scene_field.py,
   test/test_item16_values.py, test/test_item16_naming.py,
   test/test_concept_output.py, and every test file that names interpret,
   Interpret, blank, define, observe_field, promote_candidates or
   DefinitionIndex. No test that passes in the sweep (its outcomes are
   named in your prompt) may fail, apart from those known to fail by
   chance.
7. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
8. Return as tmp/item1-6/section35.md, headed "## 35. Step D2: the
   return": files changed and why; where and how the unit is made, stored
   and read; the way down you used or built; each certificate with what
   it printed, N7's numbers in full; every test ported; what in §34.1
   could not be built as written and what you built. You do not stop on a
   question of design: propose, build, mark it. Stop only if a gate
   cannot pass with its assertion as it is, and then say exactly where it
   fails.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 35. Step D2: the return (Claude on Opus, 2026-11-24)

The work of §34.2 on the text model with the scene's box. No git was run;
nothing is committed. Progress was kept in `tmp/item1-6/naming_progress.md`.
Every file changed was copied first to `/tmp/item1-6-2026-11-24/naming_before/`
(`bin/Interpret.py`, `bin/FamiliarTier.py`, `bin/ImagePercepts.py`,
`bin/Assessment.py`, `data/model.xml`, `data/model.xsd`,
`data/scene/BasicModel_scene.xml`, `test/test_item16_scene.py`,
`test/test_item17_symbols.py`). Nothing was removed.

### 35.1 Files changed, and why

| File | Why |
|---|---|
| `data/model.xml` | `architecture.createMetaConcepts`, `true`, after `sceneShape` (beside `whereAxes`). |
| `data/model.xsd` | `createMetaConcepts`, `xs:boolean`, beside `sceneShape`. |
| `data/scene/BasicModel_scene.xml` | `<createMetaConcepts>false</createMetaConcepts>`; no other file of `data/` has the key; `data/BasicModel.xml` gains nothing. |
| `bin/Interpret.py` | `InterpretLayer.create_meta_concepts`, read from the key in `__init__`. Where it is false: `reading` makes no blank object and no meta concept and gives `named(word)`; `forward` of a word with no object gives `named(word)`; `lookup_word`'s discovering path does not `define` the word's case as its object. New `named(word)`: the other member of a unit across modality that holds the word's symbol (the expected one among several, by `preferred_objects`), or the word. With the key true every path is as it was. |
| `bin/FamiliarTier.py` | `MODALITIES`, `ACROSS`; `modalities`, `apart`, `observe_units`, `units`, `raised`, `across` (new); `observe_field` keeps today's context within one modality and counts each pair across modality (`observe_units`); `promote_candidates` writes no percept feature for a literal `ACROSS` (its Sigma is the cover's). |
| `bin/ImagePercepts.py` | `noun(owner, percept)`: a value's concept, when its row is nominated and the concept brought, is given a symbol of order one (its fit, or one nominated under `('noun', concept)`). Called in `perceive` after `bring`. |
| `bin/Assessment.py` | `units_across_modality(model)`: the units across modality by order. |
| `test/test_item16_naming.py` (new) | N1 to N7. |
| `test/test_item16_scene.py`, `test/test_item17_symbols.py` | Ported (35.6). |
| `tmp/item1-6/naming_progress.md`, `naming_run_files.txt`, `naming_run*.log` | Progress, the run's selectors and logs. Not source. |

### 35.2 Where the unit is nominated, what counts the pair, how it is stored and read

- **Each concept's modality.** In `FamiliarTier.observe_field`, for each
  stream, `modalities` reads the opened field (`[S, B, L, 2]`) and the
  readings' `.where` (`owner._percept_field.where`, the same `L`): a
  concept's modality is the one of the registry's ranges `text` and `scene`
  that holds every reading at which it reads (either lane), and none where
  its readings lie in neither or in both. Where the model has no range
  `scene` (every model but the scene's), it is empty, and nothing below
  changes anything: `apart` is never true and `observe_units` finds no
  pair. So a text model, and a numeric one (the XOR gate), are as before.
- **Within one modality, today's promotion.** The focal row's context is
  the offered rows less those `apart` from it (both of a modality, and
  different). Without this the existing focal/context candidate would make
  field conjunctions of order zero over a word and a value, which 19.5,
  point 2 forbids.
- **The pair counted for itself.** `observe_units(owner, batch, present,
  kinds)`: of the offered concepts of order zero with a modality whose
  *for* lane is above `conceptUseFloor` (a concept read only against is not
  present: 35.7, 2), each pair of a `scene` concept and a `text` concept is
  offered to MTM by `observe_candidate(owner, 1, {row: for, other: for},
  literal=key, literal_key=key)` with `key = ('across', (('scene',
  value), ('text', word)))`. The literal keeps each pair its own MTM entry
  (without one, `match_candidate` would merge pairs whose centroids are
  alike); each kept close is one reading (`record_reading`), so the count is
  of openings at which the two were present together, within MTM's horizon.
- **Nominated.** In `promote_candidates`, on the ordinary path for a
  candidate above order zero: on `conceptMintCount` readings, its order is
  one above its members' (`1 + 0`); `concept_fit` first (a concept of order
  one reading exactly the two symbols); else a new concept, its LTM1
  grounding (`_ground_candidate_members`), its row at order one, its own
  symbol (`nominate_symbol(key, cid)`), and `cover_members(..., conjunctive
  =False)`, which writes the Sigma: `_populate_concept_weights` with the
  row-addressed witness gives the unit an entry, at the support's weight,
  on the thing that reads each member (`things_reading`: the member's
  symbol, its fit, or one made). The literal is remembered
  (`candidate_literal`). The only change on that path is that a literal
  `ACROSS` writes no percept feature.
- **Stored.** A unit is a concept of order one in the concept store whose
  row has two entries, on the two nouns' symbols (store columns are symbol
  indices), each symbol a row of the symbols' Pi over one concept of order
  zero: the word's symbol is the one the kind of words made for it at the
  word's admission (`('kind', words, word)`); the value's is `noun`'s. It
  is a Sigma, a list of two: it reads either symbol alone (N5).
- **Read.** `FamiliarTier.units(owner)`: each raised concept whose
  `candidate_literal` is `ACROSS`, with its order and the members of its
  pair whose symbols its Sigma reads at a weight above zero. (Today's part
  learning offers every defined row of order one a zero entry on every
  concept of the context, `_prepare_part_learning`; those are not members
  until learned, so `units` does not read the row's columns alone.)
  `across(owner, concept)`: the other members of each unit that holds the
  concept. `Assessment.units_across_modality` counts `units` by order.

### 35.3 The way down

The source's way down is `ConceptualSpace.cs_reverse_presence` (the Sigma's
transpose to the symbols a concept reads, then the symbols' Pi back to the
seats of the concepts they hold), used by `SentenceField.descended`. As it
is called there it does not serve: with no `observed` a concept's value
goes down its strongest entry alone (its case; the unit's two entries are
equal, so the first written, the word's), and with `observed` it goes down
only as far as what was observed reaches, so a member whose percepts are
absent gets nothing. And the opened field has no row for a concept whose
percepts are absent. **Built, the least that does, marked:**
`FamiliarTier.raised(owner, field, concepts, batch=0)` runs
`cs_reverse_presence` over the field's rows plus the rows of `concepts`
not among them (reading nothing), with `observed` everywhere one, as
`CaseSelection` already calls it, so nothing narrows which entry a value
goes down. The opened field itself and `descended` are not changed: the
other member is raised by this reading of the field, not written into the
field the decoder sees.

### 35.4 The certificates, as printed

From the final run (35.5). The model is the echo test's text model
(`test_item17_echo.echo_model`) with `sceneShape` and `createMetaConcepts`
as `data/scene/BasicModel_scene.xml` has them (`8 8 8`, `false`), and,
set in the test: `ConceptualSpace.attentionPromotion` true (35.7, 1); for
N7 also `ConceptualSpace.nVectors` 512 (`CORPUS_ROWS`; the echo model has
64; order caps `(512, 323, 153, 64, 32)`) and
`architecture.serialWordBuckets` `8,256` (`CORPUS_BUCKETS`; the echo model
has `8`). Rows are made in the test: the word `three`, the pixel `[3, 2, 0,
200]`; for N4 the other pixel `[5, 5, 0, 240]`; `five` as a word with no
unit. N2, N3, N5, N6 and the key-false half of N1 read one model taken
through the stages in order (a module fixture).

From the final run (`output/tests/20261124-111651-25b0a2`):

- **N1, the key.** `N1 key true: {'definitions': 1, 'blanks': 1}, the word's blank object 95; key false ('false' in the scene's configuration), after every stage: {'definitions': 0, 'blanks': 0}`. With the key true both stopping gates, and every test of run 1 that names `interpret`, `Interpret`, `blank`, `define`, `observe_field`, `promote_candidates` or `DefinitionIndex`, have their sweep outcome (35.5).
- **N2, each known alone.** `N2 the word 'three' concept 93 symbols [0]; the pixel [3, 2, 0, 200] concept 95 symbols [1]; units across modality {}`.
- **N3, the unit.** `N3 after 'three' with [3, 2, 0, 200] as often as nominates: units across modality {1: 1}; [{'unit': 99, 'order': 1, 'members': (95, 93), 'entries': {95: [2.2849], 93: [2.2849]}}]` (the Sigma's entry on each noun's symbol; the weight is the candidate's summed support, as `cover_members` writes any promotion's).
- **N4, the pair that recurs.** `N4 'three' (93) with [3, 2, 0, 200] (95) 6 times, with [5, 5, 0, 240] (97) 2 of them: united with (95,); units [{'unit': 99, 'order': 1, 'members': (95, 93), ...}]`.
- **N5, one raises the other.** `N5 the word alone: the unit reads (0.9403, 0.0), the object's concept by the way down (0.9403, 0.0) (before the unit (0.0, 0.0)); the pixel alone: the unit reads (0.9403, -0.0), the word's concept (0.9403, 0.0) (before (0.0, 0.0))`.
- **N6, interpret.** `N6 interpret of 'three' (93): 95 (its reading 95), the object's concept 95; of 'five' (98), with no unit: 98`.
- **N7, the corpus (a measure; nothing asserted).** Categories 1 and 2 (train) in order, each row presented once, training: `300 rows taught, 0 refused; order caps (512, 323, 153, 64, 32); units across modality by order {1: 28}`. By value's word, the steps it is united with: eight 0, five 0, four 0, nine 0, one 0, seven 0, six 0, three 0, two 0. Every word in a unit, with the count of steps it is united with: `{'is': 10, 'this': 2, 'what': 2, '?': 3, 'that': 1, '.': 10}`. No seed is set and the numbers move from run to run: run 1 (35.5) gave eight 1 [175], nine 1 [72], seven 1 [125], the others 0; a pytest run of the same configuration `{1: 56}` and two 1 [47]; with the echo model's buckets (`8`) 91 of category 2's rows were refused (`registered sentence exceeds constructed seen capacity`) and `{1: 49}`. A number word meets each step about once in category 2, under the count that nominates (three, within MTM's horizon of 16 closes); the frequent words meet the frequent lesser values (35.7, 2) and unite with them.

### 35.5 The runs

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py`:

- **Run 1**, the 200 files of `tmp/item1-6/naming_run_files.txt` (both gates' files, §34.2 step 6's list, and every test file that names `interpret`, `Interpret`, `blank`, `define`, `observe_field`, `promote_candidates` or `DefinitionIndex`): `output/tests/20261124-103136-d1ab5d/result.json`, `Result: test_failure, exit 1; 2572/2572 cases completed.` 2629 s. 2443 passed, 122 skipped, 1 xpassed, 6 failed. **Both stopping gates passed.** The six: `test_item17_symbols::test_71_...` (new: the call site of `ImagePercepts.noun`; ported, 35.6); `test_item17_symbols::test_cert_c_...` and `test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`, on the list of chance; `test_item17_symbols::test_54bd_...`, `test_item17_word_said_back::test_the_answer_depends_on_the_spoken_request` and `test_percept_collection_binding::...[2.0-True]`, failed in the sweep. Every other test has its sweep outcome; the 122 skipped are skipped there.
- **Final run**, after the port: both gates, `test_item17_symbols.py`, `test_item16_naming.py`, `test_item16_scene.py`, `test_item16_scene_field.py`: `output/tests/20261124-111651-25b0a2/result.json`, `Result: test_failure, exit 1; 66/66 cases completed.` 65 passed; the one failure is `test_item17_symbols::test_54bd_...`, failed in the sweep. **Both gates passed**; N1 to N7, C1 to C5, S1 to S4 passed.

### 35.6 Tests ported

| Test | Old | New |
|---|---|---|
| `test_item17_symbols.py::test_71_no_symbol_is_made_for_a_concept_and_every_symbol_is_mtms` | the callers of `nominate_symbol` are `[('FamiliarTier.py', 'promote_candidates'), ('FamiliarTier.py', 'promote_identity'), ('Spaces.py', 'things_reading')]` | the same with `('ImagePercepts.py', 'noun')`: point 2 gives a value's concept its symbol, and no existing maker makes one (no kind reads values); it is made on MTM's nomination of the value's row. Its comment says so. The rest of the test (no handle names a making for a concept; every symbol after the lesson is MTM's or a kind's) is unchanged and passes |
| `test_item16_scene.py::test_s1_the_key` | the scene's configuration differs from `BasicModel.xml` in `["/architecture/sceneShape", "/architecture/weightsPath"]` | in `["/architecture/createMetaConcepts", "/architecture/sceneShape", "/architecture/weightsPath"]` (point 1 sets the key false there); its docstring says so. Nothing else changed |

No other assertion was weakened, skipped or marked; no seed chosen.
Neither gate, nor anything a gate uses, was touched.

### 35.7 What in §34.1 could not be built as written, and what was built

1. *MTM's promotion from the opened field is off on the text model*
   (`ConceptualSpace.attentionPromotion` false in `model.xml` and in
   `BasicModel.xml`, so in the scene's configuration). `observe_field`
   returns at once without it, and `promote_candidates` promotes no
   candidate of the field. A unit across modality is made by that
   promotion, so the test's configuration sets it true; it is not set in
   `data/scene/BasicModel_scene.xml` (§34.2 names one key for that file).
   **Proposed:** the felt world's configuration sets it, when its sizes are
   set. *Marked.*
2. *A lesser value is present with a greater one.* By the Pi over values as
   built (§20.2, point 3, "to be measured"), the concept of a value reads
   *for* at a pixel of any greater value. So an image of step 200 presents
   every known lesser value's concept too, and each is counted with the
   word. N4's other pixel is the greater (240); with a lesser (90) the word
   was united with both. In the corpus this is why a word present with
   several images unites with lesser steps, and why one opening reads many
   concepts (the seen capacity of 8 was exceeded by 91 rows of N7 until the
   buckets were raised). Not changed here.
3. *The way down* (35.3): built, the least that does. *Marked.*
4. *The value's symbol* is nominated with its concept (`ImagePercepts.noun`),
   where a word's is made by the kind of words; §34.1 point 2 says each has
   one and not by what. *Marked.*
5. *Frequent words unite too.* Every word of the sentence present with an
   image is paired with it, so `this`, `is`, `.` and `what` unite with
   values as a number word does, and more often (N7). §34.1's bound says
   so for the corpus; nothing tells a name from a word that merely recurs
   with images, which is the open point of the step after.
6. *A unit once made has no single modality* (it reads at readings of
   both), so today's focal/context promotion treats it as any concept: its
   contexts may join it with words and values into classes of order one
   and two. Those are not counted as units (no literal).
7. *What `interpret` gives* with the key false is a concept of order zero
   (the value's), at its row and order; downstream readers already take a
   word that reads as itself (order zero) where no blank was made.

### 35.9 Review of step D2 (Claude, 2026-11-24)

**Accepted as a mechanism; naming does not yet work on the corpus.**
Claude ran both stopping gates, `test_item16_naming.py`,
`test_item16_scene.py`, `test_item16_scene_field.py` and
`test_item16_values.py` (run `20261124-113631-3ca9fa`): all passed.

- *What holds.* On one word with one pixel the four stages hold (N2 to
  N6), and a text model with the key true is as it was.
- *What N7 shows.* Categories 1 and 2 taught, 28 units across modality at
  order one, none of a number word with its value; they are of the
  frequent words (`is`, `.`, `?`). Two causes. (a) Step T's reading for
  only: a lesser value's concept reads at every greater value's pixel, so
  values are not told apart; §36 replaces it. (b) Pairs are counted by
  their being present together alone, and a word present with every image
  reaches the count first. The pair is to be judged by how constantly the
  two go together, each against its own occurrences, and not by how often;
  that is designed after §36 returns, on its numbers.
- *`attentionPromotion`* must be true for a unit to be made, and the test
  sets it; the felt world's configuration is to set it, with its room at
  order zero and one, when the corpus is taught through it.
- *The way down* (`FamiliarTier.raised`) is the source's own way down run
  over the field's rows and the absent member's, with nothing narrowing
  it. Accepted, and marked: the field and what the decoder sees are not
  changed by it, so "the name carries its object's properties" is so far
  a reading that can be asked for and not a state of the field.

## 36. Alec on what a basic percept of intensity is (2026-11-24), and the step that rebuilds it

> If percepts bracket the basic percept in virtue of being its parts and
> wholes, concepts live in a region (of values). They do not take on a
> specific value.

> A basic percept does not have an exact value except for its centroid. It
> generalizes over a mereological region. This is already implemented for
> letter/word percepts, and is in the doc/MD files.
>
> Your definition of a concept is significantly different from what is
> already documented.
>
> The gathering of percepts will be more clear to you when you have a
> codebook of parts, each with values for codes. A linear example, that can
> be extended to 3D:
>
> Input: .75 at location [0,0,0]
>
> Parts: .5, . 4,
> Wholes : 0.8 (must apply everywhere, trivial for a one- position look
> over .where)
>
> So those parts and wholes get a postfix of .where and .when, and 0.5 < x <
> 0.8 may become a basic percept that correctly identifies the area.
>
> So x in that formula is a region in mereological space that is identified
> by the basic percept, and pi on that space becomes a discriminator
> (present parts are evidence, absent parts are negative evidence). The
> same applies to wholes.
>
> We discussed this previously in terms of implicit and explicit
> construction: it can be found by pi, or we can add parts and wholes
> explicitly to the sparse matrix that identifies the constituent basic
> percepts that PI will operate over.
>
> Excuse my frustration, this should not be new; it should be the algorithm
> we are using in the text modality, now applied to intensity.

### 36.1 What stands, and what was built otherwise

**It stands over 29.3, 30.1 and whatever of §28 and §34 says a concept is
a region or a value has a row of its own.** The algorithm of the text
modality is applied to intensity, unchanged:

- A basic percept is a row of the first Pi over its parts and its wholes
  (`BasicPercepts.reserve(parts, wholes)`, their entries in the sparse
  matrix). Its code is the greatest of its parts' codes below and the
  least of its wholes' codes above (`subspace_bounds`). It identifies a
  region of mereological space; it has a centroid and no exact value.
- The Pi is a discriminator: an occurrence that holds the row's parts and
  lies under its wholes is evidence for, and what it lacks or exceeds is
  evidence against (`held_at`, as `read_bank` reads a form).
- A concept of order zero is the Sigma over basic percepts' rows through
  the Not, as documented. Claude's "a concept is the region between a part
  and a whole" described the basic percept.
- "A different Pi on that space" is a Pi of its own, with its own sparse
  matrix, running this algorithm: the two spaces are not mereologically
  related.

*Built otherwise, and to be undone.* Before this item a value's row of the
first Pi was empty (`member_keys` returns none for it), and steps B and T
kept it so: one row for each exact value, with no constituents. Step T
then read for only, with nothing against. Step B's first reading, by
`held_at` with both lanes, was the algorithm, on rows that were exact
values.

### 36.2 Step T2: intensity percepts by the text algorithm

1. *Codebooks with values for codes.* The catalog of parts and the catalog
   of wholes over intensity are the bins of the line of values, in order,
   as `value_code` already lays them out: a part is a lower cut, a whole
   an upper cut. They are given, as the parts of a form and the classes of
   characters are given.
2. *A basic percept of intensity* is a row over a part and a whole of
   those catalogs, with its entries in the Pi of the value space: for an
   occurrence, the greatest part it holds and the least whole it lies
   under, which is its bin. It is nominated by MTM when occurrences recur
   in its region, as a form is, and brings its concept then. Its centroid
   is the mean of the values met in it. No row is kept for an exact value.
3. *Read as a form is,* in a Pi of its own: for where the occurrence holds
   the row's part and lies under its whole; against where it holds the
   part and lies above the whole; and the row's entries on the wires for
   its place. `read_values` by `held_at`'s test on two lanes; step T's
   `parthood` on the `.what` goes. A place's row and the conjunction over
   places are as they are.
4. *The way back* gives, at each place, the centroid of the basic percept
   that a concept of the field reads for there. An image comes back to its
   percepts' centroids: exactly where one value recurs, and within a bin
   otherwise.
5. *Percepts atomic over digits* stays (`intensityWhereWidth`), and with it
   the question of `NumericPercepts.pooled`: with evidence against, see
   whether XOR's "not without the conjunction" holds without withholding a
   value's concept from the readout, and remove `pooled` if it does.

Certificates, in `test/test_item16_values.py` (V3, V9 and V11 rewritten):
V3, a row reads for inside its region, against above it where its part is
held, nothing below it; V9, a region met once has no row, and met as often
as nominates it has one row whatever exact values fell in it; V11, an image
comes back to its percepts' centroids, exactly where each value recurred;
V12, the corpus's nine values each fall in rows of their own and a
reserved step falls in a row its value's other steps made; V5 and the XOR
gate as they are. And N7 of step D2 is run again and reported.


### 36.3 Text for step T2's implementer

~~~
Item 1.6, step T2: intensity percepts by the text algorithm. Read §36 of
doc/plans/2026-11-18-item1-6-felt-math-world.md whole (36.2 is what you
build), then §31 (step T as built), §25.8 and §25.10 (step B), §33 (step
C) and §35 (step D2), and H.5. Then, before anything of intensity, read
how a form's basic percept is made and read, since you are to apply it
unchanged: doc/Mereology.md (the percept's code), doc/Spaces.md ("Level
Sets"), and in bin/BasicPercepts.py reserve, member_keys, against_keys,
subspace_bounds, code, against, held_at, read_bank; the first Pi in
bin/OrderZero.py; how MTM nominates a form's row on recurrence.

Keep tmp/item1-6/regions_progress.md current after each step.

1. Points 1 to 3 of §36.2. Say what the row's key is, what its entries in
   the Pi of the value space are, and where that Pi lives. The text
   modality's Pi and its reading are not to change.
2. Point 4, in NumericPercepts.reverse and in ImagePercepts.put_back.
3. Point 5: say what you measured and whether `pooled` stays.
4. The value beside the words (bin/ScalarPercepts.py) keeps its own
   reading; give it what it needs to be as it was, and say what.
5. test/test_item16_values.py: V3, V9, V11 rewritten, V12 added, as §36.2
   says. Port a test only where a point of §36.2 makes its assertion
   false (a test that asserted an exact value's row, or exact
   reconstruction of values that share a bin), and list each with old and
   new. No other assertion is weakened, skipped or marked; no seed is
   chosen. Neither stopping gate, nor anything a gate uses, is touched.
6. Run test/test_item16_naming.py's N7 again and report its numbers
   beside §35's.
7. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_concept_output.py,
   test/test_item16_values.py, test/test_item16_scene.py,
   test/test_item16_scene_field.py, test/test_item16_naming.py,
   test/test_item16_grid.py, test/test_item17_echo.py,
   test/test_item18_scalar.py, and the files of
   tmp/item1-6/values_run4_files.txt. No test that passes in the sweep
   may fail, apart from those known to fail by chance.
8. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
9. Return as tmp/item1-6/section37.md, headed "## 37. Step T2: the
   return": files changed and why; the row, its key and its entries; each
   certificate with what it printed; the XOR certificate before and
   after; N7 before and after; every test ported; what in §36.2 could not
   be built as written and what you built. You do not stop on a question
   of design: propose, build, mark it. Stop only if a gate, the XOR
   certificate or "without the intersection" cannot pass with its
   assertion as it is, and then say exactly where it fails.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 37. Step T2: the return (Claude on Opus, 2026-11-24)

The work of §36.3, following §36.2. No git was run, and nothing is committed.
Progress was kept in `tmp/item1-6/regions_progress.md`. Every file changed
was first copied to `/tmp/item1-6-2026-11-24/regions_before/` (`bin/BasicPercepts.py`,
`bin/NumericPercepts.py`, `bin/ImagePercepts.py`, `bin/ScalarPercepts.py`,
`test/test_item16_values.py`). The V5 output from before any edit is
`regions_before/v5_before.txt`. Nothing was removed from the tree:
`parthood`'s `.what` half and `felt`'s read of a seat were rewritten in place,
and the previous text is in `regions_before/bin/BasicPercepts.py`.

### 37.1 Files changed, and why

| File | Why |
|---|---|
| `bin/BasicPercepts.py` | **Points 1 and 2.** `cut(index, width)` is the catalogs' cut at a bin, a seat of the table of values. `perceive_intensity(value, count)` makes the row over a part and a whole. `perceive_intensities(values)` does this for every pixel of a batch. `met(row, value, count)` keeps the centroid, a running mean held in two new buffers, `centroids` and `meets`. `value_keys(row)` gives the row's entries in `value_pi`, the Pi of the value space. `value_pi` is a new `SparsePiLayer3`. `reserve` and `reserve_place` write one row to it for each table row, and `set_extra_state` clears it. **Point 3.** `value_events(bank)` is `held_at` on two lanes, times `parthood`. `parthood(bank)` now reads only the `.where` and the `.when`; its `values` argument went, because step T's `.what` comparison went. `read_values(bank, *, where_width)` reads by `value_events`, and its `values` argument went. **Point 4.** `felt(rows)` returns the row's centroid. **Step 4.** `perceive_value` remains as the row of the value beside the words. `_felt_presence` and `_seat` are split out of it and shared. |
| `bin/NumericPercepts.py` | `read` perceives with `perceive_intensities` and nominates under `('value', row)` once for each stream that holds the row. It calls `read_values` without `values`. `reverse` is unchanged as code, and its docstring now says centroid. |
| `bin/ImagePercepts.py` | `perceive` changed the same way as `NumericPercepts.read`. `joined` takes the image's events from `value_events` and no longer calls `parthood`. Docstrings changed; the code of `put_back` is unchanged. |
| `test/test_item16_values.py` | V3, V9 and V11 rewritten, V12 added, and V2, V4 and V10 ported (37.4). |

`bin/ScalarPercepts.py`, `data/` and `doc/` are untouched, and `data/BasicModel.xml`
gains nothing. No key was added: the bins are the ones `value_code` already
lays out. Neither gate's file, nor anything a gate uses, was touched.

### 37.2 The row, its key and its entries (step 1)

- **The catalogs.** A cut is bin `k` of a half `w` wide. It is the seat in
  the table of values of the bin's middle, `(k + ½)/w`
  (`BasicPercepts.cut`). Its code in that half is `value_code`'s: the bins
  at or below `k`. As a part it is a lower cut, and as a whole an upper
  cut. The cuts are given by `value_code`, and a seat is written when the
  cut is first named. *Marked:* a cut is held as a seat of the existing
  table of values, so `_boundary_codes`, `subspace_bounds`, `is_value` and
  `code` read it with no change.
- **The row.** For an occurrence of value `v`, the row is over the
  greatest part `v` holds, `cut(value_bin(v, iw), iw)`, and the least whole
  it lies under, `cut(value_bin(v, pw), pw)`. That is its bin in each half.
  - **Its key** in `index` is `((part,), (whole,))`, the two cuts and never
    the value. V9 prints it: `((-4,), (-4,))` on MM_xor for the middle third.
  - **Its code** is `subspace_bounds`, as for a form: the greatest of its
    parts' codes and the least of its wholes' codes.
  - **Its centroid** is the mean of the values met in it, counted once per
    occurrence. `met` keeps it as a running mean, so a value met again
    where it alone was met leaves the centroid exactly that value.
  - **Nomination.** The row is reserved at first sight and nominated by MTM
    on `conceptMintCount` readings under `('value', row)`. It is promoted
    and brings its concept then (`Places.bring`, as in step T).
- **Its entries in the Pi of the value space.** `('ps', (part,))` and
  `('ws', whole)`: the same key shapes a form has in the first Pi.
  - **Where that Pi lives:** `BasicPercepts.value_pi`, a `SparsePiLayer3` of
    its own, sized `capacity` rows by `2 x capacity` entries.
  - **Row numbering:** row `r` of `value_pi` is row `r` of the table. Places,
    conjunctions and forms have empty rows there.
  - **The first Pi is unchanged.** A value's row has no entries in it
    (`member_keys` returns none for a value, as before). Its against
    entries, `_refresh_pi` and `read_bank` are untouched.
- **The reading (point 3).** `read_values` reads each occurrence by
  `held_at(bank, bank.table)` on two lanes:
  - for is `part * whole`;
  - against is `part * (1 - whole)`;
  - each is times activity and `parthood(bank)`, the `.where` read by bounds
    and the `.when` by `field_factor`.

  A place's row is read in the same way on the `.what` it keeps, which is the
  cone of the values met at it, and by its wires. A conjunction is read
  through the first Pi over its members, and `held` and `intensityWhereWidth`
  apply as before. This is step B's first reading, now on rows that are
  regions.

### 37.3 The certificates, as printed

From the development runs; the bounded run's outcomes are in 37.6. The
widths are MM_xor and MM_image 3 and 3, and the scene model 64 and 64.

- **V3, read as a form is.** MM_xor, with the values `[[.1,.2],[.5,0],[.9,1.],[.5,.2]]`.
  - **Regions:** `.1` and `.2` share one row, `.9` and `1.` share one row,
    so there are three rows in all.
  - **Readings** (each pair is for, against): `region 0 at 0.1: (1,0)`,
    `region 0 at 0.5: (0,1)`, `region 0 at 1.0: (0,1)`,
    `region 1 at 0.2: (0,0)`, `region 1 at 0.5: (1,0)`,
    `region 1 at 0.9: (0,1)`, `region 2 at 0.5: (0,0)`,
    `region 2 at 1.0: (1,0)`, and `(0,0)` wherever nothing is.
  - **Keys and entries:** `keys {0: ((-4,),(-4,)), 1: ((-5,),(-5,)), 2: ((-6,),(-6,))}`.
    The entries are `([], [('ps', (-4,)), ('ws', -4)])` and so on: none in
    the first Pi, and two in `value_pi`. The cuts' values are
    `[0.1667, 0.5, 0.8333]`. `value_events` names `held_at`, `read_values`
    names `value_events`, and `numeric_read_uses: ['read_values', 'value_bank']`.
- **V9, a region nominated on recurrence.** MM_xor, `mint_count 3`. Three
  different values in the middle third were presented once each:
  - `0.4167`: rows `[0]`, not promoted, no concept;
  - `0.5`: rows `[0]`, not promoted, no concept;
  - `0.5833`: rows `[0]`, promoted, concept `0`.

  The key is `((-4,),(-4,))`, equal to the region's cuts. The centroid is
  `0.49999997` against a mean of `0.49999999`.
- **V11, back to the centroids.** MM_image. The image holds `1/12` alone in
  the first third, `11/12` alone in the last, and `5/12` and `7/12` in the
  middle third. After recurrence, `back_by_value: {0.0833: [0.0833333358], 0.9167: [0.9166666865], 0.4167: [0.5], 0.5833: [0.5]}`.
  The mean of the shared values is `0.49999999`; the two lone values come
  back exact (`torch.equal`). `silent_greatest 0.0` and
  `reverse_names_percepts False`.
- **V12, the corpus's values.** On the scene model, every free step of
  each of the nine values was perceived:
  - **Rows:** `{1: [0,1,2], 2: [3,4,5], 3: [6,7], 4: [8,9], 5: [10,11,12], 6: [13,14,15], 7: [16,17], 8: [18,19], 9: [20,21,22]}`.
    No row is shared between two values. With bins about 4 steps wide, a
    band of 7 steps falls in 2 or 3 rows.
  - **Reserved steps:** `{1: (26, 1), 2: (49, 4), 3: (76, 7), 4: (99, 8), 5: (126, 11), 6: (149, 14), 7: (176, 17), 8: (199, 18), 9: (226, 21)}`.
    Each is in its value's rows, and `rows_made_by_reserved 0`.
- **V2 (ported).** `first_presentation_greatest 0.0`. After recurrence, `back` equals
  the centroid at every place (`0.17567566` for every grid value of the first
  third); `silent_greatest 0.0`.
- **V4 (ported).** `place_rows_at_the_two_places [[1,0],[0,1]]`; `lit 63, places 63`.
- **V10 (ported).** The value's row at the places is `[[0,0],[0,1],[1,0],[1,1]]`
  and nothing at the bracket. Its concept at the bracket is `[0,0,0,0]`, and
  `as_the_readout_reads_it [0,0,0,0]`. The spans are
  `{0: value 1, 1: place 1, 2: place 1, 3: conjunction 2}`.
- **V5 and the gates.** In 37.5, and in 37.6.

### 37.4 Tests ported

Each port is in `test/test_item16_values.py`. No test outside that file was
changed, and no seed was chosen.

| Test | Old | New |
|---|---|---|
| V2 `test_v2_input_in_0_to_1` | after recurrence `torch.equal(back, given)` for the 37 grid values | `torch.equal(back, centroid)`, where `centroid` is `basic.felt` of each place's row read from the input. The grid's 37 values share three bins, so exactness is false by point 2. Every other assertion is as before |
| V3 | step T's numerical parthood | rewritten as §36.2 says (37.3) |
| V4 | `basic.parthood(bank, values)`; `basic.perceive_value(1.)` for the image's value row | `basic.parthood(bank)`, because the `.what` half went (point 3); `basic.perceive_intensity(1.)`, because `perceive_value` is now the scalar's row. The assertions are as before |
| V9 | a value nominated on recurrence | rewritten (37.3) |
| V10 | `basic.perceive_value(1.)`; `read_values(bank, values, where_width=1)` | `perceive_intensity(1.)`; `read_values(bank, where_width=1)`. The assertions are as before |
| V11 | exact reconstruction of the 37 grid values | rewritten: back to the centroids (37.3) |
| V12 | — | new |

### 37.5 XOR before and after; `pooled` (step 3)

| | Before | After |
|---|---|---|
| answers | `[1.8189894035458565e-11, 1.0, 1.0, 1.8189894035458565e-11]` | the same |
| weights | `[1.313035249710083, 1.313035249710083, -2.626070499420166]` | the same |
| against | `[0.0, 0.0, 0.0]` | the same |
| mse | `1.6543612251060553e-22` | the same |
| promoted_row | 3 | 3 |

**`pooled` stays.** I measured it by bypassing `NumericPercepts.pooled` with
a plugin that changes no file (`scratchpad/nopool/nopool_plugin.py`):

- "without the intersection" failed three times out of three, at `3.0883973221700056e-14`.
- The certificate's places' weights fell to `1.19e-8`, with the conjunction's at `-1.313`.

The reason is that evidence against does not reach XOR. Its inputs are 0
and 1, so there is one region, the last bin, and nothing lies above it.
The value's row is therefore never read against, and the readout's
greatest over the places is still an OR of the two inputs.

### 37.6 N7 (step 6) and the run (step 7)

N7 was run before and after this step, on two copies of the tree in the
scratchpad: the unchanged tree and the changed one. No seed is set.

| | units by order | words in a unit (number of steps) | number words united |
|---|---|---|---|
| §35 | `{1: 28}` | `is 10, this 2, what 2, ? 3, that 1, . 10` | none |
| before, this run | `{1: 42}` | `that 2, seven 2, is 13, this 4, nine 1, two 1, what 3, ? 5, eight 1, . 10` | eight [175], nine [72], seven [51, 125], two [47] |
| after | `{1: 10}` | `is 4, this 4, what 1, ? 1` | none |

300 rows were taught and none refused, before and after. After this step a
lesser value's concept no longer reads at a greater pixel, and the units
fall from 42 to 10. No number word is united with its value.
- **Why the frequent words are left:** pairs are still counted by being
  present together (§35.9, cause b), and a frequent word is present with
  every image.
- **Why the number words reach no count:** at width 64 a value's band falls
  in 2 or 3 rows (V12), so a number word's count is split across them.
  Each row then falls further below the count that nominates a unit.

**Bounded run.** `env -u RUN_TRAINED -u RUN_SLOW ... test/test_report.py`
with the 60 selectors of `tmp/item1-6/regions_run_files.txt`: §36.3 step 7's
list and `values_run4_files.txt`. The log is `tmp/item1-6/regions_run1.log`.

**`Result: test_failure, exit 1; 533/533 cases completed.`**
`output/tests/20261124-120650-171502/result.json`, 1291 s: 486 passed, 44
skipped, 3 failed.

- **Both gates passed:** `test_mm_xor::...::test_convergence` and
  `test_item25_completion::test_a_previously_named_shared_concept_denotes_its_raised_class`.
- `test_concept_output.py` passed whole, including "without the intersection" [0], [1] and [2].
- V1 to V12, N7, and G1 for `MM_xor` and `MM_image` passed.

The three failures:

- `test_item17_symbols::test_54bd_...` is `failed` in the sweep.
- `test_item18_perceptual_scalar::test_text_uses_the_same_percept_collection_reader`
  is on the list of chance.
- `test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`
  is on the list of chance.

The 44 skipped are skipped in the sweep, or are trained tests left out with
RUN_TRAINED unset. Every other test has its sweep outcome.

### 37.7 The value beside the words (step 4)

It keeps its own reading. What it needs, and has, is `perceive_value`: a row
whose part and whole are both the cut at the value itself, so its region
holds that value alone. That row's centroid is the value, which is what
`felt` returns for it, so `ScalarPercepts.decode` and `reconstruct` give
the number they gave before. Rows of the scalar have entries in `value_pi`
as well. `ScalarPercepts.py` is unchanged.

### 37.8 What in §36.2 could not be built as written, and what was built

1. *"A region met once has no row"* (V9). As for a form, the row is
   reserved at first sight: the occurrence needs a row id for
   `field.percepts`, and V1 reads that id on a fresh model. **Built:** "no
   row" means not promoted, not read by the Pi over values, and no
   concept. *Marked.*
2. *The catalogs as given.* The cuts are seats of the table of values at
   each bin's middle, written on first use, not a separate store.
   - A value beside the words that falls exactly on a cut's middle shares
     that seat, and with widths equal it would share that region's row.
   - No model reads both modalities. *Marked.*
3. *The centroid is weighted by occurrences:* each pixel of each
   presentation counts. *Marked.*
4. *The scalar keeps a row for its own value* (37.7), as step 4 asks. That
   is the one row left whose region is a single value.
5. *`pooled`* stays (37.5).
6. *A saved state.* `centroids` and `meets` are new buffers. A state saved
   before this step lacks them, so a strict load will refuse it (not run).
   Nothing is kept for the older form.

### 37.9 Review of step T2 (Claude, 2026-11-24)

**Accepted.** Claude ran `test_mm_xor.py` whole, the other stopping gate,
`test_item16_values.py`, `test_item16_scene_field.py`,
`test_concept_output.py`, `test_item18_places.py` and
`test_item18_scalar.py`: 42 of 42 passed.

- A basic percept of intensity is now a row over a part and a whole in a
  Pi of the value space's own, read for and against by `held_at`, with a
  centroid: the text algorithm, on intensity.
- `pooled` stays: XOR's one value lies in the last bin with nothing above
  it, so nothing reads against its row, and the readout's pooling over
  readings still makes an OR of the two places without it.
- *N7 again:* ten units across modality, of `is`, `this`, `what` and `?`,
  and no number word with its value. At 64 bins a value's steps fall in
  two or three rows, which divides its count; and the pairs are still
  counted by step D2's own counter, on positive evidence. §38 takes that
  counter out.


## 38. Step D3: the name by MTM's ordinary promotion (Alec, 2026-11-24; the text for the implementer is 38.2)

Shown N7 (no number word united with its value; the frequent words
united instead):

> My guess is that you are using positive evidence to create the
> association, and neglecting the negative evidence. "is" and "10" would
> wire together because they fire together, but they dont becuase "is" is
> often present without "10" being found in either a physical or mental
> context (the latter is necessary because the word can occur without the
> object being present). We might also need more co-occurrence trials,
> after we have ensured that they can repeat the words (and thus presumably
> know the words).

> word-object association is just another case of MLM promotion; it should
> not differ in basic form, and I think MLM is already using PI and SIGMA
> to make these associations implicitly? So Im not sure we need an explicit
> formula for their creation.

> your comment about sigma being necessary for meta concepts might dictate
> that PI can promote single elements (thus making the concept available
> for the sigma). If that is necessary, please make that happen (in a
> general way, not just for things known to be words).

("MLM" is read as MTM.)

### 38.1 What they decide

1. *No rule of its own for a name.* Step D2's counter of pairs across
   modality (`FamiliarTier.observe_units`, with `modalities`, `apart` and
   the branch on modality in `observe_field`) is the explicit formula, and
   it counted on evidence for alone. It goes. A word with its object is a
   candidate of MTM's ordinary promotion, which keeps a lane for and a
   lane against for every candidate and promotes none that is read both
   (`observe_candidate`, `promote_candidates`).
2. *A Pi may promote one element.* A concept that MTM promotes may be
   promoted alone by the Pi of the next order, a row of one member, which
   is its symbol and makes it available to a Sigma. It is general: of any
   concept, of any modality. `ImagePercepts.noun` goes.
3. *Words first, then trials.* The words are known, which the echo shows,
   before the trials of word and object together; and the number of trials
   is measured, not assumed.
4. *Measured before anything more is built.* What the ordinary promotion
   makes of a word and its object is measured first, and nothing is added
   to make the unit come out.

This stands over 34.1, point 3, and 35.9's "designed after".

### 38.2 Text for the implementer

~~~
Item 1.6, step D3: the name by MTM's ordinary promotion. Read §38 of
doc/plans/2026-11-18-item1-6-felt-math-world.md whole, then §34 and §35
(step D2 as built and its review), §37 (step T2), §19.5, §22.1 (question
1) and H.5; doc/Orders.md ("What makes what", "Where a noun begins") and
doc/Architecture.md ("What co-occurs is a necessary part of the whole it
forms; what substitutes is a sufficient part of the kind it forms"). Then
bin/FamiliarTier.py whole, and how MTM's pool keeps a candidate's
readings (bin/Memory.py or where owner.mtm is defined).

Keep tmp/item1-6/promotion_progress.md current after each step.

1. Point 1 of §38.1: take step D2's own counter and its branch on
   modality out. A concept of an image and a concept of a word in one
   opened field are offered to MTM as any two concepts are.
2. Point 2: the general rule, in MTM's promotion. Say where; say which
   makers of a word's symbol it now duplicates and whether they can go
   (do not remove one a gate or a passing test needs; list it instead).
3. data/scene/BasicModel_scene.xml sets attentionPromotion true, so that
   MTM promotes from the opened field there. No other configuration
   changes.
4. Measure, before building anything more. On the text model with the
   scene's box, key false: teach words alone until the model says each
   back; teach values alone until their regions and places are known;
   then trials of a word with its object, in two sets: (a) made in the
   test: `three` with one value every time, `is` with every image; (b)
   category 2 of the corpus. After each ten trials, up to a number you
   choose and state, record: the candidates MTM holds that have a member
   of each modality, each one's readings for and against, what was
   promoted (a Pi's row or a Sigma, at what order, over which members),
   and whether the number word and its value stand under one concept.
   Write the table into the return.
5. FamiliarTier.units and across, Interpret.named and
   Assessment.units_across_modality read what the ordinary promotion
   made: a concept of order one or above with members of two ranges of
   the registry. No tag marks it.
6. test/test_item16_naming.py: N1, N2 and N7 as they are. N3, N5 and N6
   are asserted of what the ordinary promotion makes in set (a). N4
   becomes: `is` with every image and `three` with its own value;
   `three` is united with its value and `is` with none. If the ordinary
   promotion does not make the unit, or unites `is`, do NOT add a rule to
   make it come out: leave N3 to N6 failing in the test, and say in the
   return exactly what it makes instead and which condition of
   observe_candidate or promote_candidates decides it, with what you
   would propose. That report is then the result of this step.
7. No other assertion is weakened, skipped or marked; no seed is chosen.
   Neither stopping gate, nor anything a gate uses, is touched. A text
   model with createMetaConcepts true and attentionPromotion as it is
   must behave exactly as before: say how point 2 is kept from changing
   it, or what it changes.
8. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_item17_echo.py,
   test/test_item17_symbols.py, test/test_item16_scene.py,
   test/test_item16_scene_field.py, test/test_item16_values.py,
   test/test_item16_naming.py, test/test_concept_output.py, and every
   test file that names observe_field, promote_candidates,
   nominate_symbol or maybe_raise_order. No test that passes in the sweep
   may fail, apart from those known to fail by chance (and N3 to N6 under
   step 6).
9. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
10. Return as tmp/item1-6/section39.md, headed "## 39. Step D3: the
    return": files changed and why; step 4's table; what the ordinary
    promotion makes of a word and its object and why; each certificate;
    what a text model with the key true now does differently, if
    anything; what you would propose.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 39. Step D3: the return (Claude on Opus, 2026-11-24) (Claude on Opus, 2026-11-24)

The work of §38.2, with the owner's two changes sent during it (39.3 and
39.6). No git was run; nothing is committed. Progress was kept in
`tmp/item1-6/promotion_progress.md`. Every file changed was first copied to
`/tmp/item1-6-2026-11-24/promotion_before/`. What was removed (step D2's
counter, `ImagePercepts.noun`) is there in those copies; nothing was
deleted otherwise.

**The result in four lines.** The ordinary promotion makes no unit of a
word and its object, in set (a) or in set (b). A candidate that holds a word
and a value never recurs: each opening opens a new one, with one reading,
because MTM identifies a candidate across openings only by the exact
proportions of its support's weights, and a word's lane and a value's lane
differ in their fifth digit and drift with learning
(`MediumTermMemory.match_candidate`; the parthood match beside it gives 0.0
for every pair, identical supports included). What it does promote is
concepts of order zero over values alone, whose weights are equal: an
image's value with the lesser values read against at it.

### 39.1 Files changed, and why

| File | Why |
|---|---|
| `bin/FamiliarTier.py` | **Step 1.** `MODALITIES`' old use, `ACROSS`, `modalities`, `apart`, `observe_units` removed; `observe_field` offers every concept's context as before step D2 (the focal row's context is every other offered row), and `promote_candidates` has no branch for `ACROSS`. The code of `observe_field` and of `promote_candidates`' literal branch is now that of before step D2 (`/tmp/item1-6-2026-11-24/naming_before/bin/FamiliarTier.py`). **Step 2.** `promoted_alone(owner, concept, handle)`: the owner's revised point 2 (39.3). **Step 5.** `percept_range`, `ranges`, `held_members`, and `units` rewritten: a unit is read from what the promotion made, not from a tag (39.4). `raised` and `across` unchanged. |
| `bin/ImagePercepts.py` | `noun` removed, and its call in `perceive`. |
| `bin/Interpret.py` | At a word's admission (both of `lookup_word`'s making paths), `promoted_alone(cs, word, ('alone', word))`. Nothing else. |
| `data/scene/BasicModel_scene.xml` | **Step 3.** `<attentionPromotion>true</attentionPromotion>` in `architecture`, beside `createMetaConcepts`. No other configuration changed; `data/BasicModel.xml`, `model.xml`, `model.xsd` untouched. |
| `test/test_item16_naming.py` | **Step 6.** Set (a) in the stages fixture; the words said back first (`said_back`, the echo test's teaching); N4 now of set (a); `attentionPromotion` read from the scene's configuration; N3 to N6 marked expected-to-fail by the owner's word (39.6). |
| `test/test_item16_scene.py` | S1 ported (39.7). |
| `test/test_item17_symbols.py` | `test_71`'s list of the callers of `nominate_symbol` ported (39.7). |
| `tmp/item1-6/promotion_measure.py`, `promotion_measure_{a,b}.json`, `promotion_compare.py`, `promotion_run*` | The measurement of step 4, its records, and the runs. Not source. |

`bin/Places.py` was changed and changed back (the first form of point 2
gave every concept MTM brings its symbol there); it is as it was.

### 39.2 Step 1: no rule of its own for a name

A concept of an image and a concept of a word in one opened field are now
offered to MTM as any two concepts are: in `observe_field`, each witnessed
concept that reads above `conceptUseFloor` (on either lane; places' concepts
excepted, as before) is a focal row, its context the other such rows, and
`observe_candidate(owner, order of the focal, context, pair = the focal's
(for, against))` opens or updates a candidate. `promote_candidates`
promotes a candidate whose readings within the horizon number
`conceptMintCount` and that is not read both (`min(max for, max against) >
0` refuses it).

### 39.3 Step 2: a Pi promotes one element, of text alone

The owner, sent during the step:

> The text modality promotes its zero-order concepts to first order symbols
> automatically. The reason for this is that in text, there is no need to
> generalize over correctly written text (other than typos). A concept
> (word) is represented in ASCII the same way, every time, so that the
> purpose of the PI in generalizing over that zeroth-order conceptual space
> is not needed. Presumably we should do this only for concepts whose
> members are exclusively from text modality.

*Where.* `FamiliarTier.promoted_alone(owner, concept, handle)`: where MTM's
promotion is on (`ConceptualSpace.attentionPromotion`) and every basic
percept of the concept of order zero lies in the range `text`
(`ranges(owner, concept) == {'text'}`), the concept is promoted alone by the
Pi of order one: the symbol that already holds it (its fit) is kept, else
one is nominated (`nominate_symbol(handle, concept)`), a row of that one
member. It is called where MTM's nomination brings a concept of order zero:
at a word's admission (`Interpret.lookup_word`, after the nomination
`('form', …)`, handle `('alone', word)`) and in `promote_candidates` for a
candidate promoted at order zero (handle the MTM key). Above order zero
`promote_candidates` already made the concept's symbol and still does.

*The range of a basic percept* (`percept_range`) is read from its places,
not from its being a word: a place's row by its cone decoded to an address
of the registry; a conjunction by its members; a value's row is read at an
image's places (`scene`); any other row of
the first Pi is read at a sentence's word brackets, which lie in `text`.
*Marked:* a value's row and a form's row carry no place of their own
(their `.where` entries are empty), so their range is the range of the bank
that reads them, which is fixed by their kind of row.

*A concept with any member outside the text* (an intensity's, a place's) is
not promoted automatically. Nothing new gives an intensity's concept its
symbol; `ImagePercepts.noun` is gone. What the ordinary promotion gives it
is in 39.5: nothing. No value's concept has a symbol after set (a) or set
(b), so the two or three region rows that one value's steps fall in (in set
(b): 23 rows for 9 values, e.g. steps 23, 24 and 28 for one) come under no
symbol and are gathered by nothing.

*Which makers of a word's symbol it duplicates.* The kind of words'
`things_reading` (`('kind', words, word)`, through `add_concept_edge` /
`_populate_concept_weights` / part learning). In the naming model the
kind's symbol is made first: `promoted_alone` finds it as the fit and makes
none (probe: the word 93's call returned symbol 0, handle `('kind', 91, 93)`).
`things_reading` cannot go: every kind's entry on a member with no symbol
is made there, it is the sole maker of symbols in the lesson's model
(`test_71`, 22 symbols, all `('kind', …)` or MTM's), and the gates' models
use it.

*How a text model with `createMetaConcepts` true is kept as it was* (step
7, and the coordinator's question). The rule acts only where
`attentionPromotion` is true; `data/model.xml` and `data/BasicModel.xml`
have it false, so the text model does exactly as before. Where a text model
has it true (the corpus lesson configurations, `MM_corpus_*`,
`BasicModel_corpus`), a word's symbol is made at admission unless one
already holds it; in the lesson's model of `test_71` every word was already
held by the kind's symbol when the rule ran, so it made none (22 symbols,
every handle `('kind', …)` or an MTM key; `test_71` passes). No passing test of the runs changed its outcome (39.8). Both
gates pass untouched.

### 39.4 Step 5: what reads a unit

`FamiliarTier.units(owner)`: each concept above order zero whose Sigma has
an entry of weight above zero on a symbol holding a concept that lies in
`scene` alone and on one holding a concept that lies in `text` alone
(`held_members`: the store's entries, symbol columns, the symbols' members;
`ranges`). No tag marks it. `across`, `Interpret.named` and
`Assessment.units_across_modality` read `units` and are unchanged. A model
with no range `scene` has none.

### 39.5 Step 4: the measurement

`tmp/item1-6/promotion_measure.py`, on the text model with the scene's box
(`test_item16_naming.naming_model`: the echo model, the scene's
`createMetaConcepts` false and `attentionPromotion` true, room
`CORPUS_ROWS` 512 and buckets `8,256`). Records:
`tmp/item1-6/promotion_measure_a.json`, `…_b.json`.

*Words first.* Each word taught alone (`known`, four presentations), then
as the supervised answer of itself as the echo test teaches it, until it is
said back: set (a)'s `three` and `is` were said back after one round; set
(b)'s 13 words (`eight five four is nine one seven six that this three two
what`; the marks `.` and `?` not taught alone) after one round, 13 of 13.
*Values* alone, each image four times, until its region's row and its place
were known: set (a) 3 values (steps 90, 200, 240); set (b) 23 region rows.
*Trials:* set (a) 120 (30 cycles of `three` with step 200, then `is` with
200, 240 and 90); set (b) all 200 rows of category 2 (train), in order.
Recorded after each ten.

Columns: the candidates MTM held that have a member of each modality; the
greatest number of readings any of them had within the horizon; how many
were read both (for and against at once); what was promoted; units across
modality; symbols in the table (all of one member); whether the number word
and its value stand under one concept.

**Set (a).**

| Trials | Mixed candidates | Most readings | Read both | Promoted | Units | Symbols | `three` and v200 under one |
|---|---|---|---|---|---|---|---|
| 10 | 18 | 1 | 0 | c103, order 0, over v200, v90 (no symbol: not text) | none | 2 | no |
| 20 | 28 | 1 | 0 | c104, order 0, over v200, v240, v90 | none | 2 | no |
| 30 | 30 | 1 | 0 | nothing | none | 2 | no |
| 40 to 120 | 28 or 30 | 1 | 0 | nothing | none | 2 | no |

The two symbols are `three`'s and `is`'s. No value has a symbol. A mixed
candidate's readings are its focal value's lanes: e.g. focal v200, support
`{'three': 0.761632, v90: 0.761594}`, for 0.762, against 0; focal v90 (read
against at 200), support `{'three', v200}`, for 0, against 0.762; focal
v240 with `is`, for 0.762; focal v200 at `is` with 240, against 0.762. At the
next `three` trial the same focal opens a new candidate with `{'three':
0.761654, v90: 0.761604}`. The keys run from 2 to 188 over the 120 trials.

**Set (b).**

| Trials | Mixed candidates | Most readings | Read both | Promoted | Units | Symbols | Number words under one with a value |
|---|---|---|---|---|---|---|---|
| 10 | 34 | 1 | 2 | nothing | none | 32 | none |
| 20 | 27 | 1 | 3 | nothing | none | 32 | none |
| 30 | 35 | 1 | 3 | nothing | none | 32 | none |
| 40 | 33 | 1 | 3 | nothing | none | 34 | none |
| 50 | 32 | 1 | 3 | nothing | none | 37 | none |
| 60 | 24 | 1 | 2 | nothing | none | 40 | none |
| 70 | 19 | 1 | 1 | nothing | none | 43 | none |
| 80 | 23 | 1 | 2 | nothing | none | 43 | none |
| 90 | 35 | 1 | 3 | nothing | none | 43 | none |
| 100 | 34 | 1 | 1 | nothing | none | 46 | none |
| 110 | 26 | 1 | 1 | nothing | none | 46 | none |
| 120 | 32 | 1 | 2 | nothing | none | 46 | none |
| 130 to 190 | 27 to 35 | 1 | 1 to 3 | nothing | none | 49 | none |
| 200 | 25 | 1 | 0 | nothing | none | 52 | none |

Every symbol is of one member, a concept of the text (the words, and text
concepts admitted during the trials); no value's region row is under any
symbol (`value_symbols` empty for all 23). A typical candidate: focal `?`,
support `{'is', 'this', 'what', v23, v24, v28, v47}`, one reading: an image
of step 47 reads the three rows of value 1 against, so every image brings
every lesser value's rows into the context.

**Two experiments, no source changed** (`scratchpad/d3/probe3.py`,
`probe4.py`), to find the deciding condition:

1. The parthood match in `match_candidate` scores `shared_part` 0.0 for every
   pair of candidates, the same support included (e.g. `[0, 8]` against
   `[0, 8]`): the members' codes in the similarity codebook give nothing.
   With the bar at 0.99 instead of `conceptMatchPart` 1.0 nothing changes.
2. With a candidate identified by its members alone (`set(support)`), set
   (a) promotes after 10 trials concepts of **order zero**: `'three'` with
   v200, `'three'` with v90, `'is'` with v200, `'is'` with v90, and later
   `'is'` with v240 and v90, with v200 and v90, with v200 and v240; and the
   same pair again and again (`'three'`/v200 as c105, c113, c120). `is` is
   united as readily as `three`, and with the lesser value read against.
   Still no unit across modality (order 0, not a Sigma over symbols).

### 39.6 Step 6: the certificates, and the marks

The owner, sent during the step: "lets leave the testpoint for automatic
metaconcept formation, and mark it as Xfail". By that instruction alone,
each of N3 to N6 that fails is marked `@pytest.mark.xfail(strict=False,
reason=…)`; its body is the full certificate, so it reports XPASS the day
the formation works. N1, N2 and N7 are not marked.

| | Outcome | Mark and reason |
|---|---|---|
| N1 | passes | none |
| N2 | **fails**: the value's concept has no symbol (`value_symbols []`; the word's `[0]`) | none (not to be marked). The cause is the owner's revised point 2: an intensity's concept is not promoted automatically, and the ordinary Pi of order one makes no symbol for it |
| N3 | xfail | "The ordinary promotion makes no concept of order one over the word and its value: a candidate that holds both never recurs, since its support's proportions differ at every opening (MediumTermMemory.match_candidate), so no unit across modality is made." |
| N4 | xfail | "The ordinary promotion unites neither word with any value; what set (a) promotes is concepts of order zero over values alone, an image's value with the lesser values read against it." |
| N5 | xfail | "No unit is made by the ordinary promotion, so neither the word nor its pixel met alone raises one, and the way down raises nothing." |
| N6 | xfail | "No unit is made by the ordinary promotion, so interpret of the word gives the word itself." |
| N7 | passes (a measure): 300 rows; units by order `{}`; no word united with any step | none |

What they print (final run): N1 `key true: {'definitions': 1,
'blanks': 1}, the word's blank object 95; key false … {'definitions': 0,
'blanks': 0}`; N2 said back `{'three': 'three', 'is': 'is', 'five': 'five'}`, the word
`three` 93 symbols `[0]`, the pixel 99 symbols `[]`, units `{}`; N3 units `{}`; N4 `three` united with `()`, `is` with
`()`; N5 the unit reads `(0.0, 0.0)` either way, the way down unchanged; N6
interpret of `three` gives 93 (itself), not 99.

In N5 the record of the unit's reading is `(0., 0.)` where there is no unit
(it was `None`, which made the assertion a TypeError rather than a failure).
No assertion changed.

### 39.7 Tests ported

| Test | Old | New |
|---|---|---|
| `test_item16_scene.py::test_s1_the_key` | the scene's configuration differs from `BasicModel.xml` in `createMetaConcepts`, `sceneShape`, `weightsPath` | and in `attentionPromotion` (step 3) |
| `test_item17_symbols.py::test_71_…` | callers of `nominate_symbol`: `promote_candidates`, `promote_identity`, `ImagePercepts.noun`, `things_reading` | `promote_candidates`, `promote_identity`, `promoted_alone`, `things_reading` (point 2: `noun` goes, the rule is MTM's). The test's second half (every symbol MTM's or a kind's) is unchanged and passes |
| `test_item16_naming.py` N4 | its own model: the word with its pixel, some times also with a greater one | set (a) of the stages fixture, as step 6 writes it |

### 39.8 The runs

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py`
on the 53 files of `tmp/item1-6/promotion_run_files.txt`: both gates'
files, §38.2 step 8's list, every test file that names `observe_field`,
`promote_candidates`, `nominate_symbol` or `maybe_raise_order`, and every
test file that names a configuration with `attentionPromotion` true (where
the automatic promotion of text can act). Outcomes compared with the sweep
by `tmp/item1-6/promotion_compare.py`.

- **Run 1** (`output/tests/20261124-124745-196580`): `Result: memory, exit
  137; 338/769 cases completed.` A worker passed 8 GiB in
  `test_compiled_lesson_step.py::test_one_lesson_sentence_trains_with_the_compiled_output_walk`
  (passed in the sweep). Run alone it does the same on this tree (`Result:
  memory`, 911.7 s, 8.00 GiB) and on a copy of the tree as it was before
  this step (`scratchpad/d3/before_tree`, 913.3 s, 8.00 GiB): it is not this
  step's. It is left out of the final run.
- **Run 2** (the 431 cases run 1 did not reach): three failures of
  `test_reading_round.py` (passed in the sweep) were this step's: an owner
  with no model reached `percept_range`'s read of the registry from a word's
  admission. Fixed: the registry is read only for a place's row. The three
  pass.
- **Final run**, the 768 cases of run 1 less the one above
  (`tmp/item1-6/promotion_run3_selectors.txt`, log `promotion_run3.log`):
  **`Result: test_failure, exit 1; 768/768 cases completed.`**
  `output/tests/20261124-140636-9fc4c6/result.json`. 739 passed, 23 skipped,
  4 xfailed, 2 failed. **Both stopping gates passed.** The failures: N2
  (39.6) and `test_item17_symbols::test_54bd_…`, failed in the sweep. N3 to
  N6 xfailed; N1 and N7 passed. Every other test has its sweep outcome.


### 39.9 What I would propose

1. *The identity of a candidate across openings.* The ordinary promotion
   cannot count a pair it cannot recognize twice. `match_candidate`
   recognizes by exact proportions of graded weights, and the parthood
   match reads codes these concepts do not have. Recognize a candidate by
   its members (as the experiment did), its weights summed as evidence.
   That is a change to MTM for every candidate, not a rule for names, and
   would need its own measure on the gates and the corpus models.
2. *Then the order.* With (1), the context of a focal of order zero is
   promoted at order zero, a field concept over a word and values, which
   §19.5 point 2 and §22.1 put at order one. For the meta concept to come
   of the ordinary promotion, a candidate over concepts of two modalities,
   or any candidate over concepts that have symbols, would be promoted one
   order above its members (`promote_candidates` already does so for
   `entry.order > 0`), as a Sigma over their symbols. That needs the value's
   concept to have a symbol, which after the owner's revision only the
   ordinary Pi of order one gives, and it gives none here: what of order one
   makes a symbol over a value's concept, or over the rows of one value's
   steps, is the open point.
3. *Negative evidence.* As built, a candidate's two lanes are its focal's
   reading, not the pair's: "read both" asks whether the focal concept was
   read for and against, not whether the word occurred without the object.
   The owner's "`is` is often present without `10`" is not what the lane
   against counts. With (1), `is` was united as readily as `three`. The
   lane against a candidate would have to be written at an opening that
   holds one member and not the other.
4. *A lesser value read against is present.* `used` takes a concept whose
   greater lane is above the floor, so a value read only against is part of
   every greater image's context (v90 at 200; set (b)'s three rows of value
   1 at step 47). A context of concepts read for would remove those pairs.

### 39.10 After review: N2 ported, and the meta concepts minted again

Done at the coordinator's request after the review. The files were copied
first to `/tmp/item1-6-2026-11-24/promotion_before2/`.

1. *N2 ported.* The owner's change (automatic promotion is for text alone)
   makes its assertion that a value's concept has a symbol false.

   | Test | Old | New |
   |---|---|---|
   | `test_item16_naming.py::test_n2_each_known_alone` | `alone['word_symbols'] and alone['value_symbols']` | `alone['word_symbols'] and not alone['value_symbols']`: the word has its concept and its symbol of order one, the value its concept and no symbol, and still no unit across modality. The docstring says so |

2. *`createMetaConcepts` true in the scene's configuration.* This is the
   owner's fallback: "If not, we can mint the meta concepts as before, and
   hope that the object-concept that we created for the word concept will
   begin to co-locate with its actual referent". The line is removed from
   `data/scene/BasicModel_scene.xml`, so the default (`data/model.xml`,
   true) holds. The file now differs from `data/BasicModel.xml` in
   `attentionPromotion`, `sceneShape` and `weightsPath`.
   `attentionPromotion` stays true.

   | Test | Old | New |
   |---|---|---|
   | `test_item16_scene.py::test_s1_the_key` | differs in `attentionPromotion`, `createMetaConcepts`, `sceneShape`, `weightsPath` | differs in `attentionPromotion`, `sceneShape`, `weightsPath`; docstring likewise |
   | `test_item16_naming.py::test_n1_the_key` | `assert scene_key() == 'false'` (the scene's configuration) | `assert KEY == 'false'` (the test's own model); the print names the test's model. The other assertions are unchanged |

   *Where the naming tests set the key:* `test_item16_naming.KEY = 'false'`,
   passed by `naming_model` as `architecture_overrides["createMetaConcepts"]`
   unless a caller gives `key`. Only N1's key-true half gives
   `key='true'`. `attentionPromotion` is still read from the scene's
   configuration (`scene_key('attentionPromotion')`).

**Run.** Both gates, `test_item16_naming.py`, `test_item16_scene.py`,
`test_item16_scene_field.py`, `test_item17_symbols.py`:
**`Result: test_failure, exit 1; 66/66 cases completed.`**
`output/tests/20261124-143606-579b3a/result.json`. 60 passed, 4 xfailed,
2 failed. **Both gates passed.** N1, N2 and N7 passed; N3 to N6 xfailed;
S1 passed. The two failures:

- `test_item17_symbols::test_54bd_…` failed in the sweep too.
- `test_item17_symbols::test_cert_c_…` is on the list of tests that pass or
  fail by chance. It passed in the final run of 39.8.

### 39.11 Alec's two changes during the step, and the review (Claude, 2026-11-24)

> The text modality promotes its zero-order concepts to first order
> symbols automatically. The reason for this is that in text, there is no
> need to generalize over correctly written text (other than typos). A
> concept (word) is represented in ASCII the same way, every time, so that
> the purpose of the PI in generalizing over that zeroth-order conceptual
> space is not needed.
>
> Presumably we should do this only for concepts whose members are
> exclusively from text modality.
>
> Does this make sense? it may get meta concepts to work. If not, we can
> mint the meta concepts as before, and hope that the object-concept that
> we created for the word concept will begin to co-locate with its actual
> referent, experienced as an object, and they will be treated as the same
> by higher-order reasoning (in virtue of the pi that collects them
> together within that space).

> OK. lets leave the testpoint for automatic metaconcept formation, and
> mark it as Xfail

**Accepted as a measurement.** Claude ran both stopping gates,
`test_item16_scene.py`, `test_item16_scene_field.py`,
`test_item16_values.py` and `test_item17_echo.py`: 42 of 42 passed. The
implementer's run of the naming test: N1 and N2 passed, N3 to N6 expected
to fail and failing, N7 reported.

- *The result.* MTM's ordinary promotion does not unite a word with its
  object. A candidate holding one of each is never met twice: MTM knows a
  candidate again by the proportions of its members' weights, and a
  word's and a value's differ in the fifth digit and move with learning.
  No concept of an intensity gets a symbol of order one. The lane against
  a candidate records that its focal concept was read against, which is
  not "the word present and its object not".
- *The fallback is taken:* the scene's configuration no longer sets
  `createMetaConcepts`, so meta concepts are made as before. The key
  stays, false only in the naming test's own model.
- *The expected failures* N3 to N6 are the one exception to H.5's rule
  against adding one, by his word.
- *Left for after this item:* how MTM knows a candidate again, what it
  writes against one, and what gives an intensity's concept its symbol
  (39's four proposals). They are MTM's own, in every modality.
- *Found:* `test_compiled_lesson_step::test_one_lesson_sentence_trains_with_the_compiled_output_walk`
  passes the runner's 8 GiB on this tree and on the tree before this
  step; the closing sweep finds which step it came with.


## 40. Step E: things wider than a place (Claude, 2026-11-24; the text for the implementer is 40.2)

*Alec, asked whether to stop at the sweep or build areas, intensities out
and relations of place: "Build the rest". Rests on 19.1 (the two towers; a
whole pervades its range), 29.1 and 29.5 (his example of [1,3,3,1]; parts
kept in order and placed by their overlap), 36 (the text algorithm,
unchanged).*

### 40.1 What it builds, and its certificates

As a word is a run of letters between spaces, a thing in an image is a run
of touching places that are not background. What a form is to its
characters, a thing's percept is to its pixels.

1. *A thing is a run.* In an image's bank the places that are not
   background and touch (by a side, on any axis of the box) are one
   thing. It is an occurrence beside its pixels' occurrences: its `.where`
   is the box of its places, its `.when` the moment's.
2. *Its percept, as a form's.* Its parts are its pixels' basic percepts in
   their order within its box (row by row), each at its place relative to
   the box's origin, so that the same thing at another place has the same
   parts. Its whole is the property that holds at every one of its places:
   the least upper cut all its pixels lie under. Its row is over those
   parts and that whole in the Pi of the value space, its code the
   greatest of its parts' codes below and the least of its wholes' above,
   as `subspace_bounds` gives a form's. It is nominated when the thing
   recurs, as a form is, and brings its concept then.
3. *Parts of a thing.* A run within a thing (his [1,2,1] within [1,3,3,1])
   is a part of it when it lies within it place by place by numerical
   parthood. Only what recurs is nominated; no powerset is laid out
   beforehand.
4. *Read as a form is,* for and against, by the value space's Pi; a thing's
   row reads at the thing's reading, and a pixel's row at the pixel's.
5. *The limit.* `intensityWhereWidth` is the greatest width of a thing on
   any axis: at one, no thing is wider than a place and nothing of this
   step is made, so XOR is as it is.
6. *The way back.* A thing's concept read for gives its parts back in
   their order at their places, each its centroid; where a thing's and a
   pixel's concepts both read, they agree or the difference is charged.

In `test/test_item16_things.py`, each printing what it measured:

| | Certificate | Passes when |
|---|---|---|
| E1 | A run is a thing | An image with two groups of touching pixels has two things, each with the box of its places; a pixel alone is a thing of one place and gains no second row |
| E2 | The same thing elsewhere | A thing met at one place and then at another is one row, with another `.where` |
| E3 | Parts and whole | A thing's row holds its pixels' percepts in order as parts and the least upper cut over them as its whole; its code is the greatest of its parts' codes and the least of its wholes' |
| E4 | For and against | The thing's row reads for at its own occurrence, and against at an occurrence that holds its parts and one pixel above its whole |
| E5 | Recurrence | A thing met once has no concept; met as often as nominates, it has one |
| E6 | The limit | With `intensityWhereWidth` one, no thing's row is made; the XOR gate, the XOR certificate and `test_concept_output.py` pass as they are |
| E7 | Put back | After a thing of several pixels has recurred, its image comes back at every place from the thing's concept, to its parts' centroids |
| E8 | The corpus (a measure) | Over the categories whose images hold things wider than a place: the number of things, and of rows nominated. Reported |

### 40.2 Text for the implementer

~~~
Item 1.6, step E: things wider than a place. Read §40.1 of
doc/plans/2026-11-18-item1-6-felt-math-world.md (what you build), then
§36 (the text algorithm applied to intensity; the owner's words there are
the authority), §37 (step T2 as built), §33 (step C), §29.1 and §29.5, and
H.5. Then read how a form is made of its characters and read, since you
are to apply it: doc/Mereology.md (the percept's code), doc/Spaces.md
("PartSpace", "Level Sets", "Modalities"), and in the source
BasicPercepts.reserve, member_keys, subspace_bounds, lookup, held_at and
read_bank, how a form's run and its bracket are found, and how MTM
nominates a form's row. Then bin/ImagePercepts.py, bin/NumericPercepts.py
and BasicPercepts.value_bank, read_values, value_pi.

Keep tmp/item1-6/things_progress.md current after each step.

1. Points 1 to 4 of §40.1, in the image's bank, for the numeric stem and
   for a sentence's image alike (one reader). Say what a thing's row's key
   is and what its entries in value_pi are.
2. Point 5. The key exists (intensityWhereWidth); 0 sets no limit.
3. Point 6, in NumericPercepts.reverse and ImagePercepts.put_back.
4. test/test_item16_things.py: E1 to E8. E8 uses felt_world_corpus; say
   which categories hold things wider than a place.
5. Port a test only where a point of §40.1 makes its assertion false, and
   list each with old and new. No other assertion is weakened, skipped or
   marked; no seed is chosen. Neither stopping gate, nor anything a gate
   uses, is touched. The text modality's reading is not to change.
6. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_concept_output.py,
   test/test_item16_values.py, test/test_item16_scene.py,
   test/test_item16_scene_field.py, test/test_item16_things.py,
   test/test_item16_naming.py, test/test_item17_echo.py,
   test/test_item18_places.py, test/test_item18_scalar.py, and the files
   of tmp/item1-6/values_run4_files.txt. No test that passes in the sweep
   may fail, apart from those known to fail by chance. Report the
   greatest memory of a worker and the time of test_item16_scene_field.py
   before and after.
7. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
8. Return as tmp/item1-6/section41.md, headed "## 41. Step E: the
   return": files changed and why; the thing's row, key and entries; each
   certificate with what it printed; every test ported; what in §40.1
   could not be built as written and what you built. You do not stop on a
   question of design: propose, build, mark it. Stop only if a gate, the
   XOR certificate or "without the intersection" cannot pass with its
   assertion as it is.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 41. Step E: the return (Claude on Opus, 2026-11-24) (Claude on Opus, 2026-11-24)

The work of §40.2, following §40.1. No git was run, and nothing is committed.
Progress was kept in `tmp/item1-6/things_progress.md`. Every file changed
was first copied to `/tmp/item1-6-2026-11-24/things_before/bin/`
(`BasicPercepts.py`, `NumericPercepts.py`, `ImagePercepts.py`, `Places.py`,
`FamiliarTier.py`). Nothing was removed: the loops of nomination in
`NumericPercepts.read` and `ImagePercepts.perceive` were replaced in place
by one function they now share, and the old text is in those copies.

### 41.1 Files changed, and why

| File | Why |
|---|---|
| `bin/BasicPercepts.py` | **Points 1 and 2.** `PerceptBank.things` (new field, default `None`): for each stream, each thing's occurrence and its pixels' occurrences at their places. `BasicPercepts.things`: each thing's row and its arrangement. `perceive_thing(arrangement)` makes the row; `reserve(..., thing=)` keys it; `is_value` is false for it; `value_keys` gives its entries in `value_pi`. `with_things(bank, percepts, *, where_width)` finds the runs of touching places, perceives each thing, and adds its occurrence and its reading to the bank. **Point 3 and 4.** `thing_events` (called by `value_events`) reads a thing's row at a thing's occurrence, place by place, for and against, at every shift that lays it within the occurrence. In `read_values`, a thing's reading holds its own occurrence alone. **Point 6.** `thing_at` and `thing_back` give a thing's parts back at their places. The extra state saves and restores `things`. |
| `bin/NumericPercepts.py` | `read` calls `with_things` after `Places.meet` and nominates things as values are nominated. `reverse` gives back things' parts (point 6), and where a thing and a pixel both give a place back, the lesser. |
| `bin/ImagePercepts.py` | `perceive` does the same as `read` and returns the bank with its things; `read` uses that bank. `image_rows` includes things' rows. `_given_back` (new) is the old `put_back`'s body plus the things'; `put_back` takes the lesser where both give back; `charge` adds `reconstruction_order_zero.things` where both give back. Module docstring: the thing's reading. |
| `bin/Places.py` | `nominate(owner, percepts, kind)`: the loop of nomination the two readers had, written once, used for `'value'` and `'thing'`. |
| `bin/FamiliarTier.py` | `percept_range`: a thing's row lies in `scene`, as a value's does (otherwise it would count as text, and `promoted_alone` would give it a symbol). |
| `test/test_item16_things.py` (new) | E1 to E8. |

`data/` and `doc/` are untouched, and no key was added: point 5 uses
`architecture.intensityWhereWidth`. Neither gate's file, nor anything a gate
uses, was touched. No existing test was changed (41.5).

### 41.2 The thing's row, its key and its entries (step 1)

- **The run (point 1).** In `with_things`, each run of pixel occurrences
  whose places touch by a side on any axis of the box is one run. A run of
  one place is no thing and has no second row (E1). A run wider than
  `intensityWhereWidth` on any axis is no thing; 0 sets no limit (point 5).
- **Its occurrence** comes after the pixels':
  - its span and its reading are the box of its places, from the address
    of the least corner to one past that of the greatest;
  - its `.where` wires are the box origin's;
  - its `.what` is the greatest of its pixels' codes, on two lanes;
  - its `.when` is the moment's.

  Its reading comes after the bracket, so the places and the bracket keep
  their indices (`A` places, the bracket at `A`). A thing's reading holds its
  own occurrence alone. The bracket holds pixels and things.
- **The row (point 2).** Its parts are its pixels' basic percepts in address
  order within the box (row by row), each at its offset from the box's
  origin: `arrangement = ((offset, pixel's row), ...)`.
  - **Its key** in `index` is `('thing', arrangement)`. The same thing at
    another place is the same row (E2).
  - `occurrences[row]` is the pixels' parts, their lower cuts in that order,
    and one whole: the least upper cut that all its pixels lie under (the
    greatest of the pixels' upper cuts).
  - **Its code** is `subspace_bounds`, as for a form: the greatest of its
    parts' codes and the least of its wholes'. That is the code of its
    greatest pixel's region (E3).
- **Its entries in `value_pi`** are `('ps', arrangement)` and `('ws', whole)`:
  the run of parts at their places as one key, as a form's run is, and the
  whole. It has none in the first Pi (`member_keys` and `against_keys` give
  `[]`).
- **Nomination.** The row is reserved at first sight. MTM nominates it under
  `('thing', row)` on `conceptMintCount` readings (`Places.nominate`), as
  values and forms are, and it brings its concept then with the symbol type
  `'thing'` (E5).
- **The reading (points 3 and 4)** is `thing_events`, inside `value_events`.
  - For each shift `s` that lays every one of the row's places on a place
    of the thing's occurrence: `part` is the least, over the row's places,
    of `held_at`'s part test, the pixel's code against the part's lower cut
    (numerical parthood, element by element); `under` is the least of
    `held_at`'s whole test against the row's whole.
  - For is `part * under`, and against is `part * (1 - under)`. Each lane is
    the greatest over the shifts, times activity and `parthood`.
  - So a run within a thing reads at it where it lies within it place by
    place (point 3; his `[1,2,1]` within `[1,3,3,1]` lies within it at two
    shifts).
  - A thing's row reads nothing at a pixel's occurrence, and no other row
    reads at a thing's occurrence.
  - The text modality's reading is not touched: `read_bank`, `held_at` and
    the first Pi are unchanged.

### 41.3 The certificates, as printed

From the development runs (`pytest -s`), on the code as it stands; the
bounded run's outcomes are in 41.4. The MM_image model has widths 3 and 3,
so `.2`, `.5` and `.9` fall in bins 0, 1 and 2. The scene model has widths
64 and 64.

- **E1, a run is a thing** (MM_image). The image holds a run `(3..5, 2)`
  of `.2 .5 .2`, a run `(10, 10..11)` of `.9 .9`, and a lone `.5` at
  `(20, 20)`.
  - `things [[3, 4]]`.
  - The boxes are `[(843, 846), (1074, 1103)]`, as expected.
  - The arrangements are `3: (((0,0,0),0), ((1,0,0),1), ((2,0,0),0))` and
    `4: (((0,0,0),2), ((0,1,0),2))`.
  - `rows 5`: three pixels' rows and two things'. The lone place's row is
    `[1]`, the row the first thing's middle made; it gains no second row.
- **E2, the same thing elsewhere.** `rows ([[2]], [[2]])` and
  `where ([843, 846], [1359, 1362])`; `things 1`.
- **E3, parts and whole.** The image is an L: `(3,2) .2`, `(4,2) .5`,
  `(4,3) .2`.
  - `arrangement (((0,0,0),0), ((1,0,0),1), ((1,1,0),0))`.
  - `parts (-4, -5, -4)` and `wholes (-5,)`.
  - `entries_first_pi []`, and
    `entries_value_pi [('ps', <the arrangement>), ('ws', -5)]`.
  - `code [1,1,0, 1,1,0]`, equal to the greatest part's code, which is the
    `.5` pixel's. `key ('thing', <the arrangement>)`.
- **E4, for and against.** The row of `[.2,.5,.2]`, read at the reading of
  the thing in each image:

  | Case | Read |
  |---|---|
  | own | `(1, 0)` |
  | one above, `[.2,.9,.2]` | `(0, 1)` |
  | his `[1,3,3,1]`, `[.2,.9,.9,.2]` (it lies within at two shifts) | `(0, 1)` |
  | a part not held, `[.2,.2,.2]` | `(0, 0)` |
  | the run `[.2,.5]` within the thing (point 3) | `(1, 0)` |
  | the pixel `.5` at the thing's reading | `(0, 0)` |
  | the pixel `.5` at its own place | `(1, 0)` |
  | the run `[.2,.5]` at the pixel's reading | `(0, 0)` |
- **E5, recurrence.** `mint_count 3`. The thing seen at each presentation:
  `{2: (False, None)}`, then `{2: (False, None)}`, then `{2: (True, 5)}`.
- **E6, the limit** (MM_xor, after the XOR inputs have recurred).
  `where_width 1`, `things {}`, `readings 3` with `places 2` (the places and
  the bracket, nothing more). `with_things` on two adjacent `1.`s at width 1
  gives `[[]]`. The XOR gate, the XOR certificate (V5) and
  `test_concept_output.py` whole pass as they are (41.4).
- **E7, put back.**
  - **On the numeric stem** (a thing `[.2,.5,.2]` and a lone `.9`, after
    recurrence). From the thing's concept alone: `[0.2, 0.5, 0.2, 0.0]` at
    the four lit places, the centroids `{0.2: 0.2, 0.5: 0.5}`. The lone
    pixel is not the thing's, so it gets nothing. With the pixels' concepts
    too: `[0.2, 0.5, 0.2, 0.9]`, equal to the image (`torch.equal`).
  - **With a sentence** (the scene model, `text ''`, pixels `(1..3, 1, 0)`
    at steps `51 128 51`). `ImagePercepts.put_back` from the thing's concept
    alone gives `[0.2, 0.502, 0.2]`, equal to the image felt
    (`torch.equal` over the 512 places). `reconstruction_order_zero.things`
    is `[0.]` when they agree.
  - **The charge where they differ.** The same three places at step `128`
    each, met once, so this thing has no concept yet. The pixels give back
    `.502` at each place, and the known thing `[51,128,51]` reads for there
    by numerical parthood. The lesser is put back, `[0.2, 0.502, 0.2]`, and
    `reconstruction_order_zero.things` is `0.2413`.
- **E8, the corpus (a measure).** This is the train split of each category,
  every image's things found as the reader finds them (`with_things`, at the
  scene model's `intensityWhereWidth`, 0), in a table of basic percepts of
  its own with the scene model's widths and box. "Nominated" means a row met
  in at least `conceptMintCount` (3) images.
  - **Categories that hold things wider than a place:** 3 to 25 and 27.
    Categories 1 and 2 have only single pixels, and 26 has no images.
  - **In all:** 4,975 things, 2,047 distinct rows, 544 of them nominated.

  | Category | Images | Things | Rows | Nominated |
  |---|---|---|---|---|
  | 3 | 192 | 18 | 14 | 0 |
  | 4 | 200 | 28 | 24 | 0 |
  | 5 | 200 | 30 | 29 | 0 |
  | 6 | 200 | 80 | 70 | 0 |
  | 7 | 776 | 74 | 52 | 4 |
  | 8 | 728 | 472 | 297 | 42 |
  | 9 | 613 | 399 | 264 | 29 |
  | 10 | 352 | 479 | 445 | 7 |
  | 11 | 200 | 324 | 313 | 1 |
  | 12 | 1131 | 342 | 226 | 27 |
  | 13 | 975 | 120 | 78 | 9 |
  | 14 | 200 | 154 | 131 | 2 |
  | 15 | 200 | 172 | 136 | 4 |
  | 16 | 857 | 439 | 265 | 37 |
  | 17 | 673 | 197 | 144 | 10 |
  | 18 | 706 | 57 | 34 | 8 |
  | 19 | 1180 | 157 | 93 | 19 |
  | 20 | 1172 | 494 | 318 | 38 |
  | 21 | 836 | 84 | 49 | 7 |
  | 22 | 629 | 368 | 252 | 26 |
  | 23 | 988 | 130 | 92 | 9 |
  | 24 | 533 | 67 | 52 | 3 |
  | 25 | 200 | 158 | 128 | 5 |
  | 27 | 200 | 132 | 126 | 0 |

  The per-category counts are within each category. A thing's row is its
  exact arrangement of pixels' rows, and at 64 bins a value's steps fall in
  two or three rows (V12). So one shape at one value is several rows, and
  most rows of things are not met three times.


### 41.4 The runs

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py`
with the 64 selectors of `tmp/item1-6/things_run_files.txt`, logged in
`tmp/item1-6/things_run1.log`. The selectors are:

- §40.2 step 6's list;
- `test_item16_grid.py`, `test_item17_one_reconstruction.py`,
  `test_item20_round2_inverse.py` and `test_item20_round6_reverse.py`, the
  other test files that build MM_image or read the numeric stem's reverse;
- the files of `values_run4_files.txt`.

**`Result: test_failure, exit 1; 558/558 cases completed.`**
`output/tests/20261124-151411-96c9f4/result.json`: 506 passed, 44 skipped,
4 xfailed, 4 failed. It was compared test by test with the sweep
(`tmp/item1-6/promotion_compare.py`).

- **Both stopping gates passed**, and `test_mm_xor.py` whole, V5 (the XOR
  certificate) and `test_concept_output.py` whole. The three cases of
  `test_native_cli_curriculum_learns_output_and_keeps_located_inverse`
  ("without the intersection") passed.
- E1 to E8, V1 to V12, C1 to C5, S, N1, N2 and N7 passed. N3 to N6 xfailed,
  as marked in step D3.
- **The four failures:**
  - `test_item17_symbols::test_54bd_…` is `failed` in the sweep.
  - `test_item17_symbols::test_cert_c_…` is on the list of tests that pass
    or fail by chance.
  - `test_item20_round6_loop::test_parallel_input_without_composed_sentences_books_only_field_reconstruction`
    is on the list of chance.
  - `test_item16_grid::test_g1_built_widths_are_as_before[MM_xor.xml]` is on
    the list (test order). Its MM_image case passed.
- Every other test has its sweep outcome. The 44 skipped are skipped in the
  sweep or are trained tests.

**Memory.** The greatest of a worker was 5.09 GiB (worker-002). The others
were 2.16 to 4.66 GiB, and none came near the 8 GiB cap.

**`test_item16_scene_field.py`**, alone in the bounded runner:

| | Worker time | Peak | Run |
|---|---|---|---|
| Before (unchanged tree) | 35.6 s | 1.35 GiB | `output/tests/20261124-150243-b64c11` |
| After | 34.8 s | 1.35 GiB | `output/tests/20261124-153453-335ad0` |

A development run of `test_item16_values.py`, `test_item16_scene_field.py`,
`test_item18_places.py` and `test_item16_scene.py` gave 37 of 37 passed.
On MM_image, V2 and V11 now make a thing of hundreds of places. Their
assertions hold, because the thing gives back the same centroids as its
pixels.


### 41.5 Tests ported

None. No assertion of an existing test was made false by a point of §40.1
on the runs (41.4). No assertion was weakened, skipped or marked, and no seed
was chosen.

### 41.6 What in §40.1 could not be built as written, and what was built

1. *A thing's `.where`.* A reading is an interval of addresses, and the
   wires encode one place. So a thing's reading runs from the address of
   its box's least corner to one past that of its greatest, and its wires
   are those of the box's origin. In a box laid out row by row, that
   interval holds the box and other places between its rows.
   - The reading holds its own occurrence alone, set in `read_values`, so
     no pixel of another run inside the interval is read there.
   - The box itself is recovered from the interval's two ends
     (`thing_at`).
   - *Marked.*
2. *A thing's row is read at the bracket too.* The bracket holds every
   occurrence. Under `intensityWhereWidth` a value's row is kept from
   readings wider than the limit, but a thing's row is not, since every
   reading of a thing is wider than a place in addresses. A thing wider
   than the limit is simply never made (point 5). *Marked.*
3. *Parts of a thing (point 3).* The relation is built in the reading.
   - A thing's row is read at a thing's occurrence at every shift that lays
     its places within it, place by place. So a smaller thing that recurs
     reads where it lies within a larger one (E4).
   - No sub-run is laid out or reserved. A run becomes a row only when it
     occurs as a thing itself.
   - *Built:* no stored list of parts per thing; parthood is read when
     needed. *Marked.*
4. *Whole over which places.* "For" asks that the row's places, where they
   are laid on the occurrence, lie under its whole. An occurrence with
   places beyond the row's (the row a part of it) is not told against by
   them, since the whole holds of the row's places. *Marked.*
5. *The way back from a part found at a shift.*
   - `thing_at` gives a thing's parts back only where the reading's box is
     the row's own box, which fixes the shift at zero.
   - A smaller thing's row that reads for within a larger thing gives
     nothing back, because the field keeps no shift. The places are given
     back by the larger thing's row and by the pixels.
   - *Marked.*
6. *Where a thing's and a pixel's concepts disagree,* the lesser is put
   back (§29.3, point 2).
   - On the image with a sentence the difference is charged as
     `reconstruction_order_zero.things`: the squared difference at the
     places where both give back, against the pixels' square, with the
     image's weight and category. `SentenceField` sums every
     `reconstruction_order_zero.*` term.
   - The numeric stem has no charge path, so `NumericPercepts.reverse`
     takes the lesser only.
   - Where several things give one place back, the greatest is taken, as
     among values.
   - *Marked.*
7. *E8's "rows nominated"* counts rows met in `conceptMintCount` or more
   images, each image counting once as MTM counts a reading. The scene
   model was not run over 19,000 images, and its table has no room for
   their rows. The measure uses a table of basic percepts of its own, with
   the model's widths, box and `intensityWhereWidth`. *Marked.*
8. *A new field of `PerceptBank`, `things`.* `thing_events` needs, for each
   thing's occurrence, its pixels' occurrences at their places. The field
   defaults to `None` and is set only by `with_things`, so a sentence's
   bank is as it was.
9. *Saved state.* `things` is saved in the extra state and read back with
   `state.get('things', ())`, as `conjunctions`, `forms` and `place_rows`
   are read.
10. *`data/model.xml`'s comment on `intensityWhereWidth`* still describes it
    as the limit of a value's row alone. It was left untouched because every
    gate reads that file. Proposed: add "and the greatest width, on any
    axis, of a thing".
11. *The nomination loops.* The loops in `NumericPercepts.read` and
    `ImagePercepts.perceive` were replaced by `Places.nominate`, one
    function with the same behaviour, now used for things as well.

### 41.9 Review of step E (Claude, 2026-11-24)

**Accepted.** Claude ran `test_mm_xor.py` whole, the other stopping gate,
`test_item16_things.py`, `test_item16_values.py`,
`test_item16_scene_field.py` and `test_concept_output.py`: 42 of 42
passed.

- *A thing's `.where` is an interval of addresses* from its box's least
  corner to one past its greatest, which in a box laid out row by row
  also spans places between its rows. Nothing else is read there, so it
  is harmless now; a look that takes a felt thing will need the box
  itself.
- *A smaller thing within a larger gives nothing back,* since the field
  keeps no shift. His "placing the parts according to their overlap"
  (29.5) is so far built for a thing in its own box only.
- *The corpus.* 4,975 things in 2,047 distinct rows, 544 of them met in
  three images or more: at 64 bins one shape at one value falls in
  several rows, as one value's steps do (37.9). What gathers them is the
  same open point as naming's (39.11).
- *The comment on `intensityWhereWidth`* in `data/model.xml` speaks of a
  value's row only; it is to say "no thing is wider than this", and is
  changed at the close with the gates run after it.


## 42. Steps F and G: intensities out; relations of place (Claude, 2026-11-24; the text for the implementer is 42.3)

### 42.1 F: intensities out (§24, point 1)

The model gives an image out as it gives words out. The way back that
puts an image back at the opening (step C, step E) is asked for at the
output, from the field the understanding ends with.

1. `reconstructModalities` gives, beside the sentence and the value beside
   the words, the image: for each row the intensities over the scene's
   box, or nothing where the row had no image.
2. It is read from the understanding's field by the one way back
   (`ImagePercepts.put_back`), not from anything kept from the input.
3. *Not built:* an image given in answer to words ("draw three"). That
   needs the name, which is not made automatically (39.11).

| | Certificate | Passes when |
|---|---|---|
| F1 | An image out | An image presented until it is known comes out at the output place by place, to its percepts' centroids, pixels and things alike |
| F2 | Both out | A sentence with its image gives its words out as it does without the image, and the image beside them |
| F3 | None | A row with no image gives no image out; an empty image gives one with nothing in it |
| F4 | From the field | With the field's evidence at nothing, nothing comes out |

### 42.2 G: relations of place (28.6; §23)

An axis of place is a sorted line. The relation of two things on an axis
is less-than between their places there, numerical parthood, and an
arrangement is read by such relations and is no percept of its own.

1. For the things of an image's bank (step E; a pixel alone is a thing of
   one place), a reading of every ordered pair: on each axis, whether the
   first lies wholly before the second (its greatest coordinate less than
   the other's least); and whether the two touch by a side. Each on two
   lanes, for and against, so that it can stand in a field.
2. Which end of the y axis is "above" is the corpus's, read from how
   `bin/felt_world_corpus.py` defines it, and said in the return.
3. *Not built:* the words for relations, a look that takes two things,
   and the certificate on category 14's questions, which waits for item
   1.4 (22.3, question 9). Nothing reads the relations yet.

| | Certificate | Passes when |
|---|---|---|
| G1 | Order on an axis | For two things, before on x, on y and on z each agree with their places, both ways round, and neither is before the other where their extents overlap on that axis |
| G2 | Touching | Two things that share a side touch; two a place apart do not |
| G3 | The corpus | For every image of the frozen corpus whose row records a meaning of above, below, left of, right of, on top of, under or next to between two things, the relation read from the image's bank agrees with the recorded truth. The count of images and of relations checked is printed |

### 42.3 Text for the implementer

~~~
Item 1.6, steps F and G: intensities out; relations of place. Read §42.1
and §42.2 of doc/plans/2026-11-18-item1-6-felt-math-world.md (what you
build), then §33 (step C), §37 (step T2), §41 (step E), §23, §28.6 and
H.5. Then the source: bin/ImagePercepts.py, bin/NumericPercepts.py,
BasicPercepts.value_bank and the things of step E, reconstructModalities
and its carriers in bin/Models.py, bin/ScalarPercepts.decode as the
example of a second modality at the output, bin/WhereRegistry.py (place,
address), and in bin/felt_world_corpus.py the predicates of place and how
a row records its meaning.

Keep tmp/item1-6/out_progress.md current after each step.

1. F, points 1 and 2 of §42.1. Say where the image is read from at the
   output.
2. G, points 1 and 2 of §42.2, as a function beside the image's reader.
   Say what it returns and its shape.
3. test/test_item16_out.py: F1 to F4. test/test_item16_relations.py: G1
   to G3. G3 maps a meaning's two things to the things of the image by
   their values; say how.
4. Port a test only where a point of §42 makes its assertion false, and
   list each with old and new. No other assertion is weakened, skipped or
   marked; no seed is chosen. Neither stopping gate, nor anything a gate
   uses, is touched. The text modality's output is not to change.
5. Run with the bounded runner, RUN_TRAINED and RUN_SLOW unset: both
   stopping gates, test/test_mm_xor.py, test/test_item17_echo.py,
   test/test_item16_things.py, test/test_item16_values.py,
   test/test_item16_scene.py, test/test_item16_scene_field.py,
   test/test_item16_out.py, test/test_item16_relations.py,
   test/test_item18_scalar.py, test/test_item18_recognition.py, and every
   test file that names reconstructModalities. No test that passes in the
   sweep may fail, apart from those known to fail by chance.
6. No standing doc and nothing under doc/. Nothing removed is deleted but
   moved to /tmp/item1-6-2026-11-24/.
7. Return as tmp/item1-6/section43.md, headed "## 43. Steps F and G: the
   return": files changed and why; each certificate with what it printed,
   G3's counts in full; every test ported; what in §42 could not be built
   as written and what you built. You do not stop on a question of
   design: propose, build, mark it. Stop only if a gate cannot pass with
   its assertion as it is.

Do not run git and do not commit: Claude commits what it has reviewed.
Nothing is pushed.
~~~

## 43. Steps F and G: the return (Claude on Opus, 2026-11-24) (Claude on Opus, 2026-11-24)

The work of §42.3, following §42.1 and §42.2. No git was run, and nothing is
committed. Progress was kept in `tmp/item1-6/out_progress.md`. Every file
changed was first copied to `/tmp/item1-6-2026-11-24/out_before/bin/`
(`ImagePercepts.py`, `Models.py`). Nothing was removed.

### 43.1 Files changed, and why

| File | Why |
|---|---|
| `bin/ImagePercepts.py` | **F.** `given(model, field)`: the image given out, by row, put back from the field alone by `put_back`, or `None` for a row with no reading in the box. **G.** `relations(bank)`: the relations of place among the things of an image's bank (43.3). |
| `bin/Models.py` | **F.** `_capture_understanding` carries `image_field`, the first sentence's opened field with its evidence cloned, where the batch reads an image (`ImagePercepts.imaged(self, 0)`, the test `charge` makes). `reconstructModalities` returns `image=` beside `input=` and `scalar=`: `()` where the understanding carries no image's field, else `given(self, image_field)`. |
| `test/test_item16_out.py` (new) | F1 to F4. |
| `test/test_item16_relations.py` (new) | G1 to G3. |

`data/` and `doc/` are untouched, and no key was added. Neither gate's file,
nor anything a gate uses, was touched. No existing test was changed (43.5).
The text modality's output is unchanged: `input=` is computed as before, and
F2 prints it with the image and without.

### 43.2 Where the image is read from at the output (step 1)

- **The field.** The image is read only at the first sentence's opening
  (`ImagePercepts.imaged`: `sid == 0`), into the field that opening makes
  (`SentenceField.open_field`, `_sentence_open_fields[0]`, after the value
  beside the words and the intuition are joined). That is the field the
  opening's charge puts the image back from. The understanding carries it as
  `reconstruction_carriers['image_field']`, its evidence cloned, as the scalar's
  field is carried.
- **The way back.** `reconstructModalities` calls `ImagePercepts.given`, which
  calls `put_back`: the one way back of step C and step E, the pixels' rows
  and the things' rows, the lesser where both give back. Nothing kept from the
  input is read: `put_back` reads the field's evidence, its concept ids and its
  `.where`, and the table's centroids and things.
- **A row with no image** is told from the field: it has no reading in the box
  (a `null` row's columns are empty brackets, 33.3), and gives `None`. An empty
  image has the bracket over the box, so it gives an image with nothing in it.
  A batch in which no row has an image carries no image's field, and gives
  `image=()`, as `scalar=()`.
- **Its shape.** A tuple by row; each a tensor `[places of the box]`
  (512 on the test model), detached, in address order, or `None`.

### 43.3 The relations (step 2)

`ImagePercepts.relations(bank)`, beside `put_back`, on a bank made by
`image_bank` and `with_things` (or by the numeric stem's `read`).

- **The things.** Each thing of several places (`bank.things`, step E) and
  each pixel, a thing of one place: every pixel's occurrence, including the
  pixels of a thing. In each stream the pixels come first in occurrence order,
  then the things.
- **Before on an axis.** For the ordered pair (first, second) and an axis of
  the box: the first's greatest coordinate there less than the second's least.
  This is less-than between places, numerical parthood on the line of the axis.
- **Touching across an axis.** A place of the first and a place of the second
  one apart on that axis and at one on the others, and no place in common.
- **Returns** `(occurrence, lanes)`: `occurrence` `[B, T]`, each thing's
  occurrence in the bank (`-1` where none); `lanes` `[B, T, T, 2, axes, 2]`:
  first thing, second thing, the relation (0 before, 1 touching), the axis in
  the box's order (x, y, z), and for and against. A pair where either is none,
  or a thing with itself, is nothing on both lanes. The lanes are 0 or 1: the
  places are whole numbers.
- **Which end of y is above (point 2).** The lesser. `felt_world_corpus.p_row_before(a, b)`,
  the meaning of "{a} is above {b}" (recipe 14, s01) and of "{b} is below {a}"
  (s02), is the greatest y of a less than the least y of b, with a column in
  common. Likewise the lesser x is to the left (`p_column_before`, s03 and s04).
  In the image's terms, "a above b" is a before b on y and neither before the
  other on x; "a left of b" is a before b on x and neither before the other on
  y. On top of (`p_on_top`, s05, and "under", s06) is the first with any touching;
  next to (`p_next_to`, s07) is touching across x.
- Nothing reads the relations in the model (point 3).

### 43.4 The certificates, as printed

From the development runs (`pytest -s`) on the code as it stands. The bounded
run's outcomes are in 43.6. The model is the scene model of step C (the echo
test's text model with `sceneShape` `8 8 8`, widths 64 and 64), eager
reading; an image is presented until it is known with
`test_item16_scene_field.known`. The output is read by `model.understand` and
`reconstructModalities`, in eval mode, with no gradient.

- **F1, an image out.** A thing of three places and a lone pixel:
  `F1 [[1, 1, 0, 51], [2, 1, 0, 128], [3, 1, 0, 51], [5, 5, 0, 200]]: out [(9, 0.2), (10, 0.502), (11, 0.2), (45, 0.7843)]; centroids [(9, 0.2), (10, 0.502), (11, 0.2), (45, 0.7843)]; felt [(9, 0.2), (10, 0.502), (11, 0.2), (45, 0.7843)]; thing's concepts [100], from them alone [(9, 0.2), (10, 0.502), (11, 0.2)]`.
  Each pair is (place of the box, intensity). The output equals the centroids
  at all 512 places (`torch.equal`). From the thing's concept alone (the
  carrier's evidence kept for it only) the thing's three places come out at
  their centroids and the lone pixel does not.
- **F2, both out.** Category 2's first row:
  `F2 {'text': 'this is eight.', 'scene': [[3, 2, 0, 200]]}: words out 'this is eight.', without the image 'this is eight.'; image out [(19, 0.7843)], felt [(19, 0.7843)]; without the image, image out ()`.
- **F3, none.** `F3 a row with scene null: image out (); an empty image: (512,) places, 0 lit; a batch of [[5, 5, 0, 200]] and a row with no image: [[(45, 0.7843)], None]`.
  The pair is the image's field given to two rows, the second's readings in
  the box made the empty brackets a `null` row has (43.7, 3).
- **F4, from the field.** `F4 [[1, 1, 0, 51], [2, 1, 0, 128], [3, 1, 0, 51], [5, 5, 0, 200]]: out [(9, 0.2), (10, 0.502), (11, 0.2), (45, 0.7843)]; with the field's evidence at nothing 0 of 512 places lit`.
  The understanding is the same, its `image_field` with its evidence zeroed.

G1 to G3 build each image's bank as the image's reader does (`value_bank` as
`image_bank` makes it, `with_things` at `architecture.intensityWhereWidth`),
on the scene model's box, in a table of basic percepts of its own as E8 does.
Relations are printed as (for, against) on x, y, z.

- **G1, order on an axis.** `G1 [(1, 1, 0), (2, 1, 0)] and [(4, 3, 2)]: before on x, y, z (for, against) [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], the other way round [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]; [(1, 1, 0), (2, 1, 0), (3, 1, 0)] and [(2, 4, 0)], overlapping on x: [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], the other way round [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]`.
  Neither is before the other on x where their extents overlap, nor on z
  where they share it.
- **G2, touching.** Image `[[1, 1, 0, 51], [2, 1, 0, 128], [2, 2, 0, 51], [2, 2, 1, 51], [4, 1, 0, 200], [5, 2, 0, 200]]`
  (one thing of four places and two lone pixels), touching across x, y, z,
  each way round alike:

  | Pair | Touching (x, y, z) |
  |---|---|
  | side on x, `(1,1,0)` and `(2,1,0)` | `[1,0] [0,1] [0,1]` |
  | side on y, `(2,1,0)` and `(2,2,0)` | `[0,1] [1,0] [0,1]` |
  | side on z, `(2,2,0)` and `(2,2,1)` | `[0,1] [0,1] [1,0]` |
  | a place apart, `(2,1,0)` and `(4,1,0)` | `[0,1] [0,1] [0,1]` |
  | at a corner, `(4,1,0)` and `(5,2,0)` | `[0,1] [0,1] [0,1]` |
  | the thing and its pixel `(1,1,0)` | `[0,1] [0,1] [0,1]` |

  Places that touch by a side are one thing of the bank, so the things that
  share a side are its pixels, things of one place (43.7, 5).
- **G3, the corpus.** `G3 the frozen corpus: 909 images, 909 relations checked (by predicate, true and false: {'on_top': [80, 10], 'row_before': [395, 292], 'column_before': [73, 13], 'next_to': [41, 5]}); 406 of them with the two in one thing of the bank, read by its pixels; disagreeing 0 []`.
  - **Which rows.** Every line of every `data/felt/category-NN-*.jsonl`, every
    split; every meaning whose predicate is `row_before` (above, below),
    `column_before` (to the left of, to the right of), `on_top` (on top of,
    under) or `next_to`, whose entry has an image. They are in categories 14,
    17 and 25; the inference split's rows have no image. 909 meanings, each
    on an image of its own (909 distinct row and entry): 589 recorded true,
    320 false.
  - **The two things** are the meaning's slots `a` and `b` (every frame of
    these predicates has the arguments `a, b`; a wh question's slot holds its
    answer, so the recorded truth is of the slots).
  - **How a meaning's thing is mapped to the image's things, by their
    values.** A thing of the corpus is the places of one value
    (`frame_of`, `value_of(steps)`). Each thing of the bank (each pixel, and
    each thing of several places) is given the set of values of its pixels,
    `value_of(round(intensity × 255))`; a meaning's thing `v` is every thing
    of the bank whose set is `{v}`. A thing of the bank that holds two values
    (two things of the corpus touching, which the reader joins into one run)
    is mapped to neither; its pixels are.
  - **The relation read**, from the lanes over every pair of a's things and
    b's things: before on an axis is before for every pair; sharing a line
    of the other axis is, for some pair, neither before the other on it;
    touching is touching for some pair. Above (`row_before`): before on y and
    sharing x. Left of (`column_before`): before on x and sharing y. On top
    of: above, and touching across any axis. Next to: touching across x.
  - All 909 agree with the recorded truth.

### 43.5 Tests ported

None. No assertion of an existing test was made false by a point of §42, on
the run of 43.6. No assertion was weakened, skipped or marked, and no seed was
chosen. The text modality's output is unchanged (F2; `test_item17_echo.py`
and `test_item18_scalar.py` pass).

### 43.6 The run

`env -u RUN_TRAINED -u RUN_SLOW DEVELOPER_DIR=/Library/Developer/CommandLineTools .venv/bin/python test/test_report.py`
with the 11 selectors of `tmp/item1-6/out_run_files.txt` (§42.3 step 5's
list; `test_item18_scalar.py` is the one existing test file that names
`reconstructModalities`), logged in `tmp/item1-6/out_run1.log`.

**`Result: passed, exit 0; 72/72 cases completed.`**
`output/tests/20261124-154642-148ace/result.json`, 443 s: 70 passed, 2
skipped.

- **Both stopping gates passed**, and `test_mm_xor.py` whole, the seven echo
  tests, E1 to E8, V1 to V7 and V9 to V12, S1 to S4, C1 to C5, F1 to F4,
  G1 to G3, and `test_item18_scalar.py` whole.
- The two skipped are `test_item18_recognition.py`'s two `test_90_*`, skipped
  in the sweep.
- Workers peaked at 4.01, 4.44 and 1.54 GiB, under the 8 GiB cap.

### 43.7 What in §42 could not be built as written, and what was built

1. *"The field the understanding ends with"* (§42.1). The image is read only
   at the first sentence's opening, into that opening's field, so the image is
   read from the first sentence's opened field (`_sentence_open_fields[0]`),
   carried as `image_field`. The scalar's carrier takes the last sentence's
   field; with one sentence the two are one. With several sentences the last
   field holds no reading of the image, and reading from it would give an
   empty image. *Marked.*
2. *Whether the understanding carries an image's field* is decided by the
   test the opening's charge makes (`ImagePercepts.imaged`: some row of the
   batch has a `scene`). This is no intensity from the input; the image itself
   comes from the field alone. *Marked.*
3. *A row with no image in a batch with an image* gives `None`, told from the
   field (no reading in the box). The test model reads one stream, so F3
   checks this on the image's field given to two rows, as C5 does (33.7, 4).
   A batch in which no row has an image gives `image=()`, as `scalar=()`.
   *Marked.*
4. *What comes out* is `put_back`, detached: it reads whether a concept reads
   for (33.7, 1), so an image comes out at its centroids exactly once its
   concepts read. F4's "nothing comes out" is an image of zeros: with the
   evidence at nothing no concept reads for.
5. *Touching by a side* (§42.2, point 1). Two places of the bank that touch
   by a side are one run, one thing (step E), so two things of several places
   never touch; only pixels, things of one place, touch, or a lone pixel and
   a thing where `intensityWhereWidth` keeps a run from being a thing.
   **Built:** the things are every pixel and every thing of several places,
   and touching is read across each axis (x, y, z) separately rather than as
   one relation; touching on any axis is their greatest. A pair with a place in
   common (a thing and its own pixel) does not touch. *Marked.*
6. *The lanes are crisp*, 0 or 1: places are whole numbers, read exactly. No
   reading by bounds through the wires is built; nothing reads the relations
   yet (point 3). *Marked.*
7. *Above, the corpus's.* The corpus's "above" is not before on y alone: it
   asks a column in common (`p_row_before`), and left of a row in common
   (`p_column_before`). These are composed in G3 from the relations (before on
   one axis, and neither before the other on the other axis), not built as
   relations of their own. On top of and next to likewise. *Marked.*
8. *G3 counts each meaning once*; each of the 909 meanings is on an image of
   its own.
9. *Nothing named in code by a word.* The relations are named by axis and
   order (before, touching, the axis index); the words for them are in the
   test's docstrings only, as the corpus's frames have them.

### 43.9 Review of steps F and G (Claude, 2026-11-24)

**Accepted.** Claude ran both stopping gates, `test_item16_out.py`,
`test_item16_relations.py`, `test_item16_things.py`,
`test_item16_scene_field.py` and `test_item18_scalar.py`: 28 of 28
passed.

- *The image comes out of the field of the sentence it was read with,*
  the first of a presentation. With several sentences it is not the last
  field, since nothing carries an image across sentences yet; that is the
  succession the corpus's later categories ask for and is not built.
- *The relations agree with the corpus in all 909 recorded meanings*
  (categories 14, 17 and 25). The corpus's "above" is before on y and in
  the same column, and "left of" before on x and in the same row; the
  model's relations are the two orders and touching, of which the
  corpus's words are made.
- *Two things that touch are one thing of the bank,* so touching is read
  between pixels. "Next to" of two named things is then within one thing;
  whether two things that touch stay two is a question for the round that
  names them.
- *Nothing reads the relations.* The words, a look that takes two things
  and category 14's questions wait for item 1.4.

