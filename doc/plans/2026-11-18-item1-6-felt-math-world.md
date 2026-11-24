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
| 4 | `{a} is {k}.` `{a} is not {k}.` `{a} is {aK}.` `the evens are two, four, six and eight.` `the odds are one, three, five, seven and nine.` | `is {a} {k} ?` yes, no. `what is {a} ?` {k} |
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
| 20 | `{a} moved toward {b} slowly.` then `it moved toward {c} quickly.`, and the same with the quick move first | `which move was faster, the first or the second ?` second, first. `which move was slower, the first or the second ?` first, second |
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
  first is drawn, with the question and the order of telling fixed, so
  that the answer follows from what was seen and not from the wording.
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
  not stated in.
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
  chain of two; 19, a chain through time; 24, the rule applied, eight
  that follow as yes and eight as no, and four with the premise left
  out, marked `open`, whose answer is the empty text; 25, through the
  kind; the others have none.
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
6. The held-out casts are in no train, validation, test or witness row
   of their category; every digit, every frame and every count is in
   train.
7. A scene is `null` in every entry of an `inference` row, in frame 1's
   word alone and in category 26, and nowhere else. An `inference`
   row's question has not the meaning of any of its statements. An
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
