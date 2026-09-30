# Item 5.5: where and when a sentence occurred, tense, aspect, the preposition and surface form

> **Status:** specification, 2026-09-30, written by Claude from Alec's
> decisions in conversation on 2026-09-30. Todo item **5.5**, after item 6
> and before item 5: "So 5.5?" Codex implements; Claude reviews. It closes
> the last group of the [operator catalogue](2026-09-29-operator-catalogue.md#9-the-order-of-the-pass)
> (the preposition, tense, morphology, aspect and `surface`) and changes what
> `.where` and `.when` mean. It is one check-in: "with 5.5, so that we can
> handle issues in one checkin." **Decided** is Alec's decision. *Claude's
> reading* is Claude's, to be confirmed. *Measured* means a probe run on a
> clean copy of HEAD `1678ee1f` on 2026-09-30; the probe scripts are not in
> the repository and nothing in it was edited for them.

## 1. The problem

### 1.1 The clock is inside every idea (measured)

The model clock `when_time` ticks once per batch. Every word read gets the
clock's value as its `.when` stamp, four channels holding two sine and
cosine pairs, and the stamp is muxed into every percept. ConceptualSpace
receives that muxed event, so the stamp is in every idea: in MM_grammar and
in the serial LTM fixture an idea is 14 channels, content 0–5, `.where`
6–9, `.when` 10–13, and the store keeps all 14. The stamp is the largest
part of an idea (norm about 3 to 4, against about 1.5 for the content; its
pairs are two to three times one stamp's length).

The same sentences, with the same weights, were read at nine clock values:

1. **The parse depends on when a sentence is read.** The chooser reads the
   whole idea, stamp included. Clocks 0, 200, 400, 1500 and 3000 gave a
   derivation of depth 25; clocks 25, 50, 100 and 777 gave depth 2. With
   the same derivation, the content was identical at every clock, so the
   operators do not mix time into the content; the choice does.
2. **What is stored depends on when it was read.** The same sentence was
   stored as `[0.888, 0.66, 0, 0, 0, 1.11]` at clock 0 and as
   `[0, 0.898, 0, 0, 0, 1.11]` at clock 777: one fact, two ideas.
3. **The answer depends on when it is asked.** MM_xor, trained until clock
   802, answered the same four questions with maximum error .061 at 802,
   .068 at 812, .174 at 902, .240 at 1802, .117 at 5802 and .027 at 20802.
   All four answers moved together. Repeating 802 between the others gave
   the same numbers each time, so it is the clock and not left-over state.
4. **Training chases a moving input.** The stamp changes every batch, and a
   test after training meets phases that training never saw.
5. **The stamp carries nothing comprehension needs.** Every word of a batch
   gets the same value, and what it says belongs to memory (section 3).

Every gate tests right after training, when the clock has barely moved,
which is why it was not seen.

### 1.2 What the tree holds for tense today (read)

`TenseLayer` and `AspectLayer` are identities on concepts; `_lift_when` and
`_lower_when` shift a stamp by one tick; `MorphologyLayer` is fed by
`surface_morphology.py` and `surface_tense.py`, a hand-written table of
English verb forms that nothing in the runtime reaches; `complete.grammar`
and `ladder.grammar` still declare `tense` and `morphology`. The `.when`
band lost its duration on 2026-07-04 ("exact extents belong to the record
store when aspect is built"). A row's `.when` does not exist at HEAD; item 7
adds one, stamped from the clock. Nothing reads a row's `.when` back, and
`MemoryIndex._scope_contains` reads a query's metadata, not the band.

## 2. The principle *(decided, Alec, 2026-09-30)*

"We do not reach into the conceptual space for the .when and then write
outside of the space. The .when and .where refer to the time of the speech
utterance, whereas the time conceived of within that expression is not
demuxed (although it is aligned with the temporal dimension in virtue of
being carried by the VP)."

So there are two kinds of time and they never meet in one place:

| | what it is | where it lives | written by |
|---|---|---|---|
| **occurrence** | where and when the utterance or percept occurred | a memory row's `.where` and `.when` | the reading, from outside the concept |
| **conceived** | what the sentence says about time: order ("before now"), and any stated amount ("yesterday", "for three days") | the concept, carried by the verb phrase | tense, aspect and the verb phrase's prepositions |

"Concept holds order: yes. Memory holds .when: yes." A date, when one is
needed, is worked out at recall from the row's occurrence and what the
concept says; it is never stored as a coordinate.

## 3. Where and when a sentence occurred

### 3.1 The address *(decided)*

"`.where` can be the document index, so `.when` indexes within that document
(and for each document starting from 1). That sounds like it will correct
the clock defect and allow the model to address its own LTM."

* A memory row's `.where` is its **document**; its `.when` is the
  sentence's **index within that document**, starting from 1.
* "Instead of batch indexing, use document indexing, since multiple
  documents within a batch don't have an order relation." Two documents
  have no order relation; sentences within one document do.
* Documents are in the codebook: "documents get added to the codebook, and
  revealed as codes like everything else" (3.4).
* The address is the model's handle on its own memory: sentence *k* of
  document *d*.

### 3.2 Percepts and concepts carry no `.when` *(decided)*

"Perhaps .when is a feature only of LTM or memory, not of percepts."
Perceptual spaces drop the `.when` band, so the conceptual event is its
content and `.where` (10 channels where it was 14). That removes 1.1 at its
source. A percept is always now; what it holds of past and future are
structures, not a band: STM and the recency buffer (retention), the
expectation (protention), and the change between them (item 6.5's verb
column).

*Claude's reading:* percepts keep their positional `.where` (a word's place
in the input), which the parse needs for word order; a row's `.where` is its
document. Whether a sentence's *absolute* byte offset steers the chooser as
the clock did is unmeasured and is measured here (test 2).

### 3.3 A row keeps only its start; a complement ends it *(decided)*

".where could use just a start position and a run length to determine the
end. That does not work for WholeSpace, since properties/cuts do not have a
finite size. So we can either introduce an end location, or a
complement-property that begins after the property ends. This can replace
both the refresh rule and the eternal flag." Of the complement: "That does
not give us a duration, but it may be better since we don't have to
back-patch that value. So: good. Especially given we can flip between
evidence for and evidence against." Duration is dropped: "Yes, drop
duration." None was ever added: HEAD rows have no `.when`, and item 7's
holds only a start.

* A state holds from its row's start until a later row of the same content
  with **evidence against** begins in the same document; with none, it
  holds on. Rows already carry the evidence pair `(c⁺, c⁻)`.
* Memory stays append-only: an ending is a new row, never a value written
  back into an old one. A feeling that lasts for days is two rows, its
  onset and its end: "in neither case would we want to represent the
  concept at every instant or location."
* This is the event calculus's law of inertia (Kowalski & Sergot 1986),
  the usual answer to the frame problem (McCarthy & Hayes 1969), and it
  fits item 6.5: memory stores changes, and states are what holds between
  them. WholeSpace segmentation already ends a run where its complement
  begins, and item 9b's brackets are already content-terminated.
* **Retired:** the `eternal` flag, the refresh rule, duration, `timestamp`
  (the store's own counter). A row with no occurrence of its own (a
  user-supplied truth) has the minimum start, as "just a min and max value"
  proposed.
* A query "does P hold at sentence *k* of document *d*" finds P's latest
  onset at or before *k* in *d* and checks that no complement begins in
  between. The index keeps rows by content, ordered by start within a
  document.

*Claude's reading:* reading the same document again finds the same
addresses, so its rows are refreshed in their evidence, not duplicated.

### 3.4 The situation in WholeSpace *(decided)*

The document is a whole "present in every sentence ... this is also how we
discussed telling documents apart with a stream. This suggests that
concepts need access to all parts/wholes, at least once per sentence (which
they have from the parallel read)." And: "our situation or context ...
might also be able to be presented in the WholeSpace."

* Situation wholes nest: sentence, the group read in parallel, document,
  conversation. They are **identities, never clock values**: a code that
  recurs is context; a number that changes every sentence is the defect of
  1.1 in a new place (answered "Yes").
* Indexicals ("now", "here", "this") take their referent from them, as from
  Kaplan's context of utterance (Kaplan 1989). Tense's "now" is the current
  sentence's whole.
* They seed the domain of discourse
  ([accessible mind §2.0.1](2026-09-20-accessible-mind-subsystems.md#201-words-are-a-formula-for-narrowing-attention))
  and are the widest bracket of item 6.8's one attention.

### 3.5 Every text configuration interleaves *(decided)*

"For 5.5, every text config interleaves: we have too much complexity to
manage multiple implementations." Every text configuration sets
`modeSchedule` to `interleave:N`, so each group of N sentences is read in
parallel before it is read serially, and that parallel read is how a
sentence's concepts see every part and whole (3.4). Text configurations do
not keep a plain `serial` or `parallel` schedule. Under interleave a
batch's rows are consecutive sentences (`InterleaveCursor.next_tick`), so a
row's `.when` comes from the sentence's index in its document, never from
the order in which a batch row happens to read it.

## 4. Tense and aspect: prepositions of the verb phrase with no preposition written *(decided)*

"Tense and aspect become adverbs whose action on the VP is fairly specific
(compared to a general adverb). This is similar to the prepositional phrase
... Can it be handled as a compound where the fold happens along the
transformation of the VP?" Answered yes, and: "tense and aspect can be
treated as VP PP with no P marker."

It is the catalogue's compound, "reverse sigma to the cases of the head, a
selection among them by the modifier, and sigma over what is left"
([catalogue §4.3](2026-09-29-operator-catalogue.md#43-what-follows-for-the-update)),
with the verb phrase as the head and its cases its **phases** along the
transformation: the state before, the change, the state after (the
"nucleus" of Moens & Steedman 1988). The result keeps the verb phrase's
order.

| modifier | keeps these phases | selected by |
|---|---|---|
| "in the park" | the phases located in the park | place |
| "into the park" | the state after, which is in the park | place |
| prospective, progressive, perfective, perfect | before; the change up to R; the whole; after | position relative to R |
| future, present, past | the phase that holds now: before, during, after | position relative to now |
| "yesterday", "for three days" | the phases in that stretch, relative to now | stated amount |

* "Felix walked": now falls in the walk's after-state. "Felix will walk":
  now falls in its before-state. The cases must include the before and
  after states, not only the change, or a past and a future of a whole
  event cannot be told apart in reverse.
* The progressive asserts only part of the change, so "was crossing" does
  not entail "crossed" (Dowty 1977).
* Stacked forms compose: "had been walking" is past, then perfect, then
  progressive.
* It is a **selection, not a gain**: marking a verb past twice does not make
  it more past, as "on the mat, on the mat" says no more than "on the mat".
* Languages still show the preposition these came from: Dutch *ik ben aan
  het lezen*, German *ich bin am Lesen*, English *a-hunting* (from *on
  hunting*), French *en train de*, Irish English *I'm after eating*,
  *about to*, *going to*. Temporal adpositions mostly come from spatial ones
  (Haspelmath 1997); progressives mostly from locatives, futures from
  motion and perfects from "have" and "finish" (Bybee, Perkins & Pagliuca
  1994).

**R, the reference time aspect is measured from (Reichenbach 1947)**
*(decided, Alec, 2026-09-30)*: "The time does not follow the reference."
R is now, or an event the sentence refers to, and how the event described
stands to it is an order conceived in the concept ("Mary had left when
Felix arrived": the leaving is before the arrival). A reference gives the
event, never a time: nothing reads a referenced row's `.when` to place the
sentence, which would carry an occurrence into what is conceived (section
2).

**References between sentences point back** *(proposed, Alec, 2026-09-30:
"all (inter-sentence) references are back-references"; Claude agrees)*. A
reference names a row that exists, and a row is written when its sentence
ends, so a sentence can refer only to what was read before it. Where
language refers forward across sentences ("Listen to this: the market
crashed"), the later sentence writes the link, back to the earlier word.

**Needs item 6.5.** The phases are item 6.5's frame before, verb column and
frame after. The catalogue notes that the choice of case on descent is not
yet conditioned on a modifier, for any compound; this uses that wiring.

## 5. The preposition, closed out *(decided)*

Asked: "do we need separate NPP, VPP, and spatialP and temporalP? Or just
two of those, since NP are spatial and VP are temporal?" Two, and they are
the two attachments already decided (a preposition's phrase on a noun
phrase makes an adjective, on a verb phrase an adverb). There is one
operation, the compound of section 4:

* **The head decides what is selected.** A noun phrase's cases are things,
  laid out in space; a verb phrase's cases are phases, laid out in time.
* **The object decides the test.** A place compares location ("sat on the
  mat"); a time or an event compares order ("the meeting after lunch").

There is no separate spatial or temporal preposition. Tense and aspect are
the verb phrase's prepositions whose object is now or R and whose
preposition is not written.

## 6. Surface form and markers

*Decided (Alec, 2026-09-30), on the markers:* a marker is its own leaf,
composed by the grammar, "especially when demarcated by punctuation" (with
item 6.8's word whole, "England's" is `England`, `'`, `s`); a regular form
keeps no definition row, and a marker is minted when it recurs ("Sounds
preferable"); the surface operations are trained by reconstruction alone,
under a parsimony cost ("Yes"). Irregular forms are stored and block the
rule (words and rules, Pinker 1999); "went" is found to be "go" plus the
past by its code, not by a table.

**One `surface` operator** *(Alec: "It feels better to have a surface()
operator that does one or more of these suboperations")*, with four
declared suboperations, each the reverse of another, so reconstruction
through the grammar's reverse stays exact:

| suboperation | reading | generation | examples |
|---|---|---|---|
| absorb | marker + X → X | X → marker + X | "do" in questions, "it rains", agreement -s |
| split | form → stem + marker | join | walked → walk + ed |
| insert | X → X + a silent element | drop it | "sheep" (zero plural), a left-out "that", an unstated subject |
| transpose | X Y → Y X | the same | "is Felix here?" |

Today's coded `surface` is absorb (`M·marker + content`, a learned prior
proposing the marker in generation). **One suboperation per step, and a
word may take several steps** *(decided, Alec, 2026-09-30: "yes")*. Split,
insert and transpose do not shorten the sequence, so each is admitted at
most once per position per sentence, and every step pays the parsimony
cost.

**The plural** *(decided, Alec, 2026-09-30)*. "'Cats' is not 'one order
up', it is moved down one order because of an implicit determiner", and
"'Cats purr' means for me that a cat-thing is a purring-thing". This is the
reading two truths §1 decided: "cats breathe" is a relation, "the cat region
part of the predicate's region", with the predicate nominalised as a type
([two truths §1](2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided)).
The region is what the fold's inverse leaves, the extension one order down
(catalogue §5.2, "a cat"). So:

* **"Cats"** lowers by an implicit determiner, which is plural: "I like
  the determiner 'all'. Could be 'some'. Could not be 'a' or 'the', since
  it's singular" (Alec, 2026-09-30). "Cats purr" is all cat-things being
  purring-things, the part relation between the two regions. A cat that
  does not purr adds evidence against beside the evidence for, the corner
  *both*, which prompts refinement (decided 2026-09-23): one exception does
  not make the relation false.
* **"Every"** lowers like "all" and differs from it in number: "every
  lowers like all but has a different plurality" (Alec, 2026-09-30). This
  amends 2026-09-28's "every does not lower". In a generic, "every cat
  purrs" and "cats purr" write the same part row, and their difference in
  number is the surface's (Alec, 2026-09-30: "yes").
* **"There are cats on the lawn"** is how the existential is said ("'cats
  are on the lawn' does not parse well for me"). `surface` absorbs the
  dummy "there", and the sentence is about some cats in one situation: an
  idea, as "this cat breathes" is in two truths §1.
* Claude's earlier answer, that the plural cues the set one order up, is
  withdrawn: that set is what the fold makes (decided 2026-09-28), and the
  plural names its members.

Agreement is absorbed with no effect and re-inserted from the subject in
generation.

## 7. What is removed

Under the no-legacy rule: `TenseLayer`, `AspectLayer`, `MorphologyLayer`,
`_lift_when`, `_lower_when`, `surface_morphology.py`, `surface_tense.py`,
the `tense` and `morphology` rules of `complete.grammar` and
`ladder.grammar`, the `.when` band of every perceptual space, the `eternal`
flag, `timestamp`, the clock-stamped row `.when` item 7 adds, and the tests
that exist only for these. `when_time` stays a count of training steps and
nothing semantic reads it.

## 8. Tests

Mechanism tests, unconditional:

1. **The clock is gone from ideas.** The same sentences read at nine model
   step counts give identical derivations, stored rows and answers (1.1's
   probe, inverted).
2. **Position.** The same sentence at two absolute offsets in the input
   gives the same derivation and row.
3. **The address.** Sentence *k* of document *d* is written with `.where`
   = *d* and `.when` = *k*, from 1 in each document, whichever batch row
   read it; reading *d* again refreshes the same rows.
4. **No order across documents.** Two documents read in one batch are
   never compared in time; recency and "holds at" stay within a document.
5. **The complement.** "Felix is sad" at 2 and "Felix is not sad" at 5 in
   one document: sad holds at 3 and not at 6; without the second row, it
   holds at 6.
6. **Absences.** No `.when` band on a percept or a concept; no duration,
   `eternal` flag, refresh path or `timestamp`; none of section 7 remains.
7. **The situation.** A document appears as a code in WholeSpace; no
   situation whole carries a clock value.
8. **Interleave.** Every text configuration loads with `interleave:N`.
9. **Selection.** A preposition's modifier applied twice equals it applied
   once, on a noun phrase and on a verb phrase.
10. **Surface.** Each suboperation's reverse is exact given its operands.

Learning measurements, recorded and never tuned, after item 6.5's verb
columns and item 9's prerequisite: past and future of a whole event told
apart in generation; the progressive not entailing completion; a marker
recovered in reconstruction; "went" found as "go" plus the past. The
standing XOR table passes on the candidate as on HEAD.

## 9. What it touches

* **Item 7.** Its row `.when` (a clock stamp) becomes (document, index);
  test 27's "addressed by its `.when`" becomes a lookup by that address.
* **Item 9b.** The field keeps its bracket; its interval goes with the
  percepts' `.when`.
* **Item 6.5.** Binding evidence (continuity in space and time) reads the
  rows' (document, index); the phases of section 4 are its frames.
* **Item 6.8.** The situation wholes are its widest bracket.
* **Item 6.9.** Its gates read the answer right after training, where the
  clock moves it least (.061 to .068 over ten ticks); a gate run long after
  training would not hold until this lands.
* **Item 5.** Forgetting's age term "can probably just be an increasing
  document index: more important within a document will be its salience"
  (Alec, 2026-09-30).

## 10. Questions for Alec

Answered on 2026-09-30: one suboperation per step, "yes" (section 6); "the
time does not follow the reference", and references between sentences
point back (section 4); forgetting's age is an increasing document index
(section 9); the plural lowers by an implicit plural determiner, "all"
or "some", and "every" lowers like "all", writing the same row in a
generic (section 6).

No question is open.

## 11. References

Bybee, J., Perkins, R., & Pagliuca, W. (1994). *The Evolution of Grammar.*
University of Chicago Press. Dowty, D. (1977). Toward a semantic
analysis of verb aspect and the English "imperfective" progressive.
*Linguistics and Philosophy* 1, 45–77. Haspelmath, M. (1997). *From Space
to Time: Temporal Adverbials in the World's Languages.* Lincom Europa.
Kaplan, D. (1989). Demonstratives. In Almog, Perry & Wettstein (eds.),
*Themes from Kaplan.* Oxford University Press. Kowalski, R., & Sergot, M.
(1986). A logic-based calculus of events. *New Generation Computing* 4,
67–95. McCarthy, J., & Hayes, P. (1969). Some philosophical problems from
the standpoint of artificial intelligence. *Machine Intelligence* 4.
Moens, M., & Steedman, M. (1988). Temporal ontology and temporal
reference. *Computational Linguistics* 14(2), 15–28. Pinker, S. (1999).
*Words and Rules.* Basic Books. Reichenbach, H. (1947). *Elements of
Symbolic Logic.* Macmillan.
