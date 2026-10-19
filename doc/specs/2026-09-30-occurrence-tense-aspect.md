# Item 5.5: where and when a sentence occurred, tense, aspect, the preposition and surface form

> **Status:** specification, 2026-09-30, written by Claude from Alec's
> decisions in conversation on 2026-09-30; **revised 2026-10-03 (§0): `.when`
> stays in percepts and concepts as the sentence's index in its document,
> and the address is content over the bands.** Todo item **5.5**, after item 6
> and before item 5: "So 5.5?" Codex implements; Claude reviews. It closes
> the last group of the [operator catalogue](2026-09-29-operator-catalogue.md#9-the-order-of-the-pass)
> (the preposition, tense, morphology, aspect and `surface`) and changes what
> `.where` and `.when` mean. It is one check-in: "with 5.5, so that we can
> handle issues in one checkin." **Decided** is Alec's decision. *Claude's
> reading* is Claude's, to be confirmed. *Measured* means a probe run on a
> clean copy of HEAD `1678ee1f` on 2026-09-30; the probe scripts are not in
> the repository and nothing in it was edited for them.

## 0. Revision of 2026-10-03: `.when` stays, and the address is content *(decided, Alec)*


**Implementation note, operators round 4a-0 (2026-10-07; review pending).**
The 10-03 relative `.when` value is implemented as `[i, i+1]` with the
source sentence's document position. Stable document/content address hashes
replace occurrence counters and random store namespaces. Re-reading
re-witnesses the row; the absolute model clock is written to `timestamp`
only, and full capacity raises. Source keys address the document here;
the situation code in content and learned before/after remain item 5.5.
[Receipt and reader inventory](../benchmarks/2026-10-07-operators-round4a0/).

Alec, 2026-10-03: "I see it as a valuable value when stamped inside
concepts, just as .where is currently stamped, so that concepts have
information about temporal order that they would not otherwise have access
to." And: "if we do content-based addressing over LTM, then conceptual mind
can do its own .what, .where and .when querying without any explicit index
management, which seems desirable. So I'm thinking 5.5 should not remove the
.when."

The defect of §1.1 was the **value**, not the band: a global clock, the same
for every word of a batch and different every batch, carrying time of
reading rather than order, at a norm that dominated the content. The band
stays, with the value `.where` already has one level up:

- **§3.2 reverses.** Percepts and concepts carry `.when`. Its value is the
  sentence's **index within its document**, from 1, encoded as `.where` is
  (the existing `when_encoding` applied to the index instead of `when_time`)
  and scaled like `.where`, so that it informs the chooser without
  dominating. The same sentence of the same document gets the same stamp at
  every reading; sentences of one document are ordered; documents are not.
  Every word of a sentence carries the sentence's stamp (word order is
  `.where`), so its ideas carry it, and the row written at the closing
  carries it: memory's `.when` and the concept's `.when` are one value,
  written once.
- **§3.1's address is the row's own content and bands**, queried by
  content. The **document is a code**: the situation whole of §3.4, present
  in every sentence's content, never a band value (a whole is a type; a value
  that changes every sentence in WholeSpace is §1.1's defect in a new place).
  So the bands keep their positional meaning: `.where` the place in the input
  (word order, as today), `.when` the place in the document. "What did
  sentence *k* of document *d* say", "what holds now" and "what was said
  before this" are one kind of query. No index is kept beside the store;
  `MemoryIndex._scope_contains`, which reads a query's metadata today,
  becomes a band match. **No `.where` or `.when` is passed to the LTM query
  as a parameter** (Alec, 2026-10-03): the query is a conceptual event,
  matched over the whole vector. **No mask, pre-defined or learned** (Alec):
  a band the query leaves at zero contributes nothing to the match, so the
  filter's own content selects. The query events are formed by the thought
  and generate operators from the current idea, and those operators are
  trained by what uses the recall (expectation, output): "the information
  relevant to time will find its way into the gradient, and knowledge about
  before and after will be learned through experience".
- **§3.3's recency and "holds at *k*"** are content queries over `.what`,
  `.where` and `.when` within a document; the index "by content, ordered by
  start" is not a separate structure.
- **§7** no longer removes the `.when` band of the perceptual spaces; the
  clock-stamped row `.when` becomes the index-stamped one. **Test 6**
  inverts (the band is present and equals the index); **test 3** becomes a
  content query; **test 1** stands (the same sentences at nine step counts
  give identical derivations, rows and answers).
- **Order must be learnable, so the encoding is structural.** `.when`
  takes the form `.where` already has: an endpoint-sum bracket, [start, end]
  as centre and extent, so that order is a simple function of the channels
  and an operator producing "before this" or "after this" from the current
  idea has something linear to learn (a bare phase would make "before"
  ambiguous across its period). Containment on the bracket remains available
  as a hard read for reasoning, as `.where`'s mereology has it; the recall
  filter itself is soft similarity. Nothing codes before and after; the
  retired `_lift_when`/`_lower_when` were a hand-written version of the
  operator that is now learned.
- §2 is unchanged: the band is occurrence order, written by the reading from
  outside the concept; what the sentence says about time stays in the verb
  phrase (§4).

§3.4 (the situation as identities in WholeSpace) stands and stays in 5.5.
§3.5 (interleave everywhere) is delivered by item 6.8's open read, which
deletes the schedule knob it names. The sections below are as written on
2026-09-30; where they conflict with this revision, the revision governs.

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

## 12. Amendment of 2026-10-17: location is subjective, order is the stream's history, markers belong to operators *(decided, Alec, 2026-10-17; written by Claude)*

Decided in dialogue on 2026-10-17 before the 5.5 mechanism round; where it
conflicts with §0–§7, this amendment governs.

- **`.where` is the subjective location of the experience**: where in the
  field the percept occurred. Two documents can occupy the same location (a
  book in front of the face), so a document is not a `.where` region and
  takes no whole row; §3.4's situation wholes in WholeSpace and the
  remaining "documents as situation codes in content" are withdrawn in that
  form. What tells a book apart is its history: the cover was looked at
  before the pages. **The model reads its context from LTM**: its own
  history (below) and the content queries of §0.
- **Order is the stream's history.** A stream is one subject's sequence of
  experience (one batch row across its presentations). Within a stream every
  row is ordered by occurrence, which is the store's own order; streams are
  not ordered with each other. *Decided (Alec, 2026-10-17, after the landing
  review):* a row that is merely updated by re-witnessing does not move; its
  place in the history is its first occurrence. `.when` stays the sentence's index within its
  document (§0). The stream's history chain replaces every reader of the
  global clock: the recency chain and the occurrence centroid's weight.
  *Claude's reading, to be confirmed:* a document is then an individual of
  the referent ladder, minted at its first sentence and bound by continuity,
  referred to from each sentence's row by a slot as an NP refers to its
  individual; the open contexts (conversation ⊃ document ⊃ paragraph ⊃
  sentence) are the stream's recently bound individuals, and "what was said
  in *d*" is a reference query.
- **Sub-word wholes nest.** A recurring sub-run inside the predefined word
  whole (item 6.8's maximal run of letters) is a nested whole: letters ⊂
  morpheme ⊂ word, since we are parsing words. The word stays the attention
  unit; a leaf's code carries its parts, so "walked" is walk plus ed at the
  code level and an irregular ("went") has no nested whole. `split` leaves
  the grammar. *Decided (Alec, 2026-10-17, after the landing):* admission uses
  the **type criterion** — a residual such as `ed` is a nested whole when it
  recurs across distinct word types, not when one word is read twice — and
  the nested affix is a **temporal affix that operates as a spatial
  preposition does**: the marker leaf of the verb phrase's preposition of §4–5,
  so `walked` is the preposition operator with `ed` as its written marker on
  `walk`. *Decided (Alec, 2026-10-17, later):* **nested admission learns one pattern
  for one rule at a time.** It is not a general carving statistic over the
  vocabulary (a probe of the landed rule carved 7 of the 17 4e words at shared
  letter pairs). A stage introduces one marker pattern: the grammar is run
  without the marker for tense, then the marker is turned on and a set of
  sentences is tested, as a human learns one generalization at a time. With
  no pattern named, no word is carved; with a pattern named, only words whose
  remainder is a witnessed stem carve into stem plus that marker, and
  irregulars stay whole.
- **A marker belongs to its operator** — to one or more of its selections,
  and a selection may need several markers; the map is many-to-many
  (§13.1, 2026-10-19). A marker-headed operator absorbs the
  marker leaf when it is present and supplies its own learned marker when it
  is absent (a unary face of the same implementation whose reverse emits no
  leaf). Insertion cannot occur in an arbitrary context with an arbitrary
  word, so `insert` leaves the grammar; the zero plural, the omitted
  relativizer and the unstated subject are the operator without its leaf,
  the last bound at the close by reference. Tense and aspect, §4–5's
  prepositions "with no preposition written", are this same face.
- **Transpose stays**, as an operator on the two newest constituents (so it
  acts on sub-trees), self-inverse and exact; source order is strict and the
  swap explicit and charged. The parsimony cost of §6 is the field's existing
  step cost on every step; no separate surface charge.
- **§8 tests, amended.** Test 3: sentence *k* of document *d* is written at
  `.when` = *k* from whichever stream read it, and reading *d* again
  re-witnesses the same rows. Test 4: no order across streams; within a
  stream, order by occurrence. Test 7 is withdrawn with the situation
  wholes. Test 10 covers absorb, transpose and the unary face's reverse.

## 13. Across languages: the operators are the universality *(decided, Alec, 2026-10-19; written by Claude)*

Alec, 2026-10-19: "In English tense and aspect modify the verb, and
prepositional modification takes care of modifying the verb with positional
information. But is this true for all languages?" It is not. "We cannot
mention any languages in the code, or even special case them. A UG needs
to parse any human language." And: "There need be no table; I am
proposing that operators replace the table. If we don't have the
operators to do that, then our grammar is not universal."

So this section is the operators and their capabilities, and the claim
that they suffice. There is no inventory of placements, no language, and
nothing in the mechanism that knows which arrangement it is reading. If a
human language turns out to write a selection in a way these operators
cannot reach by exposure, that is a finding that the grammar is not
universal, recorded against the catalogue — never a case added.

### 13.0 The rule

No identifier, branch, parameter, configuration, fixture label or test
name names a language or a language-specific form. §7's removal of the
table of English forms stands and no table for any language replaces it.
Verified 2026-10-19 at `9207f5e`: no language name in `bin/`,
`data/*.xml`, `data/model.xsd` or test identifiers; "English" survives in
two docstrings in `Models.py`, in fixture sentences, and in test comments
that disclaim a parsing claim. The docstrings go with the 3.5 landing.

### 13.1 The operators, minimal *(Alec, 2026-10-19: "the operators should be minimal … just one rule called twice")*

Minimality rule: a capability that can be had by calling an existing
rule again is not a rule. So there is no set-valued selection, no
double-location rule, no spatial and temporal pair, no affix face — each
of those is one of the following called once more.

1. **`position`** — the one rule of §4–5: unfold the head to its cases,
   select among them by the object, fold what is left. It reads its two
   operands and nothing else: the head's cases are whatever the head has
   (things, phases, the phases of a role held over time), and the test is
   whatever the object's content supports (location, order). **Called once
   per selection.** Two selections — a place for the after-state and the
   whole event; a path with two landmarks; past, then perfect, then
   progressive — are the rule called twice or three times, in the
   surface's order, each call idempotent (§4). A leaf that a language uses
   for two selections is the rule called twice on one leaf: the first call
   absorbs it, the second runs without one. **Called with or without a
   leaf**: the same rule, whose reverse emits a leaf exactly when one was
   read (§12's unary face). **Not called** keeps every case, which the
   answer reader reads as *unknown*, never *now* — every sentence's state
   before its marker is read, and how a sentence with no marker is read.
2. **The marker leaf** — one kind of leaf, minted on recurrence at a
   marker position (§6, §12). Its reading is how many times, and with what
   object, `position` is called when it is absorbed; that is learned by
   exposure and never declared. It is read at its position, independently
   of any other reading the same form has elsewhere. A free word and a
   nested whole inside a form are the same leaf: the nested whole is made
   at admission by the type criterion (§12), on a stem of any class, and
   is not a grammar rule.
3. **`surface`** — absorb, on either side of the head, and transpose, on
   any two newest constituents. Nothing else: `split` and `insert` left
   the grammar in §12, and no face is added here.
4. **The leaf's own content** — not a rule. A verb whose meaning already
   places its after-state is a verb; `position` finds the phase located
   in the content and is not called.

**The claim.** Any way a human language writes a selection of time or
place — on the form of the head or of the object, as a free leaf on either
side, as nothing, inside the leaf's own meaning, as one leaf for several
selections or several leaves for one — is these three, each called as
many times as the sentence needs, reached by exposure under one
configuration. That is the universality of this part of the grammar, and
it is measured (§13.2), not configured. A language that needed a fourth
is a finding against the catalogue, recorded, never a case added.

### 13.2 The gate (for the tense and aspect pass of item 3.5)

**One grammar, one configuration, nothing named**, reads a generated
corpus in several variants and passes §8's test 10 (the surface reverse
exact) and the phase questions of §4 under each. The variants are the
*generator's* choices about where it writes the marker, named by what the
generator did and never by a language:

- the temporal marker written as part of the verb's form;
- the temporal marker written as a separate leaf, the verb's form
  unchanged;
- the spatial marker written as part of the noun's form;
- one leaf written for two selections (`position` called twice on one
  leaf);
- two leaves for two selections on one head (`position` called twice);
- no temporal marker written at all, where the phase questions answer
  *unknown*.

The generator (`test/constituent_corpus.py`) is the only place these
choices exist, and it is test tooling, not the model. The gate passes when
every variant is learned by the same mechanism with no change to it. No
natural-language corpus is needed for this gate, so it does not wait for
the developed corpus that the learning claims wait for. Tense and aspect
also need the phases populated, which is the VP temporal-difference
teacher of item 3.5.

### 13.3 Literature

The requirement is real: tense and aspect are written on an auxiliary, as
free particles, or not at all (Comrie 1976, 1985; Dahl 1985; Li & Thompson
1981); tense is written on nouns (Nordlinger & Sadler 2004; Evans 1995);
path is inside the verb in verb-framed languages and on a satellite in
satellite-framed ones (Talmy 2000); one preverb may encode direction and
perfectivity at once; the same selections move between word, affix and
zero over time (Bybee, Perkins & Pagliuca 1994; Haspelmath 1997). The
localist tradition (Anderson 1971) holds that time and place are one kind
of relation, which is what §5 decided.

## 14. Two differences: the verb across time, the modal across possibility *(Alec, 2026-10-19; written by Claude)*

> **Modals and counterfactuals are a second difference. They extend the
> space of events into possibility space.** (Alec, 2026-10-19: "a big part
> of my work in cognitive set theory and the whole part.")

This section records the two differences side by side, because the
second is as large a part of the design as the first and had been
mis-stated in discussion as "modals do not teach". They do. They are the
difference that opens the event into the possibilities it is one of, and
the operation that makes a whole of which the actual event is a part.
Alec's books, *Cognitive Set Theory* and *The Whole Part*, are the
source, cited as the books attached to his ORCID record,
[0000-0002-6570-7571](https://orcid.org/0000-0002-6570-7571) (Alec,
2026-10-19: "just reference them as books attached to my ORCID"); this section gives the mechanism's reading of them and
marks what is his and what is Claude's to be confirmed.

### 14.1 The first difference: the verb is a group element acting on the identity *(Alec's direction, 2026-10-19, corrected the same day; Claude's reading, to confirm)*

Alec: "Verbs restrict the all-ones temporal dimensions (whichever those
turn out to be) to alter the temporal possibility space that the noun
phrase otherwise has … If 'Alec runs', then the identity that covers Alec
undergoes a change, since that entity may be tired after the completion of
the sentence." Then, correcting the picture: "I am trying to understand the
vector space we are creating in mereological and Boolean terms, and that
may be overly constrictive. I am probably just asking for an **identity
element in an abelian or some other kind of group as defined by the
operation of the VP** (which may be defined by object before and object
after, but a strict subtraction would probably not be sufficient; we had a
matrix operation previously where the verb defined the diagonal, and that
seems like a much more likely transformation than an actual difference)."

The reading:

- **The verb is an element of a group acting on the identity's frame.**
  A verb phrase is a map `v: before ↦ after` on the identity's occurrence
  frame (the shared index: the perceptual chart at order 0, the two
  evidence lanes per symbol above it; item 6.5's frame before, verb
  column, frame after). The verbs form a group `G` under composition: an
  identity element `e` (the verb that changes nothing), an inverse for
  each verb (the change undone), and composition for verbs in sequence.
  "All-ones" named `e`, not the top of a Boolean lattice. **An undefined
  verb is `e`**: meaningless, read as *unknown*, never as false.
- **The form the code already has.** `<verbSpectrum>` (`Language.py`,
  confirmed 2026-06-20) applies the verb as a diagonal in the chart:
  `VP(NP) = chart⁻¹(e^{w_v} ⊙ chart(NP))`, with `w_v` the verb's sparse
  log-eigenvalues from its code, zero-initialized and soft-thresholded,
  and `unapply` applying `e^{−w_v}`. That is the group of positive
  diagonals under multiplication: abelian, `e` the unit diagonal (`w = 0`),
  inverse the reciprocal, composition the sum of logs. The verb defines
  the diagonal, as Alec recalls; the first difference is **not a
  subtraction but a quotient**, `w = chart(after) − chart(before)`
  coordinatewise — a difference of logs, which is what "the verb defines
  the diagonal" means as a teacher. The follow-on already noted in the
  code, an NP-conditional basis `Q` so the verb acts as `Q·diag(e^w)·Qᵀ`,
  is what a non-abelian sequence ("sat, then stood" against "stood, then
  sat") would need; until then order between verbs is carried by the rows
  (§3.3, §12), not by the group. **A verb with several participants is
  not a larger tensor**: "give" is `VP → VP NP` applied twice (Alec,
  2026-10-19), the catalogue's one rule called again (§13.1's minimality),
  each application its own diagonal on its own participant.
- **Recognizability is the verb's support over the noun phrase's
  extent** (Alec, 2026-10-19). "Break" makes the object unrecognizable
  because the verb's support covers most of the object's extent: the
  after-frame no longer matches the identity that the before-frame was an
  occurrence of. The map is still invertible — `unapply` returns the
  before-frame exactly — but inversion is informational, what it was, and
  does not make the after-object recognizable *as* the original. "Alec
  runs" leaves Alec recognizable because the support is small against his
  extent. So the fraction of the identity's extent that the verb's support
  covers is a measure the close reads: below it the after-frame
  re-witnesses the same individual, above it a new individual is minted
  (the pieces) and bound to the old by the row (item 6 §15 items 2–3:
  mint, then match).
- **Support, zero by default.** The verb's support is the set of
  coordinates where `w_v ≠ 0`. A fresh verb is `e` everywhere — support
  empty, undefined. A coordinate enters the support when the log-ratio
  recurs across occurrences (the type criterion of §12), and the sparsity
  threshold is the admission. Outside the support there is no expectation
  (κ = 0). A correlate that recurs for one identity and not for others
  (tiredness after running: Alec, not a car) never enters the verb and
  stays an expectation on the identity (item 4).
- **Where the quotient is undefined, nothing is learned.** A coordinate
  that is zero in the before-frame has no log-ratio; that occurrence
  teaches nothing there. The frame's coordinates are presences and lanes,
  never signed, so a positive diagonal keeps them in the chart; adding a
  property ("reddens") is the against lane of red driven toward zero and
  the for lane scaled up where it was present at all.
- **Two algebras, not one.** The Boolean and mereological picture is the
  right one for *selections*: `position` (§4–5, §13) is an idempotent meet
  with no inverse — on the mat twice is on the mat; past twice is past. The
  verb proper is the other algebra: a group element, with an inverse and
  not idempotent — ran twice is more than ran once. Tense, aspect and the
  preposition select among a change's phases; the change itself is the
  group element. The dimensional `lift`/`lower` stays in FutureWork.
- **States are not group elements.** A synchronic statement ("Alec is
  tired", "the box contains apples", "cats breathe") is a relation and a
  relative truth (Alec, 2026-10-19; two truths §1); it is a selection
  (part of a region) located in a situation, and its start and end come
  from the rows (§3.3, a complement ends it). An idea with a verb proper
  is a change: an element other than `e`. (The earlier draft's "a state
  restricts every phase alike" is withdrawn.)
- **The teacher** (item 3.5's verb line, formerly "VP vectors by temporal
  difference"): per occurrence, the identity's frames before and after the
  sentence are read from the stream's history (its occurrence rows; item
  6's occurrences, nothing new); `w` is their coordinatewise log-ratio;
  across occurrences the recurring coordinates pass the threshold into the
  support. The participant the quotient is taken on is the verb's content
  ("broke the glass" changes the glass); an impersonal ("it rained") acts
  on the situation, §12's document individual. `<verbSpectrum>` already
  has the sparse log-eigenvalues and the exact `unapply`; what is missing
  is this teacher.

### 14.2 The second difference: the modal extends the event into possibility space *(Alec, 2026-10-19, decided in principle)*

The first difference is taken between phases of one actual history. The
second is taken between the actual event and the events it could have
been — across possibility at a time, not across time in one possibility.
A modal or a counterfactual does not fail to assert a change; it asserts
a **set of events of which the actual is a member**: it makes a whole of
which the actual event is a part, and the second difference is that whole
minus that part — the merely possible. In §14.1's terms, the first
difference picks an element `v` of the group; the second extends `v` to a
set of elements of which `v` is one, and a counterfactual names a member
of that set differing from `v` at one coordinate.

| | first difference | second difference |
|---|---|---|
| taken between | the phases of one event (before, after) | the actual event and the set it is extended into |
| algebra | the group: an element, an inverse, not idempotent | a selection over worlds: idempotent (might-might is might), no inverse (§14.4) |
| direction | from the default `e`: a change | from the default *all possible worlds* (§14.4): the indicative narrows to the actual; a modal narrows less, so relative to the indicative it extends |
| what is written | the identity's occurrence rows (ideas, ultimate) | a part relation: the actual event a part of the possible set (a relation, relative) |
| the default | `e`, "exists": no change | all possible worlds; the indicative is the selection of the actual |
| examples | "Alec ran", "the box contains apples" | "Alec might run", "Alec could have run", "if Alec had run, he would be tired", "Alec must run" |

Readings of the forms, in the set-theoretic terms:

- **Possibility** ("might", "can", "could"): the event's restriction is
  re-opened on the named coordinates; the actual is one member of the
  extended set. The against lane is not asserted; the for lane is asserted
  *of the set*, not of the actual.
- **Necessity** ("must", "has to"): the extended set is narrowed back to
  the event on every member — the whole is restricted as the part is.
  Necessity is possibility with the complement withheld, which is why the
  two are duals.
- **Counterfactual** ("if Alec had run, he would be tired"): a member of
  the extended set that differs from the actual in exactly the
  antecedent's restriction, with the consequent asserted of that member.
  It is the second difference taken at one coordinate, then the first
  difference run forward inside it.
- **Future and prospective** (§4) sit between the two: a prospective is
  the first difference's before-phase; "will" with no evidence yet is the
  second difference's extension of now.

### 14.3 The minimal mechanism: the modal is the verb phrase's preposition over worlds *(Alec and Claude, 2026-10-19; decided)*

Alec proposed `S → NP VP MP`, "similar to the VP except that it is a
difference over VPs instead of NPs", and asked for a binary replacement.
There is one, and it adds no rule: the modal phrase is `position` (§13.1)
with a **world** as its object, exactly as tense is `position` with a time
and a locative is `position` with a place:

    S  → NP VP
    VP → VP PP          position, called once more
    PP → P  W           a marker leaf, possibly absent, and a world

The test the object supports is **part-of**: the actual world is a part of
the worlds the modal keeps. The default "in all possible worlds" (§14.4)
is the call with no selection — every world kept. "Might", "could",
"must", "would", the conditional and the subjunctive are marker leaves
read by exposure like "yesterday" and "into"; a language that writes mood
on the verb's form is the nested-admission case of §12. No `MP` rule
exists, and the ternary rule is not needed: it flattened two bracketings
that the binary grammar keeps apart.

**The two bracketings are the two truths** (Alec, 2026-10-19), written
in the two-truths syntax Alec fixed the same day — an absolute truth is
`S`; a relation between two truths is `S REL S`:

| reading | derivation | written as | truth |
|---|---|---|---|
| *de re*: of Alec, running is possible | the modal's `position` on the verb phrase, before it acts on the noun phrase | `S REL S` read as membership: `[Alec]` in the set `[running things PP(W)]` — written as taxonomy evidence, not a row, the predicate nominalized and located in the worlds the modal keeps | relative (two truths §1, as "cats breathe") |
| *de dicto*: that Alec runs is possible | the modal's `position` on the sentence | `S`: `[Alec runs] PP(W)` — the sentence's own occurrence row, located in the worlds the modal keeps | absolute, in those worlds |

The syntax is the store's own shape (Alec, 2026-10-19: "that corresponds
exactly to the use of LTM and STM slots"; [two truths §1, amended 2026-10-19](2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided)):
a row has three positions, argument, kind, argument. An absolute truth `S`
fills one position with its fused point — a vector, possibly unminted, in
STM and in LTM1; minted on recurrence. A relative truth `S REL S` fills all three, the kind being a
relation that takes a truth as an argument (`DEF`, *said*, *if-then*) and
each argument an idea or a REF to a row. Parthood and kind membership are
not rows but folds in the main tree (meronomy, taxonomy). So the *de re*
reading writes membership evidence — Alec in the set of running things in
the kept worlds — and the *de dicto* reading writes the fused point of
`[Alec runs]` as an LTM1 row with the world `W` as an address column. No new row kind is needed for the modal.
Each `S` has its noun phrase, verb phrase and world, defaulting to
everything, occurs and all possible worlds (§14.4); `[Alec]` and
`[running things]` are such sentences with their defaults, and the modal
sits in the second one's `PP(W)`. "Alec might run" has both readings in
English and languages mark the difference; under the minimal operators the
chooser picks the derivation by exposure, as it picks any attachment, and
the scope ambiguity is the attachment ambiguity. In §14.1's terms the *de
re* reading extends the group element `v` and the *de dicto* reading
extends the resulting state.

What is written, then:

- **A world is an individual**, minted as §12's document individual is and
  bound by continuity; the actual world is the default, and a possible
  world differs from it in the restrictions the modal names. The part
  relation between worlds is the world individual's own row, as a
  document's is.
- ***De re*** writes `S REL S` as membership evidence in the taxonomy:
  `[Alec]` in `[running things PP(W)]` — a relative truth, not a row.
  Nothing enters Alec's history as having happened.
- ***De dicto*** writes `S`: the sentence's occurrence row located in the
  world — an absolute truth, true there. The actual stream's rows are
  untouched unless the world is the actual one (the indicative).
- **The teacher** for the modal is the extension itself: the kept worlds'
  frame is the actual frame with the named restriction re-opened
  (possibility) or imposed (a counterfactual antecedent). That is a
  target, so modals train; what they train is the world's relation and
  its minting, never the identity's occurrences. "Modals do not teach" is
  withdrawn.
- **Reading back:** "did Alec run?" reads the actual stream and finds
  nothing (*unknown*, or *no* if the complement was asserted); "could
  Alec have run?" reads the relation or the located row. Whole and part
  are both there.

This supersedes rev 1 of this section (situations as the object and a
part relation between situations as the only row); situation and world
are the same individual kind, and the row written depends on the
bracketing.

### 14.4 Defaults: the empty sentence is wide-open attention *(Alec, 2026-10-19)*

Alec: "We can assume some NP for every sentence, by default everything.
Default VP is 'exists'. Default MP is 'in all possible worlds'. Taken
together, these mean that the empty sentence is an assertion that
everything exists everywhere, which is a wide-open attention." (MP is the
modal phrase; Alec's gloss "model".)

Every sentence has the three, and each default is the top of its algebra:

| phrase | default | the top of | a word narrows it by |
|---|---|---|---|
| NP | everything | the part lattice: the whole field, the all-ones attention mask | restriction (words as restrictors, 2026-09-28) |
| VP | exists | the change group: the identity element `e`, asserted | a change: an element other than `e` |
| MP | in all possible worlds | the possibility lattice: every world | selection: the indicative keeps the actual world; "might" keeps a set containing it; "must" keeps every accessible world |

So the empty sentence asserts that everything exists in all possible
worlds — the field before any word, attention wide open — and a sentence
is read as a narrowing from there in each of the three. This resolves two
things above:

- **"Exists" is `e` asserted.** The default VP is the identity element
  with evidence in the for lane; an *undefined* verb is `e` with no lane
  filled. "Remain" and "stay" are "exists" said of a span. The two lanes
  keep the three apart, which is the only place the "undefined is the
  identity" rule needed them (§14.1).
- **Whether "exists" is the right word** *(Alec, later the same day: "I
  wonder if exists makes a narrowing assertion, since ideas do not exist?
  Perhaps there is no English word for it.")* The default verb phrase must
  be the identity element asserted and nothing more, and "exists" asserts
  more: existence, which an idea does not have (ideas are conceived;
  existence is a matter of evidence, [ExistenceEvidence](../ExistenceEvidence.md)).
  Claude's proposal: the project's own word is **occurs** — every sentence
  writes an occurrence row, and "the noun phrase occurs" claims presence
  to the mind in a situation and no more; it holds of an idea, a percept
  and a relation alike, and it narrows nothing. English offers "obtains",
  "there is", "appears" and the bare copula, each carrying a connotation
  "occurs" does not. The spec reads "exists" above as "occurs" until Alec
  decides; existence proper is a further relation asserted by evidence.
- **The modal is a selection, not a group element.** It belongs to the
  same algebra as tense, aspect and the preposition — idempotent, with no
  inverse — but over worlds rather than phases or places; §14.2's table
  is corrected accordingly. The second difference is still the whole minus
  the part: the worlds the modal keeps, less the actual. Under §13.1's
  minimality this makes the modal `position` called with a world as its
  object, which is what §14.3 proposed with "situation" for "world".

### 14.5 Open for Alec

- Whether the first difference's group is left abelian (the diagonal in a
  fixed chart, as `<verbSpectrum>` has it) for item 3.5, with the
  NP-conditional basis `Q` as the later step.
- Whether the possible situation is a minted individual (§14.3) or a
  lane beside the two evidence lanes; the individual keeps the lanes at
  two and reuses §12's situation, which is why it is proposed.
- Whether necessity is to be written as the dual (possibility with the
  complement withheld) or as its own marker reading.
- Whether this section moves to its own spec when item 3.5 reaches the
  verb line; the first difference is that line's teacher.

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

Added 2026-10-19 (§13): Anderson, J. M. (1971). *The Grammar of Case:
Towards a Localistic Theory.* Cambridge University Press. Comrie, B.
(1976). *Aspect.* Cambridge University Press. Comrie, B. (1985). *Tense.*
Cambridge University Press. Dahl, Ö. (1985). *Tense and Aspect Systems.*
Blackwell. Evans, N. (1995). *A Grammar of Kayardild.* Mouton de Gruyter.
Li, C. N., & Thompson, S. A. (1981). *Mandarin Chinese: A Functional
Reference Grammar.* University of California Press. Nordlinger, R., &
Sadler, L. (2004). Nominal tense in crosslinguistic perspective.
*Language* 80(4), 776–806. Talmy, L. (2000). *Toward a Cognitive
Semantics*, vol. 2. MIT Press.

Added 2026-10-19 (§14): Rogers, A. *Cognitive Set Theory*; Rogers, A.
*The Whole Part* ([thewholepart.com](https://thewholepart.com)) — the books
attached to the author's ORCID record,
[0000-0002-6570-7571](https://orcid.org/0000-0002-6570-7571).
