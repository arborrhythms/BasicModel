# Parts of speech: a class is a bundle of places (Claude, 2026-11-15)

A specification of how the model learns which words go where. It is
round C of item 1.8 (§10). It has no text for Codex yet. A rule marked
*default* is Claude's choice where Alec has not spoken.

## 1. Alec's words (2026-11-15)

"I have some concern that the way we are arriving at the parts of
speech classes is not sufficiently rich. For example, we talked about
the collapse of some formal argument parameters into common classes
(what is operated on by a verb and an adjective are the same thing).
But we also have that some classes need to be split (we might need a
difference in part of speech between pronouns and nouns that verbs do
not distinguish). Hopefully that difference can be exposed with
operators like bind(), but can you please think about the languages
that you know, and take a look at the code, and let me know if the way
we are learning parts of speech (i.e. as concepts over words and the
argument specification of operators) is being learned sufficiently
such that it will help us learn new vocabulary rapidly once we get to
five years old by assessment?"

"I would like you to spec the parts-of-speech work while codex
continues with the assessment tool. Your work is probably essential
for language learning to take place."

"a new word will be known by the parts of speech on either side and
permissibility of the grammar".

Earlier, and standing:

- 2026-10-21: "Unit tests can force parts of speech; they may take a
  while to learn, and we need to test without that."
- 2026-11-08: the formal arguments are concepts one order above words,
  initially zero, and parts of speech have special construction and
  grammar use only; the rest is the general concept architecture
  (`doc/DevelopmentHistory.md`, "formal arguments reuse general
  conceptual learning").
- `doc/Orders.md`: "positions that share their words draw their
  arguments together, which is how many formal arguments come to few
  parts of speech."

## 2. What is built

Read in the first machine's working tree, which holds rounds A and B
uncommitted.

- **A formal argument is a concept.** `PartsOfSpeech.formal_arguments`
  names one for every input and output of every rule of the grammar
  (`verb_argin1`, `modifier_argin2`, `bind_argin1`, ...); there are
  about seventy for `data/complete.grammar`. Each is seated one order
  above the words (`prebuild_formal_arguments`).
- **A word joins an argument by a kept reading.** At the kept close
  `PartsOfSpeech.observe_kept` replays the reading's journal; each
  operator applied adds its operands' words to its input arguments and
  the words its result carries (`flow_mask`) to its output argument
  (`ConceptHierarchy.observe_member`). A membership is a constituent
  edge and one trust view of the word's direction. Nothing is written
  from a reading that was not kept, and there is no bar on the kept
  reading's worth.
- **A word's fit to an argument** is the coverage of the argument's
  vector by the word's direction (`ConceptHierarchy.fit`,
  `PartsOfSpeech.similarities`). It is added to the compose chooser's
  scores (`bin/Language.py`, the occupation terms) and fed to the
  generate walk (`GenerateOccupation`). It is a bias, never a rule of
  what is legal.
- **What round B removed.** At the landing source two passes moved
  geometry across orders: an argument's direction was refreshed toward
  its members (`ConceptHierarchy.refresh`), and each word was moved
  toward its argument classes (`ConceptHierarchy.condition`). That is
  how "positions that share their words draw their arguments together"
  and how a word's position was its degree of membership. Both are
  removed under §30.7 of the 1.8 specification (placement compares
  rows of one order only; a class takes no centroid of its members),
  which Claude wrote. *Claude's reading, not measured:* a word's fit
  now compares a word placed among words with an argument placed among
  arguments, two placements made apart, so the chooser's occupation
  term may carry little.
- **Nothing merges or splits a class.** `PartsOfSpeech.collapse`
  reports the cosines between arguments and has no caller.
- **A word never seen** fits every argument at zero, and nothing reads
  its class from the sentence.
- **No form is used:** no ending, prefix or capital.
- **Measured but not gated.** The lessons' criteria read each
  operator's definiteness (`PartsOfSpeech.operation`: how far its
  members fit it better than other arguments, times evidence); all
  their thresholds are undeclared.
- **The determiner operators** (`definite`, `universal`, `some`) are
  commented out of `complete.grammar` since round A. By Alec's
  decision of 2026-11-13 a determiner is a case of the adjective, so
  `the` falls under `modifier`.

## 3. What languages ask of a class

A part of speech is a bundle of places: a word that stands in one
place of the bundle may stand in the others. It is marked by where a
word stands, and in many languages by its form; it is not its meaning.

- *Many places, one class.* The subject of a verb, its object and the
  object of a preposition are one class (`Felix ran`, `the dog saw
  Felix`, `the cat sat on Felix`). So are the operand of a modifier and
  the operand of a verb. This is the merge Alec names.
- *One place, several classes.* `Felix`, `he` and `the cat` can all be
  a subject. Only `cat` takes `the` and `fast`; only `he` needs an
  earlier referent, and becomes `him` as an object; only `Felix` names
  one thing and takes no determiner. A verb does not divide them; the
  modifier, `bind` and the determiner do. This is the split.
- *Like meaning, different class:* `quick` and `quickly`; `eat`, which
  may stand without an object, and `devour`, which may not.
- *Unlike meaning, one class:* `cat`, `idea`, `Tuesday`.
- *Quick learning.* A child of four who hears `a wug` once says `two
  wugs` and `the wug ran`: a word heard in one place of a bundle is
  used at once in the bundle's other places. That is what the
  assessment's unknown probe tests (`doc/plans/2026-11-15-assessment.md`,
  §6).

## 4. Whether what is built is enough

No, and the first reason is not about this design.

1. *What is written now is not evidence.* Memberships come from
   whatever reading is kept, and in the lesson the kept readings leave
   out half the words and apply `bind` 738 times in 160 sentences
   (assessment §3). The classes cannot be judged until the assessment's
   first rungs pass, and no bar would have shown it.
2. *A class by direction serves meaning, not place.* A class's
   direction is where its members point, and a word fits it by
   pointing the same way. That holds while nouns are things and verbs
   are actions; it cannot hold `idea` with `cat`, nor part `quick`
   from `quickly`.
3. *One place does not reach the bundle.* Each of seventy arguments is
   a class on its own. A word kept once in one place is a member of
   that place only. With the conditioning removed (§2) nothing carries
   it to the bundle's other places.
4. *A new word's place is not read from the sentence.* It may come out
   right, since the known words' fits pull them to their places and
   leave the new word the place that remains; nothing ensures it.

## 5. Alec's design (2026-11-15)

Asked what a part of speech should be, given three choices (a class of
words by their places; classes over arguments joined by shared words;
the landing's conditioning restored), Alec answered:

"we need conditioning, I think it would be best to leverage the
existing conceptual mechanism since all words are already concepts.
So would they be drawn together naturally if we create the 70
arguments as concepts one order above the words, and activate them
when they are used in the sentence? That is what we want to happen;
then there would even be conceptual pressure to select words and parts
of speech, and therefore operators, based on priming: the chooser
would not have to work as hard outside of that structure."

And, asked which kept readings should write a word's places: "I think
the last answer addresses this."

So no mechanism special to parts of speech is added: the arguments are
concepts, they are active where they are used, conditioning draws words
toward them, and priming carries the rest. The class of words by their
places, which Claude had proposed (§4 and an earlier §5), is withdrawn.

### 5.1 Would they be drawn together? (Claude's answer to Alec's question)

**The words, yes, with conditioning.** If each word is drawn toward the
arguments it is used in, words used in the same places come to point
the same way: `cat` and `dog`, both after `the` and before a verb, draw
together; `fast` is drawn to the modifier's other place and stays
apart; `he`, drawn to `bind`'s place and never to the modifier's, stays
apart from `cat` though both are subjects. That is the split, from the
general mechanism.

**The arguments, only if what draws them together is being filled by
the same word.** Placement draws together concepts active in the same
context. If the context of an argument is the whole sentence, then in
`the fast cat ran` the modifier's two places (filled by `fast` and by
`cat`) are drawn together, as are a verb's subject and object places in
every transitive sentence: the places that occur together, which is
not a part of speech. If the context of an argument is the other
arguments filled by the same word in that sentence, then `cat`'s
places (the modifier's operand, the verb's subject) are drawn together
and `fast`'s place is not: the places a word shares, which is the
merge Alec named, "what is operated on by a verb and an adjective are
the same thing". Rule 3 takes the second.

**Priming follows.** With the arguments in the field, what the model
expects includes them, and the chooser's term for a word at a place can
read that expectation (rule 5).

### 5.2 Up and down: an EM between words and their classes (Alec, 2026-11-15)

"I don't know if I answered this, but sub and dom concepts should do an
EM algorithm; the word (sub) occurs and pulls its POS (doms) toward one
group. The active POS dom will also pull its word subs together. So
that hierarchical clustering should already be happening in virtue of
the way that vertical priming (up and down) influences distribution
within an order."

So the two halves are: *up*, a word that occurs draws the arguments it
fills toward each other (one group: the bundle); *down*, an argument
that is active draws the words that fill it toward each other. Each
half moves concepts within one order, by what is active one order away.

**What happens today** (read in the source by Claude, not measured):

- *Up: half of it.* A present word activates the arguments it has
  filled: since round B the membership also writes the argument's
  definition over its words (`ConceptHierarchy.observe_member`), and
  the field that placement reads folds definitions upward
  (`Spaces.placement_context`, `FieldProjection.close`). Those
  arguments are then placed among the arguments active in the whole
  sentence (`MereologicalCodes.place_rows`), so they are drawn toward
  every other word's arguments as well as toward each other.
- *Down: no.* Placement moves a word only relative to the words present
  (its own order). Nothing activated one order up reaches a word's
  placement: the fold runs upward only, and the downward pass
  (`ConceptHierarchy.condition`) was removed in round B.

So the clustering Alec describes is not happening yet: the down half is
missing, and the up half is diluted by the sentence. Rules 3 and 4 are
the two halves: rule 3 makes the up pull the word's own group (Alec:
"pulls its POS (doms) toward one group"), and rule 4 restores the down
pull ("the active POS dom will also pull its word subs together").

### 5.3 Each in its own space (Alec, 2026-11-15)

After Claude had written that both passes of the landing source would be
reinstated, Alec corrected it:

""a class taking no centre of its members" is actually correct, let's
dig in. A sub should pull its doms together in dom space as a centroid
of the doms. A dom should pull its subs together in sub space. But the
doms location in dom space does not influence the subs location in sub
space. Does that make sense? The centroid defined by a dom or sub is
unrelated to its own position."

It does. Each order has its own space. A membership between two orders
only says which concepts of one order are grouped together, and the
grouping is done in their own space: a word's classes are drawn toward
their own centroid among classes, and a class's words toward their own
centroid among words. Neither the landing's refresh (a class placed at
its words' centroid) nor its conditioning (a word moved toward its
class's position) does that: each moved a row by the positions of the
other order. Both are therefore replaced, not reinstated (rules 3, 4,
4b).

One consequence: the fit of §2, which compares a word's position with
an argument's position, compares two spaces; a word has to be compared
with the argument's centroid of its words instead (rules 4c and 5). That
is also what lets a word heard in one place fit the others: it lies near
the words of those places.

## 6. The rules

1. *The arguments* are concepts one order above the words, as built
   (`PartsOfSpeech.formal_arguments`, `prebuild_formal_arguments`).
2. *Active where used* (Alec). When a kept reading puts a word token in
   an argument, that argument is present in the sentence's field at
   order one, at the token's place (its `.where`), with the word's
   evidence. `PartsOfSpeech.observe_kept` writes it, at the kept close,
   beside the membership it writes today. The field then holds, at
   each word, the places it took, and placement, `expect`, `intuit` and
   recall read them as they read any concept. Every kept reading
   writes (Alec: "I think the last answer addresses this"); the
   conceptual mechanism's evidence and trust carry its weight.
3. *Up: a word draws its classes together, in their own space* (Alec,
   §5.3). At the kept close, for each word of the reading, the
   arguments active at its place (rule 2) are each moved, by the
   existing placement step (`placement_step`) and its negative samples,
   toward the centroid of that word's arguments (*default:* all the
   arguments it is a member of). The context is arguments only; the
   word's own position takes no part.
4. *Down: a class draws its words together, in their own space* (Alec,
   §5.3). At the kept close, for each argument active in the reading,
   the words present that fill it are each moved toward the centroid of
   that argument's words (*default:* all its members, present or not),
   with the same step. The context is words only; the argument's own
   position takes no part. Absent words do not move (*default*).
4b. *No position crosses an order.* A class is not placed at its
   members' centroid and a word is not moved toward its class's
   position: §30.7, rules 2 and 3, of the 1.8 specification stand, and
   `ConceptHierarchy.refresh` and `condition` are not reinstated as
   they were (each moved a row by positions of the other order).
4c. *A class's centroid of its words* is kept apart from the class's
   own position: a sum of its members' positions in the words' space,
   updated when a member joins or moves (*default*), and read as the
   class's extension. Likewise a word's centroid of its classes in the
   classes' space.
5. *Fit and priming.* A word's fit to an argument
   (`ConceptHierarchy.fit`, `PartsOfSpeech.similarities`) is the
   likeness of the word's position to the argument's centroid of its
   words (rule 4c), both in the words' space, and no longer to the
   argument's own position, which is in another order's space. The
   expected activation of the argument at the word's place (`expect`,
   read as compose reads it today) is added to it (Alec: "conceptual
   pressure ... based on priming"). The sum is the word's occupation
   term at that argument in the compose chooser (`bin/Language.py`) and
   in the generate walk (`GenerateOccupation`). *Default:* the two are
   added.
6. *A new word.* Nothing else is built. Its first reading rests on the
   known words' fits and priming and on what the grammar allows; once
   kept, rule 4 draws it at once toward the words of the places it took,
   and by rule 5 it then fits every place whose words are near it,
   including places it has never stood in (the unknown probe, points 1
   to 3).
7. *Tests may force* a word's places, directly, and say so (Alec,
   2026-10-21).
8. *Read.* The table of arguments and their cosines
   (`PartsOfSpeech.collapse`) is reported with the certificates, and a
   word's nearest arguments are part of the assessment's record of a
   word.

**Not in this round.** Form (endings, capitals) as evidence of class;
agreement; any class over words beside the arguments.

## 7. Certificates

1. *Active where used.* After a kept reading of `the fast cat ran`, the
   sentence's field holds, at `cat`, the arguments `cat` filled, and at
   `fast` the one `fast` filled, and nothing else at order one from
   this rule.
2. *The merge.* On sentences in which the same nouns are a verb's
   subject, a verb's object, a modifier's operand and a preposition's
   object, those four arguments come to be nearer to each other than to
   the modifier's own place; the cosines are reported before and after.
3. *The split.* With `he` read where it binds an earlier referent and
   never after `the` or a modifier, and `cat` after both, `he` and `cat`
   come to point apart, while each stays near the verb's subject place.
4. *Within one order.* After a kept close, moving an argument's own
   position beforehand changes no word's position, and moving a word's
   position beforehand changes no argument's position (rule 4b). A word
   moves only toward words and an argument only toward arguments; a
   frozen row does not move.
4a. *Both halves together* (§5.2). Two sets of words that fill two
   disjoint sets of arguments, over repeated sentences, come apart
   between the sets and together within each, and their arguments group
   the same way; with rules 3 and 4 switched off (the source as it is)
   neither separates. Both are reported, before and after.
5. *Priming.* With an argument expected at a word's place, the
   chooser's term for that word at that argument is larger than without,
   by the expected activation.
6. *A new word.* After one kept reading of `the wug ran`, `wug`'s fit
   for the verb's object place, where it has never stood, is nearer to
   `cat`'s than it was before the reading.
7. *Storage.* No table of word by argument; the field rows and the
   placements round-trip a checkpoint.
8. *The unknown probe* of the assessment is read and reported on a
   model whose known probe passes to rung 6; a measurement, with no
   bar, until a model passes it.
9. *Gates and seconds.* Both stopping gates hold; seconds for a kept
   close, before and after.

## 8. What it changes

- Every kept close: the field at order one, word placements (rule 4),
  argument placements (rule 3).
- The chooser's and the generate walk's occupation terms (rule 5).
- Nothing in the grammar file, the arguments' seats or their names.

## 9. Decided, and what is left

- What a part of speech is: Alec's design, §5.
- Which readings write: every kept reading (Alec).
- Up and down, as an EM between words and their arguments, each order in
  its own space (Alec, §5.2 and §5.3): rules 3, 4, 4b, 4c.
- What the arguments cannot part (§11): decided.
- *Left as defaults:* which rows move in rules 3 and 4, how a centroid
  is kept (rule 4c), and the two terms of rule 5 added.

## 10. Where this round sits (decided, Alec, 2026-11-15)

"I think at this point we probably need to accept as baseline whatever
the tests report ... So I think codex can make the tool, capture what
it reports, and finish rounds B and C (including your parts of speech
work). Then we will at least have a correct codebase on which to merge
the performance work."

The order, chosen by Alec: round B finished with its results as the
baseline; round C, this specification; round D, the rewrite of
`SigmaLayer` and `PiLayer`; then the merge of the `performance` branch;
then the sentence as one block (the round drafted as round C in §12 of
`doc/plans/2026-11-14-attention-in-thinking.md`). The teacher of the
walk of attention (§9.2 of the assessment) waits until after the merge.

Round C also makes the principle that no space moves another hold
across the architecture (Alec, 2026-11-15):
`doc/plans/2026-11-15-spaces-apart.md`.

## 11. What the arguments cannot part (put to Alec)

A word's class is read from the arguments it fills, so two words that
fill the same arguments cannot come apart, whatever else differs. The
grammar has one `modifier` for what modifies a thing and what modifies
an action, so `fast` (in `the fast cat`) and `quickly` (in `ran
quickly`) both fill `modifier_argin1`, and so does `the`, a case of the
adjective since 2026-11-13. Likewise `eat` and `devour` fill the verb's
same places. They differ only in what they combine with: a noun or a
verb, an object or none.

- *Accept it for now* (Claude's recommendation). The rungs to rung 9
  need no such split, and the assessment will show where it is first
  needed.
- *A place is also its partner's class:* an argument is read together
  with the class of the other operand (`modifier_argin1` beside a noun,
  beside a verb), so that the same position with different partners
  gives different places. More places, each learned the same way.
- *Separate operators in the grammar* for what modifies a thing and what
  modifies an action. It names in the grammar what the first choice
  would learn.

**Alec's answer, 2026-11-15:** "I guess parts of speech have to make a
grammatical difference if they embody a difference, so let's assume
that operator arguments will be sufficient. If we cannot distinguish, it
is re@son to introduce a new operator." So the arguments are taken as
sufficient; where the assessment finds two classes the arguments cannot
part, the remedy is a new operator in the grammar.
