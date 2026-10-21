# Parts of speech: one order-2 set per operand slot, taught one operation at a time

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record. Record: [two truths §1, "The row, field by field"](2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided); [bind plan §6–§6.3](../plans/2026-10-19-bind-thought-operation.md); [occurrence spec §13–§14](2026-09-30-occurrence-tense-aspect.md); [expectation spec rev 5](2026-10-08-expectation-at-every-level.md).
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - Parthood, subsumption and kind statements ("cats are animals") are **folds** in the meronomy and the taxonomy over symbols, not rows; equality is membership in a singleton; no `REL_PARTOF` row exists.


> **Status:** specification, 2026-10-18, written by Claude from Alec's
> decisions in conversation on 2026-10-17 and 2026-10-18 under item 5.5's
> learning sequence (stage 4e). **Decided** is Alec's decision; *Claude's
> reading* is to be confirmed. Governed by [Orders](../Orders.md) (orders are
> fuzzy sets of symbols one order down; membership by evidence; two ladders,
> one fold) and by the 2026-10-10 decision that parts of speech form from
> participation and are the only part-of-speech information the chooser reads.
> Codex implements; Claude reviews.

## 1. Why *(decided)*

The stage 4e gate failed three times with the same shape: the compose
chooser must decide the first word pair (fold a determiner with a noun,
wait on a determiner with an adjective, intersect an adjective with a noun)
from raw word content, and a few dozen presentations do not teach that.
What makes the chooser effective is knowing the parts of speech. They are
not to be tuned into the chooser's menu for a 72-sentence stage ("I'm wary
of making tests pass in situations where the amount of data they process
is drastically less than a human corpus"); they are learned by teaching,
one at a time.

## 2. What a part of speech is *(decided, 2026-10-18)*

- **Prototypes, one per formal argument.** The parts of speech are defined
  by unit-length vectors, one for every formal input and output argument
  over all operators (every `argin` and `argout` of every compose
  operation). Some collapse: there are fewer actual parts of speech than
  arguments, and collapse is two prototypes coinciding; no merge rule.
- **A categorizer maps a word to its most likely part of speech.** Its
  input is distributional (§2a): the word's sparse count of occupations over
  the formal arguments in kept readings, normalized; its output is a
  distribution over the prototypes, and the word stores the argmax as its
  single part of speech, recomputed whenever its counts change. We know what
  a noun is in virtue of recognizing nouns by where they occur, not by
  accumulating a conceptual mapping. *Alec:* this contradicts the stored
  sparse-set form stated earlier, and has a much smaller memory footprint.
  Confirmed 2026-10-18: "Ok, let's try this."
- **The prototypes are orders in the symbol ladder** (order-2 kinds over
  words), not rows in the truth store.

## 2a. Linguistic types are distributional *(Alec's worry, 2026-10-18; Claude's resolution to confirm)*

Alec: object categories can be divided cleanly into classes, because objects
have a structural presence in conceptual space; words are arbitrary, so a
common mapping over word-concepts (as opposed to object-concepts) will be
exceedingly difficult to find, even though "parts of speech" are linguistic
types. *Resolution:* this is Orders.md's two ladders. A part of speech is a
type on the forms ladder and is defined distributionally: the prototypes are
the formal arguments themselves, the basis of the participation space, and a
word's part of speech is the single argument class it most occupies in kept
readings; the evidence that computes it is the word's short sparse count of
occupations over the arguments, not its concept. Conceptual resemblance
enters only where it is structural, through the referents ladder: a word
whose concept is tied to an individual column names a thing, which is a
prior for the noun class and nothing else. All other classes arise by
occupation and, under one operation at a time, by elimination. Nothing tries
to find a common conceptual mapping over all words.

## 2b. Which results carry their inputs' words *(Alec, 2026-10-18: the result of an intersection or union is the same as its input types; Claude's table from the catalogue, to confirm)*

Occupation flows from a result to the arguments it later fills by this rule,
read off the catalogue (`GRAMMAR_LAYER_CLASSES`: arity, declared head,
order change):

| Operators | Result's type | Words the result carries |
|---|---|---|
| `intersection`, `union`, `conjunction`, `disjunction`, `product`, `sum` (symmetric, no head) | the inputs' type | all operands' words, each counted *(product and sum added 2026-10-18: a multiplicative or additive modification is symmetric and keeps its inputs' type)* |
| `not`, `non` (unary modifiers) | the operand's type | the operand's words |
| `adverb` (modifier + head), `preposition`, `surface`, `bind`, `compound` (declared head) | the head's type | the head's words (`adverb` must declare its head: it has none today) |
| `lower`, `definite` and their unary faces (order −1), `verb` (clause form VP) | a new type | the head operand's words: the bare noun, or both words of an intersected description; the verb for a VP. An NP carries those words into `lift` |
| `lift`, `equal`, `part`, `whole`, `implies`, `generic`, `ask` | a clause or relation | nothing (a clause has no word occupant) |

Every operator keeps its own output argument (the formal set stays
exhaustive, as decided); the identity of a type-preserving output with its
inputs is not assumed but enforced only in how words flow, so the collapse
table measures the claim (`intersection_argout1` ≡ its inputs).

Selecting a head operand does not select a single lexical word within it.
Thus both `black` and `cat` flow through `lower(a, intersection(black, cat))`
and into `lift`, after which neither is carried. Their chooser probabilities
add, one contribution per carried occurrence. Operations outside this table
with no declared head retain the existing empty carrier; this pass makes no
new type claim for those operations. *Decided 2026-10-19:* `product` and
`sum` are symmetric carriers like `intersection` and `union` (table above);
their results keep the inputs' type and carry every operand's words. Menu
admission alone does not declare an output type; the table row does. The
code's flow mask must list them before another adjective lesson.

*Linguistic note (Claude).* Coordination preserves the conjuncts' type
(Williams 1981's Law of Coordination of Likes); the counterexamples ("Pat is
a Republican and proud of it", "slowly and with great care", Sag, Gazdar,
Wasow & Weisler 1985; Bayer 1996) are unlike categories that satisfy the
same selector, i.e. the same formal argument of the selecting operator,
which is exactly what argument-level classes provide. Tense does not change
a conjunct's part of speech: in converb and clause-chaining languages
(Japanese -te, Korean -ko, Turkish -ip, Hindi -kar, Swahili narrative -ka-,
Amharic; Haspelmath 1995, 2007) tense or finiteness is marked once for the
chain, and in English, Latin or Russian each conjunct is marked but the
marks must agree; either way tense is an operator on the coordinated verb
phrase, the verb phrase's preposition of §4–5 applied once, and the
conjuncts stay one part of speech. Where languages use different
conjunction words by type (Japanese *to* for nouns, *-te/-shi* for verbs;
Chinese *hé* versus *érqiě*; Swahili *na* versus sequence marking), the
difference is in the marker word, learned as word-to-operator exposure, not
in the operator's result type.

## 3. How the categorizer is learned *(decided; membership is not taught)*

- **By occupation, from kept readings only.** When a kept reading applies an
  operation, each word an operand carries (§2b: a word itself; for a phrase,
  the words its producing operation passes on) gains one count on that
  argument, and the result carries words onward by the same table. The
  prototypes themselves are the formal arguments and need no training;
  collapse is reported as the cosine between arguments' occupant
  distributions and is never merged by rule. The bracket
  annotation's operands never contribute. No labels, no clustering, no
  learned-category codebook. The noun prior of §2a (a word whose concept is
  tied to an individual column) is recorded as a table in this pass, not
  yet read by the chooser.
- **Measured starting point (Claude, 2026-10-18, read-only probe of the
  stage-3 artifact).** The existing concept codes of the 4e words carry no
  class structure: on the meaning band, mean cosine within nouns .06, within
  adjectives .11, between nouns and adjectives .06, all at the .2 meaning
  norm; on the form band every pair is about .35–.42 (letters); leave-one-out
  nearest-centroid classification of the ten content words with codes is
  5/10, chance. Seven words (`chases`, `sees`, `a`, `the`, `is`, `on`, `near`)
  have no codebook row. So, as the artifact stands, a categorizer over the
  conceptual representation can only memorize the arguments the training
  words have occupied; recognition of a *new* noun by resemblance needs the
  grounding stages (identity and permanence; words with their objects) to
  give things and properties distinct meaning structure, which this artifact
  does not have. The NP stage therefore measures two things apart: the
  categorizer's assignment of the seen words (by occupation) and its
  generalization to held-out words (by resemblance), the second recorded
  without being gated until the grounding stages provide structure.
- **One operation or one part of speech at a time.** The order is arbitrary;
  what is not arbitrary is that a stage introduces one new operation (or one
  new class). First NPs, over many NPs as whole sentences (NP → S, with no
  VP). Then NP VP: the grammar already knows the NP words' classes, learns
  the next operation (the one that corresponds to verbs), and the new words'
  class follows from occupation and, where the categorizer is silent, by
  elimination.

## 4. How the chooser reads the classes *(decided in direction; Claude's reading of the form)*

The chooser's decision to apply an operation to the two newest operands
reads, for each input argument, the categorizer's probability that the
operand occupies it (its normalized occupation count on that argument, with
a floor for an unseen word). *Claude's reading:* that term enters the
operation's score directly, replacing the category vector the chooser
receives today (§7). A phrase operand reads through the words it
carries (§2b) until output typing exists.

**Output prototypes** (the `argout` vectors) classify results, so that
`lift`'s first input converges with the determiner operators' output; this
should happen eventually and is not in the first pass. Until then the output
prototypes are updated by occupation but not read.

## 5. Curriculum *(decided in form; sizes Claude's decision)*

Each stage's menu is the operations taught so far plus one new one. Each
stage's receipt names the artifact it produces, and the next stage's
certificates restore it (artifact → fixture).

1. **NP stage.** A generated corpus over the 4e vocabulary: every
   determiner, adjective and noun combination, with and without the
   adjective, each as a whole sentence (NP → S, no VP), plus the bare
   adjective-noun descriptions as unlowered controls, with the 4b-style
   question where one applies; a held-out quarter by the usual
   counterbalance (committed as `data/noun_phrases`). Menu: STOP, `lower`,
   `definite`, `intersection`, `product`. The stage is a sequence of prerequisite
   lessons, one unknown each, each gated on its own held-out items before
   the next restores its artifact: `lower` on bare "a" NPs, `definite` on
   bare "the" NPs, `description` (modification on bare
   adjective-noun pairs), `adjective` ("a" + adjective + noun, where
   modification is scheduled inside an NP), `definite adjective`. After
   the last lesson the stage's gates run once on the whole held-out quarter
   from the last artifact, which is the NP VP stage's fixture. The bracket
   CE lesson on the student's states stays inside each lesson as the
   teacher of the operator choice (it never supplies occupation evidence).
   Adjective-noun annotations name a `modification` bracket; its fold target
   sums the admissible `intersection` and `product` actions, including their
   reference variants. The kept operator is reported under §5a;
   reconstruction and the questions are the keep credit, as now.
2. **NP VP stage.** The 4c sentences; menu adds `lift` (and `verb` with its
   object NP at the same stage or the next, one new operation per stage).
3. **Relations.** `preposition` and `surface` as the next operation(s).

Budgets are declared per stage before the run, as 4e's were; no stage
extends its budget to pass.

## 5a. Gates are behaviors *(decided, Alec, 2026-10-18)*

"Whatever does the job, great; in writing a UG, we do not demand which
operators must be chosen, only which behaviors must be exhibited." No lesson
gate names an operator. A lesson's gates are the behaviors its sentences
exhibit: the held-out constituents (brackets), the answers to its questions,
one individual per singular NP (one order-1 occurrence, no duplicates or
collisions), and the stamps and the kept operators are reported, never
required. The bare-description control (no occurrence for "small cat") is a
report, not a gate; whether a bare description licenses a later reference is
a behavior to be measured by a later sentence, not demanded of a reading.

## 6. Measurement

The read-only probe of §3 has been run; the NP stage starts from its result.

Per stage, on held-out sentences of the stage's kinds:
- exact brackets for the stage's operations (the 4e grader, unchanged);
- each word's part of speech (argmax argument) with its counts, before and
  after the stage; the collapse table (cosines between arguments' occupant
  distributions); the noun-prior table (which words' concepts are tied to
  individual columns); earlier classes unchanged by the new stage;
- collapse reported as the agreement between slot sets (one table);
- the §5a behavior gates once the NP prerequisite sequence has run, from
  the last artifact on the whole held-out quarter at the declared budget;
  kept operators, lowering stamps and bare-description occurrences reported.

Mechanism certificates (artifact restoration is phase-gated): occupation counts move only on kept
readings, never on the annotation; a word's single part of speech is the
argmax of its counts and is recomputed when they change; the chooser's score
reads the categorizer's probability for each input argument; the stage menu
holds only the declared operations; artifact restoration is exact.

## 7. What is replaced

The chooser today receives a per-word category vector from a category
codebook (`MetaSymbolCategoryLearner`, `categoryCodebook` in
`BasicModel.xml`): a fixed number of categories learned by vector
quantization of role participation, with per-operator category priors. The
slot sets replace that source; the codebook, its context and its priors are
retired when the sets pass the certificates above (no legacy paths). The
reference-expanded menu stays as it is, measured by the first-pair logit
diagnostic; it is revisited only if the taught classes do not overcome its
prior.

## 8. Open

Resolved 2026-10-18: the categorizer is distributional (§2a) and results
carry words by §2b. Deferred: output typing (§4).

**Grounding is textual, not perceptual** *(decided, Alec, 2026-10-19)*. The
model sees percepts of words (letter parts and wholes) and forms
word-concepts from them. A word-concept is a zeroth-order concept that
references an object-concept of any order ("Bill" an individual, "water" a
mass, "chair" a count kind); one concept may reference another through the
LTM's reference structures (the definition rows and the meta references).
Since the model experiences only text, it learns a word by minting a random
object-concept, and learns about that object explicitly: a definition ("a
table has a top and four legs") adds parts to the conceptual mereology, a
kind statement ("tables are furniture") adds to the conceptual hierarchy.
The difference between mereology and hierarchy is whether the item collects
symbols, which makes it abstract. So there is no perceptual grounding stage:
the §3 probe's empty intersections are the expected state of freshly minted
object-concepts, and the resemblance a categorizer could bootstrap from
arrives through hierarchy statements, not percepts. Lessons demand
behaviors (§5a), never non-empty meanings.
Superseded reading, kept for the record: that a word's class is read from
the component families — individual columns for things, property columns
for properties, change directions for changes, no content for markers — and
that the earlier curriculum stages are what ground them. The probe of §6
measures what the stage-3 artifact already recognizes. Deferred: output
typing (§4).
