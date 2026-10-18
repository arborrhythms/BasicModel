# Parts of speech: one order-2 set per operand slot, taught one operation at a time

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

## 3. How the categorizer is learned *(decided; membership is not taught)*

- **By occupation, from kept readings only.** When a kept reading applies an
  operation, each operand's word (a phrase's head word) gains one count on
  that argument. The prototypes themselves are the formal arguments and
  need no training; collapse is reported as the cosine between arguments'
  occupant distributions and is never merged by rule. The bracket
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
receives today (§7). A phrase operand reads through its head word until
output typing exists.

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
   adjective, each as a whole sentence (NP → S, no VP), with the 4b-style
   question where one applies; a held-out quarter by the usual
   counterbalance. Menu: STOP, `lower`, `definite`, `intersection`. The
   bracket CE lesson on the student's states stays inside the stage as the
   teacher of the operator choice (it never supplies membership evidence);
   reconstruction and the questions are the keep credit, as now.
2. **NP VP stage.** The 4c sentences; menu adds `lift` (and `verb` with its
   object NP at the same stage or the next, one new operation per stage).
3. **Relations.** `preposition` and `surface` as the next operation(s).

Budgets are declared per stage before the run, as 4e's were; no stage
extends its budget to pass.

## 6. Measurement

The read-only probe of §3 has been run; the NP stage starts from its result.

Per stage, on held-out sentences of the stage's kinds:
- exact brackets for the stage's operations (the 4e grader, unchanged);
- each word's part of speech (argmax argument) with its counts, before and
  after the stage; the collapse table (cosines between arguments' occupant
  distributions); the noun-prior table (which words' concepts are tied to
  individual columns); earlier classes unchanged by the new stage;
- collapse reported as the agreement between slot sets (one table);
- the four 4e gates once all four families' stages have run, from the last
  stage's artifact, at the declared budget.

Mechanism certificates, unconditional: occupation counts move only on kept
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

**The recognizer's input** (§2–3): the probe shows the artifact's concept
codes carry no class structure, so recognition by resemblance waits on the
grounding stages; until then the categorizer classifies by occupation. Which
earlier stage is responsible for giving things and properties distinct
meaning structure, and its gate, is the next question for the curriculum.
Superseded reading: that a word's class is read from
the component families — individual columns for things, property columns
for properties, change directions for changes, no content for markers — and
that the earlier curriculum stages are what ground them. The probe of §6
measures what the stage-3 artifact already recognizes. Deferred: output
typing (§4).
