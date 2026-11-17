# The assessment: what a child has learned, read from its own trees (Claude, 2026-11-15)

A specification of a teaching tool and of the simple cases it tests
first. It is not scheduled and has no text for Codex. A rule marked
*default* is Claude's choice where Alec has not spoken.

## 1. Alec's words (2026-11-15)

"If we are failing many tests, we should ensure that we are learning
the most simple cases first.

Some testing tool that looks at a text dump of the STM tree (parse
tree) for a sentence would probably help us to know what words are
being found, what operators are being applied and where. As I
mentioned before we need to gate complex operators on the success of
simple operators. so a simple gate for s set example corpus (learn a
word as a concept (object permanence), learn 20 words, learn "Felix",
"Felix ran", the cat ran, the fast cat ran, he ran, felix ran quickly,
the fast cat ran quickly) would be a probe that we could apply across
all tests (that should have learned those operators and words), what
they actually learned."

"Lets call that teaching tool "the assessment", which will test the
child first on the known probe, and possibly an unknown probe (a new
word will be known by the parts of speech on either side and
permissibility of the grammar)."

## 2. What exists

- *Ten lessons* (`data/lessons/recipes/00-words.json` to
  `09-relations.json`, `bin/lesson_stages.py`), each with a criterion
  read by `bin/eval_lessons.py`: the share of words whose definition
  is trusted, the definiteness of an operator's occupant classes, and
  answers. None reads which words reached a reading or which operator
  was applied where.
- *The kept reading's journal.* `reading_forest`
  (`test/scalar_question_curriculum.py`, built for §38.3 of the 1.8
  specification) turns a kept program's actions into a nested list:
  operators at the nodes, source places at the leaves.
- *A lesson run.* `corpus_lessons.measure` runs the lesson corpus
  `data/corpus-lessons.json` (22 sentences such as `This is Felix.`
  and `What is this?`, four epochs, 160 presentations).

Nothing prints a tree, and no test is gated on a simpler one.

## 3. A first dump, and what it found

A read-only prototype in the session's scratch directory (nothing in
the tree was changed, and it is not the tool of §4): the lesson above,
on the configuration `explore-1.xml`, with every kept sentence's tree
written out.

| Sentence | Words that are leaves of the kept reading | Kept tree, first epoch | Kept tree, last epoch |
|---|---|---|---|
| `This is Felix.` | `is`, `.` | `marker(multiply(bind.open x3(is), .))` | `non(verb(non(is), .))` |
| `What is this?` | `is`, `this`, `?` | `ask(multiply(bind.open x2(multiply(bind x3(is), this)), ?))` | `non(verb(non(implies(ask(is), this)), ?))` |
| `That is Ciao.` | `is`, `.` | `non(marker(surface(bind.open x3(is), .)))` | `non(verb(marker(is), .))` |
| `Who is here?` | `is`, `here`, `?` | `non(surface(bind.open x2(verb(bind.open x3(is), here)), ?))` | a run of eighteen `bind.open` over `verb(...)` |

Over the 160 kept sentences:

- *Fewer than half the words reach the grammar:* 248 of 544 word
  places are leaves. The first word of a sentence is a leaf in 0 of
  160. The last word of a statement, which is the name being taught
  in `This is Felix.`, is a leaf in 35 of 80.
- *It is attention that leaves them out.* A word is a leaf of the
  grammar only where the walk of attention accepted it
  (`ModelAttention.handoff`: the leaf mask is `reading.accepted`). In
  `This is Felix.` the walk takes 13 of its 32 acts and stops. It
  mints `is` and the full stop. On `This` and on `Felix` it takes
  field acts (union, complement, intersection) and never mints them.
- *This is not new with item 1.8.* The same lesson on the landing
  source `3926005f12` gives the same accepted places for the same
  sentences, 55 in a hundred of the word places against 59 now.
- *One-word operations run on.* `bind` is applied 738 times and
  `bind.open` 478 times in 160 trees, against 175 `verb`. A `bind` or
  a `bind.open` is repeated ten or more times on one operand in 57
  trees.

So a lesson whose subject is `Felix` is read without `Felix`. This is
what the criteria of §2 could not show, and it is the first thing the
assessment is for. Where and when the walk began to leave words out
was not traced further back than the landing source.

## 4. The tool

1. *One function reads one sentence and returns its record,* in
   evaluation mode and with no gradient, through the ordinary
   `understand`. *Default:* it lives in `bin/` beside
   `eval_lessons.py`, built on `reading_forest`, which moves there.
2. *The record of a sentence holds:*
   - each word place: its form, whether the word is known, its word
     row and the objects it denotes, and what attention did with it
     (minted, entered by its parts, or left);
   - the kept reading as a tree, with the operator's name at each
     node and the word at each leaf, a repeated one-word operation
     written once with its count, and an open reference marked;
   - what was kept: the meaning's references, and the answer and its
     evidence kind where one was asked.
3. *As text* a record is a few lines, for example:

   ```
   "This is Felix."
     words   This[known, left]  is[known, minted]  Felix[known, left]  .[minted]
     tree    non(verb(non(is), .))
     missing This, Felix
   ```
4. *It reads; it does not teach.* It writes no memory, moves no clock
   and draws no random number that the run would otherwise draw
   (*default:* it restores the model's state as an evaluation read
   does today, and a test asserts it).
5. *A probe is a list of sentences with what each must show:* every
   word a leaf, and the tree. A probe's result is a table, one row
   for a sentence: passed or not, and the record where not.

## 5. The known probe: the simple cases, in order

A rung is assessed only where the rungs below it have passed.
*Default:* the trees are written with the grammar file's own operator
names when the probe is built, and are reviewed before use; they are
not given here.

| Rung | What is presented | What must show |
|---:|---|---|
| 1 | one word, alone, several times | it is minted each time; it is the same concept each time; its definition is trusted (object permanence) |
| 2 | twenty words | each as rung 1; no two share a concept; each is said back |
| 3 | `Felix` | the word denotes one object; the reading is that one leaf |
| 4 | `Felix ran` | both words are leaves; one fold; `Felix` in the verb's subject place |
| 5 | `the cat ran` | all three are leaves; the determiner on `cat`, then the verb |
| 6 | `the fast cat ran` | the modifier on `cat`, inside the determiner |
| 7 | `he ran`, after `Felix ran` | `he` is bound to the earlier referent |
| 8 | `Felix ran quickly` | the modifier on the verb |
| 9 | `the fast cat ran quickly` | rungs 6 and 8 together |

Every rung also requires that no one-word operation is repeated on an
operand. Where a rung ends a stage of a lesson, the reading is also the
same in three rounds in a row.

*Gating.* A test of an operator names the rung it stands on. It first
assesses that rung's probe on its own fixture; where the rung fails,
the test reports "not ready" with the record, and its own assertion
is not counted as a failure of the operator. A lesson's stage is not
begun until the rung below it passes.

*Applied across the tests.* The same probe is read on any fixture or
checkpoint that claims those words and operators, and the table says
what it learned. *Default:* a rung is a frame with places for a kind
of word (a name, a noun, a verb, a modifier of each, a pronoun), as
the lessons' recipes are, and each fixture fills them from its own
words; a rung whose kind of word a fixture lacks is "not applicable"
there. The ladder's own lesson fills them from a vocabulary of its
own.

## 6. The unknown probe

Alec: "a new word will be known by the parts of speech on either side
and permissibility of the grammar".

A word never seen is put where a known word stood in a sentence of the
known probe: a new word for `cat` in rung 5, for `ran` in rung 4, for
`fast` in rung 6, for `quickly` in rung 8.

1. *The first reading.* With the new word in it, the kept tree is the
   rung's tree, and the new word is the leaf where the known word was.
2. *One presentation.* After the sentence is kept once, the new word
   fits the arguments the known word fits better than it fits the
   others (`PartsOfSpeech.similarities`).
3. *The class and not the one place.* The new word is then read
   rightly in the rung's other sentences, in places it has not been
   seen in: after `the wug ran`, in `the fast wug ran`.
4. A new word is assessed only where the known probe passes at that
   rung.

Point 3 is what makes vocabulary quick to learn, and §8 says why it
is not expected to pass as the classes are learned today.

## 7. Open

- Whether the walk of attention is taught to mint every word first
  (rung 1 as its own lesson), or is wired to seek the next word until
  it does (Alec, of an earlier case: "Id prefer a small cheat to
  "seek for the next word" in the current attention mechanism until
  we have a better test for it"). The assessment is that better test.
- Where the assessment sits beside item 1.8: decided, it comes first
  (§9).
- The rungs' sentences: §5, *default*.

## 8. Parts of speech: what is learned today, and whether it is enough

Alec: "we talked about the collapse of some formal argument parameters
into common classes (what is operated on by a verb and an adjective
are the same thing). But we also have that some classes need to be
split (we might need a difference in part of speech between pronouns
and nouns that verbs do not distinguish). Hopefully that difference
can be exposed with operators like bind(), but can you please think
about the languages that you know, and take a look at the code, and
let me know if the way we are learning parts of speech (i.e. as
concepts over words and the argument specification of operators) is
being learned sufficiently such that it will help us learn new
vocabulary rapidly once we get to five years old by assessment?"

### 8.1 What the code does (a survey by a sub-agent; three statements checked by Claude)

- Each formal argument of each operator is a concept of its own, one
  order above words (`PartsOfSpeech.formal_arguments`: `verb_argin1`,
  `modifier_argin2` and so on; about seventy for `complete.grammar`).
- A word's class is two things: it is recorded as a member of each
  argument it has occupied, and it has a *fit* to every argument,
  which is the likeness of the word's meaning direction to the
  argument's direction, scaled by the argument's trust
  (`ConceptHierarchy.fit`, `PartsOfSpeech.similarities`).
- A membership is written from the kept reading only, at
  `_commit_sentence` (`PartsOfSpeech.observe_kept`), one view for each
  word in each argument, with no bar on the reading's worth.
- The fit is added to the compose chooser's scores (`bin/Language.py`,
  the occupation terms). It is a bias and not a rule of what is legal.
- Nothing merges two arguments' classes (`collapse` reports their
  cosines and is never called), and nothing splits one.
- A word never seen fits every argument at zero. Nothing reads its
  class from its neighbours or from which readings the grammar
  allows.
- No ending, prefix or capital is used for a class.
- The lessons' bars on an argument's definiteness are all undeclared.
- Since round A the determiner operators (`definite`, `universal`,
  `some`) are commented out of `complete.grammar`.

### 8.2 What languages ask of a class

A class is a bundle of places: the words that may stand in one place
of the bundle may stand in the others. What marks it differs by
language and is nowhere the meaning:

- *the neighbours.* In English `the _ ran` makes a noun and
  `Felix _ quickly` a verb; a child of four inflects a word it has
  heard once (`a wug`, `two wugs`).
- *the form.* `quick` and `quickly` mean the same and stand in
  different places; in Turkish, Finnish or Russian the ending carries
  the class and the role.
- *agreement with a particular function word.* German gender and
  Mandarin classifiers split the nouns by which article or classifier
  they take, and the split follows no meaning.

Cases of the two kinds Alec names:

- *A merge:* what a verb takes as its subject, what a verb takes as
  its object and what follows a preposition are one class.
- *A split a verb does not make:* `Felix`, `he` and `cat` can each be
  a subject, but only `cat` takes `the` and `fast`, only `he` changes
  to `him` as an object and needs an earlier referent, and only
  `Felix` names one thing. So the split is made by the places of
  other operators, the modifier's, the determiner's and `bind`'s, as
  Alec expects.
- *Same meaning, different class:* `eat` and `devour` (one may stand
  without an object), `quick` and `quickly`.
- *Different meaning, same class:* `cat`, `idea` and `Tuesday`.

### 8.3 Whether it is enough

Not yet, for four reasons, the first of which is not about the design.

1. *What is written today is not evidence.* Memberships are written
   from whatever reading is kept. In the lesson the kept readings
   leave out half the words and apply `bind` 738 times in 160
   sentences (§3), so the first word of a sentence and most names
   have no class at all, and `bind`'s argument holds whatever was on
   the stack. The classes cannot be judged until rungs 1 to 4 pass,
   and no bar is declared that would have shown it.
2. *The fit is by meaning, and a class is by place.* It serves the
   first words, where things are nouns and actions are verbs. It
   cannot make the split of `quick` from `quickly`, nor hold `idea`
   with `cat`, since an argument whose members have unlike meanings
   has no direction for a new word to be like.
3. *One place does not give the bundle.* Seventy arguments are
   seventy classes. A new word kept once in `the wug ran` is a member
   of the places it stood in and fits no other place, unless its
   meaning has moved. The step from one place to the class is what a
   five-year-old has and what §6, point 3, tests.
4. *A new word's class is not read from the sentence.* It may come
   out so in part, since the known words' fits pull them to their
   places and leave the new word the place that remains; nothing
   ensures it, and it is what §6, point 1, tests.

**What Claude suggests** (not decided): that the class be read from
the memberships themselves. Two arguments whose members are the same
words are one class, which is the merge; two sets of words that share
one argument and differ in the others are two classes, which is the
split, and `bind`'s and the modifier's arguments are what divide
`he`, `Felix` and `cat`. A word's fit to an argument is then how far
the words it shares places with stand there, and a new word seen once
in one place takes the rest of that class's places with it. Meaning
stays as the first evidence, before there are places to count. Form
and agreement are later rungs.

## 9. Decided: the assessment first (Alec, 2026-11-15), and the text for Codex

"Yes, assessment first". Round B of item 1.8 pauses at §39 with its
source as it is; it continues once the rungs its certificates stand on
pass.

The first round builds the tool and the known probe, reads what the
present fixtures and the lesson have learned, and finds why attention
leaves words out. It repairs nothing in attention: whether reading
words in turn is taught or wired for now is with Alec.

~~~
Item 1.8, round B, pauses at §39 of
doc/plans/2026-11-11-item1-8-last-architecture-round.md, on Alec's
word: "Yes, assessment first". Build the assessment of
doc/plans/2026-11-15-assessment.md (read all of it first, §3 and §5
above all). Work on the present working tree. Commit and push nothing
until Alec says so; never push main, and never force-push. Leave
bin/Attention.py, bin/ModelAttention.py, bin/SentenceAttention.py and
bin/AttentionTraversal.py as they are; the assessment reads attention
and changes nothing in it. Round D is not yours to build yet.

1. The tool (§4). bin/Assessment.py: one function reads one sentence
   through the ordinary understand, in evaluation mode with no
   gradient, and returns its record: each word place (form, known or
   not, word row, its objects, and what the walk of attention did
   with it: minted, entered, or left, from model._attention_words),
   the kept reading as a tree with the operator at each node and the
   word at each leaf, a repeated one-word operation written once with
   its count, an open reference marked, and what was kept. Move
   reading_forest from test/scalar_question_curriculum.py into it. A
   record prints as the few lines of §4, point 3. A test shows that
   reading a probe leaves the model as it was: memory, clocks, random
   streams, parameters.
2. The known probe (§5). data/assessment/known_probe.json holds the
   nine rungs as frames with places for kinds of words, as the
   lessons' recipes are (bin/game_corpus.py render), and what each
   rung must show. Write each rung's expected tree with the operators
   of data/complete.grammar as it is now, and mark it proposed: it is
   reviewed by Claude and Alec in your return before it gates
   anything. Where the grammar has no operator for a rung (the
   determiner operators are commented out since §27.1), say so; do
   not add one.
3. What is learned now. Read the probe, filled from each fixture's
   own words, on: the lesson of data/corpus-lessons.json on
   doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml
   after its four epochs; the fixtures of round B's certificates
   (test/test_item18_*.py) after their own teaching; and the scalar
   fixtures of certificate 90 after their curriculum stage. Return one
   table per source: rung, passed or not, and the record where not.
4. Why attention leaves words. In the lesson, the first word of a
   sentence never reaches the grammar and a statement's last word in
   35 of 80 (§3). Find the producer: which acts the walk takes on
   those words, why it ends with budget left, and whether it is the
   walk's chooser or its legality (Attention.narrowing_mask). Report
   it with a trace. Repair nothing in attention; stop there and write
   for Alec and Claude whether reading each word is to be taught as
   the first rung or wired for now.
5. Gating. Each test of an operator in round B's certificates names
   the rung it stands on (a mark the test runner reads). Add the
   check of §5 that reports "not ready" with the record where that
   rung fails; do not yet turn any failure into "not ready" in the
   sweep, and change no assertion.

Return as §10 of doc/plans/2026-11-15-assessment.md: the tool and its
test; the probe with its proposed trees; the tables of point 3; the
producer of point 4 with its trace; the marks of point 5; the source
snapshot; and every edit, if any, in the five attention places of
§31.3 of the 1.8 specification (there should be none).
~~~

### 9.1 Decided: reading each word is taught first (Alec, 2026-11-15)

"Teach it first". Reading every word of a sentence, in order, is the
first rung's own lesson, with the assessment as its measure. It is
not wired. Alec had said the same of round C on 2026-11-14: "We should
teach reading words in order as a preliminary to reading".

**What teaches attention today** (read in the source by Claude). There
are two walks:

- *The walk that decides which words reach the grammar*
  (`Attention.narrow_words`, called from `ModelAttention`). Its chooser
  (`attend` in `bin/Language.py`) takes a forced or replayed act but no
  target, and nothing teaches it. This is the walk that leaves `This`
  and `Felix` out (§3).
- *The order in which accepted words are read* (`SentenceField`, from
  `AttentionTraversal.FieldTraversal`). It has a teacher:
  `AttentionLesson.reading_lesson` names the words in order, and the
  read is taught by cross entropy (`lesson_cross_entropy`). The stages
  `reading_two` and `reading_four` of
  `test/attention_curriculum_gates.py` use it, and the constituent
  experiment requires that stage before it runs.

So the earlier reading curriculum taught the order of the words that
were accepted, and never which words are accepted. Teaching the first
walk is new work.

**What it touches.** The first walk's teacher goes in
`bin/Attention.py` and `bin/ModelAttention.py`, which §31.3 of the 1.8
specification left untouched because the `performance` branch rewrote
the same walk as steps (`narrow_step`). Where it is built is put to
Alec. Point 4 of §9's text (the producer, with no repair) stands
either way, and the teacher's design follows from what it finds: if
the walk's chooser declines a legal mint, a target teaches it; if the
mint is not legal there (`Attention.narrowing_mask`), that is a design
question and comes back first.

### 9.2 Decided: the first walk's teacher is built here, on main (Alec, 2026-11-15)

"Here, on main". §31.3 of the 1.8 specification is lifted for
`bin/Attention.py` and `bin/ModelAttention.py`, for this teacher only.
Every edit there is listed for the merge of the `performance` branch,
which rewrote `narrow_words` as steps. `bin/SentenceAttention.py`,
`bin/AttentionTraversal.py` and `Models._run_sentence_word_bricks`
stay as they are unless the teacher cannot be built without them, and
any edit there is listed too.

**The teacher** (Claude's rules; *defaults* marked).

1. *The same lesson names the words for both walks.*
   `AttentionLesson.reading_lesson` already names a sentence's words in
   order for the second walk. While it is set and the model trains, it
   also gives the first walk its target.
2. *The target of a round* is the legal act (by
   `Attention.narrowing_mask`) that reaches the first word of the
   sentence not yet minted or entered, by the fewest acts, and then
   mints it, or enters its parts where the word is not known. No field
   act is a target. A test shows that following the targets reaches
   every word, in order, in the fewest acts, on sentences of one to
   six words.
3. *As the second walk is taught.* The target act is the one taken
   (as `FieldTraversal.select(taught=)` takes it), and its cost is the
   cross entropy of the chooser's probability of it, booked as that
   walk's `lesson_cross_entropy` is. The choice's own estimator
   (alternatives, replay, the draw) is not used on a taught round.
4. *The lesson* (*default*). The sentences of rungs 3 to 9, filled from
   the ladder's own vocabulary, every word named in order, on the
   lesson's configuration
   (`doc/benchmarks/2026-11-11-whole-paths/configurations/explore-1.xml`),
   at its configured learning rate. A round presents each sentence
   once.
5. *It ends on its own criterion* (*default*, as §38.3 of the 1.8
   specification). After each round the assessment reads every
   sentence with no lesson and no gradient. The round is good when
   every word place of every sentence is minted or entered and none is
   left. The stage ends at the third good round in a row; its bound is
   200 rounds, and is not changed to obtain a pass. Held out: the same
   frames filled with words the lesson did not use, read at the end,
   reported and not gated.
6. *Before it is measured* the lesson's sentences, its vocabulary and
   its held-out words are written in `data/assessment/` and are not
   changed after the first measurement.

### 9.3 Amendment to §9's text (for Alec to paste)

~~~
Amendment to the assessment round (§9 of
doc/plans/2026-11-15-assessment.md). Alec, 2026-11-15: "Teach it
first" and "Here, on main". Read §9.1 and §9.2 first.

Point 4 continues past its diagnosis. If the producer is the walk's
chooser declining a mint that narrowing_mask allows, build the first
walk's teacher of §9.2 in bin/Attention.py and bin/ModelAttention.py,
which you may now edit for this teacher only. List every edit there
with its reason, for the merge of the performance branch.
bin/SentenceAttention.py, bin/AttentionTraversal.py and
Models._run_sentence_word_bricks stay as they are unless the teacher
cannot be built without them; list any edit there too.

If instead the mint is not legal where the word is left
(narrowing_mask), build nothing in attention: stop and write that for
Alec and Claude, with your recommendation.

Then run the lesson of §9.2, points 4 to 6, exactly as written, and
read the known probe again on the lesson's configuration after it.
Do not change the bound, the learning rate, a seed or a capacity to
obtain a pass; if the bound is reached, report each round's record
and stop.

Return as §10, as §9's text asks, with also: the labeler and its test;
the teacher and a test that fails before it; the stage's rounds and
the round at which it ended or its failure; the held-out reading; and
the known probe's table before and after.
~~~

### 9.4 The teacher waits until after the merge (Alec, 2026-11-15)

Asked whether the first walk's teacher is built now or after the merge
of the `performance` branch, Alec chose "Defer until after merge". §9.2
and §9.3 wait, and §31.3 of the 1.8 specification stands as it was.
This round builds the tool, reads what is learned now, diagnoses why
attention leaves words (point 4, with no repair), and its tables are
part of the baseline (§43 of the 1.8 specification).


## 10. Codex return: assessment without the first-walk teacher (2026-11-15; complete for review)

The assessment is complete under §9.4 and item 1.8 §43.1. The finished-source
records and all 34 source tables are in §10.12–15; earlier subsections retain
the intermediate findings and superseded questions. The first-walk teacher
remains deferred. Alec's §11 decision keeps mint purity unchanged. The
read-only record now identifies the word or object concepts supplying each
native lane. No attention code is changed, and no commit or push is made.

### 10.1 The read and its restoration

`Assessment.read_sentence` calls the ordinary `understand` in evaluation mode,
without a gradient. Its input is text, optionally with a scalar and earlier
sentences; it supplies no reading, reference or answer. The kept admission's
program supplies the tree. `reading_forest` moved from
`scalar_question_curriculum.py`; that helper imports the same function.
The record includes the pre-read lexical lookup, word row, denotations,
effective definition evidence, minted/entered/left status, kept references
and openings, and, when requested, the ordinary answer and its evidence kind.
Consecutive identical unary operators print once with their count.

The probe is an in-process transaction over model-owned objects, tensors and
containers. It restores original identities, not a newly loaded checkpoint:
loading cannot restore a live MTM frame or an existing autograd graph.
Python, NumPy, CPU torch and initialized accelerator random streams, module
training modes and model-owned generators are restored too. The independent
test compares parameters, state dict, structural extras, corpus continuation,
clock and random streams. A second test raises after an ordinary read and
checks the same restoration. It also checks identical successive reports.
No model transaction is kept and no teacher is added.

The first focused run exposed two tool/test errors: a nonexistent
`ConceptualMeaning.open_roles` read, then the independent comparison helper's
use of dataclass equality on tensor fields. The tool now reads
`ThoughtReferences.open_slots`; the helper compares each dataclass field.
The restoration and curriculum tests then passed, six in 8.84 seconds.
With the proposed-ladder checks and native fixture reports, the focused run
passed 25 tests in 23.12 seconds.

### 10.2 Proposed known trees, not gates

`data/assessment/known_probe.json` declares the nine frames and marks the
whole specification proposed. `AssessmentProbe` renders frames through
`game_corpus.render`. A caller declares its own word kinds; the code does not
infer a kind from a spelling. A lower rung that is unavailable or fails
blocks a later available rung. Missing fixture kinds and grammar operators
are reported separately as not applicable. These statuses do not change a
test's outcome.

Leaf numbers below are source-word positions. These are proposals for review,
including the orientation of the unavailable determiner; no determiner was
added. `modifier` projects its left head, so the prefix noun modifier has
operands in the order noun, modifier.

| Rung | Proposed tree | Availability |
|---:|---|---|
| 1 | `0` | one declared word; three restored reads |
| 2 | `0` for each word | twenty distinct declared words; each as rung 1 and said back |
| 3 | `0` | a declared name, with exactly one denotation |
| 4 | `verb(0,1)` | name and verb |
| 5 | `verb(definite(0,1),2)` | unavailable: §27.1 commented out `definite` |
| 6 | `verb(definite(0,modifier(2,1)),3)` | unavailable for the same reason |
| 7 | `verb(bind(0),1)` after the rung-4 context | pronoun and verb, bound to an earlier reference |
| 8 | `verb(0,modifier(1,2))` | name, verb and verb modifier |
| 9 | `verb(definite(0,modifier(2,1)),modifier(3,4))` | unavailable for the same reason |

The three reads for "several times" are a stated tool choice; they teach
nothing. Every examined rung checks its leaves and proposed tree and rejects
repeated unary operators. The unavailable determiner prevents later rungs
from claiming readiness under the literal all-lower-rungs rule. This is a
limit of this proposed ladder, not a reason to add an operator in this round.

### 10.3 Marks and fixture observations

The runner reads `assessment_rung(number)`. Round-B stack, expectation,
intuition, word trust, lexical form and placement tests declare rung 1;
marker tests declare rung 3; native recognition declares rung 4. The report
is attached without suppressing an assertion, skip or failure. Fixtures
with a native taught text vocabulary are read after their own teaching,
before cleanup, including on an assertion failure. Tensor-only structural
fixtures report that they have no taught text vocabulary. The numeric
mixing lexical-form fixture likewise has no native text input.

Expectation's isolated omission override is removed before the assessment:
it is a teacher fixture that always hides its second word, not an ordinary
reading of a one-word probe. The original draw is restored, no learned
parameter or assertion is changed. `corpus_lessons.measure` adds one optional
`after_training` observer before cleanup so the trained model can be assessed
without a second training path.

The source-matched tables, lesson attention trace and final source snapshot
are pending the runs below. The first-walk teacher remains deferred by §9.4.
No edit has been made in any of the five protected attention places for the
assessment.


### 10.4 The attention producer (diagnosis, no repair)

The diagnostic lesson completed its 160 presentations. Its first attempt
also completed 160, but the observer tried to stack an empty set during the
following assessment read. The observer now handles an empty field and
keeps assessment reads out of the training comparison count. This was an
instrumentation failure, not a model or curriculum repair. The subsequent
diagnostic completed the assessment too, in 485.7353542909841 seconds;
these are preliminary measurements, before the finished-source run.

The native trace distinguishes legality from selection. On the post-lesson
read of `This is Felix.`, `This` and `Felix` are known but left out. The
actual tree is `non x2(surface(non(is), sign(.)))`. Attention spends **13**
acts and has **19** left. Its action addresses are **5, 1, 3, 4, 11, 7, 17,
13, 15, 16, 19, 20, 8**, followed by STOP (**192**). With six operations per
bracket these are: complement the input; descend/split; intersection then
union on `This`; complement and descend/split the rest; complement and
descend/split the last portion; intersection then union on `Felix`; descend
into the unknown full stop, mint it, then mint `is`.

At the word brackets, `This` and `Felix` both have `both=True`,
`singular=True`, `pure=False`, `neither=False`. `ModelAttention.native_word_poles`
reads their current native membership at their own extents, taking the
maximum over the lexical word and its defined objects, separately on each
lane. `Attention.narrowing_mask` requires pure, singular, word for mint;
known word brackets cannot split, and descent at a word requires neither.
Consequently none of the three progress operations is legal for these
both-lane word readings. Intersection and union of one word, and exchanging
its two nonzero lanes, do not make it pure. Once each field operation has
been used, those words have no remaining legal operation.

The next menu is empty. `Language.attend` masks STOP to minus infinity
whenever any legal action exists; with none it gives STOP zero. Thus the
chooser does **not** choose an early STOP over an available mint. It chooses
the preceding field/split order, but legality strands the two word brackets.
`ModelAttention.handoff` exposes only `reading.accepted` as grammar leaves.
There is no later grammar operation that can restore a word excluded there.

The same diagnostic on `What is this?` leaves `What`, with tree
`ask x2(preposition(non(is), ask(change(sign(this), ?))))`. It spends **12**
acts and leaves **20**; at the end `What` has the same both-lane state and
empty progress mask. The first-walk teacher is deferred. In its later
review, a teacher must confront this unavailable target; changing a
chooser's preference alone cannot select a masked-out mint. No legality,
chooser, teacher or protected attention code is changed in this round.

### 10.5 Checkpoint at the round-B mechanism question

This is **not the completed assessment return**. The source checkpoint is
`601945ecc9f1039711ab766be0e9b353e375c8e432df87a2c7cea4ac9202a547`.
The current six assessment tests pass in **9.19 seconds**, including a
contextual probe and the check that a shared verb cannot count as the
previous subject. The all-item-1.8 preflight reports **208 passed, one
skipped, three deselected, 98.03 seconds**; the three deselections are the
long certificate-81 case and the two native certificate-90 cases, which
still require their specified runs. This was a focused preflight, not the
default sweep.

Two integration tasks were found during review and remain unbuilt at the
stop in item 1.8 §44.2. The bounded pytest worker currently discards
successful teardown reports and their assessment user properties. The
explicit JSONL observer retains the tables, but the default runner must
retain them too, without changing test outcomes. Also, the transaction's
external `MemoryActivity` deferred-read journal must be saved and restored
when a probe is called inside an already-open trial; the current restoration
tests enter without such an enclosing journal. Neither limitation is
claimed covered by the passing tests above.

The lesson table is preliminary: rung 1 fails (the known word `Felix` is
left, with no tree, no minted word and no trusted single denotation);
rung 2 is not applicable because this source declares 23 words rather than
twenty; rungs 3, 4 and 7 are blocked by the lower failure; rungs 5, 6 and 9
lack the commented-out determiner; rung 8 lacks a declared verb modifier.
The source-matched tables for all native fixtures, both scalar curriculum
stages, and the final lesson remain owed. The attention diagnosis in §10.4
stands as a diagnosis of its recorded run, not a claim that the pending
finished-source measurements have run. No attention teacher or protected
attention edit was made.


### 10.6 Integration repairs after the checkpoint

The two integration tasks of §10.5 are now built. The evaluation
transaction also restores the external deferred-read activity journal,
including when invoked inside an enclosing kept trial. The new test starts
with one deferred read, observes the real native read, and requires exactly
the original journal and one final activity increment afterward. Before the
repair it left three reads and additional writes. The bounded worker now
retains assessment properties delivered during teardown on the original
call report. Its integration test deliberately fails the underlying test
and verifies that both its original error and the not-ready report remain.
Before the repair the assessment was missing. Neither change alters an
operator assertion or turns a failure into not-ready.

Both new checks failed before (**two failed, 7.63 seconds**) and pass after;
the focused run passes 17 checks, with only the separately unfinished
turnover fixture failing at that time. The full source-matched tables and
final sweep remain in progress. The protected attention places are unchanged.


### 10.7 First frozen-source lesson table and attention trace

Source `7bbac217ac4de1b8677648fd5814958c81df46518766941d4ca5c4c18c435424`,
unchanged throughout the 160-sentence lesson and its probes. Three restored
reads of the first rung agree; they are evaluation observations, not three
additional teaching rounds.

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `Felix`: known, left; no leaf or tree; no trusted single object. Each of the three restored reads has the same record. |
| 2 | Not applicable | This source declares 23 forms, not twenty. |
| 3 | Blocked | The lower rung has not passed. |
| 4 | Blocked | The lower rung has not passed. |
| 5 | Not applicable | `definite` is commented out in the grammar. |
| 6 | Not applicable | `definite` is commented out in the grammar. |
| 7 | Blocked | The lower rung has not passed. |
| 8 | Not applicable | No verb modifier is declared in this source's vocabulary. |
| 9 | Not applicable | `definite` is commented out in the grammar. |

There are **22 ordinary sentence traces** after teaching. For `This is
Felix.`, the kept tree is `ask(surface(non(is), lower(.)))`: `This` and
`Felix` are known but left, `is` is known and minted, and the full stop is
unknown and minted. The actual action addresses are
**4, 1, 5, 3, 10, 7, 8, 16, 13, 17, 15, 19, 20, STOP(192)**. The walk
spends **13/32**. On the word brackets for `This` and `Felix`, the last
applicable legality record says **both=true, pure=false, neither=false,
singular=true, word=true, can_split=false, has_parts=true**; all three
progress actions are false. At STOP, every action logit is negative infinity
and STOP is zero, with no legal action. The chooser is not selecting STOP
over a legal mint. The same unchanged `narrowing_mask` conditions and
`ModelAttention.handoff` producer described in §10.4 apply.

`What is this?` takes **4, 1, 5, 3, 10, 7, 8, 16, 13, 14, 19, 20,
STOP(192)**, spending **12/32**; its first word is likewise left. Across
the lesson, **304/544 lexical places** reach the grammar, the first word
**0/160** times and the final lexical word of a statement **48/80** times.
These are this run's numbers; the earlier prototype used another draw.
The lesson's 0/80 answers, 0/160 equal compose/generated fields and order
occupancies are written out in item 1.8 §44.6. This is diagnosis only; the
first-walk teacher remains deferred to the performance merge.


The marker measurement subsequently found an observer-only None-program
read (item 1.8 §44.7). Its repair changes the source snapshot, so the table
above remains the diagnostic table on its stated source; the final-source
lesson and fixture tables will follow it. The attention diagnosis is
unchanged, and there is still no teacher or protected attention edit.

### 10.8 Tables retained by the completed diagnostic sweep

These tables belong to source `7bbac217ac4de1b8677648fd5814958c81df46518766941d4ca5c4c18c435424`.
The worker retained **31 assessment reports**, including their teardown
properties, without altering the original test outcomes. Eight fixtures
have native text vocabularies; their tables follow. In each, rung 1 uses
three restored reads. The other 23 fixtures have no taught text vocabulary
and report that fact, rather than an invented learned reading. The final
source tables and both native scalar-curriculum tables remain owed.

**`test/test_item18_expect.py::test_81_expectation_restores_the_omitted_native_word`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `alpha`: known, minted; objects [113]; trusted definition; `bind.open x18(sign(alpha))`. different tree; repeated unary operator Three identical restored reads. |
| 2 | Not applicable | fixture declares 4 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_expect.py::test_82_omission_is_the_only_teacher_and_consumers_detach[sigmaPi]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `alpha`: known, minted; objects [111]; trusted definition; `ask(alpha)`. different tree Three identical restored reads. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_expect.py::test_82_omission_is_the_only_teacher_and_consumers_detach[predictor]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `alpha`: known, minted; objects []; untrusted definition; `sign(alpha)`. different tree; definition not trusted; word does not denote one object Three identical restored reads. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_intuit.py::test_84_intuit_reads_once_at_observation_and_no_relation_row`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `alpha`: known, minted; objects []; untrusted definition; `bind.open x19(alpha)`. different tree; repeated unary operator; definition not trusted; word does not denote one object Three identical restored reads. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_lexical_form.py::test_91_reading_and_free_generation_take_the_mapped_words_form[aligned]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `hello`: known, minted; objects [110]; trusted definition; `sign(hello)`. different tree Three identical restored reads. |
| 2 | Not applicable | fixture declares 1 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_lexical_form.py::test_91_spelling_presence_reads_form_and_keeps_its_gradient`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Passed | `wug`: known, minted; objects [110]; trusted definition; `wug`. The proposed leaf and single trusted object checks pass. Three identical restored reads. |
| 2 | Not applicable | fixture declares 1 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_word_trust.py::test_88_scalar_definition_reads_one_after_the_same_six_views`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | `alpha`: known, minted; objects [110]; trusted definition; `lift(alpha)`. different tree Three identical restored reads. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_word_trust.py::test_88_derived_code_reads_do_not_rebuild_definition_index`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Passed | `alpha`: known, minted; objects [110]; trusted definition; `alpha`. The proposed leaf and single trusted object checks pass. Three identical restored reads. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

The following **23 sources** each report all nine rungs **not applicable**:
"fixture exercises structure or numeric tensors; no taught text vocabulary".
Their declared prerequisite is still retained and their mechanism assertion
still runs. They do not become skipped tests or successful language probes.

- `test/test_item18_expect.py::test_82_omission_teacher_keeps_the_staged_input_bracket` (prerequisite 1).
- `test/test_item18_intuit.py::test_infer_ordinary_derivation_does_not_claim_a_knowing_deposit` (prerequisite 1).
- `test/test_item18_intuit.py::test_85_infer_keeps_the_old_three_rule_supports` (prerequisite 1).
- `test/test_item18_intuit.py::test_83_thought_teaches_the_open_map_and_an_unseen_conclusion_still_needs_inference` (prerequisite 1).
- `test/test_item18_lexical_form.py::test_91_reading_and_free_generation_take_the_mapped_words_form[mixing]` (prerequisite 1).
- `test/test_item18_marker.py::test_86_marker_preserves_form_and_has_an_exact_selected_inverse` (prerequisite 3).
- `test/test_item18_marker.py::test_86_compose_and_generate_choose_each_kernel` (prerequisite 3).
- `test/test_item18_marker.py::test_86_retired_unary_case_has_no_grammar_and_compound_stays` (prerequisite 3).
- `test/test_item18_marker.py::test_86_pinned_sentence_keeps_the_selected_kernel` (prerequisite 3).
- `test/test_item18_placement.py::test_91_context_includes_the_present_tokens_types_before_likeness` (prerequisite 1).
- `test/test_item18_placement.py::test_91_observed_formal_membership_is_also_a_field_definition` (prerequisite 1).
- `test/test_item18_placement.py::test_91_one_step_uses_only_context_and_negatives_of_the_rows_order` (prerequisite 1).
- `test/test_item18_placement.py::test_91_classes_have_no_form_and_are_placed_by_the_same_context` (prerequisite 1).
- `test/test_item18_recognition.py::test_scalar_percept_and_its_collection_restore_with_their_typed_references` (prerequisite 4).
- `test/test_item18_stack.py::test_79_stack_learns_the_four_disjoint_word_combinations` (prerequisite 1).
- `test/test_item18_stack.py::test_79_stages_are_the_shared_pi_and_sigma_layers` (prerequisite 1).
- `test/test_item18_stack.py::test_79_stack_construction_leaves_the_global_draw_unchanged` (prerequisite 1).
- `test/test_item18_stack.py::test_80_numeric_certificate_keeps_its_bar_with_the_stack_built` (prerequisite 1).
- `test/test_item18_word_trust.py::test_88_read_word_fixes_its_objects_radius_without_freezing_its_content` (prerequisite 1).
- `test/test_item18_word_trust.py::test_88_stated_definition_keeps_the_callers_trust[0.0]` (prerequisite 1).
- `test/test_item18_word_trust.py::test_88_stated_definition_keeps_the_callers_trust[0.75]` (prerequisite 1).
- `test/test_item18_word_trust.py::test_89_expectation_selects_among_orders_and_is_local_to_the_stream` (prerequisite 1).
- `test/test_item18_word_trust.py::test_89_written_prediction_is_local_across_constructed_streams` (prerequisite 1).

The finished diagnostic sweep itself reports 6,754 passed, 330 skipped,
10 failed, one expected failure and one unexpected pass over 7,096 unique
selectors. Nine fixture failures are ported in item 1.8 §44.8; the verbal
proof exhausts LTM0 before its read-back and is being diagnosed. Its
capacity exception is not an assessment readiness result. The native
certificate-90 fixtures are RUN_SLOW and absent from this default sweep.
No first-walk teacher or protected attention edit was made.

### 10.9 Stop at the remaining memory-policy question

Item 1.8 §44.9 records the completed diagnostic sweep's remaining failure:
the verbal proof reaches a full LTM0 at a closed boundary, but the
256-sentence minimum interval prevents the next activity pass. The next
close correctly raises. The question is whether capacity pressure overrides
that interval, as it already does for serving writes. No such policy change
has been made. Both stopping gates passed; the proof failed in setup, not
on its reconstruction assertion.

The current checkpoint is
`a03b0d0953661e51d5b18c986432898dc2c2710801516b1356c726e01b41e185`.
The tables of §10.7–10.8 retain the earlier source explicitly stated there;
they do not claim to be the finished-source tables. Both native scalar
curriculum tables and the final-source measurements remain owed when the
memory question is decided. The assessment transaction, native reader,
proposed frames and reporting integration stand; the first-walk teacher
remains deferred. No commit, push or protected attention edit was made.

### 10.10 The reissued teacher amendment meets its legality stop

Alec reissued §9's assessment text and §9.3's conditional teacher amendment
while the approved memory-pressure change was beginning. This conflicts
with the later deferral in §9.4 and item 1.8 §43. I have asked which governs;
no teacher or attention change is made while that is pending.

The recorded trace already identifies the amendment's **illegal-mint**
case, not its case of a chooser declining an available mint. In the lesson
read of `This is Felix.` (§10.7), both omitted word brackets are singular,
known words with both lanes present: `both=true`, `pure=false`,
`neither=false`, `word=true`, `can_split=false`, `has_parts=true`.
`Attention.narrowing_mask` computes:

- split: `both & can_split & ~word` — false;
- descent: `has_parts & (~word | neither)` — false;
- mint (still locally named `gloss` in the protected file):
  `pure & singular & word` — false.

The walk takes actions **4, 1, 5, 3, 10, 7, 8, 16, 13, 17, 15, 19, 20**,
then STOP **192**, with **19/32** acts left. At that last choice all act
logits are minus infinity and STOP is zero. This is not STOP beating a
legal mint. `ModelAttention.handoff` passes only accepted word places to
the grammar; a later grammar teacher cannot restore the excluded words.
The read of `What is this?` has the same blocked progress at its first word
and stops with **20/32** acts left. Those are the recorded source-matched
observations, not newly trained results.

**Recommendation for Alec and Claude:** keep "teach it first" and resolve
word-acquisition legality before adding that teacher. A concrete rule for
review is to permit acquiring a known, singular word when its evidence is
on both lanes, carrying both lanes unchanged into the grammar. Acquiring
the word need not assert its content as true. The chooser would still have
to learn to take that legal act in the requested order; this proposal does
not force the next word or clear an evidence lane. It changes the existing
purity requirement and is therefore a design decision, not an implemented
repair. I recommend retaining the already-decided performance-merge
placement for this attention work rather than reopening the protected
files during round B.

Under the reissued conditional text, this is the required stop before a
labeler, teacher, teaching stage or held-out teacher measurement. None is
claimed built or measured. The tool, proposed probe and diagnostic tables
already returned above stand; the two completed scalar-curriculum tables
and the final-source measurements remain owed. The current assessment
recheck reports **8 passed in 12.34 seconds**: its seven tool/transaction
checks and the bounded-runner check retaining the original failure together
with its assessment. Reporting readiness does not replace assertions.

The source snapshot remains
**`a03b0d0953661e51d5b18c986432898dc2c2710801516b1356c726e01b41e185`**.
There are **zero new edits** in the four protected attention files or
`Models._run_sentence_word_bricks`. No code was committed or pushed.

### 10.11 Deferral confirmed; observations resumed

Alec clarified that the earlier assessment text was pasted by mistake.
§9.4 and item 1.8 §43/§43.1 govern. The first-walk teacher remains deferred
until after the performance merge. The approved memory pressure change is
built as item 1.8 §44.11 records. Validation on
`e75d3cd124d6278a0f9825688d2749c803f07f5457e6875bf32e8773b8616fcd`
was interrupted for Alec's subsequent read-only attribution amendment,
with both stopping gates passed and no unexpected failure. §44.12 records
that attempt separately. No protected attention edit was made.

### 10.12 Native lane contributors, under Alec's §11 decision

The proposed both-lane mint in §10.10 is **not taken**. Mint's purity stays;
the narrowing design and the first-walk teacher wait until after the
performance merge. The tool now records the native meanings that supplied
each word's lanes, as Alec requested in §11. It changes no attention rule.

`Assessment._native_word_fields` observes the existing call to
`ModelAttention.native_word_poles` and captures the field returned by its
existing `cs_read_memberships` call. It makes **no second field read**.
At the word's own byte extent it records the word's concept and every
defined object's concept, their physical rows and their separate for/against
values. Each lane names every positive maximum contributor, including ties.
These are the native input lanes **before the walk's field operations**;
they are not inferred from definition trust or from a later handoff pair.
`read_sentence` includes this record even when detailed tracing is disabled.
For a word left with both native lanes, `format_record` prints each lane's
value and its contributing word/object identities and rows.

The new
`test_native_lane_attribution_keeps_extent_identity_and_all_tied_sources`
checks a word contributing the for lane, two objects tied for against,
and another occurrence of that same word where one object supplies both
lanes. An unrelated concept contributes neither; unsorted identities and
the distinct extents are retained. The test verifies exactly one native
field call and restoration of the wrapped functions. The existing full
transaction test now requires the native record and compares the complete
tree and walk with this observer disabled under the same restored state.
Its memory, clocks, parameters and random-stream checks remain.

Before implementation, those two checks fail (**2 failed, 3.52 seconds**):
the observer and the record are absent. The final focused tool/runner run
reports **9 passed, 41 unrelated tests deselected, 18.18 seconds**. This is
not a substitute for the complete default sweep. No fixture's capacity,
seed, learning rate or training presentations changed.

The amended validation source is
**`eb87db17045ad78c83bab06903209ee73f2d8f3bc6a953caf26d99fb92faf1c2`**.
The lesson, fixture tables and required round-B measurements are running
on that source. They will report the actual sources of the capitalized
and lowercase words' lanes, without treating case as a rule in code.

### 10.13 Finished-source lesson: which meaning supplies each lane

The four-epoch, 160-sentence lesson completes on
`eb87db17045ad78c83bab06903209ee73f2d8f3bc6a953caf26d99fb92faf1c2`.
The following are ordinary restored probe reads **after teaching**, not
the early training trees compared in item 1.8 §44.13. Each source below is
the actual native field at the stated word extent. The concept identity
and physical row are kept distinct.

| Text and word | Extent | For: value and source | Against: value and source | Walk |
|---|---|---|---|---|
| `This is Felix.`: `This` | [0,4] | 1.0, word concept 117, row 17 | 1.0, the same word concept 117, row 17 | Left |
| `This is Felix.`: `Felix` | [8,13] | 1.0, word concept 113, row 13 | 1.0, the same word concept 113, row 13 | Left |
| `What is this?`: `What` | [0,4] | 1.0, word concept 118, row 18 | 0.9999999403953552, the same word concept 118, row 18 | Left |
| Both texts: `is` | [5,7] | 0.0, no contributor | 1.0, word concept 121, row 21 | Minted |
| `What is this?`: `this` | [8,12] | 0.0, no contributor | 1.0, word concept 124, row 24 | Minted |

`This`, `Felix` and `What` have no defined objects in this trained model's
read. They are **one concept holding both lanes**, not different objects
winning different lanes. The read of `is` also inspects object 125, row
2418, at **(0,0)**; `this` also inspects object 143, row 2421, at **(0,0)**.
Those objects contribute neither maximum. These observations distinguish
the two cases raised in §11; they do not establish capitalization as a
cause, and no case or spelling rule is added.

For `This is Felix.`, the walk takes **1, 4, 3, 5, 7, 13, 16, 15, 17, 8,
19, 20**, then STOP **192**, with **20/32** acts remaining. The first and
third word brackets have `both=true`, `pure=false`, `neither=false`,
`singular=true`, `word=true`, `can_split=false`, `has_parts=true`. All
ordinary choices are illegal at STOP; its logit is zero. For `What is
this?`, the acts are **1, 4, 3, 5, 7, 13, 14, 8, 19, 20**, then STOP,
with **22/32** remaining; its first word has the same both-lane mask.
Thus the new source still reaches the existing legality boundary. Under
Alec's decision that boundary is recorded for later narrowing work, and
purity is not relaxed. The tool retained 22 distinct sentence traces.

### 10.14 Amended-source assessment tables

These **32 tables** are on source
`eb87db17045ad78c83bab06903209ee73f2d8f3bc6a953caf26d99fb92faf1c2`:
31 reports from the completed fixture reads in the completed default sweep,
plus the completed four-epoch lesson. Eight fixture sources and the lesson
have native text vocabularies; 23 structural/numeric sources explicitly
have none. The two completed scalar-curriculum reports are in §10.15. No proposed tree gates an outcome, and these readings train nothing.

Of the eight native fixture sources, **one passes rung 1 and seven fail**;
the lesson also fails rung 1. The records below state the failure instead
of replacing its operator assertion. Rung 2 requires the proposed twenty
words: no fixture declares exactly that vocabulary, and the lesson declares
23. It is reported unavailable, without selecting another vocabulary or
claiming a twenty-word result. The unavailable determiner and the literal
lower-rung prerequisite rule remain as stated in §10.2. These limitations
of the proposed ladder remain for review.

**`test/test_item18_expect.py::test_82_omission_teacher_keeps_the_staged_input_bracket`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_expect.py::test_81_expectation_restores_the_omitted_native_word`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | missing leaves; different tree; word not minted. Record: `alpha`; 0 alpha: known, left, row 12, objects [119], trusted; native alpha at [0, 5]: for 0.0013334155082702637 from word 109 (row 12); against 0.9997738599777222 from word 109 (row 12). Tree `(empty)`. Missing [0]. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 4 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_expect.py::test_82_omission_is_the_only_teacher_and_consumers_detach[sigmaPi]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree; repeated unary operator. Record: `alpha`; 0 alpha: known, minted, row 12, objects [111], trusted. Tree `bind.open x18(non(alpha))`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_expect.py::test_82_omission_is_the_only_teacher_and_consumers_detach[predictor]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree; definition not trusted; word does not denote one object. Record: `alpha`; 0 alpha: known, minted, row 12, objects [], untrusted. Tree `sign(alpha)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_intuit.py::test_infer_ordinary_derivation_does_not_claim_a_knowing_deposit`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_intuit.py::test_85_infer_keeps_the_old_three_rule_supports`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_intuit.py::test_84_intuit_reads_once_at_observation_and_no_relation_row`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree; definition not trusted; word does not denote one object. Record: `alpha`; 0 alpha: known, minted, row 12, objects [], untrusted. Tree `non(alpha)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 2 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_intuit.py::test_83_thought_teaches_the_open_map_and_an_unseen_conclusion_still_needs_inference`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_lexical_form.py::test_91_reading_and_free_generation_take_the_mapped_words_form[aligned]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree. Record: `hello`; 0 hello: known, minted, row 12, objects [110], trusted. Tree `sign(hello)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 1 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_lexical_form.py::test_91_reading_and_free_generation_take_the_mapped_words_form[mixing]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_lexical_form.py::test_91_spelling_presence_reads_form_and_keeps_its_gradient`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree. Record: `wug`; 0 wug: known, minted, row 12, objects [110], trusted. Tree `sign(wug)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 1 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_marker.py::test_86_marker_preserves_form_and_has_an_exact_selected_inverse`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_marker.py::test_86_compose_and_generate_choose_each_kernel`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_marker.py::test_86_retired_unary_case_has_no_grammar_and_compound_stays`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_marker.py::test_86_pinned_sentence_keeps_the_selected_kernel`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_placement.py::test_91_context_includes_the_present_tokens_types_before_likeness`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_placement.py::test_91_observed_formal_membership_is_also_a_field_definition`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_placement.py::test_91_one_step_uses_only_context_and_negatives_of_the_rows_order`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_placement.py::test_91_classes_have_no_form_and_are_placed_by_the_same_context`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_recognition.py::test_scalar_percept_and_its_collection_restore_with_their_typed_references`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_stack.py::test_79_stack_learns_the_four_disjoint_word_combinations`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_stack.py::test_79_stages_are_the_shared_pi_and_sigma_layers`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_stack.py::test_79_stack_construction_leaves_the_global_draw_unchanged`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_stack.py::test_80_numeric_certificate_keeps_its_bar_with_the_stack_built`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_word_trust.py::test_88_read_word_fixes_its_objects_radius_without_freezing_its_content`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_word_trust.py::test_88_stated_definition_keeps_the_callers_trust[0.0]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_word_trust.py::test_88_stated_definition_keeps_the_callers_trust[0.75]`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_word_trust.py::test_89_expectation_selects_among_orders_and_is_local_to_the_stream`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`test/test_item18_word_trust.py::test_88_scalar_definition_reads_one_after_the_same_six_views`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree. Record: `alpha`; 0 alpha: known, minted, row 12, objects [110], trusted. Tree `ask(alpha)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_word_trust.py::test_88_derived_code_reads_do_not_rebuild_definition_index`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Passed |  Record: `alpha`; 0 alpha: known, minted, row 12, objects [110], trusted. Tree `alpha`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_word_trust.py::test_89_written_prediction_is_local_across_constructed_streams`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1–9 | Not applicable | fixture exercises structure or numeric tensors; no taught text vocabulary |

**`explore-1 lesson after four epochs`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | missing leaves; different tree; word not minted; definition not trusted; word does not denote one object. Record: `Felix`; 0 Felix: known, left, row 13, objects [], untrusted; native Felix at [0, 5]: for 1.0 from word 113 (row 13); against 1.0 from word 113 (row 13). Tree `(empty)`. Missing [0]. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 23 words, not twenty |
| 3 | Blocked | lower rung has not passed |
| 4 | Blocked | lower rung has not passed |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Blocked | lower rung has not passed |
| 8 | Not applicable | fixture has no declared verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |


### 10.15 Completed scalar curricula and final assessment return

Both certificate-90 fixtures reached their fixed **200-round bound**:
**600 training presentations and 600 evaluation readings each**, with
**zero exact readings and zero good rounds**. The failed producer is the
retained compose tree, including repeated open binds; the later 18
presentations and recognition calls are not reached. Item 1.8 §44.17 gives
the final trees, the taught-choice probabilities and the measured runtimes.
These are the read-only known probes after those failed stages, before
fixture cleanup. They do not train a word or substitute a supplied reading.

**`test/test_item18_recognition.py::test_90_present_percept_is_said_by_its_class_without_naming_an_object`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree; repeated unary operator. Record: `alpha`; 0 alpha: known, minted, row 12, objects [110], trusted. Tree `bind.open x19(alpha)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

**`test/test_item18_recognition.py::test_90_named_percept_does_not_mix_native_spellings`**

| Rung | Result | Record or reason |
|---:|---|---|
| 1 | Failed | different tree; repeated unary operator. Record: `one`; 0 one: known, minted, row 12, objects [110], trusted. Tree `bind.open x19(one)`. 3 restored reads, 1 distinct records. |
| 2 | Not applicable | fixture declares 3 words, not twenty |
| 3 | Not applicable | fixture has no declared name |
| 4 | Not applicable | fixture has no declared name, verb |
| 5 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 6 | Not applicable | definite is commented out in complete.grammar (§27.1) |
| 7 | Not applicable | fixture has no declared pronoun, name, verb |
| 8 | Not applicable | fixture has no declared name, verb, verb_modifier |
| 9 | Not applicable | definite is commented out in complete.grammar (§27.1) |

The complete collection is **34 tables**: 23 structural/numeric sources
with no taught text vocabulary, ten native fixture sources, and the
four-epoch lesson. **One of the ten native fixture sources passes rung 1;
nine fail.** The lesson fails it too. Every unavailable or blocked rung
retains its reason. The trees remain proposals for review, not sweep gates.
The two scalar probes above each have one known, minted, trusted word,
physical row **12**, with object **110**; the failing producer is the
**19 repeated open binds**, rather than an unknown or unminted word.
Each record is identical across its three restored reads.

The added native-lane observer's regression checks fail before it and pass
after it (§10.12); the complete default sweep on this same source passes
**6,766 tests**, skips **330**, reports **one expected failure and one
unexpected pass**, and has **zero unexpected failures**. Both stopping
gates pass. The two native scalar cases separately report their acquisition
assertions as non-strict expected failures under the baseline rule. All
four supervised commands exit **0**, without a stopping event, and their
source checks are unchanged. The full measurement results, baseline
producers and test ports are in item 1.8 §44.13–18.

Returned `bounded_tests.source_snapshot`, **1,184 inputs**:
**`eb87db17045ad78c83bab06903209ee73f2d8f3bc6a953caf26d99fb92faf1c2`**.
The four protected attention files are byte-identical to the landing commit.
`Models._run_sentence_word_bricks` is text-identical to returned round A;
there are **zero new edits in all five places** for this assessment. No
teacher, both-lane mint or new narrowing act is built. The exact lane
contributors in §10.13 remain evidence for the teacher's round after the
performance merge. Receipts stay local and are not linked. Nothing is
staged, committed or pushed.


## 11. Both lanes at once: narrowed before it is learned (Alec, 2026-11-15)

Alec's answer to the question of §10.10: "a concept that has conflicted
evidence cannot be learned if both for and against are present *at the
same time*: the attention has to be narrowed to avoid ambiguity before
a concept is learned (although it can be learned with both sets of
evidence if it has been created that way in MTM)."

Asked again, he made it exact: a known word whose concept holds both
lanes "CANNOT be minted if both lanes are active at once." It may be
minted when the reading activates one of its lanes.

**What it decides.** The purity that `Attention.narrowing_mask` asks of
a mint (`pure & singular & word`) is right and stays. The rule
proposed in §10.10, to mint a known word with both lanes present, is
not taken. The way past a word read with both lanes is to narrow
attention until one lane remains. The one exception is a concept whose
MTM entry was made with both lanes (as a basic percept's pair is, §18
of `doc/plans/2026-11-14-sigma-pi-layers.md`): it may be learned
carrying both.

**Where both lanes come from today** (read by Claude in the source; not
measured). `ModelAttention.native_word_poles` reads the native field at
the word's own extent for the word's row and each of its objects
(`owner.definitions.objects(word)`), and keeps the largest value on each
lane separately. So a word's two lanes may come from two of its
concepts, each read with one lane, or from one concept read with both at
that extent. The two cases narrow differently (*Claude's reading*):

1. *Two concepts, one lane each.* The ambiguity is between concepts of
   one word, and narrowing means attending one of them. The walk has no
   act for this today: its acts narrow the span (split, descent) or
   rewrite the field (complement, intersection, union, exchange of the
   lanes), and none chooses among a word's concepts.
2. *One concept with both lanes at the word's extent.* The ambiguity is
   over the word's parts, and narrowing means descending into them.
   `narrowing_mask` forbids descent at a known word unless neither lane
   is present.

Codex's trace does not say which case holds for `This`, `Felix` and
`What`. Noted and not checked: each word reported left is capitalized,
each lowercase word reported (`is`, `this`) was minted, and every
sentence of the lesson begins with a capital (`form_key` does not fold
case, so `This` and `this` are two forms).

**When.** With the first walk's teacher, after the merge of the
`performance` branch (§9.4); nothing in attention changes in round B.
In round B the assessment's record of a word left with both lanes
names which of its concepts gives each lane (Alec: "Yes, add it"), so
the design of the narrowing act starts from what is read.


## 12. The proposed ladder, reviewed (Alec and Claude, 2026-11-15)

Codex's proposed trees of §10.2 are accepted, with two changes Alec chose
("Determiner as modifier"):

- *A determiner is a modifier* (§27.1 of the 1.8 specification). Rungs 5, 6
  and 9 are available. With `modifier` projecting its left head:
  rung 5, `the cat ran`, is `verb(modifier(1,0),2)`; rung 6, `the fast cat
  ran`, is `verb(modifier(modifier(2,1),0),3)`; rung 9, `the fast cat ran
  quickly`, is `verb(modifier(modifier(2,1),0),modifier(3,4))`. Whether
  `the` binds an earlier referent is chosen at run time (`bind`), as for
  any operator, and is not part of the tree.
- *Rung 2 is at least twenty words* (*default*), so a source that declares
  more, as the lesson does, is read.

The other trees stand: rung 1 `0`, rung 3 `0` with one denotation, rung 4
`verb(0,1)`, rung 7 `verb(bind(0),1)` after its context, rung 8
`verb(0,modifier(1,2))`. The ladder still reports and gates nothing in the
sweep.

The identity lesson Alec described the same day (numbers spoken of as
individuals, with the value seen in the scalar modality) is FutureWork only:
`doc/FutureWork.md`, "Identity taught by a second modality".
