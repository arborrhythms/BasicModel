# ModelAge

A model has a mental age: one number, in years, determined by the assessment
(`bin/Assessment.py`). The age says which tasks the model has learned. It is
a parameter of the model and of its XML, and the assessment is the only thing
that writes it.

The age exists so that a test is run only against a model that is ready for
it. A test of understanding or identity asks the assessment for a mental age.
If the model is younger, the test reports that the model is not qualified at
that age, and does not count as a failure of what the test set out to
measure.

## How to read the chart

Each row is a task with the age at which it is counted as learned. The ages
follow the age at which a child learns the same thing. A model's age is the
greatest age in the chart with every measured row at or below it passed.

A model here takes the words of a sentence in order and says them back. That
is nearer to a child hearing and repeating than to a child reading print, and
the ages are set accordingly.

The column "Basis" says where the age comes from:

- *sourced*: a published norm or study, listed under Sources. Published ages
  are ranges and differ between sources; the chart takes a point inside the
  range.
- *estimate*: no source was found in the check of 2026-11-22; the age is a
  judgement and may move.

The column "Measured by" names what the assessment measures. "None yet" means
the row has no test: it is in the chart because it may serve as a sub-goal
when a later task has to be broken down.

## The chart

### Before words

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 0.70 | A thing is kept when it is out of the look (object permanence) | sourced: search for a fully hidden object at 8 to 12 months (Piaget); looking-time evidence from 3.5 months (Baillargeon) | the ladder's rung "one word" |
| 0.75 | A heard form is said back as a form, with no meaning asked | estimate | none yet |

### Single words

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 1.00 | A word is said back | sourced: first word at 10 to 14 months | each word alone as the input, the word as the output |
| 1.50 | Ten words are said back; a thing is named by one word | sourced: at least ten words at 18 months | the rung "Felix" |
| 1.70 | Twenty words are said back | sourced: between ten words at 18 months and fifty at 24 | the rung "twenty words" |
| 2.00 | Fifty words are said back | sourced: about fifty words at 24 months, then a spurt | none yet |

### Words together

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 1.75 | Two words are taken in order and said back | sourced: two-word combinations at 18 to 24 months | two-word sentences, read in order and stopped at the end |
| 2.00 | A noun and a verb (who did what) | sourced: Brown's stage I, 12 to 26 months | the rung "Felix ran" |
| 2.00 | A property and a thing ("fast cat") | estimate: among Brown's stage I relations | none yet |
| 2.25 | "What" is answered | sourced: "what" before 2;02, then "where" (Wong and Ingram 2003) | "what is cat ?" |
| 2.50 | A sentence of three words is taken in order and stopped | sourced: Brown's stage II, mean length 2.0 to 2.5 at 27 to 30 months | the lesson's three-word sentences |
| 2.50 | "in" and "on" | sourced: Brown's morphemes 2 and 3, 27 to 30 months | none yet |
| 2.50 | Two unrelated words are said back in order | sourced: span tests are usable from 2.5 years | the stage `reading_two` |
| 2.50 | "Where" is answered | sourced: follows "what" | none yet |
| 2.50 | More than one ("cats") | sourced: Brown's morpheme 4 | none yet |
| 2.75 | A thing is named in a sentence with "is" | sourced: the uncontractible copula, Brown's morpheme 7, 27 to 39 months | "This is Felix." |
| 2.75 | A relation of place in a sentence | sourced: "on" and the copula | "cat is on mat" |
| 3.00 | A determiner | sourced: articles, Brown's morpheme 8, 28 to 46 months | the rung "the cat ran" |
| 3.00 | A sentence of four words is taken in order and stopped | sourced: sentences of 3 to 4 words at 3 years | the lesson's four-word sentences |
| 3.00 | "Who" is answered | sourced: "what", "where", "who" come before "how", "why", "when" (Bloom and others 1982) | "what sleeps ?" stands nearest |
| 3.25 | A pronoun for a third person ("he") | sourced: pronouns used more consistently at 3 years | the rung "he ran" |
| 3.25 | A determiner, an adjective, a noun and a verb | estimate | the rung "the fast cat ran" |
| 3.50 | Three unrelated words are said back in order | sourced: a span of three or less is still common at 5 | none yet |
| 3.50 | "Why" and "how" are answered | sourced: later than "what", "where", "who" | none yet |
| 4.00 | A sentence of five or six words is taken in order and stopped | sourced: sentences of 5 to 6 words at 4 years | the lesson's longest sentences, and held out |
| 4.25 | An adverb | estimate | the rungs "Felix ran quickly" and "the fast cat ran quickly" |
| 4.50 | One thing is compared with another ("bigger than") | estimate | "x is bigger than y" |
| 5.00 | Four unrelated words are said back in order | sourced: mean forward span about four items at 5 years | the stage `reading_four` |

### Counting

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 2.75 | "One" is told from more than one | estimate: the one-knower of Wynn's levels; the levels are sourced, their ages were not found | none yet |
| 3.00 | The counting words are said in order, to "4" | estimate | one counting word for each word of a sentence |
| 3.25 | "Two" and "three", each learned in turn | estimate: two-knower and three-knower | none yet |
| 3.75 | Each thing is counted once, and the last word said is how many | estimate: the cardinal-principle knower | counting, with each word looked at once |
| 4.50 | The counting words are said in order, to "10" | estimate: "counts to 10" is on the CDC's list for 5 years as recalled; the page's wording was not confirmed in the check | counting, to "9" |
| 5.00 | Counting on request with nothing to count: "Please count from one to 15" | estimate | none yet |
| 5.00 | What comes after a counting word, asked anywhere in the list | estimate | none yet |

### Forms and logic

These rows are things the model does today that the other groups do not
cover. Their ages are estimates.

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 0.75 | A known word is recognised by its form, and is not seen inside a longer word that only shares its letters | estimate: familiar word forms are recognised before the first word is said | `test/test_item17_forms.py` |
| 2.25 | "Not": what is absent is told from what is present | estimate | the and, or, not certificates of the Sigma and Pi layers |
| 3.00 | "And" and "or" over two properties | estimate | the same certificates |
| 6.00 | One or the other but not both (exclusive or) | estimate: a rule over two features at once is late in children | `test/test_mm_xor.py`, the standing gate |

## Metal Baby's age today

The model is named Metal Baby. Measured on 2026-11-22, at `1f48e6c2a3`:

**Its age is 0.70.** Object permanence is the only row of the ladder that is
both measured and passed, and the next measured row fails.

| Row | Result today |
| --- | --- |
| 0.70, object permanence | Passed in 3 of the 4 test models that have text; failed in 1 ("unknown word"). The ladder's eight higher rungs were not reached: no test model declares the vocabulary they need. |
| 0.75, a word recognised by its form | Passed. |
| 1.00, a word said back | Not measured. The measure is being built. |
| 1.75, two words taken in order and said back | Failed. After the reading lesson, 0 of 6 two-word sentences are read in order in each of three starts, though the first word is taken first in 6 of 6. Every word of a sentence is found (64 of 64), but not in order, and "is" is never taken. |
| 2.25 to 5.00, with the words handed to the model in order | The curriculum's stages `reading_two`, `reading_four`, `identities`, `noun_verb` and `relations` pass when the model does not have to take the words itself. They do not count toward the age, since taking the words in order is the row that fails. |
| 6.00, exclusive or | Passed (the standing gate). |

So the model is uneven: it has learned things a child learns late (exclusive
or, relations between named things) while failing one a child learns before
two, taking two words in order. The age reports the first failure, not the
best result.

In a toy outside the model, the method now being built reads untaught
lengths in order, stops at the end and counts to "4" exactly. That is the
toy's result and not the model's.

## How the tasks are taught

The tasks are taught in the chart's order. Each word is taught alone until
every word is said back. Then sentences are read over those words, and their
words are counted. Then the grammar is taught, one rule at a time. A stage is
left when its measure passes, not after a fixed number of presentations.

Reading and counting are two outputs over the same input sentences, on the
same model and the same weights. No concept's activation is set from outside
to choose a task: every activation comes from the input or from the output.

Counting on request differs from both. The input does not change while the
count goes on, so nothing in the input can say what comes next. Each counting
word said is stored as an episode of long-term memory, and the next is
recalled from it: a loop of several steps that runs in long-term memory.

## Status

The chart was checked against published sources on 2026-11-22. Rows marked
*estimate* were not confirmed. The assessment today has the nine rungs from
"one word" to "the fast cat ran quickly", and reaches only the first, because
no test model declares the vocabulary the others need. The rows for a word
said back, reading in order and counting are being built in item 1.7. The
`age` parameter is not yet in the model or its XML: the age above was worked
out by hand from the results named beside it.

## Sources

- Brown's stages and the fourteen grammatical morphemes (ages added by later
  authors, not by Brown):
  <https://jcu.pressbooks.pub/app/uploads/sites/16/2024/12/Brown‘s-stages-of-language-development.pdf>,
  <https://www.csun.edu/~vcoao0el/de361/de361s61_folder/tsld016.htm>
- First words, vocabulary at 18 and 24 months, two-word combinations:
  <https://www.healthlinkbc.ca/living-well/parenting/parenting-toddlers-12-36-months/toddler-growth-and-development/toddlers-1>,
  <https://myhealth.alberta.ca/Health/pages/conditions.aspx?hwid=ue5419>,
  <https://www.chla.org/blog/advice-experts/learning-talk>
- Sentence length and pronouns at 3 to 5 years:
  <https://www.healthlinkbc.ca/healthwise/speech-and-language-milestones-ages-3-5-years>
- Object permanence: <https://www.simplypsychology.org/object-permanence.html>,
  <https://britannica.com/science/object-permanence>
- The order of question words: <https://core.ac.uk/download/210883054.pdf>
  (reviewing Bloom, Merkin and Wootten 1982, and Wong and Ingram 2003)
- Number-knower levels (Wynn 1990, 1992):
  <https://sites.socsci.uci.edu/~lpearl/courses/readings/SarneckaWright2013_ExactNumber.pdf>,
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10598441/>
- Counting milestones: <https://cdc.gov/act-early/milestones/5-years.html>
- Forward span in young children: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4310560>
