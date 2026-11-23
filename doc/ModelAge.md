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

Every row is measured the same way: an input is given to the model and its
own output is read. There is input, there is reconstruction, and there is
output; nothing reaches into the model to set or to read what is inside it in
order to decide a row (Alec, 2026-11-22). Saying words back and reading them
in order is mostly animal cognition, and understanding of a higher order is
human cognition, but that is a description of the tasks and not two ways of
measuring them.

What decides a row is the output: the answer the model returns. What teaches
the model is more than that. A training step charges two costs together, the
reconstruction of the input (R) and the answer against the answer that was
supplied (A), and it keeps and credits a reading by R + A
([Training](Training.md#what-questions-and-the-two-primary-costs),
[GradientFlow](GradientFlow.md)). An answer is said only from a thought the
model has concluded; with no conclusion it is silent. So a row can fail in
two places: the input is not put back, or the model has nothing concluded to
say.

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
| 1.00 | A word is said back (the echo test) | sourced: first word at 10 to 14 months | each word alone as the input and as the supplied answer; the answer the model returns spells the word |
| 1.50 | Ten words are said back; a thing is named by one word | sourced: at least ten words at 18 months | the rung "Felix" |
| 1.70 | Twenty words are said back | sourced: between ten words at 18 months and fifty at 24 | the rung "twenty words" |
| 2.00 | Fifty words are said back | sourced: about fifty words at 24 months, then a spurt | none yet |

### Words together

| Age | Task that must be learned | Basis | Measured by |
| ---: | --- | --- | --- |
| 1.50 | Two words are said back in order as a sequence, with no understanding asked | estimate: a child repeats two words before it combines two of its own | none yet |
| 1.75 | Two words are taken in order and said back as a sentence | sourced: two-word combinations at 18 to 24 months | two-word sentences, read in order and stopped at the end |
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
| 3.50 | Singing: a counted sequence with a number left out is said whole | estimate; the task is still to be talked through (Alec, 2026-11-22) | none yet |
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

**Its age is 0.75.** Object permanence and recognising a word by its form
pass, and the next row, a word said back, fails.

| Row | Result today |
| --- | --- |
| 0.70, object permanence | Passed in 3 of the 4 test models that have text; failed in 1 ("unknown word"). The ladder's eight higher rungs were not reached: no test model declares the vocabulary they need. |
| 0.75, a word recognised by its form | Passed. |
| 1.00, a word said back | **Failed**, by the model's own output, on `main`. Each of 19 words presented alone for 30 rounds, three starts: 0 of 19 in every round; 19 of 19 in every round; 19 then 0. The word's own concept is exact where the word is read. But the input is reproduced from the sentence's composed root, and "This." is a composition of the word and its mark by whichever rule the grammar took; some rules pass the word through and others do not. Given the word's own concept, the way back does give the word (19 of 19). A closing mark is never said back, since it is never made a word. Measured again on the branch `item17-r6d11` at `9d2fc4751a`, with a mark made a word and each leaf put back through the perceptual inverse: 0 of 19 in three starts, taught or not. Whatever leaf the output is given is spelled exactly and at its place, but the walk stops at the root and gives one leaf (the word, the mark, another word or nothing), and "This." is the word and its mark together. The root has to be taken apart by the inverse of what composed it; that is not yet built. Measured again at `11b916f86c`, with the root taken apart by the inverse of each rule that composed it, and that the only reconstruction: every word in some starts and none in others (19, 19, 19, 19, 18 and 0 of 19 over three starts, untaught and after 3 rounds; in an earlier run 19, 16, 5, 17, 19 and 1). Which rule the grammar took for the word and its mark decides it: some rules give the word back, four have no inverse and give nothing, and one gives back another word. Where a word is said back, the idea gives one of the two leaves; the other, and the place of both, are read from what the model kept of the way up. Where the reading applied no rule, the word is put back from its own basic-level concept. Measured again at `f396314f9e`, with nothing read from what was kept of the way up (Alec: "The idea is what does the reconstruction"): 0 of 19 in every start and round. The best case, the rule `part`, gives back the word and not its mark ("This." is put back as "This"); the other rules give nothing. No rule gives back both of its operands from the idea alone, and an untaught grammar draws any rule. The grammar's rules are next made available one at a time, each taught until it can be undone. *By the answer the model returns,* which is this row's measure: empty. In Codex's trial at `23a594e537` (the lesson model given only "Felix", with no mark, and taught the pair Felix, Felix three times) the returned answer was the empty string before and after, 0 of 4, with no meaning in the answer and no word from the output walk, while the reading's own reconstruction said "Felix". Not yet measured by Claude. A word alone concludes no thought, so the answer has nothing to say. |
| 1.75, two words taken in order and said back | Failed. After the reading lesson, 0 of 6 two-word sentences are read in order in each of three starts, though the first word is taken first in 6 of 6. Every word of a sentence is found (64 of 64), but not in order, and "is" is never taken. At `11b916f86c`, by the one reconstruction: 0, 6 and 6 of 6 untaught, and 0, 0 and 1 of 6 after 3 rounds of the words alone, when a reading leaves a word or the mark out of what it composes, and what is left out is not put back. |
| 2.25 to 5.00, with the words handed to the model in order | The curriculum's stages `reading_two`, `reading_four`, `identities`, `noun_verb` and `relations` pass when the model does not have to take the words itself. They do not count toward the age, since taking the words in order is the row that fails. |
| 6.00, exclusive or | Passed (the standing gate). |

So the model is uneven: it has learned things a child learns late (exclusive
or, relations between named things) while failing one a child learns before
two: saying one word back. The age reports the first failure, not the best
result.

In a toy outside the model, the method now being built reads untaught
lengths in order, stops at the end and counts to "4" exactly. That is the
toy's result and not the model's.

## How the tasks are taught

The tasks are taught in the chart's order, in these stages (Alec,
2026-11-22). A stage is left when its measure passes, not after a fixed
number of presentations.

1. *The echo test* (age 1.00): a word is given, and the same word is the
   answer. It is the stage being built now.
2. *A sequence of two words, with no understanding* (age 1.50): two words are
   given and said back in order.
3. *Reading* (age 1.75 and on): a sentence of two words, then longer ones,
   taken in order and said back. Reading comes before counting.
4. *The counting test* (age 3.00 and on): one counting word for each word of
   a sentence.
5. *Singing* (age 3.50): a counted sequence with a number left out is said
   whole. It is still to be talked through.

The grammar is taught with reading, one rule at a time: a rule is available
to the model only once it has been taught.

Counting needs internal representations: what is said is not in the input, so
it has to be something the model holds. The echo test is the first place
where the answer is said from what the model holds of a word, and that is why
it comes first.

Reading and counting are two outputs over the same input sentences, on the
same model and the same weights. All content of the input is a word for both:
a closing mark is read, said back and counted as a word is. No concept's activation is set from outside
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
