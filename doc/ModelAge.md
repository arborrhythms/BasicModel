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

## The chart

Each row is a task with the age at which it is counted as learned. The ages
follow, roughly, the age at which a child learns the same thing. A model's age
is the greatest age in the chart with every row at or below it passed.

A model here takes the words of a sentence in order and says them back. That
is nearer to a child hearing and repeating than to a child reading print, and
the ages are set accordingly.

| Age (years) | Task that must be learned | How the assessment measures it |
| ---: | --- | --- |
| 0.70 | A word is kept when it is out of the look (object permanence) | the ladder's rung "one word" |
| 1.00 | A word is said back | each word alone as the input, the word as the output |
| 1.50 | Twenty words are said back | the rung "twenty words" |
| 1.75 | Two words are taken in order and said back | two-word sentences, read in order and stopped at the end |
| 2.00 | A thing is named | "This is Felix."; the rung "Felix" |
| 2.25 | A noun and a verb | the rung "Felix ran" |
| 2.50 | A sentence of three to five words is taken in order and stopped | the lesson's sentences, and sentences held out |
| 2.75 | A determiner | the rung "the cat ran" |
| 3.00 | The counting words are said in order, to "4" | one counting word for each word of a sentence |
| 3.25 | An adjective; a "what" question is answered | the rung "the fast cat ran"; "what is cat ?" |
| 3.50 | A pronoun | the rung "he ran" |
| 4.00 | Each thing is counted once, to "9" | counting, with each word looked at once |
| 4.25 | An adverb | the rungs "Felix ran quickly" and "the fast cat ran quickly" |
| 4.50 | Relations | "cat is on mat"; "x is bigger than y" |

## How the tasks are taught

The tasks are taught in the chart's order. Each word is taught alone until
every word is said back. Then sentences are read over those words, and their
words are counted. Then the grammar is taught, one rule at a time. A stage is
left when its measure passes, not after a fixed number of presentations.

Reading and counting are two outputs over the same input sentences, on the
same model and the same weights. No concept's activation is set from outside
to choose a task: every activation comes from the input or from the output.

## Status

The chart is a first list (2026-11-22). Its ages are estimates and have not
been checked against the developmental literature. The assessment today
measures the nine rungs from "one word" to "the fast cat ran quickly"; the
rows for a word said back, reading in order and counting are being built in
item 1.7, and the `age` parameter is not yet in the model or its XML.
