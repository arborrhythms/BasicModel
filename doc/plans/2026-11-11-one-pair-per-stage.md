# One pair across all stages, one reader, and omission

2026-11-11. The specification of the round that follows item 2.0's round 7.
It came out of item 1's baseline ([record](2026-11-11-item1-baseline.md) §8,
which holds the measurements) and of Alec's decisions there. Claude wrote
it; Codex builds it. The names below were read on round 7's working tree on
2026-11-11 and checked again on its returned source `fde035b1464d`.
The file keeps the name of the arrangement first specified, one pair for
each stage, which Alec replaced the same day (§1.1).

## 1. Decided (Alec, 2026-11-11)

1. **Whole paths, compared once at the end.** "I'm pretty sure that we only
   need one grammatical pair across all compose, thinking, and generation."
   And, against a keep at the end of every stage: "The compose exploration
   might be lost at the initial stage because the keeping acts like a local
   minimum on the error surface, whereas if we have one departure per
   sentence our minimum is evaluated only at final stage over all stages.
   It will probably be slower, but more thorough. If we wish to increase
   the credit, we might introduce a second explore path." Decided: "Yes,
   default one explore".
2. **One reader.** "I am concerned that "XOR_grammar's twin" is a workaround
   that was added because something in the architecture failed to learn
   appropriately during testing." Question tool: "Retire it everywhere
   (Recommended)". The record is in baseline §8.10: the twin was added on
   2026-10-06 for an affine reader that could not read one of the two
   compositions, and that cause was withdrawn on 2026-10-07.
3. **Output prices the understanding through the thought.** "we want both
   reconstruction and output to influence a single understanding; so if I
   understand correctly, we need to price it through thought (this is also
   why we remove the projection of the output gradient onto the
   reconstruction gradient, i.e. to prevent gradient fighting)."
4. **The answer trains on both compositions.** "quick take: why wouldnt the
   answer train on both cmopositions?" On Claude's reply that no good
   reason remained: "Yes, both". Told afterwards of the rule's recorded
   cost, which that reply had left out: "Keep both". The rule and the cost
   are in §3.6.
5. **The builder.** "Codex, after round 7".
6. **Silence is the answer where none was asked.** "Perhaps output in cases
   where there is no supervised answer should supervise silence (i.e.
   penalize any answer). That would seem to teach both answering questions
   and not talking when no question was asked." Decided: "Yes, count
   silence".
7. **Output follows from the thought, never from the understanding.** "all
   that needs to happen is that thinking or the intuitive step reduce the
   understanding to zero (i.e. after comprehension). Output is never a
   direct consequence of understanding the question; it is a consequence of
   the answer produced by thinking about the question."
8. **An adaptive loss over the input, by omission.** "My guess is that we
   will want to create an adaptive loss function over the input to teach
   the model before item 0, so minimally a maskRate for both forms and
   words. It ought to be able to drop both percepts and concepts. Ensuring
   a high, long-term reconstruction rate (e.g. 90%) would allow learning
   while at the same time creating an raising bar for prediction." And:
   "The kind should be omitted using an out-of-band signal, not corrupted,
   because the best model reproduces corruption on seeing corruption."
   Answers of the same day: the rate held is "Of everything" in the input;
   what was hidden is restored by "Reconstruction only"; a dropped word
   loses "Its whole form"; the omission is "On by default, but you can
   override to get existing tests to pass"; and it is built "Inside the
   whole-path round". The rules are §3.11.
9. **Reconstruction uses the higher-order understanding; no next-sentence
   judge.** On Claude's answer that the record's way to train System 2 on
   monologue is the next sentence: "No, I do not think predictive
   processing is a good solution; reconstruction is better... So to train
   system2 on that, we might do reconstruction as augmented by compose()
   and generate() without doing any thinking()? So call generate() on the
   LTM0 idea to alter the order-0 conceptual contents before doing
   reconstruction? Does that make sense? otherwise we are not using our
   higher-order understanding in order to reconstruct, which does not feel
   psychologically valid." That is the order-zero reconstruction as it has
   been since his correction of 2026-11-10 (loop spec §9), and the second
   stage of every path here (§3.1).
10. **The XOR test tests this design.** Asked whether a numeric head should
   keep reading the composed state, as the XOR gate needs: "We have
   converged on a model design at this point: so the purpose of the xor
   test is to test that design on the xor problem, not to make a new model
   architecture". And, asked whether a red XOR test stops the round: "Yes,
   it stops the round. A continuous output (and continuous input) can be
   treated as alternate modalities. Using a linear output head on top of
   order 0 concepts instead of verbal reconstruction can be a model option.
   So with that caveat, you can keep xor close to its current form." And,
   shown that a linear head on order-zero concepts cannot read XOR in this
   fixture (§3.12): "Sorry; when I said order-0, I meant no grammar
   (higher-order concepts are in fact required for xor)." §3.12.

### 1.1 The arrangement this replaces

On Claude's recommendation Alec first chose "One pair per stage
(Recommended)": compose, the reconstruction read-out, the thought and the
output read-out each with its own explore and its own keep. He then asked:
"Can I presume that the question about training stages only on the "kept
(exploit/explore ?)" means that there are exploit/explore paths for all
stages, and we decide which is kept at the end of every stage? If so,
should we consider only which path to keep across all stages?" Claude
answered that a whole-path keep would be right only "if the stages' choices
interacted, or if an explore could not reuse its trunk", and that neither
held. Alec debated it (decision 1's second quotation), and he was right.
Two errors of Claude's:

- *The stages do interact.* They are a chain. The reconstruction read-out's
  words become the order-zero field (`OutputOccurrences.generated`,
  installed by `SentenceField.publish`), and the thought reads that field
  (`_selected_thought_memory`) and writes into it
  (`Intuition.write_knowing`). This is Alec's two inverses of 2026-11-10:
  with symbols on, the grammar inverse determines the order-zero layer.
  The first specification kept the reconstruction walk on $R$ alone, which
  is a keep at a local minimum one stage after the one Alec named, and it
  broke its own rule that the price is the output.
- *The cost.* With each explore's whole tail run again, a keep for each
  stage costs about 8 walks and 4 thought episodes a sentence, not the 5 to
  7 walks first stated.

## 2. Today

A pair is `WalkTrials.generation_pair`: a greedy walk, one departure, an
explore walk, the cheaper kept. A lesson sentence makes eleven (baseline
§2.5, §8.2). Nine of them are prices. Inside each compose trial
`BasicModel._sentence_path_cost` runs a `'decoder'` pair for $R$ and two
`'output'` pairs for $A$, the presented reader's and the twin's, and
`ThoughtAnswer.scorer` runs two more for each thought trial. Measured on 20
lesson sentences (baseline §8):

- The answer pairs inside a trial unfold the sentence's meaning
  (`_sentence_reader_error`). What the sentence then says is unfolded at the
  batch end from `resolveAnswer`'s idea, another idea in 18 of 20.
- That batch-end pair trains nothing ("The batch-end read is reporting
  only"). The reader is trained on read-outs that are not the output.
- 39 of 40 thought trials are priced by reading the kept composition again,
  with draws of their own. In 4 of 19 sentences the explore thought was kept
  on a difference that came from that draw.
- The twin reads the presented reader's roles and differs from it by
  `question_conditioners`, which trains in the twin and not in the reader it
  copies.

The reading already has the rule this round extends: one departure a
sentence, drawn by `SentenceCredit.departure` over the attention, compose,
candidate and bind walks, the explore reading run greedily from it, and one
comparison on $R + A$.

## 3. The sentence after the round

### 3.1 Paths

Alec, 2026-11-11: "whole path is an integration of the reconstruction and
output paths, both of which share the input path but do not share the same
reverse (or at least call the same reverse mechanism from two different
conceptual content spaces)." The reverse mechanism is the generate walk
with the occurrence writer and `reverse`. Reconstruction calls it on the
composed understanding; output calls it on the thought's answer.

A path is one complete derivation of a sentence, a chain of four stages:

1. the reading: the attention, compose, candidate and bind walks, which end
   in a composition;
2. the reconstruction read-out of the composition
   (`SentenceField.sentence_generated`), whose words become the order-zero
   field;
3. the closing thought on that field (`ThoughtClosing.close`);
4. the output read-out of the thought's answer: the answer's idea by
   `_resolve_answer` with the episode as `selected_thoughts`, then one
   `reverseOutput` walk.

*The output walk's operand is what the thought concluded, and nothing
else* (decision 7). `_resolve_answer` already forms it so for a row whose
episode it is given: the concluded answer (`Output.thought_answer_meanings`,
live through `Intuition.live_meaning` where an intuition was selected) when
the episode is accepted, and zero when it is not. Its other default, the
field's own meaning as the answer of a row with no episode (source
`identity`), is not used at a sentence close, and
`_sentence_reader_error`'s own derivation from `observation['meanings']`
goes. So does `_answer_occupancy`'s fallback to the field's own role mask,
which today gives an unanswered row occupied roles of value zero: an
unanswered row has no occupied role. A path whose thought concluded nothing
unfolds nothing, and the sentence says nothing.

*A continuous output is a modality of this design* (decision 10). Where
the verbal output unfolds the answer through the generate grammar into
words, a linear head may read the concepts, higher orders included, with no
grammar: a model option, not another architecture. A path then has no
output walk, and its $A$ is the head's cost on the path's own sentence
state. `XOR_grammar.xml` is the configuration that uses it (§3.12).

A path's cost is $C = R + A$: $R$ as the registry totals reconstruction
now, on the path's reconstruction walk, and $A$ the output walk's cost
against the row's target (`_output_target_cost`; the head's cost with a
numeric head). On data that supervises
output the target is the supplied answer where one was asked and silence
otherwise, on every row (decision 6): `_answer_trial_cost`'s `asked_only`
form, which set $A$ to zero on a row that was not asked, goes.

Silence is an empty operand, not a walk taught to stop. The two read-outs
are one generate walk (2.0 spec §2), and reconstruction trains it to unfold
the very state that a penalty on speaking would train it to leave folded.

The *exploit path* takes the greedy choice everywhere. An *explore path* is
the exploit path with one departure: at one eligible round of one walk the
alternative action is taken, and every later choice is greedy again.
`architecture.explorePaths` (new; default 1) is the number of explore paths
a sentence makes in training.

### 3.2 The departure

`SentenceCredit.departure` draws it as now: a walk uniformly among the
walks that have an eligible round, a round uniformly within that walk, the
action from the chooser's alternative distribution. It draws over seven
walks instead of four: the reconstruction walk, the thought and the output
walk join attention, compose, candidates and binds. The exploit path runs
first and supplies every walk's eligible rounds. Explore paths of one
sentence take distinct departures, and each path's proposal correction is
that of the draw actually made. A configuration without an output walk has
none in the draw. Paths are kept row by row, as `sentence_pair` selects
rows now.

Where the data supervises no output (`has_supervised_outputs` false) there
is no $A$. The thought and the output walk are then not in the draw; the
paths are carried to their reconstruction walks and compared on $R$, and
the kept path runs its thought as now. §6 has Alec's direction for such
text.

### 3.3 The trunk

An explore path runs nothing that lies before its departure. A departure in
the output walk makes one more output walk and no second reading,
reconstruction walk or thought. From its departure a path is computed to
the end: a departure in the reading computes the composition, the
reconstruction walk, the thought and the output walk again; one in the
reconstruction walk computes that walk, its field, the thought and the
output walk; one in the thought computes the episode from its fork and the
output walk. Inside the departed walk the actions before the departure
repeat the exploit walk's, as `exploit_actions` makes them now.

### 3.4 Every path ends before the write

`ThoughtClosing.close` leaves `_commit_sentence`: a path's thought runs
before the sentence's row is written. What the closing thought takes from
the commit today, and where a path's thought takes it from:

- the admitted clause (`finish_clause`, `frozen_values`): the path's clause
  in `observation['clauses']`, whose scored values admission already
  preserves;
- the closing image (`ClosingImage.form` on
  `disc.last_expectation_comparison`): the path's prediction, which "owns
  each trial's estimate";
- the row's canonical references (`closing_question`'s `committed` branch,
  `store.read_reference_bindings`): the path's reference plan
  (`reference_close.preview`, `prepare`);
- the row's own occurrence, kept out of its evidence (`_query_exclude`):
  nothing is written yet, so nothing is excluded;
- the timeline event (`TimelineRead.request`): to be placed by Codex.

A thought episode's effects are restored after it
(`WalkTrials.ThoughtTrialState`, which round 7 extends to both clocks).
`_commit_sentence` then writes the kept path's row and its thought's
records, in today's order and with today's ticks. It runs no thought and
chooses nothing again, as it already does for a bind ("Reattach the scored
decisions to the admitted copies"). If an operation of the thought reads
the sentence's own row, the return names it. The alternative there is a
write that a rejected path undoes; which was taken is reported.

### 3.5 One keep, and the credit

`SentenceCredit.comparison` compares the paths once, at the end, on
$R + A$. The cheapest path is kept. An explore path is kept only if it is
strictly cheaper than the exploit path. The kept path is what is written
and what the sentence says.

Each explore path's departure is credited by `score_function` with its own
difference from the exploit path on the whole $R + A$, at the chooser that
owns the departed walk: the reading's walks as now, the generate policy for
either read-out (`WalkTrials.generate_credit`'s form), the thought's
chooser for the thought (`ThoughtCredit.complete`'s form). Every credit's
proposal correction counts the walks of the draw, as
`draw['walk_count'] * draw['walk_rounds']` does for the reading now.
Expectation trains its predictors and credits no departure, as now.

### 3.6 What trains

The sentence's one update (`_sentence_train_step(surrogate(...))`):

- *Reconstruction and expectation* train on every path, as they train on
  both compose trials now (`sentence_pair`). Each path's registry is scaled
  by $2/(1+K)$ for $K$ explore paths, so the total is today's and does not
  depend on `explorePaths`.
- *The answer* trains from the kept path's output walk. It shares that
  weight equally with every explore path that departed in compose and,
  where there is one, with the exploit path. With one explore path this is
  `SentenceCredit.reader_weights`' rule with the draw, which the twin had:
  half from each path at a compose departure, whichever is kept. For every
  other departure the kept path trains alone: a departure in
  attention, among the candidates or in a bind is another reading of the
  input, to which the supplied answer does not apply (operators update §17,
  round 2b: "The reader was trained on misreadings"), and a path that
  departed in the thought or a read-out is not asked to give the answer
  from a conclusion it did not keep.
- No other answer cost exists. This round cuts nothing from the graphs of
  the walks that train: their cotangent reaches the understanding through
  whatever of the thought carries a gradient, and 2.0 spec §1 projects it
  against reconstruction's on the units both write.

*Why both compositions.* A reader trained only on what is kept reads the
kept composition better because it is kept. That is the bias the twin was
added to offset, and with one reader it returns unless the reader sees
both.

*The rule's recorded cost.* It is the rule of rounds 2c and 2d. The
rejected composition's share does not decay as the chooser commits, and in
round 2d it cost the reader its fit (class 2 of 10; operators update §21),
on roots that interfered. On the gate the roots have not interfered since
2026-10-07 (§27). Elsewhere this is not measured: §4's measurement reads
the reader's cost on the kept path for it.

### 3.7 One reader

`AnswerComparison` and `comparison_reader` go from every configuration,
`XOR_grammar.xml` included. $A$ is the one reader's cost.

### 3.8 The batch end reports

Where a sentence close made the output, `ask`'s `reverseOutput` makes no
pair and takes no departure. It replays the kept path's output walk
(`kept_actions`) and asserts that the replay is the sentence's output.
Where no sentence close makes the output, the batch end's read-out is the
whole derivation, and the same mechanism applies there with the output walk
as its only walk.

### 3.9 What goes

No shim and no second mechanism. One mechanism remains for every choice
(2.0 spec §3): the reading's pair, extended.

- `WalkTrials.generation_pair`, `output_pair` and `thought_pair` with their
  keeps (`kept_walk`), and the draw table: `_sentence_generation_draws`,
  `_sentence_generation_cursors`, `paired_departure`, `paired_action_draw`;
- `ThoughtAnswer.scorer` and `train_readers`: a thought is priced by its
  path. `bound_view` stays where `_commit_sentence` installs the kept
  episode's meaning as the sentence's record, and where a numeric head
  reads a path's state;
- `bin/AnswerComparison.py`; `comparison_reader`;
  `_sentence_comparison_registry`, `_sentence_comparison_reader_weights`,
  `_sentence_comparison_reader_updates`; the `thought_comparison.` terms;
  the state-dict handling of `comparison_reader.weights.*`;
- `SentenceCredit.reader_weights` without a draw (the presented reader's
  kept-only form);
- `_sentence_reader_error`'s `AnswerDerivation` built from
  `observation['meanings']`, and at a sentence close `_resolve_answer`'s
  `identity` default (the field's own meaning as the answer) with
  `_answer_occupancy`'s fallback to the field's role mask;
- `_answer_trial_cost`'s `asked_only` form.

### 3.10 Counts

On a row of data that supervises output the exploit path makes 2 walks and
1 thought episode. An
explore path adds 2 walks and 1 episode when it departs in the reading or
the reconstruction walk, at most 1 walk and 1 episode when it departs in
the thought, and 1 walk when it departs in the output walk. With one
explore path: at most 4 walks and 2 episodes a sentence, against 22 and 2;
about 3.7 and 1.9 if every walk has an eligible round. Each further explore
path adds at most 2 walks and 1 episode. Where the data supervises no
output: one reconstruction walk for each path, and the kept path's thought.
With a numeric head there is no output walk, and a path makes one walk.
One chooser is credited for each explore path.

### 3.11 Omission: an adaptive loss over the input

*What exists.* On the sentence path nothing is hidden today
(`gaussian_window_word` "is NOT target-hiding"; baseline §8.13). The one
hiding mechanism, `create_ir_mask`, runs on the path without the per-word
loop: it zeroes the WHAT slice of percept slots drawn at `maskRate` and
keeps their WHERE and WHEN ("MASK is the all-zeros vector").

*What is hidden.* In training, once for a sentence and before it is
perceived, some of its content is omitted at two sizes:

- *a word:* each word independently, at the word rate $m_w$, loses every
  one of its percept slots;
- *a form:* each percept slot of a word that was not dropped, independently,
  at the form rate $m_f$.

An omitted slot has zero content and keeps its place and its time, as
`create_ir_mask` writes it. No learned mask code is used. Padding is never
drawn, at least one word of a sentence is kept whole (no slot of it
omitted), and nothing is omitted outside training. Every path of the sentence reads the same
omission (§3.1: a sentence is perceived once).

*Out of band, and no leak.* Nothing the model reads carries an omitted
content: not the word slab that `perceive` is given, not
`perceptualSpace._forward_input` (the native indices, spans and seed
events), not the word's resolved rows and concept ids, not its reference
evidence. Two sentences that differ only in an omitted content give the
same understanding, the same reconstruction, the same expectation state and
the same writes; only their costs differ.

*What restores, and against what.* Reconstruction, through the
understanding and without the thought (decisions 8 and 9). The order-zero
reconstruction is what restores: compose builds the idea from the degraded
input, the generate walk unfolds that idea into the order-zero contents
(`SentenceField.sentence_generated`, `OutputOccurrences.generated`), and
the system-1 inverse renders them (`reverse`; the term
`reconstruction_order_zero`). Its target is the input as it was before the
omission, at every place, kept as loss-side data
(`SentenceField.native_target`, the staged input `_staged_concepts_in`,
`record.word_values`): "targets, never decoder inputs". A reconstruction of
zero at an omitted place is charged like any other miss. So omission trains
compose and generate by reconstruction's gradient and credits their
choosers by path; the thought is outside it.

The input reconstruction (`reconstruction_reverse`) inverts the observed
field without the grammar and has nothing to restore from: it is costed
where content was observed and not at omitted places. Asked whether that
was right, Alec: "we want to compare the order-zero reconstruction as
augmented by higher-order-understanding but without thinking". Claude's
reading: what is compared with the original at an omitted place is the
order-zero reconstruction, through compose and generate; the grammar-free
one is not. Expectation's observation errors at an
omitted place take the original as target only where that target does not
also become state the model reads; otherwise the place is left out. In no
cost is the zero content a target. The output and its cost are unchanged:
what the understanding lacks, the thought and the output do not get.

The walk starts from the path's composed end state, as now. That is the
idea the kept path then writes, except where the close fuses a clause
(baseline §8.4).

*The rates adapt to hold the reconstruction rate.* For each size the model
keeps two long-term accuracies of the kept path's order-zero
reconstruction: its bytes, decoded as `SentenceField.reconstruct` decodes
the input reconstruction's now, compared with the staged input before the
omission. $v$ is on what was visible and $a$ on what was omitted. A
slot is right when its bytes are; a word is right when all of its bytes
are. Forms are counted over slots of words that were kept, words over
words. With the target rate $r$ (`architecture.omissionTarget`, default
.9),

$$m = \operatorname{clip}\!\left(\frac{v - r}{v - a},\ 0,\ m_{\max}\right),$$

with $m = 0$ while $v \le r$ and $m = m_{\max}$ once $a \ge r$
(`architecture.omissionLimit`, default .5). This is the share that makes
the reconstruction rate of everything equal $r$: nothing is hidden until
the model reconstructs what it sees at the target rate; with $v = 1$ a
tenth is hidden while nothing is restored, a fifth at $a = .5$, a half at
$a = .8$. The averages are exponential over
`architecture.omissionHorizon` sentences (default 1,024), $a$ counts as
zero until something of that size has been omitted, and the two rates and
four averages are saved with the model and reported for each epoch. The
rate itself is the bar: it says how much of its input the model can do
without.

*On by default.* `data/model.xml` sets the target, so every
sentence-training configuration omits. An existing test or configuration
that omission breaks may set `omissionTarget` to 1, which hides nothing;
the return lists every such override with its reason. `XOR_grammar.xml` and
the other XOR proofs set it to 1: with two words, an omitted operand leaves
their answer undefined. `maskRate` and `create_ir_mask` on the path without the per-word
loop are unchanged.

### 3.12 The XOR test on this design

Decision 10. The grammar XOR gate tests this design in its continuous
output modality, and stays close to its present form.

- *The head.* A linear head on the concepts with no generate grammar:
  `answerSynthesis` stays false in `XOR_grammar.xml`, and `_forward_head`
  reads each path's composed sentence state (the root and end slots, which
  are of higher order) as it does now. "higher-order concepts are in fact
  required for xor" (Alec).
- *What stays.* The four two-word sentences and their numeric classes, the
  compose grammar and widths, and the certificates
  `TestXorGrammarLearnsXor::test_xor_class_accuracy` (behind `RUN_TRAINED`
  since item 2.3) and `TestXorGrammarReconstruction`. No thought, rule or
  second reader is added for the fixture.
- *What the round changes there.* The twin goes as everywhere (§3.7), so
  the one head reads the kept path and trains by §3.6's rule. A path makes
  one walk, the reconstruction walk, and is compared on $R$ plus the head's
  cost.
- *It stops the round.* Ten unseeded trainings; five or more must pass the
  class certificate. With fewer the round is not offered: the failing runs
  are diagnosed (§5) and the cause repaired at its producer, or the
  diagnosis is returned if the repair needs a decision. The count is
  Claude's: the certificate passes about two initializations in three
  today (2 of 3 and 4 of 6 on record), so five of ten is missed by a sound
  round about one time in thirteen, and a rate of one in five reaches it
  about one time in thirty. Seven of ten would stop a sound round four
  times in ten.
- Omission stays off for this fixture (`omissionTarget` 1): with two words,
  an omitted operand leaves the answer undefined.

*The thought in this fixture* (Claude's reading). Decision 7 is about what
the output walk unfolds. A continuous head has no walk, and this fixture
has no thought that concludes: `XOR_grammar.xml` owns its grammar and
declares a `<compose>` section only, where the lesson configurations name a
grammar file with thought entries (`complete.grammar`), and in 48 of 48
closes its thought returned the sentence's own meaning with no producing
operation and no evidence kind. So the head reads the understanding as
composed. What a continuous head reads where a thought does conclude is
not decided (§6).

*Why not order zero* (measured on round 7's working tree, one unseeded
training that converged; scripts `xor_thought_check.py` and
`xor_order_zero_b.py`, not in a receipt). Claude first read "on top of
order 0 concepts" literally. The field that `sentence_generated` unfolds
from the composed end state held exactly the two words' concepts at
amplitude 1, all of order zero, in every sentence at each of thirteen
checkpoints from the first batch to the last (norm $\sqrt{2}$). Write the
sentences $x_1 = h + w$, $x_2 = h + t$, $x_3 = l + w$, $x_4 = l + t$: every
affine read $f$ has $f(x_1) - f(x_2) - f(x_3) + f(x_4) = 0$, and the
classes $0, 1, 1, 0$ need $-2$. The field departs from that sum only in the
order of its two words (the walk gives two of the sentences transposed,
which the reconstruction gate allows; $|x_1 - x_2 - x_3 + x_4|$ is exactly
2 for that reason at every checkpoint), and a head that read XOR there
would read the layout. On the composed root the same quantity is 0 at the
first batch and between 1.37 and 1.98 afterwards (norm 8.2): the
combination is in the higher-order concept, as Alec says.

## 4. Certificates and the one measurement

*Certificates*, in the fast suite:

1. Counts: with `explorePaths` 1 a sentence makes at most 4
   `BasicModel.generate` calls and 2 thought episodes, with 2 at most 6 and
   3, whether or not its answer was asked. A departure in the output walk
   makes exactly one more walk and no second reading, reconstruction walk
   or thought.
2. An explore path equals the exploit path before its departure and
   differs there by the one drawn action.
3. The keep is the cheapest path on $R + A$, and a tie keeps the exploit
   path. A constructed departure in the reconstruction walk that changes
   the thought's answer is kept or rejected, and credited, on $R + A$ and
   not on $R$.
4. The kept path is what is written and said: `_commit_sentence` runs no
   thought and chooses nothing, the batch-end output is the kept path's
   output walk, and the batch end draws nothing.
5. From the kept path's output walk a nonzero gradient reaches a parameter
   that produces the understanding. Where the departure lay in compose both
   paths' output walks contribute at equal weight; for any other departure
   no other answer cost contributes.
6. No `comparison_reader`: no attribute, parameter or state-dict key. A
   checkpoint that carries `comparison_reader.weights.*` loads with those
   keys dropped and named.
7. Silence and the operand. No path unfolds the sentence's own meaning as
   its answer: wherever the path's thought concluded nothing the operand
   has no occupied role, the walk emits nothing and the sentence says
   nothing. On a statement such a path pays no silence cost, a path whose
   thought concluded an answer pays it, and the comparison counts it.
8. Omission is out of band: an omitted slot has zero content and its own
   place and time; padding is never drawn, a sentence keeps one whole word,
   evaluation omits nothing, and every path of a sentence reads the same
   omission.
9. No leak: two sentences that differ only in an omitted word are the same
   to the model through the close (understanding, reconstruction,
   expectation state, writes) and differ in cost. In the order-zero
   reconstruction a zero at an omitted place is charged and the original
   there costs nothing; the input reconstruction is not costed there.
10. The rate rule on its cases: $m = 0$ while $v \le r$; $m = 1 - r$ at
   $v = 1$, $a = 0$; $m = m_{\max}$ at $a \ge r$; and the rates and
   averages survive a checkpoint.
11. `omissionTarget` 1 hides nothing, and a seeded run with it is identical
   to the same run with the omission code removed from the path.
12. The ports, each with its old and new text: every test that names
   `comparison_reader`, `AnswerComparison`, `generation_pair`,
   `output_pair`, `thought_pair` or the draw table. For the twin, on round
   7's tree: `test_operators_round2c.py`, `test_operators_round2e.py`,
   `test_generation_selection_backward.py`,
   `test_supplied_answer_training.py`, `test_sweep_thought_save.py`,
   `test_identity_experiment.py`, `identity_experiment.py`,
   `operators_run_audit.py` and `observer_probe.py`.

*One measurement*, on the frozen candidate, one thread a process:

- *The XOR test* (§3.12). Ten unseeded trainings of `data/XOR_grammar.xml`
  with `explorePaths` 1 and `omissionTarget` 1, without the twin. Five or
  more pass the class certificate, or the round stops. For each run: the
  four answers and the kept composition of each sentence.
- The standing XOR proofs once each, the sum control at $\tfrac14$.
- *The lessons, in three arms.* Round 7's `single-0.0-float32` lesson arm
  (the default configuration) at `explorePaths` 1 and at 2, both with
  omission on, and at `explorePaths` 1 with `omissionTarget` 1, against
  round 7's receipt of that arm:
  acquisition and answers; for each sentence its walks, thought episodes
  and seconds; the credited departures counted by walk; and for each epoch
  the two omission rates and the four accuracies. This reads Alec's
  "slower, but more thorough", what a second explore path buys, and whether
  the bar rises at all on a corpus this small.
- *The cost of 3.6's answer rule.* In the XOR trainings and the lesson
  arms: the reader's cost on the kept path over training, and the count of
  sentences on which a rejected composition trained.
- One complete sweep.

*The toy, run for the reader's rules.* The credit-loop toy of 2026-10-06
(two compositions of four sentences, an affine reader, a one-logit chooser
credited by the answer difference) with three reader rules, 40 seeds, 400
epochs. The second composition is either readable by an affine map
(operators update §27) or not (the regime of 2026-10-06). Settled means the
reader's error on the kept composition is below .05 at the end; the table
is at 400 epochs.

| second composition | one reader on the kept composition | one reader on both | twin |
|---|---|---|---|
| readable | 40 of 40 | 36 of 40 | 40 of 40 |
| unreadable | 33 of 40 | 37 of 40 | 36 of 40 |

Price noise of .05 and an Adam step on the chooser change no cell by more
than two. At 1,600 epochs every rule settles in 40 of 40 seeds in both
regimes (corrected 2026-11-11: the rules differ in how fast the loop
settles, not in where it ends). Where both compositions are readable, one
reader trained on what is kept settles soonest and the twin adds nothing:
the reader fits what is kept, so what is kept reads better. Where a
composition cannot be read and the chooser starts on it, one reader is
still there at 400 epochs in 5 to 7 seeds; if the grammar XOR certificate
is slow or fails without the twin, the toy points at roots the reader
cannot read, not at the judge. The toy's roots are fixed, so it cannot show
the answer's gradient making a composition readable, and under no rule does
it reproduce finding 14's unsettled runs. The script and the full table are
a local receipt, `doc/benchmarks/2026-11-11-one-reader-toy/`.

*The toy on 3.6's answer rule* (a second run, 40 seeds, no noise). Where both
compositions are readable, one reader on both settles later than one reader
on the kept composition (median epoch 82 against 52) and keeps a small
error on the kept composition (worst seed .037 after 1,600 epochs, against
0). Where one is unreadable the two rules settle alike (35 and 36 of 40 at
400 epochs, 40 and 40 at 1,600); the first run's lead for "both" there did
not repeat. So the toy shows the rule's cost and not its benefit. The
benefit on record is the model's, in rounds 2c and 2d: the chooser left a
wrong start only when the reader trained on both. A guard was tried and
rejected in the toy: projecting the rejected composition's answer gradient
against the kept one's, as 2.0 spec §1 does for output against
reconstruction, keeps the fit where both are readable (worst .002) and
strands 5 of 40 seeds on an unreadable start.

The toy has one departure and no chain of stages: it bears on the reader's
rules of §3.6 and §3.7, not on whole paths against a keep for each stage. Its
affine reader is the numeric gate's, which decision 10 keeps; it is the
record of why the twin is not needed there.

## 5. Two findings to diagnose in the round

1. *The presented reader's question conditioner does not train.* After 76
   lesson sentences the `question_conditioners` weight is exactly zero and
   no gradient has reached it, while the twin's copy trained every sentence
   (baseline §8.3, §8.7). With one reader it is the only conditioner. Find
   the cut and repair it so that the conditioner trains from the kept
   path's output walk, or report why it cannot.
2. *The XOR test.* The gate does not converge in about a third of
   initializations today (item 2.3 spec §33.1, finding 14; never
   diagnosed). Any run of §4's ten that does not pass is traced once by its
   recorded `torch.initial_seed()`: for each epoch the kept path's
   composition for each sentence, the head's cost and the loss. The stage
   that does not learn is reported.

## 6. Not in the round

- *The conditioner's and the attention read's place.* The question
  conditioner and `_generation_read` change the state between the thought
  and the output walk, and no departure explores them (baseline §8.4). The
  round leaves them where `_materialize_answer_idea` and `generate` apply
  them. Whether they belong to the thought, or go, is open for Alec.
- *The other XOR proofs.* `test_mm_xor.py` (the native convergence gate),
  `test_grounded_xor.py` and the rest run configurations without the
  per-word loop. Decision 10 was given for the grammar gate; they are not
  changed. Continuous input as a modality is named in decision 10 and not
  specified here; it is future work, with XOR's conversion to a System 1
  model on numeric input
  ([FutureWork](../FutureWork.md#numeric-input-as-a-modality-xor-as-a-system-1-model-and-one-model-class-alec-2026-11-11-to-specify)).
- *A continuous head after a thought.* Whether a linear head reads the
  thought's conclusion or the understanding, where a configuration has
  both a head and thought entries, is not decided; no gate of this round
  has both. Other configurations with a numeric head on the per-word loop
  are not changed and are listed in the return.
- *More than one departure on a path.* A path that needs two departures to
  pay is not found by paths of one. Not built.
- *A next-sentence judge of the thought.* Considered and not to be built.
  Alec asked how the monologue corpus trains System 2; Claude pointed to
  the record's next-sentence prediction (2026-09-15; 2.0 spec §8); Alec:
  "No, I do not think predictive processing is a good solution;
  reconstruction is better" (decision 9). On text that supervises no
  output, System 2 is trained through the order-zero reconstruction, which
  runs compose and generate; the thought is not trained there, and nothing
  judges it against the next sentence.
- *Other kinds of omission.* A concept withheld from a word whose form is
  seen (the unfamiliar word), a constituent, and corruption in place of
  omission are not built. Alec chose a word's whole form and omission.
- *A conclusion that is kept and not said.* On a statement an accepted
  conclusion of the thought becomes the answer and then pays for being
  said. Whether a thought may keep a conclusion without saying it is open.
- *What the intuitive step alone may say.* `ThoughtAnswer.accepted` does
  not accept evidence of kind `intuition`, so an intuition by itself leaves
  the operand zero. Decision 7 names "thinking or the intuitive step"; the
  round does not change `accepted`.
- *`separateChooser` true.* The paths are the same. The certificates are
  for the default, where no update lies between a path's end and the
  commit.
- *Item 1's exact steps* (`Generative.MenuStore`; no second walk without a
  departure) are on branch `item1-generate-walk`. This round does not need
  them and does not include them.

## 7. Estimates that were not measured

The counts of §3.10 are arithmetic on the rules, not measurements. A keep
for each stage was estimated at about 8 walks and 4 thought episodes; one
explore path at about 3.7 and 1.9, two at about 5.4 and 2.7, each assuming
that every walk has an eligible round. With one explore path a chooser is
credited on about one sentence in seven. The lesson arms of §4 measure
these. Whether omission starts at all on the lessons is not known: it
waits for the visible reconstruction rate to pass the target.

## 8. Codex text: land round 7, then build this round (Claude, 2026-11-11; for Alec to paste; final)

Round 7's return is reviewed and accepted (2.0 spec §23.22), and by the 2.0 spec's §13.2 its landing is
folded into this build text: steps 1 to 3 land round 7, item 4 carries one wart of the review, and items 5
to 15 are the round. The names in §3 were checked again on round 7's returned tree `fde035b1464d`.

*Reviewed before the hand-off (Claude, 2026-11-11), four repairs.* (1) As first written, an empty operand
would have left `XOR_grammar.xml`'s affine head nothing to read, since its closing thought concludes nothing in
48 of 48 sentences. Claude proposed to keep the head outside decision 7; Alec made the continuous output a
modality of the design, a linear head on the concepts with no grammar, and the gate a stopping one (decision 10,
§3.12). (2) An unanswered row
must have no occupied role: `_answer_occupancy` falls back to the field's role mask today, so a zero operand
would still have been walked (§3.1). (3) The omission draws are independent, a word at the word rate and a slot
of a kept word at the form rate (§3.11). (4) The accuracies read the kept path's order-zero reconstruction, and
paths are kept row by row (§3.2, §3.11). Not checked: that today's walk on an operand with no occupied role
emits nothing; certificate 7 tests it.

~~~
Item 2.0: land round 7, then build the whole-path round. Round 7 is accepted (2.0 spec doc/plans/2026-11-03-gradient-projection.md §23.22). Steps 1 to 3 land it and change no code, data or test; the whole-path round starts from that landed main with a clean tree.

1. Check. The live tree's source hash by §20.33's method equals fde035b1464d97b49bdd15251bbb65135e0d72cea7f03ae4aff754b6844b485b, round 7's returned source. Since the return the handwritten documents changed by appends and by Claude's replacement of nineteen "working tree, not landed" markers with "accepted 2026-11-11". Run test/test_doc_links.py once on the live tree; it must pass. No make test: the one sweep is on da78077358c5 with the whole-file follow-ups of §23.20, and no source file changed since the return.

2. Commit, on main, every file round 7 changed, added or deleted, with the handwritten documents: the 2.0 spec with its reviews, todo.md, Architecture, FutureWork, GradientFlow, Params, Philosophy, Reasoning, STM, Training, the forgetting and accessible-mind specifications, doc/plans/2026-11-11-item1-baseline.md and doc/plans/2026-11-11-one-pair-per-stage.md. The receipts under doc/benchmarks/ stay local under the existing ignore rule; force nothing in. Commit message in the form of 16a6a79b81: subject "Land accepted item 2.0 round 7"; body: land the returned source fde035b1464d, measured and swept as da78077358c5, with the acceptance in section 23.22; what it carries (episodic memory on model time: one tick for a sentence and for each new thought idea on one ladder, both clocks restored at the epoch reset, ages for the learned readers, entries of episodicPercepts percepts beside LTM0 rows, episode anchors for relation-only closings, recall by time and reinstatement; the conceptual mixer on time differences; the six order repairs at the occurrence writer's contract; the tensor clock view); validation (the eight fits and four lesson arms, the one sweep with its whole-file follow-ups, the reviewer's seeded identity check of the clock view); and section 23.22's carried list. No model or assistant identifier anywhere in the message.

3. Record the landing: append one line to todo.md after its last 2.0 update and one line to the 2.0 spec after §23.22, "Landed as <hash>" with the date; commit as "Record item 2.0 round 7 landing"; then push main once. Report both hashes in your return. Then continue with the whole-path round at once, from this landed main, without waiting.

4. Carried from §23.22, from round 7's receipt and with no new run: name the parameter and the sentence of the largest optimizer gradient in each lesson arm (203 to 1,377, 413 to 1,095 and 4.26 to 7,069 against the accepted source), and say whether one producer explains the rise.

Whole-path round: one exploit path and explore paths compared once at the end, one reader, and omission over the input. Alec's decisions and the rules are in doc/plans/2026-11-11-one-pair-per-stage.md (§1 decisions, §3 rules with §3.11 for omission, §4 certificates and the one measurement, §5 findings, §6 what is not in the round); the measurements behind them are in doc/plans/2026-11-11-item1-baseline.md §8. It starts from the main that step 3 leaves. A failed test_convergence or shared-concept case stops the round; anything else is repaired at its producer under the recorded rules or recorded. No commit or push of this round.

5. Paths (§3.1). A path is one complete derivation of a sentence: the reading (attention, compose, candidates, binds); the reconstruction read-out (SentenceField.sentence_generated), whose words become the order-zero field; the closing thought on that field (ThoughtClosing.close); the output read-out of the thought's answer (_resolve_answer with the episode as selected_thoughts, then reverseOutput). The output walk's operand is what the thought concluded and nothing else: where it concluded nothing the operand has no occupied role and the walk emits nothing; it is never the field's own meaning (_resolve_answer's identity default and _answer_occupancy's fallback to the field's role mask are not used at a sentence close), and _sentence_reader_error's own derivation from observation['meanings'] goes. A continuous output is a modality of the design: a linear head on the concepts, higher orders included, with no generate grammar, in place of the verbal output. A path then has no output walk and its A is the head's cost on the path's own sentence state. XOR_grammar.xml uses it (item 14); other configurations with a numeric head on the per-word loop are not changed, and you list them. Its cost is R + A; on data that supervises output A's target is the supplied answer where one was asked and silence otherwise, on every row, and _answer_trial_cost's asked_only form goes. The exploit path is greedy everywhere; an explore path is the exploit path with one departure and greedy after it. architecture.explorePaths (new, default 1) is the number of explore paths a sentence makes in training.

6. The departure (§3.2). SentenceCredit.departure draws it as now, over seven walks instead of four: the reconstruction walk, the thought and the output walk join attention, compose, candidates and binds. The exploit path runs first and supplies every walk's eligible rounds. Explore paths of one sentence take distinct departures, each with the proposal correction of the draw actually made. A configuration without an output walk has none in the draw; paths are kept row by row, as sentence_pair selects rows now. Where the data supervises no output there is no A: the thought and the output walk are not in the draw, the paths are compared on R at their reconstruction walks, and the kept path runs its thought as now.

7. The trunk (§3.3). An explore path runs nothing that lies before its departure: a departure in the output walk makes one more output walk and no second reading, reconstruction walk or thought. From its departure it is computed to the end.

8. Every path ends before the write (§3.4). ThoughtClosing.close leaves _commit_sentence: a path's thought runs on its clause, prediction and reference plan, with its effects restored through WalkTrials.ThoughtTrialState. _commit_sentence writes the kept path's row and its thought's records, in today's order and with today's ticks, runs no thought and chooses nothing again. If an operation of the thought reads the sentence's own row, name it; the alternative there is a write that a rejected path undoes, and you report which you took.

9. One keep, and the credit (§3.5). SentenceCredit.comparison compares the paths once on R + A; the cheapest is kept, and an explore path only if strictly cheaper than the exploit path. Each explore path's departure is credited by score_function with its own difference from the exploit path on the whole R + A, at the chooser that owns the departed walk, its proposal correction counting the walks of the draw.

10. What trains (§3.6). Reconstruction and expectation on every path, each path's registry scaled by 2/(1+K) for K explore paths. A from the kept path's output walk, its weight shared equally with every explore path that departed in compose and, where there is one, with the exploit path (with one explore path this is SentenceCredit.reader_weights' rule with the draw). No other answer cost exists. Cut nothing from the graphs of the walks that train.

11. What goes (§3.7 to §3.9). WalkTrials.generation_pair, output_pair and thought_pair with their keeps; _sentence_generation_draws, _sentence_generation_cursors, paired_departure and paired_action_draw; ThoughtAnswer.scorer and train_readers; AnswerComparison and comparison_reader in every configuration, XOR_grammar.xml included, with their registries, weights, counters and the thought_comparison terms; reader_weights without a draw. Where a sentence close made the output, ask's reverseOutput replays the kept path's output walk and draws nothing; where none did, the batch end's read-out is the whole derivation and the same mechanism applies with the output walk as its only walk (§3.8). No shim and no second mechanism.

12. Diagnose (§5). (a) The presented question_conditioners weight is exactly zero after 76 lesson sentences and no gradient reaches it: find the cut and repair it so that the conditioner trains from the kept path's output walk, or report why it cannot. (b) Any XOR training of item 14 that does not pass: one trace by its recorded torch.initial_seed(), for each epoch the kept path's composition for each sentence, the head's cost and the loss; report the stage that does not learn (item 2.3 spec §33.1, finding 14: about a third of initializations do not converge today, mechanism open). Nothing is restored or added to pass a gate.

13. Omission (§3.11). In training, once for a sentence before it is perceived, omit content at two sizes: each word independently at the word rate loses every one of its percept slots, and each percept slot of a word that was not dropped is omitted independently at the form rate. An omitted slot has zero content and keeps its place and time, as create_ir_mask writes it; no learned mask code; never padding, one word of a sentence always kept whole, never outside training; every path reads the same omission. Nothing the model reads carries an omitted content (the word slab given to perceive, perceptualSpace._forward_input, the word's resolved rows and concept ids, its reference evidence). The order-zero reconstruction (compose's idea, the generate walk, the occurrence writer, reverse; reconstruction_order_zero) restores, without the thought: its targets (SentenceField.native_target, _staged_concepts_in, record.word_values) are the input before the omission at every place, as loss-side data. The input reconstruction (reconstruction_reverse) is costed where content was observed and not at omitted places. The zero content is a target of no cost; expectation's observation error at an omitted place takes the original as target only where that does not become state the model reads, and otherwise leaves the place out. Each rate is clip((v - r)/(v - a), 0, omissionLimit) from the long-term accuracies v on visible and a on omitted items of the kept path's order-zero reconstruction, its decoded bytes against the staged input before the omission (0 while v <= r; a counts as 0 until something was omitted), r = architecture.omissionTarget (new, default .9, in data/model.xml), architecture.omissionLimit (new, default .5), averaged over architecture.omissionHorizon sentences (new, default 1024); the two rates and four averages are saved with the model and reported for each epoch. omissionTarget 1 hides nothing: set it in an existing test or configuration that omission breaks, and in XOR_grammar.xml and the other XOR proofs, and list every override with its reason. maskRate and create_ir_mask on the path without the per-word loop are unchanged.

14. The XOR test (§3.12; Alec: "the purpose of the xor test is to test that design on the xor problem, not to make a new model architecture"; "A continuous output (and continuous input) can be treated as alternate modalities. Using a linear output head on top of order 0 concepts instead of verbal reconstruction can be a model option. So with that caveat, you can keep xor close to its current form."; and "when I said order-0, I meant no grammar (higher-order concepts are in fact required for xor)"). XOR_grammar.xml keeps its four sentences, numeric classes, compose grammar, widths and both certificates; answerSynthesis stays false there and _forward_head reads each path's composed sentence state (root and end slots) as now: the linear head on the concepts with no generate grammar. The twin goes there as everywhere, so the one head reads the kept path and trains by item 10's rule; a path there makes the reconstruction walk only and is compared on R plus the head's cost; omissionTarget is 1 there; add no thought, rule or second reader for the fixture. The gate stops the round: ten unseeded trainings, five or more passing the class certificate (it passes about two initializations in three today); with fewer, diagnose the failing runs (item 12) and repair the cause at its producer, and if the repair needs a decision return the diagnosis without offering the round.

15. Certificates and the one measurement as §4: the twelve certificates (the seventh is silence and the operand, the eighth to the eleventh omission); the XOR test of item 14 (ten unseeded trainings with explorePaths 1, five or more passing, reporting for each run the four answers and the kept composition of each sentence); the standing XOR proofs once; round 7's single-0.0-float32 lesson arm in three arms (explorePaths 1 and 2 with omission on, explorePaths 1 with omissionTarget 1), with walks, thought episodes and seconds for each sentence, the credited departures counted by walk, and the two omission rates and four accuracies for each epoch; in all of them the reader's cost on the kept path over training and the count of sentences on which a rejected composition trained; one complete sweep. With the landing: model.xsd and Params for explorePaths and the three omission parameters, and Architecture's "Operators round 2e: two answer-owned readers" and GradientFlow's credit sections amended by date.

Testing: direct before-and-after cases and the targeted files while you build; the measurement of §4 and make test once (env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected) when you offer the freeze. Receipt under doc/benchmarks/<date>-whole-paths/ with commands.json, frozen-source.json, before-source.json (round 7's landed source), source.patch, ports.json (every port's before and after text) and offered-documents.json sealed before the sweep. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud.

Return as §9 of doc/plans/2026-11-11-one-pair-per-stage.md, through Alec for Claude's review.
~~~
