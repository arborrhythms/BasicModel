# Item 2.0: one gradient, one choice mechanism, the order-zero loop and behavioral chains

2026-11-03. The specification of item 2.0 (`todo.md`), which follows item 2.5's completion round: Alec's decisions and Claude's proposals from the gradient-factoring discussion that followed the review of the §39 return (`doc/plans/2026-10-27-item2-5-thinking.md` §40), the order-zero loop of the "Thinking is Optional" conversation (§8), and behavioral chains in MTM at order zero (§9). The Codex text for the round is written after the review of the §41 return.

## 1. Projection replaces jitter (Alec, 2026-11-03)

With `separateChooser` false, the sentence close sums every owner's cost into one surrogate and calls backward once (`SentenceTraining.surrogate`, `Models._backward_training_loss`). Only the reconstruction owner is scaled by the jitter strength. `separateChooser` false becomes the default (Alec, 2026-11-03: one update per sentence). The single backward becomes two gradient calls:

- $g_R = \nabla L_R$ and $g_A = \nabla L_A$ over every optimizer parameter, the first call retaining the graph. The sides are listed in §2.
- For each unit $u$ with $\langle g_A^u, g_R^u\rangle < 0$: $g_A^u \leftarrow g_A^u - \frac{\langle g_A^u, g_R^u\rangle}{\|g_R^u\|^2}\, g_R^u$.
- The gradient is $g_R + g_A$, followed by the one optimizer step, as now.

Reconstruction's gradient is never altered. Output keeps every component that does not oppose it; that is the slack. No weight, tolerance or amplitude balance enters, because geometry alone fixes the removed component. A unit with one writer is unchanged, so the projection acts only where both sides write.

**Units.** Each parameter tensor is one unit. Concept tables are projected per concept row and definition coefficients per edge. On a whole table, agreement on one concept would offset a conflict on another. Also, the projection adds a multiple of $g_R$ to output's gradient, which would enlarge the step on rows output never touched.

**Jitter.** `reconstructionJitter` stays, with default 0 as now. It can be retrofitted as an annealing of reconstruction's strength if rich input does not supply the slack; rich input is the preferred method. The projection is invariant to the scale of $g_R$, so jitter would only reweight the final sum.

**Adam.** The projection guarantees that output's gradient adds no first-order increase of $L_R$. Adam rescales each coordinate afterwards, so the guarantee holds for the gradient but not exactly for the step. If the receipts show reconstruction rising where output was projected, take the inner product in Adam's metric instead: weight each coordinate by $1/(\sqrt{\hat v}+\epsilon)$ from the optimizer's second moment.

**Receipts.** Per unit and per epoch, report:
- the cosine of $g_R$ and $g_A$ before projection;
- the fraction of steps on which output was projected;
- the fraction of $\|g_A\|$ removed.

The existing dissonance and understanding norms of `SentenceTraining.gradient_receipt` remain.

The per-owner path remains for comparison. That is `separateChooser` true and `ObjectiveOwnership.backward_owned`, whose docstring reads "no gradient projection". Tests that assume the per-owner backward need repair when the default changes.

## 2. Two chains, and which side each term is on

Alec (2026-11-03) described two complete gradient chains, input to reconstruction and input to output. Both share the gradient from the input up to the understanding and are separate after it. The code, traced 2026-11-03:

- **Today, with `separateChooser` true, there is no shared trunk.** The answer reads a cut copy of the understanding, so A trains only its readers.
- **With `separateChooser` false (§1), the trunk is shared.** Perception codes, definitions and the compose operators are reached by both chains.
- **After the understanding, the two configurations differ.**
  - In `data/model.xml`, answers are read directly from the end slots, so the chains are separate after the understanding, as Alec described.
  - In `data/BasicModel.xml`, answers are generated (`answerSynthesis` true). Both chains run back out through the same operators and the same generate policy: the reconstruction decoder and the answer walk are one `_output_generate_walk`. The projection therefore also acts on those shared parts.

| Chain | Terms |
|---|---|
| R | the reconstruction owner (field, part, iterations, the decoder's walk residual, forward discard, content, reference), `compose_lesson`, expectation, attention, the penalty (once), the decoder walk's credit, and the R part of each choice reward |
| A | output (answer, prediction and the readers), `generate_lesson`, the answer walk's residual, the output walk's credit, and the A part of each choice reward |

**Each choice reward splits by chain.** *Superseded the same evening by §8: under the loop, reconstruction precedes thought, so every choice reward is on the output side whole. The split below is kept as the record.* This follows from Alec's picture.
- **The rewards.** They are `reconstruction.compose_score_function`, `expectation.bind_score_function`, `attention.candidate_departure` and the thought score function.
- **The split is exact.** Each reward is $\pi(a)\,\Delta C/q(a)$ with $\Delta C = \Delta R + \Delta A$. Because a reward is linear in $\Delta C$, it divides into $\pi(a)\,\Delta R/q(a)$ in the R chain and $\pi(a)\,\Delta A/q(a)$ in the A chain. `SentenceCredit.components` already keeps R and the answer apart.
- **Effect on the chooser.** Suppose a departure helps the answer but hurts reconstruction. The projection then removes the answer part on the chooser, so the chooser is not trained toward that departure. Where reconstruction is indifferent, the answer part acts in full.
- **Implementation note.** The two parts must share the combined tie mask, $\Delta C \ne 0$. Otherwise an exact tie with $\Delta R = -\Delta A \ne 0$ would leave a net update after projection.
- **Keep rule.** Which trial is kept is still decided by R + A together. The receipts count the departures whose $\Delta R$ and $\Delta A$ disagree in sign, which is where the keep rule and the projected credit part.

**The answer walk's residual moves to A.** It is booked as `reconstruction.walk_residual` today (`Models.reverseOutput`), so under one summed backward, reconstruction's gradient runs down the answer path.

Compose is part of reconstruction and generate is part of output (Alec, 2026-11-03), so `compose_lesson` is on the R side and `generate_lesson` on the A side.

The generate policy also decodes the understanding for reconstruction (`Models._reconstruct_trial`). The decoder walk's credit is judged by identity cost (reconstruction alone), so it sits on R. The output walk's credit is judged by the answer, so it sits on A. The two are summed into one `reconstruction.walk_credit` today and need separate terms. The projection then gives the policy's reconstruction role priority over its output role.

Code to change: `backward_owned` reads `compose_lesson`, `generate_lesson` and `chooser` as reconstruction. `GrammarLessons` already books `generate.choice` with objective output.

The retired teachers leave the table: `walk_policy` (§3), the decomposition teacher (§4), and the verb and modal teachers (§5).

## 3. One mechanism for every choice (Alec's point 2)

Generate does try two walks, in two places:
- **The reconstruction decoder** (`Models._reconstruct_trial`) costs a greedy walk and one departure by identity cost.
- **The output** (`WalkTrials.output_pair`) does the same by answer cost.

Three things still differ from compose:

| | compose, attention, thought | generate (decoder and output walks) |
|---|---|---|
| Walks | greedy and one departure | greedy and one departure |
| Departure step | uniform over eligible steps | uniform over eligible steps (`WalkTrials.departure_at`) |
| Alternative action | $\phi$-mixture (`Expectation.alternative_probabilities`) | the policy's softmax at $\tau = 1$, greedy action excluded (`Language.sample_eligible_logits`) |
| Credit | $\pi(a)\,\Delta C/q(a)$, $q$ including the uniform step draw (`SentenceCredit.score_function`) | cross-entropy toward the explored action where it won, nothing where it lost (`WalkTrials.generate_credit`) |
| Teacher | none (lessons on labelled input) | `walk_policy`: cross-entropy on the actual compose tree's detached states. Each leaf teaches STOP; each parent teaches its rule's generate face |

Unification:

1. Generate draws its alternative from `alternative_probabilities`, with the same $\phi$ and $\tau$.
2. Generate's credit becomes `score_function`, with the same proposal correction and $\Delta C$ from its own comparison. In each case $\Delta C$ is the part of R + A that the choice can change:
   - for the decoder, identity cost plus residual;
   - for the output, answer cost plus residual.
3. **`walk_policy` is retired** (Alec, 2026-11-03: "walk-policy forcing should be avoided if possible: the teacher should not be reaching into the internals of the model"). Generate learns only from its comparisons, like every other chooser, plus generate lessons where labelled data supplies them.

**History.**
- **Origin.** `walk_policy` came from Claude's proposal in the 6.8 plan §16.4 (`doc/plans/2026-09-27-item-6-8-one-attention.md`). It landed in operators round 1 (`73cd7b71ba`), where it replaced the free walk's straight-through policy gradient.
- **STOP.** The failure that motivated it was a free walk that always chose STOP. That was fixed instead by making STOP-versus-supported-undo an eligibility mask (6.8 §11.6, Alec), after which reconstruction rose from 0/10 to 6/10 (§12) and then to 10/10 (§17, 2026-10-05). The 10/10 came the day before `walk_policy` landed (2026-10-06).
- **Precedent.** The teacher reintroduced the imitation that the 2026-09-15 output contract had excluded (`doc/DevelopmentHistory.md`, contract 5: "no gold parse is assumed", "no output imitation credit").

**Not a substitute.** Making the compose tree's inverse a component of generate's proposal would put a compose trace into the free walk. That breaks the invariant "no compose trace enters free generation" (`doc/plans/2026-10-27-item2-7-corpus.md` §38.1), so it is not offered.

**Risks to measure.**
- **Signal.** The policy's signal falls from a target at every node to at most one credited departure per walk.
- **Starting bias.** The policy starts biased to STOP (`GeneratePolicy` bias 2.0).
- **Purity of the signal.** The decoder judge includes the walk residual, an internal round trip, beside the external identity cost.

**Tests that change.**
- `test_operators_round1.py::test_walk_teacher_covers_binary_unary_and_leaf_stop_without_code_gradient` is the teacher's own test and goes with it.
- These read `_last_decomposition_walk_teacher` or the term's name and need editing:
  - `test_grammar_word_learning.py::test_normal_text_reconstruction_updates_the_grammar_chooser`;
  - `test_free_reconstruction.py::test_free_trial_uses_no_reference_or_offsets_and_keeps_identity_audit`;
  - the `review17` probe run by `test_review14_contracts.py`.
- These assert that the policy changes every batch, and the teacher was the writer that guaranteed a nonzero gradient:
  - `test_output_walk.py::test_runbatch_trains_generate_policy_from_reconstruction`;
  - `::test_runbatch_answer_availability_never_reassigns_the_decoder`;
  - `::test_runbatch_output_masks_rows_without_reassigning_decoder_credit`.
- `test_explicit_dimensions.py::TestXorGrammarReconstruction::test_piecewise_overall_at_least_50_pct` needs a run.
- `test_mind_generativity.py::test_current_grammar_recovery_by_depth_training_and_chain_length` hand-writes the same forcing in its training loop.

Every choice then has one proposal family, one comparison, one judge and one credit: compose operation and location, attention, thought, and generate.

## 4. The decomposition teacher (proposed: retire)

`reconstruction.decomposition` (`Models._decomposition_teacher_loss`) has the character of `walk_policy`. It teaches the `DecompositionChooser` which operand pair the compose tree used at each node, evaluated on the tree's own parent states. It is on in default training.

- **Sole writer.** It is the only writer of the chooser's five weights. The features are detached, the choice is a hard argmax, and no credit reaches the pair choice.
- **Without it.** The weights stay at their initial $[1,0,0,0,0]$, which is exactly the residual argmin. That is the fix the 6.9 plan §24.4 recommended for the read-back collapse, and `doc/Architecture.md` already expects the chooser to become a lookup once inverses are exact.
- **Measured benefit.** Ordered pair picks went from 2/4 to 4/4 (6.8 §17, §18). Reconstruction is scored by multiset, which was 40/40 either way.
- **Measured harms.** The 6.8 §23 run-10 miss, where the argmin would have been right. The §41.3 divergence, where one step took the chooser's weight norm from 1.0 to $3.74\times10^8$; float32 seed 2 later reached a loss of $7.78\times10^{21}$.

Proposal: retire the teacher and keep the chooser at the residual argmin. If ordered-pair accuracy later shows a need, bring the pair choice under the one mechanism (§3) rather than restoring the teacher. Pair alternatives would then be departures in the decoder's comparison, credited by identity cost.

Tests:
- `test_decomposition_chooser.py::test_teacher_uses_resolved_input_rows_when_the_legacy_word_lane_is_absent` and `test_native_attention_field.py::test_decomposition_targets_follow_actual_reversed_candidate_reads` test the teacher and go with it.
- Tests of the static `DecompositionChooser.teacher_loss` go if that method is deleted.
- Tests of the chooser's initial behaviour stay. `::test_learned_orientation_wins_among_roundoff_exact_pairs_in_any_bank_order` would then describe a state production no longer reaches.

## 5. Verbs and modals learn by participation (Alec, 2026-11-03)

Alec ruled that by default, verbs and modals are trained by their functional participation in the sentence. The verb and modal teachers (`VerbLearning`, `ModalLearning`) are retired. Like `walk_policy`, both take their targets from the model's own derivation:
- **Verb teacher.** Its input is the recorded `lift` operand state. Its target is the referent's medium-term value, which the model's own compose transaction set.
- **Modal teacher.** It pairs two compose frames from the same tick.

The verb teacher trains the `lift` operator's `MeaningAction` (`Language.verb_action`), and the modal teacher trains the `adverb` operator's (`Language.modifier_action`). Only the embedding configurations build these actions (`data/BasicModel.xml`, `data/BasicModel_corpus.xml`). `data/model.xml` is numeric and builds none.

**What trains them without the teachers** (traced 2026-11-03, read-only, with one update per sentence):
- **The same sentence's answer, where it has one.** Both configurations set `answerSynthesis` true, and the answer reads the clause's state through question conditioning and the output walk.
- **The compose chooser's credit, through its live features.** This moves the action's outputs in the chooser's feature space. It says nothing about change.
- **Reconstruction.** The inverse undoes each action exactly, so for intransitive verbs and adverbs it contributes nothing. For transitive verbs it contributes a defect (below).

Nothing from a later sentence reaches them. Medium-term memory (MTM) carries state across sentences, but no gradient goes with it:
- every MTM write is detached;
- each close cuts the graph;
- seats re-read stored values detached;
- prediction trains only the predictor.

Before this plan, the only training that used MTM across sentences was the verb teacher itself.

Under `separateChooser` true, the value still configured, only reconstruction writes the actions. The teachers therefore go off together with the switch to `separateChooser` false (§1). Otherwise the actions would be trained by the defect alone.

**The first difference.**
- A `lift`'s output becomes the subject's new state in its identity seat (`ChunkJournal`).
- Whatever reads that state with gradient tells the verb what the state should have become. The verb therefore learns the change that makes later reads of its referent right, rather than a fitted ratio of after to before.
- It also learns on the input it actually receives, the composed verb-phrase meaning. The teacher used the verb word's own code.

**The second difference.**
- An adverb makes its clause modal, which moves the clause to the alternative timeline (`ChunkJournal`: a modal clause forces timeline 1).
- Its result is written only to the subject's alternative-timeline seat. The actual seat keeps the verb's result.
- The modifier therefore learns what readers of the alternative seat need beyond what readers of the actual seat need. The second difference is the gap between what the two seats must carry, with no comparison term.

**The gap.** Today the only reader with gradient is the same sentence:
- a story verb ("Felix went to the park") receives no credit from a later question about Felix;
- later questions read a modal's alternative state only symbolically (`TimelineRead`).

The teachers were bridging that gap.

**Credit crosses sentences through MTM: a re-read with no target.** Alec (2026-11-03): "Credit should cross sentences in virtue of training over MTM, which does maintain state across sentences." The re-read is how. When a later sentence seats a referent, its stored value $v$ is presented as $v + A_\theta(b) - \mathrm{sg}(A_\theta(b))$:
- $b$ is the recorded input of the value's latest event: the before-state, the basis and the modifiers. The verb-phrase meaning would need to be recorded in place of the verb symbol.
- $\mathrm{sg}$ stops the gradient.

The presented value equals $v$, so the next sentence still sees state. But the later sentence's own answer and chooser now credit the action $A_\theta$ that made it. Nothing is compared with a target; the later sentence's external objectives are the judge. The verb is credited by later reads of its referent, and the modifier by later reads of the alternative seat.

Limits of the re-read:
- it credits only the latest event on each seat;
- it takes the gradient at the current parameters, for a value made with older ones;
- truth and existence answers about modality stay symbolic;
- a referent that is never read again under gradient teaches nothing.

**Verb groups keep their assigner, through MTM.** The assigner is `VerbGroups.observe` (`MeaningKernels`), and its only caller is `VerbLearning.observe`.
- **How it works.** It runs on the host and moves no weight. On a verb's third distinct failed fit (residual above `verbGroupFit`, .1), it joins the best existing group or opens a new one. Its residuals are no-gradient evaluations of the action under each group against the referent's later MTM value: a cross-sentence comparison over MTM.
- **What is kept.** `VerbLearning.observe` keeps the pairing, the residuals, `groups.observe` and the taught marking. Only its loss (`errors.add('verb_teacher', ...)`) and the `register` calls at the close go. "Keep verb groups for intransitives" (Alec, 2026-10-23; `doc/plans/2026-10-23-verb-and-modal.md` §3) then still holds.
- **Who shapes the bases.** Every group starts as the identity, and no fit moves its basis toward the observed change. An assigned group is used by the forward `lift`, so the same sentence's readers write it, and with the re-read, so do later reads of the referent. Participation shapes the bases; the assigner chooses among them.

Caveats on the assigner:
- **It compares against MTM.** Under §8, MTM is the state of affairs that later input sets, so the comparison is against the world as read, not against the model's choices. Settled.
- **It evaluates the verb word's own code,** while the forward `lift` receives the composed verb-phrase meaning. It should read the recorded verb-phrase meaning, which the re-read records anyway.
- **It also runs during evaluation.** It is not gated by training, so evaluation can assign groups.

**A defect to fix in any case.** The derivation inverse calls `reverse_blocks` without the forward's action context (`DerivationReconstruction.inverse_step`). The walks' recomposition (`Generative.native_recompose`) also passes none.
- **Transitive verbs.** A transitive `lift` acts in its object's basis going forward but is undone in the identity basis, so reconstruction pushes transitive verbs' object bases toward the identity.
- **Grouped intransitive verbs.** An intransitive verb with a group is affected the same way.
- **Adverbs.** The `adverb` action always uses its one basis, forward and inverse, so it is unaffected.

The fix is to pass the recorded object value and group into `reverse_blocks`, which already accepts them.

**Rate.** Alec's ordering still applies to the actions themselves: codes first, then verbs, then modals. With no teacher term left to slow down, optimizer parameter groups set it: $\kappa\eta$ for the `lift` action and $\kappa^2\eta$ for the `adverb` action, with $\eta$ the base learning rate (0.001) and $\kappa = .1$ proposed. Wherever one objective is the main writer of a coordinate, Adam divides out a loss weight, so the rate must sit on the step. The receipt reports each action's realized relative step $\|\Delta\theta\|/\|\theta\|$ per epoch beside the codes'.

**Tests that go with the teachers:**
- the four tests of `test_verb_learning.py`;
- `test_modal_timelines.py::test_a_modifier_without_an_equal_time_alternative_teaches_nothing`;
- `test_modal_timelines.py::test_equal_time_teacher_uses_only_defined_coordinates_once_and_defers_its_graph`.

## 6. Other trained terms evaluated on the model's internals

A sweep of every trained registration in `bin/` (2026-11-03, read-only) found no other teacher of a choice. It did find terms that are evaluated on the model's own states. None of them supplies a target to a choice, so the proposal is to keep them this round:

| Term | What it constrains | Default |
|---|---|---|
| `reconstruction.walk_residual` | operator round trip: recomposing the selected inverse's children against the walk's own parent; also in the decoder and output judges | on |
| `reconstruction.forward_discard` | meaning lost at coordinates where a step's inverse is undefined, along the kept derivation | on with the sentence field |
| `<category>.definition_fold` | each class's sigma read covering the members the model itself minted or read as witnessed; possibly sigma's main gradient writer (unverified) | conditional |
| `reconstruction.reference` | reward only: the record of the model's own bind choice | on, weight .1 |
| `reconstruction.field` | the core reconstruction: a training-only reader follows the committed compose tree to the model's own codebook atoms of the input words | on |

**Note on `reconstruction.field`.** While the field is on, byte-level reconstruction (`free_bytes`) is not trained. Every reconstruction writer therefore targets the model's own encodings or its self-consistency, not raw input. The remedy is decided in §8: reconstruction from order 0, without grammar. It is not this round's work.

## 7. Decisions and open points

Confirmed (Alec, 2026-11-03):
- **Generate.** Its proposal and credit are made the same as compose's, and `walk_policy` is retired (§3).
- **Decomposition teacher.** It is retired, and the chooser stays at the residual argmin (§4).
- **Verbs and modals.** They learn by participation, and their teachers are retired (§5).
- **Credit across sentences.** Credit crosses sentences through MTM (§5), which the re-read implements.
- **The §6 terms.** They are kept this round. They are recorded in [Gradient flow](../GradientFlow.md#terms-on-internals) and in [Future work](../FutureWork.md), "Terms evaluated on the model's own internals: kept this round".

Closed by the "Thinking is optional" loop (§8, Alec, 2026-11-03, evening):
1. **Choice rewards.** Whole on the output side; no split (§8 supersedes the split in §2).
2. **The assigner.** It compares against the state of affairs, which later input sets. It stays.
3. **Rates.** The loop orders the levels structurally. Proposed: measure, build no rates.

For Alec's yes, from §8:
- `compose_lesson` on the output side, since compose is thought under the loop;
- thought's judge on unlabelled text is the next sentence's order 0;
- no $\kappa$ rates, measurement only;
- the loop as the next design item, with its own plan.

Carried into the same round:
- `separateChooser` false as the default (§1);
- the item 60 metrics, which also judge this change. They drop chooser variance as a criterion and use instead:
  - held-out cost;
  - acquisition;
  - answers;
  - each chooser's departure win rate.

Their per-epoch traces for the generate policy and the decoder's identity cost show whether retiring `walk_policy` costs acquisition.

## 8. Thinking is optional: the loop that closes the open points (Alec, 2026-11-03, evening)

Alec settled the three open points of §7 in a parallel conversation ("Thinking is Optional", 2026-11-03, 17:11 and 17:31). His statements, in his words:

1. **MTM is a state of affairs.** "We might want to use it as a state of affairs of the world, something that is not entirely permanent or transitory. It would be used to track changes in objects, and could thus be used to train verbs and modal operators." And: "MTM is needed to support a state of affairs, in which static concepts are combined with dynamic verbs."
2. **MTM is for thought and action only.** "Is it possible that MTM is not involved in perception/reconstruction, but only involved in thought and action? That means the perception and imagination are entirely free of grammar, which is the only mechanism that needs an identity store. It also implies that the identity store operates only on concepts of order-1 and higher."
3. **Reconstruction from order zero; thought from order zero to order zero.** "Reconstruction should be initiated from order-zero concepts rather than explicit understanding (STM0 and STM1). Understanding is a thought process that transforms order-zero to a novel order-zero. Output, on that model, can be the same as reconstruction (except that it uses a transformed understanding). In this architecture, there would be one input to order-0 concepts, one reconstruction at that point, one thought process from order-0 back to order-0, and an output at that point."

**The loop.** Input $\to$ order 0 $\to$ decode is reconstruction. Order 0 $\to$ thought $\to$ order $0'$ $\to$ the same decode is output. Thought holds everything with grammar: compose, STM0 and STM1, MTM, LTM and the higher orders. Perception and the decoder hold none of it. The loop closes without thought, which is why thinking is optional.

**What it closes.**

*Open point 1, the chains.* Perception (input to order 0) is shared by both chains. The decoder is shared too, by design: every reconstruction step trains the decoder that output uses. Thought is in the output chain only. Grammar is thought, so every choice is in the output chain only, and reconstruction does not depend on any choice. Consequences:
- A choice's $\Delta R$ is zero by construction. Each choice reward is on the output side whole, and the split of §2 is not needed.
- The projection acts on perception and the decoder only. Thought's parameters have one writer.
- The keep rule compares the trials by their output cost and work. Reconstruction is the same for both trials, since it precedes thought.
- "Compose is part of reconstruction" (§2) held while reconstruction ran back through the understanding. Under the loop, compose is thought. Alec (2026-11-03, late): "Compose is thought in the sense that it is symbolic understanding. ... compose, think, and generate grammars all operate as system 2, on top of system 1." So the three grammars are System 2, and perception, order 0 and the decoder are System 1. The grammar lessons from labelled data, the code's `compose_lesson` and `generate_lesson` owners, are on the output side with their grammars. Alec dislikes the name "compose-lesson"; the owners should be named for the three grammars, compose, think and generate, when the code changes.
- The order-zero to order-zero operation within a step is `intuit()`, the one-pass order-zero calculus (Alec, 2026-11-03, late: "'Thought's judge' seems like the intuit() from zero order concepts to zero order concepts"). Across time steps the same operation is expectation: "Next-sentence prediction is what we were calling expectation, the similar operation across time steps." So on a sentence with no supplied answer, System 2 is judged by expectation against the next input's order 0, and reconstruction by the current input. System 1 learns from System 2, but not by a comparison of results. Alec (2026-11-03, late): "I think system 1 does learn from system 2, but mainly by system 2's activation of the conceptual space. That will train the expectation (which should still be separate from intuit())." So System 2's activations are what expectation learns to predict; `intuit()` and expectation remain separate operations, and the evaluation of their combination is the one recorded in [Future work](../FutureWork.md), "`intuit`: a learned thought operation as the completeness fallback", in its amendment that the meaning-to-meaning map already exists.

*Open point 2, the verbs, the modals and the assigner.* MTM holds the state of affairs: an identity's static concepts at order 0, and the verbs that change them, with more than one state per identity. A verb is the difference between two states of one identity over time, and a modal is the difference between two states that cannot both be actual. The after-state is the later sentence's reading of the identity, so it is set by input. Training over MTM therefore compares a verb with the world as read, not with the model's own choices:
- The group assigner's residuals compare the verb under each group with the identity's later state. That is the design's own definition of a verb, so the assigner stays as it is (§5), reading order-0 states.
- The verb and modal teachers are retired in their present form (§5), where the states were compose frames. They return as training over the state of affairs: the verb is trained to map the identity's earlier order-0 state to its later one, and the modal to map the actual state to the alternative state at equal time. Both states are readings of input. The re-read of §5 stays beside this, so that later answers also credit the verb. Alec (2026-11-03, late): "The verb and modal teachers are MTM." Their training is MTM's own function, and belongs with it.

*Open point 3, the rate ordering.* The conversation does not speak to rates. The loop orders the three levels structurally: order 0 is trained by reconstruction on every sentence, a verb only on pairs of sentences about one identity, and a modal only on actual and alternative pairs. Each level therefore gets fewer updates than the one below it. Proposal: build no $\kappa$ rates, report the realized relative step of each level per epoch, and add rates only if the receipt shows the ordering violated. *For Alec's yes.*

*The §6 terms.* Reconstruction from order 0 removes `reconstruction.field`'s path through the compose tree, `forward_discard` along the kept derivation, and the decoder's walk residual from reconstruction. The grammar walk belongs to thought, so word order in output comes from thought. This supersedes the future-work item "Move reconstruction onto the free decoder": the decision is stronger, reconstruction from order 0 without grammar.

**MTM at order zero: behavioral chains (Alec, 2026-11-03, late; decided, before step 0).** "We might also have MTM operate on order zero concepts as a way of creating behavioral chains? Animals have that, and we might want to store those in MTM before promotion to chained event concepts." Then: "Behavioral chains should be implemented before step 0. Just as it identifies across nouns currently, it can also be used to identify 'the same' event (via gestalt properties, etc), which might even help in the creation of nouns." So MTM holds sequences of order-0 states beside its identities, kept until recurrence promotes a chain to a chained event concept, the LTM0 idea and LTM1 binary-chain layout specified for a later round (`doc/Architecture.md`, identity threading note). Event identity is the point: as MTM threads mentions into one identity today, it threads occurrences into one event by their gestalt properties (proximity, common fate), and a thing that persists through a chain of events is a candidate noun. This sits beside "the identity store operates only on concepts of order-1 and higher": a chain at order 0 is a sequence, not an identity, so the two hold together if MTM's order-0 contents are chains only.

It is scheduled before step 0, the large training run, and needs its own design item. That item must specify:
- what an order-0 chain is in MTM's terms: its states, its `.when` stamps and its continuity rule;
- the gestalt properties by which two occurrences are the same event, and how that matching relates to the bind of nouns;
- promotion: the recurrence rule that turns a chain into a chained event concept, and the LTM0 and LTM1 rows it writes;
- how an identity threaded through a chain becomes a noun candidate.

Recorded in [Future work](../FutureWork.md).

**Acceptance checks** (from the conversation):
- Feed an input's own order 0 through the decoder as if it were thought's output. The result must equal the reconstruction exactly.
- Ablate STM and MTM. Reconstruction must be unchanged.

**Tensions noted in the conversation.**
- Kahneman's object files are perceptual (tracking through occlusion and motion). With MTM out of perception, that persistence is carried without identity, by STM chunks or priming. The grounding of MTM in object files is then revised: MTM is the discourse referent and file card, at order 1 and above.
- Imagination renders grammar-free, but choosing which particular to imagine draws on MTM or LTM. The hand-off is one way: thought passes forms down, and the percept side never addresses MTM.
- Thought's last step must land on the order-0 codebook. A free vector off that manifold decodes to nothing useful.
- Output is a sequence of order-0 points. The decoder is applied per item, and the ordering comes from thought.

**Confirmed (Alec, 2026-11-03, late).** "It's basically what we have, but I think the gradient splitting gets cleaner, because reconstruction ceases to involve the grammar (although the generative grammar is still necessary for output)." The two chains under the loop are therefore:

| Chain | Parameters |
|---|---|
| Shared | perception, input to order 0; the order-0 decoder, concept to form |
| Reconstruction only | nothing |
| Output only | System 2: the compose, think and generate grammars with their choosers, STM, MTM, LTM and the higher orders. The generative grammar, the generate walk and its policy, is what output still needs |

The projection acts on the shared row only. Every grammar parameter has one writer, output. Today's reconstruction decoder is a generate walk, so the generate policy, the decoder's walk residual and the decoder walk's credit (§2, §3) are interim bookings on the reconstruction side; under the loop they all move to output.

One thing to measure when the loop is built: on unlabelled text, grammar is trained today by reconstruction. Under the loop it is trained by the next sentence's order 0 alone, plus work. Acquisition on FineWeb then rests on next-sentence prediction, and the item 60 metrics should show whether that signal is enough.

**Sequencing.** This is an architecture change, not this round's work. In this round:
- §1 (projection), §3 (generate under one mechanism), §4 (decomposition teacher retired) and §5 (verb and modal teachers retired, the re-read, the assigner kept) are built as written.
- Choice rewards go on the output side whole, with no split. That is where the loop puts them, and it needs no new machinery.
- Nothing moves reconstruction yet. The §6 terms stay, as decided.

The loop itself is the next design item, for Alec to place. It needs its own plan: the order-0 decoder, MTM timelines with more than one state per identity, the state-of-affairs training of verbs and modals, and the two acceptance checks.
