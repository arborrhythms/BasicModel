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

**The step** (corrected 2026-11-04, thinking spec §44). Reconstruction-owned parameters are stepped by SGD with momentum .9 and no variance normalization (`Models.py` about 2882 to 2889), not Adam; this covers the concept rows, the definition edges, the primitive memberships and the operator tensors. The projection guarantees that output's gradient adds no first-order increase of $L_R$. The step also carries .9 of the previous velocity, so the guarantee holds for the gradient and not exactly for the step. If the receipts show reconstruction rising where output was projected, project the momentum-carried step instead of the gradient. A unit stepped by Adam rescales each coordinate after the projection; there the inner product can be taken in Adam's metric, weighting each coordinate by $1/(\sqrt{\hat v}+\epsilon)$.

**Prerequisite (2026-11-04).** Item 2.5's arms with `separateChooser` false showed optimizer gradient norms of $1.05 \times 10^{11}$ on the primitive memberships, $9.9 \times 10^{9}$ on a lift operator and $2.1 \times 10^{8}$ from the chooser objective; the default arm peaked at 916,140 (thinking spec §44, finding 6). Item 2.0 makes `separateChooser` false the default, so its measurements wait until those producers are attributed (item 2.5's item 75) and repaired.

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
- **Without it.** The weights stay at their initial $[1,0,0,0,0]$, which is exactly the residual argmin. It gives priming no weight, as the 6.9 plan §24.4 did for the word read-back, and `doc/Architecture.md` already expects the chooser to become a lookup once inverses are exact.
- **Measured benefit.** Ordered pair picks went from 2/4 to 4/4 (6.8 §17, §18). Reconstruction is scored by multiset, which was 40/40 either way.
- **Measured harms.** The 6.8 §23 run-10 miss, where the argmin would have been right. The §41.3 divergence, where one step took the chooser's weight norm from 1.0 to $3.74\times10^8$; float32 seed 2 later reached a loss of $7.78\times10^{21}$.

Decided (Alec, 2026-11-03; moved into item 2.5 on 2026-11-04, thinking spec §44, decision (a)). The teacher retires in item 2.5's completion round. The chooser is the fixed joint residual argmin over form and both meaning lanes, out of every optimizer group. It is a chooser of the generate grammar, System 2, whose one writer is output (§8).
- *A learned policy later.* If answers later show errors of pair orientation, or near-ties among approximate inverses, the pair choice joins output's comparison (`WalkTrials.output_pair`, answer cost plus residual) under §3's proposal and credit. It keeps the joint residual in the score, takes no feature from the derivation's operand order, and uses a fixed score scale, never a division by the parent's energy (thinking spec §41.3). It is not credited by the decoder's identity cost, which the loop removes from reconstruction ([loop spec](2026-11-04-item2-0-order-zero-loop.md) §3.2).
- *Two dependences remain for that round.* The pair shortlist is by form cosine times priming, so meaning arbitrates only among the K form-nearest words per side (`Language.py` about 13711). Alec: meaning matters, particularly with large lexicons. Also, the decoder's basis includes a bounded snapshot of the forward's own activations (`Models.py` about 20186 to 20197).
- *The XOR gate after the loop.* The `XOR_grammar.xml` gate keeps decoding the composed root through generate, with its contract unchanged: four of four multisets, transpositions allowed. It does not move to the input reconstruction from the order-0 field, which never reads the root.

Tests: ported in item 2.5 (thinking spec §44.1, item 65).

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

**Rate.** Alec's ordering still applies to the actions themselves: codes first, then verbs, then modals. With no teacher term left to slow down, optimizer parameter groups set it: $\kappa\eta$ for the `lift` action and $\kappa^2\eta$ for the `adverb` action, with $\eta$ the base learning rate (0.001) and $\kappa = .1$ proposed. These actions are reconstruction-owned and stepped by SGD with momentum, not Adam (thinking spec §44). The rate therefore sits on the step, in the optimizer's parameter groups, rather than on a loss weight, so that it orders the actions without changing what their objectives measure. The receipt reports each action's realized relative step $\|\Delta\theta\|/\|\theta\|$ per epoch beside the codes'.

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

Answered (Alec, 2026-11-03, late):
- **Rates:** "Sure, keep in model file." No rate ordering is built; the ratio $\kappa$ is a parameter of the model file, default 1, so the ordering can be set if the receipts show it violated; the realized relative step per level is reported.
- **The compose lesson's side:** Alec answered "If this is just learning MTM, and we have since resolved that it should operate on order zero also, then it should go in conceptual space as a method called from forward(), and probably ramsified." That is taken as the placement of MTM's linking (§9); the lesson owners follow their grammars, as §8 says. *Alec corrects the reading if it was meant otherwise.*
- **D7, the residual split:** closed by the loop, "except as the grammar influences current symbolic activation state": System 2's activations are what expectation learns to predict.
- **The `XOR_grammar.xml` gate and the uniform fraction:** "Please refine after more feedback from the run." Both decisions wait for item 64's reruns ([thinking spec §42](2026-10-27-item2-5-thinking.md)).
- **The loop** is item 2.0's last part, not a separate item.
- **The supplied `yes` admission** is carried into item 2.0 pending Alec's confirmation of that home over item 0.

Carried into the same round:
- `separateChooser` false as the default (§1);
- the item 60 metrics, which also judge this change (the held-out set behind held-out cost is decided at the start of item 0, Alec, 2026-11-03; until then the round reports the other three). They drop chooser variance as a criterion and use instead:
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

**MTM at order zero: behavioral chains (Alec, 2026-11-03, late; decided, before step 0).** "We might also have MTM operate on order zero concepts as a way of creating behavioral chains? Animals have that, and we might want to store those in MTM before promotion to chained event concepts." Then: "Behavioral chains should be implemented before step 0. Just as it identifies across nouns currently, it can also be used to identify 'the same' event (via gestalt properties, etc), which might even help in the creation of nouns." So MTM holds sequences of order-0 states beside its identities, kept until recurrence promotes a chain to a chained event concept in the built LTM layout: LTM0 ideas and LTM1 rows of two tagged references (`OCCURRENCE`, `SYMBOL`, `RELATION`; layout version 14), which has no writer of temporal links today. Event identity is the point: as MTM threads mentions into one identity today, it threads occurrences into one event by their gestalt properties (proximity, common fate), and a thing that persists through a chain of events is a candidate noun. This sits beside "the identity store operates only on concepts of order-1 and higher": a chain at order 0 is a sequence, not an identity, so the two hold together if MTM's order-0 contents are chains only.

It is scheduled before item 0, the full training session (Alec's "step 0"), and its design is §9. That design specifies:
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

The loop is item 2.0's last part. Its design in code names, after Alec's refinement (the decoder is the existing `reverse`, with the higher-order reverse involving symbols separated out; reconstruction keeps its place with the parallel conceptual field as intermediate state; output turns order-0 concepts into order-0 concepts and calls the same `reverse`), is [2026-11-04-item2-0-order-zero-loop.md](2026-11-04-item2-0-order-zero-loop.md), with six decisions for Alec in its §5.

## 9. Memory: one file, and MTM as one linking function across time (Claude, 2026-11-03, re-specified)

The design is its own file, [2026-11-03-item2-0-behavioral-chains.md](2026-11-03-item2-0-behavioral-chains.md). A first version with a separate chain entry, its own continuity rule, gestalt matcher, promotion branch and forgetting roots was withdrawn by Alec the same evening as too complex. Alec: "Placement should be a single file called memory that contains STM, MTM, and LTM. Note that MTM is just linking things across time; causality for events, identity for nouns, processes for verbs. It's the same function in each case." The re-specification:

- **One file.** `bin/Memory.py` holds `STMChunks`, `MediumTermMemory` with `Token`, `Timeline` and `Event`, `LTM0` (out of `Layers.py`) and `LTM1`, names and behaviour unchanged, the old modules removed.
- **One function.** `ConceptualSpace.link(kind, occurrence, ...)`, called from `forward()` at the kept close, links occurrences across time in three kinds: identity (mentions of one thing, order 1 and above, today's bind), process (the events of one identity, today's timelines), causality (consecutive order-0 states of one course of events, new). It is written once, with no store-specific names inside it: a kind, the accessible entries of that kind, one feature row per candidate (content meet, continuity in the where and when bands, and for the ordered kinds the first difference, which is common fate), one small policy per kind with the withhold dummy, the chooser's choice, the ordinary credit.
- **The entry** is the existing `Token` with a `link` field naming the kind; a causality entry's occurrences are the kept sentences' order-0 ideas in time order. No new entry type, no new knob; the pool's capacity, horizon and ids are shared.
- **Promotion** is one rule for every kind: `conceptMintCount` recurrences promote the entry to a concept of its occurrences through `promote_identity`'s path, writing membership rows and, for the ordered kinds, successor rows between consecutive states. A promoted causality entry is the chained event concept.
- **Nouns.** An identity linked through `conceptMintCount` consecutive states of one causality entry receives a mention witness per state and mints with no new count.
- **Constraints kept.** No order-0 identity is opened; causality entries are never prepended, so MTM stays out of reconstruction; every write stays detached; the credit is the output-side credit of §2.

Three decisions remain in its §5: the move, one policy per kind, and process entries not promoted on their own.

## 10. The rounds and the round-1 text (Claude, 2026-11-08)

Item 2.3 landed on 2026-11-08 (BasicModel `59cd466077`, [trust spec §35–§36](2026-11-05-item2-3-trust.md)); item 2.0 begins. It is one item in six rounds, each with its own freeze, as todo.md lists them under item 2.0 ("Rounds"): behaviour-neutral moves; repairs and the measured "before"; one gradient (§§1–2); one choice mechanism (§§3–5); memory and basic percepts as one round (§9, the [memory](2026-11-03-item2-0-behavioral-chains.md) and [basic-percepts](2026-11-04-item2-0-basic-percepts.md) specs); the [order-zero loop](2026-11-04-item2-0-order-zero-loop.md) (§8). The hand-off is [2026-11-08-handoff-2-0.md](2026-11-08-handoff-2-0.md). Round 1 carries thinking spec §58's SHIFT note and §54 finding 3; the text below is final. Codex's return is §11.

### 10.1 Text for Codex: round 1

~~~
Item 2.0, round 1: behaviour-neutral moves. The spec is doc/plans/2026-11-03-gradient-projection.md (§§1-9; §9 points to the memory spec doc/plans/2026-11-03-item2-0-behavioral-chains.md, whose §2 and §3.1 govern the move) and todo.md's item 2.0 entry ("Rounds", round 1); this text is spec §10. Start from the landed main 87e74ff435 with a clean tree. This round changes no behaviour: every certificate below is "the same as before", and the one make test at the freeze is expected to report the landing's counts (7509 passed, 325 skipped, 1 xpassed) plus the new direct cases. Build nothing of rounds 2 to 6 (MTM's link function, the gradient, the chooser, basic percepts, the loop).

1. One memory file (memory spec §3.1, D1). bin/Memory.py holds the three stores with names and behaviour unchanged: the STMChunks constants and functions (FIELDS, empty, locate, deposit, keep_close and the rest); ShortTermMemory out of Layers.py (decided here: it moves, since it owns STM's live buffers and is the STM Alec named; its shift/reduce scorer moves with it untouched, an entanglement for a later round, recorded); MediumTermMemory with Token, Timeline and Event (state version 10 unchanged); LTM0 out of Layers.py (its mixins OccurrenceRows, ClauseRows and EntryRead stay in Occurrence.py, ClauseRow.py and EntryRead.py: they are not stores); and LTM1. bin/STMChunks.py, bin/MediumTermMemory.py and bin/LTM1.py are deleted; every import site in bin/ and test/ is repointed (several hundred lines, local imports included); no compatibility alias or re-export, under the no-legacy rule. The file's docstring states the one boundary at which all three change: the kept close. State-dict keys do not depend on module paths. Certificates: a checkpoint written by the landed source loads unchanged on this source (write it from the landed tree before the move and keep it in the receipt); the state-key inventories (test_truth_ideas_routing and its kin) are unchanged. If any load or save depends on a module path it fails loud and the return records it; nothing is aliased.

2. Thought ideas stamped and found by .when (loop spec doc/plans/2026-11-04-item2-0-order-zero-loop.md §3.5 misfit 2, gate 10 and D5; this is D5's behaviour-neutral part, the rest of D5 is the loop round's). Today ThoughtChunks.put deposits a thought idea at a synthetic .where (address(identifier + 1), four coordinates disjoint from the percept bands) with no .when, and _index, held, read and release find it by chunks.locate(state, where). After: every idea put deposits carries the .when of the thought step that made it, from a thought-step clock that advances once per deposited idea so that .when identifies the idea, encoded in the four-coordinate when band through RowTime.encoder or a thought ladder of the same width (the thought working set is its own constructed STM, ThoughtChunks.attach; if sentence ideas ever share it, the thought encoding must be disjoint from model time as the address bands are today; record which holds); it is found by that .when (locate on the WHEN band) and its .where is empty (-1). address goes, and the two address computations in Thoughts.py with it. The duplicate-occupant and absent-chunk raises of _index carry over to .when. Certificates: a deposited thought idea reads back by its .when with its .where empty; two ideas of one thought have distinct .when; the thought tests (test_thought_infer, test_thought_intuition, test_thought_operation_catalog, test_item6_2_thinking, test_item25_repair) pass unchanged apart from assertions on the synthetic address, which are ported with before/after text.

3. The substitution no closing reaches (thinking spec §54 finding 3). ReferenceClose.commit's one-leaf substitution (the branch occurrence[:1] == ('sym',) and word_occurrences and len(item['leaves']) == 1, about line 1240) and its word_occurrences parameter are retired; the word_occurrences=read_word_occurrences argument at the resolver.commit call in Models.py (about line 21016) goes; commit_read_words stays, called for its effect. A one-leaf description whose occurrence is a symbol now raises as the multi-leaf case does ("description ... has no own occurrence"), fail loud. Record the retirement in the return with the finding's text. Certificate: the reference tests (test_reference_close, test_reference_arrival_review, test_reference_occurrence_views, test_reference_review10, test_bind_thought, test_memory_consolidation) pass unchanged.

4. No row for a SHIFT at the source (thinking spec §58's note for this round). Models.py maps a SHIFT's column -1 through clamp to slot 0's journal column (about line 13171), and the Reading's operation_values, operation_refs, operation_relations, case_values and case_selected index with columns.clamp_min(0) (about lines 13246 to 13257), so a SHIFT's row in each table is a copy of row 0. Give a SHIFT no row: its position carries no operation row (a sentinel the readers skip, or the tables sliced to operations, whichever the readers already assume; record which) and nothing is copied from row 0. ChunkJournal.capture's kind mask stays. Certificates: a direct case in which a reading with a SHIFT has no table row equal to row 0's; test_chunk_journal_origins::test_shift_alias_records_real_operation_and_closing_writes_it passes unchanged; the journal and sentence tests pass.

5. Documents, by append or exact replacement: Architecture (the one memory file and the kept close as the boundary at which STM, MTM and LTM change; the thought idea's .when; the SHIFT's absent row; and exact replacement of its line-10 sentence "The LTM0 idea/LTM1 binary-chain layout and eight shared STM chunks are specified for a later round, not implemented here."), STM.md (the module paths; ShortTermMemory in Memory.py), Testing.md one line, FutureWork's memory entry gains the pointer to bin/Memory.py; your return as §11 of the 2.0 spec.

Carried, not built here (record them in the return): §54 finding 2 (an elided role binds nothing under containment; round 5 with the full reference form), findings 4 and 5 (the output gradient under the shared backward and the projection's momentum tail; round 3), findings 6 and 7 (records).

Testing: fast checks only until the implementation is finished. Each item's direct before-and-after cases, the targeted files (the memory tests test_medium_term_memory, test_stm_chunks, test_stm_substrate, test_ltm0_location, test_ltm_chain_grows, test_ltm_consolidation, test_familiar_tier; test_reference_close, test_chunk_journal_origins, test_sentence_journal, test_truth_ideas_routing; the thought tests of item 2; the files you repoint) and make test once when you offer the freeze (env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected; the XOR class, the pending-premise pair and the shared-concept case run inside it). No item 71 runs, no lesson arms, no repeats, no pre-checks. Receipt under doc/benchmarks/<date>-item2-0-round-1/ with commands.json, frozen-source.json, before-source.json (the landed tree's source f770cd132dbc...), source.patch, ports.json (every port's before/after text), offered-documents.json sealed before the sweep, and the list of moved files; every file your return links to exists before the sweep starts. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud. Return through Alec for Claude's review; no commit or push.
~~~

### 10.2 Round 2 under the fast-checks rule (Alec, 2026-11-08)

Round 2 is listed as "repairs and the measured before" (todo.md, item 2.0, "Rounds"): the repairs, and item 71's runs with the item 60 metrics recorded as the baseline. Those runs are long, and the standing rule since 2026-11-06 ([thinking spec §58](2026-10-27-item2-5-thinking.md)) is fast checks only until the implementation is finished. The conflict was put to Alec with the recommendation that round 2 build its repairs under fast checks and the measurements wait for the end of item 2.0. Alec: "Fast checks only." Consequences:

- **Round 2 builds the repairs** and certifies each with its direct before-and-after cases, the targeted files and one `make test` at its freeze: fail loud in the reverse's stage loop and in `runBatch` ([loop spec §3.9](2026-11-04-item2-0-order-zero-loop.md)); the derivation inverse given the forward's action context (`DerivationReconstruction.inverse_step` and `Generative.native_recompose` passing the recorded object value and group to `reverse_blocks`, §5); one WholeSpace ([basic-percepts spec §7](2026-11-04-item2-0-basic-percepts.md), D3, with its checkpoint migration and the persisted-address certificate); the supplied `yes` admission (the three-stage report for every supplied form and the certificate on the admission path, todo.md item 2.0, "Carried"); and the sentence-82 capture rule installed at `ReferenceClose.commit`'s deferred `mtm.bind`, a rule on every lesson-13 run, exercised when those runs happen. The item "any event at or above $10^8$ under `separateChooser` false that item 71 reports, repaired at its producer" is empty today: item 71's last runs (thinking spec §56 to §58) reported no such event in any audited class; it is re-opened only if the end-of-2.0 runs show one.
- **The measured "before" moves to the end of item 2.0**, with every other measurement of the item: item 71's runs and the item 60 metrics (held-out cost excepted until item 0) are taken then, on round 2's frozen source, which its freeze hash identifies, and on the finished tree, so that rounds 3 to 6 are still judged against a baseline that precedes the gradient change. Until then no round of 2.0 makes a learning claim; each certifies its mechanism, as 2.3's rounds did.
- The round-2 Codex text is written after the review of the round-1 return, on Alec's word, and carries the fast-checks rule as every text does.

## 11. Round 1 return (2026-11-08): behavior-neutral memory move and source repairs

Offered source `2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa` (1004 files), from landed main
`87e74ff4351faec544f1fb51d4ff9c98fcfacc71`. Work began in a clean detached checkout;
the live checkout's pending §10, todo and handoff edits were preserved. The
ignored corpus manifest is part of the landed source inventory and was copied
unchanged before the baseline was recorded. The complete landed inventory is
`f770cd132dbc04d6ea20964cabc4fabf734511f7c9908a88240fd55cd472507e`.
The finished edits are in the live BasicModel checkout, uncommitted and unpushed,
for Alec to return to Claude. The freeze records below are sealed before the
one complete default sweep; its eventual result is reported in the already
created result file, without editing this offered prose.

### 11.1 What moved, and what stayed the same

[Memory.py](../../bin/Memory.py) now owns STM's constants and functional
operations, `ShortTermMemory`, `Name`, `Token`, `Timeline`, `Event`,
`MediumTermMemory`, `LTM0` and `LTM1`. STM and LTM1 use different numerical
reference tags, so the STM constants/functions remain together in the
`STMChunks` namespace inside Memory; their names, values and default operations
are preserved. ShortTermMemory's complete shift/reduce scorer moved unchanged
(AST comparison certified); its coupling to live memory remains an entanglement
for a later round. OccurrenceRows, ClauseRows and EntryRead remain in their
original mixin files. The kept close is the common commit boundary stated in
the module docstring. The three old modules are absent; Layers has neither
moved class as a compatibility export; the import census finds no retired
import site in bin or test.

Alec's additional subclassing direction is implemented by the stateless
`MemoryStore` base shared by ShortTermMemory, MediumTermMemory, LTM0 and LTM1.
It owns the existing tensor clone/detach/CPU-copy operation. STM trial snapshots
can retain gradients, while committed copies detach them. There is no base
constructor, buffer, parameter, capacity change or checkpoint field. This
shares existing code without adding a common linking or lifecycle policy.

The [native landed checkpoint](../benchmarks/2026-11-08-item2-0-round-1/landed.ckpt)
and [populated memory checkpoint](../benchmarks/2026-11-08-item2-0-round-1/landed-memory.pt)
were written before the move. The native strict loader and all memory restorers
load them unchanged: exact tensors, exact state-key inventories, MTM version
10, no module alias or re-export and no module-path load/save failure. The
populated certificate includes STM content/bands/evidence, an MTM identity,
an LTM0 idea and an LTM1 relation. The unchanged state-key assertions, including
test_truth_ideas_routing, passed. [Before inventory](../benchmarks/2026-11-08-item2-0-round-1/checkpoint-before.json),
[load certificate](../benchmarks/2026-11-08-item2-0-round-1/checkpoint-after.json),
[move and scorer audit](../benchmarks/2026-11-08-item2-0-round-1/move-audit.json).

### 11.2 Thought time, retired substitution and SHIFT rows

Thought puts now allocate `.when` from a per-stream thought clock, once per
deposited idea. Existing-idea updates and reads do not advance it. Exact WHEN
lookup drives read, held, release, replay and query operands; WHERE is explicitly
four -1 coordinates. The thought and sentence working sets share the owner
passed to attach, so thought time uses four exact 16-bit digits offset by 2,
disjoint from the model-time percept band. The synthetic address function and
both computations in Thoughts are gone. The generic `ThoughtRecord.address`
checkpoint field is retained and points to WHEN; it is not an old-module alias.
The new clock is live STM state and participates in trial rollback. Duplicate
occupied stamps, missing chunks, exhausted capacity and clock overflow fail
loud. The direct cases check distinct stamps for two ideas, read-back, empty
WHERE, coexistence with a sentence stamp, eviction and trial restoration.

The retirement is thinking spec §54 finding 3, whose text was:

> *A substitution that no closing reaches.* `ReferenceClose.commit` substitutes a one-leaf mention's word occurrence when the mention's address is a symbol (about line 1216), and `Models` passes `word_occurrences` for it (about line 21002).
>    - Its one production caller passes the rows of `write_clause`, where every mention now has an occurrence of its own.
>    - Before item 110, only the defective multi-leaf root reached the substitution.
>    - The substitution and the argument are retired in 2.0 round 1, with a record. `commit_read_words` stays.

The branch and parameter are removed. Models still calls commit_read_words for
its effect. Both one-leaf and multi-leaf symbol-addressed descriptions now raise
“description ... has no own occurrence.” Two test callers retained the retired
keyword; their calls were ported, with exact text in ports.json. Their native
occurrence assertions remain, and test_reading_round additionally proves the
mention is none of its word occurrences. No reference test was retired.

Reading tables retain action alignment, which their readers already use.
Nonoperations have column -1 and empty sentinel entries: zero numerical values,
false flags and -1 references, including zero operation probability. No SHIFT
or transpose is gathered from a choice or journal row; fold/rewrite operations
retain their original rows and gradients. ChunkJournal.capture's kind mask is
unchanged, and its existing SHIFT-alias closing certificate passes unchanged
apart from its import port. The direct source case checks every affected table
and proves there is no gradient contribution from a nonoperation row.
[Direct landed cases](../benchmarks/2026-11-08-item2-0-round-1/direct-before.json),
[direct offered cases](../benchmarks/2026-11-08-item2-0-round-1/direct-after.json).

### 11.3 Validation, documents and receipt

The bounded targeted run completed 2,144/2,144 cases over 149 files: 2,087 passed,
55 skipped and two failures from obsolete word_occurrences calls in the tests.
After the call ports, those two cases and the reference-reconstruction file
passed all 16 checks. The earlier focused memory/reference run passed 288 with
7 skipped. Development also caught and repaired a mechanical torch.where
rename typo; an inventory test's sandboxed compiler-cache failure passed under
the normal authorized process permissions. These attempts are retained in the
[commands](../benchmarks/2026-11-08-item2-0-round-1/commands.json) and
[targeted summary](../benchmarks/2026-11-08-item2-0-round-1/targeted-summary.json).
No seed was pinned or selected, no item 71 run or lesson arm was run, no capacity
was resized, and no existing test was retired and no language-specific branch was added.
There are eleven new direct parameterized cases in test_item20_round1; all
existing test function inventories remain unchanged.

Architecture has the exact replacement of the obsolete line-10 sentence and
an appended record of the module, kept close, shared base, thought WHEN and
SHIFT sentinels. STM has the new module paths and an appended record; Testing
has one line; FutureWork's memory entry points to Memory. The old thinking
spec's one live LTM1 source hyperlink now points to Memory. The link check also
found one old receipt's live LTM1 source pointer; only that pointer is replaced,
with its original bytes preserved in the [historical link port](../benchmarks/2026-11-08-item2-0-round-1/historical-link-port.json).
The initial link check passed 999 files and found that one obsolete pointer;
its repair is checked before the freeze. The pending §10 and handoff prose are preserved, including the concurrent §10.2 fast-checks decision for round 2.

The one freeze command is `env -u RUN_TRAINED -u RUN_SLOW make test`, with no
selection or deselection. The XOR class, test_convergence, both pending-premise
orders and the shared-concept case run inside it. The expected count starts
from 7,509 passed, 325 skipped and one permitted XPASS, plus the eleven new
cases and any additional documentation-file case. The run's actual counts,
status and critical-case results belong to the
[sweep result](../benchmarks/2026-11-08-item2-0-round-1/sweep-result.json) and
[final audit](../benchmarks/2026-11-08-item2-0-round-1/final-audit.json);
[sweep log](../benchmarks/2026-11-08-item2-0-round-1/sweep.log).
These files exist before the sweep starts; only execution evidence is filled
in afterwards. This return makes no advance claim that the sweep passed.

Receipt: [before source](../benchmarks/2026-11-08-item2-0-round-1/before-source.json),
[frozen source](../benchmarks/2026-11-08-item2-0-round-1/frozen-source.json),
[source patch](../benchmarks/2026-11-08-item2-0-round-1/source.patch),
[every port's before/after text](../benchmarks/2026-11-08-item2-0-round-1/ports.json),
[moved files](../benchmarks/2026-11-08-item2-0-round-1/moved-files.json),
[offered documents](../benchmarks/2026-11-08-item2-0-round-1/offered-documents.json).
Source and offered documents are checked unchanged after the sweep. Nothing
is committed or pushed by this return.

### 11.4 Carried, not built

Thinking spec §54 finding 2 remains round 5: an elided role has no content and
binds nothing under containment; the full reference form must supply content.
Findings 4 and 5 remain round 3: under shared backward, output reached the
percept-to-word readout and answer readers, not codes/operators/definitions;
the projection's momentum-driven tail remains, with outward-momentum
projection only a recorded proposal. Findings 6 and 7 remain records: the
lesson arms bound or exactly answered at most 1/80, so their numerical gate
was not a learning claim; the weekly slow record was stale, and make test_all
belongs to items 0 and 1. No linking function, gradient projection, chooser
unification, basic-percept implementation or order-zero loop was built here.

### 11.5 Review of the §11 return (Claude, 2026-11-08): accepted; round 1 is frozen at `2d2abfe10878…`

**Verified.** The live checkout's source by the receipt method (`test/bounded_tests.source_snapshot`, SHA-256 of the sorted JSON) is `2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa` over 1004 files and equals `frozen-source.json` file for file; `before-source.json` is the landed tree's `f770cd132dbc…` over 1005 files ($1005 - 3 + 2 = 1004$: three modules deleted, `bin/Memory.py` and `test/test_item20_round1.py` added). `source.patch` covers 202 files and `ports.json` 193 files in 479 hunks; every hunk's before text matches `87e74ff435` and its after text matches the live tree at the recorded lines, and the 201 entries of `git status` are all in the patch. The move was read by AST: every function and class of `STMChunks.py`, `MediumTermMemory.py` and `LTM1.py`, and `LTM0` and `ShortTermMemory` out of `Layers.py`, compared with its copy in `Memory.py`; the differences are the `@staticmethod` decorator and the `STMChunks.` qualification inside the namespace class, `locate`'s new `band` keyword (default `WHERE`, so existing callers are unchanged), module-level imports in place of the former local ones, and `_snapshot_tensor` in place of the same `detach`/`cpu`/`clone` sequences (`snapshot(detach=False)` still clones without detaching). The shift/reduce scorer is unchanged. `Layers.py` holds neither class; no retired import or module path remains in `bin/`, `test/`, `data/` or the live documents. The 163 import-only files were classified by machine and the 39 others read by hand. Receipt: one `env -u RUN_TRAINED -u RUN_SLOW make test`, exit 0, 7847 cases, 7521 passed, 325 skipped, 1 xpassed, which is the landing's $7509 + 11$ new cases $+ 1$ (the hand-off file's link case); `test_convergence`, the shared-concept case, both pending-premise orders and the `XOR_grammar.xml` reconstruction gate passed inside it; the documents were sealed ten seconds before collection and unchanged after; the source was unchanged after; the three result files existed as placeholders before the sweep and were filled after, as §11.3 says. My reruns from the scratchpad: the checkpoint certificate (the landed `landed.ckpt` and the populated STM, MTM, LTM0 and LTM1 state load exactly on the offered source, state-key inventories unchanged, MTM version 10, no alias) and the direct cases reproduce Codex's `checkpoint-after.json` and `direct-after.json` byte for byte; the targeted set of 149 files: 2144 cases selected and completed, 2089 passed, 55 skipped, none failed, exit 0 (Codex's own run of the same set, before its two call ports, had 2087 passed and those two failing); the eleven new cases passed inside it (`output/tests/20261108-173355-29a506`, log in my scratchpad).

**Approved as built.**
1. *One memory file.* `STMChunks` is a namespace class because its reference tags differ in value from LTM1's; `MemoryStore` is a stateless base with one static method and no constructor, field or checkpoint key, which §10 did not ask for: Codex records it as Alec's direction, and the acceptance below assumes that is so. To break the import cycle (`Memory` imports `Layer` and `ClauseRows`), the `NULL_CONCEPT` imports in `Layers.py` and `ClauseRow.py` moved from the module level into the seven methods that read the tag. Recorded as a wart: the tag constant's home is a leaf module in a later round; no behaviour depends on it.
2. *Thought ideas by `.when`.* The per-stream clock on `ShortTermMemory` advances once per deposited idea and never on a read or update; the stamp is the former address digits, four exact 16-bit digits offset by 2 (so every coordinate is at least 2, while `RowTime.encoder` stays in $[0, 1]$: disjoint), with `.where` $= (-1, -1, -1, -1)$. The recorded choice: the thought and sentence working sets share the owner passed to `attach`, so the encoding is disjoint from model time, as §10 required in that case. The clock is restored by `ThoughtTrialState` and is not checkpointed, like `_thought_contents`; a checkpoint inside a thought episode was never supported. `ThoughtRecord.address` keeps its checkpoint key and now holds the `.when`: a naming debt, recorded, since a key rename would change every checkpoint. Duplicate stamps, absent chunks, a full working set and clock overflow raise.
3. *The substitution.* The one-leaf branch and the `word_occurrences` parameter are gone, every symbol-addressed description raises, `commit_read_words` is still called; the finding's text is in §11.2; the two test callers are ported and `test_reading_round` now also asserts the mention is none of its word occurrences.
4. *No row for a SHIFT.* Codex kept the tables aligned to action positions and gives a non-operation an empty sentinel row (zeros, false flags, $-1$ references, probability 0) with no lookup; an operation without a choice or journal column now raises. I read every reader of `operation_values`, `operation_refs`, `operation_relations`, `operation_probabilities`, `case_values` and `case_selected` (`ClauseJournal`, `SentenceCredit`, `EqualityLearning`, `DerivationReconstruction`, the generate lesson in `Models`): each branches on kinds 1 and 2 before indexing, so no sentinel is read. The direct case shows the row-0 alias gone and no gradient from a non-operation row.
5. *Documents.* Architecture's line-10 sentence replaced exactly and a dated section appended; STM.md, Testing.md and FutureWork as asked; one live link in the thinking spec and one in a 2.5 receipt retargeted from the deleted `LTM1.py`, the original bytes kept in the receipt.

**Recorded, not findings.** The two development repairs (`torch.when`, a mechanical rename caught by the thought tests before the freeze; one compiler-cache write refused by the sandbox and passed under the normal process) are in `commands.json`. `test_item25_repair`'s recalled-rows criterion is ported from the address digit to the `_rule_row` binding, with before and after text. §54's findings 2 to 7 are carried as §11.4 says.

**Verdict.** Accepted. `2d2abfe10878…` is round 1's frozen source. Codex commits and pushes on Alec's word, by §11.6; the WikiOracle submodule bump follows the push, as for item 2.3. The acceptance assumes Alec confirms the `MemoryStore` base as his direction; if he reverses it, a small round removes the base and its four-case certificate with one `make test`, and the landing text is withdrawn. Nothing is committed by me.

### 11.6 Codex text (Claude, 2026-11-08; for Alec to paste; final under the assumption that Alec confirms the `MemoryStore` base as his direction)

~~~
Item 2.0 round 1: land the frozen source. The round is accepted (2.0 spec doc/plans/2026-11-03-gradient-projection.md §11.5). Commit and push; change no code, data or test. Fast checks only, as throughout.

1. Check. The live tree's source hash by the receipt method (test/bounded_tests.source_snapshot(Path('.')), sha256 of json.dumps(files, sort_keys=True)) equals 2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa, the frozen source of doc/benchmarks/2026-11-08-item2-0-round-1. Since that sweep the handwritten documents changed only by appends (2.0 spec §11.5 and §11.6, one todo.md update). Run test/test_doc_links.py once on the live tree (BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest -q test/test_doc_links.py); it must pass. No make test: the one green sweep on this source (7521 passed, 325 skipped, 1 xpassed, exit 0) is its certificate and nothing in the source changed.

2. Commit, on main, every file the round changed, added or deleted: the modified tracked files, the deletions of bin/STMChunks.py, bin/MediumTermMemory.py and bin/LTM1.py, and the untracked bin/Memory.py, test/test_item20_round1.py and doc/plans/2026-11-08-handoff-2-0.md. The receipts under doc/benchmarks/ stay local under the existing ignore rule; force nothing in. Commit message in the form of 59cd466077: subject "Land accepted item 2.0 round 1"; body: land freeze 2d2abfe10878 with its certificates and the section 11.5 acceptance; what it carries (STM, MTM, LTM0 and LTM1 in one memory file with the kept close as their shared commit boundary and the three old modules removed; the stateless shared snapshot base; thought ideas stamped and found by their thought step's .when with an empty .where; the retired one-leaf word-occurrence substitution; no journal row for a SHIFT or transpose, empty sentinels in place of the row-0 alias); validation: one default make test on the frozen source, the landed-checkpoint load certificate, the direct before-and-after cases, test_item20_round1; generated evidence remains local under the ignore rules. No model or assistant identifier anywhere in the message.

3. Record the landing: append one line to todo.md after its last 2.0 update and one line to the 2.0 spec after §11.6, "Landed as <hash>, 2026-11-08"; commit as "Record item 2.0 round 1 landing"; then push main once (git push -u origin main). Report both hashes to Alec.

4. Then stop. Round 2 (repairs under fast checks; the measurements wait for the end of 2.0, §10.2) is a separate text; build nothing of it now.
~~~

**Confirmed (Alec, 2026-11-08):** the base class was his direction ("I asked for a base class to consolidate code"), and `MemoryStore` is its name because `Memory` is the module ("I think memory is taken"). The assumption of §11.5 and §11.6 is discharged; the landing text stands as final.

### 11.7 One Codex text for the landing and round 2 (Claude, 2026-11-08; Alec: "add the round two text to this prompt, otherwise we take too many iterations to finish 2.0")

Alec's instruction folds round 2 into the landing text, so that Codex lands round 1 and continues into round 2 without a further exchange. Round 2 is the repairs under fast checks of §10.2; the measurements wait for the end of item 2.0. The text below supersedes §11.6's step 4 ("then stop"); its steps 1 to 3 are §11.6's landing unchanged. One assumption is stated inside it: the supplied `yes` admission's home is this round (todo.md, item 2.0, "Carried": "Alec to confirm this home over item 0"); if Alec moves it to item 0, that item is withdrawn and nothing else in the text changes. The one-WholeSpace item changes behaviour on the parallel path (it reads trained memberships where it read the untrained copy 0), so round 2's certificates are "the same as before" only where a path already read the trained copy; every other change is certified by a direct before-and-after case. The round's return is §12.

~~~
Item 2.0: land round 1, then build round 2. Round 1 is accepted (2.0 spec doc/plans/2026-11-03-gradient-projection.md §11.5); Alec confirmed the MemoryStore base as his direction (§11.6). Steps 1 to 3 land round 1 and change no code, data or test; round 2 starts from that landed main with a clean tree. Fast checks only throughout: direct before-and-after cases, the targeted files, and one make test when you offer round 2's freeze. No item 71 runs, no lesson arms, no item 60 metrics, no repeats, no pre-checks: the measurements of item 2.0, the measured "before" included, are taken at the end of the item (§10.2).

1. Check. The live tree's source hash by the receipt method (test/bounded_tests.source_snapshot(Path('.')), sha256 of json.dumps(files, sort_keys=True)) equals 2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa, the frozen source of doc/benchmarks/2026-11-08-item2-0-round-1. Since that sweep the handwritten documents changed only by appends (2.0 spec §11.5 to §11.7, todo.md updates). Run test/test_doc_links.py once on the live tree (BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest -q test/test_doc_links.py); it must pass. No make test: the one green sweep on this source (7521 passed, 325 skipped, 1 xpassed, exit 0) is its certificate and nothing in the source changed.

2. Commit, on main, every file round 1 changed, added or deleted: the modified tracked files, the deletions of bin/STMChunks.py, bin/MediumTermMemory.py and bin/LTM1.py, and the untracked bin/Memory.py, test/test_item20_round1.py and doc/plans/2026-11-08-handoff-2-0.md. The receipts under doc/benchmarks/ stay local under the existing ignore rule; force nothing in. Commit message in the form of 59cd466077: subject "Land accepted item 2.0 round 1"; body: land freeze 2d2abfe10878 with its certificates and the section 11.5 acceptance; what it carries (STM, MTM, LTM0 and LTM1 in one memory file with the kept close as their shared commit boundary and the three old modules removed; the stateless shared snapshot base; thought ideas stamped and found by their thought step's .when with an empty .where; the retired one-leaf word-occurrence substitution; no journal row for a SHIFT or transpose, empty sentinels in place of the row-0 alias); validation: one default make test on the frozen source, the landed-checkpoint load certificate, the direct before-and-after cases, test_item20_round1; generated evidence remains local under the ignore rules. No model or assistant identifier anywhere in the message.

3. Record the landing: append one line to todo.md after its last 2.0 update and one line to the 2.0 spec after §11.7, "Landed as <hash>, 2026-11-08"; commit as "Record item 2.0 round 1 landing"; then push main once (git push -u origin main). Report both hashes to Alec in your return. Then continue with round 2 at once, from this landed main, without waiting.

Round 2: repairs under fast checks (todo.md item 2.0 "Rounds", round 2, as amended by 2.0 spec §10.2). Build nothing of rounds 3 to 6 (the projection and separateChooser false, the chooser unification and walk_policy's retirement, the verb and modal teachers, the link function, basic percepts, the loop). A failed test_convergence or shared-concept case stops the round; anything else is repaired at its producer under the recorded rules or recorded.

4. Fail loud in the reverse and in runBatch (loop spec doc/plans/2026-11-04-item2-0-order-zero-loop.md §3.9). Models.runBatch wraps reverseReconstruct in except Exception and zeroes lossRev ("never let a reconstruction edge case stop the training step", about line 14884); the stage loop of Models._reverse_body (about 24272) swallows a RuntimeError at a stage. Both handlers go: a decoder that raises inside a stage propagates out of _reverse_body, out of reverseReconstruct and out of runBatch. ConceptualSpace.reverse (Spaces.py about 18925) reads an undefined name _inputShape on ProjectionBasis configurations, a latent NameError; repair it to the subspace's recorded input shape. Whatever the removed handlers were hiding is a producer defect: repair it at its producer, never by narrowing a test or re-adding a handler; a repair that needs a decision not in the record stops that item with the evidence. Certificates: a direct case in which one stage is patched to raise and the raise propagates through all three callers; the reverse and reconstruction files (test_dual_input_contract, test_radix_recon_render, test_free_reconstruction, test_output_walk, test_conceptual_recurrence's combine round-trip) pass; a ProjectionBasis configuration runs ConceptualSpace.reverse.

5. The derivation inverse given the forward's action context (2.0 spec §5, "A defect to fix in any case"). Layers.reverse_blocks(parent, reference, *, counts, reference_right, action_context=None, whole=None) already takes the forward's action context (object_value and the verb group), but DerivationReconstruction.inverse_step (about line 97) calls it with none, so a transitive lift that acted in its object's basis is undone in the identity basis and reconstruction pushes transitive verbs' object bases toward the identity; a grouped intransitive verb is affected the same way; the adverb action has one basis and is unaffected. Pass the recorded action context of the journal's transaction at that call, and at Generative.native_recompose's compose_blocks for the proposal's forward map, from the same record the forward used (the lift operand's object value and group as the journal records them; where the record is absent the inverse raises, fail loud, rather than defaulting to the identity). Certificates: a direct before-and-after case in which a transitive lift with a non-identity object basis round-trips exactly through inverse_step after the change and did not before (numbers in the receipt); the same for an intransitive verb with a group; an adverb round-trip unchanged; the reconstruction and verb files (test_verb_learning, test_modal_timelines, test_free_reconstruction, test_grammar_word_learning) pass, ported with before/after text only where a test asserted the identity-basis inverse.

6. One WholeSpace (basic-percepts spec doc/plans/2026-11-04-item2-0-basic-percepts.md §7 D3 and gate 8; Alec: "we should not need copies"). Exactly one WholeSpace on every configuration, whatever bindingDepth is: the construction loop in Models (about 16658 to 16844) builds one instance; self.wholeSpaces (the nn.ModuleList) goes, with no alias, and every reader of wholeSpaces[0] or wholeSpaces[-1] (Models about 6899, 8472, 11261; Spaces about 15756; ConceptLessons 23; ModelCandidateAttention; ModelAttention; Occurrence; Mereology; WhereRegistry; checkpoint_migrations; the about 61 references in Models) reads self.wholeSpace. bindingDepth keeps counting the ConceptualSpace stages and the ramsification slots. The four latent inconsistencies close by construction: the word fill (WordIdentity.refresh_property_fills) and the serial open read the same members; the byte cut and the lessons teach the one inventory; PerceptField and SentenceAttention give a property row one where-address (today 27 apart on data/BasicModel.xml). Checkpoints: migrate_wholespace_checkpoint (checkpoint_migrations.py) keeps the copy whose members or observed differ from their initial rows (the last on a serial checkpoint, copy 0 on a parallel one), raises if more than one differs, merges copy 0's host extras (_standalone_run_bytes, _property_rows_used, the LBG-split rows) and drops the rest; only checkpoints written since 2026-10-27 need it, older ones already fail on the WholeSpace width. The where registry's wholes slice shrinks by (T-1) x nVectors, which shifts the symbol and conceptual offsets: certify whether any persisted record (LTM0 bands, MTM, the journal, the registry's own saved slices) holds an absolute address in those slices, and migrate it if so, with the certificate in the receipt either way. ModelCandidateAttention's input width falls (189 inputs on data/BasicModel.xml): its weights in old checkpoints are rebuilt and the rebuild reported. Two readings change on the parallel path and are reported, not repaired here: PerceptProperties.reverse and reverse_poles divide by a byte sum floored at 1e-12 (about 214 and 222) and now read trained memberships through _reverse_field: count in the receipt how often the floor is reached on the reverse tests, with no change to the division (basic percepts replace that read in round 5); FormCentroids' compose-time whole read keeps its logic and its presence weighting. Certificates: one instance on every configuration (a direct case over data/model.xml, data/BasicModel.xml and one XOR configuration); the migration's selection, its raise on two differing copies and its merge of copy 0's extras, on one serial and one parallel checkpoint written before the change; the persisted-address certificate; PerceptField and SentenceAttention agreeing on a property row's address; the word fill and the open reading one members; the about 34 test files that count or key WholeSpace copies ported with before/after text; data/model.xsd's and doc/Spaces.md's descriptions of the copies replaced exactly; Params.md where it names them.

7. The supplied yes admission (2.7 corpus spec doc/plans/2026-10-27-item2-7-corpus.md §38.1 decision F1 and §47: 31 of 32 lesson-13 absences were "yes" missing from the terminal bank, absent in all 576 applicable banks). Its home is this round; if Alec moves it to item 0 this item is withdrawn. Build §38.1's three-stage report for every supplied form (admitted, shortlisted, exact route available), recorded by the lessons runner beside its lexical forms, and certify the admission path from Corpus.admit_training_forms to Generative.lexical_candidates: a direct case supplies a one-word answer whose form is in the training vocabulary and shows it admitted, shortlisted and routable in every applicable terminal bank; "yes" is present in every applicable bank on the lesson-13 fixtures without running a lesson. No special case for any word or language: the rule is for every supplied form.

8. The sentence-82 capture rule (todo.md item 2.0 "Carried"; 2.7 corpus spec about line 3451: ValueError('an identity requires one finite working value') in MediumTermMemory._value, reached from the deferred mtm.bind in ReferenceClose.commit, never captured, not recurred in eight later runs). At ReferenceClose.commit's mtm.bind calls (about 1339 to 1379 on the current tree), when MediumTermMemory._value (bin/Memory.py about 1055) rejects, capture before re-raising: the rejected working value, the token and its members, the pool state, the stream, document and position, into the run's receipt directory (the lessons runner's output directory when one is set, else output/); then re-raise unchanged. No clamp, no fallback, no retry; the repair is at the producer when a capture shows it. Certificate: a direct case injects a non-finite working value and shows the capture file written and the raise propagated; corpus_lessons.measure runs with the rule installed on one presentation, no lesson arm.

9. Events at or above 1e8 under separateChooser false (round 2 as listed): item 71's last runs reported none in any audited class (thinking spec §56 to §58), so there is nothing to repair; record that the item is empty and re-opened only if the end-of-2.0 runs show such an event.

10. Documents, by append or exact replacement: Architecture (the reverse and runBatch fail loud; the inverse with its action context; one WholeSpace and the migration; the admission report; the capture rule), Spaces.md and data/model.xsd (the copies, exact replacement), Params.md, GradientFlow where it names the identity-basis inverse, Testing.md one line, FutureWork (the deferred measurements point to §10.2; the PerceptProperties floor count); your return as §12 of the 2.0 spec, with each item's before-and-after numbers.

Carried, not built here (record them in the return): §54 finding 2 (round 5), findings 4 and 5 (round 3), findings 6 and 7 (records); the item 60 metrics and item 71's runs (the end of item 2.0, §10.2); the momentum-carried projection and a learned pair policy (only if the receipts call for them).

Testing: fast checks only until the implementation is finished. Each item's direct before-and-after cases, the targeted files (the reverse and reconstruction files of item 4; the verb and reconstruction files of item 5; the about 34 WholeSpace files, the checkpoint and migration tests, test_coordinate_registry and the attention files of item 6; the corpus admission and lessons files of items 7 and 8; the files you port) and make test once when you offer the freeze (env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected; the XOR class, the pending-premise pair and the shared-concept case run inside it). Expect round 1's counts (7521 passed, 325 skipped, 1 xpassed) plus the new direct cases, less any case a port changes, each named. No item 71 runs, no lesson arms, no repeats, no pre-checks. Receipt under doc/benchmarks/<date>-item2-0-round-2/ with commands.json, frozen-source.json, before-source.json (round 1's landed source 2d2abfe10878...), source.patch, ports.json (every port's before/after text), the two pre-change checkpoints of item 6, offered-documents.json sealed before the sweep; every file your return links to exists before the sweep starts. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud. Return through Alec for Claude's review; no commit or push of round 2.
~~~

**Confirmed (Alec, 2026-11-08): round 2 is the home of the supplied `yes` admission** ("Yes, round 2"), not item 0. The assumption stated in §11.7 item 7 is discharged; §7's "pending Alec's confirmation of that home over item 0" is closed. The §11.7 text stands as final with no open assumption.

Landed as 6becc3dc47c88d787406ff94f1a4ba6acbf95eec, 2026-11-08


## 12. Round 2 interim return through Alec (2026-11-08)

**Historical interim status; decoding interpretation amended by §13 below.**
Repairs and targeted certificates, not an offered freeze. At this return,
item 5's free-generation routing was recorded as needing a decision. Its
recorded derivation path was implemented; the independent items below were
implemented and checked. The one
`env -u RUN_TRAINED -u RUN_SLOW make test` has **not run**. No round-2 commit or
push has occurred. This return is for Alec to pass to Claude; no external
message was sent.

### 12.1 Landing and scope

Round 1's live source matched
`2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa` (1004 files).
Its landing-only document check passed **1000 cases in 124.52s**; its accepted
**7521 passed, 325 skipped, 1 xpassed** default sweep was not repeated.
The code landing is `6becc3dc47c88d787406ff94f1a4ba6acbf95eec`; the landing-line
commit is `04164eddb4ed0773a849d8e5e491979e39b8cd51`. One successful
`git push -u origin main` pushed both. Round 2 began on that clean landed main.

The local [command record](../benchmarks/2026-11-08-item2-0-round-2/commands.json)
retains successes, failures, interrupted instrumentation and the reasons for
corrective targeted runs. The [before source](../benchmarks/2026-11-08-item2-0-round-2/before-source.json),
[current source](../benchmarks/2026-11-08-item2-0-round-2/current-source.json),
[patch](../benchmarks/2026-11-08-item2-0-round-2/source.patch) and
[exact ports](../benchmarks/2026-11-08-item2-0-round-2/ports.json) are local evidence.
`current-source.json` is an interim snapshot, not the final `frozen-source.json`.
The final frozen source and sealed offered-document manifest wait for the
routing decision and completed implementation, before the single default sweep.

### 12.2 Fail-loud reverse and recorded action inverse (items 4–5)

Before, the injected decoder exception propagated through **0/3** callers;
`_reverse_body` swallowed it and subsequently failed elsewhere, while the other
two callers did not propagate it. After, **3/3** propagate the same injected
exception object: `_reverse_body`, `reverseReconstruct`, and `runBatch`.
Removing the handlers exposed an aligned-path attempt to invert a mixing
combine that did not exist. The producer now invokes that inverse only for
mixing; it still raises if a required mixing combine is absent.

ProjectionBasis reconstruction uses the recorded input shape and its native
coefficient carrier. Generic codebook reverse materialization incorrectly
reshaped those coefficients as feature vectors; the producer now materializes
coefficients before applying the projection inverse. The configured
ConceptualSpace reverse case passes with the expected finite output shape.
The required reverse files pass **33, skip 28**. No handler was restored.

The journal now preserves the selected forward object's value, presence and
verb group beside the numerical operation frame. Replay and forward-discard
accounting read that record; a missing non-adverb action context raises.
The context owns its value snapshot while preserving its gradient path.
`native_recompose` accepts the same explicit context. Direct double-precision
results ([before](../benchmarks/2026-11-08-item2-0-round-2/direct-before.json),
[after](../benchmarks/2026-11-08-item2-0-round-2/direct-after.json)):

| Action | Inverse maximum error before | After | Recomposition error after |
|---|---:|---:|---:|
| Transitive, non-identity object basis | 0.008796397368599718 | 1.1102230246251565e-16 | 0 |
| Intransitive, verb group | 0.001113328976186767 | 1.1102230246251565e-16 | 0 |
| Adverb | 1.1102230246251565e-16 | 1.1102230246251565e-16 | 0 |

**Item 5 stops at the free-proposal boundary.** Replaying an existing sentence
has a particular forward transaction to undo. Generating a new sentence can
propose several verbs and objects and has no such transaction. Item 5 asks for
the same forward record, while §3 and the Generative module contract forbid
feeding the input's compose history into free generation. Selecting a matching
input-journal record would violate that boundary; that suggestion was withdrawn.
One possible resolution is to derive context from the proposal's own verb and
object identities. The record does not yet authorize/specify that routing.
The [pending decision](../benchmarks/2026-11-08-item2-0-round-2/pending-generation-routing.json)
separates implemented replay from unimplemented free-proposal context construction
and routing. No fallback or journal lookup was added to free generation.

### 12.3 One WholeSpace and persistence (item 6)

`wholeSpaces` is removed without an alias. Every stage references `wholeSpace`;
bindingDepth still counts ConceptualSpaces and ramsification slots. All three
required configurations have exactly one registered WholeSpace. Before, the
serial checkpoint had **4** copies and the parallel checkpoint **3**; after,
each has **1**. The native pre-change
[serial checkpoint](../benchmarks/2026-11-08-item2-0-round-2/serial-before.ckpt) and
[parallel checkpoint](../benchmarks/2026-11-08-item2-0-round-2/parallel-before.ckpt)
strictly load, choosing copies **3** and **0**, respectively. Both native
geometries reject a second changed inventory and preserve copy 0's host extras.
The landed producer's native LBG split at row **9** retains its exact members
and observed mask when merged into a retained copy 2; conflicting row contents
raise. BasicModel.xml has no initially empty split seat. Split rows on initially
empty, used copy-0 rows are host additions, not a second trained base inventory.

The [native before](../benchmarks/2026-11-08-item2-0-round-2/checkpoints-before.json)
and [after](../benchmarks/2026-11-08-item2-0-round-2/checkpoints-after.json) certificates
record offsets, strict loads and candidate-attention rebuilding. Serial wholes
shrink **36 → 9** rows, shifting later addresses by **27**; candidate input width
falls **1,171,068 → 1,170,879** (**189**). Parallel wholes shrink **192 → 64**,
shifting later addresses by **128**; candidate width falls **16,236 → 15,340**.
All four candidate-attention parameters are rebuilt and reported. The selected
property parameter's optimizer slots survive; rebuilt reader slots start fresh.

The [persisted-address certificate](../benchmarks/2026-11-08-item2-0-round-2/persisted-address-certificate.json)
identifies absolute bands in LTM0 slots/recurrence/epoch snapshots, LTM1 percepts,
MTM values/events and corpus continuation; these migrate. Registry slices are
reconstructed, and no derivation journal is persisted. Both native checkpoint
samples have **0** moved occupied addresses. An explicit nine-address fixture
covers the persisted locations, including every MTM event-value position, and
preserves identity keys and original tensors. Old addresses
`[0,9,12,15,20,21,26,27,34]` become `[0,9,12,12,14,15,20,21,28]`.

The [reader certificate](../benchmarks/2026-11-08-item2-0-round-2/reader-certificate.json)
records one membership tensor for word fills and **2** serial open reads, plus
both readers assigning property row **0** address **8192**. The 32 existing test
and audit files with WholeSpace ports retain their cases; exact before/after
text is in ports.json. The old duplicate WholeSpace alias setup and duplicated
fingerprint entry are removed because the shared owner is already represented.
No test function was retired.

Parallel attribution now reads the shared trained inventory. FormCentroids'
compose read and presence weighting are unchanged. The
[floor observation](../benchmarks/2026-11-08-item2-0-round-2/reverse-floor-counts.json)
found **0 calls, 0 denominators, 0 floor hits** in each of `reverse` and
`reverse_poles` on the required default reverse selection. This is explicitly
empty coverage of those functions. The
[observer control](../benchmarks/2026-11-08-item2-0-round-2/floor-observer-control.json)
detects **1/2** and **3/4** floor hits on a separate two-row case. The divisions
are unchanged; basic percepts replace this read in round 5.

### 12.4 Admission, rejecting-close capture and empty large-event item (7–9)

The admission producer already admits the direct fixture's supplied answers.
The generic report now records admitted, shortlisted and exact-route-available
for every supplied lexical form at every generated bank, beside the lessons
runner's lexical forms and in `lexical-admission.jsonl`. It does not alter
ranking. The fixture census is **20/20 supplied-form banks passing all three
stages before and after**, including **8/8 yes banks**. The direct one-word
cases include a second arbitrary form. These are fixture banks, not a claim
that the historical 576 trained banks have been rerun. No lesson was run for
admission. [Before](../benchmarks/2026-11-08-item2-0-round-2/admission-before.json) ·
[After](../benchmarks/2026-11-08-item2-0-round-2/admission-after.json).

ReferenceClose's deferred binds capture a rejected MTM working value, token,
members, pool and stream/document/position, then re-raise the same exception.
The unchanged commit `no_grad` boundary remains in place. The direct injected
non-finite value produces **1 capture file** and the original exception object.
`corpus_lessons.measure` completed **1 presentation, 1 sentence, 1 batch** with
the rule installed; this was a micro case with no lesson arm and no claim of
lesson acquisition. [Capture rule certificate](../benchmarks/2026-11-08-item2-0-round-2/capture-rule-certificate.json).

The [large-event item](../benchmarks/2026-11-08-item2-0-round-2/large-events.json)
is empty: the last item 71 runs found no event at or above `1e8` in any audited
class. It reopens only if the end-of-2.0 runs show one. No item 71 run was made.

### 12.5 Validation and carried work

The wider ownership/checkpoint/admission selection produced **375 passed,
32 skipped, 5 failed**. Three failures were optional-owner accesses exposed by
the new ownership; two were compiler-cache sandbox denials. The five affected
cases pass after the producer repair and cache access. Capture/lessons targets
pass **70, skip 4**. The expanded persisted-address case passes separately.
Journal, inverse, verb and reconstruction results, including the repaired
journal record-width producer and native ProjectionBasis case, remain in
commands.json. There are **19 new direct cases** and no retired test functions;
the document-link check passed **1000 cases in 111.21s**. These targeted
counts are overlapping selections, not a substitute for a default-suite count.
The XOR convergence class, pending-premise pair and shared-concept case remain
reserved for the one final sweep. No convergence or shared-concept gate failure
has occurred; those gates have not been pre-run.

Nothing from rounds 3–6 is built. Carried from thinking §54: finding **2**
(elided-role content/full reference form) belongs to round 5; findings **4**
(output gradient does not reach understanding) and **5** (momentum-only
projection tail, including 318/320 re-clamps) belong to round 3; findings **6**
(at most 1/80 bound or exact lesson answers) and **7** (stale weekly slow record)
remain records. `make test_all` belongs to items 0 and 1. Item 60 metrics,
item 71 runs and the measured before wait until the end of item 2.0 under
§10.2. Momentum-carried projection and a learned pair policy remain conditional
on the receipts. No seed pinning, resize, lesson arm or end-item panel was used.


## 13. Alec's clarification: shared short-term strata and generative inversion (2026-11-08)

**Decided.** STM0 and STM1 are short-lived versions of LTM0 and LTM1, sharing
their implementation. Reconstruction, input-derived conditioning and MTM use
the same stored structure. The detailed amendment, the operation-representation
question and the implementation audit are in the
[memory plan §10](2026-11-03-item2-0-behavioral-chains.md#10-shared-stmltm-strata-and-decoding-from-an-idea-alec-2026-11-08).
This is more sharing than round 1's stateless snapshot base; it is planned work,
not an assertion that the current chunk/journal arrangement already provides it.

**The §12.2 blocker is withdrawn in its stated form.** A syntactic inverse is
not required to be exact or to have all the original forward witnesses. A
single LTM0 idea must support surface realization using active concepts,
matching and the verb's available inverse. Absence of an input transaction is
therefore not a reason to require or invent one for free generation. The
recorded-action round trips in §12 remain limited numerical replay certificates;
they must not become a prerequisite for the general decoder. The rule that a
broken calculation fails loudly remains; an underdetermined decomposition is
not by itself a broken calculation.

The earlier §5 and §11.7 item 5 language about reusing forward context applies
to replaying that known application. It does not require the free decoder to
recover the historical object's value or group from an input journal. A
candidate's forward check must agree with the candidate's own proposed action.
The existing round-2 code still requires review/completion against this
clarification; no new default sweep, code change or implementation acceptance
is implied by this documentation amendment.

**Conditioning.** Learn its context from input composition and use the composed
grammar for top-down context, with limited influence on syntactic categories
while grammar is unreliable. Architecture-generated sentences do not need a
separate context-learning pass. This conditioning context is distinct from a
verb application's object/group arguments; no new numerical weighting schedule
is specified here.

**Representation direction, clarified by Alec.** STM0 holds operator ideas as
well as word/concept ideas, with operations represented as ordinary concepts;
STM1 keeps ordered binary references. Alec suggests constructing the primitive
operator concepts a priori. The [memory plan §10.5](2026-11-03-item2-0-behavioral-chains.md#105-operator-ideas-in-stm0-ordered-binary-branches-in-stm1-alec-2026-11-08)
distinguishes a reusable operator concept from its temporary occurrences and
a priori identity/execution availability from learned content and confidence.
The exact nested application convention and initialization policy remain to be
specified. Argument categories can constrain application without being assumed
to identify every operation uniquely. No new per-branch operation column is
introduced by this direction. Current LTM1 already has `operator` and
`grammar_operation` trust/gating fields; their dependencies must be reconciled
with the concept representation, not silently dropped.

### 13.1 Codex text: round 2 continuation to the freeze (Claude, 2026-11-08, evening; for Alec to paste; final)

~~~
Item 2.0 round 2, continuation: complete item 5 against the 2.0 spec §13 and the memory plan §10.2 (doc/plans/2026-11-03-item2-0-behavioral-chains.md), then offer the freeze. Everything else in your §12 interim return stands. Fast checks only: direct before-and-after cases, the targeted files, and one make test when you offer the freeze. No item 71 runs, no lesson arms, no item 60 metrics, no repeats, no pre-checks (§10.2). Build nothing of rounds 3 to 6, nothing of memory plan §10 (the shared STM0/STM1 strata, the operator concept seats, the collapse of the pinned sentence's tree, authority onto the operator concepts' trust: planned work for the memory round; §10.6 keeps LTM1.operator and LTM1.grammar_operation as they are), and nothing of §13's conditioning paragraph. A failed test_convergence or shared-concept case stops the round; anything else is repaired at its producer under the recorded rules or recorded.

1. Item 5, completed. Replay stays as built (§12.2): DerivationReconstruction.inverse_step and the forward-discard accounting read the journal's action record (ChunkJournal.action_context: object_value, object_present, group), a missing non-adverb record raises, and the recorded round trips are its certificate. Free generation changes. §11.7 item 5's "from the same record the forward used" for the proposal's forward map is withdrawn by §13, and the decision in pending-generation-routing.json is made by it: a candidate's forward check uses the candidate's own proposed action. At the free walk's native_recompose calls (Generative about lines 112, 165 and 610) pass an action context built from the candidate itself: object_value is the candidate's object operand, the operand the forward records as object_value for that operation; object_present is that operand's availability for the candidate (the negation of its missing flag); group is the group the forward assigns the candidate's verb operand, computed by the same function the kept path uses when it writes the record (locate it where lang[19] fills the record, Models about 20084 and 22952). Where a candidate has no object or no group, the identity basis, exactly as MeaningAction.forward states ("unknown groups and absent objects have exactly the identity basis"). No journal, STM or desired-output read on the free path, as Generative's module contract says; nothing raises for an absent record there: an underdetermined decomposition is not a broken calculation (§13); a non-finite value still propagates. Add the resolution to pending-generation-routing.json (status resolved by §13, with the rule above); delete nothing. Certificates: a direct before-and-after case in which a parent composed by lift in a non-identity object basis is checked, for the free candidate that proposes that object, in that object's basis: the recomposition residual is within EXACT_RESIDUAL_TOLERANCE after the change and was not under the identity-basis check before, numbers in the receipt; the same for an intransitive candidate whose verb has a group; an adverb candidate unchanged; a case that runs the free walk with no action record and no live STM and shows it neither reads nor raises for them; the reconstruction and verb files (test_free_reconstruction, test_output_walk, test_verb_learning, test_modal_timelines, test_grammar_word_learning) pass, ported with before/after text only where a test asserted the identity-basis check of a free candidate.

2. Items 4, 6, 7, 8 and 9 as returned in §12. Re-read each once against §13 and record in your return "unaffected" or the change made; the §12.2 handlers stay removed, the one WholeSpace and its migration stand, the admission report, the capture rule and the empty large-event item stand.

3. Documents, by append or exact replacement, fast: Architecture's item-5 text says that replay reads the recorded action context and that free generation checks a candidate in the candidate's own object basis and group; GradientFlow where it names the identity-basis inverse; FutureWork one line pointing to memory plan §10 as the memory round's planned work (shared strata, operator seats, the collapse at the end of the pinned sentence, authority onto the operator concepts' trust; §10.6: the two LTM1 columns stay until that round); Testing.md if a line is needed for the new direct cases. Run test/test_doc_links.py once after the appends; it must pass.

4. The freeze. When item 5's certificates and the targeted files pass: write frozen-source.json by the receipt method (test/bounded_tests.source_snapshot(Path('.')), sha256 of json.dumps(files, sort_keys=True)); keep current-source.json as the interim record; before-source.json stays round 1's landed source 2d2abfe10878...; source.patch and ports.json cover every change since that source, item 5 included; seal offered-documents.json before the sweep over every handwritten document the return links to, including today's appends to this spec (§13, §13.1), the memory plan (§10 to §10.6) and todo.md; create sweep-result.json and final-audit.json as placeholders before the sweep; every file your return links to exists before the sweep starts. Then exactly one env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected; the XOR class, the pending-premise pair and the shared-concept case run inside it. Expect round 1's counts (7521 passed, 325 skipped, 1 xpassed) plus the new direct cases (your §12.5 counted 19; item 5 adds its own), less any case a port changes, each named. Source and sealed documents unchanged after the sweep, certified in final-audit.json.

5. Return as §14 of this spec: the §12 interim numbers carried forward where unchanged, item 5's before-and-after numbers, the §13 re-read of items 4 and 6 to 9, the sweep counts, and the carried list (your §12.5's: §54 findings 2, 4, 5, 6 and 7 as placed there; item 60 metrics, item 71 runs and the measured before at the end of item 2.0 under §10.2; the momentum-carried projection and a learned pair policy only if the receipts call for them; memory plan §10 to the memory round). Through Alec for Claude's review. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud. No commit, no push of round 2.
~~~

### 13.2 Schedule with fewer passes (Alec, 2026-11-08, evening)

Alec: "lets schedule a few fewer passes", then "Merge 3 and 4". The rounds
after round 2 (todo.md, item 2.0, "Rounds") are rescheduled without cutting
their content:

- **Rounds 3 and 4 are one round** with one freeze, one review and one
  landing: `separateChooser` false with the one-sided projection and the
  sides (§§1–2, with 2.5 §54 findings 4 and 5), and generate under compose's
  proposal and credit, `walk_policy` and the verb and modal teachers retired,
  the XOR gate run once, output typing. Round 5 (memory and basic percepts)
  also carries the memory plan §10 amendment. Round 6 is the loop.
- **The end-of-item measurements are part of round 6's freeze** (§10.2:
  item 71's runs and the item 60 metrics on round 2's frozen source and the
  finished tree), returned with round 6, not as a pass of their own.
- **Every landing is folded into the next build text**, as §11.7 was. Round
  6's landing is §11.7's steps 1 to 3 on Alec's one line to Codex; no further
  Claude text.
- **Reviews accept with recorded warts** unless a gate fails: a failed
  `test_convergence` or shared-concept case stops a round; anything else is
  carried into the next text, as the 2.3 reviews did.

Three Claude texts remain after round 2's §14 return: land 2 and build 3/4;
land 3/4 and build 5; land 5 and build 6 with the measurements. Rounds 5 and
6 stay apart: round 5 is already a merge and has grown, and the loop depends
on it.

## 14. Round 2 offered freeze, through Alec for Claude (2026-11-08)

Item 5's free-generation routing is completed under §13. Replay retains its
recorded action context and missing-record error. Free candidates propose their
own action context from the supplied concept bank: `lift` uses the object of
its proposed verb phrase, whose bounded native decomposition supplies that
operand and its availability. A lexical verb has no object. Both paths use
`LanguageSpace.verb_group_table` and `Generative.verb_groups` for the same
declared verb assignment. Unknown groups and absent objects keep MeaningAction's
identity basis. Pair ranking, the selected inverse and all free recompositions
use that proposal, without input-journal, live-STM or desired-output reads.
The temporary candidate tensors allocate no STM/LTM rows or operator seats.
The [routing record](../benchmarks/2026-11-08-item2-0-round-2/pending-generation-routing.json)
preserves its earlier statements and adds the §13 resolution.

### 14.1 Item 5 certificates

The same fixed candidate operands and native operation states were checked on
the landed round-1 source and the repaired source. The residual is the largest
of the form and two meaning-lane MSEs; `EXACT_RESIDUAL_TOLERANCE` is `1e-8`.
[Before](../benchmarks/2026-11-08-item2-0-round-2/free-action-before.json) ·
[After](../benchmarks/2026-11-08-item2-0-round-2/free-action-after.json).

| Free candidate | Identity-basis check before | Candidate-basis check after |
|---|---:|---:|
| Transitive, non-identity object basis | 9.368469618548211e-5 | 7.703719777548943e-34 |
| Intransitive, assigned verb group | 7.852916027480765e-6 | 7.703719777548943e-34 |
| Adverb, unchanged | 7.703719777548943e-34 | 7.703719777548943e-34 |

The after meaning MSEs are exactly zero in all three cases; the remaining
form rounding error is at most `5.551115123125783e-17` in coordinates.
The [twelve new direct cases](../../test/test_item20_round2_generation.py)
pass, including free walks with journal/STM reads forbidden, eager and compiled
selected pullbacks, mixed-stream object/group isolation, unknown-basis identity
and a non-finite action that still raises. The compiled cases exposed an
object-vector contraction pullback whose strides differed from the unused
conditional branch. The producer now uses the equivalent linear weight view;
both compiled cases and nine existing action/replay certificates pass. No
existing test assertion was ported for item 5. Initial compiled failures and a
receipt fixture's missing group `used` count are retained in the command logs;
the corrected native fixture and fixed-operand checks pass.

The five requested files (`test_free_reconstruction`, `test_output_walk`,
`test_verb_learning`, `test_modal_timelines`, `test_grammar_word_learning`) pass
**60, skip 7** in **132.61s**. The completed direct file passes **12** in
**19.20s**. These are targeted selections, not the default sweep.
[Commands and logs](../benchmarks/2026-11-08-item2-0-round-2/commands.json).

### 14.2 One §13 re-read of the independent repairs

| Item | §13 result | Carried §12 evidence |
|---|---|---|
| 4, fail-loud reverse | **Unaffected.** Removed handlers remain absent; ProjectionBasis keeps its native coefficient carrier. | Injected exception propagation **0/3 → 3/3**; required reverse files **33 passed, 28 skipped**. |
| 6, one WholeSpace | **Unaffected.** Shared inventory, strict migration and address translation stand. | Native serial **4 → 1**, parallel **3 → 1**; retained copies **3**, **0**. Wholes **36 → 9**, **192 → 64**; later addresses shift **27**, **128**. Candidate widths **1,171,068 → 1,170,879**, **16,236 → 15,340**. Both strict loads pass, two changed inventories reject, native LBG row **9** survives. |
| 7, admission report | **Unaffected.** Free-bank group staging does not change retrieval, ranking or reports. | **20/20** supplied-form banks pass all three stages before/after, including **8/8 yes** banks; no trained-bank or lesson rerun. |
| 8, rejecting-close capture | **Unaffected.** The same exception is captured and re-raised; commit stays under `no_grad`. | **1** rejected-value capture; **1 presentation, 1 sentence, 1 batch** micro case. |
| 9, large-event item | **Unaffected; empty.** | No audited event at or above `1e8` in the last item-71 records; no item-71 run here. |

The replay certificate remains: object inverse error
**0.008796397368599718 → 1.1102230246251565e-16**, grouped intransitive
**0.001113328976186767 → 1.1102230246251565e-16**, adverb
**1.1102230246251565e-16** unchanged, recomposition **0** afterward.
The persisted-address fixture keeps the nine mappings listed in §12.3; native
samples have **0** moved occupied addresses. The shared-reader certificate
still has **2** serial reads and property row **0** at address **8192**.
Floor observation remains explicitly empty (**0 calls, 0 denominators,
0 hits** in each function); the separate control remains **1/2**, **3/4**.
The overlapping wider targeted results remain **375 passed, 32 skipped,
5 initially failed**, with all five repaired cases passing, and capture/lessons
**70 passed, 4 skipped**. Their receipts remain linked in §12.

### 14.3 Sealed offer and the single sweep

This section is written before the sweep so the offered prose can remain
unchanged. **The actual sweep counts, critical-gate outcomes and verdict are
the fields in [sweep-result.json](../benchmarks/2026-11-08-item2-0-round-2/sweep-result.json)**;
the placeholder is filled only after the one execution. Likewise,
[final-audit.json](../benchmarks/2026-11-08-item2-0-round-2/final-audit.json)
records whether both the source and sealed documents stayed unchanged.
A placeholder or failed audit is not a passing freeze.

The exact command is `env -u RUN_TRAINED -u RUN_SLOW make test`, without a
selection or deselection. The XOR class, pending-premise pair and shared-concept
case belong inside it; no gate was pre-checked. Expected counts are round 1's
**7521 passed, 325 skipped, 1 xpassed**, plus **31 new direct cases** (the
previous **19**, plus item 5's **12**): **7552 passed, 325 skipped, 1 xpassed**.
No retired case or changed parameterization reduces that expectation.
A failed convergence/shared-concept gate stops the round; other failures are
recorded on this frozen source, without another full sweep.

[frozen-source.json](../benchmarks/2026-11-08-item2-0-round-2/frozen-source.json)
uses `bounded_tests.source_snapshot(Path('.'))` and SHA-256 of
`json.dumps(files, sort_keys=True)`. The round-1
[before-source.json](../benchmarks/2026-11-08-item2-0-round-2/before-source.json)
remains `2d2abfe10878a190926963ac5727ce6275e292da0665aff6368fbd1b51790faa`;
[current-source.json](../benchmarks/2026-11-08-item2-0-round-2/current-source.json)
remains the historical interim snapshot. The
[complete patch](../benchmarks/2026-11-08-item2-0-round-2/source.patch) and
[exact before/after ports](../benchmarks/2026-11-08-item2-0-round-2/ports.json)
cover all changes since landed round 1, including the 32 existing test/audit
ports and all four new direct files. No test function is retired.

[offered-documents.json](../benchmarks/2026-11-08-item2-0-round-2/offered-documents.json)
seals the handwritten documents before the sweep, including this spec's §§13,
13.1, 13.2 and 14, the [memory plan §§10–10.6](2026-11-03-item2-0-behavioral-chains.md),
[todo.md](../../todo.md), [Architecture](../Architecture.md),
[GradientFlow](../GradientFlow.md), [FutureWork](../FutureWork.md),
[Testing](../Testing.md), [Spaces](../Spaces.md) and [Params](../Params.md).
The once-only post-append document-link result is also recorded in the sweep
receipt before execution. Every file linked by this return exists before the
sweep starts. Round 2 is not committed or pushed.

### 14.4 Carried work

Thinking §54 finding **2** stays in round 5; findings **4** and **5**, including
the **318/320** momentum-only projection re-clamps, stay in the merged round
3/4 of §13.2. Findings **6** (at most **1/80** bound/exact lesson answers) and
**7** (stale weekly slow record) remain records. `make test_all` belongs to
items 0 and 1. Item 60 metrics, item 71 runs and the measured before stay at
the end of item 2.0 under §10.2, inside round 6's freeze under §13.2.
Momentum-carried projection and a learned pair policy remain conditional on
those receipts. Memory plan §10 stays in the memory round: shared strata,
operator seats, pinned-sentence collapse and authority on operator-concept
trust. Its §10.6 keeps both LTM1 provenance columns until that round.
No conditioning, rounds 3–6 implementation, seed pinning, resize, lesson arm
or end-item panel is included in this offer.

### 14.5 Review of the §14 return (Claude, 2026-11-09): accepted; round 2 is frozen at `8cde24380d44…`; one red case, ported at the landing

Verified on the live checkout, logs under Claude's scratchpad:

- **Source.** The live tree's hash by the receipt method equals [frozen-source.json](../benchmarks/2026-11-08-item2-0-round-2/frozen-source.json): `8cde24380d447493b0500898e6cfe4b6242d794ac763ebaf9061aab7be4e7387` (1008 files). [before-source.json](../benchmarks/2026-11-08-item2-0-round-2/before-source.json) is round 1's landed `2d2abfe10878…` (1004 files). 57 files differ between them: 20 under `bin/`, `data/model.xsd`, the 32 test ports and the 4 new test files, as [source.patch](../benchmarks/2026-11-08-item2-0-round-2/source.patch) (4408 lines, read in full) and [ports.json](../benchmarks/2026-11-08-item2-0-round-2/ports.json) list (no retired test function; no item-5 assertion port). HEAD is `04164eddb4`, equal to `origin/main`; round 2 is not committed or pushed.
- **Documents.** The 121 files of [offered-documents.json](../benchmarks/2026-11-08-item2-0-round-2/offered-documents.json) re-hashed now: none changed; the sealed hash `76b749ca…` is the one [final-audit.json](../benchmarks/2026-11-08-item2-0-round-2/final-audit.json) records before and after the sweep. Every document change is an append or an exact replacement; §14 follows §13.2.
- **The sweep.** One `env -u RUN_TRAINED -u RUN_SLOW make test`, 7878 cases, 921 s: **7551 passed, 325 skipped, 1 xpassed, 1 failed** against the expected 7552. The gates ran inside it and passed: `test_mm_xor.py::TestMMXorConvergence::test_convergence` (6.6 s), the shared-concept case `test_item25_completion.py::test_a_previously_named_shared_concept_denotes_its_raised_class`, and the pending-premise pair `test_math_chain_section14.py::test_forced_ordinary_pending_premise_in_both_orders[False]`/`[True]` (the live pair's two skips are the standing `RUN_SLOW` skips, as in round 1). The one failure is `test_operators_round1.py::test_walk_teacher_covers_binary_unary_and_leaf_stop_without_code_gradient`: `AttributeError: 'types.SimpleNamespace' object has no attribute 'verb_groups'` at `BasicModel._decomposition_walk_teacher_loss` (bin/Models.py about 20359), reproduced here (48 passed, 1 failed over the four new files and that file). The producer reads `bank.verb_groups`, a field every real bank carries (`PrimedSymbols.verb_groups`, default `None`, bin/SentenceUnderstanding.py about 81); the certificate's hand-built `SimpleNamespace` bank predates the field and omits it. The repair is the one-line fixture port `verb_groups=None`, with before/after text in ports.json, not the `getattr` default Codex proposed in `sweep-result.json`: a default in the producer would hide a missing field on a real bank, and the bank's contract is the dataclass.
- **Certificates**, rerun from copies of the receipt's scripts against the live tree: free-action after, direct after, admission after, reader, floor-observer control and checkpoints-after (both native checkpoints, strict loads). All six JSONs equal Codex's apart from the run path: after residuals `7.7e-34` with meaning MSE 0 and maximum error `5.55e-17` in all three cases; replay inverse `1.11e-16` and recompose 0; the injected stage failure propagates through `_reverse_body`, `reverseReconstruct` and `runBatch`; 20/20 supplied-form banks admitted, shortlisted and routable; property row 0 at address 8192 with the wholes slice `[8192, 8201)`; floor control 1/2 and 3/4; serial copy 2 and parallel copy 0 retained, two changed copies rejected, native split row 9 merged, four candidate-attention tensors rebuilt.
- **Targeted files.** The five generation files **60 passed, 7 skipped** (149 s; Codex 60/7). The five reverse files **33 passed, 28 skipped** (120 s; Codex 33/28). Capture and lessons **70 passed, 4 skipped** (21 s; Codex 70/4). Action-layout regressions **9 passed** (10 s; Codex 9). The 39-file WholeSpace, checkpoint and admission selection **380 passed, 32 skipped** (232 s; Codex 375 passed, 32 skipped, 5 failed before the compiler-cache repair, then 5 passed).
- **Items 4 to 9 against §11.7 and §13.** The two handlers are gone; the aligned path skips the mixing-only combine inverse instead of swallowing its error; `ProjectionBasis` reverse reads `self.inputShape` and materialises before the basis inverse. The journal frame widens from `3D` to `4D+3` with the forward's object value, presence, group and a validity flag (`ChunkJournal.value_record`/`action_context`), read by `DerivationReconstruction.reconstruct` and `SentenceCredit.forward_discard`; a missing non-adverb record raises; `LanguageOperationChoice.action_context` is cloned at the selected position so a later refresh cannot replace it. Free generation proposes from the supplied bank only (`propose_actions`, `ActionCandidates`, `candidate_action_context`), threaded through `inverse_menu`, `selected_inverse`, `selected_residual`, `eligibility_menu`, `decoder_eligibility`, `_bounded_binary_reconstruction` and the walk-teacher loss, with the verb group from `LanguageSpace.verb_group_table` shared with the kept path. One `wholeSpace` is built at stage 0 and every `cs.wholeSpace_ref` points at it; the registry's wholes slice is one inventory; the migration selects the changed copy, merges copy 0's extras and split rows, translates persisted bands and rebuilds the candidate reader. The supplied-form report follows retrieval in `lexical_candidates` with the lessons-runner sink; `ReferenceClose._bind_with_capture` captures and re-raises; item 9 is recorded empty.

**Recorded warts, no repair in this round.**

1. `test_definition_writer._checkpoint_model` was ported to `model.spaces, model.conceptualSpaces, model.wholeSpace = None, [cs], []` (`spaces` is `None`, `wholeSpace` an empty list); it passes because that model never walks `self.spaces` and `hasattr([], 'property_extras')` is false. The intended line is `[], [cs], None`: a one-line port at the landing, before/after text recorded.
2. The one WholeSpace keeps `is_last = True` and `quantize = False` (bin/Models.py about 16817); no reader of a WholeSpace consults either (the `is_last` reader is the merge glue), so the former per-stage `quantize = not is_last` was already dead. Two dead attributes, for the memory round to delete.
3. `_migrate_whole_addresses` recognises a persisted location band by re-encoding its decoded index and comparing within `2e-5` (bin/checkpoint_migrations.py about 258); content that happened to equal a valid band would be rewritten. Certified on the fixture and on both native checkpoints (0 moved occupied addresses); it runs only on multi-copy checkpoints written before this round.
4. `Generative.walk_bank` resolves concept ids for the verb-group table with a Python loop over every bank row per batch (about line 329), a host island beside the one that already builds surfaces. Measured with the other host loops at the end of the item, not now.
5. `ReferenceClose._bind_with_capture` reads the rejected value out of `MediumTermMemory._value`'s frame (a local named `value`) by walking the traceback; a renamed local would lose the capture silently while the raise still propagates. The producer form is for `_value` to attach the rejected value to its exception; for the memory round, which owns `Memory.py`.
6. The failed walk-teacher case above: a one-line fixture port at the landing.

**Verdict: accepted under §13.2.** No gate failed; the one failure is a fixture port. **To confirm with Alec, before the landing text:** land the frozen source plus the two one-line test ports (`verb_groups=None` in the walk-teacher fixture; `[], [cs], None` in `_checkpoint_model`), rerun only `test_operators_round1.py` and `test_definition_writer.py`, and record the landed hash beside the sweep certificate on `8cde2438…` with the two-line delta named; or run a second full sweep after the ports. Claude recommends the former: the delta is two test lines, production is untouched, and §14.3 records other failures on the frozen source without another sweep.

**Confirmed (Alec, 2026-11-09: "Fine with landing").** Round 2 lands with the two one-line test ports and a targeted rerun of the two files; no second sweep. The landing is folded into the next Codex text with the merged round 3/4.

### 14.6 One Codex text: land round 2, build the merged round 3/4 (Claude, 2026-11-09; for Alec to paste; final)

Steps 1 to 3 land round 2 as §14.5 confirmed; items 4 to 11 build rounds 3 and 4 as §13.2 merged them. Two readings of recorded text are stated inside the text as its assumptions, not left open: the decoder walk's credit stays on the R side this round (§2, and §8's list of interim bookings that move only with the loop), while every other choice reward is on A whole (§8 "Sequencing"); and §54 finding 5 of the thinking spec is read as calling for the small producer repair (the clamp's stored outward velocity), while the momentum-carried gradient projection of §1 keeps its condition on the end-of-item receipts. The rates parameter is named `training.actionRateRatio` ($\kappa$, default 1; §7, "keep in model file").

~~~
Item 2.0: land round 2, then build the merged round 3/4. Round 2 is accepted (2.0 spec doc/plans/2026-11-03-gradient-projection.md §14.5); Alec confirmed the landing form (§14.5, "Fine with landing"): the frozen source plus two one-line test ports, a targeted rerun of the two files, no second sweep. Steps 1 to 3 land round 2; round 3/4 starts from that landed main with a clean tree. Fast checks only throughout: direct before-and-after cases, the targeted files, the one gate run of item 8, and one make test when you offer the freeze. No item 71 runs, no lesson arms, no item 60 metrics, no repeats, no pre-checks: item 2.0's measurements are taken at the end of the item (§10.2, §13.2). Build nothing of rounds 5 and 6 (memory plan §10, basic percepts, the loop, the measurements). A failed test_convergence or shared-concept case stops the round; anything else is repaired at its producer under the recorded rules or recorded.

1. Port and check. Two one-line test ports, with before/after text in doc/benchmarks/2026-11-08-item2-0-round-2/landing.json: in test/test_operators_round1.py the walk-teacher case's SimpleNamespace bank (about line 94) gains verb_groups=None, the field every PrimedSymbols carries; in test/test_definition_writer.py _checkpoint_model (about line 108) reads model.spaces, model.conceptualSpaces, model.wholeSpace = [], [cs], None. Nothing under bin/ changes. Run the two files (BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest -q test/test_operators_round1.py test/test_definition_writer.py); both green. Record the tree's source hash by the receipt method (test/bounded_tests.source_snapshot(Path('.')), sha256 of json.dumps(files, sort_keys=True)) as landed-source.json beside frozen-source.json, naming the two files as the whole difference from 8cde24380d447493b0500898e6cfe4b6242d794ac763ebaf9061aab7be4e7387. Since the sweep the handwritten documents changed only by appends (§14.5 and its confirmation, todo.md lines); run test/test_doc_links.py once; it must pass. No make test: the one sweep on 8cde24380d44… (7551 passed, 325 skipped, 1 xpassed, 1 failed, the failure the first port) plus the two-file rerun is the certificate.

2. Commit, on main, every file round 2 changed, added or deleted, with the two ports: the modified tracked files under bin/, data/model.xsd, doc/, test/ and todo.md, and the four new files test/test_item20_round2_admission.py, test_item20_round2_generation.py, test_item20_round2_inverse.py, test_item20_round2_memory.py. Receipts stay local under the existing ignore rule; force nothing in. Commit message in the form of 6becc3dc47: subject "Land accepted item 2.0 round 2"; body: land freeze 8cde24380d44 with its certificates and the section 14.5 acceptance, plus the two test-fixture ports named; what it carries (the reverse and runBatch fail loud, the aligned path skipping the mixing-only combine inverse; the ProjectionBasis reverse on the recorded input shape; the forward's action context recorded beside the journal frame and read by the derivation inverse, forward discard and free generation's candidate-owned proposals; one WholeSpace with its checkpoint migration; the supplied-form admission report; the rejecting-bind capture; the large-event item empty); validation: one default make test on the frozen source with its one recorded fixture failure, the two-file rerun on the landed source, the direct before-and-after cases, the four round-2 direct files; generated evidence remains local under the ignore rules. No model or assistant identifier anywhere in the message.

3. Record the landing: append one line to todo.md after its last 2.0 update and one line to the 2.0 spec after §14.6, "Landed as <hash>, 2026-11-09"; commit as "Record item 2.0 round 2 landing"; then push main once (git push -u origin main). Report both hashes to Alec in your return. Then continue with round 3/4 at once, from this landed main, without waiting.

Round 3/4: one gradient and one choice mechanism (todo.md item 2.0 "Rounds", 3 and 4 as §13.2 merges them; spec §§1–5, §7, and §8 "Sequencing"). Every "about" line number is on the landed source.

4. separateChooser false is the default (§1). data/model.xml line 18 and the code defaults (Models.py about 1790, Language.py about 6704) read false; the per-owner path stays for separateChooser true (ObjectiveOwnership.backward_owned; its module docstring "no gradient projection" now names the path it describes). The single-backward branch of Models._backward_training_loss (about 2662 to 2669) becomes the projection: the costs of SentenceTraining.current_costs are summed into two sides, R and A (item 5), reconstruction scaled by the drawn strength as surrogate does and each penalty once; two torch.autograd.grad calls over every optimizer parameter, the R call retaining the graph. For each unit u with <g_A^u, g_R^u> < 0: g_A^u <- g_A^u - (<g_A^u, g_R^u> / ||g_R^u||^2) g_R^u; then p.grad = g_R + g_A and the one optimizer step as now. Units: each parameter tensor; the concept tables (similarity_codebook.W) per concept row; the definition coefficients (_concept_allocator.layer(0).values) per edge; a unit stepped by Adam (the adaptive and attention groups, Models about 2866 to 2890) takes the inner product in Adam's metric, each coordinate weighted by 1/(sqrt(v_hat)+eps) read from that optimizer's state, the identity before its first step. Reconstruction's gradient is never altered; a unit with one writer is unchanged. reconstructionJitter stays, default 0. Receipt, per unit per epoch, in SentenceTraining.gradient_receipt beside the existing dissonance and understanding norms: the cosine of g_R and g_A before projection, the fraction of steps on which output was projected, the fraction of ||g_A|| removed; the read-only measure() path stays as it is. Certificates: a direct case with two hand-built costs over one shared tensor and one single-writer tensor, showing the opposing component removed exactly, the agreeing component kept, the single writer untouched, reconstruction's gradient unchanged, and an Adam-stepped unit projected in its metric; a direct case on data/model.xml running one runBatch under the default, the receipt's three fields populated and the optimizer stepping once; the seven files that assume the per-owner backward (test_generation_selection_backward, test_grounded_xor, test_item25_repair, test_item25_completion, test_one_backward, test_one_backward_standing, test_packed_reconstruction_parity) pass, ported with before/after text only where they asserted the separate backward or its single-writer audit.

5. The sides (§2 as §8 "Sequencing" amends it). R: the reconstruction owner's terms (field, part, iterations, the decoder's walk residual, forward discard, content, reference), compose_lesson, expectation's predictor terms, attention's reading lesson and candidate cross-entropy, the penalties once, and the decoder walk's credit (judged by identity cost; an interim booking until the loop, §8). A: the output owner (answer, prediction, the readers), generate_lesson, the answer walk's residual, the output walk's credit, and every other choice reward whole (reconstruction.compose_score_function, expectation.bind_score_function, attention.candidate_departure, the thought score function): "every choice is in the output chain only" (§8). Code: the answer walk's residual, booked today as reconstruction.walk_residual.* from Models.reverseOutput (about 9731), becomes output.walk_residual.*, while the decoder's (Models._reconstruct_trial, about 20261 and 20288) keeps its name; _sentence_walk_credits (WalkTrials.stage_generate_credit) splits into a decoder list and an output list, registered as reconstruction.walk_credit and output.walk_credit (Models about 21693 to 21695); the four rewards' registrations (Models about 20035, 21715 and 21729; ThoughtCredit.register) carry objective output; registry_costs (ObjectiveOwnership about 15 to 35) routes every term by these objectives into the two sides. No term's value changes, only its side. Certificate: a direct case reading registry_costs on a registry holding one term of each name and asserting the side each lands on.

6. Generate under compose's proposal and credit; walk_policy retired (§3). Both generate walks (Models._reconstruct_trial; WalkTrials.output_pair) draw their alternative action from Expectation.alternative_probabilities with compose's configured fraction and temperature, through Expectation.sample_alternative, in place of Language.sample_eligible_logits at unit temperature; the departure step stays WalkTrials.departure_at. Their credit is SentenceCredit.score_function with the proposal correction (q including the uniform step draw, as Models._compose_score_function_loss builds it, about 19989 to 20035) and delta C from the walk's own comparison (the decoder: identity cost plus residual, the costs _reconstruct_trial already stacks; the output: answer cost plus residual, output_pair's costs), in place of WalkTrials.generate_credit's cross-entropy; the exact-tie rule delta C != 0 is shared. walk_policy goes: Models._decomposition_walk_teacher_loss (about 20319 to 20375), observation['walk_loss'] (about 20492), the reconstruction.walk_policy registration (about 21691) and _last_decomposition_walk_teacher. GeneratePolicy keeps its STOP bias (a risk to measure, §3). Generate lessons stay (GrammarLessons.generate_loss, generate.choice, objective output). Certificates: a direct case in which a winning generate departure is credited by score_function with the recorded probability, correction and delta C, and an exact tie has no graph; a direct case that the alternative draw's probabilities are alternative_probabilities' mixture over the eligible actions with the greedy action excluded; the files §3 names, ported with before/after text: test_operators_round1's walk-teacher case retires with its teacher (recorded); test_grammar_word_learning::test_normal_text_reconstruction_updates_the_grammar_chooser and test_free_reconstruction::test_free_trial_uses_no_reference_or_offsets_and_keeps_identity_audit read the credit instead of the teacher; test_review14_contracts' review17 probe; test_output_walk's three runbatch cases assert a policy change where a departure won, not on every batch; test_mind_generativity::test_current_grammar_recovery_by_depth_training_and_chain_length loses its hand-written forcing.

7. Verbs and modals by participation; teachers retired; the re-read (§5, §8). Retire the loss and its registration only. VerbLearning.observe keeps the pairing of previous and current entries, the residuals, groups.observe and the taught marking, and drops errors.add('verb_teacher', ...) and the deferred records; VerbLearning.register and ModalLearning.register go with _verb_teacher_records and _modal_teacher_records (Models about 21887 to 21895; PresentationMemory's list names); ModalLearning.observe keeps its pairing and marking for its report and drops its loss; the verbTeacherWeight and modalTeacherWeight knobs (Models about 2401 to 2406; data/model.xsd 855 to 856; Params.md about 420 to 430) and ReferenceClose.commit's verb_teacher_weight, teacher_records, modal_teacher_weight and modal_records arguments go; the report lists last_verb_teacher and last_modal_teacher keep their contents under names without "teacher" (recorded). MeaningAction.teacher stays as the assigner's no-grad residual. The assigner (VerbGroups.observe) runs only in training, and its residuals read the verb-phrase meaning the forward lift received: Memory.Event records that meaning beside its verb symbol (a tensor field, the symbol kept for the report), written where the event is committed (ReferenceClose._commit_events), and VerbLearning's residuals use it in place of point(event.verb). The re-read: where PrependedSymbols.symbolize presents a stored value to the next sentence (about line 98, owner.mtm.value_of(key, timeline).detach()), present v + A(b) - sg(A(b)) for a seat whose latest event on that timeline has a recorded input b: A is MeaningAction's forward on the event's before state, the recorded verb-phrase meaning and its basis or group, as MeaningAction.teacher evaluates it, and on the alternative timeline the adverb action over the event's modifiers; sg is the stop-gradient. The presented value equals v exactly; where no event or no recorded input exists, v as today. Nothing else about MTM changes: every write stays detached, each close cuts the graph. Rates: one model-file parameter, training.actionRateRatio (kappa, default 1, finite in (0, 1]; data/model.xsd, Params.md), scaling the lift action's optimizer group to kappa*eta and the adverb action's to kappa^2*eta inside the reconstruction SGD groups (Models about 2880 to 2886); the receipt reports each action's realized relative step ||delta theta||/||theta|| per epoch beside the codes'. Certificates: a direct case that the re-read's presented value equals v to the bit and its gradient reaches the action's parameters through a later sentence's answer and nothing else; a direct case that a seat without a recorded input presents v detached; the assigner's third-failed-fit rule unchanged on the recorded verb-phrase meaning, with the residuals on the verb word's code and on the phrase meaning in the receipt; test_verb_learning's four cases and the two test_modal_timelines cases §5 names ported with before/after text to the kept functions (pairing, residuals, assignment, marking), or retired with a record where they asserted the loss alone; kappa = 1 leaves every step unchanged and kappa = .1 orders the three realized steps (direct cases).

8. The XOR gate, once. After item 6, run the trained xor_grammar phase of test_explicit_dimensions.py::TestXorGrammarReconstruction once, its artifact produced fresh as item 71 did (test/fixtures/phase-artifacts.json names the producer), with the chooser at the residual argmin: four of four multisets expected, transpositions allowed, no seed or budget change, no repeat. A failure stops nothing; record it with the run's evidence for Claude's review.

9. Output typing (todo.md item 2.0, "Output typing", decided). PartsOfSpeech.observe_kept already teaches the argout faces (about 240 to 242); the generate side now reads them. For each generate choice the GeneratePolicy's menu features gain the occupation scores of the same measure the inputs use (PartsOfSpeech.argument_scores, as Models about 18091 and 21102 build the compose side's table): the parent's fit to the rule's argout1 concept and each proposed child's fit to its argin face; STOP contributes zeros; fixed operations without declared arguments contribute zeros. Scores, not masks: the type collapses to parts of speech as Language.md "Distributional parts of speech" describes, and the policy learns their weight. Certificates: a direct case that the features equal argument_scores on the same words; a direct case that, after observe_kept has taught both faces, a candidate occupying its argout face scores above one that does not; the generation and output-walk files pass.

10. Finding 5 of the thinking spec §54: the primitive memberships were re-clamped on 318 of 320 updates and the last updates moved by momentum alone. The clamp (PerceptProperties.project, from Models._project_sentence_parameters, about 19953) discards each outward push, but the SGD velocity keeps it, so an inward gradient must first cancel the stored outward velocity. Repair at the producer: after the step and the clamp, zero the momentum buffer's component at every coordinate the clamp bound, in the outward direction, in whichever reconstruction SGD group owns primitive_properties.members (Optimizer.py about 594 to 672 for RowLocalSGD; torch SGD's momentum_buffer otherwise); the clamp itself is unchanged. Certificate: a direct case in which a coordinate at the bound with outward velocity moves inward on the first inward gradient after the change and did not before, with the numbers. The momentum-carried gradient projection of §1 is not built: its condition, reconstruction rising where output was projected, is read from the end-of-item receipts.

11. Documents, by append or exact replacement: Architecture (the projection and its two sides; generate under one mechanism; the retired teachers and the re-read; output typing; the clamp's velocity), GradientFlow (the sides table as built; the projection receipt), Params.md and data/model.xsd (separateChooser's default, the retired knobs, actionRateRatio), Testing.md one line, Language.md where it names the argout faces, FutureWork (the momentum-carried gradient projection and a learned pair policy stay conditional on the end-of-item receipts; the loop moves the decoder's bookings to output), ObjectiveOwnership's module docstring; your return as §15 of the 2.0 spec, each item with its before-and-after numbers.

Carried, not built here (record them in the return): memory plan §10 and basic percepts (round 5); the loop and the measurements (round 6, §10.2); §54 finding 2 (round 5); the decoder's bookings moving to output (the loop).

Testing: fast checks only until the implementation is finished. Each item's direct before-and-after cases; the targeted files (the seven per-owner files of item 4; test_free_reconstruction, test_output_walk, test_grammar_word_learning, test_review14_contracts and test_mind_generativity for item 6; test_verb_learning, test_modal_timelines, test_reference_close and test_meaning_kernels for item 7; the files you port); the one gate run of item 8; and make test once when you offer the freeze (env -u RUN_TRAINED -u RUN_SLOW make test, nothing deselected; the XOR class, the pending-premise pair and the shared-concept case run inside it). Expect the landed counts (7552 passed, 325 skipped, 1 xpassed) plus the new direct cases, less each case a port retires or re-parameterizes, each named. Receipt under doc/benchmarks/2026-11-09-item2-0-round-3-4/ with commands.json, frozen-source.json, before-source.json (the landed round-2 source of step 3), source.patch, ports.json (every port's before/after text), offered-documents.json sealed before the sweep, placeholders filled only by the one execution; every file your return links to exists before the sweep starts. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud. Return through Alec for Claude's review; no commit or push of round 3/4.
~~~
