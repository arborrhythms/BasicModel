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
