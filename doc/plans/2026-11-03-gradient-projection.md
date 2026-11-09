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

Landed as acbe466e871df474f864bfd403e11c0921441529, 2026-11-09

## 15. Merged round 3/4 return (2026-11-09)

Round 2 landed as `acbe466e871df474f864bfd403e11c0921441529`; its landing record
is `e4b72d69899a98676c6cbe032901a54018365e9e`. Main was pushed once. The two
fixture ports passed (34 cases), and the append-only document check passed
(1000 cases), without another round-2 sweep. The
[landing receipt](../benchmarks/2026-11-08-item2-0-round-2/landing.json) records
both exact ports and the landed source `cff4fb245235c247ce8a14a582c883b5c574766acba221a95fe0cf4f0fd56f4e`.

Round 3/4 is offered, uncommitted and unpushed, as source `40676b500b3a292b51d9edfa6babb783ceec5d525ca41dbdb09362216d92822c`.
This text is sealed before the one default sweep. Its final outcome, counts,
critical-gate decisions and source/document equality checks are filled by that
execution in the [sweep result](../benchmarks/2026-11-09-item2-0-round-3-4/sweep-result.json)
and [final audit](../benchmarks/2026-11-09-item2-0-round-3-4/final-audit.json);
this pre-sweep return does not assert a sweep result before it exists.

### 15.1 Items 4–6: gradients, sides and one choice mechanism

- **4 — Default projection.** `separateChooser` changes from true to false in
  defaults and code. On the shared-tensor direct case, the former combined
  gradient `[-1, 3]` becomes `[1, 3]`; R remains `[1, 0]`, the agreeing A
  `[2, 3]` is unchanged, and the single writer remains `[4]`. In Adam's
  metric `[1, .25]`, the shared sparse row changes from `[-1, 4]` to `[0, 5]`;
  the other rows remain `[1, 1]` and `[2, 3]`. The `data/model.xml` probe uses
  the model's constructed input shape, changes no capacity, takes one step,
  and populates cosine, projection fraction and removed-answer fraction.
  [Projection cases](../benchmarks/2026-11-09-item2-0-round-3-4/projection.json),
  [Adam case](../benchmarks/2026-11-09-item2-0-round-3-4/adam-projection.json),
  [native default receipt](../benchmarks/2026-11-09-item2-0-round-3-4/native-default-projection.json).
  The native numeric probe reports the existing placeholder-answer and absent
  percept-target warnings; it certifies update wiring, not learned reconstruction.
- **5 — R/A routing.** Sixteen representative terms retain their values and
  enter the declared side. Output residual and credit have separate collectors;
  decoder credit stays R. Compose, bind, candidate-departure and thought rewards
  are A whole. Reader exposure masks do not suppress or duplicate these choice
  terms; regularizers enter R once. [Routes](../benchmarks/2026-11-09-item2-0-round-3-4/routes.json).
- **6 — Proposal and credit.** The alternative distribution changes from
  `[0, .731059, .268941]` at unit temperature to the configured mixture
  `[0, .561230, .438770]` in the direct case (fraction .5, temperature 2).
  With selected probability `.244728`, two eligible departure steps and
  delta C `-2`, correction is `3.563603` and credit is `-1.744230`.
  The old winning-departure CE was `1.407606`; the former masked tie retained
  a graph, while the shared score function now returns a graph-free tie.
  `reconstruction.walk_policy` and its forced-tree producer are absent.
  Generate lessons and STOP bias remain.
  [Old-producer comparison](../benchmarks/2026-11-09-item2-0-round-3-4/before-direct.json),
  [proposal-credit case](../benchmarks/2026-11-09-item2-0-round-3-4/generate-credit.json).

### 15.2 Items 7–10: participation, gate, typing and clamp

- **7 — Action participation.** Verb and modal teacher registrations go from
  present to absent; pairing, residuals, assignment and marking remain.
  MTM event version 11 adds the detached forward verb-phrase operand;
  old versions 8–10 load it as absent. Direct ordinary and alternative reads
  preserve v bit for bit, reach lift/modifier parameters through a later answer,
  and leave all stored inputs without gradients. Missing-input reads remain
  detached. The assigner's three failed fits report groups `[-1, -1, 0]`;
  the word-code residual is `1.278706`, versus `9.429863` on the recorded phrase,
  which is the residual actually used. Evaluation does not assign groups.
  κ=1 gives equal relative steps `.1, .1, .1`; κ=.1 gives `.1, .01, .001`
  for codes, lift and adverb. [Re-read](../benchmarks/2026-11-09-item2-0-round-3-4/reread.json),
  [absent input and legacy event](../benchmarks/2026-11-09-item2-0-round-3-4/reread-no-input.json),
  [phrase assignment](../benchmarks/2026-11-09-item2-0-round-3-4/phrase-assignment.json),
  [rates](../benchmarks/2026-11-09-item2-0-round-3-4/action-rates.json).
- **8 — One fresh XOR gate.** The unchanged producer ran once, 400 epochs,
  with residual-argmin chooser, no pinned seed and no budget change. The
  maintained reconstruction class passed 4/4 word multisets; two surfaces
  are transpositions. The fresh checkpoint and its SHA-256 are recorded in
  the [gate result](../benchmarks/2026-11-09-item2-0-round-3-4/xor-gate/result.json).
  This is the gate after item 6, not an item-71 panel or an end-of-item measurement.
- **9 — Output typing.** Three occupation features replace the former absence
  of output type features. They equal `argument_scores`'s candidate coverage
  on the same words. After kept participation and the ordinary hierarchy
  refresh, the occupied parent scores `.5`, the unoccupied parent `0`, both
  child input faces `1`, and STOP `0`. Undeclared fixed faces remain zero.
  Existing quality weights survive checkpoint loading; the new occupation
  weight starts at zero. [Feature certificate](../benchmarks/2026-11-09-item2-0-round-3-4/output-typing.json).
- **10 — Clamp velocity.** At upper/lower bounds, velocity `[-2, 2]` previously
  kept coordinates at `[1, 0]` after the first inward gradient. Clearing only
  outward velocity gives `[.9, .1]`. An interior velocity `-.5` remains;
  its coordinate moves from `.5` to `.545`. Both ordinary and compact SGD
  produce these results. [Velocity certificate](../benchmarks/2026-11-09-item2-0-round-3-4/clamp-velocity.json).

### 15.3 Item 11: documents, ports and the offered freeze

Architecture, GradientFlow, Params/XSD, Language, FutureWork and Testing are
updated. The unchanged read-only measurement path remains beside the new
per-unit projection and realized-step receipts. Twelve direct cases are added.
Two cases retire with their sole loss producer: the round-1 forced walk teacher
and the deferred verb-teacher registration case. Other affected tests retain
pairing, marking, residual, checkpoint, update and decoding assertions, with
exact [before/after ports](../benchmarks/2026-11-09-item2-0-round-3-4/ports.json).
The seven gradient files and specified generation/action files were run;
producer/fixture failures and their targeted repairs remain visible in the
[command log](../benchmarks/2026-11-09-item2-0-round-3-4/commands.json).
The compile-cache permission failure was rerun with its cache in writable
`/private/tmp`; no test was deselected to bypass it.

The expected default count is the landed 7552 passed, 325 skipped, 1 xpassed,
plus 12 direct cases minus 2 named retirements: 7562 passed, 325 skipped,
1 xpassed (7888 selected). Raw unittest subtest reports are counted separately
from unique cases. The single command is `env -u RUN_TRAINED -u RUN_SLOW make test`.
A failed convergence or shared-concept gate stops the round without a retry.
[Before source](../benchmarks/2026-11-09-item2-0-round-3-4/before-source.json),
[frozen source](../benchmarks/2026-11-09-item2-0-round-3-4/frozen-source.json),
[patch](../benchmarks/2026-11-09-item2-0-round-3-4/source.patch),
[sealed documents](../benchmarks/2026-11-09-item2-0-round-3-4/offered-documents.json)
and [result index](../benchmarks/2026-11-09-item2-0-round-3-4/result.json)
exist before that execution.

**Carried:** memory plan §10, basic percepts and thinking-spec §54 finding 2
remain round 5; the loop and §10.2 measurements remain round 6. The loop moves
the decoder's bookings to output. Momentum-carried projection and a learned
pair policy remain conditional on end-of-item receipts. Nothing of those rounds,
no lesson arms, no item-60 metrics and no repeated gate fits is built here.

### 15.4 Review of the §15 return (Claude, 2026-11-09): round 2 landed; round 3/4 is NOT landable as frozen: 49 red cases (48 test ports, one defect), gates green, one repair pass then the landing

**Round 2 landing.** `acbe466e871d` ("Land accepted item 2.0 round 2") carries exactly the two one-line ports of §14.5 and nothing else beyond the freeze; `e4b72d6989` records it; main equals origin/main. The landed source `cff4fb245235c247ce8a14a582c883b5c574766acba221a95fe0cf4f0fd56f4e` (1008 files) equals before-source.json of the round 3/4 receipt.

**Source and documents.** frozen-source.json `40676b500b3a292b51d9edfa6babb783ceec5d525ca41dbdb09362216d92822c` (1014 files) equals the live tree; 42 files differ from the landed source (18 under bin, 2 under data, 7 under doc, 9 ported test files, 3 new modules ActionLearning/GenerateOccupation/GradientProjection, 3 new direct test files). source.patch (2524 lines) and ports.json (9 files, 2 retirements, 12 added cases) read in full. offered-documents.json seals the 7 changed .md files (`aff327ca…`), recomputed equal; no other .md differs from HEAD.

**Gates, sweep, certificates.** Codex's one sweep (7888 cases, 15 min): test_convergence, the shared-concept case and the pending-premise pair passed; 7513 passed, 325 skipped, 1 xpassed, 49 failed against the expected 7562/325/1. §15 was sealed before the sweep and does not describe the failures; result.json records "completed with recorded failures". A first `make test` died in a sandbox on `sysctl hw.memsize` before collecting anything (sandbox-launch.json); the counted sweep is the one real run. I reran the twelve direct cases from the live tree into the scratchpad (5 s): ten JSONs equal Codex's to $10^{-6}$; native-default-projection.json differs only in one unit's cosine and the realized step (unseeded initialization), and on that one-batch probe no unit was projected in either run (fraction_projected 0 for all 4112 units), so it certifies wiring, not an opposition. The XOR gate ran once: 4/4 multisets at 400 epochs; the checkpoint's sha256 equals result.json.

**The 49 red cases** (each reproduced alone; each port verified green on a scratch copy; none is in ports.json). 48 are test ports under the new design, one is a producer defect; none is a retirement.

- Cause A, five cases: a SimpleNamespace language without `_tree_layer`, which the walk body (Models ~13419) and `WalkTrials.generate_credit` now read for the exploration fraction and temperature: test_decoder_exploration (two cases), test_inverse_work (two, one helper), test_review10_contracts::test_decoder_departure_samples_lower_rank_and_keeps_greedy_suffix (shares the first fixture). Port: `_tree_layer=lambda arity: SimpleNamespace(exploration_fraction=.5, exploration_temperature=1.)`; every assertion holds.
- Cause B, six cases: menu features 5 → 8 and the walk arrays 9 (the step's legal mask rides as a ninth lane, unstated in §15): test_generative_residual_finite (shape 8), test_generation_budget (two, shape 9), test_generation_catalog (the new `occupation_score.weight` key), test_final_generation and test_generate_repair (both bound to the retired cross-entropy credit `generate_policy_credit`; they move to the score-function credit with 9-wide features, `greedy`/`eligible`/`costs` in the audit, the sign of $\Delta C$ in the assertions, `_sentence_decoder_walk_credits` as the collector, and the exact-tie rule in place of `wins.zero_()`).
- Cause C, five cases: retired or renamed terms and MTM version 11: test_decomposition_chooser (`walk_policy` now asserted absent), test_item20_round2_memory (version 11), test_case_views (the event tuple's last slot is now `verb_meaning`; the bad `case_value` goes at slot −2), test_bind_thought and test_reference_review10 (the thought score function is objective output).
- Cause D, four cases: `ownership_gradient_diagnostics` reports sides `R`/`A` in projection mode: test_reading_curriculum (`['R']`), test_native_attention_field::…credits_only_scorer_parameters (`['A']`), test_objective_ownership (zero gradient instead of None; writers $\subseteq$ {R, A}), test_joint_objectives::…ownership_and_backward_dispatch (a `torch.nn.Module` owner, expected $g_R + \Pi(g_A) = [2, 4]$).
- Cause E, twelve cases: single mode is the default (one backward per sentence after the commit, no per-trial step, no batch-end step, no perception fork, no pullback): test_compose_pair_driver, test_joint_objectives::…real_runbatch (`['single']`), test_native_attention_field::…charges_only_kept_trial (capture the trial registry as the author's test_free_reconstruction port does), test_sentence_comparison (two: the pullback case and the replay case, whose fixture labels both trials exploit so the batch-end answer stays trained and hits the fail-loud raise at Models ~15207; it supplies no targets), test_sentence_compose (`['commit', 'step']`), test_supplied_answer_training (`autograd.grad` in place of the pullback), test_output_path_supervised (`_single_sentence_objectives`), test_operators_round2c (four: the closing thought's exposure trains in the same sentence step, so one output step, not two), test_operators_round2e (same).
- Cause F, five cases: direct chooser-layer tests read the config default, so the live compose features share the graph and the test's own second backward collides: test_compose_operations, test_compose_review, test_signal_router_layer, test_signal_router_unary (each a one-line `separate_chooser = True` on the layer if that path stays), test_live_references (`retain_graph=True` on the test's first backward).
- Cause G, eleven cases: the answer-side stop-gradients at Models ~12763 (reverseOutput's idea), ~17562 and ~17581 (`_forward_head`), ~20674 (`_sentence_answer_error`) have been conditional on `separate_chooser` since item 2.5's landing; under the default the trunk is shared, as §2 states, and the projection is what protects reconstruction. test_trial_policy_ownership (two), test_understanding_readout (four), test_prepared_answer_boundary (three; the percept case's gradient exists and is identically zero), test_operators_round2 (one) assert the gradient present instead of None. This is the design, not a removed cut; recorded here so Alec can object.
- DEFECT, one case, nondeterministic (unseeded; failed 5 of 8 solo runs): test_learning_readiness::test_native_progress_and_trained_artifact_evaluation_path. Cross-sentence graph carry on the default training path. At the single-path sentence boundary only `stm`, `lang` and `feedback` are detached (Models ~21888–21893); `sub`, `sym` and the word-aligned slabs the next sentence reads (inputSpace `_ar_part_reference_vectors`, `_ar_whole_reference_vectors`, `_ar_readout_coefficients`; symbolSpace `_word_reference_presences`; the reconstruction stack's `_choice_values`, `_choice_action_records`, `_choice_log_probabilities`; `_tensor_pushed_ideas`, `_tensor_sentence_roots_live`, `_packed_sentence_roots`, `_attention_words`, `_last_decoder_trace`) keep the previous sentence's graph. The second sentence's R gradient then traverses the first sentence's nodes: when the first sentence had an A cost, its A traversal ran with `retain_graph=False` (GradientProjection ~94) and freed them, raising "backward through the graph a second time" at GradientProjection ~93; when it had none, the traversal succeeds silently and the second sentence's gradient crosses the optimizer step. The per-owner path masked both (`fork_perception` on detached leaves per trial; `backward_owned` always retains). Packed streams of two or more sentences are the ordinary path, so this must be repaired before landing: detach every carried word-aligned tensor at the boundary exactly as `stm`/`lang`/`feedback` are, with a certificate (a seeded two-sentence packed batch whose first sentence has an output-side term steps twice; a per-tensor `autograd.grad(..., retain_graph=True)` scan finds no second-sentence cost reaching a first-sentence node).

**Two further findings from the patch.** (1) Models ~9895's dense-synthesis placeholder `_last_output_walk_trace` still allocates 5-wide features while the walk arrays are 9-wide; a dense-synthesis model trained with a score would reach `generate_credit` with four columns and fail as test_final_generation did. The width belongs in one named constant (Generative) read at ~9895, ~13458, ~13494 and by `generate_credit`. (2) The `separateChooser` true path is kept but degraded: the compose, bind, candidate-departure and thought rewards are now objective output, so `backward_owned` hands them to the readers' parameters and the compose chooser, the resolver and the attention scorer get no credit in that mode; the `chooser` owner is dead (ObjectiveOwnership's branch, Models ~2600, SentenceTraining ~175, GradientProjection.sides); no test in the default sweep runs that mode. Under "no legacy code paths" my recommendation is to delete it (the knob in model.xml/xsd/Params, `backward_owned`, the four conditional detaches made unconditional in the live direction, the `chooser` owner), porting the five Cause-F and two Cause-E tests that would otherwise pin `separate_chooser = True` to single mode or retiring them with a record; the per-owner diagnostics stay as the projection receipt. Alec decides.

**Warts, recorded; the one-line ones go into the repair pass.** `amp_scaler.scale(total_loss)` at Models ~2666 computes a value and discards it. `_projection_writers` is never reset (its per-owner predecessor is reset at Models ~14199). `GeneratePolicy.forward` tolerates 5-wide features (`features.shape[-1] == 5`), a producer tolerance for old fixtures; remove it and port test/generation_fixtures.suffix (5-wide at line 23). `LanguageSpace.generate_policy_credit` (~12674) has no caller left. `VerbLearning.observe`'s `point` parameter is unused. `VerbLearning.reread` runs its action forward on every read including evaluation (the correction is exactly zero there). model.xml's "Adam learning rate" comment now sits above `actionRateRatio`. `getattr` defaults that only serve fixtures: GenerateOccupation ~8–12 (`parts_of_speech`, `_generate_*_rules`), ActionLearning's callers at Models ~2886 (`languageSpace`, `action_rate_ratio`), GradientProjection.units ~29 (`conceptualSpaces`). The occupation weight is migrated twice (GeneratePolicy._load_from_state_dict and LanguageSpace ~12644). Recorded without repair: GenerateOccupation.features is a host loop over rules per menu; PartsOfSpeech `_value_arguments` is a staging buffer reassigned at `prepare_values`; `registry_costs` recognizes choice rewards by name suffix.

**Verdict.** No gate failed, so the round continues (§13.2). The frozen tree is not accepted for landing: 49 red cases and one producer defect on the default path. One repair pass follows: the defect, the placeholder width, the 48 ports with before/after text, the one-line warts, the §15 record of the ninth lane; then one sweep expecting 7562 passed, 325 skipped, 1 xpassed (less any case the separate-path decision retires, each named).

**To confirm with Alec before the Codex text.** (a) Delete the `separateChooser` true path (recommended) or keep it and repair its routing. (b) The pass: one Codex text that repairs, sweeps once and lands on a green sweep in the §11.7 form, returning the landing record as §16 (recommended; every repair is specified here and the sweep is the certificate), or a repair text returned for review before the landing.

**Decided (Alec, 2026-11-09).** (a) "keep it, it's an option, not a legacy path": the `separateChooser` true path stays and its routing is repaired, so that every choice reward reaches the parameters it reached before the round in that mode, while the projection keeps the sides of §8. (b) "your recommendation": one Codex text that repairs, sweeps once and lands on a green sweep; the landing record returns as §16. The build-5 text follows §16.

### 15.5 One Codex text: repair round 3/4, one sweep, landing on green (Claude, 2026-11-09; for Alec to paste; final)

Preface for Alec. Owner and side become two attributes of a cost term: the owner says which parameters receive it under `separateChooser` true, the side says whether it is R or A under the default. That is the smallest change that keeps both paths exact. The carry repair retains the batch's perception graph across sentences, as the per-owner path always did, and cuts only what a finished sentence produced. The Cause-G ports record that the answer reads the concluded idea live under the default; if you want that cut unconditional, say so before pasting.

~~~
Item 2.0, merged round 3/4: repair pass, one sweep, landing on green. The reviewer's findings are 2.0 spec §15.4 (doc/plans/2026-11-03-gradient-projection.md); the landed round-2 source is cff4fb245235c247ce8a14a582c883b5c574766acba221a95fe0cf4f0fd56f4e; your §15 freeze is 40676b500b3a292b51d9edfa6babb783ceec5d525ca41dbdb09362216d92822c. Alec decided (2026-11-09): the separateChooser true path stays as an option and its routing is repaired; this one text repairs, sweeps once and lands on a green sweep. Fast checks only until the implementation is finished. Receipt under doc/benchmarks/2026-11-09-item2-0-round-3-4-repair/: commands.json; before-source.json (the §15 freeze); frozen-source.json; source.patch against the landed round-2 source (the whole round); ports.json with every port of the round, the 48 below included, before/after text; offered-documents.json sealed before the sweep; sweep-result.json; final-audit.json; landing.json. Return as §16 of the 2.0 spec, each item with its before-and-after numbers. No commit before step 7.

1. Cross-sentence graph carry (§15.4, the defect). Invariant: at the entry of every sentence after the first in a packed stream, a cost of that sentence reaches its own graph and the batch forward's perception graph (the parameter-derived word slabs computed before the sentence loop), and no node created between an earlier sentence's entry and its optimizer step. Two mechanisms. (a) GradientProjection.backward (about 93 to 94): both traversals retain the graph (the A call retain_graph=True like the R call), so the batch's perception graph survives the first sentence's step for the later sentences, as backward_owned always did; the graph is released when the batch's references drop. (b) At the single-path sentence boundary (Models about 21888 to 21893, where stm, lang and feedback are detached after the step), one named helper detaches every tensor the finished sentence produced and the next sentence reads: the reconstruction stack's _choice_values, _choice_action_records and _choice_log_probabilities; _tensor_pushed_ideas, _tensor_sentence_roots_live, _packed_sentence_roots, _attention_words, _last_decoder_trace, and whatever else the certificate scan finds. For the threaded word states sub and sym, cut only the finished sentence's writes: rebuild each as torch.where(written, state.detach(), perception_state), the batch forward's pre-loop values standing at the unwritten positions, so the perception graph stays reachable for the next sentence's words. inputSpace._ar_part_reference_vectors, _ar_whole_reference_vectors, _ar_readout_coefficients and symbolSpace._word_reference_presences are batch-level perception products and stay attached under (a). Certificate, a direct case, unseeded (the assertion is structural and holds for every draw): a two-sentence packed stream under the default whose first sentence has an output-side term takes two optimizer steps without a raise, and a scan at the second sentence's entry over the model's and the spaces' carried tensors and the threaded sub/sym finds no tensor with a grad_fn created between the first sentence's entry and its step (autograd.grad(cost2, tensor, allow_unused=True, retain_graph=True) over the first sentence's choice records is None). test_learning_readiness::test_native_progress_and_trained_artifact_evaluation_path then passes eight consecutive solo runs; record the eight.

2. One walk-feature width. One named constant in Generative (the menu's eight quality-and-occupation lanes plus the legal lane: 9), read at Models about 9895 (the dense-synthesis placeholder, now 5-wide), about 13458 and 13494, and by WalkTrials.generate_credit's split of the legal lane. GeneratePolicy.forward (Language about 12358 to 12364) loses the features.shape[-1] == 5 branch and always adds occupation_score(features[..., 5:]); test/generation_fixtures.suffix (line 23) builds 9-wide features. Record the ninth lane (the step's legal mask, kept so the credit can rebuild the departure-step logit mask) in Architecture's round 3/4 section and in §16.

3. Owners for separateChooser true, sides for the default. Error.add and Error.error (Layers about 14305 and 14346) take side='R' or 'A', defaulting by owner (output, generate_lesson and chooser to A, every other owner to R). Registrations: reconstruction.compose_score_function objective 'chooser', side A (Models about 20052, back from 'output'); thought_score_function objective 'chooser' when its category is reconstruction, else its category (ThoughtCredit about 23 to 25, as before the round), side A; expectation.bind_score_function objective 'expectation', side A (about 21679); attention.candidate_departure objective 'attention', side A (about 21691); output.walk_credit objective 'chooser', side A (about 21658; the decoder's reconstruction.walk_credit stays objective reconstruction, side R); output.walk_residual.* objective 'reconstruction', side A (about 20548), rows None, whole, like the decoder residual. ObjectiveOwnership.registry_costs keys each sum by the owner when the side is the owner's default and by owner + '.A' (or '.R') otherwise; choice terms keep rows None; the choice_rewards flag and the name-suffix test go, and SentenceCredit.reader_costs filters on the keys output and penalty.output as before. GradientProjection.sides: A is the keys output, generate_lesson, chooser and every key ending in '.A'; R is the rest; reconstruction scaled by the drawn strength; each penalty once. ObjectiveOwnership.backward_owned: the owner is the key before its first '.', the parameters owners[owner], the pullback traversed for reconstruction, chooser and attention as before the round. Effect: under separateChooser true every choice reward and the output walk's residual and credit reach the parameters they reached before the round (the chooser set is the reconstruction set, Models about 2600); under the default only the keys change. Certificates: the routes direct case in test_item20_round34_projection asserts owner and side per term (routes.json gains the owner); a direct case under separateChooser true in which a compose score function with an advantage trains the chooser's logits and nothing in the readers (test_review14_contracts' probe returns to backward_owned(costs, {'chooser': (logits, unrelated)}) and set(costs) == {'chooser'}); the option's ported tests of item 4 run that mode.

4. The 48 test ports (§15.4 causes A to G; before/after text in ports.json). Cause A, the fixture language gains _tree_layer=lambda arity: SimpleNamespace(exploration_fraction=.5, exploration_temperature=1.): test_decoder_exploration's decoder() about line 22 and its inline language about line 184; test_inverse_work's language(ops) helper about line 26 (both cases); test_review10_contracts::test_decoder_departure_samples_lower_rank_and_keeps_greedy_suffix shares decoder(). Cause B: test_generative_residual_finite line 70 shape (2, 1, 8); test_generation_budget line 95 shape (1, 2, 4, 9), both parameters; test_generation_catalog line 116 adds 'generate_policy.occupation_score.weight'; test_final_generation lines 72 to 80: the language gains _tree_layer and binds only generate_policy_logits, features ones(3, 1, 2, 9), the audit adds greedy ones(3, 1) long, eligible ones(3, 1) bool and costs, the assertion becomes (loss[:2] > 0).all() and loss[2] < 0 (the sign of delta C: +.1, +9.9, -.5); test_generate_repair lines 187 to 207: GeneratePolicy(2, 3), the language gains _tree_layer and binds only generate_policy_logits, features ones(3, 4, 3, 9), the audit adds greedy [[2, 0, 1, 0]] three times, eligible ones(3, 4) bool and costs [[1, 0], [0, 1], [0, 0]], the collector is _sentence_decoder_walk_credits, the assertion credit[0] < 0 and credit[1] > 0 and credit[2] == 0, and the wins.zero_() tail becomes the exact-tie rule (costs zeros(3, 2): zero credit without a graph). Cause C: test_decomposition_chooser line 102 asserts 'reconstruction.walk_policy' not in the registry; test_item20_round2_memory line 198 version 11; test_case_views line 257 writes the bad case_value at slot -2, (*event[:-2], bad_value, event[-1]); test_bind_thought and test_reference_review10 line 235 need no port once item 3 restores the bind owner (verify that errors.total(objective='expectation') returns the term). Cause D: test_reading_curriculum line 96 writers ['R']; test_native_attention_field::test_native_candidate_departure_credits_only_scorer_parameters lines 133 to 134 ['A']; test_objective_ownership line 95 the gradient present (zeros) and line 116 set(row['writers']) <= {'R', 'A'}; test_joint_objectives::test_model_parameter_ownership_and_backward_dispatch line 194 an owner that is a torch.nn.Module carrying outputSpace, synthesis_parameters and errors, line 205 expected [2., 4.]. Cause E: test_compose_pair_driver line 64 _build('<separateChooser>true</separateChooser>'), the option's per-trial certificate; test_joint_objectives::test_real_runbatch_steps_sentence_trials_then_batch_only_objectives line 236 ['single'] for MM_xor_loopback, lines 259 and 263 recording 'single' when _sentence_backward holds with _sentence_trial None and when _single_sentence_objectives is set; test_native_attention_field::test_native_reads_one_supported_word_and_charges_only_kept_trial captures the trial registry through a _sentence_path_cost wrapper as your test_free_reconstruction port does and reads registries[-1]._terms at line 40; test_sentence_comparison::test_sentence_trials_keep_their_own_perception_pullbacks line 37 adds architecture_overrides={'separateChooser': True} and line 63's read counts are re-derived with the chooser owner restored by item 3 (before/after recorded); test_sentence_comparison::test_replaying_real_greedy_derivation_ties_in_every_row lines 93 to 94 set inputSpace.data.has_supervised_outputs False and target torch.empty(4, 0) (its replay labels both trials exploit, so no reader update and the fail-loud raise at Models about 15207); test_sentence_compose line 186 ['commit', 'step'] twice; test_supplied_answer_training line 73 torch.autograd.grad(error.mean(), params, retain_graph=True, allow_unused=True); test_output_path_supervised line 272 m._single_sentence_objectives; test_operators_round2c line 145 one output step (the closing thought's exposure trains in the same sentence step; comment lines 143 to 144 say so); test_operators_round2e line 120 {1.}. Cause F, the option's direct certificates: test_compose_operations after line 19 result.separate_chooser = True; test_compose_review after line 11, test_signal_router_layer after line 25 and test_signal_router_unary after line 33 step.separate_chooser = True; test_live_references line 395 backward(retain_graph=True). Cause G: test_trial_policy_ownership line 43, test_understanding_readout line 45, test_prepared_answer_boundary lines 161 to 162 and test_operators_round2 line 210 assert the gradient present (the trunk is shared under the default, §2; the message says so); test_prepared_answer_boundary line 226 gradients[0] is None or not bool(gradients[0].abs().any()). A port that does not hold as written is repaired on its own fixture with the assertion's claim unchanged and recorded; no producer gains a default for a fixture.

5. The one-line warts of §15.4: delete amp_scaler.scale(total_loss) at Models about 2666; reset _projection_writers beside _objective_gradient_writers at about 14199; delete LanguageSpace.generate_policy_credit (about 12674); drop VerbLearning.observe's unused point parameter and its callers' lambdas (ReferenceClose about 1424, the tests); VerbLearning.reread returns value unchanged when torch.is_grad_enabled() is False; move model.xml's "Adam learning rate" comment back above learningRate; remove the getattr defaults in GenerateOccupation.features (parts_of_speech, _generate_binary_rules, _generate_unary_rules; every LanguageSpace has them) and in GradientProjection.units (conceptualSpaces); keep the one occupation migration at LanguageSpace about 12644 and drop GeneratePolicy._load_from_state_dict. Recorded without repair: the host loop in GenerateOccupation.features; PartsOfSpeech._value_arguments reassigned at prepare_values; the two getattr at Models about 2886 stay if any constructed BaseModel lacks languageSpace (say which in §16).

6. Documents, by append or exact replacement: Architecture (the ninth lane; owner and side as two attributes; the option's restored routing; the sentence-boundary detach and the retained batch graph), GradientFlow (the sides table gains the owner column), Testing one line, Params.md's separateChooser row (an option: both paths named). §16 lists every item with its numbers.

7. Sweep and landing. Seal frozen-source.json and offered-documents.json, then run env -u RUN_TRAINED -u RUN_SLOW make test once, outside the sandbox that denied sysctl, nothing deselected. Expected: 7562 passed, 325 skipped, 1 xpassed plus the new direct cases of items 1 and 3, each named; the XOR class, the pending-premise pair and the shared-concept case run inside it. Green means every selected case passed or skipped as expected. On green, land in the §11.7 form: landing.json and landed-source.json (the frozen hash is the landed hash; nothing changes after the sweep); commit on main every file the round changed, added or deleted since cff4fb24…, the message in the 6becc3dc47 form: subject "Land accepted item 2.0 round 3/4"; body: the §15.4 review and this repair; what the round carries (the default projection with per-unit sides; owners kept for separateChooser true; generate under the proposal and the score-function credit, walk_policy retired; the teachers retired, the assigner on the recorded verb-phrase meaning, the re-read, actionRateRatio; output typing; the clamp velocity; the sentence-boundary detach); validation (the one sweep's counts, the direct certificates, §15's one gate run); generated evidence local under the ignore rules; no model or assistant identifier. Append "Landed as <commit hash>, 2026-11-09" after §15.5 in the 2.0 spec and a line in todo.md; commit "Record item 2.0 round 3/4 landing"; push once. On any red case: no commit, no push; §16 lists the failures with their tracebacks and your diagnosis, nothing retried.

Carried: memory plan §10, basic percepts and thinking-spec §54 finding 2 (round 5); the loop and §10.2's measurements (round 6); the decoder's bookings move to output with the loop; momentum-carried projection and a learned pair policy stay conditional on the end-of-item receipts. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud.
~~~

## 16. Merged round 3/4 repair and landing return (2026-11-09)

This is the repair authorized in §15.5, preserving `separateChooser=true` as an
option. It starts at the §15 source `40676b500b3a292b51d9edfa6babb783ceec5d525ca41dbdb09362216d92822c`
and offers `420c0c0b388ba24da5b67248d1bffb37dd98570a2bb3fa4febc72b9a395981f0`. The whole-round patch is against landed round 2,
`cff4fb245235c247ce8a14a582c883b5c574766acba221a95fe0cf4f0fd56f4e` (main
`e4b72d69899a98676c6cbe032901a54018365e9e`). This prose is sealed before the
single sweep; its outcome and any landing are filled by that execution in the
[sweep result](../benchmarks/2026-11-09-item2-0-round-3-4-repair/sweep-result.json),
[final audit](../benchmarks/2026-11-09-item2-0-round-3-4-repair/final-audit.json)
and [landing record](../benchmarks/2026-11-09-item2-0-round-3-4-repair/landing.json).

1. **Sentence graph lifetime.** Both traversals now retain the batch graph
   (R/A retention `true/false` → `true/true`). The unseeded two-sentence
   certificate takes **2** steps with an explicit first-sentence A term.
   At the second entry its scan finds **0** earlier-sentence graph nodes in
   the carried tensors; all three eligible old choice-record pullbacks are
   `None`, while batch perception remains reachable. The scan found additional
   carries: sampling scale, expectation comparison, gradient-objective reports,
   percept-field values/events, arrival loss and full STM buffers. Detaching only
   the active STM prefix retained the inactive rows' old graph through `cat`;
   the helper now cuts the whole buffer. Threaded sub/sym preserves original
   perception values at unwritten positions via `where`. The reference vectors
   and readout coefficients remain live. In this native fixture the initial
   symbol-presence slab is absent; sentence publication creates it, so the helper
   cuts those sentence writes and preserves original unwritten positions when a
   pre-loop slab exists. This distinguishes that replacement from a batch
   perception product. The review reported **5 failures in 8** native solo runs;
   the repair passes **8 consecutive runs out of 8**, unseeded, each in its own
   single-test process. [Boundary certificate](../benchmarks/2026-11-09-item2-0-round-3-4-repair/sentence-boundary.json),
   [eight solo runs](../benchmarks/2026-11-09-item2-0-round-3-4-repair/eight-solo.json).

2. **One feature width.** The dense placeholder was **5** wide while real traces
   were **9** wide; all now read `Generative.WALK_FEATURE_WIDTH = 9`. Policy
   inputs are always **8** lanes (five quality, three occupation); lane **9** is
   the legal-action mask used to rebuild departure logits. The 5-wide policy
   fallback is removed. The suffix fixture supplies 9 lanes. Strict native-menu
   and toy-fixture checks passed **77**, skipped **3**; fixtures with no declared
   arguments explicitly supply an empty inventory.

3. **Owner and side.** Each error term now records **2** independent attributes.
   Six formerly output-owned reward/residual families regain their declared
   owners: compose and reconstruction thought → chooser; bind → expectation;
   candidate departure → attention; output credit → chooser; output residual →
   reconstruction. Their side remains A; decoder credit/residual remain R.
   Sixteen direct routes preserve each weighted value at **6** and assert its
   side and owner. Reader exposure uses output keys, without suffix-based choice
   detection. The option's direct compose case moves logits
   `[.2, -.1, .4]` → `[.174672, -.043737, .369065]` toward the cheaper departure;
   reader and unrelated gradients remain `None`. The default projection's
   original hand-built gradient remains `[1, 3]`, with R unchanged at `[1, 0]`.
   [Routes with owners](../benchmarks/2026-11-09-item2-0-round-3-4-repair/routes.json),
   [optional-owner certificate](../benchmarks/2026-11-09-item2-0-round-3-4-repair/separate-owner.json).

4. **Ports.** The [complete port manifest](../benchmarks/2026-11-09-item2-0-round-3-4-repair/ports.json)
   retains landed, reviewed and repaired text, including all **48** reviewed
   non-defect cases. Bind/thought expectation assertions remain unchanged and
   pass with restored owners. The pullback fixture's old fixed **6** calls becomes
   **4**, plus **1** for a non-tied explored chooser reward; the final targeted
   draw records **5**. It retains the claim that each trial uses its own
   perception at the same pre-update parameter version.
   [Pullback counts](../benchmarks/2026-11-09-item2-0-round-3-4-repair/perception-pullbacks.json).
   One combined targeted command was interrupted after **267 passed, 23 skipped,
   9 failed** while constructing a large codebook; its diagnostics are retained.
   Five failures were missing toy inventories, already repaired in the strict
   menu checks. The other four were port mistakes (a misplaced return, an unused
   lambda and two unrelated AMP expectations), repaired in their fixtures.
   Remaining files ran separately with one CPU thread; their sole initial
   mismatch was the pullback count above, which then passed. The error/reader
   routing checks passed **16**. No producer default was added for a fixture.
   [All commands and outcomes](../benchmarks/2026-11-09-item2-0-round-3-4-repair/commands.json).

5. **Small repairs.** The discarded projection AMP scale call, dead
   `generate_policy_credit`, unused `observe` operand and four fixture-only
   attribute defaults are removed. Projection writers reset once per batch.
   Evaluation re-read action calls go from an unnecessary forward to **0**
   under `no_grad`; training keeps the bit-exact local pullback. Occupation
   checkpoint migration goes from **2** sites to **1**, in LanguageSpace. The
   Adam-rate comment again labels `learningRate`. The two ActionLearning call-site
   defaults remain: bare `BaseModel()` checkpoint hosts are constructed without
   `languageSpace` or configured `action_rate_ratio`, including
   `test_structural_checkpoint` and `test_learning_readiness`'s progress roundtrip.
   The GenerateOccupation host loop and reassigned PartsOfSpeech staging buffer
   remain recorded without repair.

6. **Documents and case accounting.** Architecture names the ninth lane, both
   cost attributes and both graph lifetimes; GradientFlow has the owner column;
   Params names both supported paths; Testing records the new certificates.
   The round adds **14** direct cases (the prior **12**, plus
   `test_packed_sentence_boundary_cuts_prior_writes_and_retains_perception` and
   `test_separate_chooser_option_routes_compose_credit_without_reader_updates`).
   The same **2** teacher-only cases recorded in §15 remain retired; no further
   case retires or changes parametrization. [Before source](../benchmarks/2026-11-09-item2-0-round-3-4-repair/before-source.json),
   [frozen source](../benchmarks/2026-11-09-item2-0-round-3-4-repair/frozen-source.json),
   [whole-round patch](../benchmarks/2026-11-09-item2-0-round-3-4-repair/source.patch),
   [sealed documents](../benchmarks/2026-11-09-item2-0-round-3-4-repair/offered-documents.json).

7. **One sweep; landing only on green.** The expected unique count is
   **7,564 passed, 325 skipped, 1 xpassed** (**7,890 selected**): landed 7,552
   passes plus 14 additions minus 2 retirements. The command is
   `env -u RUN_TRAINED -u RUN_SLOW make test`, outside the sandbox that denied
   `sysctl`, with nothing deselected. Native subtest reports are also preserved
   separately. The XOR class, pending-premise pair and shared-concept case run
   inside it. §15's single fresh XOR gate remains **4/4**; this repair adds
   **0** gate fits. On green, [landed source](../benchmarks/2026-11-09-item2-0-round-3-4-repair/landed-source.json)
   has the frozen source hash; the landing record contains both commits and the
   one push. On red, no commit/push occurs and the sweep receipt retains every
   traceback, with the diagnosis recorded in this return.

**Carried:** memory plan §10, basic percepts and thinking-spec §54 finding 2
remain round 5; the loop and §10.2 measurements remain round 6. Decoder bookings
move to output with the loop. Momentum-carried projection and a learned pair
policy remain conditional on end-of-item receipts. No seed pinning, model resize,
new language-specific code, lesson arms or end-of-item measurements are introduced.


### 16.1 Single-sweep result: red; not landed

The one sweep completed all **7,890** selected cases: **7,553 passed, 324 skipped, 1 xpassed, 12 failed**.
Source and offered documents stayed unchanged throughout the execution.
Of the earlier 49 failing cases, **48 pass** and **1 fails**.
The structural graph and optional-owner certificates, convergence, shared-concept,
pending-premise pair and ordinary XOR class all passed, as recorded in the sweep receipt.
The one-case skip difference is `test_data_loader_list_outputs_survive_worker_pickling`: its existing spawn-only skip did not trigger, and it passed (previously skipped). No selection or test changed to produce that difference.
No test was retried after the freeze. No commit or push was made; the source
remains `420c0c0b388ba24da5b67248d1bffb37dd98570a2bb3fa4febc72b9a395981f0` and main remains `e4b72d69899a98676c6cbe032901a54018365e9e`.

The remaining failures are incomplete fixture ports, including the deliberately
removed producer tolerances and an insufficient pullback-count condition. They were not repaired after this frozen run.
The complete tracebacks and per-case diagnoses follow and are also retained in
[failure diagnoses](../benchmarks/2026-11-09-item2-0-round-3-4-repair/failure-diagnoses.json).
This result-only appendix is the authorized red return, appended after the
sweep and its equality audit; the sealed offered prose remains in the patch.

**1. `test/test_echoic_decoder.py::test_same_walk_infers_binary_and_unary_without_a_journal[False]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_echoic_decoder.py:122: in test_same_walk_infers_binary_and_unary_without_a_journal
    decoded = fn(event, bank)
              ^^^^^^^^^^^^^^^
test/test_echoic_decoder.py:118: in walk
    return Models.BasicModel._output_generate_walk(owner, event, 8,
bin/Models.py:13486: in _output_generate_walk
    _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
bin/Models.py:441: in _reconstruction_while_loop
    values = body(*values)
             ^^^^^^^^^^^^^
bin/Models.py:13397: in body
    if menu_read is None else menu_read.read(parent, live, **menu_options))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:663: in read
    result = inverse_menu(self.language, parent, live, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**2. `test/test_identification_evidence.py::test_abstract_unfold_descends_each_sigma_rung`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_identification_evidence.py:123: in test_abstract_unfold_descends_each_sigma_rung
    result = unfold_idea(language, basis, basis[2], 8, activation=torch.ones(3) * 2,
.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py:158: in no_grad
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:429: in read
    return function(language, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/MemoryIndex.py:71: in unfold_idea
    menu = inverse_menu(language, top, gate,
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**3. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[value-object]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
test/test_item20_round2_generation.py:94: in walk
    return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
bin/Models.py:13486: in _output_generate_walk
    _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
bin/Models.py:441: in _reconstruction_while_loop
    values = body(*values)
             ^^^^^^^^^^^^^
bin/Models.py:13397: in body
    if menu_read is None else menu_read.read(parent, live, **menu_options))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:663: in read
    result = inverse_menu(self.language, parent, live, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**4. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[value-group]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
test/test_item20_round2_generation.py:94: in walk
    return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
bin/Models.py:13486: in _output_generate_walk
    _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
bin/Models.py:441: in _reconstruction_while_loop
    values = body(*values)
             ^^^^^^^^^^^^^
bin/Models.py:13397: in body
    if menu_read is None else menu_read.read(parent, live, **menu_options))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:663: in read
    result = inverse_menu(self.language, parent, live, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**5. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[gradient-object]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
test/test_item20_round2_generation.py:94: in walk
    return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
bin/Models.py:13486: in _output_generate_walk
    _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
bin/Models.py:441: in _reconstruction_while_loop
    values = body(*values)
             ^^^^^^^^^^^^^
bin/Models.py:13397: in body
    if menu_read is None else menu_read.read(parent, live, **menu_options))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:663: in read
    result = inverse_menu(self.language, parent, live, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**6. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[gradient-group]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
test/test_item20_round2_generation.py:94: in walk
    return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
bin/Models.py:13486: in _output_generate_walk
    _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
bin/Models.py:441: in _reconstruction_while_loop
    values = body(*values)
             ^^^^^^^^^^^^^
bin/Models.py:13397: in body
    if menu_read is None else menu_read.read(parent, live, **menu_options))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:663: in read
    result = inverse_menu(self.language, parent, live, **options)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**7. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[compiled_gradient-object]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py:1276: in compile_wrapper
    raise e.with_traceback(
E   torch._dynamo.exc.UncapturedHigherOrderOpError: This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:
E
E   raised exception AttributeError("'SimpleNamespace' object has no attribute 'parts_of_speech'")
E     Higher Order Operator: torch.while_loop
E
E   from user code:
E      File "/Users/arogers/github/WikiOracle/basicmodel/test/test_item20_round2_generation.py", line 94, in walk
E       return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
E              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           basis_valid=torch.ones(1, 3, dtype=torch.bool),verb_groups=groups,
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           constituents=structure,constituent_valid=None if structure is None else torch.ones(1, 1, dtype=torch.bool),
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           candidate_limit=3,return_trace=True)
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 13486, in _output_generate_walk
E       _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 439, in _reconstruction_while_loop
E       return torch.while_loop(condition, body, values)
E              ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/while_loop.py", line 227, in while_loop
E       return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())
E              ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/while_loop.py", line 224, in flat_body_fn
E       return body_fn(*carried, *additional)
E              ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 13396, in body
E       menu = (inverse_menu(language, parent, live, **menu_options)
E               ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Generative.py", line 721, in inverse_menu
E       return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
E                                       ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/bin/GenerateOccupation.py", line 8, in features
E       speech = language.parts_of_speech
E                ^^^^^^^^^^^^^^^^^^^^^^^^
E
E   Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```

</details>

**8. `test/test_item20_round2_generation.py::test_free_walk_has_no_action_record_or_live_stm[compiled_gradient-group]`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_item20_round2_generation.py:101: in test_free_walk_has_no_action_record_or_live_stm
    result = walk(event)
             ^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py:1276: in compile_wrapper
    raise e.with_traceback(
E   torch._dynamo.exc.UncapturedHigherOrderOpError: This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:
E
E   raised exception AttributeError("'SimpleNamespace' object has no attribute 'parts_of_speech'")
E     Higher Order Operator: torch.while_loop
E
E   from user code:
E      File "/Users/arogers/github/WikiOracle/basicmodel/test/test_item20_round2_generation.py", line 94, in walk
E       return BasicModel._output_generate_walk(FreeOwner(), value, 6, basis=words,
E              ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           basis_valid=torch.ones(1, 3, dtype=torch.bool),verb_groups=groups,
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           constituents=structure,constituent_valid=None if structure is None else torch.ones(1, 1, dtype=torch.bool),
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E           candidate_limit=3,return_trace=True)
E           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 13486, in _output_generate_walk
E       _, stack, depth, emitted, count, actions, alternatives, _, _, tops, exact_path, residuals, features = _reconstruction_while_loop(
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 439, in _reconstruction_while_loop
E       return torch.while_loop(condition, body, values)
E              ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/while_loop.py", line 227, in while_loop
E       return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())
E              ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/.venv/lib/python3.12/site-packages/torch/_higher_order_ops/while_loop.py", line 224, in flat_body_fn
E       return body_fn(*carried, *additional)
E              ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Models.py", line 13396, in body
E       menu = (inverse_menu(language, parent, live, **menu_options)
E               ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/test/../bin/Generative.py", line 721, in inverse_menu
E       return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
E                                       ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E     File "/Users/arogers/github/WikiOracle/basicmodel/bin/GenerateOccupation.py", line 8, in features
E       speech = language.parts_of_speech
E                ^^^^^^^^^^^^^^^^^^^^^^^^
E
E   Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"
```

</details>

**9. `test/test_one_backward.py::test_thought_and_generate_features_stay_detached_when_compose_is_live`**

Missed fixture port after removal of the five-lane policy branch. test_one_backward supplies features of shape (1, 2, 5); the strict policy consumes eight menu lanes, so its occupation slice has zero columns and cannot multiply the three-column weight. The fixture needs (1, 2, 8), preserving its detached-feature and policy-gradient assertions.

<details>
<summary>Recorded traceback</summary>

```text
test/test_one_backward.py:247: in test_thought_and_generate_features_stay_detached_when_compose_is_live
    policy(semantic[None],features).square().sum().backward()
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1787: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1798: in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/Language.py:12363: in forward
    return logits + self.occupation_score(features[..., 5:]).squeeze(-1)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1787: in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1798: in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py:134: in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E   RuntimeError: mm: mat1 and mat2 shapes cannot be multiplied (2x0 and 3x1)
```

</details>

**10. `test/test_reference_index.py::test_unfold_candidates_are_the_current_priming_surface`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_reference_index.py:21: in test_unfold_candidates_are_the_current_priming_surface
    cold = unfold_idea(language, basis, basis[0], 4, activation=active)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py:158: in no_grad
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:429: in read
    return function(language, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/MemoryIndex.py:71: in unfold_idea
    menu = inverse_menu(language, top, gate,
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**11. `test/test_math_chain_repair.py::test_unfold_reads_the_readers_priming_row`**

Missed fixture port after removal of the producer's getattr fallback. This test's SimpleNamespace language lacks parts_of_speech and the explicit generate-rule inventory required by GenerateOccupation.features. Its toy algebra needs an explicit empty argument inventory; the behavioral assertion should remain unchanged. The compiled variants fail while capturing the same missing attribute. No producer fallback should be restored.

<details>
<summary>Recorded traceback</summary>

```text
test/test_math_chain_repair.py:78: in test_unfold_reads_the_readers_priming_row
    result = unfold_idea(language, basis, basis[row], 4, activation=lambda: weights, stream=row)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py:158: in no_grad
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
bin/Generative.py:429: in read
    return function(language, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/MemoryIndex.py:71: in unfold_idea
    menu = inverse_menu(language, top, gate,
bin/Generative.py:721: in inverse_menu
    return menu._replace(occupation=occupation_features(language, parent, menu.left, menu.right))
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
bin/GenerateOccupation.py:8: in features
    speech = language.parts_of_speech
             ^^^^^^^^^^^^^^^^^^^^^^^^
E   AttributeError: 'types.SimpleNamespace' object has no attribute 'parts_of_speech'
```

</details>

**12. `test/test_sentence_comparison.py::test_sentence_trials_keep_their_own_perception_pullbacks`**

The count port is still too narrow. The sweep observed [0, 0, 1, 1, 1] while the final compose non_tied flag was false, so the fixture expected only [0, 0, 1, 1]. It incorrectly treats that final compose flag as the sole source of an extra per-owner pullback. Other restored chooser or reconstruction.A costs can also traverse perception; this traceback does not identify which supplied the fifth read. A further fixture repair should observe each trial's actual cost owners and preserve the same-version, trial-specific-pullback claim. No rerun was made to attribute that fifth read. This failure supersedes the pre-sweep item 4 claim that the conditional count was sufficient.

<details>
<summary>Recorded traceback</summary>

```text
test/test_sentence_comparison.py:65: in test_sentence_trials_keep_their_own_perception_pullbacks
    assert called == [0, 0, 1, 1] + ([1] if chooser else []), called
E   AssertionError: [0, 0, 1, 1, 1]
E   assert [0, 0, 1, 1, 1] == [0, 0, 1, 1]
E
E     Left contains one more item: 1
E     Use -v to get more diff
```

</details>

### 16.2 Review of the §16 return (Claude, 2026-11-09): the repair accepted; 12 red cases, all fixture ports; landing with the ports

**Source and documents.** frozen-source.json `420c0c0b388ba24da5b67248d1bffb37dd98570a2bb3fa4febc72b9a395981f0` (1016 files) equals the live tree; before-source.json equals the §15 freeze `40676b500b3a…`; 53 files differ between them (13 under bin plus the new `bin/SentenceBoundary.py`, data/model.xml, 37 ported test files, the new test_item20_round34_repair.py). The production delta was read in full against Codex's before-tree copy. The seal over eight documents recomputes equal except this spec, which §16.1 appended after the sweep; final-audit.json records that append as the authorized red return. HEAD is still `e4b72d6989`; nothing committed or pushed.

**What the repair built, and that it matches §15.5.** (1) Both traversals of GradientProjection.backward retain the graph; `SentenceBoundary.finish`, called at the single-path boundary beside the stm/lang/feedback detach, detaches the finished sentence's choice records, roots, decoder trace, attention words, sampling scale, expectation comparison, gradient-objective reports, STM's live buffers whole (the active-prefix detach left the inactive rows' graph reachable through `cat`), the percept field and the last arrival, and rebuilds the threaded sub/sym and the symbol-presence slab by `torch.where(written, detached, perception)`, so unwritten positions keep the batch's perception graph. The structural certificate finds 0 leaked nodes over 14 carried tensors at the second sentence's entry, the three old choice-record pullbacks None, the perception reachable, two optimizer steps; test_learning_readiness passed 8 of 8 unseeded solo runs (eight-solo.json). (2) `Generative.WALK_FEATURE_WIDTH = 9` read at the dense placeholder (Models ~9892), the walk arrays and `generate_credit`; the 5-wide policy branch and `_load_from_state_dict` are gone; `generation_fixtures.suffix` is 9-wide; `undeclared_arguments` supplies a toy language an empty inventory. (3) `Error.add`/`error` take `side`, defaulting from `ObjectiveOwnership.default_side(term_owner(...))`; `registry_costs` keys by owner, or owner + '.A'/'.R' when the side is not the owner's default; `backward_owned` reads the owner before the first '.'; the registrations carry exactly the owner-and-side pairs of §15.5 item 3 (routes.json: 16 terms, each with owner and side); the option's direct case moves the compose logits toward the cheaper departure with the reader's gradient None. (5) The dead AMP line, `generate_policy_credit`, the unused `point`, the GenerateOccupation and `units` getattr defaults and the second occupation migration are gone; `_projection_writers` resets per batch; `reread` returns the value unchanged under no_grad; the model.xml comment is back above learningRate. The two ActionLearning call-site getattr defaults stay, with the reason recorded (bare `BaseModel()` checkpoint hosts have no languageSpace).

**Certificates.** The fourteen direct cases rerun from the live tree into the scratchpad (7.8 s): thirteen JSONs equal Codex's; native-default-projection.json differs in one unit (unseeded; in my run one unit was projected, in Codex's none). Gates in the sweep: test_convergence, the shared-concept case, the pending-premise pair, the XOR reconstruction class, the two new certificates all passed.

**The sweep.** 7890 cases: 7553 passed, 324 skipped, 1 xpassed, 12 failed, against the expected 7564/325/1; the one skip-to-pass is `test_data_loader_list_outputs_survive_worker_pickling`, whose spawn-only skip did not trigger in this environment. Of the 49 red cases of §15.4, 48 pass. The 12 are fixture ports, none a producer defect, each diagnosed in failure-diagnoses.json: ten toy languages without `parts_of_speech` after the getattr default was removed as §15.5 asked (test_echoic_decoder, test_identification_evidence, test_item20_round2_generation's `FreeOwner` with its six parameters including the two compiled variants, test_reference_index, test_math_chain_repair; the port is `undeclared_arguments(language)` from generation_fixtures); one 5-lane features fixture in test_one_backward::test_thought_and_generate_features_stay_detached_when_compose_is_live (8 lanes); and the pullback-count port in test_sentence_comparison::test_sentence_trials_keep_their_own_perception_pullbacks, whose fixed list `[0, 0, 1, 1] + [1 if non_tied]` met `[0, 0, 1, 1, 1]`: under the option `backward_owned` traverses the pullback once per cost key among reconstruction, chooser and attention, and `output.walk_residual.*` (owner reconstruction, side A) is a key of its own beside `reconstruction`, so the count is the number of such keys with a graph per trial; the fixture derives it from each trial's costs instead of a fixed list, keeping its claim (each trial's own perception at one parameter version).

**Warts, recorded.** `SentenceBoundary.finish` detaches by a list of attribute names; a new carried attribute would leak until the scan certificate catches it, which is what the certificate is for. Under the option two keys with one owner traverse the pullback twice (`reconstruction` and `reconstruction.A`); summing keys per owner before the traversal would save one pass. The dual-process and three-stage design is not in Architecture.md or Philosophy.md; its text for Codex is [2026-11-09-item2-0-dual-process-architecture.md](2026-11-09-item2-0-dual-process-architecture.md) (Alec, 2026-11-09), to be added at the round-5 build.

**Verdict.** The production repair is accepted; the tree is not landable until the 12 fixture ports are in. The production source does not change for them. Recommended landing form, as round 2's: the 12 ports with before/after text, a rerun of the six files, landing in the §11.7 form with "Land accepted item 2.0 round 3/4", no second sweep; alternative: one more sweep before the landing. Either way the landing is folded into the round-5 build text, as §13.2 schedules.

**Confirmed (Alec, 2026-11-09: "good recomendations").** Round 3/4 lands with the 12 fixture ports and a rerun of the six files, no second sweep; the landing is folded into the round-5 build text (§16.3). Round 5's seat initialization: identity and execution interface declared at construction, direction, confidence and participations learned (memory plan §10.5, decided the same day).

### 16.3 One Codex text: land round 3/4, build round 5 (Claude, 2026-11-09; for Alec to paste; final)

Preface for Alec. §16.2 said six files; the twelve ports touch seven. Assumptions stated inside the text: the LTM1 authority columns are replaced by one reference to the operation's seat concept, no row added, and the word-concept column retires where the operator word is already an endpoint; the verb and modal state terms are relative errors against the no-change prediction, so no weight knob returns; a supplied training form is declared a basic percept at admission while forms met in reading promote by recurrence, which keeps round 2's supplied-form report true; the pinned sentence's working form is the nested schematic unless Codex finds the flatter form cheaper, recorded. If any of these is not your intent, say so before pasting.

~~~
Item 2.0: land the merged round 3/4, then build round 5 (memory, basic percepts, the pinned sentence and the seats, verbs over the state of affairs). The review of the §16 return is 2.0 spec §16.2 (doc/plans/2026-11-03-gradient-projection.md); the offered source is 420c0c0b388ba24da5b67248d1bffb37dd98570a2bb3fa4febc72b9a395981f0, the landed round-2 source cff4fb245235c247ce8a14a582c883b5c574766acba221a95fe0cf4f0fd56f4e, main e4b72d69899a98676c6cbe032901a54018365e9e. Alec decided (2026-11-09): round 3/4 lands with the twelve fixture ports and a rerun of their seven files, no second sweep; the operator seats declare identity and execution interface at construction and learn direction, confidence and participations. Fast checks only until the implementation is finished. Steps 1 to 3 land; items 4 to 10 build round 5 and freeze it for review as §17; no commit or push of round 5.

1. Port and check. Twelve ports, before/after text in doc/benchmarks/2026-11-09-item2-0-round-3-4-repair/landing.json; nothing under bin/ or data/ changes. (a) The toy languages gain the declared empty inventory by test/generation_fixtures.undeclared_arguments(language), wrapping the SimpleNamespace at test/test_echoic_decoder.py line 25, test/test_identification_evidence.py line 117, test/test_item20_round2_generation.py line 25 (the module language FreeOwner reads; this clears all six parameters, the two compiled ones included), test/test_reference_index.py line 17 and test/test_math_chain_repair.py line 74. (b) test/test_one_backward.py line 246: features torch.ones(1, 2, 5, ...) becomes torch.ones(1, 2, 8, ...). (c) test/test_sentence_comparison.py lines 62 to 65: in place of the fixed list, wrap ObjectiveOwnership.backward_owned to record, per call, the number of costs with a graph whose key before its first '.' (after 'penalty.' is removed) is reconstruction, chooser or attention and has parameters in owners; assert called == [0] * count_0 + [1] * count_1, keep versions[0] == versions[1], and record the counts in perception-pullbacks.json beside the before list. Rerun the seven files (BASICMODEL_DEVICE=cpu PYTHONPATH=bin .venv/bin/python -m pytest -q test/test_echoic_decoder.py test/test_identification_evidence.py test/test_item20_round2_generation.py test/test_reference_index.py test/test_math_chain_repair.py test/test_one_backward.py test/test_sentence_comparison.py, with the writable inductor cache the repair used); all green. Record the tree's source hash by the receipt method as landed-source.json beside the repair's frozen-source.json, naming the seven files as the whole difference from 420c0c0b…. The handwritten documents changed since the sweep only by appends (§16.1, §16.2, two confirmation paragraphs, a memory-plan paragraph, a todo line, the new file doc/plans/2026-11-09-item2-0-dual-process-architecture.md); run test/test_doc_links.py once; it must pass. No make test: the repair's one sweep (7553 passed, 324 skipped, 1 xpassed, the twelve fixture failures) plus the seven-file rerun is the certificate.

2. Commit, on main, every file the round changed, added or deleted since cff4fb24…: the modified tracked files under bin/, data/, doc/, test/ and todo.md, the new modules bin/ActionLearning.py, bin/GenerateOccupation.py, bin/GradientProjection.py and bin/SentenceBoundary.py, the new tests test/test_item20_round34_actions.py, test_item20_round34_projection.py, test_item20_round34_repair.py and test_item20_round34_typing.py, and doc/plans/2026-11-09-item2-0-dual-process-architecture.md. Receipts stay local under the ignore rules; force nothing in. Message in the form of 6becc3dc47: subject "Land accepted item 2.0 round 3/4"; body: land the repaired freeze 420c0c0b with the §15.4 and §16.2 reviews and the twelve fixture ports named; what the round carries (the default projection with per-unit sides and owners kept for separateChooser true; generate under the proposal and the score-function credit, walk_policy retired; the verb and modal teachers retired, the assigner on the recorded verb-phrase meaning, the re-read, actionRateRatio; output typing as generate-policy features; the clamp velocity; the sentence-boundary detach with both traversals retaining the batch graph); validation (the repair's one sweep with its twelve fixture failures ported and rerun, the fourteen direct certificates, §15's one XOR gate); generated evidence local under the ignore rules; no model or assistant identifier.

3. Record. Append "Landed as <commit hash>, 2026-11-09" after §16.3 in the 2.0 spec and a line in todo.md; commit "Record item 2.0 round 3/4 landing"; push once. Then round 5 starts from that tree; before-source.json of its receipt is the landed hash.

4. Memory: one link, references, promotion (memory spec doc/plans/2026-11-03-item2-0-behavioral-chains.md §§3.2 to 3.5 as §9 and its 2026-11-05 additions amend them; §3.1 landed in round 1). ConceptualSpace.link(occurrence, candidates, *, value, where, when, stream), called from forward() at the kept close in place of ReferenceClose.commit's bind and _commit_events, with three callers: reference sites (today's binds and commit_read_words' familiar candidates, questions included), verb events, and each kept sentence's order-0 state (never an estimate, a question, a cover, a modal or an unlowered idea). A member is a reference: the .when of the STM0 or LTM0 entry and the concept referenced within that idea; a word reads the concept it denotes, a mention without a word reads its own idea, a known individual or the promoted concept is one reference, a kept thought's written idea is a member. Values are computed from the members when read (MediumTermMemory.value_of over references); Token.value, the timelines' stored values and the join in open and bind retire; Event keeps before, verb_meaning, basis and modifiers as the recorded forward operands of the re-read, and its after retires with the stored values; the state version becomes 12, old versions load with their stored values dropped and reported. Re-entry (EntryRead.recall_identities, recall_chunks) records the recalled member's link time as the stream's current close and joins an individual already open in MTM instead of opening a duplicate. The path: link order is time order; merge interleaves by close clock (test_reference_arrival_review ported with before/after). Enforced: the window (a collection is a candidate only while its latest member is within mtmHorizon kept closes, read from refinement['mentions'], which every link and re-entry now records) and likeness (one order, one kind set by the first member; containment: one of IdentityRows.meet_ratios' two coverages exactly 1). Scored at zero weight: where-band overlap with the tail member, the gap over mtmHorizon, the first difference against the tail member, presence in STM's chunks; the policy row grows from eight to twelve features, old checkpoints zero-padded and reported; the choice is credited by the departure rule, output side whole. Candidates from MTM's pool and STM's chunks alike (D4). Runs form concepts at every level: an extended event's state is computed from its members when read, one order lower, no update sets it; the run keeps its path order. Promotion by the one rule: conceptMintCount recurrences within horizon promote a collection one order above its members through promote_identity's path, writing membership rows (OCCURRENCE, m_i) -> (SYMBOL, cid) by _resolve_null_token's writer and successor rows (OCCURRENCE, m_{i-1}) -> (OCCURRENCE, m_i) with m_i's bands (LTM1.add, witness=True); the epoch boundary withdraws them and keeps the concept; ChainRead.path(cid) reads the path back with a visited set. Nouns from chains: an identity that is the participant of conceptMintCount consecutive links of one collection of states receives one mention witness per link and mints through promote_identity with no new count. The judge: reconstruction.reference (Models about 20364; ReferenceReconstruction.cost) becomes memory.link_coverage on every link, owner reconstruction, side R, its gradient reaching what the coverage reads, the choice discrete. Certificates: memory spec §4 items 2 to 9 (the three callers on one fixture, with two consecutive kept closes linking two order-0 states, a recurrence, a distant close taking the dummy, a modal clause adding no link; the window and likeness; the path after merge; re-entry's link time and the join; promotion with membership and successor rows, the id released, the path read back and stopped on a cycle, the boundary withdrawing the rows; nouns from chains; no concept at order 0, collections of states absent from stage's leading field, the input reconstruction unchanged with STM and MTM emptied; the judge in the registry); the unchanged MTM, identity, timeline, familiar-tier, forgetting and reference files on the same hash, ported with before/after where D4 changes a choice, each named with the reason.

5. Verbs and modals over the state of affairs (loop spec §3.8; 2.0 spec §8: "the verb and modal teachers are MTM"). At the kept close, in training only, inside link's caller for events: for an event member whose identity has a later reference, memory.verb_state is a relative error (Error.error): the error is the cube-chart squared distance on defined coordinates (ModalLearning.residual's form) between the lift action's forward on the recorded operands, verb_action()(before, verb_meaning, basis or group), and the identity's computed value at its next reference, detached; the baseline is the same distance between before and that value, the no-change prediction, so the term needs no weight. memory.modal_state likewise for the modifier action: the actual timeline's computed value mapped through the event's modifiers against the alternative timeline's computed value at equal time, baseline the distance between actual and alternative. Both owner reconstruction, side R, registered in the close registry (_sentence_close_errors) so the single backward carries them; their gradient reaches the lift and adverb parameters (the actionRateRatio groups) and nothing stored; the assigner and the re-read stay as landed. Certificates: a direct two-sentence fixture where the second reading moves the identity's state: both terms finite, the verb term's gradient reaching verb_action().readout and no stored tensor, the error equal to the baseline when the action is the identity map, the modal term on an alternative() timeline; the round 3/4 re-read case still passing; under evaluation no term registered.

6. Shared strata, the pinned sentence, the seats, authority (memory plan §10.1, §10.5, §10.6; the seat policy decided 2026-11-09). STM0 holds live ideas with LTM0's storage and read implementation and a retention policy; STM1 holds ordered binary branches with LTM1's two tagged references per row; STMChunks stays the eight-entry recent buffer (stmCapacity). During the pinned sentence (pin, the stack flag of ThoughtChunks.put) its derivation tree lives in these strata: word occurrences and operator-concept occurrences in STM0, ordered binary branches in STM1, the operator present in the referenced content and not in a column; nested pairs pair(pair(operator, left), right) and pair(operator, argument), or a flatter working form if cheaper, recorded; sized by the sentence-length constant and preallocated (n word and n-1 operator occurrences, at most 2(n-1) branches), separate from stmCapacity, never resized. At the kept close the tree collapses in STMChunks.keep_close: the sentence's compacted idea stays in STM0 as a recent entry, the relation rows PartClosing._write adds stay in STM1 and LTM1, the branches and operator occurrences are freed; LTM0 and LTM1 hold compacted ideas and relations only. Reconstruction's derivation inverse (DerivationReconstruction), conditioning and MTM read the shared structure; the ChunkJournal record and the Composition objects stop being an authoritative second tree (views at most; no hidden side table), and decoding from a retained LTM0 idea requires none of it (§10.2: generative matching). Operator concept seats: one concept per primitive operation of the configured grammar (_compose_binary_rules and _compose_unary_rules, named by operation_name), seated at construction with its identity and its execution interface (the native operation implementation; FIXED_OPERATIONS membership read from the seat), its direction, confidence, participations, relations and applicability learned from experience (PartsOfSpeech.observe_kept teaches the seat's participations as it teaches the argument concepts); no seat enters a lexical example or gets a surface spelling. Authority onto the seats' trust: LTM1.refresh_effective's write(row) scales a row by min(the endpoints' trusts, the operation seat's trust by positive_length, gates.cap(named, operation=the seat's name)); the categorizer's occupations.operation(name)['trust'] factor retires in favour of the seat's trust; the grammar_operation catalog index becomes the reference to the seat concept, the one integer the row keeps (no row added; buffer renamed), and the operator word-concept column retires where the operator word is an endpoint of the row, concept(operator) being that endpoint's trust; a row whose operator word is not an endpoint is recorded with its case and keeps the reference. _primitive_evidence and the gate-reopening function (gates.advance over measurements) are preserved; a checkpoint migration renames the column and drops the retired one, reported. Certificates: memory plan §10.4's list (one data/read implementation for STM0/LTM0 and STM1/LTM1; common reads by reconstruction, conditioning and MTM; reference validity across short-term expiry; two operations with compatible argument categories distinguished by their seat references; one retained LTM0 idea unfolding to a surface with the input journal and the input STM absent, judged as meaning-compatible realization); after a kept close STM0 holds the compacted idea and no operator occurrence and STM1 no branch of that sentence; a row's effective truth equals the min above on a hand-built row; the input reconstruction unchanged.

7. Basic percepts (basic-percepts spec doc/plans/2026-11-04-item2-0-basic-percepts.md §§3.1 to 3.3, §8, §9; D1 and D2 decided; D3 landed in round 2). BasicPercepts: a preallocated table of basicPerceptCapacity rows (data/model.xml and model.xsd, 262144 where the percept store has word identity; lesson configurations may set less; Params.md), a row recording its parts (identity atoms, nested rows, radix chunks) and wholes (properties) with its code at their centroid (the identity join of the parts and the property meet of the wholes, MereologicalCodes._subspace_bounds moved into perception), laid out in the item 2.8 subspaces; a full table raises. Its read at a bracket ties parts and wholes live: the minimum that fold_presence takes over both towers today, with the complement pole from the property lanes, so the parallel field keeps its gradient to PrimitiveProperties.members and the fill written without gradient goes. Order-0 concepts collect basic percepts: features column 2*bp + pole; the literal writers (_populate_concept_weights, add_concept_feature, lookup_word, FamiliarTier, ConceptLessons, _witness_concept_poles) write basic-percept literals; cs_read_memberships reads basic-percept evidence per bracket and folds by the existing sigma and pi (_compose_order0 unchanged, the working[:n0] contract kept); the lexical gate and the cross-tower minimum leave the concept read; a concept that is not a word unit collects the basic percepts carrying its property or part (D2). Symbols have no mereological read: MereologicalCodes.derive stops deriving order-0 forms from the two towers and FormContainment.audit's premises move to basic percepts. The compose-time whole read (FormCentroids, MeaningKernels.centroid, with item 2.5's presence weighting) and the per-word typed slots (IndexedSigmaConceptsFromPercepts, admit_references) keep their logic and read basic percepts; thinking spec §45.8's gradient receipts are re-taken after the change. MTM creates basic percepts (§9): an occurrence of a basic-level percept, a word Interpret.lookup_word admits, is a link member at the perceptual level, its parts and wholes never members alone; a collection recurring conceptMintCount times within horizon promotes to a BasicPercepts row, which replaces admission as the writer; before promotion the occurrence reads and reverses through its own parts and wholes; a supplied training form (Corpus.admit_training_forms) is declared a basic percept at admission, so round 2's three-stage report and yes in every applicable bank stand. The input reconstruction reads no MTM state: promoted rows persist in the table. The reverse through basic percepts (§3.4) is round 6's, with the loop's reverse. Certificates: basic-percepts §4 items 1 to 5 and 7 (preallocation from model.xml and a full table raising; every admitted word one basic percept at its centroid; no part or whole literal column left in order-0 features; the cross-tower minimum absent from cs_read_memberships and present in the basic-percept read; the parallel field's gradient reaching members through it; FormContainment.audit on basic percepts), promotion of a recurring basic-level collection writing one row, the supplied form declared at admission, and the §45.8 receipts.

8. The elided role's content (thinking spec §54 finding 2, carried here with the full reference form). At the kept close a reference site with no word, an elided role, takes as its content the description the expected role's witnesses established, the identity's prior reference in that role read as a member reference, so that link has a candidate for it; where no witness established the role the site provides no identifying content, as §§50 and 52 certify. G3 (test/test_bind_thought.py about 232 to 238) drops its fixture-supplied content and passes on the native supply, ported with before/after.

9. Documents, by append or exact replacement: doc/plans/2026-11-09-item2-0-dual-process-architecture.md §§1 to 4 into Architecture.md as a dated section "Dual process: System 1, knowing and System 2" with a status bullet at the document's head, decided-not-built statements marked with their round, and §5 into Philosophy.md under "Object Files, File Cards and Dharmakīrti's Particulars", both by append; the memory spec §6's documents (Architecture: the one link and the references; the FutureWork pointer; Orders by append, order 1's continuity as every collection's connected path); the memory plan §10's (Architecture: the shared strata, the collapse, the seats, authority onto their trust); the basic-percepts spec §3.5's (Orders §1 and §9, Mereology, Architecture's "Two spaces, one index" and its order-alternation passage, Spaces.md, the item 2.8 spec's dated note) and Params.md with model.xsd for basicPerceptCapacity; GradientFlow (memory.link_coverage, memory.verb_state and memory.modal_state with owner and side; reconstruction.reference retired); Testing one line; your return as §17 of the 2.0 spec, each item with its before-and-after numbers.

10. Carried, not built here: round 6 (reconstruction from the parallel field, reverse as the one decoder through basic percepts, generate with the occurrence writer, loop spec D1 to D6, the §10.2 measurements, the decoder's bookings to output); momentum-carried projection and a learned pair policy, conditional on the end-of-item receipts; events as mereological neighbours (future work); item 1.5's port onto the new features layout.

Testing: fast checks only until the implementation is finished. Each item's direct certificates; the targeted files (the memory, identity, timeline, familiar-tier, forgetting and reference files for items 4 to 6 and 8; the word-identity, mereology, containment, parts-of-speech and compose-centroid files for item 7; every file you port); one make test when you offer the freeze (env -u RUN_TRAINED -u RUN_SLOW make test, outside the sandbox, nothing deselected; the XOR class, the pending-premise pair and the shared-concept case inside it). Expect the landed counts (7565 passed, 324 or 325 skipped by the spawn-only skip, 1 xpassed) plus the new direct cases, less each case a port retires or re-parameterizes, each named. Receipt under doc/benchmarks/2026-11-09-item2-0-round-5/ with commands.json, before-source.json (the landed round-3/4 source of step 3), frozen-source.json, source.patch, ports.json (every port's before/after text), offered-documents.json sealed before the sweep, placeholders filled only by the one execution; every file your return links to exists before the sweep starts. No seed pinning, no resize, nothing retired without a record, no language named in code, fail loud. Return through Alec for Claude's review; no commit or push of round 5.
~~~
