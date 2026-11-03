# Gradient projection and one choice mechanism

2026-11-03. Alec's decisions and Claude's proposals from the gradient-factoring discussion that followed the review of the §39 return (`doc/plans/2026-10-27-item2-5-thinking.md` §40). None of this is in a Codex addendum yet. These items join the next round, after the review of the §41 return, once Alec confirms the points in §7.

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

## 2. Which side each term is on

| Side | Terms |
|---|---|
| R | the reconstruction owner (field, part, iterations, walk residual, forward discard, content, reference), `compose_lesson`, expectation, attention, the penalty (once), the decoder walk's credit |
| A | output (answer, prediction and the readers), `generate_lesson`, the output walk's credit, and every credit judged by R + A |

The credits judged by R + A are:
- `reconstruction.compose_score_function`;
- `expectation.bind_score_function`;
- `attention.candidate_departure`;
- the thought score function.

They go on the A side whole (proposed, §7):

- Their judge is the keep rule's R + A cost. Kept whole, a chooser is trained toward what the keep rule keeps.
- Splitting $\Delta C = \Delta R + \Delta A$ across the sides would remove the A part wherever the two parts disagree in sign. The chooser would then be trained away from departures the keep rule kept.
- With compose features live, the compose credit writes the concepts it reads, so the chooser can use semantic information (Alec, 2026-11-03). On the A side those writes are subordinate to reconstruction. On weights that only credits write, the credit applies in full.

Compose is part of reconstruction and generate is part of output (Alec, 2026-11-03), so `compose_lesson` is on the R side and `generate_lesson` on the A side.

The generate policy also decodes the understanding for reconstruction (`Models._reconstruct_trial`). The decoder walk's credit is judged by identity cost (reconstruction alone), so it sits on R. The output walk's credit is judged by the answer, so it sits on A. The two are summed into one `reconstruction.walk_credit` today and need separate terms. The projection then gives the policy's reconstruction role priority over its output role.

Code to change: `backward_owned` reads `compose_lesson`, `generate_lesson` and `chooser` as reconstruction. `GrammarLessons` already books `generate.choice` with objective output.

The two retired teachers (§3, §4) leave the table, and the verb and modal teachers leave both sums (§5).

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

## 5. Verb and modal teachers (Alec's concern, saved)

- The verb teacher trains the `lift` operator's `MeaningAction` (`Language.verb_action`). The modal teacher trains the `adverb` operator's (`Language.modifier_action`).
- Both detach their inputs (`MeaningAction.teacher`, `ModalLearning.teacher`). Each therefore writes only its action's readout, group reflections and object vectors.
- Compose uses the same action when reading (the `Layers` forward) and when inverting (`reverse_blocks`). Reconstruction therefore writes it too, and under the single backward so does output.
- Both are registered at the close with weight .1, in the reconstruction owner, so jitter scales them with reconstruction.

**Both also take their targets from the model's own derivation (inventory, 2026-11-03).** That bears on §3's ruling:
- **Verb.** Its input is the operand state of the recorded `lift` action in the compose frames. Its target is the referent's medium-term value, which the model's own compose transaction set, joined with the re-mention's composed value at the later close.
- **Modal.** Its target and input are two compose frames from the same tick: the alternative timeline's result and the actual timeline's result, paired by the model's identity handle. It is a commutation constraint between two outputs of the same adverb action.

Neither teaches a choice, and neither is an external observation. Alec to rule whether they stay (§7). If they stay, the rate method below applies.

Alec's principle (2026-11-03) is that their rate of adaptation must be less than that of their underlying constituents:
- the verb action's constituents are the concept codes it maps (the state before and after, the verb, the object);
- the modal action's constituents are events the verb action produced, plus modifiers.

The order is therefore: codes, then the first difference (the verb), then the second difference (the modal).

Proposed method:

- **The rate lives on the step, not on the loss weight.** Wherever a teacher is the main writer of a coordinate, Adam divides out its weight, and the step is about $\eta$ whatever the weight is.
- **The teachers get their own step.**
  - A third gradient call computes the teacher terms alone. It is cheap, because their inputs are detached: the graph is just the action applied to a few stored vectors.
  - Its own Adam state applies the result, at $\kappa\eta$ for the verb teacher and $\kappa^2\eta$ for the modal teacher.
  - $\eta$ is the base learning rate (0.001) and $\kappa$ a configured ratio (.1 proposed).
- **Each order of difference then adapts a factor $\kappa$ slower than what it is built on.** Reconstruction's own writes to the actions keep the full rate, and the teacher terms leave the R and A sums.
- **The ordering is measured.** The receipt reports the realized relative step $\|\Delta\theta\|/\|\theta\|$ per epoch for the codes, the verb action and the modal action.

A simpler alternative is optimizer parameter groups at $\kappa\eta$ and $\kappa^2\eta$ for the two actions. That saves the third call. But it also slows reconstruction's writes to the actions, and it leaves the teacher-to-reconstruction mix on them unchanged.

## 6. Other trained terms evaluated on the model's internals

A sweep of every trained registration in `bin/` (2026-11-03, read-only) found no other teacher of a choice. It did find terms that are evaluated on the model's own states. None of them supplies a target to a choice, so the proposal is to keep them this round:

| Term | What it constrains | Default |
|---|---|---|
| `reconstruction.walk_residual` | operator round trip: recomposing the selected inverse's children against the walk's own parent; also in the decoder and output judges | on |
| `reconstruction.forward_discard` | meaning lost at coordinates where a step's inverse is undefined, along the kept derivation | on with the sentence field |
| `<category>.definition_fold` | each class's sigma read covering the members the model itself minted or read as witnessed; possibly sigma's main gradient writer (unverified) | conditional |
| `reconstruction.reference` | reward only: the record of the model's own bind choice | on, weight .1 |
| `reconstruction.field` | the core reconstruction: a training-only reader follows the committed compose tree to the model's own codebook atoms of the input words | on |

**Note on `reconstruction.field`.** While the field is on, byte-level reconstruction (`free_bytes`) is not trained. Every reconstruction writer therefore targets the model's own encodings or its self-consistency, not raw input. A later round could move reconstruction onto the free decoder, which chooses its own tree, against bytes and percepts. That would remove the last training-only path through compose internals. It is not proposed now.

## 7. For Alec to confirm

1. Credits judged by R + A whole on the A side, with the decoder walk's credit split out onto R (§2).
2. Generate's proposal and credit made the same as compose's; `walk_policy` retired (§3; the retirement is Alec's ruling).
3. The decomposition teacher retired, and the chooser kept at the residual argmin (§4).
4. Whether the verb and modal teachers stay, given that they constrain the model's own states (§5). If they stay: their own step at $\kappa\eta$ and $\kappa^2\eta$ with $\kappa = .1$, or the parameter-group alternative.
5. The terms in §6 kept this round.

Carried into the same round:
- `separateChooser` false as the default (§1);
- the item 60 metrics, which also judge this change. They drop chooser variance as a criterion and use instead:
  - held-out cost;
  - acquisition;
  - answers;
  - each chooser's departure win rate.

Their per-epoch traces for the generate policy and the decoder's identity cost show whether retiring `walk_policy` costs acquisition.
