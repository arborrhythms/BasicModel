# Gradient projection and one choice mechanism

2026-11-03. Alec's decisions and Claude's proposals from the gradient-factoring discussion that followed the review of the §39 return (`doc/plans/2026-10-27-item2-5-thinking.md` §40). None of this is in a Codex addendum yet. These items join the next round, after the review of the §41 return, once Alec confirms the points in §5.

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
| R | the reconstruction owner (field, part, iterations, and the teachers `reconstruction.decomposition` and `reconstruction.walk_policy`), `compose_lesson`, expectation, attention, the penalty (once) |
| A | output (answer, prediction and the readers), `generate_lesson`, every comparison credit |

The comparison credits are:
- `reconstruction.compose_score_function`;
- `expectation.bind_score_function`;
- `attention.candidate_departure`;
- the thought score function;
- `reconstruction.walk_credit`.

All of them go on the A side whole (proposed, §5):

- Each is judged by the keep rule's R + A cost. Kept whole, a chooser is trained toward what the keep rule keeps.
- Splitting $\Delta C = \Delta R + \Delta A$ across the sides would remove the A part wherever the two parts disagree in sign. The chooser would then be trained away from departures the keep rule kept.
- With compose features live, the compose credit writes the concepts it reads, so the chooser can use semantic information (Alec, 2026-11-03). On the A side those writes are subordinate to reconstruction. On weights that only credits write, the credit applies in full.

Compose is part of reconstruction and generate is part of output (Alec, 2026-11-03), so `compose_lesson` is on the R side and `generate_lesson` on the A side.

The generate policy also decodes the understanding for reconstruction (`Models._reconstruct_trial`), so it has writers on both sides. The `walk_policy` teacher is on R. Both walk credits and the generate lesson are on A. The projection reconciles them on the policy's tensors.

Code to change: `backward_owned` reads `compose_lesson`, `generate_lesson` and `chooser` as reconstruction. `GrammarLessons` already books `generate.choice` with objective output.

The verb and modal teachers leave both sums (§4).

## 3. One mechanism for every choice (Alec's point 2)

Generate does try two walks, in two places:
- **The reconstruction decoder** (`Models._reconstruct_trial`) costs a greedy walk and one departure by identity cost.
- **The output** (`WalkTrials.output_pair`) does the same by answer cost.

Both credits are summed into `walk_credit`. Three things still differ from compose:

| | compose, attention, thought | generate (decoder and output walks) |
|---|---|---|
| Walks | greedy and one departure | greedy and one departure |
| Departure step | uniform over eligible steps | uniform over eligible steps (`WalkTrials.departure_at`) |
| Alternative action | $\phi$-mixture (`Expectation.alternative_probabilities`) | the policy's softmax at $\tau = 1$, greedy action excluded (`Language.sample_eligible_logits`) |
| Credit | $\pi(a)\,\Delta C/q(a)$, $q$ including the uniform step draw (`SentenceCredit.score_function`) | cross-entropy toward the explored action where it won, nothing where it lost (`WalkTrials.generate_credit`) |
| Teacher | none (lessons on labelled input) | `walk_policy`: cross-entropy on the actual compose tree's detached states. Each leaf teaches STOP; each parent teaches its rule's generate face |

So the softmax over the two walks does not unify `walk_policy` with `walk_credit`: the teacher's targets come from the input's compose tree, not from either walk. Proposed unification:

1. Generate draws its alternative from `alternative_probabilities`, with the same $\phi$ and $\tau$.
2. Generate's credit becomes `score_function`, with the same proposal correction and $\Delta C$ from its own comparison. In each case $\Delta C$ is the part of R + A that the choice can change:
   - for the decoder, identity cost plus residual;
   - for the output, answer cost plus residual.
3. `walk_policy` stays as a teacher on the R side, because it is not a choice. It states that generate inverts compose. It gives every node a target on every sentence, where a comparison credit gives one departure per walk. A later option is to make the compose tree's inverse one component of generate's proposal, so that it is learned through the same comparison. That would remove the teacher, at the cost of its per-node signal.

Every choice then has one proposal family, one comparison, one judge and one credit: compose operation and location, attention, thought, and generate. Teachers whose targets come from the input are R-side terms.

## 4. Verb and modal teachers (Alec's concern, saved)

- The verb teacher trains the `lift` operator's `MeaningAction` (`Language.verb_action`). The modal teacher trains the `adverb` operator's (`Language.modifier_action`).
- Both detach their inputs (`MeaningAction.teacher`, `ModalLearning.teacher`). Each therefore writes only its action's readout, group reflections and object vectors.
- Compose uses the same action when reading (the `Layers` forward) and when inverting (`reverse_blocks`). Reconstruction therefore writes it too, and under the single backward so does output.
- Both are registered at the close with weight .1, in the reconstruction owner, so jitter scales them with reconstruction.

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

## 5. For Alec to confirm

1. Comparison credits whole on the A side (§2).
2. Generate's proposal and credit made the same as compose's, with `walk_policy` kept as an R-side teacher (§3).
3. The teacher step at $\kappa\eta$ and $\kappa^2\eta$ with $\kappa = .1$, or the parameter-group alternative (§4).

Carried into the same round:
- `separateChooser` false as the default (§1);
- the item 60 metrics, which also judge this change. They drop chooser variance as a criterion and use instead:
  - held-out cost;
  - acquisition;
  - answers;
  - each chooser's departure win rate.
