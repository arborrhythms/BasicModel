# Conflicts between the training objectives

> **Status:** analysis and proposal, written by Claude on 2026-10-01 at
> Alec's request. Nothing in the code is changed by this document. Alec
> answered the questions of §7 the same day; the design that follows from
> his answers is §8, revised in §10. There is no separate item (Alec,
> 2026-10-01: "No separate item, it would take too long. Add it to the
> current work, or even better, the next work"): the cost function is
> item 6.9's next work, after the memory repair, the production
> measurement and the suite trim
> ([6.9 plan §12](../plans/2026-09-29-item-6-9-xor-grammar.md#12-remaining-steps-2026-10-01)).

## 1. What is asked, and why now

Alec, 2026-10-01: "The gradient and what it reaches had to change in order
to get things to work; that's good. But it reintroduces a conflict in the
gradient objectives; reconstruction and output. So I want to make sure that
a detailed description of the gradient is captured in the doc/*.md, and
that we do a more careful analysis of how to avoid conflicts in the cost
function should the objectives be at odds with one another."

The decision of 2026-09-20 avoided the conflict by construction: every
objective differentiates only its own computation, the state is cut at the
concluded idea, and the shared operators are the one intended coupling,
with their disagreement measured rather than corrected
([next-sentence plan §8.4](../plans/2026-09-15-next-sentence-as-the-production-objective.md#84-gradient-boundaries-and-learning-evidence)).
Item 6.9 showed that with the cut in place the answer cannot teach the
grammar a composition it can read: XOR_grammar met its bar in 0 of 10 runs,
and in 9 of 10 once a supplied answer's error reached the chooser and the
codes ([6.9 plan §3.13](../plans/2026-09-29-item-6-9-xor-grammar.md#313-after-step-5-what-holds-xor-back-measured-2026-09-30-and-10-01)).
Alec then decided that a supplied answer trains the whole path (step 5a).
Reconstruction, expectation and the answer now meet on the same parameters
inside one cost, so they can pull those parameters in different directions.

The description of the gradient itself is kept in
[GradientFlow](../GradientFlow.md); this document is the analysis.

## 2. The objectives and what they share, after step 5a

Every sentence that closes is derived twice, by the greedy derivation and
by one explore derivation. Both are costed under the same parameters, then
each trial's cost is backpropagated and the optimizer steps once for each.
The strictly cheaper trial is kept and committed, detached
([6.9 plan §4 step 5](../plans/2026-09-29-item-6-9-xor-grammar.md#4-the-plan)).
After the last sentence the batch takes one more optimizer step on the
batch-end objectives.

| Objective | Where it is costed | Its weight | What its gradient reaches |
|---|---|---|---|
| Reconstruction, tied | each trial, when the configuration reconstructs in the loop | `reconstructionScale` | compose (chooser and operators) through the recorded derivation's tied inverse, the leaves, perception through the trial's pullback |
| Reconstruction, perceptual | batch end | `reconstructionScale` | perception's reverse and the per-word state it renders |
| Expectation | each trial | `interLossWeight`, `interContrastiveWeight` | the predictor and its live sources; the chooser through the trial's costs (measured, [6.9 plan §3.4](../plans/2026-09-29-item-6-9-xor-grammar.md#34-what-the-answers-error-trains)) |
| Grammar lesson | each trial, when reading lessons are on | `grammarLessonWeight` | the compose and generate policies and operators it supervises |
| **Supplied answer, in the trial (new)** | each trial of a sentence with a supplied answer | none today: added as it is, in the output's units | the reading map (the numeric head, or generate in `answerSynthesis` configurations), and through the uncut understanding the chooser, the operators, the leaves and perception |
| Supplied answer, at batch end | batch end | `1 - reconstructionScale` | the reading map only: the understanding is cut there |
| Thought and generate policies | their own explicit credit | their weights | their choosers only |

Two facts decide how far the new term reaches:

* **The codes depend on the configuration.** Where `conceptualContextLearningRate`
  is set (the eleven canonical configurations, `BasicModel.xml` among them),
  the concept dictionary is outside autograd and only the sentence-local
  rotation moves it; the answer reaches the chooser and the operators but
  not the codes. Elsewhere (XOR_grammar, MM_grammar) the dictionary is a
  parameter, and every objective that reaches a leaf moves it.
* **The weights differ between the trial and the batch end.** In
  XOR_grammar `reconstructionScale` is 0.1, so the batch end weighs the
  answer 0.9 while the trial adds it at 1. In `BasicModel.xml`
  `reconstructionScale` is 1.0, which gives the answer no weight in that
  batch-end sum; the supplied answer in the trial has weight 1. (How the
  `answerSynthesis` path weighs its batch-end answer is to be confirmed by
  the measurement of §6.)

The shared parameter groups are therefore: perception, the codes (where
trainable), the compose chooser, the compose operators and their tied
inverses, the generate path, and the reading map.

## 3. How a conflict shows up

1. **Direction.** Two objectives' gradients on the same parameters point
   apart (their cosine is negative). The summed step can then raise one
   objective's cost while it lowers the other's.
2. **Magnitude.** One objective's gradient is much larger than another's,
   so it decides the step whatever the other needs. The answer term enters
   the trial in output units with no weight, while reconstruction enters in
   byte cost under `reconstructionScale`; nothing makes their sizes
   comparable. The existing diagnostic has seen ratios of thousands
   ([GradientFlow, per-operator agreement](../GradientFlow.md#per-operator-agreement)).
3. **Selection.** The kept trial is the one whose summed cost is lower. A
   trial that answers better and reconstructs worse can win, and the
   committed history then records that trade. This is a conflict in the
   choice, not in the gradient, and no gradient diagnostic sees it.
4. **Capacity.** Even optimized perfectly, no single setting may satisfy
   both. XOR needs an understanding in which the two words interact; a
   reconstruction that admits only transpositions needs one from which both
   words can be recovered. Both can hold at once (an injective, XOR-readable
   encoding of four sentences exists), but whether they can with ten code
   numbers and min, max and `not` is an empirical question.
5. **Turns.** Each sentence's two trials and the batch end each take a full
   optimizer step, at different parameter versions, and Adam's moment
   estimates mix all the objectives. A conflict can therefore appear as
   oscillation between steps rather than as one negative cosine.

## 4. What is measured today, and what is not

The per-operator agreement diagnostic (`branchDiagnosticsEvery`) reports,
for each named grammar operator, the weighted gradient norms of the
reconstruction, output and expectation branches, their cosines, and the
norm ratios; three negative observations in a row mark persistent
opposition. It is evidence only: no clipping or projection is applied, as
decided on 2026-09-20. Step 5a adds the trial's answer to the trial's
`output` objective, so the diagnostic now includes it.

What it does not cover:

* **The chooser, the codes and the reading map.** It compares named
  operators only, and excludes codebooks. Step 5a's new reach is exactly
  the chooser and the codes.
* **One parameter state.** It sums each trial's gradient across the actual
  optimizer updates, so a cosine mixes parameter versions. Since step 5
  both trials are costed under the same parameters before either step,
  which is the moment a clean, single-state comparison can be taken.
* **The choice between trials.** Nothing records whether the kept trial
  was worse than the other on some objective.
* **What a conflict costs.** A negative cosine says the objectives
  disagree, not that either got worse. The outcome that matters is each
  objective's own cost with and without the other.

## 5. Ways to avoid or resolve a conflict

Each is listed with what it guarantees, what it costs here, and how it
fits what has been decided.

* **A. Comparable scales.** Put every term in comparable units before it
  is summed: a fixed weight per objective, a running normalization by each
  cost's own size, uncertainty weighting (Kendall, Gal and Cipolla, 2018),
  or balancing gradient norms on the shared parameters (GradNorm, Chen et
  al., 2018). It addresses magnitude, not direction. It is cheap. The
  answer-construction costs of the What path are already normalized
  independently ([Training](../Training.md)), and the trial's answer term
  is not.
* **B. A priority stated as a constraint.** Reconstruction (and Alec's
  "errors are transpositions") becomes a constraint: its cost may not rise
  above a budget, and the answer is optimized within it. A Lagrange
  multiplier rises while the budget is exceeded and falls back when it is
  met (Platt and Barr, 1988; used for reconstruction budgets as GECO by
  Rezende and Viola, 2018). It keeps one backward per trial, it states the
  priority explicitly, and its one extra number per constraint is easy to
  log. It needs a budget, for example the reconstruction cost without the
  answer term.
* **C. Pareto choice between the trials.** A trial is kept only if it is
  no worse than the other on any objective, within a tolerance, and better
  on at least one; otherwise the greedy trial stays. This removes the
  selection conflict of §3.3 entirely, changes no gradient, and costs
  nothing. It can keep the greedy trial more often, which slows what the
  explore trials teach.
* **D. Gradient surgery on the shared parameters.** Where two gradients
  conflict, project each onto the normal plane of the other (PCGrad, Yu et
  al., 2020); or step along the smallest-norm convex combination of the
  objectives' gradients, which no objective opposes to first order (MGDA,
  Désidéri, 2012; Sener and Koltun, 2018); or variants (CAGrad, Liu et al.,
  2021; Nash-MTL, Navon et al., 2022). This guarantees no first-order rise
  of any objective, but it needs one backward per objective, and on
  2026-09-20 the global reconstruction-priority projection was removed in
  favour of measuring. Two careful comparisons found that well-weighted
  plain sums usually match these methods (Kurin et al., 2022; Xin et al.,
  2022), which argues for A and B first.
* **E. Separate parameters where they conflict.** The answer's gradient
  reaches only parameters that reconstruction does not use, for example a
  small answer-specific map on top of the shared composition, or a part of
  the chooser. This was the 2026-09-20 solution in its strongest form. Step
  5a deliberately chose sharing: §3.13 of the 6.9 plan shows that choosing
  between the trials by the answer is not enough, while reaching the codes
  and the chooser is, so separation can only be partial.
* **F. Resolve a persistent conflict by refinement.** If two objectives
  pull one code or operator apart for many batches, the code is taken to
  conflate two things: it is split or a new row is minted, rather than
  pulled to a compromise. This fits the project's existing use of
  dissonance (the BOTH corner prompting refinement into parts; luminosity
  in the forgetting spec). It needs its own design.
* **G. Different speeds.** The answer's gradient into the shared parameters
  takes a smaller learning rate, or starts later, so reconstruction settles
  first and the answer adjusts within it. Simple, with no guarantee.

## 6. Proposal

*As first proposed; the design decided from Alec's answers is §8.*

**Stage 1, measure (no decision needed).** On XOR_grammar with step 5a,
and on `BasicModel_answers_tied_benchmark.xml` with supplied answers:

1. A reach matrix: for each objective, which parameter groups receive a
   nonzero gradient (perception, codes, chooser, operators and tied
   inverses, generate, reading map). It replaces inference with
   measurement in [GradientFlow](../GradientFlow.md).
2. Every weight each objective carries, in the trial and at batch end, in
   each configuration.
3. At one parameter state, before a sentence's first optimizer step, the
   per-group gradient norms and pairwise cosines of reconstruction,
   expectation and the answer, for both trials. The chooser and, where
   trainable, the codes are included.
4. A selection audit: how often the kept trial is worse than the other on
   reconstruction or on expectation, and by how much.
5. The outcome: each objective's own cost at the end of training with the
   answer term and without it (the cut), on the same configurations. For
   XOR_grammar this includes the reconstruction gate of step 6.
6. The usual magnitude of every term of every cost, in the trial and at
   the batch end, per configuration (needed for §8.2).

**Stage 2, cheap safeguards (decisions Q2 and Q3).** Weigh the trial's
answer term as the batch end weighs the answer, in comparable units (A);
and keep a trial only if it is no worse on any objective (C).

**Stage 3, if stage 1 shows that the answer costs reconstruction.**
Reconstruction becomes a constraint with a multiplier (B), the budget being
its cost without the answer term. Projection (D) stays out unless B fails.

**Later.** Refinement on persistent conflict (F), designed with the
forgetting spec's dissonance.

## 7. Questions for Alec

1. **Priority.** When reconstruction and a supplied answer conflict, does
   reconstruction take precedence, as a bound the answer may not push it
   past, with the answer learned within that bound? (Example: in
   XOR_grammar, the answer may reshape the codes only while every sentence
   still reads back as its own words, in either order.)
   **Alec:** "A mind has to see what is there before it can make good
   decisions about it."
2. **Weight.** Should the trial's answer term carry the same weight as the
   batch end gives the answer, in comparable units? (Today it is added at 1
   in output units, against `reconstructionScale` 0.1 for reconstruction in
   XOR_grammar.)
   **Alec:** "Terms in a single cost function should have approximately
   equal magnitudes before any differential weighting."
3. **The choice between trials.** Keep a trial only when it is no worse on
   any objective, instead of by the summed cost?
   **Alec:** "I'd say keep the trial if it's better overall (subject to 1)."
4. **Projection.** Does the 2026-09-20 rule stay, measure but never
   project, unless stage 1 shows a conflict the constraint cannot hold?
   **Alec:** "I don't know that as a rule."
5. **Order.** Stage 1 in item 6.9's receipt, and stages 2 and 3 as a new
   item before the operators update?
   **Alec:** "You can decide when to implement."

## 8. The design, from Alec's answers (2026-10-01)

### 8.1 Seeing comes first

Reconstruction, the model seeing what is there, takes precedence over a
supplied answer (answer 1). It is enforced in two places.

* **In the gradient.** In each trial of a sentence with a supplied answer,
  the answer's gradient on the parameters it shares with reconstruction is
  compared with reconstruction's gradient. Where they oppose (a negative
  inner product), the opposing component is removed from the answer's
  gradient before the step; reconstruction's own gradient is never
  altered. The reading map, which reconstruction does not use, receives the
  answer's gradient in full. To first order, then, a step taken for the
  answer cannot make reconstruction worse. This is a projection, used
  because it states the decided precedence; it is not a general correction
  of every disagreement (answer 4). Its price is one extra backward, paid
  only by sentences that come with an answer, and only in configurations
  whose trial reconstructs (`reconstructInLoop`).
* **In the choice between the trials.** The explore trial is kept when its
  total cost is lower ("better overall") and its reconstruction cost is not
  higher than the greedy trial's ("subject to 1", answer 3). Where a
  trial has no reconstruction term, as in XOR_grammar, the second
  condition always holds and the choice is by the total.

Where reconstruction is costed only at the batch end (XOR_grammar's
perceptual reconstruction), a trial cannot see its gradient. Stage 1's
comparison of each objective's cost with and without the answer term shows
whether that reconstruction suffers. If it does, a supervised trial gets
its own reconstruction term so the two rules above can protect it.

### 8.2 Equal magnitudes, then weights

Every cost that sums terms, each trial's cost and the batch-end cost,
divides each term by its own running scale before any weight is applied
(answer 2). The scale is a moving average of the term's detached
magnitude, kept per term and per cost, started at the first observation,
held above a small floor, and saved in checkpoints. Both trials of a
sentence use the scales as they stood before either step, so their
comparison stays equal. The weights then express deliberate priority only.

The present weights were set against raw magnitudes (for example
`reconstructionScale` 1.0 in `BasicModel.xml` and 0.1 in XOR_grammar,
while the trial's answer term has none). Stage 1 reports each term's usual
magnitude and what each weight is for. The weights after normalization
are proposed with the 6.85 hand-off for Alec's confirmation: 1 for every
term, unless a weight states a priority that §8.1 does not already carry.

### 8.3 What is not adopted

General gradient surgery among all objectives (PCGrad, MGDA and their
variants) is not adopted: the precedence of §8.1 is the one conflict
decided so far. It is reconsidered only if stage 1 or 6.85's receipt shows
persistent opposition that §8.1 and §8.2 do not resolve. Separate
parameters (E), refinement (F) and different speeds (G) stay as options.

### 8.4 Order

Decided by Claude, as Alec left it open (answer 5):

1. **Stage 1 in item 6.9's receipt.** It measures and changes nothing, and
   item 6.9 lands the answer's whole-path training that it is about.
2. **The suite trim**, part 4 of the October 1 hand-off, so that the many
   sweeps of 6.85 run faster.
3. **Item 6.85: §8.1 and §8.2**, with the weights set from stage 1, then
   the XOR table, XOR_grammar's ten runs, MM_grammar's ten runs and the
   reconstruction benchmarks, against 6.9's receipt.
4. **The operators update**, which changes the operators that every
   objective shares, under 6.85's cost function.

## References

Chen, Badrinarayanan, Lee, Rabinovich (2018), *GradNorm: Gradient
Normalization for Adaptive Loss Balancing in Deep Multitask Networks*,
ICML. Désidéri (2012), *Multiple-gradient descent algorithm (MGDA) for
multiobjective optimization*, Comptes Rendus Mathématique. Kendall, Gal,
Cipolla (2018), *Multi-Task Learning Using Uncertainty to Weigh Losses for
Scene Geometry and Semantics*, CVPR. Kurin, De Palma, Kostrikov, Whiteson,
Kumar (2022), *In Defense of the Unitary Scalarization for Deep Multi-Task
Learning*, NeurIPS. Liu, Liu, Jin, Stone, Liu (2021), *Conflict-Averse
Gradient Descent for Multi-task Learning*, NeurIPS. Navon, Shamsian,
Achituve, Maron, Kawaguchi, Chechik, Fetaya (2022), *Multi-Task Learning as
a Bargaining Game*, ICML. Platt, Barr (1988), *Constrained Differential
Optimization*, NeurIPS 1987 proceedings. Rezende, Viola (2018), *Taming
VAEs*, arXiv:1810.00597. Sener, Koltun (2018), *Multi-Task Learning as
Multi-Objective Optimization*, NeurIPS. Xin, Ghorbani, Gilmer, Garg, Firat
(2022), *Do Current Multi-Task Optimization Methods in Deep Learning Even
Help?*, NeurIPS. Yu, Kumar, Gupta, Levine, Hausman, Finn (2020), *Gradient
Surgery for Multi-Task Learning*, NeurIPS.

## 9. Stage 1 results, and a question on equal magnitudes (2026-10-01)

Measured on XOR_grammar, one unseeded run with step 5a and one with the
answer cut ([receipt](../benchmarks/2026-10-01-item6-9-review/objective-conflicts/README.md)).
The native benchmark's two runs stopped at the 8 GiB worker guard before
their first measurement, so nothing is known yet about the production path.

* **Outcome.** With step 5a the run ends at error .028, all four right;
  with the cut at .195, two right. Neither reads back.
* **Reach.** In a trial, expectation reaches perception, the codes, the
  chooser and its predictor; the supplied answer reaches the codes, the
  chooser and the reading map, and no longer perception (the separator is
  off the grammar's stack). Reconstruction is absent from XOR_grammar's
  trials.
* **Direction.** On the chooser expectation and the answer oppose: cosines
  −.03 and −.60 in the two trials of the first batch; on the codes −.12 and
  −.02.
* **Selection.** Of 1,600 comparisons, the kept trial was the worse for
  expectation in 297 (mean worsening .0032, largest .137).
* **Magnitude.** At the first batch the two costs are of a size
  (expectation .158, answer .250), yet on the codes the answer's gradient
  is 31 times smaller than expectation's (.0009 against .028), and on the
  chooser 1.6 to 3.8 times smaller. Over training expectation's cost falls
  to about .0005 and the answer's to about .028.

**The question this raises for §8.2.** Equal cost magnitudes do not give
equal pull on the shared parameters: the costs above are of a size while
their gradients on the codes differ thirty-fold. Dividing each cost by its
running size would, late in training, multiply expectation's gradient
about fifty times more than the answer's, so the answer would shape the
codes even less. "Approximately equal magnitudes" can therefore mean the
cost terms, as §8.2 has it, or the size of their gradients on the
parameters they share (the GradNorm reading of §5 A). Alec decides which.

## 10. Alec's answers to §9 and to the 6.9 review (2026-10-01, evening)

1. *Train read-back in XOR_grammar?* **Alec:** "I'm not sure what 'reconstruct
   in trial' means. We should be reconstructing input in all cases, in
   addition to doing output (if there is supervision). Typically there is
   no supervision, but expectation can always learn, and add a gradient
   (presumably it will be strongly influenced by previous LTM entries for
   that batch/document, or in-sentence data)."
2. *Equal magnitudes of what?* **Alec:** "If we want to weight output in
   relation to reconstruction, their relative errors need a similar norm.
   Not sure the best way to do that, I would look to existing literature,
   and make sure that the loss terms are described clearly in the doc (and
   probably implemented in the error class, which used to exist in
   Layers.py)."
3. *The production measurement.* **Alec:** "Measure at a smaller batch, or
   put it in run_slow."

### 10.1 Reconstruction in every configuration

There are two reconstructions today, and the question named the wrong one
plainly. *Perceptual* reconstruction re-renders the input from the per-word
percepts and never passes through the grammar; every configuration has it.
Reconstruction *through the understanding* reads each sentence back from
its end state through the grammar's inverse, so it trains compose to keep
what it read; only five configurations have it (`reconstructInLoop`:
`BasicModel.xml`, its three benchmarks, and `MM_grammar_wording`).
XOR_grammar has the first and not the second, which is why nothing trains
its grammar to read back. Alec's answer is read here as: every
configuration reconstructs its input through the understanding, the answer
is added when an answer is supplied, and expectation always learns. The
switch that turns reconstruction through the understanding off is retired
(no legacy paths); its cost in time is measured on the configurations that
gain it.

### 10.2 Relative errors of a similar norm, kept in one place

Each term enters a cost as a *relative error*: its error divided by the
error a trivial predictor would make on the same targets, computed from the
targets themselves and detached. For a squared error that is the fraction
of the target's variance left unexplained; for a cross-entropy over bytes
or categories it is the cross-entropy relative to the targets' own entropy;
for a presence term, its error relative to predicting the base rate. A
relative error is 1 for a predictor no better than trivial and 0 for a
perfect one, whatever the units, so reconstruction, expectation and the
answer become comparable before any weight is applied, and the weights
then state only priority.

The scale is taken from the data, not from the model's own running loss.
Dividing by the running loss (the log-loss balancing of loss-scale methods:
IMTL-L, Liu et al., 2021; DB-MTL, Lin et al., 2023; and in effect the
uncertainty weighting of Kendall et al., 2018) gives every term the same
relative rate of progress, which multiplies the gradient of a term the
model has nearly solved: measured in §9, it would raise expectation's pull
on the codes about fifty-fold late in training. A relative error against
the data keeps a nearly solved term small. Whether the terms' gradients on
the shared parameters are then of a similar size is measured (the stage-1
cosines and norms); balancing gradient norms directly (GradNorm, Chen et
al., 2018; IMTL-G) stays in reserve for persistent imbalance.

Every term is defined and summed in the `Error` registry of `Layers.py`
(it exists, with `add`, `total`, `breakdown` and a covariance of terms
across batches, but today it only records: `runBatch` assembles the trained
sum itself). In item 6.85 the registry becomes the one place where each
term is named, its baseline and relative error computed, its weight
applied, and the trial and batch-end totals summed; the trained totals are
read from it. [GradientFlow](../GradientFlow.md) and
[Training](../Training.md) describe every term: what it compares, its
baseline, its weight, where it is costed and what its gradient reaches.

### 10.2a The baseline, made exact (Claude, 2026-10-01, answering Codex)

Codex found two cases where "the error of a trivial predictor of the
term's own targets" has no positive value: a term whose targets do not
vary in a batch (the idea/relation kind of `InterSentenceLayer._kind_loss`,
when every sentence is an idea, as in XOR_grammar), and a structural
penalty with no targets at all (the proximal L1 on the concept readout in
the production configuration; definition sparsity when enabled).

* **The trivial predictor is the uninformed one.** It knows nothing of the
  targets except their scale: uniform over the outcomes for a categorical
  or binary term (zero logits), so the baseline is log K, log 2 for a kind
  or presence term and log 256 for a byte; and the origin for a squared
  error, so the baseline is the mean squared norm of the targets over the
  term's active entries, detached. A predictor fitted to the batch's own
  rate or mean is not trivial: it has seen the targets it is scored on,
  which is why it can be perfect.
* **One ratio per term:** the term's mean error over its active entries
  divided by the baseline's mean over the same entries (a ratio of means,
  not a mean of ratios), so a row with small targets cannot dominate.
* **Penalties are not errors.** A term with no targets (readout L1,
  definition sparsity, any regularizer) is not normalized: it keeps its own
  coefficient as a regularization strength, is recorded in the `Error`
  registry under its own category, and is reported apart from the sum of
  relative errors. The proximal L1 stays an optimizer step. A squared-error
  term whose targets are all exactly zero is a pull toward zero, and is
  treated the same way.

With these, every relative error is 1 for a predictor that knows nothing
and 0 for a perfect one, no term's divisor can vanish, and no floor or
running scale is needed.

*Why a nearly solved term stays small (Alec's concern, 2026-10-01: that
parity across domains "will make noise significant even for an almost
solved problem").* The divisor is fixed by the data: it does not shrink as
the model learns. A term the model has nearly solved has a relative error
near 0 and its noise is scaled by the same fixed factor as before, so it
stays as small as it is. The amplification Alec describes happens when the
divisor tracks progress, either the model's own running loss or a baseline
fitted to the batch, which falls to zero when the targets do not vary;
those are the two choices this rule rejects. The rule puts the terms on one
scale, so that a weight means the same thing for each of them; it does not
make the errors equal. Balancing their gradients to equal size would bring
the same amplification back, which is one more reason it stays in reserve.

**Reconstruction in every configuration, scoped.** The prototype failed on
the old reading paths (missing BPE surface data), on grammar-free serial
paths (no concluded state to read back from), and on sentences with no
admitted surface candidates. Reconstruction through the understanding
applies to every configuration on the reading paths that stay (meronomy,
in both bindings). The old reading modes are not ported: their
configurations are listed as exempt until item 12 of the trim moves them
to meronomy, right after 6.9 closes. A configuration with no grammar has
no understanding to read back through; it keeps perceptual reconstruction
and is listed. A sentence with no admitted surface candidate contributes
no reconstruction term; it is counted and reported, not treated as an
error.

### 10.3 The production measurement

Alec, on the choice between a smaller batch and `RUN_SLOW`: "If it's under
the slow guard, you might as well make it production-sized." The native
measurement becomes a `RUN_SLOW` test at the benchmark's own batch. The
slow run gives it a per-worker memory ceiling that fits (it needed about
9 GiB against the sweep's 8 GiB), stated in the slow-run target and in the
receipt; the sweep's guard is unchanged. Its first run is stage 1's
production measurement, and the weekly slow run repeats it.

### 10.4 Order

*Superseded by [6.9 plan §12](../plans/2026-09-29-item-6-9-xor-grammar.md#12-remaining-steps-2026-10-01): the cost function is item 6.9's next work, not a separate item.*

No separate worktrees (Alec). In the one working tree, in order:

1. **Stage 1, continued (measurement only).** The native arms as the
   production-sized `RUN_SLOW` test of §10.3; and XOR_grammar with reconstruction through the
   understanding (a receipt-local copy of the fixture), with and without
   the trial answer, ten runs of each gate in the answer arm: the first
   direct measurement of reconstruction and the answer in one trial cost.
2. **The suite trim**, items 1 to 10 of the October 1 hand-off. Items 11
   (venv) and 12 (old reading modes) wait until 6.9 closes, since both move
   its baseline.
3. **Item 6.85**: reconstruction through the understanding everywhere
   (§10.1), relative errors in the `Error` registry (§10.2), the precedence
   and selection rules of §8.1, and the documentation of every term.
4. **Close item 6.9** under that cost function: both gates, the control,
   the XOR table and MM_grammar's runs.

### References added

Groenendijk, Karaoglu, Gevers, Mensink (2021), *Multi-Loss Weighting with
Coefficient of Variations*, WACV. Lin, Jiang, Ye, Zhang, Chen, Chen, Liu,
Kwok (2023), *Dual-Balancing for Multi-Task Learning*, arXiv:2308.12029.
Liu, Li, Kuang, Xue, Chen, Yang, Liao, Zhang (2021), *Towards Impartial
Multi-task Learning*, ICLR. Liu, Johns, Davison (2019), *End-to-End
Multi-Task Learning with Attention* (dynamic weight average), CVPR.
