# Gradient flow across the architecture

This document states which objective trains which parameters: the owner and
side of every registered cost, the places where a gradient stops, how the
terms are normalized and added, and how hard choices are credited. How
training runs (corpora, lessons, the loop) is in [Training](Training.md).

## Current gradient boundaries and loss normalization

Gradient ownership and the numerical definition of a loss are separate choices.
Alec's direction is to avoid balancing terms where possible. No term is
balanced adaptively by running losses or gradient norms; each relative term is
normalized by its own detached uninformed baseline (below), which fixes the
units in which terms add. Whether that normalization suits a loss is judged
for each loss; the choice of owners neither requires nor removes it.

| Parameters | `separateChooser=false` (default) | `separateChooser=true` |
|---|---|---|
| Understanding parameters: codes, grammar operators and their inverses, conceptual definitions, the compose chooser | Reached by reconstruction R and by the answer side A; A loses only its component that opposes R in each unit | Written only by R-owned costs; the owner-restricted backward excludes supplied-answer gradients |
| Expectation predictors, including `ActivationPredictor` | Their own local objectives; sources, targets and consumed forecasts are detached | Same |
| Candidate attention | Its reading lesson (R) and departure credit (A); consumers detach its output | Its own owner, which also receives the sentence's R and A through the admission read |
| Intuit's learned weights | Output only; no reconstruction or expectation gradient | Same |

`separateChooser` defaults to `false` (`data/model.xml`): each sentence takes
the gradients $g_R$ of R and $g_A$ of A over every optimizer parameter,
removes from $g_A$ only its component that opposes $g_R$ in each unit, and
makes one optimizer step with $g_R + g_A$
([projected gradients](#default-projected-gradients)).
`separateChooser=true` is a supported option, not a legacy path: each cost
then trains only the parameters of its declared owner, whatever its side
([owners and sides](#parameter-owners-and-sides)). `reconstructionJitter`
(default 0) draws one strength $s \sim U[1-j, 1]$ per sentence and scales the
reconstruction owner's R term alone, in either mode. Thought and generate
chooser features remain detached. A separate output residual band was
considered and not adopted; the conceptual residual split remains
[future work](FutureWork.md#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29).

Both settings compare completed paths by detached `R + A` in
`SentenceCredit.comparison`: each stream keeps its strictly cheapest path, a
tie keeps the greedy one, and the difference supplies chooser credit
([whole paths](#whole-path-credit-and-concept-episodes-2026-11-11)).
Expectation's error enters neither the keep nor the credit: a departure judged
by its own surprise would teach the chooser to make the input predictable
rather than to explain or answer it. Even with disjoint parameter owners,
rescaling a cost can change which path is kept. This makes the units of those
costs consequential without establishing a need for automatic balancing or any
particular normalization formula.

### The normalization contract and the generation exception

The general rule (2026-10-01) is implemented by `Layers.Error.error`: register raw errors and a **detached**
uninformed baseline, sum contributions over the same active entries of a named
term, then divide once. For squared error this is
`sum((prediction - target)^2) / stop_gradient(sum(target^2))` when the baseline
is positive. Categorical/binary baselines are `log K`/`log 2`. An exactly zero
squared baseline gives an ordinary squared penalty; an empty mask gives zero.
The uninformed baseline is the error of a predictor that knows nothing of the
targets but their scale: the origin for a squared error, zero logits for a
categorical or binary one. A relative error is therefore 1 for that predictor
and 0 for a perfect one. The divisor is fixed by the targets and does not
shrink as the model learns, so a nearly solved term stays small; a divisor
that tracked progress (a running loss, or a baseline fitted to the batch's
own mean or rate, which vanishes when the targets do not vary) would amplify
its noise.
The purpose is comparable loss units before configured priorities, not equal
gradient norms or forcing nearly solved losses back to equal magnitude.

The generation residual is the exception to a target baseline.
`Generative.absolute_residual` measures per-lane mean squared error in the
walk's own chart: cube codes in their coordinates, numeric walks after `asinh`
on both values. The walk sums the chosen steps and the unresolved frontier, and
the sentence registers form, for and against separately through `Error.error`,
each with baseline one and the active-stream mask
([the generation residual](#the-generation-residual)).

**Numerical failures are repaired at their producer.** A non-finite value is a
producer's bug and is repaired where it is made. A choice's menu is an
observation: it is built without autograd, and only the selected inverse is
differentiated as an operand, per stream; a losing trial or a rejected proposal
contributes no edge, so a rejected operator receives no gradient and keeps its
momentum. A quotient whose derivative can overflow, such as the boundary
solution of the mean inverse in `MeaningCodes.recover_lanes`, keeps its exact
forward and bounds each local partial at the `_BoundedAtanh` slope,
$1/(1 - 0.9^2) \approx 5.26$, before the arriving cotangent is applied
(`_BoundedInverseRatio`); a non-finite cotangent still propagates. No
`nan_to_num`, clamp, floor, detached denominator or gate hides a value. Before
every optimizer step `preflight_finite_gradients` (`bin/Optimizer.py`) checks
every optimizer-owned gradient, coalescing sparse ones, and raises
`NonFiniteGradients`, which names every affected parameter and carries the
per-objective gradient vectors already collected. `MODEL_FINITE_GRAD_GUARD` is
`auto` (on unless a parameter is on CUDA), `on` or `off`.

## Parameter owners and sides

**One forward, one update.** A sentence runs forward once per path. Every
reader sees live values: output reads the understanding in STM, attention
reads perception, the decoder reads the completed understanding. The graph is
differentiable end to end. Every registered cost has an owner and a side, R or
A, which `Error.add` and `Error.error` record separately. By default the side
decides: the R and A gradients are taken over all optimizer parameters and A
is projected against R. With `separateChooser=true` the owner decides:
`ObjectiveOwnership.backward_owned` calls `torch.autograd.grad` once per owner,
only with respect to that owner's parameters, so an error can be
differentiable through a parameter without training it, and an audit fails if
any parameter receives a second writer (Alec, 2026-10-02: "the gradients
separate over the weights"). Either way each sentence makes one optimizer step.

**Who owns what** (`Models.objective_parameter_groups`):

| Owner | Parameters | Trained by |
|---|---|---|
| reconstruction | perception codebooks, conceptual definitions, the compose and generate operators, the compose chooser, and every parameter no other owner claims | the input and order-zero reconstruction losses and compose lessons; the chooser rewards (owner `chooser`) use the same parameter set |
| expectation | the conceptual components (`IndependentComponents`, including the bind resolver `reference_close`), the shared `ActivationPredictor` (`_inter_predictor`) and word predictor, the concept predictor and the percept predictor (`PerceptExpectation`: context and moments update at each arrival, and its parameters receive the accumulated arrival error once, in the sentence's update) | expectation terms whose targets and sources are detached, so a predictor never trains what it predicts; also `expectation.bind_score_function` |
| output | output space, answer attention and record reader, `intuit`'s weights and the generate policy; named output concepts also own their sparse coefficients | supplied-answer costs; the generate policy also takes the generate lessons (owner `generate_lesson`) |
| attention | the candidate attention | `attention.reading_lesson` and `attention.candidate_departure`, plus, in the separate backward, the sentence's R and A through the admission read (`AttentionCredit`, a cotangent to the candidate attention only) |

**Sides.** Compose and thought rewards are
`output.compose_score_function` and `output.thought_score_function`, owner
chooser, side A; the chooser's parameter set is the reconstruction set. The
output walk's departure credit is owner chooser, side A; its residual lanes
are side A and, since their objective is reconstruction, owner
reconstruction. The reconstruction walk keeps its residual and departure
credit under owner reconstruction, side R (Alec, 2026-11-10). Bind and
candidate-departure rewards are expectation/A and attention/A.

**Where reconstruction is a loss and where it is an audit.** The input's
live parallel field trains `reconstruction_reverse.what/where/when` through
basic percepts and the common native decoder. The sentence's untransformed end
slots traverse the generate grammar, the occurrence writer and the same reverse
to train `reconstruction_order_zero.what/where/when` against the same detached
native input. Generated word order and brackets come from the writer's cursor. The decoder's byte identity and
coverage are value-only audits; the compose-forest inverse remains a System 2
report and tool. `reconstruction.free_percept` keeps its forecast residual,
with context read from the parallel field rather than the sentence's end slots.

**Expectation and reconstruction (decided 2026-10-29).** Percept
expectation predicts each arrival from prior context; its context and moments
update at that arrival, and its predictor's parameters receive the accumulated
local arrival error once, in the sentence's update (`PerceptExpectation`). The
forward path carries the full signal, `expected + residual`; the residual is
signed, so an expected absence is negative. The inverse side is split by
residual: the completed understanding's inverse learns only
`observed - stop_gradient(expected)`, using the value frozen at arrival, so no
top-down gradient reaches expectation and expectation's loss reaches nothing
else. Where an expectation gain is configured, the gain multiplies the signal
detached (`Expectation.gain_field`), so a predictor cannot learn to mispredict
in order to brighten a signal or to over-predict in order to silence it. At
the conceptual layer the expectation is `ActivationPredictor`, whose predicted
next state is also the spread that primes attention, the chooser and
retrieval, trained only by its own error against the detached next state;
`intuit` is the top-down component, a Sigma over Pi with its own weights
trained only by output. The conceptual split and the correction-output form
are [future work](FutureWork.md#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29).

**How output reaches the understanding (decided 2026-10-29; projected since
2026-11-09).** Attention and expectation are each isolated by a stop-gradient
at the place where their output is consumed. Reconstruction and output are
not isolated from each other by default: both write the understanding's
parameters, because a strongly partitioned syntax and semantics is doubtful
as a psychological model (Alec, 2026-10-29). The choosers' log-probabilities
enter the same objective as the costs, so the sentence's one update carries
both the pathwise gradient through the values the choices produced and the
score-function gradient of the choices themselves
([the chooser's credit](#perceptions-codes-and-the-choosers-credit)). On a
shared unit, output keeps every component of its gradient that does not
oppose reconstruction ([projected gradients](#default-projected-gradients)).
`reconstructionJitter` is the earlier means to the same end and stays
available at its default of 0. It scales reconstruction's strength and never
its target: a zero-mean perturbation of a target leaves the expected gradient
unchanged, while a weaker reconstruction term lets output move the shared
parameters.

## Whole-path credit and concept episodes (2026-11-11)

A sentence is costed over whole paths: one greedy path and `explorePaths`
explore paths, each departing once from the greedy one. Each path runs
reading, the reconstruction generate walk, the occurrence writer and reverse,
thought, and then the answer generate walk and reverse; all paths are costed
at the same parameters before the sentence's one update. A path's comparison
cost is R + A, each stream keeps its strictly cheapest path, and a tie keeps
the greedy one. On data that supervises output, a path's A is its answer cost
on every row: the supplied answer where one was asked and silence otherwise,
so a path that speaks where nothing was asked pays for it, and silence takes
part in which path is kept. The direct field reconstruction remains
grammar-free. The numeric modality uses the path's composed state through one
affine head. The path and its two costs are drawn in
[Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11).

A single departure is drawn over attention, compose, candidate, bind, reconstruction, thought or output. Every departure receives its whole-path change in R + A with the proposal correction of that draw, reading its own path's compose departure mask and attention state. Every ordinary cost trained on every path is scaled by 2/(1+K): reconstruction, expectation, compose and generate lessons, attention and their regularizers. The answer trains on the union of the kept path, explore paths departing in compose, and the greedy path when that set includes a compose departure; weights sum to one per stream. Both compositions train the reader because a reader trained only on what is kept reads the kept composition better because it is kept, which biases the comparison toward the incumbent; with one reader the bias returns unless the reader sees both. A departure in attention, among the candidates or in a bind is another reading of the input, to which the supplied answer does not apply, and a path that departed in the thought or a read-out is not asked to answer from a conclusion it did not keep. The rule's known cost is that the rejected composition's share does not decay as the chooser commits. Answer costs and `penalty.output` already carry those reader weights; chooser and `.A` departure costs are booked once and receive no additional all-path multiplier. The exploit's one reading lesson is added once outside the path sum. A no-departure path has an independent registry container, so subsequent bookings cannot leak between paths. The reconstruction/output walk residuals and credits retain their declared owners and R/A sides. Default gradient projection and optional separate owner pullbacks each make one sentence update. There are no `thought_comparison` terms or separately trained comparison-reader parameters.

Hard derivations train only what they select: a path's costs reach the operators it applied, and an operator that no path selects receives no gradient beyond the negligible derivative through its candidate score. Its route to credit is a departure, an explore path that selects it, trains it with that path's costs and is compared with the exploit on R + A. The proposal's uniform floor (`explorationUniformFraction`) gives every eligible alternative a positive probability at each departure, and `explorePaths` sets how many explore paths a sentence draws; the two set how quickly an unused operator is tried.

New question-conditioner widths join the optimizer before the same sentence's first backward. Live thought results and output conditioning retain the selected path's graph; only choice evidence and durable storage use their existing detach boundaries. Batch-end reporting uses the recorded output without another walk or loss. On data without supervised outputs, paths compare on R and only the kept reconstruction runs a thought.

Omission's original contents are detached loss-side targets only, captured before identity admission. `reconstruction_order_zero` restores the original input; `reconstruction_reverse` scores visible positions. `reconstruction.free_percept` excludes omitted brackets from both its numerator and its normalization. Loss-only codes for unseen target bytes neither admit percepts nor consume the forward random stream. Expectation excludes omitted observations that would otherwise leak content into state. Rates and accuracy averages are checkpointed statistics, with no gradient. Episodic concept entries are also detached: only their references and signed values are stored, while fields and semantic truth rows retain both poles. Time-only recall rebuilds the poles from the signs, lays out references in reading order and publishes their field directly, and direct `reverse` can read it without a reconstruction charge.

## Default projected gradients

The default (`separateChooser=false`) takes two `autograd.grad` traversals
over all optimizer parameters, one for each side, keeps the reconstruction
gradient whole, removes from the answer gradient only its opposing component,
and makes one optimizer step. The separate-owner path remains available with
`separateChooser=true`.

| Side | Owner in the optional separate path | Costs |
|---|---|---|
| R | reconstruction | Input and order-zero reconstruction (`reconstruction_reverse`, `reconstruction_order_zero`), `reconstruction.part`, `reconstruction.free_percept`, the reconstruction walk's residual and departure credit, memory link coverage and state errors, definition folds. |
| R | compose_lesson | Compose lessons. |
| R | expectation | Predictor terms. |
| R | attention | Reading lessons and candidate cross-entropy. |
| R | Each parameter's owner | Regularizers, once in the projected aggregate. |
| A | output | Answers, prediction and readers. |
| A | generate_lesson | Generate lessons. |
| A | reconstruction | The output walk's residual lanes (their objective is reconstruction), whole. |
| A | chooser | Output-walk credit, compose rewards and reconstruction-category thought rewards, whole. |
| A | expectation | Bind and expectation-category thought rewards, whole. |
| A | attention | Candidate-departure rewards, whole. |

`Error.add` and `Error.error` retain owner and side separately. Registry keys use
the owner alone for its default side, or append `.A`/`.R` for the other side.
The separate path strips that suffix and routes to the owner's parameters,
including its existing perception pullback. The default reads the sides of the
same costs. Both default traversals retain the batch graph; sentence-boundary
cuts release completed sentence writes without cutting batch perception.

Only the `reconstruction` key's R aggregate is multiplied by the drawn
`reconstructionJitter` strength; its default remains zero jitter. A shared
regularizer enters R once. Unit projection uses a tensor, concept row or durable
definition edge; Adam coordinates have weight `1/(sqrt(v_hat)+eps)` from existing
state, with identity before the first step. A single-writer unit is unchanged.
Sparse row gradients retain sparse support, without densifying the concept table.

For each unit $u$ whose two gradients oppose, $\langle g_A^u, g_R^u\rangle_M < 0$,
the answer's gradient becomes
$g_A^u - \frac{\langle g_A^u, g_R^u\rangle_M}{\lVert g_R^u\rVert_M^2}\, g_R^u$,
with inner products in the unit's optimizer metric $M$. Reconstruction's
gradient is never altered, and output keeps every component that does not
oppose it: that is output's slack. No weight, tolerance or balance of
amplitudes enters. Geometry alone fixes the removed component, which does not
depend on the scale of $g_R$, so jittering reconstruction's strength would only
reweight the final sum; the projection replaces the jitter as the way output
shapes shared parameters. Concept tables are projected per concept row and
definition coefficients per edge: over a whole table, agreement on one concept
would offset a conflict on another, and the added multiple of $g_R$ would
enlarge the step on rows output never touched. The guarantee is first order:
output's gradient adds no first-order increase of the reconstruction loss.
Reconstruction-owned parameters step by momentum descent, so it holds for the
gradient and not exactly for the step.

Why this conflict, and only this one: reconstruction, the model seeing what is
there, takes precedence over a supplied answer. Removing only the answer's
opposing component in each shared unit states that precedence directly, so to
first order a step taken for the answer cannot make reconstruction worse,
while parameters reconstruction does not use receive the answer's gradient in
full. General multi-objective gradient surgery (PCGrad, MGDA and their
variants) is not adopted: well-weighted plain sums usually match it, and this
precedence is the one conflict decided. If persistent opposition appears, the
options are separate owners (`separateChooser=true`), refinement of a
persistently conflicted code into two, and different learning speeds.

`SentenceTraining.gradient_receipt` adds per-unit epoch averages
`cosine_before`, `fraction_projected` and `fraction_answer_removed`, with a step
count. The cosine and removed norm use the unit's optimizer metric; zero-vector
cosines and removed fractions are reported as zero. `realized_steps` reports
root-sum-squared actual displacements divided by root-sum-squared pre-step
parameter norms for codes, lift and adverb, with null for a zero denominator.
The existing read-only `measure()` dissonance and understanding norms are retained.

The decoder and answer generate credits use their own complete paired costs,
including residuals, and the same proposal-corrected score function as compose.
Exact ties do not create a gradient. No walk, verb or modal teacher remains.
A stored MTM value receives only the current action's local straight-through
pullback from its recorded forward input; earlier sentence graphs stay cut.

## Perception's codes and the chooser's credit

- **Perception's codes have one writer, perception's reconstruction.** In the
  conceptual derivation the percept prototypes and their evidence are
  detached; the sentence path's pair search, byte scorer and answer head
  send no gradient to them, and no code moves under the sentence path. (A
  straight-through blend through the pair search once pulled every candidate
  code onto every read-back target and merged the codes.)
- **Attention's credit is detached at the sentence handoff.** The
  input-attention walk shares the compose chooser's parameters. Its
  straight-through value, the cached perception and the perception pullback
  would otherwise give those parameters a second, pathwise writer beside the
  score-function term, even when the compared costs tie. The handoff is
  therefore detached, and an attention choice learns by the score-function
  term of its own departure.
- **A chooser's gradient is the score-function estimator (decided
  2026-10-05).** The chooser's
  parameters enter the path only through which operation is chosen, so the
  exact gradient of the expected cost with respect to them is the
  score-function term at the sampled departure, with the exploit path's cost
  as the paired baseline (self-critical sequence training): the surrogate is
  $p_\theta(a_{dep} \mid s)\,(C_{explore} - C_{exploit}) / q(a_{dep})$, with the
  costs and the proposal probability $q$ detached, one term per departure;
  $\nabla p / q$ is the importance-weighted $\nabla \log p$. The departure is
  drawn walk first, then round, then alternative: `W` counts walks with an
  eligible round and `R_walk` eligible rounds in the drawn walk, and the
  alternative comes from a proposal that mixes a uniform floor over the
  value-distinct eligible alternatives (`explorationUniformFraction`) with the
  tempered policy, so $1/q = W \cdot R_{walk} / q_{alt}$, which is
  $W \cdot R_{walk} \cdot K$ when the draw is uniform over $K$ alternatives.
  No differentiable decoder is needed; a tie teaches nothing and the greedy
  argmax supplies no chooser gradient. The choice itself is a hard argmax
  with no straight-through estimator. Compose, attention, candidate, bind,
  thought and both generate walks are all credited this way.
- **Why reconstruction alone cannot judge a choice.** Operators with exact
  inverses, such as conjunction and disjunction over the same two words,
  recover the same words, so their reconstruction costs tie. What the choice
  asserts shows only in the answer and in what follows, which is why paths
  are compared on R + A rather than on reconstruction alone.
- **No soft relaxation.** DARTS-style superposition, Gumbel-softmax sampling
  and mixture-of-experts routing all need the loss to be smooth in the mixed
  output downstream of the choice. Here it is not: the hard pair pick makes
  the reconstruction cost piecewise constant in the root, so a router
  gradient through a mixture is zero almost everywhere, and the mixture
  gradient that favours whatever already dominates (the documented DARTS
  collapse) is the mechanism that once merged codes through a
  straight-through blend. The paths therefore serve three purposes besides
  the estimator: coverage independent of the chooser's confidence (the
  proposal's uniform floor), the commit (the understanding kept is the better
  of actual derivations) and an exact zeroth-order cost comparison. A dense
  pathwise signal, if ever wanted, enters as a control variate (REBAR,
  RELAX), not as a second sum.
- **The pair search is hard.** Candidate codes detached, the residual
  absolute in the code's own chart, and no straight-through blend: with codes
  perception's and operators parameterless, nothing on the compose side
  needs a gradient through the inverse, and the blend's gradient was a biased
  proxy (the true pair's margin) weighted by the decoder's uncertainty. The
  byte scorer's bank codes are detached. Selected numerical inverses retain
  operand gradients, but the free walk supplies no policy gradient.
- **The decomposition chooser is fixed.** `DecompositionChooser` has the
  constant weights `[1, 0, 0, 0, 0]`, outside every optimizer group, and
  selects the joint residual argmin over form and both meaning lanes; no
  teacher trains it. `LanguageSpace._bounded_binary_reconstruction` scores
  each shortlisted dictionary pair with `Generative.absolute_residual` on its
  fold and its observed parent, in the code's own coordinates: the selected
  feature is the negative of the sum of the form MSE and the mean of the two
  meaning MSEs, so a parent whose evidence tends to zero still gives a
  continuous pair score.
  [Producer certificate](../test/test_reconstruction_magnitudes.py)
- **One affine reader for both gates.** XOR_grammar's class reader and the
  sum control read the raw root through the same output-owned affine head;
  the root's address bands are zero. A normalization of the root is a
  nonlinearity the control cannot hold and is not the affine read.
- **Certainty is the activation.** Forms are at full presence; the leaf's
  activation, the projection coefficient, carries how sure the reader is.

## One understanding record, two consumers

[SentenceUnderstanding](../bin/SentenceUnderstanding.py) is built once per
path. The tied reconstruction and the supplied answer receive that same
object. It carries the concluded root, three end slots and depth, packed
sentence roots/depths, per-word target values/rows/validity, sentence id,
and a primed-symbol snapshot. No rule IDs, arities, operand positions or
journal columns enter the record or the answer reader. It carries no
witness offsets or numerical operation frames. Temporary clause-closing
frames preserve the performed relation semantics and are discarded at the
closing; neither reconstruction nor the answer reads them. Addresses select evidence;
they are not semantic magnitudes.

The snapshot contains each row's own sentence symbols plus up to
`reconstructionBasisLimit` others whose relevance weight is above neutral. It
is taken after the sentence's seen write and shared by every path.
`Space.prime_seen` decays the per-stream `[B,V]` seen ledger and bumps the
seen rows; the relevance weight it returns is one plus the signed desire
channel and, where the parallel path supplies it, the detached predicted
concept activation of `ActivationPredictor`, so the prediction is the spread.
No diffusion over definition edges or occurrence rows remains, and no stream
primes another. Symbols with surfaces are byte candidates; all valid rows
enter answer context with their priming weights. Signed inhibition below
neutral is preserved.

The numeric head adds an answer-owned, zero-initialized linear reader of the
root (D), three masked end slots (3D) and the echoic priming-weighted code sum
(D, read detached); its extra width is 5D. There are no rule histograms or
original per-word features. The linear reader preserves additive evidence for
the sum-only control. With `readerBlockNormalization=true`, each form and
meaning block is divided by the maximum norm of that block in the shared code
snapshot, clamped below by one. The echoic bank uses its own shared block
scales. These scales are common to all paths' roots and streams; a composed
root is never normalized by its own norm. Thus the reader remains affine and
the sum control remains additive. This scaling is a reader operation and never
changes composition or stored meanings.

The learned `PrimedSymbolReader` is excluded from `_forward_head` for every
configuration. Its scorer and consume gate remain available to
`reverseOutput` and to the row-owned STM/LTM reads of selected thought.
Generation consumes an owned operand without changing the reconstruction
carrier. Thought preserves its metadata and charges its shared work meter.
Keys remain detached. No numeric gate can learn a nonlinear bag-of-words
answer through this reader.

## State paths and shared parameters

`resolveAnswer()` prepares an owned `AnswerDerivation`; `reverseOutput()`
consumes it without repeating resolution. By default live thought results and
output conditioning retain the selected path's graph; under
`separateChooser=true` numeric and generated answers stop at the concluded
state, the contextual operands and the record features. Generation keeps the
declared numerical hosts and their tied interfaces, without a second
optimizer owner or checkpoint copy. Grammar lessons cannot move those hosts.

Reconstruction and output share one **order-zero decoder**. Its two
reconstruction chains are:

```text
System 1: perception -> live a_0 -> reverse (basic percepts -> native parts) -> reconstruction_reverse
System 2: untransformed sentence end slots -> generate -> occurrence writer -> the same reverse -> reconstruction_order_zero
```

Both losses use the detached per-byte native input as target. Both compare the
full event extent, padding the shorter sequence with zero events; missing and
extra events remain charged. The second enters generate; the common `reverse`
itself remains grammar-free and reads no compose forest. Reconstruction scores
the completed sentence, never word by word: a per-word recovery cannot show
that later folds and the sentence's ending retained the earlier input, so the
ending is inside the fidelity check, and input targets only score, never
seed. Configurations without the serial sentence path (`word_brackets=False`)
book only `reconstruction_reverse`. `reverse(x, *, field)` requires a field
on aligned configurations and reads no input perception spans. Its sparse
native activity holds one selected row, strength and continuous percept vector
per covered position. The native spelling buffers are allocated at construction;
vectorized decoding never allocates a batch-by-position-by-inventory slab.

**Episodic entries and time bands (2026-11-11).** An episodic entry is
selected under `torch.no_grad` and stored detached (`EpisodicMemory.select`,
`write`), and a field reinstated by a recall by time books no loss term of its
own: the percept side of long-term memory is System 1's content and trains
nothing. Every learned reader takes a `.when` as an age or a difference
(`RowTime.age_interval`, `age_band`, `relative_idea`, `relative_to`,
`difference`), so no learned read depends on the clock's phase. See
[Architecture](Architecture.md#episodic-and-semantic-memory-in-the-two-systems-2026-11-11).

Output is `generate -> occurrence writer -> reverse`. Generate infers binary
undo, unary undo or STOP over its bounded bank. The existing legal-action and
recomposition tests still determine eligibility. Its emitted terminals select
the bank's lexical row by unprimed `readback_scores`, using priming only to
break exact identity ties; their independent for/against
activations and new output brackets become the field. Input spans and desired
answers do not choose these rows. The output comparison gives the departure
its detached cost advantage through score-function credit, owner output/A.
Its three residual lanes are side A (owner reconstruction, by their objective). The reconstruction walk uses
the same generate, writer and reverse implementation, and compares its native
reconstruction loss plus its selected residual. Its `reconstruction.walk_residual`
lanes and `reconstruction.walk_credit` belong to reconstruction/R. No
compose-tree teacher trains the generate policy. Shared numerical inverse operators retain their ownership;
`separateChooser=true` still restricts parameter writes by owner.

Thought deposits carry a differentiable order-zero reading, so this same
native reverse can credit selected intuition weights. Expectation reads a
detached order-zero target after the kept thought episode, on serial and
parallel paths. Numeric heads retain their forward readout with
`answerSynthesis=false`.

Compose, generate and reverse share each kernel; a free inverse searches
through that same compose kernel. These operations impose no code norm
constraint. Target spellings, priming weights and selection addresses are
detached. Durable LTM writes, finished episodes and retained expectations
detach; they cannot reopen an old graph. Query masks, reference matching and
work accounting remain hard bookkeeping.

**Optimizers.** Reconstruction uses gradient descent with momentum 0.9 at the
configured learning rate. Its compact row-local form retains a float32
momentum prefix, updates only observed nonzero-gradient rows (or rows with an
active proximal penalty), and leaves unobserved rows unchanged. It does not
divide a step by gradient magnitude. Readers and expectation predictors keep
Adam. The L1 proximal threshold for a reconstruction-owned coefficient is
`lr * strength`; answer-owned adaptive coefficients keep their Adam metric.
Within reconstruction's descent the lift action learns at `actionRateRatio`
times the configured rate and the adverb action at its square (default 1).
Reconstruction steps by momentum descent because of indifference: once it is
satisfied it is indifferent among many derivations and code placements, and
its gradient to the chooser and the codes becomes tiny and noisy. Adam divides
each gradient by its running magnitude, so such a gradient would still move a
weight about one learning rate per step in a random direction, and the
derivation and the roots would drift under the reader; a step proportional to
the gradient leaves a weight whose objective is at its floor where it is.

## Trained total and diagnostic objectives

`Layers.Error` owns the trained sum and applies
[the normalization contract](#the-normalization-contract-and-the-generation-exception)
to every relative term.

Every relative term has weight 1 unless the configuration supplies a
priority, including inherited, explicitly stated `model.xml` priorities.
`reconstructionScale` multiplies reconstruction terms independently.
`Error.total(kind='relative')` and `Error.total(kind='penalty')` report the
two sums separately; `total()` supplies their trained sum. A row registry
normalizes with the shared target sum, then returns row contributions whose
active-row mean is that same training objective. A small-target row does
not get a separate large inverse scale. `breakdown()` records raw error,
baseline, entries, weight, contextual multiplier, kind, ownership and row
selection values. `trained=False` metrics never enter the optimizer total.

All retained serial grammar readings reconstruct through their understanding.
A reading without a grammar keeps perceptual reconstruction. A sentence with no
admitted surface candidate has no reconstruction term and is counted.
A zero-weight term does not require its source registry.

The inventory below gives each active term's definition. "Origin" means the
detached target mean square over exactly the error's active entries. R is the
input and order-zero reconstruction; A is the answer side; E is the
expectation predictor. Under `separateChooser=true` every A term stops at the
understanding. With it false (the default), the answer reads the understanding
live and shares the trunk with R, less its component that opposes R in each
unit. Every E term detaches both its source and target. Lesson inputs are
detached, so grammar lessons train only their choosers and write no operator;
under `separateChooser=true` the restricted backward also prevents it. Owner
and R/A side are independent. In the last column "trial" means each path of
the sentence and "batch" the batch-end registration. The full route is
`ObjectiveOwnership.registry_costs`.

| Registered term | Comparison and uninformed baseline | Priority | Owner and gradient |
|---|---|---|---|
| `answer.what/where/when` | Numeric output or generated surface band vs supplied target; origin per band | corresponding event-band scale | Trial and batch; A |
| `reconstruction.free_bytes`, `reconstruction.coverage` | Common reverse bytes vs observed input; identity and missing-position audits | `reconstructionScale` | Value only, `trained=False` |
| `reconstruction.what/where/when` | Masked percept prediction vs detached arrival; origin per band | reconstruction scale × band scale | reconstruction/R |
| `reconstruction_reverse.what/where/when` | Live input field reversed against detached native events; origin per band | reconstruction scale × band scale | reconstruction/R; perception, definitions and basic percepts, no grammar |
| `reconstruction_order_zero.what/where/when` | Untransformed sentence end slots → generate → new occurrences → common reverse against the same native events | reconstruction scale × band scale | reconstruction/R; compose, grammar inverse, selected definitions and native decoder |
| `reconstruction.free_percept` | Decoded native percepts against the frozen percept forecast residual | `reconstructionScale` | reconstruction/R; field context, detached predictor |
| `reconstruction.part` | Test lesson's common-reverse percept readback | `reconstructionScale` | reconstruction/R |
| `output.compose_score_function`, `output.thought_score_function` | Detached completed-trial cost advantage times the selected log probability | whole term | chooser/A for compose and reconstruction-category thought |
| `output.walk_credit` | Generate departure's score-function credit | whole term | chooser/A |
| `expectation.concepts` | Prediction against detached post-thought order-zero knowing | 1 | expectation/R; predictor only |
| `leaf_distill` | Historical root decoder's predicted exact leaves vs retained leaves; origin | `leafDistillWeight` | Batch; decoder and live input root where enabled |
| `expectation.activation` | Predicted two-lane bank activation vs the next row's actual activation; bounded MSE | `sentenceExpectationLossWeight` | Trial or batch, once; shared predictor only, sources and targets detached |
| `expectation.word` | Pre-arrival word activation vs the admitted word's two lanes; bounded MSE | `wordExpectationLossWeight` | Trial or batch; shared predictor at the word read, sources and targets detached |
| `expectation.symbols` | Pre-compose bank activation vs the resulting symbolic field's two lanes; bounded MSE | `wordExpectationLossWeight` | Trial; shared predictor at the S read, sources and targets detached |
| `expectation.components` | Retained innovations plus the current sentence's; one-sparse sparse-coding energy over the change columns only (below) | `independenceWeight` | Trial; E owns columns and relevance; the existing closing comparison credits the global grammar choice |
| `grammar.compose.choice` | Joint compose probability vs annotated operator; `log(choice count)` | `grammarLessonWeight` | Trial; compose chooser; unlabelled waiting sites excluded |
| `grammar.generate.choice` | Generate action logits vs annotated action; `log(choice count)` | `grammarLessonWeight` | generate_lesson/A; generate policy only |
| `grammar.generate.operands` | Generated children vs annotated detached codes; origin | `grammarLessonWeight` | Reporting only (`trained=False`); no operator gradient |
| `embedding.positive/negative` | Sampled lexical logits vs positive/negative membership; `log 2` separately | `embeddingScale` | Batch; joint perceptual embedding |
| `conceptual_sbow.positive/negative` | Cosine logits vs context/non-context membership; `log 2` separately | `conceptualSimilarityScale` × existing SBOW strength | Batch; conceptual codes on the enabled parallel path |

`expectation.components` is a sparse-coding energy over change only.
Individuals are native concept rows (`IdentityRows.individuals`) and receive
no sparse-coding pressure. A kept sentence's innovation, its fused meaning
minus the expectation's prediction, is read in the coordinates of those rows
for each of the three roles, and the change dictionary
(`IndependentComponents.verbs`) explains it as $\Delta_t = B\,e_t$: the columns
of $B$ are fixed patterns of change over individual coordinates and $e_t$ is
one-sparse (one verb phrase per sentence), so the coding is a vector
quantization of change. The energy is squared reconstruction plus a Laplace
activation prior, unit column directions, column incoherence and a learned
relevance scale per column; inference is one projection to a bounded support,
one small differentiable normal-equation solve and a soft threshold. The
prior is heavy-tailed for a reason: an isotropic Gaussian regularizer
(SIGReg, considered and not adopted) is invariant under rotation and cannot
prefer any axes, while a sparse source model can. The
population is the retained observations plus the current sentence, so the
statistic does not depend on batch size; an innovation leaves it when its
occurrence leaves retained LTM.

Targetless terms retain their strength and are recorded apart from relative
errors. They have no fabricated target or normalization denominator:

| Term | Definition and strength | Owner and gradient |
|---|---|---|
| `definition_sparsity` | Rank-ordered soft-L0; `definitionSparsityScale` | Batch; definition coefficients |
| `gate_l1` | Absolute lift/lower raw gates; `gateL1Lambda` | Batch; operator gates |
| `concept_readout_l1` | Active sparse readout L1; configured `l1Lambda`, averaged over distinct observed concepts | Trial/batch report only; one proximal application per applicable optimizer step, never another autograd penalty |
| `truth.falsity`, `truth.balance` | Stored-truth incompatibility and forbidden-corner penalties; the caller's `truth_loss_weight` (default 0) and balance strength | Batch; live operands if present |

A thought choice learns by the whole-path score function like every other
choice; it has no policy loss of its own (`selectedThoughtPolicyWeight` is
rejected at configuration). Raw `output`, reconstruction, reverse and
embedding totals, grammar-lesson summaries and missing-candidate counts are
reporting-only entries. Standalone embedding pretraining is separate from
these costs.

<a id="terms-on-internals"></a>
### Terms evaluated on the model's own internals (Alec, 2026-11-03)

Rule (Alec, 2026-11-03): a teacher does not reach into the model's internals.
No term takes its target from the model's own derivation to force a choice.
Every chooser (compose, attention, bind, thought and both generate walks)
learns only from its own comparisons, by the proposal-corrected score
function, plus lessons where labelled data supplies a target. No walk-policy,
decomposition, verb or modal teacher exists under this rule; verbs and modals
learn by their participation and by
[memory's state errors](#memory-readers-and-state-errors). The generated
order-zero reconstruction takes the place of the earlier field, iteration,
forward-discard, content and reference reconstruction terms. A term evaluated
on the model's own states is kept only when it supplies no target to a choice;
the kept terms follow. What remains to revisit is in
[Future work](FutureWork.md#terms-evaluated-on-the-models-own-internals-alec-2026-11-03).

| Term | What it constrains | Weight, owner and side |
|---|---|---|
| `reconstruction.walk_residual.{form,for,against}` | Reconstruction walk's selected inverse recomposition and unresolved frontier | Weight 1, baseline 1; reconstruction/R, whole rows |
| `reconstruction.walk_credit` | Reconstruction departure's proposal-corrected score function on its complete native loss and residual | Weight 1; reconstruction/R, whole rows; exact ties have no graph |
| `output.walk_residual.{form,for,against}` | Output walk's selected inverse recomposition and unresolved frontier | Weight 1, baseline 1; side A, owner reconstruction (its objective), whole rows |
| `<category>.definition_fold` | Witnessed member evidence against the class's sparse definition | Weight 1; existing category owner and side |
| `memory.link_coverage` | Coverage of the selected retained reference's link | Weight 1; detached value in choice comparisons, live derivative in reconstruction/R |

The reconstruction targets are native percept events; conceptual activation
and symbols are intermediate states. Generated bytes remain detached reports,
while the continuous face carries each native event loss.

## Memory readers and state errors

| Term | Owner | Side | Live derivative |
| --- | --- | --- | --- |
| `memory.link_coverage` | reconstruction | R | What the chosen link's coverage reads; choice discrete |
| `memory.verb_state` | reconstruction | R | Lift action readout on recorded operands; later reference target detached |
| `memory.modal_state` | reconstruction | R | Modifier action on equal-time actual/alternative references; targets detached |

A link's comparison between paths uses a detached coverage value; the kept
close's registry carries the live coverage derivative in the single backward.
State errors are registered at the kept close only in training, with the
no-change squared cube-chart distance as baseline. Parallel-field,
compose-centroid and typed-slot property gradients pass through basic
percepts; the presence derivative remains the directly weighted carrier.

## The generation residual

**What the residual is.** Generation reads a concept back into words by a
walk over a stack: at each step the top code (the parent) is split by an
inverse binary operation, rewritten by an inverse unary operation, or
emitted as a word (STOP). For every legal proposal at the step, the menu
(`Generative.eligibility_menu`) composes the proposed pieces forward again
and measures how well that reproduces the parent, separately for the form
block and the two meaning lanes (for and against). That measurement is the
residual. It is used four ways:

1. *Chooser features.* Each proposal carries five detached numbers into the
   generate chooser (`InverseMenu.features`, scored by
   `LanguageSpace.generate_policy_logits`): the form residual, the two lane
   residuals, an exactness flag and the stack effect ($+1$ binary, $0$
   unary, $-1$ STOP).
2. *Exactness.* A proposal is exact when its residuals are at most
   $10^{-8}$; an exact binary pair or exact terminal pre-empts every
   approximate proposal at that step (the pair-first rule).
3. *Admissibility.* A residual must be finite: a finite-domain proposal whose
   residual is not finite raises at its producer.
4. *The loss.* The chosen proposal's residual at each step, summed over the
   walk, plus the energy left unresolved on the stack, is the reconstruction
   term `reconstruction.walk_residual`. On the gradient path it is recomputed
   through the chosen recomposition (`Generative.selected_residual`), so its
   gradient trains the grammar operators and their inverses.

**The measure (decided 2026-10-30; Alec: no ratios of error terms).** The
residual is an absolute error in the code's own chart, in all four uses:

- *Symbolic codes* are membership vectors in the unit cube (every admissible
  symbolic proposal is checked `in_cube`), so the mean squared error per
  lane is already bounded in $[0, 1]$ and in fixed units. The symbolic
  exact-pair test in `LanguageSpace.decoder_eligibility` uses the same
  measure.
- *Numeric codes* (the math chain's walks, `cube_domain=False`) are not
  membership coordinates and span magnitudes. Their error is the squared
  difference in an $\operatorname{asinh}$ chart,
  $(\operatorname{asinh} r - \operatorname{asinh} p)^2$: absolute near zero,
  logarithmic for large magnitudes, continuous, and with no division.
- *Exactness* reads the same absolute error against the same tolerance,
  $10^{-8}$, in either chart.
- *Admissibility*: an absolute error of finite codes is finite, so the
  finiteness test is an assertion that fails loudly, never a silent
  exclusion from the menu.

**The loss.** Let $\operatorname{chart}(x) = x$ on symbolic cube walks and
$\operatorname{chart}(x) = \operatorname{asinh}(x)$ on numeric walks. For each
of form, for and against, a chosen step contributes the mean of
$(\operatorname{chart}(\mathrm{recovered}) - \operatorname{chart}(\mathrm{parent}))^2$
in that lane. The walk adds these errors and the unresolved stack's per-entry
mean-square chart energy, without averaging across steps. The sentence close
registers three terms, `reconstruction.walk_residual.form`, `.for` and
`.against` (`output.walk_residual.*` for the output walk), each through
`errors.error(name, per_stream_sum, 1., mask=active, category='reconstruction')`:
one entry per active stream and a baseline of exactly one, so the registry
averages over streams and divides once per term, and each lane stays in its
own units. No floor, cap, detachment of the error or dropped term is used.

**Why not a ratio.** A residual divided by the parent's own energy is
discontinuous at zero: the against lane is usually exactly zero or nearly so,
where the ratio is the raw error times an arbitrarily large factor. That
measure once produced a loss of about $10^{29}$ and a non-finite gradient.
Within one step every proposal shares the parent, so the division never
changed the ranking among proposals; it only rescaled the features, and
through the admissibility test it could remove a correct proposal from the
menu. The absolute measure changes the scale of $R$ against $A$ in the path
comparison and in the single backward; no balancing weight is introduced to
compensate.

## Pullbacks of particular operations

**Selected fields and the verb reflection.** For a mixed batch, the concluded
sentence and the costed thought output replay the selected integer actions
once. Each stream keeps the fields of its selected path, and only that
realization supplies the pullback from those fields. The parameterized verb
reflection rescales its direction by an exact power of two before taking the
squared norm, which preserves the reflection for every nonzero direction and
avoids underflow in its intermediate divisor.
Certificates: [the selected fields](../test/test_generation_selection_backward.py),
[the reflection](../test/test_meaning_kernels.py).

**Whole presence and gradients.** For total membership $t$, the whole's presence is $p=\min(1,t)$. A centroid moves its parts form $x$ toward the whole mean $m$ by $p/2$, using a lerp for exact rails. Its differentiable carrier is $\sum_r w_r c_r/\max(1,t)$, so no presence consumer divides by a small mass. For $0<t<1$, the membership pullback is $g\mathbin{\cdot}(c_r-x)/2$; for $t\ge1$, it is $g\mathbin{\cdot}(c_r-m)/(2t)$. The code pullback is $w_r g/(2\max(1,t))$. At $t=0$ the consumer contributes nothing and sends no whole gradient. The inverse first recovers $(parent-(p/2)m)/(1-p/2)$, then splits the counts; candidate search uses the same equations. Its forward uses the detached exact-rail mean and presence, while its derivative comes from the direct carrier expression. At $t\ge1$ this forward is bitwise $2\,parent-m$. [Presence and inverse certificates](../test/test_whole_presence.py)

**Literal whole membership gradients.** The independent `weighted_mean` read still returns the convex mean of the property codes at their active weights. `_WholeMean` interpolates one property at a time to retain exact cube rails. Its derivative uses the complete span. For a mean $m$, total positive mass $T$ and incoming gradient $g$, membership $w_i$ receives $g\mathbin{\cdot}(c_i-m)/T$ and code $c_i$ receives $(w_i/T)g$. A zero cumulative prefix inside an occupied span has no role in those derivatives. An entirely empty span keeps its existing zero value and first-membership tangent. This literal read retains its normalized derivative; presence consumers use the bounded carrier described above. [Producer and compiled derivative certificate](../test/test_form_centroid_numerics.py)

**Hierarchical conceptual conditioning.** `ConceptHierarchy.condition`
executes under `torch.no_grad` at the kept sentence boundary. It applies the
conceptual placement step to native membership context, from higher orders
downward, with a shared snapshot per order, then refreshes member centroids.
This directional update preserves forms and stored confidence radii, and the
optimizer's projection separately keeps concept lengths at their stored radii.
No hierarchy update sends a gradient or a placement target into perceptual
form coordinates. The placement of order-zero meanings at kept closes
(`MereologicalCodes.place_rows`) is likewise a detached index computation
with no optimizer owner.
[Architecture](Architecture.md#hierarchical-conceptual-context-alec-2026-11-08)
specifies the chart and membership rules.

**The derivation inverse.** The compose-forest inverse
(`DerivationReconstruction`) is a report and a tool, not a trained loss. It
inverts each recorded operation from that operation's own journal: the
selected forward action's object value, object-presence flag and verb group
are kept beside its numerical frame, and a missing non-adverb action record
raises instead of selecting an identity basis. When a consumer refreshed its
operand from a bound STM identity, the inverse translates the recorded
publication by the discrepancy between the recovered operand and the operand
actually read (additively for form coordinates, in the operation's centered
atanh cube chart for the two membership lanes), so a direct edge returns the
recovered value exactly. An unfinished reading keeps every surviving root with
its gradient. The input's measured leaves never supply these values. Free
proposals, by contrast, use their proposed object and the shared verb-group
lookup, with the identity basis for an absent object or unknown group, and
need no input record.
See [the inverse](../bin/DerivationReconstruction.py) and
[its certificates](../test/test_reconstruction_magnitudes.py).

## Ownership audit

`branchDiagnosticsEvery` reports `[objective-ownership]`: every trainable
optimizer parameter, its sole declared owner and the writers actually observed
during backward. Inactive parameters remain listed and are not counted as
reached. Under `separateChooser=true` a duplicate owner or a second objective
writer raises an error; under the default the observed writers are the two
sides, R and A, and the epoch report adds each unit's projection statistics
([projected gradients](#default-projected-gradients)).
Writer separation is not an acceptance claim or a guarantee that the
objectives improve together.

The one projection in training is the default per-unit projection of A against
R, which keeps no state beyond the epoch report; no per-operator projection,
norm rebalance or conflict counter exists, and `reconstructionPriority`,
`outputGradientRatio` and `reconstructionLossTolerance` are rejected at
configuration. The stateless gradient math utilities remain available for
read-only measurements. Separate weights remove direct shared-weight
competition, but reconstruction can still change the forward inputs of the
answer or expectation predictors.

The read-only epoch report (`SentenceTraining.gradient_receipt`) gives the
norms of the R and A gradients that reach the understanding: the Euclidean
norms of the weighted parameter-coordinate vectors accumulated over the epoch,
not averages of step norms. Understanding coordinates are the optimizer-owned
codebooks, grammar operators and inverses, and conceptual definition matrices
of reconstruction's ownership group; placed code buffers and answer-owned
readers are excluded. Vectors are grouped by stable parameter name, and a
sparse definition sums its surviving edge coordinates by allocation ID across
replacements. Under `separateChooser=true` the A contribution to these
coordinates is zero by construction, and the per-operator dissonance reports
hypothetical cross-owner directions. The report uses the vectors already
collected, without another backward or a `.grad` write, and its time is
excluded from training step time.

## Verification

[Ownership tests](../test/test_objective_ownership.py) check restricted writers,
expectation source/routing cuts, perception pullback ownership, a common path
record and the answer cut. [Priming tests](../test/test_sentence_priming.py)
check that batch rows never prime each other, that serial reading keeps its
prediction apart from placed rows, and the part graph's constructed capacity.
[Selection tests](../test/test_reconstruction_precedence.py) retain reconstruction
precedence; [sentence comparison tests](../test/test_sentence_comparison.py)
check equal-parameter costs before updates.

Superseded mechanisms and their measurements are recorded in
[DevelopmentHistory](DevelopmentHistory.md).
