# Expectation across the architecture

Expectation is a gain on what arrives, never a subtraction from it (decided
2026-10-19/20). At the symbol level the content a reader receives is
$c = m \odot a \odot o$: attention's gain $m$, expectation's gain $a$ and the
arriving signal $o$, with the residual reported beside it in two lanes,
excess and omission. No sign-reversed image of the prediction is added to
content. At the percept level the residual is signed and is computed beside
the observed signal, which the forward path carries unchanged
([Percept arrivals](#percept-arrivals)).

`SymbolExpectation` owns one activation predictor shared by the word, composed
symbol, closing and document reads. It predicts independent for/against
activations over native identity rows. It has no byte, ARMA, root-only or
slot-structured prediction head. The three-role memory envelope retains a
representative identity; the full symbol bank is the forecast.
Each level predicts what it is about to receive from what has already arrived
at it, never from its own prediction, so it predicts under zero input as well,
and no level's expectation is derived from another's by inversion: one wrong
estimate at the top cannot become a wrong expected operand at every round, and
the error stays bounded per level. Expectation selects; it calls no grammar
operator (Alec, 2026-11-01): percept expectation selects active percepts on
both paths, concept expectation selects active concepts on the parallel pass
only, and on the serial path expectation selects no concept.

The conceptual spread that primes attention, the chooser and retrieval is
the conceptual space's own `ActivationPredictor` (`concept_predictor`,
`ConceptSpread`): it predicts the next order-zero knowing from the observed
one, its frozen prediction is read detached, and it learns only from the
detached post-thought order-zero view (decided 2026-10-30). Placement does
not read the spread, since codes must not read a value computed from codes.
Expectation proposes nothing: top-down proposal belongs to `intuit`, a
separate layer whose weights are trained only by its use in output, with no
gradient from reconstruction or expectation.

The content gain is $m\,(1 - g\kappa + g\kappa s)$ for gain strength $g$ and
confidence $\kappa$, where $s$ is the saturated sum of
absolute lane residuals divided by expected lane mass plus epsilon. The
entire saturation is detached, including its denominator, so the predictor
receives gradient only through the residual, never through the gain; otherwise
it could mispredict to brighten the signal or over-predict to silence it. Each
symbol keeps one gain over both evidence lanes and retains its raw identity.
The gain is one scalar per symbol, never one per coordinate: a per-coordinate
gain changes a vector's direction, not only its magnitude, so a cat whose
every feature but its whiskers was predicted would become a whiskers vector
and be identified as whiskers. Identification needs the whole, so
per-coordinate salience goes to attention (where to look) and content is only
scaled. The form is divisive by design: expectation and repetition
suppression reduce response rates rather than subtracting into negative
firing, contrast gain control divides by the local mean, and at full gain a
perfectly stable percept passes nothing, as stabilized retinal images fade.
At full confidence and no surprise the gain falls to its floor $1 - g$, which
sets how much of a perfectly expected arrival the model keeps. A fully
expected symbol is still seen, twice over: as the prime, present before it
arrived, and as content dimmed toward the floor. Consumers that need content
read the attenuated whole, and the estimate when they ask what was expected;
salience, retention, orienting and the predictor's training read the residual
lanes.

Expected activation separately primes selection. It is never added to
comprehension, so top-down expectation reaches perception only through
attention, biasing where the reading looks, never as content: a hallucination
here is a concept conceived without perceptual evidence, not a percept
predicted into existence. The shipped defaults are four zero gains,
`expectationPrime=true` and `explorationUniformFraction=0.5` (Alec). The
selection prime and the exploration fraction are separate from the gains, and
a nonzero gain must first pass the zero-gain
[parity configuration](../test/fixtures/item4-parity.json), which keeps
prediction enabled, turns off the compose and candidate-attention selection
prime and restores uniform eligible alternatives ($\varphi = 1$). The
expectation prime is not capped, in binding menus or elsewhere
(`OperationSelectionLayer.prime_bias` is the uncapped, detached prior): the
form cue and the expectation are legitimate non-uniform priors over
referents.

Candidate attention's percept salience is the signed residual against the
last arrival's expectation (`attention_residual`, scaled by
`perceptExpectationGain`); the perceptual moments are maintained but unread,
and codebook lookup reads the original coordinates. Prediction-error salience
has two known limits. An unpredictable source (Bernoulli noise) keeps a
residual however well its mean and variance are predicted, so a noisy source
is never rejected as uninformative; and a forecast of marginal statistics
cannot detect a change only in the order of transitions. Novelty is not a
separate salience term: the dial is `hetTolerance` $\tau$ on the candidate's
evidence lanes,
$(c^+, c^-) \leftarrow (c^+, c^-) - (1 - \tau) \min(c^+, c^-)$, left at 1, a
no-op; toward 0 the both corner, the expected held beside the arrived,
collapses to its net lane and nothing captures.

Closing confidence $\kappa$ is per symbol, the maximum of its two predicted
activations. It is never taken from the representative envelope's role slot.
Word and S reads use $\kappa = 1$: their predicted lane magnitudes supply the local
certainty, without a second confidence gate. Closing reads retain an
explicit presence-confidence gate.

A native handle identifies an activated symbol in its bank address domain.
Without a handle, a bank identity requires raw-key distance at most `1e-5`.
An unknown composed value passes whole and contributes no target to its
nearest bank key. The STM keeps the raw identity and original projection
coefficient. Compose scores a separate gained activation view, including
operators that read a coefficient from a norm; its selected candidate retains
the raw identity. Order-zero poles receive P at perception and S once at
their symbol consumer. The S read uses the actual for/against lanes,
including purely negative evidence. Repeated reads cannot compound S. The
sentence prime and S read are cleared before closing thoughts and before the
next peer-word input. P's gain affects the incoming content, but the S
residual and its gain both compare the original lane pair with the S prior.
P cannot change that local observation or manufacture an omission for S.

The local residual has excess and omission signs for each evidence lane.
It trains only the predictor: both target and all source context are detached.
The row and document gains compose multiplicatively when thought reads the
closing. A cold document has no document attenuation. Open grammatical
objects keep their expected prime and are spared attenuation. Deep
expectations have no domain outside the admitted region: there the prior, the
residual and the content are exactly zero (`Expectation.gain_field`), so no
omission is reported for a symbol attention never visited, while an expected
but absent symbol inside the region does count as an omission. `non` removes
both the positive and the negative evidence of a dimension, where `not` swaps
the two (Alec, 2026-11-12), so a dimension withheld by `non` has no prior at
all: no expected prime, no attenuation and no omission. The raw observation,
prior and content remain available to awareness even when the content gain is
zero. An omission can support an explicit checked `not` inference; it is
never an external observation.

## One pair, one observation

`LTM0` is the single durable owner. A warm external boundary writes
an `estimate` row before its `observation` (or interrogative `question`) row and
links their stable occurrences in both directions. Only the observation enters
the chronological predictor context. Provisioned truths and internal thoughts
are not additional external observations. Estimate rows are never facts or their
own corroboration, and ordinary LTM evidence readers exclude them.

An estimate predicts content, not time. Its row is written with the zero
`.when` band, the model's mark for no time, also when it is re-witnessed, and
it consumes no tick of the clock; the arrival's idea is the one entry stamped
at its tick, so the time LTM keeps is the arrival's. Recall by time and
content reads skip estimates. The predictor receives time only as an input:
the stamps in its context are expressed as ages from the present, and four
`RowTime` coordinates after its $4D$ context carry the recent age, never an
absolute stamp.

The estimate retains native identity, order and two activation logits for
every symbol. Full keys are retained for the observation's support and the
representative envelope; absent keys are explicit, never reconstructed from
later mutable weights. Before an observation is linked, a future prior may
retain every key. Checked bindings and scope come from preceding source
occurrences. They never copy the incoming target. Provenance records those
occurrences, the external stream/document and the intended observation. The
observation retains its own raw carrier and metadata. When the durable row
fuses several observed symbols, its `_observed_symbol_activations` metadata
retains the original two-lane target, aligned by the prior's native IDs.
Recomputing that target by quantizing the fused description would change the
measured residual, so checkpoint replay uses the retained target, including
negative evidence.

`expectation_pair(gain=..., object_mask=...)` reconstructs the prior and derives
content, gain, expected activation, excess and omission. At a close the
image's object mask spares the roles of the field's actual open slots (an
open relation spares its relation slot), for every entry; it reads no
grammatical mode. There is no third durable delta or conceived row.
Restoration reads the exact stored observation, so it also works at zero
content gain. A new pair requires capacity for both rows and fails before
publishing either when forgetting is required. Reads resolve stable
occurrences through the store index, including after compaction; they do not
scan unrelated rows.

Surprise is the detached mean squared activation residual over the soft union
of observed and expected support: each symbol's weight is $\max(o, \hat{e})$ across
its two lanes, and the weighted error $E$ is bounded as $E/(1+E)$. Confident
omissions count; empty bank padding does not dilute the score. Both writers
apply the same withheld marks and formula, independent of content norm and
gain. Local prediction training uses bounded activation MSE. Unknown surprise
is `-1`. An estimate whose checkpoint lacks its symbol-bank prior is rejected
at load.

The semantic sidecar of `LTM0`
([STM](STM.md#10-ltm-as-the-chain-of-stm-end-states)) fingerprints the links
with scope and source text, checks both directions on restore, and
participates in origin-compaction reachability. A tensor-only restore leaves
rows with missing required sidecar metadata unavailable. There is no second
memory owner.

## Local prediction credit

Prediction residuals train only the shared activation head. There is no
anticipatory-query reward or separate expectation policy. Reading the object
mask of a nested question follows its owned occurrence on the controller's
ordinary work meter. The held prior remains immutable when its local loss is
replayed against a newly arrived observation.

An order-zero arrival has two local training reads of the shared head: the
word read predicts the next admitted word from its preceding word context;
the S read predicts the symbolic field before the ensuing composition and
scores the resulting field. These distinct causal contexts and targets are
intentional. They do not apply the content gain twice. Each local loss keeps
its declared weight; neither enters the R+A judge.

Source encodings, observed targets, thought effects and durable history are
detached. Only the shared predictor receives local E gradients; in the one
backward every consumer reads expectation's output detached at the point of
use, so its parameters receive exactly their own local terms. Both the
whole-path keep rule and chooser credit use `R + A`; E is independent. This
includes thought departures, even when R and A disagree. Work remains metered
and reported, without an additional work-price term in the judge.

Judged by its surprise, a departure would teach the chooser to make the world
predictable (the dark room); judged by reconstruction and the answer, it is
rewarded for explaining the input or getting the outcome right. Both are
measured against the same whole observation and outcome, never against
attenuated content, since a chooser scored against dimmed content would come
to prefer omission. A departure's alternative is drawn from a mixture: a
fraction $\varphi$ (`explorationUniformFraction`) uniform over the eligible
non-greedy alternatives and the rest from the policy at
`explorationTemperature`, so each keeps probability at least
$\varphi / (K - 1)$ among $K$ eligible actions whatever the policy has learned;
credit uses the actual mixture probability. The paths themselves are described
in [Architecture](Architecture.md#whole-paths-and-concept-episodes-2026-11-11).

The forecast is staged before its target arrives, from the previous accepted
observation; every path of the sentence reads that snapshot, and the context
takes the kept external observation once, after the comparison, so
speculative paths and generated thoughts never count external evidence twice.

## Percept arrivals

At a percept arrival with live expectation $e$ and observation $x$, the
expectation is frozen before the arrival (`expected = e.detach()`) and every
path of the sentence reads that frozen value; `residual = x - expected` is
reported in two lanes, excess and omission, and the local loss $\tfrac12\lVert e - \operatorname{sg}(x)\rVert^2$
over the valid domain trains the expectation alone
(`PerceptExpectation.arrive`). The forward value is the observed signal: an
identity chart (`recombine_percept`) gives the content the value $x$ with
derivative one, and `expected + residual` is computed beside it only to report
cancellation. So a perfectly expected arrival still carries its whole signal
and task gradient, and an expected absence ($x = 0$) has residual $-e$. The
cut makes the ownership boundary explicit: detaching only the add-back while
subtracting a live $e$ would be a different learning rule (a negative
downstream gradient to expectation), and processing the two parts by
different functions changes the rule too, since $f(p) + f(r) \ne f(p + r)$
for a nonlinear $f$. A later learner that should learn only the error needs
an explicit residual target; being fed `expected + residual` does not give it
one.

The context and moments update at each arrival as detached state, so the
forecast at arrival $k$ reads the state left by arrival $k - 1$; the
predictor's parameters move once, in the sentence's one optimizer step, which
applies every arrival's accumulated local loss exactly once, and there is no
private optimizer.

Reconstruction is the inverse-side dual (`percept_reconstruction`). At the
percept leaves the inverse's native form is translated into a signed
correction against the expectation frozen at arrival, with the detached
teacher $x - \operatorname{sg}(e)$; adding the same expectation back gives the
full-signal reconstruction error. The translation separates the losses and
their gradients, so no reconstruction gradient reaches expectation and
expectation's local loss reaches nothing else, without claiming any reduction
in the inverse's capacity. The expectation used is the value recorded at
arrival, not a fresh forecast, since by the close the predictor has moved. An
inverse that outputs only the correction, and the conceptual split, are future
work ([FutureWork](FutureWork.md#the-residual-split-a-stopped-subtraction-or-the-correction-alec-2026-10-29)).

## Expectation and desire

Expectation and desire read the same primed field and are never one vector.
Expectation seeks homogeneity: it acts against the arrival, as a gain toward
the floor, has a local error (the residual) and trains the predictor. Desire
seeks to contribute: it acts with the arrival as a prime, has no local error,
and the policy it biases learns only from the outcome, the judge's $R + A$.
They are kept apart by what each writes to. Expectation writes the content
gain and the residual lanes; desire is a separate signed surface
(`_desire`) added to the detached relevance weights that prime selection,
floored at zero, so it reweights what is offered and removes nothing from a
menu. Nothing self-generated is written into comprehension's content. Merging
the two, as treating a preference as a prior expectation does, produces the
dark-room problem.

## Migration and regression dispositions

`architecture.prediction` is rejected as a retired setting. A checkpoint from
the retired slot and ARMA predictors loads as follows: its `cast.*` tensors
are discarded at the discourse loader, with no input projection to replace
them; its slot and ARMA head tensors migrate to a fresh shared activation
predictor; and its retained observations and their semantic sidecar survive.
The attention context receives zero-initialized columns for the bank prime
and the percept residual. Percept moments update only during training on
external inputs and are checkpointed; an older checkpoint starts them cold.
Transient pending graphs do not survive a checkpoint restore. The
expectation-free proposal view has a zero prime and a zero percept residual,
with only the open request and gist as needs.

The regression contracts are causal predictions, detached ownership,
identity preservation at zero gain, full-bank pair restoration, exactly-once
external observations, document/row isolation, local E training, R+A selection,
and the actual probability correction for the exploration mixture. The
behavioural certificates are: a sentence repeated until its symbols are
predicted at confidence still passes content at the gain floor and is answered
and stored correctly (the fully expected cat,
[test_item4_native.py](../test/test_item4_native.py)); a symbol expected at
confidence and then absent appears in the omission lane of the residual and
nowhere in content, and an omission supports only a checked `not` inference;
and no content coordinate is ever negative (`gain_field` asserts it).
