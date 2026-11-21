# Expectation across the architecture

Expectation is a gain on what arrives, never a subtraction from it (decided
2026-10-19/20). At the symbol level the content a reader receives is
$c = m \odot a \odot o$: attention's gain $m$, expectation's gain $a$ and the
arriving signal $o$, with the residual reported beside it in two lanes,
excess and omission. No sign-reversed image of the prediction is added to
content. At the percept level the residual is signed and is computed beside
the observed signal, which the forward path carries unchanged
([Percept arrivals](#percept-arrivals)).

Expectation is not prediction: it is an aid to understanding and supplies
missing data, and it works on percepts and on the field, never on a whole
sentence or on an idea, which is a computed value (Alec, 2026-11-17). Nothing
estimates a sentence. No read is made at a sentence's close or for the next
row of a document, no sentence is taken as a target, and nothing answers what
comes next.

`SymbolExpectation` owns one activation predictor shared by two reads during
the read of a sentence: the word read (`WordArrivals`) and the field read of
the symbols a word's composition gives (`SymbolField`). It predicts
independent for/against activations over native identity rows, over the whole
symbol bank. It has no byte, ARMA, root-only or slot-structured prediction
head.
Each level predicts what it is about to receive from what has already arrived
at it, never from its own prediction, so it predicts under zero input as well,
and no level's expectation is derived from another's by inversion: one wrong
estimate at the top cannot become a wrong expected operand at every round, and
the error stays bounded per level. Expectation selects; it calls no grammar
operator (Alec, 2026-11-01): percept expectation selects active percepts on
both paths, concept expectation selects active concepts on the parallel pass
only, and on the serial path expectation selects no concept.

The conceptual spread that primes attention, the chooser and retrieval is
`expect`, and it has no map of its own. It is the reconstruction made in
stages on the structure's own Pi, Not and Sigma (Alec, 2026-11-19: "a
reconstruction that happens in stages: we are predicting the state of
percepts, concepts, and output. using the existing architecture will allow us
to to sparse in interconnection, which should help amplify the salience").
What is expected is what the vertical spread gives on each order's matrices
([Architecture](Architecture.md#the-vertical-spread-and-the-way-back)), in
three stages: the percepts expected at order zero, the concepts expected at
each order, and the output. It is built in, turned on by a parameter of the
model. At the open pass (`ConceptSpread.open_prediction`) the spread reads the
sentence's observed field and publishes what it gives (`_expected_knowing`),
which attention, the chooser, retrieval and `ConceptSpread.rule_order` read
detached. It is taken again, without publishing, for the forecast of each
look ([below](#the-forecast-of-a-look)).

Its teacher is the reconstruction's error: each stage is compared with what
the whole sentence has at that stage. Where `Omission.stage` hid part of the
sentence, the error at what was hidden teaches it; the concepts' stage,
`expectation.concepts`, compares what is expected from the lossy field with
the field with the hidden units present, which is detached, and a sentence
with nothing hidden gives that stage no cost. Omission teaches expectation
alone. What is expected is written apart from what is observed: it "is not
counted as evidence, it is just removing some of the burden from later
layers. it should be used only to guess at network activation" (Alec,
2026-11-19). So the error at a hidden word teaches the rows that hold things
together, the symbols above, and no entry from one word's row to another
word's concept at the same order, which would make a concept present for a
word that is not there. Placement does not read the spread, since codes must
not read a value computed from codes: its output is not written to the seen
ledger, and no cost of it moves a place.

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
predicted into existence. The shipped defaults are two zero gains
(`perceptExpectationGain`, `symbolExpectationGain`), `expectationPrime=true` and `explorationUniformFraction=0.5` (Alec). The
selection prime and the exploration fraction are separate from the gains, and
a nonzero gain must first pass the zero-gain
[parity configuration](../test/fixtures/item4-parity.json), which keeps
prediction enabled, turns off the selection prime of compose and of `attend`
and restores uniform eligible alternatives ($\varphi = 1$). The
expectation prime is not capped, in binding menus or elsewhere
(`OperationSelectionLayer.prime_bias` is the uncapped, detached prior): the
form cue and the expectation are legitimate non-uniform priors over
referents.

The percept residual that `attend` reads is the signed residual against the
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

The word and S reads use $\kappa = 1$: their predicted lane magnitudes supply
the local certainty, without a second confidence gate.

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
Open grammatical
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

## Intuition's map and the read to a point

Intuition is not expectation (Alec, 2026-11-19: "if expectation comes from
the arch, please let intuit have the pi-not-sigma that exists outside of the
architecture. We should not call it "the expected field", since it is called
from an operator, whereas expect is built in to a model parameter").
`intuit` is an operator of thought, and it has a Pi, Not and Sigma of its own
that stands outside the architecture: a stacked sigma-pi map (`SigmaPiStack`)
from the observed concepts of order zero, both lanes, to the same concepts. A
stacked sigma-pi map is a row of fused pairs (`SigmaPiLayer`), each the
structure of every order: a Pi of unsigned rows, each row the product of its
inputs' memberships on one lane raised to weights at zero or above, then a
Sigma, $\tanh(x\,U)$ with no bias, whose entries are for or Not and which
gives one signed number in $[-1, 1]$ for each output
([Architecture](Architecture.md#sigma-and-pi-layers)). It starts by adding
nothing, since its last Sigma's weights start at zero. When thought calls
`intuit`, the map gives the *intuited field*; the operator joins it into its
stream's knowing at the observed places, marked as intuition, and carries it
up the orders by the closure of the definitions (`FieldProjection.close`);
other orders and other streams are kept. Nothing reads the intuited field as
expected, so expectation is never evidence, with no exception. Its teacher is
the field that thought leaves, with the cost of the output (Alec,
2026-11-20: "restore what you approved on 2026-11-12, the field that thought
leaves, with the output's cost").

A field can also be read to one point: many concepts to one value of a code's
width. With no sentence comprehension the read is the centroid of the field's
codes, weighted by their evidence (`FieldProjection.key`), as the number
modality reads its field (`NumericPercepts`). Where sentences are comprehended
the conceptual space that owns the concepts builds a stacked sigma-pi of its
own for it (`idea_stack`; `ideaHidden` 64, `ideaLayers` 1), and a model that
comprehends no sentence builds none. Its input is the observed order-zero
field on both lanes, held detached at the sentence's open
(`IdeaRead.open_read`); its teacher is the idea that comprehension leaves: after
the kept close, the read of that field is compared with the kept reading's
root idea (`record.meaning_root`), detached, by squared error over the streams
present (`expectation.idea`, `IdeaRead.complete`). Its gradient reaches only
its own stack. The expected point is this read of the expected field
(`IdeaRead.expected_point`); nothing reads it yet.

So there are three teachers: expectation learns from the reconstruction in
stages, omission among it; intuition's map learns from the field that thought
leaves and from the output; and the read to a point learns from the idea that
System 2 concludes. Why a point is kept at all, as a key and not as what is
reasoned on, is in
[Philosophy](Philosophy.md#expectation-as-a-gain-attention-as-exclusion).

## The forecast of a look

Before a look is perceived, expectation forecasts what should be at its place
from everything seen elsewhere (Alec, 2026-11-16: "span left out").
`LookForecast` keeps a path's located readings. For a look it pools, on each
lane of each order-zero row, the greatest reading among those whose extent
lies outside the look; a reading that overlaps the look, even an earlier
reading of the same place, gives nothing. It takes the spread on
that field without publishing it (`ConceptSpread.expected`) and closes the
forecast upward with `FieldProjection.close`. Every order-zero address is
forecast, a silent one too. There is one gain field: the look's activity is
the `attention` of `Expectation.gain_field`, and the look's forecast is its
`expected`. What the look returns beyond the forecast is the **excess**, and
what was forecast beyond what it returns is the **omission**, row by row and
lane by lane, each summed over the rows only after the split; these are the
surprise `attend` reads. The forecast and both numbers are detached.

At the opening glance each word gets its pair from a forecast with that word
left out, all words in one batch. In "This is Felix." `Felix` is forecast from
`This`, `is` and `.`: a word the context makes likely reads with little
excess, a new name with much. A later look records its own pair in the
sentence's record (`_sentence_look_fields`) and replaces a word's latest pair
only when the look returns that one word; a look over several words reads the
sum of its words' latest pairs. `attend` reads the pairs through two columns
of its context projection that start at zero (`AttentionContext.project`), so
an untaught `attend` asks as it would without them. The comparison is reported and is
not a cost: expectation is taught stage by stage at each look, by what the
look brings at each order.

## The observation and its surprise

`LTM0` is the single durable owner, and it keeps observations, never
expectations: no forecast survives its sentence and no estimate row is
written. What the kept close writes is the sentence's own row, and the same
close admits that row, once, to its stream's recent situation
(`SymbolExpectation.include_occurrence`), by its occurrence with the row's
symbols and activations. The situation holds the address and reads the row
from the store when the word read runs; it is bounded, and cleared when a new
document begins. Provisioned truths, speculative paths and internal thoughts
are not admitted.

The shared predictor predicts content, not time. It receives time only as an
input: the stamps in its context are expressed as ages from the present, and
four `RowTime` coordinates after its $4D$ context carry the recent age, never
an absolute stamp.

Since nothing estimates a sentence, no surprise of a sentence is computed: the
close passes none, and the row it writes starts at `activityDefault`
([STM](STM.md#10-ltm-as-the-chain-of-stm-end-states)). The writer still takes
a surprise in $[0, 1]$ as a row's starting activity, and `-1` for unknown.

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
its declared weight (`expectation.word` at `wordExpectationLossWeight`,
`expectation.symbols` at `symbolFieldLossWeight`); neither enters the R+A
judge.

Source encodings, observed targets, thought effects and durable history are
detached. Only the expectation maps (the shared predictor, the read to a
point and the percept predictor) receive local E gradients; in the one
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

Each forecast is staged before its target arrives; every path of the sentence
reads that snapshot, and the recent situation takes the kept external
observation once, at the kept close, so speculative paths and generated
thoughts never count external evidence twice.

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
predictor; and its retained observation rows survive. A checkpoint that holds
the retired `intuit` stack, or an `expect` stack on a conceptual space that
does not own the concepts, loads with those tensors discarded and a warning;
one with no read to a point initializes it, with a warning.
The attention context receives zero-initialized columns for the bank prime
and the percept residual. Percept moments update only during training on
external inputs and are checkpointed; an older checkpoint starts them cold.
Transient pending graphs do not survive a checkpoint restore. The
expectation-free proposal view has a zero prime and a zero percept residual,
with only the open request and gist as needs.

The regression contracts are causal predictions, detached ownership,
identity preservation at zero gain, no forecast written to memory, exactly-once
external observations, the recent situation's isolation by document, local E training, R+A selection,
and the actual probability correction for the exploration mixture. The
behavioural certificates are: a sentence repeated until its symbols are
predicted at confidence still passes content at the gain floor and is answered
and stored correctly (the fully expected cat,
[test_item4_native.py](../test/test_item4_native.py)); a symbol expected at
confidence and then absent appears in the omission lane of the residual and
nowhere in content, and an omission supports only a checked `not` inference;
and no content coordinate is ever negative (`gain_field` asserts it).
