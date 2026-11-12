# Expectation across the architecture

> **Amendment of 2026-10-20 (read first).** Decisions of 2026-10-19/20 supersede statements below where they conflict; the text is kept as the record.
> - Naming (2026-10-21, Alec): LTM0 is the occurrence store (old LTM1); LTM1 is its relation stratum (old LTM2); LTM2 denotes relations over relations. The old LTM3 name is retired. Earlier names below belong to the retained record.
> - Rows: an LTM1 row is **one** concept slot; an LTM3 row is **three symbol references** (argument, kind, argument) of kind definition, attribution or conditional. The three-slot relation row with a REL code in the middle, null outer vectors, `refs`, `rel_type`, `role_mask`, `order`, `polarity`, `grammatical_mode`, `row_ids`, the address columns and the semantic sidecar are superseded: a row holds its slots, `.when` (model time, a column composition never reads), `truth_plus`/`truth_minus` (the lanes, renamed from `c_plus`/`c_minus`), the access count (renamed from `witness_count`) and `surprise`, and nothing else. **Amended (decided 2026-10-23; recorded 2026-10-30):** an LTM1 row is two tagged references (each to an LTM0 occurrence, a symbol or another LTM1 row, its tag saying which) with its kind read from the referents, not three. The unused `LTM0` columns are removed and the row has this layout (Alec, 2026-11-01).
> - Identity lives only in the major structures: a **minted** concept row at the shared index (its content the fold of its occurrences, rebuildable) or a **short-term** concept (the NP's fold point with its `.where`/`.when` band, for the situation). The nouns dictionary, the resolver's count tables and the ICA event chart are off the identity path; minting is by recurrence in LTM1; **`bind`** is the only binder (a thought operation, once per site), determiners are cue inputs, and its content term is the meet, not distance.
> - `.where` is inside the concept's reserved band (no column); **symbols have `.where` and `.when`**, concepts keep the extent and never the clock; `.when` is model time, sinusoidally encoded; a stream is the batch row that wrote a row, and a document is the cover row that starts it in that stream (reused at every later presentation; a row belongs to the document of the latest cover before it), so "sentence *k* of document *d*", source address hashes, timestamps and situation wholes are superseded (2026-10-24).
> - Expectation is a **gain field** $c = m \odot a \odot o$ (attention's gain $m$, expectation's gain $a$, the arriving signal $o$) with the residual in two lanes, never a subtraction or a sign-reversed image; per level with a local error (no inversion across levels — the invertible path is item 3's); one activation predictor over the symbol bank (the slot-structured `SentenceExpectation`, `BracketExpectation`, the byte level and the ARMA heads are retired); keep and credit both read `R + A`; defaults gains 0, prime on, `φ = .5`.
> - Amended 2026-10-29/30 (Alec): at the percept level the residual is signed and the forward path carries `expected + residual`, so "never a subtraction" is superseded there (built in item 2.7's final round); at the concept level the expectation is `ActivationPredictor`'s predicted next state, which at item 2.5 round B is also the spread that primes attention, the chooser and retrieval; `intuit` is a separate top-down layer with its own weights, trained only by output; one backward with each band stopped at its output is the decided direction. The round B and one-backward parts are built (accepted as implemented, Alec, 2026-11-01).


`SymbolExpectation` owns one activation predictor shared by the word, composed
symbol, closing and document reads. It predicts independent for/against
activations over native identity rows. It has no byte, ARMA, root-only or
slot-structured prediction head. The existing three-role memory envelope
retains a representative identity; the full symbol bank is the forecast.
The conceptual spread that primes attention, the chooser and retrieval is
the conceptual space's own `ActivationPredictor` (`concept_predictor`,
`ConceptSpread`): it predicts the next order-zero knowing from the observed
one, its frozen prediction is read detached, and it learns only from the
detached post-thought order-zero view (decided 2026-10-30).

The content gain is `m * (1-g*κ + g*κ*s)`, where `s` is the saturated sum of
absolute lane residuals divided by expected lane mass plus epsilon. The
entire saturation is detached, including its denominator. Each symbol keeps
one gain over both evidence lanes and retains its raw identity. The gain is
one scalar per symbol, never one per coordinate: a per-coordinate gain changes
a vector's direction, not only its magnitude, so a cat whose every feature but
its whiskers was predicted would become a whiskers vector and be identified as
whiskers. Identification needs
the whole, so per-coordinate salience goes to attention (where to look) and
content is only scaled. The form is divisive by design: expectation and
repetition suppression reduce response rates rather than subtracting into
negative firing, contrast gain control divides by the local mean, and at full
gain a perfectly stable percept passes nothing, as stabilized retinal images
fade. At full confidence and no surprise the gain falls to its floor $1 - g$,
which sets how much of a perfectly expected arrival the model keeps.

Expected activation separately primes selection. It is never added to
comprehension. The shipped defaults are four zero gains,
`expectationPrime=true` and `explorationUniformFraction=0.5` (Alec). The sole
[parity configuration](../test/fixtures/item4-parity.json) keeps prediction
enabled, turns off the compose and candidate-attention selection prime and
restores uniform eligible alternatives ($\varphi = 1$). It is a gate fixture;
zero attenuation alone does not promise the previous selection law. Reference
variants start with zero-initialized candidate-value columns and reference
projection in the shared chooser, so the chooser's own content term is
neutral at initialization, and both neutral paths receive ordinary
gradients. The expectation prime is not capped, in binding menus or
elsewhere (`OperationSelectionLayer.prime_bias` is the uncapped, detached
prior): the form cue and the expectation are legitimate non-uniform priors
over referents. The initial-uniformity check therefore reads the chooser's
own content score under the parity configuration, with the prime off and the
form term held out.

Candidate attention's percept salience is the signed residual against the
last arrival's expectation (`attention_residual`, scaled by
`perceptExpectationGain`); the perceptual moments are maintained but unread,
and codebook lookup reads the original coordinates. Prediction-error salience
has two known limits. An unpredictable source (Bernoulli noise) keeps a
residual however well its mean and variance are predicted, so a noisy source
is never rejected as uninformative; and a forecast of marginal statistics
cannot detect a change only in the order of transitions. Richer transition
prediction and belief-shift novelty are proposals, not part of the mechanism.

Closing confidence κ is per symbol, the maximum of its two predicted
activations. It is never taken from the representative envelope's role slot.
Word and S reads use κ=1: their predicted lane magnitudes supply the local
certainty, without a second confidence gate. Closing reads retain an
explicit presence-confidence gate.

A native handle identifies an activated symbol in its bank address domain.
Without a handle, a bank identity requires raw-key distance at most `1e-5`.
An unknown composed value passes whole and contributes no target to its nearest bank key. The STM
keeps the raw identity and original projection coefficient. Compose scores a
separate gained activation view, including operators that read a coefficient
from a norm; its selected candidate retains the raw identity. Order-zero
poles receive P at perception and S once at their symbol consumer. The S read
uses the actual for/against lanes, including purely negative evidence. Repeated
reads cannot compound S. The sentence prime and S read are cleared before
closing thoughts and before the next peer-word input.
P's gain affects the incoming content, but the S residual and its gain both
compare the original lane pair with the S prior. P cannot change that local
observation or manufacture an omission for S.

The local residual has excess and omission signs for each evidence lane.
It trains only the predictor: both target and all source context are detached.
The row and document gains compose multiplicatively when thought reads the
closing. A cold document has no document attenuation. Open grammatical
objects keep their expected prime and are spared attenuation. Deep
expectations have no domain outside the admitted region: there the prior, the
residual and the content are exactly zero (`Expectation.gain_field`), so no
omission is reported for a symbol attention never visited, while an expected
but absent symbol inside the region does count as an omission. A dimension
withheld by `non` has no prior at all: no expected prime, no attenuation and
no omission. The raw observation, prior and content remain available to
awareness even when the content gain is zero. An omission can support an
explicit checked `not` inference; it is never an external observation.

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
retain every key.
Checked bindings and scope come from
preceding source occurrences. They never copy the incoming target. Provenance
records those occurrences, the external stream/document and the intended
observation. The observation retains its own raw carrier and metadata. When
the durable row fuses several observed symbols, its
`_observed_symbol_activations` metadata retains the original two-lane target,
aligned by the prior's native IDs. Recomputing that target by quantizing the
fused description would change the measured residual. Checkpoint replay uses
the retained target, including negative evidence, rather than doing so.

`expectation_pair(gain=..., object_mask=...)` reconstructs the prior and derives
content, gain, expected activation, excess and omission. At a close the
image's object mask spares the roles of the field's actual open slots (an
open relation spares its relation slot), for every entry; it reads no
grammatical mode. There is no third durable delta or conceived row.
Restoration reads the exact stored observation, so it also works at zero
content gain. A new pair requires capacity for both rows and fails before
publishing either when forgetting is required. Reads resolve stable occurrences through the store index,
including after compaction; they do not scan unrelated rows.

For current pairs, surprise is the detached mean squared activation residual
over the soft union of observed and expected support: each symbol's weight is
`max(o, ê)` across its two lanes, and the weighted error is bounded by
`s/(1+s)`. Confident omissions count; empty bank padding does not dilute the
score. Both writers apply the same withheld marks and formula, independent
of content norm and gain. Local prediction
training uses bounded activation MSE. Unknown surprise is `-1`. An estimate
whose checkpoint lacks its symbol-bank prior is rejected at load.

The existing semantic sidecar fingerprints links with scope and source text,
checks both directions on restore, and participates in origin-compaction
reachability. A tensor-only restore leaves rows with missing required sidecar
metadata unavailable. No second memory owner is introduced.

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
its existing declared weight; neither enters the R+A trial judge.

Source encodings, observed targets, thought effects and durable history are
detached. Only the shared predictor receives local E gradients. Both the
whole-path keep rule and chooser credit use `R + A`; E is independent. This
includes thought departures, even when R and A disagree. Work remains metered
and reported, without an additional work-price term in the judge.

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
and task gradient, and an expected absence ($x = 0$) has residual $-e$. Even
with ordinary derivatives through both occurrences of $e$, the task gradients
to a predictor-only parameter would cancel, so writing the signal this way
trains nothing by itself; the cut makes the ownership boundary explicit.
Detaching only the add-back while subtracting a live $e$ would be a different
learning rule (a negative downstream gradient to expectation), and processing
the two parts by different functions changes the rule too, since
$f(p) + f(r) \ne f(p + r)$ for a nonlinear $f$. A later learner that should
learn only the error needs an explicit residual target; being fed
`expected + residual` does not give it one.

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
menu. Nothing self-generated is written into comprehension's content. Merging the two, as treating a preference as a
prior expectation does, produces the dark-room problem.

## Migration and regression dispositions

`architecture.prediction` is rejected as a retired setting. The old comprehension projection (`InterSentenceLayer.cast`/`prime`),
serial `_c_prior`, parallel seed slab, per-source residual fire flag and its
arm/fire APIs are gone. Old `cast.*` checkpoint tensors are discarded at the
owning discourse loader; there is no replacement input projection. Source-row
batch sizing now has an explicit host integer; sentence completion and SVO
reset retain their own existing owners.

The previous attention context receives zero-initialized columns for the bank
prime and the percept residual, and loses the retired structured forecast
slot. The expectation-free proposal view has a zero prime and a zero percept
residual, with only the open request and gist as needs.
Percept moments update only during
training on external inputs and are checkpointed; an
older checkpoint starts them cold. Old slot/ARMA head tensors migrate explicitly
to a fresh shared activation predictor, while retained observations and their
semantic sidecar survive. Transient pending graphs do
not survive checkpoint restore.

The relevant regression contracts are causal predictions, detached ownership,
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

All four gain strengths default to zero in both canonical configurations. A
nonzero operating point must first pass the native zero-gain parity gate; the
all-gains-.5 point is a candidate, not a default. The selection prime and the
exploration fraction $\varphi = .5$ are separate from these gains and stay enabled
(Alec).
