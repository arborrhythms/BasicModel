# Item 4 (formerly 4.5): expectation at every level — a gain field, surprise as the signal

*Draft rev 5, 2026-10-19 (Claude, from Alec's direction on 2026-10-08/09 and
his five decisions on 2026-10-19). Item 4 closed as a mechanism on 2026-10-20 (gates at item 0);
[the execution record](../plans/2026-10-19-item4-start.md) distinguishes
implementation, mechanism checks and open learning gates. §8 records the
decisions and the questions still open; §7's toys run before the hand-off
is written
([simulate before specifying](../plans/2026-10-05-operators-update.md#22-round-2e-hand-off-to-codex-claude-2026-10-06-rewritten-after-the-toy)).
Item 4.5 became item 4 on 2026-10-19 (todo); the file keeps its name.*

*Rev 5 changes (Alec, 2026-10-19):* the image is no longer a subtraction.
Expectation and attention are **one gain field** over a signal that is
purely attenuated, never negative (§1, §4.1, §4.3); the residual is a
second output in two lanes, carrying omission (§4.1). Expectation is
**per level with a local error**: percepts get a mean-and-variance
predictor whose normalized residual drives the mask and whose pooled value
attenuates the region's content as one scalar (§3, §4.5); concepts get a
top-down prediction of **activations over the symbol bank** from the
symbolic activation bank and STM, not a prediction of deep structure
inside conceptual space (§3); the unfolding by inversion (§3.1) and the
structured sentence predictor are withdrawn, and the invertible-path
program returns to item 3 (§3.1). **Expectation and desire** are two
signals with opposite aims and different supervision (§4.7). The fully
expected cat is still seen: awareness reads the prime and the attenuated
whole, learning reads the lanes (§4.8). Toys 3–4 leave with the inversion;
toys 7–8 are added (§7).

*Rev 2 changes (Alec's replies to rev 1):* percepts are not expected —
attention over percepts, expectation over concepts only (§3, §4; rev 1's
two-lane perceptual surprise withdrawn; rev 5 restores a two-lane
residual as a *channel*, not as content); exploration must be free, at
least to a degree, of expectation (§5.2); no full-derivation escape —
"rolling the dice multiple times and expecting all sixes" — and an
age-appropriate corpus is needed regardless (§5.3).

*Rev 3 changes (Alec's answers to rev 2's seven questions):* the top-down
route is 6.1's mask, wired as one layer over the current perceptual and
symbolic activation (§4.2); the departure's alternative is a mix that can
reach every alternative (§5.2); trials are judged by `R + A`, without `E`
(§5.2); the cost is read at the end of the sentence, and the Viterbi-like
principle Alec asked about is verified in a toy (§5.3); a graded set here
and the age-appropriate corpus before the final train (§5.4); rev 3's
"the compose round needs no predictor of its own and no content width"
(§3.1) is superseded by rev 5.

*Rev 4 changes (Alec, 2026-10-09, after the literature review in
[doc/research/reports/Expectation outside attention.md](../research/reports/Expectation%20outside%20attention.md)):*
attention attenuates expectation and signal together with one gain (§4.3,
now the gain field of rev 5); deep expectation is per region, absent
rather than attenuated outside it; the heterogeneity tolerance
`hetTolerance` (§4.4) is the answer to "is novelty a salience term", left
at 1; the shallow field-wide prior (§4.5, now decided in its minimal form)
and the novelty term it feeds; toys 5–6 added (§7).

## 0. Sources

- Alec, 2026-10-08 (the direction):

  > The plan is to use it as a predictor, but it should basically operate
  > per-level (perceptual prediction, conceptual prediction, etc) and be used
  > as a negation that reduces signal to surprise. Optimistically, it will
  > dovetail with the exploit/explore architecture, which will explore an
  > approximation of soft superposition at every level (although there may be
  > situations in which we need randomness and a full derivation to escape
  > local minima in the exploit surface. If we don't do that, we will need a
  > learning dataset that is gradual and avoids those multi-step minima).

- Alec, 2026-10-08 (replies to rev 1):

  > Perhaps we do not expect percepts, we just have areas of attention over
  > percepts, and expecting plays a role only over concepts? This does not
  > explain hallucination, but perhaps that can work in virtue of top-down
  > expectation?
  >
  > If we are arranging exploit to be a per-level exploitation, rather than a
  > full derivation (reasonable), then we need exploration that is free (at
  > least to some degree) of expectation.
  >
  > I still am not sure that explore needs a full derivation; it's like
  > rolling the dice multiple times and expecting all sixes. We need an
  > age-appropriate corpus regardless.

- Alec, 2026-10-19 (rev 5's direction, five messages):

  > I like the idea that percepts are attended and concepts are predicted
  > top-down. But we can still have expectation doing some of the heavy
  > lifting. So attention is guided by high-level strategy, but perceptual
  > content is not. A simple perceptual predictor, however, that minimally
  > removes mean and variance would be essential. "Top down" or "high-level
  > strategy" here means access to the bank of symbolic activations and the
  > short term memory bank. But the prediction does not have to be a
  > function from conceptual space to conceptual space: it can be a
  > top-down prediction of conceptual activations which influence the idea
  > that is built in conceptual space (thinking about cats may prime a
  > concept that makes a thought about cats more likely, but without the
  > complexity of predicting a syntactic deep structure within conceptual
  > space directly).
  >
  > I'm differentiating expectation (opposite sign, seeks homogeneity, can
  > train per-level since the error is known per level) from desire (same
  > sign, seeks to contribute its own conceptual content top-down).
  >
  > If we see a cat, and the cat is fully expected, then it will be as if
  > no cat is seen unless higher-order awareness has access to the
  > expectation.
  >
  > If we wish to attenuate what we predicted, we might take the difference
  > of actual and predicted as an attenuation factor: we attenuate where
  > there is little or no difference.
  >
  > Unification of attention and expectation under a gain field is the
  > right move. Since attention drives toward zero outside its area of
  > interest, and expectation also drives toward zero, we should be passing
  > a signal which is purely attenuated; in percepts the absolute value of
  > the difference can be used as a gain.

- The negative image: [accessible mind §2.6](2026-09-20-accessible-mind-subsystems.md#26-expectation).
- Expectation at every bracket: [item 6.8 plan §1](../plans/2026-09-27-item-6-8-one-attention.md).
- Expectation by inversion, the layer-wise comparison, the audit: operators
  plan [§10.3](../plans/2026-10-05-operators-update.md#103-the-invertible-forward-path-reconstruction-as-an-audit),
  [§12.2](../plans/2026-10-05-operators-update.md#122-two-complete-derivations-subtracted-layer-by-layer),
  [§12.4](../plans/2026-10-05-operators-update.md#124-spsa-and-the-layer-wise-comparison-that-replaces-it-alec-2026-10-06),
  [§13](../plans/2026-10-05-operators-update.md#13-the-scheme-as-confirmed-alec-2026-10-06);
  [GradientFlow, the target scheme](../GradientFlow.md#the-target-scheme-october-6-decided-scheduled-by-rounds);
  [FutureWork, expectation by inversion](../FutureWork.md#expectation-by-inversion-and-the-reconstruction-audit-decided-2026-10-06-unscheduled)
  (the record this spec replaces).
- Attention as a mask over `.where`: todo item 6.1;
  [stream-state plan §5](../plans/2026-10-08-stream-state.md).
- Gradual training: [FutureWork](../FutureWork.md#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29).

## 1. The idea

Perception is attended, not expected: over percepts there are only regions
of attention (the mask of item 6.1, `non` — selection, no signed carrier),
plus one content-free predictor of each coordinate's running mean and
variance, whose normalized residual says where to look (§3, §4.5).
Every **conceptual** level holds an expectation of what it is about to
receive, formed before it receives it from the symbolic activation bank
and STM — an expected **activation over the symbol bank**, not an expected
structure — and meets what arrives with a **gain**: what was predicted is
attenuated, what was not passes whole, nothing is ever subtracted and
nothing goes negative. Attention and expectation are the same kind of
operation, a gain field in `[0, 1]`, and compose by multiplication (§4.1,
§4.3). What a level passes up is therefore the whole, dimmed where it was
expected; a perfectly predicted concept passes on only the floor `1 − g`.
Beside the content runs a second output, the **residual in two lanes**,
which is what trains the predictor, sets retention, drives orienting and
carries omission (§4.1). The level keeps the whole observation `o` in the
store (pointer and tag, §4.8).

The per-level expectation is the **exploit**: at each level the greedy choice
is what the model expects of itself there, and the expected activation
primes the field the chooser and the mask read. Expectation is not
**desire**: expectation seeks homogeneity and trains locally on its own
residual; desire seeks to add content and trains only on the outcome
(§4.7). **Exploration** is a departure at one level, drawn and judged at
least partly free of expectation, so that what is learned is not only what
was already expected. A departure is one choice, completed greedily, never
a chain of random choices. Multi-step minima are not escaped by
randomness; the corpus is graded so that each improvement the model needs
is one departure away.

## 2. What exists today (verified against the working tree, 2026-10-08)

| level | owner | what is predicted | image | where the surprise goes |
|---|---|---|---|---|
| byte | `BracketExpectation` | declared, off | — | — |
| word | `WordExpectation.word_distribution`, its own ARMA(p, q) | categorical over the native candidate bank; teacher-forced in training | `ExpectedWords.negative_image` | the word surprise column, read only by the answer reader |
| compose round | — | nothing | — | — |
| sentence | `SentenceExpectation` (NP1/VP/NP2, presence, kind) | the next idea | `Meaning.negative_image`, `@no_grad`, concept face, at the closing | thought reads `c`; storage and the target keep `o`; residual → row `surprise` |
| row chain | `BracketExpectation` | declared, off | — | — |

Sources detached everywhere; only predictors train. **The trial** (`WalkTrials`,
`Models`): one departure per sentence; *where* is uniform (first among
eligible walks, then rounds — `departure_at`); *which alternative* is sampled
from the chooser's own policy with the greedy action excluded, at unit
temperature (`sample_eligible_logits`); the suffix is greedy; both trials are
costed at the owner step on `R + E + A`; strict-lower `R` keeps explore; the
score-function credit uses the total. The chooser reads the expectation image
as context (`Language` ≈5052).

So expectation already enters exploration in two places: the alternative is
drawn from a policy that reads the image, and the trial is judged partly by
`E`.

*Implementation status (2026-10-19, later):* Codex has an uncommitted
candidate in the working tree built from this revision
([execution record](../plans/2026-10-19-item4-start.md),
[toy and native receipts](../benchmarks/2026-10-19-item4-expectation/README.md)):
`Expectation.py` (the gain carrier of §4.1, the percept moments of row P,
one shared activation predictor for S/R/D, the `φ`-mix proposal law),
`SymbolField.py`, per-level gains and the new parameters in `model.xsd`.
Item 4 has since closed as a mechanism (step 0 round 2, 2026-10-20); Claude's reviews are in the
execution record's last section and in §8 below.

*Rev 5 note (verified 2026-10-19 at `9207f5e`):* `word_distribution` is
already a prediction of activations over a bank (the native candidate
bank), which is the form §3 generalizes to every order; `SentenceExpectation`
predicts a slot-structured idea, which §3 retires; the assimilative
`_c_prior` path that §2.6's spec scheduled for deletion is gone from
`bin/`; `hetTolerance` exists in `AttentionObjective`; the priming
surface's desired/hated intent channel exists in `Spaces` (§4.7).


## 3. The levels: a local error at each (rev 5)

Each level has its own predictor and its own residual, measured against
that level's own observation. No level's expectation is derived from
another's by inversion, so no level's error carries another's: the error is
bounded per level and per coordinate. Each predictor is autoregressive over
what has already arrived at its level (the bank, STM, the region), never
conditioned on its own prediction, so every level still predicts under zero
input.

| # | level | carrier | predictor (reads) | residual | the gain acts on |
|---|---|---|---|---|---|
| P | percept | `[0, 1]` cube coordinates, unsigned | running mean `μ` and variance `σ²` per coordinate; no content | `z = \|o − μ\| / (σ + ε)`, saturated to `[0, 1]`, per coordinate | `z` is the mask's salience input per coordinate; the region's content is attenuated by **one scalar** `ā`, `z` pooled over the attended region (§4.5) |
| S | symbol, every order (shared index) | activation × identity row; lanes `(c⁺, c⁻)` | **one top-down predictor**: the symbolic activation bank and STM → expected activation `ê` over the symbol bank, both lanes; `word_distribution` is its order-0 case | per symbol, two lanes `relu(o − ê)`, `relu(ê − o)` | one scalar per symbol on its activation; identity untouched |
| R | row (sentence closing) | the row's symbols | the same predictor read at the closing: the row's symbol activations against `ê` (the slot-structured `SentenceExpectation` retires) | per symbol in the row | the row's symbol activations as thought reads them; the store keeps `o` |
| D | row chain / document | rows in LTM | the same predictor read over STM for the next row; the declared `BracketExpectation` chain level is subsumed | per symbol of the next row | as R |

One predictor at S, R and D, read at three moments, is my consolidation of
Alec's "top-down prediction of conceptual activations" (§0); since orders
are fuzzy sets of symbols over one index
([Orders](../Orders.md)), its error is per symbol and is *reported* per
order. Whether the weights are shared across orders or one set per order
is an implementation choice for the hand-off (default: shared, per-order
reporting); it does not change the bounded-error property. What the
predictor reads is the primed field — the spreading activation of the
priming surface and the conceptual STM — which is exactly what 6.1's mask
layer reads, so expectation and attention share an input as well as a
mechanism.

The compose round has **no expectation of its own and none derived**. A
round combines symbols whose activations already carry the per-symbol gain
of S; the round's product is a symbol of a higher order whose gain is read
when it enters the bank. The departure (§5) is a choice of the chooser at a
round and is unaffected.

### 3.1 Withdrawn: the unfolding by inversion (rev 3) — moved to item 3

Rev 3 derived each round's expectation by unfolding the expected idea down
the actual tree given the actual co-operand, and the word's expectation
from it. That couples the levels: one wrong estimate at the top becomes a
wrong expected operand at every round and a wrong expected leaf — the
error of every level aggregated into the lowest. Rev 5 withdraws it. With
it leave the derived round expectation, `κ = 0` on undefined inverses, and
the content-width question, none of which per-level expectation needs.

The invertible-path program — exact inverted targets for the maps,
expectation's sources going live, reconstruction retiring to an audit
where the inverse is exact (operators plan §10.3, §12.2, §13; GradientFlow's
target scheme) — did not depend on expectation and expectation no longer
depends on it. It returns to **item 3** with the width `D ≈ c·k·log N` and
the audit, and item 0's condition reads it from there. Here, the
predictors' sources stay **detached** everywhere, as on 2026-09-20: only
the predictors train on the residual; nothing upstream is trained to be
predictable.

The byte level retires from `BracketExpectation` (declared, never on; no
legacy paths are kept); the perceptual predictor of §3 row P is not a byte
predictor but a per-coordinate statistic. The perceptual rung's *both*
corner (the prompt to divide a bracket) remains attention's evidence, as in
6.8.

## 4. The gain field, and where hallucination comes from

### 4.1 The gain at every level, and the residual beside it (decided, Alec, 2026-10-19)

The image of §2.6.1 (`n = −g·(1−m)⊙κ⊙ê`, `c = o + n`) is **withdrawn** as the
form of expectation's effect on the signal. In its place, two outputs per
level:

    content:   c = m ⊙ a ⊙ o,     a = (1 − g_L) + g_L · s(r),   r = o − ê
    residual:  (r⁺, r⁻) = (relu(o − ê), relu(ê − o))

where `m` is attention's gain (6.1's region, exact containment, §4.3),
`a ∈ [1 − g_L, 1]` is expectation's gain, `s(r) ∈ [0, 1]` is a saturating
measure of how far the arrival is from the prediction (for percepts the
normalized `z` of §3; for symbols the lane residual relative to the
expected magnitude, `s = min(1, (r⁺ + r⁻) / (‖ê‖ + ε))` as the default to be
tuned in toy 7), `g_L` is the level's gain in model.xml (0 = beginner's
mind, nothing attenuated; 1 = the fully predicted passes nothing), and the
object mask `m_obj` of §2.6 (the open roles of the active question are
spared) multiplies into `a` as before. Where the predictor has no
confidence (`κ = 0`) `s = 1`: nothing is attenuated.

Properties, each of which the subtraction lacked:

- **Never negative, never off the chart.** A gain keeps every coordinate in
  its chart: the `[0, 1]` percept cube, the positive lanes. The order-0
  residue of 2026-09-28 (reflecting a point through the origin leaves the
  cube) disappears, and §2.6.2 item 1 — *sensation is never subtracted
  from* — holds for percepts and concepts alike: attenuated, not
  subtracted from.
- **One mechanism with attention.** Attention drives the signal to zero
  outside its region; expectation drives it toward the floor where it was
  predicted. Both are gains in `[0, 1]` and compose by multiplication
  (§4.3). Expectation's effect on the signal is therefore a
  **non-affirming exclusion**, the same kind as attention's. Philosophy's
  classification (expectation = `not`, attention = `non`) moves: `not`
  lives only in the against lane `c⁻` (and in thought); Alec's 2026-09-20
  statement 1, prediction as sign reversal, is withdrawn with it.
- **Omission has a channel.** A gain on `o` cannot show what failed to
  arrive (`o = 0` times anything is 0). The residual carries it: the
  expected-but-absent symbol has `r⁻ = ê` — expected, not seen — which is
  the *against*-lane form of conceived absence (dṛśyānupalabdhi, §2.6.8),
  and for percepts `z = μ / (σ + ε)` opens the mask on the dark spot, where
  the content is darkness. The residual is what trains the predictor (delta
  rule, §2.6.4), sets retention priority (the `surprise` column), feeds
  orienting (§4.4, §4.5) and conceives absence; the content never does.
- **The dark-room guard changes form, not substance.** The predictor
  receives gradient only through the residual, never through `a`:
  otherwise it could mispredict to brighten the signal or over-predict to
  silence it. One `detach` on `s(r)`'s input; the purity invariant of
  §2.6 (for fixed input, `o`, the codes and the derivation are independent
  of `ê` and `g`) is unchanged.
- **Attenuation is per symbol, never per feature.** A per-coordinate gain
  on a percept vector changes its direction, not only its magnitude: a cat
  whose everything but whiskers was predicted becomes a whiskers vector,
  and the codebook snaps to whiskers. Identification needs the whole. So
  the per-coordinate `z` goes to the mask ("attend to the novel part"),
  and content inside a region is attenuated by one scalar; at S the gain
  is one scalar per symbol's activation (symbol = activation × identity
  row) and the identity is untouched (toy 8).

Physiology is the check that this is the wanted form: expectation
suppression and repetition suppression are reductions in rate (divisive),
not subtractions into negative firing; retinal and cortical contrast gain
control divides by the local mean (Weber's law), which is row P's `z`; and
at `g = 1` with a slow mean predictor a perfectly stable percept passes
nothing, which is what stabilized retinal images do — they fade within
seconds (Riggs 1953; Ditchburn & Ginsborg 1952), and the eye refreshes
them by microsaccades. The floor `1 − g_L` is the knob that says how much
of that we want.

### 4.2 Hallucination by top-down expectation

With no expectation of perceptual *content* (row P predicts a mean and a
variance, never a percept, rev 5), a hallucination cannot be a percept
predicted into existence. It is a **concept conceived without perceptual
evidence**: the estimate entering comprehension with the sign it has in
production (§2.6.3, reafference — positive in `<generate>`; in comprehension, since rev 5, not a negative sign but the gain: the prime is positive into selection and what was predicted is attenuated; the old wording "negative in
comprehension). The two directions already exist; hallucination is the
production sign where the comprehension sign belongs, or an estimate whose
`for` lane is filled where attention found nothing. Imagery and dreaming are
the same path run on purpose: `<generate>` unfolding an expected idea down
to the percepts.

That gives top-down expectation one legitimate route into perception, and
it is **attention, not percepts**: an expected concept, inverted to its form,
can bias where the mask opens ("look for what you expect" — biased
competition; Alec, 2026-09-20 statement 2: the expected is recognized more
quickly). The mask still only selects (`non`); it adds nothing. A
misdirected mask is the model's form of perceptual set, and its errors are
mis-*attended* percepts, not invented ones.

**Decided (Alec, 2026-10-08):** the route is item 6.1's mask, wired as **a
single layer** whose input is the current perceptual and symbolic activation
— the spreading activation of the priming surface. "That is sufficiently top
down for now." 4.5 builds nothing here; what is expected reaches the mask
through the symbolic activation it already reads.

### 4.3 One gain over signal and expectation (decided, Alec, 2026-10-09; the gain field, 2026-10-19)

Attention attenuates both the signal and the expectation over the
unattended field, with one gain: the region weight of 6.1's mask multiplies
the whole. Rev 4 wrote this as `c = g·(o + n)`, never `g·o + n` — with the
signal attenuated and the image not, the unattended field fills with
omission surprise and the mask hallucinates absences. Rev 5 makes the rule
exact by construction: expectation is itself a gain, `c = m ⊙ a ⊙ o`
(§4.1), so outside the region (`m = 0`) nothing passes and nothing is
"expected but not seen" in the content; omission lives only in the
residual, which the mask reads as salience.

The deep expectations — the per-level predictions of §3 — are **per
region**: realised inside a region, and absent, not attenuated, outside
it. The predictors read the regions, not a lag window (the stream-state
plan's stage 6, "expectation from the right history"; today
`word_distribution` and `SentenceExpectation` condition on ARMA and
`context_window` lags).

What 6.1 has built is this rule's implementation: region membership is exact
containment in the forward (zero outside; the reference bank and the
conceptual STM are masked by it, and the in-STM predictor reads the masked
STM), while the attention MLP reads the whole field's spreading activation
unattenuated. Comprehension gain 0 outside the regions, placement gain 1:
the literature's two tiers (the report's levels (a) and (b)) as a hard split.
A floor on the comprehension path is wrong — unadmitted items would leak
into compose and the budget would go. (Expectation's floor `1 − g_L` of
§4.1 is a different thing: it applies inside the region, to admitted
content.) A parameter attenuating the MLP's field input under focused
attention (Woldorff 1991; Molloy 2015) is a later refinement, not 6.1's.

### 4.4 `hetTolerance`: the answer to "is novelty a salience term" (decided, Alec, 2026-10-09)

Alec: the question gets a parameter for an answer. The mind tends to make
reality non-contradictory; the tetralemma becomes Boolean at some degree of
intolerance for heterogeneity. The dial is `hetTolerance = τ`, applied to
the evidence lanes at the symbolic level (percepts are unsigned; the pair
exists only for symbols and meanings, which is also where priming lives):

    (c⁺, c⁻) ← (c⁺, c⁻) − (1 − τ) · min(c⁺, c⁻).

At τ = 1 (the default, left in for now) nothing changes. At τ = 0 the shared
overlap is removed: one lane is zero, the other holds `|c⁺ − c⁻|`; *both*
collapses, *neither* (0, 0) survives, since absence of evidence cannot be
netted into presence — three corners, for/against exclusive and the unknown.
No signed scalar appears at any setting; both lanes stay positive
(the two-lane rule, [operator catalogue, Lanes](2026-09-29-operator-catalogue.md)). It is one op where the lanes are
read into the mask input; it may be declared with 6.1 or here.

It is also the capture dial: the held *both* corner — the expected beside
the arrived — is the surprise that orienting reads (report, level (c));
τ → 0 is the rationalising mind, set-match wins and nothing captures.

### 4.5 The shallow prior: decided in its minimal form (Alec, 2026-10-19); the novelty term still proposed

The report's finding: prediction runs outside attention only for low-level,
located, first-order regularities — where, when, form, the transition from
the previous item — and attention scales that residual as a gain; identity
and next-item expectation leave no error signature outside attention
(Richter & de Lange 2019, preregistered, BF10 0.18–0.25; Bekinschtein 2009's
global rule; lexical tracking only for the attended talker). Orienting is a
third system, driven by violation of a learned prior, not by unfamiliarity
(Vachon, Hughes & Jones 2012: no capture before a rule exists, d < 0.11;
capture at its first violation, d ≈ 0.9–1.4; habituating), subordinate to
set-match and stochastic.

**Decided (Alec, 2026-10-19), the minimal form:** the shallow field-wide
prior is row P of §3 — a running mean and variance per percept coordinate,
computed wherever the field is active, not per region; no content, no
transition structure. Its normalized residual `z = |o − μ| / (σ + ε)`,
saturated, is the mask's salience input per coordinate, and pooled over an
attended region it is the scalar by which that region's content is
attenuated (§4.1). The division by `σ` is not optional: a raw `|o − μ|`
passes a noisy coordinate at high gain forever, since noise is always
different, and the normalized form is what contrast gain control does.
`ε` and the saturation are model.xml parameters. **Measured limit (Codex,
toy 6, 2026-10-19):** normalization does *not* reject the noisy TV — a
correctly predicted mean and variance still give a normalized residual of
about 1 for Bernoulli noise, so an irreducibly noisy coordinate stays at
full salience; and marginal statistics cannot see a change that is only
in transition order. Rev 5's sentence claiming otherwise is withdrawn.
What the minimal statistic buys is the floor and the cold start; the
noisy TV is answered only by the belief-shift novelty term below, which
remains proposed.

**Still proposed:** the richer transition prior (form, class and position
from the previous item) and the **novelty term** as that residual's belief
shift (prediction gain, not raw residual energy — persistent error without
belief change is not surprise: the snow paradox), fed into region placement
a step behind onset, subordinate to priming and set-match, cancelled when
the deviation itself becomes regular. Low priming alone is not a capture
signal; low priming plus a violated shallow prior is. Toys 5–6 remain.

### 4.6 Non-affirming negation withholds a dimension from the prior (Alec, 2026-10-15)

`non` on a modifier (*a non-heavy paper weight*) clears both evidence lanes
on every aspect that bears on the modified dimension **and blocks this
level's prior on that dimension**: the kind's expectation (paper weights
are typically heavy) must not fill what the sentence withheld. The row
therefore carries a per-dimension *withheld* mark beside the two lanes;
without it *non-heavy paper weight* and *paper weight* are the same row
and the prior would hallucinate *heavy* back. The mark is content the
inverse reads (it re-emits *non-heavy*) and the forward discard charge
counts as retained, not discarded. `non` applies to modifiers only — on a
noun it erases the thing — and its position is learned by exposure, not
declared ([Philosophy](../Philosophy.md#the-two-negations-on-a-modifier-not-heavy-and-non-heavy-alec-2026-10-15)).
Gate: a corpus with negated modifiers; the expected operand at the
withheld dimension is zero, not the prior; *non-X* is reconstructed.
(Rev 5: "blocks the prior" means no prime and no attenuation on that
dimension — `s = 1`, nothing expected there — not a subtracted zero.)

### 4.7 Expectation and desire (Alec, 2026-10-19)

Two top-down signals read the same primed field and must not be one
vector:

| | expectation | desire |
|---|---|---|
| aim | homogeneity: attenuate what was predicted | contribute content: make the wanted more likely |
| sign | against the arrival (a gain toward the floor) | with the arrival (a prime) |
| error | local, known per level: the residual | none locally; only the outcome, the judge's `R + A` at the end of the sentence (§5.2) |
| trains | the predictor, per level, delta rule | the chooser's policy, score-function credit |
| writes to | the gain `a` on content; the residual lanes | selection: the chooser's logits and 6.1's mask (`non`, floor 0, never a veto) — in comprehension; content only in production |
| home today | `word_distribution`, `SentenceExpectation` → §3's predictor | the priming surface's desired/hated intent channel (`Spaces`) and the chooser |

Psychologically these are Berridge and Robinson's *wanting* against
*learning*, and Schultz's reward prediction error is the desire-side
outcome minus the expectation-side forecast; Sokolov's habituation is the
expectation side alone. Active inference merges the two by making
preference a prior expectation, and the dark-room problem is what that
merger costs; this spec keeps them apart, which is also the structural
separation of `R`, `E` and `O` over weights asked for on 2026-10-02.

**The tension and its resolution.** If desire wrote excitation into the
content vector that expectation reads, expectation would learn to predict
the self-generated excitation and attenuate it — the reafference principle
(von Holst & Mittelstaedt 1950; one cannot tickle oneself, Blakemore,
Wolpert & Frith 1998). The resolution is a separation of what each writes
to, not an order of operations: expectation reads the primed field and
writes the gain; desire reads the primed field and writes to **selection**;
neither writes into the content vector during comprehension, so there is
nothing to re-prime or cancel and the stored row stays honest. Desire
contributes **content only in production**: the thought loop builds from
the primed field with the positive sign (§2.6.3's reafference sign). When
that product re-enters comprehension, expectation attenuates it as
self-generated — wanted: it is what keeps imagined cats from being stored
as seen cats (reality monitoring, Johnson & Raye 1981). The limit is
deliberate: desire can make a cat thought more likely to be built, and a
cat recognized faster, but cannot make a cat appear in a field where
attention found none — otherwise it is §4.2's hallucination. The
assimilative `_c_prior` path (predicted end state added into composition)
stays deleted; the prime goes to the field the chooser and mask read, not
into composition.

### 4.8 The fully expected cat: higher-order awareness has the expectation (Alec, 2026-10-19)

Alec: "if we see a cat, and the cat is fully expected, then it will be as
if no cat is seen unless higher-order awareness has access to the
expectation." It has it in three ways, the third an amendment:

1. **The store, and the predictor.** The closing keeps `o`; the estimate
   is retained by the predictor as its last estimate per address (the row
   amendment of 2026-10-20 moves the estimate keys out of the row:
   [two truths §1, "The row, field by field"](2026-09-16-two-truths-ideas-and-relations.md#1-definitions-decided));
   the pair of estimate and observation is still readable
   ([ExpectationRetention](../ExpectationRetention.md)), so the residual is
   a change of origin, not a loss.
2. **The primed field.** The expected activation is written positively into
   the field the chooser and mask read (§4.7), so the next level receives
   the cat twice: as the prime, top-down and present before the cat
   arrived, and as the content, bottom-up, attenuated where the prime
   already had it. The whole is prime plus content through two channels
   with different gains and different learning. A fully expected cat
   arrives through the first channel: seen, as an expectation confirmed.
3. **The consumers (amendment to §2.6's list, "thought reads `c`").** Under
   the gain the content is never zero for `g_L < 1`, so a consumer that
   reads only `c` still sees the dimmed cat. Where thought needs content
   it reads the attenuated whole (and the estimate row when it asks what
   was expected); where it needs novelty — salience, retention priority,
   orienting, the predictor's training — it reads the residual lanes.
   Alec's 2026-09-20 statement 1 ("for a perfect prediction we then
   conceive of nothing") is amended to *nothing new*: the cat was conceived
   in anticipation.

This is the predictive-coding answer and the evidence matches: what is
perceived is the prediction confirmed, not the error; expected stimuli
evoke reduced but sharper responses, not absent ones (Kok et al. 2012;
Summerfield & Egner 2009); self-touch is felt dampened, not unfelt. The
worry is a real phenomenon at the limit — the fully expected drops from
report (driving on autopilot; inattentional blindness, Mack & Rock 1998) —
and what makes it retrievable afterwards is that the expectation was held
somewhere accessible. Press, Kok & Yon's "perceptual prediction paradox"
frames the pairing exactly: *sharpening* (the expected enhanced, here the
prime in the field) and *dampening* (the expected attenuated, here the
gain), both present.

At `g_L = 0` there is no attenuation and everything arrives as content; at
`g_L = 1` with a perfect predictor everything arrives as prime. Awareness
is the same cat either way; what is learned, retained and attended is not.

## 5. Explore and exploit

### 5.1 Exploit per level

At each conceptual level the greedy choice, made with that level's
expectation priming the field it reads, is the exploit. Its residual trains
that level's predictor (delta rule, §2.6.4) and nothing else: sources stay
detached (§3.1). Reconstruction stays a loss; its retirement to an audit is
item 3's.

### 5.2 Exploration free of expectation

Expectation can enter a departure at three points; each can be made free
of it separately:

| point | today | free of expectation |
|---|---|---|
| where to depart (level, round) | uniform | already free |
| which alternative | the chooser's policy, greedy excluded, which reads the image | uniform among eligible alternatives, or the policy read at `g = 0` (beginner's mind, §2.6.6) |
| how the two trials are judged | `R + E + A` | `R + A` only — evidence and outcome, not predictability |

**Decided (Alec, 2026-10-08):** the alternative is drawn from a **mix** —
a fraction `φ > 0` uniformly over the eligible non-greedy alternatives, the
rest from the policy at `g = 0` — so that every alternative can sometimes be
hit (probability at least `φ / (K − 1)`, whatever the policy has learned).
Trials are judged by **`R + A`**, not `E`: "a clean line". `φ` is a
model.xml parameter; a static check asserts `φ > 0`.

The third is the one that matters most. If a departure is judged by its
surprise, the chooser learns to make the world predictable — the dark room
(§2.6.9), which the guard there forbids for anything that decides what is
conceived. Judged by reconstruction and the answer, a departure is rewarded
for explaining the input or getting the outcome right, and expectation is
left to train only the predictors. The middle point is a degree: uniform is
fully free and wastes draws as the policy sharpens; `g = 0` keeps the
learned policy and removes only the image.

### 5.3 One departure, one die, read at the end: the Viterbi-like principle

Alec's point: needing several coordinated random choices is rolling for all
sixes. So no chained random derivation: a departure is **one** choice at
**one** level, everything after it greedy, and the cost (`R + A`) is read at
**the end of the sentence** (decided, Alec, 2026-10-08), so a choice can learn
from the answer. Rev 1's "global departure" at a rate ε is withdrawn.

**The principle (Alec asked to verify it).** Reading one departure plus a
greedy completion at the end measures `Q(s, a)` — the cost of choice `a` in
state `s` followed by the completion. That is the *right* measure,
Bellman's `Q(s, a) = V*(s')`, exactly when the greedy completion from `s'`
is already optimal. Two consequences:

1. **The last step is always credited exactly**: its completion is empty.
   So the last round learns first, without help.
2. **Each earlier step is credited exactly once every step after it is
   optimal** — backward induction, Viterbi's principle of optimality. If the
   data supplies the problems in that order — the last step alone, then the
   last two, … — every new step is learned against an optimal completion,
   and no local minimum of the exploit surface is ever entered. Without that
   order, completions *off* the greedy path are never learned (a single
   departure reaches them only to be completed by an untrained policy), so
   early departures are credited against bad suffixes and the policy settles
   where it is.

**Verified in a toy** ([2026-10-08 Viterbi departure toy](../benchmarks/2026-10-08-viterbi-departure-toy/README.md)):
a tabular chooser, depth 5, branching 3, random terminal costs, 200
instances, one departure (mixed alternative) credited at the end. Fraction
ending on the optimal derivation: **root-only 0.04** (unchanged at 3.3× the
budget), **graded "one more layer at a time" 0.96**, the same subproblems
unordered 0.32–0.43. The order matters, not only the content. Limits: no
generalization across states (the model's chooser shares parameters), Markov
state, terminal cost only.

What "one more layer" means for the model: a derivation's *last* round is
its closing operation, so the graded set adds rounds at the **bottom** —
first items whose derivation is one operation over already-known
constituents, then items one operation deeper, each new item's upper rounds
already met. In language, that is phrases before clauses before sentences:
the age-appropriate order.

### 5.4 The corpus (decided, Alec, 2026-10-08: "a bit of this with 4.5, and a bit before the final train")

- **In item 4 (then 4.5):** a small graded set built for the gate — the standing fixtures'
  vocabulary arranged so each item requires one more round than the items
  before it, read in that order and also shuffled. The gate is the difference
  (§9). This is the Viterbi principle measured in the model, not only the toy.
- **Before the final train:** the age-appropriate corpus
  ([gradual training](../FutureWork.md#gradual-training-an-age-ordered-corpus-read-in-stages-proposed-2026-09-29):
  Wordbank → `interpret` → AO-CHILDES by age band) goes into todo item 3
  (corpus at target size), so item 0 reads it first.
- **The stall diagnostic** stays: per level, the explore-win rate at zero
  while the level's surprise stays high says the data step is too steep
  there.

### 5.5 Is a tunable softmax enough? (Alec, 2026-10-08) — no; it lives inside the departure

Alec asked whether, with random sampling, an explicit explore is needed, or
a softmax with a tunable temperature `τ` can balance explore and exploit by
itself. Measured in the same toy
([follow-up](../benchmarks/2026-10-08-viterbi-departure-toy/README.md#follow-up-does-a-tunable-softmax-replace-the-departure-alec-2026-10-08)),
with the softmax given twice the episodes so compute matches: on the graded
order the departure reaches the optimal derivation in **0.98** of instances;
the softmax, at its best `τ` and learning rate, **0.54–0.66**; root-only, at
most 0.12.

The explicit departure is not about randomness, which the softmax supplies
equally well. It is about the *comparison*: two derivations of the same
input that differ in exactly one choice, the greedy one as the baseline,
and a greedy (optimal, once learned) finish. That is what makes the credit
exact under the Viterbi condition. Sampling every choice credits all of them
from one shared cost and measures the value of a noisy finish.

So the two are not alternatives. **The tunable softmax is the departure's
alternative draw**: `τ` (with the uniform floor `φ`) sets how far from the
greedy choice a departure looks, and the paired trial does the credit. The
only randomness is that one draw, plus where the departure lands. `τ` joins
`φ` as a model.xml parameter.

## 6. What it touches (for the hand-off, after §8)

- `Meaning.negative_image` → the gain of §4.1 (content) plus the two-lane
  residual, per level; `expectation_surprise` per level reads the lanes;
  `g_L`, `ε`, the saturation and `hetTolerance` in model.xml; one `detach`
  on the gain's input.
- Row P: a running mean and variance per percept coordinate; `z` into
  6.1's mask layer as a per-coordinate input; `ā` pooled per region as the
  region's content gain. The codebook never sees a per-coordinate gain.
- Row S: `word_distribution` generalized to the symbol bank at every
  order, reading the symbolic activation bank and the conceptual STM, both
  lanes; per-symbol residual, reported per order. Its output written into
  the primed field (the prime), not into composition.
- Rows R and D: `SentenceExpectation`'s slot-structured prediction retires;
  the row's and the next row's expectation are the S predictor read at the
  closing and over STM. `BracketExpectation` removed (byte and chain).
- The trial: unchanged from rev 3 (departure level uniform; the `φ`-mix;
  judged by `R + A` at the end of the sentence; per-level records).
- Desire (§4.7): no new mechanism; the desired/hated channel and the
  chooser are its home; the audit is that nothing of it writes into
  content during comprehension.
- The graded set (§5.4) and its shuffled control.
- Docs: spec §2.6 (image → gain; the residual channel; percepts attended,
  with the mean/variance statistic; consumers amended per §4.8);
  Philosophy's `not`/`non` classification of expectation (§4.1);
  Architecture "Surprise at every layer"; FutureWork's expectation-by-
  inversion section pointed at item 3; operators plan §12.2 and §13
  marked: expectation detached from the invertible path; todo item 3
  takes the inversion program.

## 7. Toys before the hand-off

1. **Done:** one departure read at the end, graded vs root-only vs unordered
   (§5.3).
2. **Judging with and without `E`.** A two-option chooser where one option
   explains the input and the other is more predictable; judged by
   `R + E + A` and by `R + A`. Records the dark-room drift that the decision
   avoids.
3. *Withdrawn with §3.1* (partial unfolding) — item 3.
4. *Withdrawn with §3.1* (inverted targets vs detached sources) — item 3.
5. **The shallow prior and capture** (§4.5): without the field-wide
   statistic, unexpected tokens outside the regions are never admitted and
   region placement collapses onto early winners; with it, capture appears
   only after a regularity is learned, habituates when the deviation
   becomes regular, returns when the higher-order rule breaks — the Vachon
   signature.
6. **Residual energy vs prediction gain** as the novelty quantity: only the
   latter stops chasing an irreducibly noisy token stream.
7. **Gain vs subtraction** (§4.1). A stream mixing a predictable and an
   unpredictable component, one predictor, the two forms of the image.
   Measured: whether the predictor or a chooser downstream can exploit the
   gain (with and without the `detach`); whether omission survives in the
   against lane; the content's chart (no negative coordinates) ; the
   choice of `s(r)` and the floor. The subtraction is the control.
8. **Per-coordinate vs scalar attenuation before the codebook** (§4.1).
   Percepts whose predicted part is large: the codebook's identification
   under a per-coordinate gain against a scalar region gain. Expected:
   identity preserved only under the scalar.

## 8. Decisions and open questions

Decided 2026-10-08 (Alec): percepts are attended, not expected; the top-down
route is 6.1's single-layer mask over perceptual and symbolic activation;
the alternative is a `φ`-mix able to reach every alternative; trials judged
by `R + A`; the cost read at the end of the sentence; a graded set here,
the age-appropriate corpus before the final train (item 3). Rev 3's
"the compose round derives its expectation; the width stays in item 3" is
superseded by rev 5 (no derivation at all; the inversion program moves to
item 3).

Decided (Alec, 2026-10-08, rev 3's two questions): **one departure per
sentence**, its level drawn uniformly among the conceptual levels (two
departures read at one end-of-sentence cost would confound each other's
credit); the graded order is **one more round per item, bottom-up** —
phrases, then clauses, then sentences.

Decided (Alec, 2026-10-09): one gain over signal and expectation; deep
expectation per region, absent outside it; 6.1's exact-containment mask
with the MLP reading the whole field is the implementation (§4.3);
`hetTolerance` τ on the lanes, left at 1, as the answer to
novelty-as-salience (§4.4).

Decided (Alec, 2026-10-19, rev 5): expectation is **per level with a local
error**, never derived across levels (§3, §3.1); percepts are attended, with
a **mean-and-variance** predictor only (§3 row P, §4.5); concepts get a
**top-down prediction of activations** over the symbol bank from the
symbolic activation bank and STM, not of deep structure (§3); expectation
and **desire** are two signals — opposite aims, local vs outcome error —
and desire writes to selection, not content, in comprehension (§4.7);
higher-order awareness has the expectation through the store, the prime
and the amended consumers (§4.8); expectation acts as an **attenuation
factor from the residual**, and attention and expectation are **one gain
field** over a purely attenuated signal (§4.1, §4.3); for percepts the
absolute difference is the gain's source, **normalized by the predicted
variance and applied per coordinate to the mask, per region to content**
(verified and corrected, §4.1, §4.5).

Decided (Alec, 2026-10-19, evening, on Codex's candidate):
- **Gain defaults: `g_L = 0` at every level** until the §9 zero-gain
  parity gate has run; Codex's 0.5 is withdrawn from the training config
  and becomes a measured setting after the gate, never a default written
  first.
- **The keep rule reads `R + A`**, the same judge as the chooser's credit
  (§5.2's "judged by `R + A`" covers both; §2's "strict-lower `R` keeps
  explore" is the pre-rev-5 state).
- **The open role keeps its prime and is spared attenuation** (§4.1,
  §4.2); Codex's withholding of the prime on a question's open role is
  reversed. Only `non`-withheld dimensions lose the prime (§4.6).
- **The `non` operator change is taken with item 4**: `NonLayer` clears
  both evidence lanes and writes the withheld mark, as §4.6 says, covered
  by the withheld-mark regression.

Taken from Codex's candidate, implementation choice recorded: the row and
document gains compose multiplicatively when thought reads the closing
(one predictor read at two moments, two gains on one row's symbols).

Decided (Alec, 2026-10-19, late, B3): **the selection prime stays on and
`φ = 0.5` stays as the shipped default**, independently of the four zero
attenuation gains. The sole declared §9.1 parity configuration has all four
gains zero, selection prime off and the previous uniform alternative law
(`φ = 1`); prediction and its residual/store cycle stay enabled. It is used
only for the parity gate. Measure the enabled shipped behaviour against it
before the landing sweep; the earlier disabled-expectation comparison does
not certify this configuration. The declaration is
[`test/fixtures/item4-parity.json`](../../test/fixtures/item4-parity.json).

Taken as my consolidation, for Alec to object to: one predictor at S read
at three moments (§3); the default `s(r)` and the floor `1 − g_L` (§4.1);
`SentenceExpectation`'s structured prediction retiring rather than running
alongside (§3).

Measured, not decided: a tunable softmax does not replace the departure; it
becomes the departure's alternative draw (§5.5). **Open:** the richer
transition prior and the novelty term as belief shift (§4.5, toys 5–6);
whether S's weights are shared across orders (§3); §7's toys 2, 7 and 8
before the hand-off.

## 9. Gates (proposed)

- The standing XOR fixtures and the math chain unchanged or better with every
  level on; each level's `g_L = 0` reproduces the previous landing.
- Per level: the residual falling on the standing fixtures; no negative
  coordinate anywhere in content (an assertion, not a measurement); the
  collapse diagnostic flat; explore-win rate and the stall diagnostic
  reported.
- **The fully expected cat:** a sentence repeated until its symbols are
  predicted at confidence passes content at the floor and is still answered
  and stored correctly (§4.8).
- **Omission:** a symbol expected at confidence and then absent shows in
  the against lane of the residual and nowhere in content (§4.1).
- **Identity under attenuation:** the codebook's identification of a
  percept is unchanged by the region gain at every `g_P` (§4.1, toy 8).
- The graded set read in order beats the same set shuffled on the
  derivation-optimality rate (the toy's measure in the model); reported, with
  the shuffled run as the control.
