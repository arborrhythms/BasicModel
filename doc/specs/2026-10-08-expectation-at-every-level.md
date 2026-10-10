# Item 4.5: expectation at every conceptual level — prediction as negation, surprise as the signal

*Draft rev 3, 2026-10-08 (Claude, from Alec's direction and his two rounds
of replies the same day). Nothing here is built. §8 records the decisions and
the questions still open; §7's toys run before the hand-off is written
([simulate before specifying](../plans/2026-10-05-operators-update.md#22-round-2e-hand-off-to-codex-claude-2026-10-06-rewritten-after-the-toy)).*

*Rev 2 changes (Alec's replies to rev 1):* percepts are not expected —
attention over percepts, expectation over concepts only (§3, §4; rev 1's
two-lane perceptual surprise withdrawn); exploration must be free, at least
to a degree, of expectation (§5.2); no full-derivation escape — "rolling the
dice multiple times and expecting all sixes" — and an age-appropriate corpus
is needed regardless (§5.3).

*Rev 3 changes (Alec's answers to rev 2's seven questions):* the top-down
route is 6.1's mask, wired as one layer over the current perceptual and
symbolic activation (§4.2); the departure's alternative is a mix that can
reach every alternative (§5.2); trials are judged by `R + A`, without `E`
(§5.2); the cost is read at the end of the sentence, and the Viterbi-like
principle Alec asked about is verified in a toy (§5.3); a graded set in 4.5
and the age-appropriate corpus before the final train (§5.4); the compose
round needs no predictor of its own and 4.5 needs no content width (§3.1).

*Rev 4 changes (Alec, 2026-10-09, after the literature review in
[doc/research/reports/Expectation outside attention.md](../research/reports/Expectation%20outside%20attention.md)):*
attention attenuates expectation and signal together — the region weight
multiplies the sum, `c = g·(o + n)` (§4.3); deep expectation is per region,
absent rather than attenuated outside it; the heterogeneity tolerance
`hetTolerance` (§4.4) is the answer to "is novelty a salience term", left at 1;
the shallow field-wide transition prior and the novelty term it feeds are
proposed, not decided (§4.5); toys 5–6 added (§7).

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
of attention (the mask of item 6.1, `non` — selection, no signed carrier).
Every **conceptual** level holds an expectation of what it is about to
receive, formed before it receives it, and meets what arrives from the
expected point — the negative image, so what the level passes on is
**surprise**, and a perfectly predicted concept passes on nothing. The level
keeps the whole observation (`o = c − n`, a change of origin, §2.6.1).

The per-level expectation is the **exploit**: at each level the greedy choice
is what the model expects of itself there. **Exploration** is a departure at
one level, drawn and judged at least partly free of expectation, so that what
is learned is not only what was already expected. A departure is one choice,
completed greedily, never a chain of random choices. Multi-step minima are
not escaped by randomness; the corpus is graded so that each improvement
the model needs is one departure away.

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

## 3. The levels: conceptual only

| # | level | carrier | the next thing expected | from below (next step) | from above (prior, by inversion) |
|---|---|---|---|---|---|
| C0 | word (order-0 concept) | code + lanes `(c⁺, c⁻)`, magnitude | the next word | `WordExpectation` (exists) | the expected leaf operand of the unfolding (§3.1) |
| C1 | compose round | operand rows | each round's operands | — (none; §3.1) | the expected idea unfolded through the committed operations (§3.1) |
| C2 | sentence / row | idea `[NP1, VP, NP2]` | the next idea | `SentenceExpectation` (exists) | the chain (C3) |
| C3 | row chain / document | rows in LTM | the next row; the next thought step | the chain predictor (declared, off) | — (the one unconditioned prediction) |

Each level combines a predictor from below (teacher-forced, conditioned on
what has already arrived, never on what it predicts) with a prior from above
where the path is invertible. Before the path is invertible each level has
only its predictor from below — which is what lets the item land in stages.
This withdraws §12.2's "the word-level expectation follows from the row-level
predictor by inversion rather than by its own predictor".

### 3.1 The compose round, and why 4.5 needs no content width

A round's expectation is not predicted; it is **derived**. The sentence level
expects an idea; the actual sentence's derivation is a tree of committed
operations; unfolding the expected idea down that tree gives an expected
operand at every round, and the actual operand against it is the round's
surprise (§12.4: "conditioned on the actual structure; expectation completes
no derivation of its own"). No separate round predictor is needed.

The unfolding is done **given the actual co-operand** at each node: knowing
the result's expectation and the other operand that was actually there, the
inverse of the operation gives the expected value of this one. That is
exact for the maps (verb, adverb, lift: invertible given the modifier) and
partial for the rest — `∧ = min` returns the operand only where the result
lies below the co-operand, a Hadamard-type binding only where the co-operand
is nonzero. Where the inverse is undefined there is no expectation: `κ = 0`
on those coordinates, so no image and no surprise there. One expects only
what the operation let through.

The content width `D ≈ c·k·log N` arose for a different inverse: recovering
*both* operands from the root alone by clean-up against the bank, which is
what the **reconstruction audit** needs (todo item 3). Unfolding the
expectation along the actual tree never cleans up against the bank, so it
does not need that width. It stays in item 3.

The byte level retires from `BracketExpectation` (declared, never on; no legacy
paths are kept). The perceptual rung's *both* corner (the
prompt to divide a bracket) remains attention's evidence, as in 6.8; it never
needed expectation.

## 4. Negation, and where hallucination comes from

### 4.1 The image at every conceptual level

`n = −g·(1−m)⊙κ⊙ê`, `c = o + n` (§2.6.1), unchanged, applied by each level to
what it passes up, with its own gain `g_L` and confidence `κ_L`; concept face
only (§2.6.2 amendment 3). §2.6.2 item 1 — *sensation is never subtracted
from* — is now the whole of perception's relation to expectation.

### 4.2 Hallucination by top-down expectation

With no expectation of percepts, a hallucination cannot be a percept
predicted into existence. It is a **concept conceived without perceptual
evidence**: the estimate entering comprehension with the sign it has in
production (§2.6.3, reafference — positive in `<generate>`, negative in
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

### 4.3 One gain over signal and expectation (decided, Alec, 2026-10-09)

Attention attenuates both the signal and the expectation over the
unattended field, with one gain: the region weight `g` of 6.1's mask
multiplies the **sum**,

    c = g · (o + n),

never `g·o + n`. With the signal attenuated and the image not, the unattended
field fills with omission surprise — expected, nothing seen — and the mask
hallucinates absences. Under one gain the surprise scales down without
inverting.

The deep images — the per-level expectations of §3 (word, sentence,
document, the next) — are **per region**: realised inside a region, and
absent, not attenuated, outside it. The predictors read the regions, not a
lag window (the stream-state plan's stage 6, "expectation from the right
history"; today `word_distribution` and `SentenceExpectation` condition on
ARMA and `context_window` lags).

What 6.1 has built is this rule's implementation: region membership is exact
containment in the forward (zero outside; the reference bank and the
conceptual STM are masked by it, and the in-STM predictor reads the masked
STM), while the attention MLP reads the whole field's spreading activation
unattenuated. Comprehension gain 0 outside the regions, placement gain 1:
the literature's two tiers (the report's levels (a) and (b)) as a hard split.
A floor on the comprehension path is wrong — unadmitted items would leak
into compose and the budget would go. A parameter attenuating the MLP's
field input under focused attention (Woldorff 1991; Molloy 2015) is a later
refinement, not 6.1's.

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

### 4.5 The shallow prior and the novelty term (proposed; pending Alec)

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

Proposed accordingly, for 4.5 if accepted: one **shallow field-wide
transition image** beneath the regions — the first-order prediction of the
arriving item's form, class and position from the previous item — computed
wherever the field is active, not a region; its residual is a salience
candidate, not admitted content. The **novelty term** is that residual's
belief shift (prediction gain, not raw residual energy — persistent error
without belief change is not surprise: the snow paradox, the noisy-TV
pathology), fed into region placement a step behind onset, subordinate to
priming and set-match, cancelled when the deviation itself becomes regular.
Low priming alone is not a capture signal; low priming plus a violated
shallow prior is. Without the shallow prior there is no novelty term, and
6.1 has nothing to revise for it.

## 5. Explore and exploit

### 5.1 Exploit per level

At each conceptual level the greedy choice, made with that level's
expectation in context, is the exploit. Its surprise trains that level's
predictor (delta rule, §2.6.4). Sources go live and maps take inverted
targets only where the level is exactly invertible (§12.2, §13 point 2);
reconstruction retires to an audit level by level, and stays a loss
elsewhere (§13 point 3).

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

- **In 4.5:** a small graded set built for the gate — the standing fixtures'
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

- `Meaning.negative_image` → per conceptual level; `expectation_surprise` per
  level; gain and confidence per level in model.xml.
- `BracketExpectation`: byte level removed; row-chain level on.
- `WordExpectation`: surprise to the word level's consumers, not only the
  answer reader.
- The trial: departure level chosen uniformly among conceptual levels; the
  alternative from the `φ`-mix; judged by `R + A` at the end of the
  sentence (`E` leaves the trial cost); per-level records (departures,
  explore wins, the stall diagnostic).
- The unfolding (§3.1): expected operands along the actual tree, given the
  actual co-operand; `κ = 0` where an inverse is undefined; inverted targets
  for the maps where the inverse is exact.
- The graded set (§5.4) and its shuffled control.
- Docs: spec §2.6 (image at every conceptual level; percepts attended, not
  expected; hallucination as the sign of production in comprehension);
  GradientFlow's target scheme (trial cost `R + A`);
  Architecture "Surprise at every layer" and "Two hard derivations";
  FutureWork's section replaced by a pointer; operators plan §12.2's
  word-level sentence marked superseded.

## 7. Toys before the hand-off

1. **Done:** one departure read at the end, graded vs root-only vs unordered
   (§5.3).
2. **Judging with and without `E`.** A two-option chooser where one option
   explains the input and the other is more predictable; judged by
   `R + E + A` and by `R + A`. Records the dark-room drift that the decision
   avoids.
3. **Partial unfolding** (§3.1): on the operators' kernels, the fraction of
   coordinates where the expected operand is defined, and whether surprise
   restricted to them still trains the predictors.
4. **Inverted targets vs detached sources** (FutureWork's candidate): the XOR
   fixtures, with the collapse diagnostic.
5. **The shallow prior and capture** (§4.5, if accepted): without the
   field-wide shallow prior, unexpected tokens outside the regions are never
   admitted and region placement collapses onto early winners; with it,
   capture appears only after a transition regularity is learned, habituates
   when the deviation becomes regular, returns when the higher-order rule
   breaks — the Vachon signature.
6. **Residual energy vs prediction gain** as the novelty quantity: only the
   latter stops chasing an irreducibly noisy token stream.

## 8. Decisions and open questions

Decided 2026-10-08 (Alec): percepts are attended, not expected; the top-down
route is 6.1's single-layer mask over perceptual and symbolic activation;
the alternative is a `φ`-mix able to reach every alternative; trials judged
by `R + A`; the cost read at the end of the sentence; a graded set in 4.5,
the age-appropriate corpus before the final train (item 3); the compose
round derives its expectation (no predictor); the width stays in item 3.
Taken as agreed from rev 1 unless Alec objects: the two sources combined by
confidence; §12.2's word-level sentence withdrawn.

Decided (Alec, 2026-10-08, rev 3's two questions): **one departure per
sentence**, its level drawn uniformly among the conceptual levels (two
departures read at one end-of-sentence cost would confound each other's
credit); the graded order is **one more round per item, bottom-up** —
phrases, then clauses, then sentences.

Decided (Alec, 2026-10-09): one gain over signal and expectation,
`c = g·(o + n)`; deep expectation per region, absent outside it; 6.1's
exact-containment mask with the MLP reading the whole field is the
implementation (§4.3); `hetTolerance` τ on the lanes, left at 1, as the
answer to novelty-as-salience (§4.4).

Measured, not decided: a tunable softmax does not replace the departure; it
becomes the departure's alternative draw (§5.5). **Open:** the shallow
field-wide transition prior and the novelty term it feeds (§4.5), and with
them toys 5–6. §7's toys 2–4 remain before the hand-off.

## 9. Gates (proposed)

- The standing XOR fixtures and the math chain unchanged or better with every
  level on; each level's switch off reproduces the previous landing.
- Per level: expectation discrepancy falling; the collapse diagnostic flat;
  explore-win rate and the stall diagnostic reported.
- The graded set read in order beats the same set shuffled on the
  derivation-optimality rate (the toy's measure in the model); reported, with
  the shuffled run as the control.
